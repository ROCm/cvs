'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

HW/SW support-matrix compatibility gate.

A single data file per category (``training_support_matrix.json`` /
``inference_support_matrix.json``) declares, per ``(framework, model)``, what is
*unsupported* (deny-list). At run time -- after the container is launched -- a
suite calls ``run_support_matrix_gate()`` with the framework, model, and the
parameter values the run will use (e.g. the precisions across the enabled
sweeps). The library detects the GPU arch and ROCm version on the nodes and stops
the run when a declared constraint is violated.

Enforcement is controlled only by the ``CVS_SUPPORT_MATRIX_GATE`` environment
variable (see README.md): unset/ON/1/TRUE enforce (fail + stop); OFF/0/FALSE warn
and continue. The library never imports pytest; the suite turns a blocking result
into a test failure.

Everything lives in this one module on purpose -- it is small enough that
splitting it into loader/detect/evaluate/gate files would only add indirection.
'''

import json
import os
import re
from pathlib import Path

from cvs.lib import globals
from cvs.lib.utils_lib import get_model_from_rocm_smi_output

log = globals.log

GATE_ENV = "CVS_SUPPORT_MATRIX_GATE"

_DATA_DIR = Path(__file__).resolve().parent

# rocm-smi reports mi300x/mi325/mi350/mi355; the data files use the canonical
# "<family>x" tokens. Normalize so both sides compare equal.
_ARCH_ALIASES = {
    "mi300": "mi300x",
    "mi325": "mi325x",
    "mi350": "mi350x",
    "mi355": "mi355x",
}

NOT_IN_MATRIX_MSG = (
    "the framework or model under test is not available in the support matrix. "
    "This means this framework/model is not validated / characterized yet with CVS."
)

_OFF_VALUES = {"0", "off", "false", "no", "disable", "disabled"}


class GateResult:
    '''Outcome of a gate call.

    ``passed`` is True when the run may proceed (supported, unknown workload, or
    the gate is disabled). ``violations`` lists the deny matches found regardless
    of the enforcement mode, so the caller can log them.
    '''

    def __init__(self):
        self.passed = True
        self.not_in_matrix = False
        self.enforced = True
        self.violations = []
        self.warnings = []
        self.platform = {}


def gate_enforced():
    '''Return True when the gate should fail+stop on a violation.

    Controlled by ``CVS_SUPPORT_MATRIX_GATE``; unset defaults to enforce. Only the
    explicit OFF values disable it.
    '''
    raw = os.getenv(GATE_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in _OFF_VALUES


def normalize_arch(token):
    '''Lower-case, trim, and map a GPU token to its canonical ``<family>x`` form.'''
    name = str(token or "").strip().lower()
    return _ARCH_ALIASES.get(name, name)


def parse_version(text):
    '''Parse a ROCm version string into a (major, minor, patch) int tuple.

    Tolerates a build suffix (``6.4.0-123`` -> ``(6, 4, 0)``) and a missing patch
    (``7.0`` -> ``(7, 0, 0)``). Returns None when no version is present.
    '''
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", str(text or ""))
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def _cmp_version(a, b):
    return (a > b) - (a < b)


def load_matrix(category):
    '''Load and lightly validate the ``<category>_support_matrix.json`` data file.'''
    path = _DATA_DIR / f"{category}_support_matrix.json"
    with open(path) as fp:
        data = json.load(fp)
    frameworks = data.get("frameworks")
    if not isinstance(frameworks, dict):
        raise ValueError(f"{path.name}: missing or malformed 'frameworks' object")
    known = {normalize_arch(t) for t in data.get("known_arch_tokens", [])}
    if known:
        _validate_arch_tokens(data, known, path.name)
    return data


def _validate_arch_tokens(data, known, fname):
    '''Reject arch tokens in the data that are not in ``known_arch_tokens`` (typos).'''
    for fw, fw_block in data.get("frameworks", {}).items():
        for model, entry in (fw_block.get("models") or {}).items():
            if not isinstance(entry, dict):
                continue
            tokens = list(entry.get("unsupported_gpu_arch") or [])
            for value_map in (entry.get("unsupported_params") or {}).values():
                for arches in (value_map or {}).values():
                    tokens.extend(arches or [])
            for token in tokens:
                if normalize_arch(token) not in known:
                    raise ValueError(
                        f"{fname}: {fw}/{model} references unknown arch token {token!r} (not in known_arch_tokens)"
                    )


def find_model_entry(matrix, framework, model):
    '''Return the model entry dict for ``(framework, model)``, or None if absent.

    Matches the normalized model name against the model key or any of its
    ``aliases``.
    '''
    fw_block = (matrix.get("frameworks") or {}).get(str(framework or "").strip().lower())
    if not fw_block:
        return None
    models = fw_block.get("models") or {}
    wanted = _norm_model(model)
    if not wanted:
        return None
    for key, entry in models.items():
        candidates = [key] + list(entry.get("aliases") or [])
        if any(_norm_model(c) == wanted for c in candidates):
            return entry
    return None


def _norm_model(name):
    '''Normalize a model identifier for lenient matching.

    Lower-cases, takes the basename of a path or HF ``org/model`` id, and drops
    separators, so ``llama-3.1-8b``, ``llama3.1_8B``, ``llama3_1_8b`` and
    ``/root/models/Llama-3.1-8B`` all compare equal. The matrix therefore matches
    the preset name a training config uses and the HF id or local path an
    inference config uses, without one alias per spelling.
    '''
    text = str(name or "").strip().lower()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return re.sub(r"[\s._-]+", "", text)


# The config key that names the model is not unified across suites yet: training
# and inference configs use different spellings. Try all of them, at the top level
# and inside the common sub-blocks, so the model is found wherever it lives.
# ``model_repo`` is intentionally excluded: it is a filesystem path, not a model id.
# (Unifying these keys in the configs is a separate follow-up.)
_MODEL_KEYS = ("model_id", "hf_model_id", "model_name", "hf_model_name", "model")
_MODEL_CONTAINERS = ("model", "server_params", "model_params", "train_params", "inference", "tokenizer")


def _field(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _as_model_string(value):
    '''Coerce a candidate field into a model-name string, or None.

    Accepts a bare string, or an object/dict that itself carries the id under one
    of the known keys (e.g. a ``model`` block with ``.id``).
    '''
    if isinstance(value, str):
        return value.strip() or None
    if value is None:
        return None
    for key in ("id",) + _MODEL_KEYS:
        inner = _field(value, key)
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
    return None


def served_model_name(variant_config):
    '''Best-effort served-model identifier from a suite's variant config.

    The key that names the model varies across training and inference configs
    (``model_id``, ``hf_model_id``, ``model_name``, ``hf_model_name``, ``model``,
    ``model_repo``), and it may sit at the top level or inside ``model`` /
    ``server_params`` / ``model_params`` / ``train_params`` / ``inference`` /
    ``tokenizer``. This tries each key in each place and returns the first match;
    "" when none is found (the gate then treats the workload as not-in-matrix,
    i.e. unchecked).
    '''
    for key in _MODEL_KEYS:
        found = _as_model_string(_field(variant_config, key))
        if found:
            return found
    for container_name in _MODEL_CONTAINERS:
        container = _field(variant_config, container_name)
        if container is None:
            continue
        found = _as_model_string(container)
        if found:
            return found
        for key in _MODEL_KEYS:
            found = _as_model_string(_field(container, key))
            if found:
                return found
    return ""


def _host_text(raw):
    '''Extract text from a per-host exec result that may be a str or a dict.'''
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("output", "") or ""
    return ""


_PROBE_TIMEOUT_S = 30


def _exec(handle, cmd, timeout=_PROBE_TIMEOUT_S):
    '''Run ``handle.exec`` with a short timeout; fall back if the backend lacks it.'''
    try:
        return handle.exec(cmd, timeout=timeout)
    except TypeError:
        return handle.exec(cmd)


def _confident_arch(text):
    '''Return a canonical arch token only when the probe output actually names it.

    Reuses ``get_model_from_rocm_smi_output`` (the existing CVS parser), but that
    helper defaults to ``mi300x`` for unrecognized output, so ``mi300x`` is trusted
    only when the text really names it. Anything else -- empty output, a banner, or
    an arch the parser does not know (e.g. MI308X) -- returns None so the gate fails
    instead of assuming a default GPU.
    '''
    body = (text or "").strip()
    if not body:
        return None
    arch = normalize_arch(get_model_from_rocm_smi_output(body))
    if arch == "mi300x" and not re.search(r"MI300X", body, re.I):
        return None
    return arch


def detect_platform(handle):
    '''Detect ``{host: {"arch": <token|None>, "rocm": <version|None>}}`` via the handle.

    ``handle.exec`` routes into the launched container, so both probes reflect the
    container the workload will run in. GPU arch comes from ``rocm-smi`` with
    ``amd-smi`` as a fallback (both parsed by the existing
    ``get_model_from_rocm_smi_output``); an unrecognized GPU yields ``arch=None``.
    ROCm comes from ``/opt/rocm/.info/version`` then ``amd-smi`` / ``hipconfig``.
    '''
    arch_probe = (
        "rocm-smi --showproductname 2>/dev/null; rocm-smi -a 2>/dev/null; "
        "amd-smi static --asic 2>/dev/null; amd-smi list 2>/dev/null"
    )
    arch_out = _exec(handle, "bash -c " + _shq(arch_probe)) or {}
    rocm_probe = (
        "if [ -f /opt/rocm/.info/version ]; then cat /opt/rocm/.info/version; "
        "else amd-smi version --json 2>/dev/null || hipconfig --version 2>/dev/null || true; fi"
    )
    rocm_out = _exec(handle, "bash -c " + _shq(rocm_probe)) or {}

    hosts = set(arch_out) | set(rocm_out)
    platform = {}
    for host in hosts:
        platform[host] = {
            "arch": _confident_arch(_host_text(arch_out.get(host, ""))),
            "rocm": _extract_rocm_version(_host_text(rocm_out.get(host, ""))),
        }
    return platform


def _shq(cmd):
    import shlex

    return shlex.quote(cmd)


def _extract_rocm_version(text):
    '''Pull a ROCm version out of ``/opt/rocm/.info/version`` text or amd-smi JSON.'''
    raw = (text or "").strip()
    if not raw:
        return None
    if raw[:1] in "{[":
        try:
            payload = json.loads(raw)
            if isinstance(payload, list) and payload:
                payload = payload[0]
            if isinstance(payload, dict):
                ver = payload.get("rocm_version")
                if ver:
                    return str(ver).strip()
        except json.JSONDecodeError:
            pass
    match = re.search(r"\d+\.\d+(?:\.\d+)?", raw)
    return match.group(0) if match else None


def evaluate(entry, params, platform):
    '''Return ``(violations, warnings)`` for a model entry against the platform.

    ``params`` maps a parameter name to the values the run will use, e.g.
    ``{"precision": ["bf16", "mxfp4"]}``. Each rule is a deny: a match is a
    violation. An **undetected GPU arch is a violation** (the gate cannot certify a
    GPU it cannot identify). Undetected ROCm stays a warning, since a model only
    gates ROCm when it declares a min/max.
    '''
    violations = []
    warnings = []
    unsupported_arch = {normalize_arch(a) for a in (entry.get("unsupported_gpu_arch") or [])}
    unsupported_params = entry.get("unsupported_params") or {}
    rmin = parse_version(entry.get("required_rocm_min_version"))
    rmax = parse_version(entry.get("required_rocm_max_version"))

    for host, info in sorted(platform.items()):
        arch = info.get("arch")
        rocm_str = info.get("rocm")
        rocm = parse_version(rocm_str)

        if not arch:
            violations.append(
                f"{host}: could not detect a supported GPU arch (rocm-smi/amd-smi returned no recognizable device)"
            )
        else:
            if arch in unsupported_arch:
                violations.append(f"{host}: GPU arch {arch} is not supported for this model")
            for pname, values in (params or {}).items():
                deny = unsupported_params.get(pname) or {}
                for value in values:
                    v = str(value).strip().lower()
                    if arch in {normalize_arch(a) for a in (deny.get(v) or [])}:
                        violations.append(f"{host}: {pname}={v} is not supported on {arch}")

        if rmin or rmax:
            if rocm is None:
                warnings.append(f"{host}: could not detect ROCm version; skipping ROCm range check")
            else:
                if rmin and _cmp_version(rocm, rmin) < 0:
                    violations.append(
                        f"{host}: ROCm {rocm_str} is below the required minimum "
                        f"{entry.get('required_rocm_min_version')}"
                    )
                if rmax and _cmp_version(rocm, rmax) > 0:
                    violations.append(
                        f"{host}: ROCm {rocm_str} is above the supported maximum "
                        f"{entry.get('required_rocm_max_version')}"
                    )
    return violations, warnings


def run_support_matrix_gate(category, framework, model, params, handle):
    '''Run the support-matrix gate and return a ``GateResult``.

    Loads the category data file, looks up ``(framework, model)``, detects the
    platform via ``handle``, evaluates the deny rules, and applies the
    ``CVS_SUPPORT_MATRIX_GATE`` enforcement mode. All outcomes are logged here; the
    caller only needs to act on ``result.passed``.
    '''
    result = GateResult()
    result.enforced = gate_enforced()

    try:
        matrix = load_matrix(category)
    except Exception as exc:  # noqa: BLE001
        if result.enforced:
            log.error(
                "SUPPORT-MATRIX: could not load %s matrix (%s); failing (set %s=OFF to bypass)",
                category,
                exc,
                GATE_ENV,
            )
            result.passed = False
            result.violations = [f"support matrix for '{category}' failed to load: {exc}"]
        else:
            log.warning("SUPPORT-MATRIX: could not load %s matrix (%s); gate disabled, continuing", category, exc)
        return result

    entry = find_model_entry(matrix, framework, model)
    if entry is None:
        result.not_in_matrix = True
        log.info("SUPPORT-MATRIX: %s (framework=%s, model=%s)", NOT_IN_MATRIX_MSG, framework, model)
        return result

    result.platform = detect_platform(handle)
    log.info("SUPPORT-MATRIX: framework=%s model=%s detected=%s", framework, model, result.platform)

    violations, warnings = evaluate(entry, params, result.platform)
    result.violations = violations
    result.warnings = warnings
    for warning in warnings:
        log.warning("SUPPORT-MATRIX: %s", warning)

    if not violations:
        log.info("SUPPORT-MATRIX: PASS - %s/%s supported on the detected platform", framework, model)
        return result

    for violation in violations:
        if result.enforced:
            log.error("SUPPORT-MATRIX: UNSUPPORTED - %s", violation)
        else:
            log.warning("SUPPORT-MATRIX: UNSUPPORTED - %s", violation)

    if result.enforced:
        result.passed = False
    else:
        result.passed = True
        log.warning("NOTE: %s is disabled - continuing despite the violation(s) above.", GATE_ENV)
    return result
