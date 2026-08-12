'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

"""
Paired A/B RCCL regression test.

For each (collective, data type, NCCL env-combo) this runs a reference build (A)
and a candidate build (B) back-to-back, interleaved, for a configurable number of
repeats. Because both sides run on the same nodes within the same allocation,
environmental noise is largely common-mode and cancels in the paired comparison.

All pass/fail logic lives in ``cvs.lib.regression_lib`` (pure, unit-tested). This
module is only the cluster orchestration: it produces the A and B result samples
and hands them to the detector.

Modes (config: rccl.ab_regression):
  - control_mode=true : both sides use the reference build (A=B). Used to measure
    the on-hardware run-to-run noise floor, derive thresholds, and prove the
    detector reports zero regressions on an identical build.
  - control_mode=false: reference build vs candidate build (real detection).

See cvs/input/config_file/rccl/rccl_ab_config.json.sample for the config shape.
"""

import os
import json
import itertools
import copy

import pytest

from cvs.lib import rccl_lib
from cvs.lib import regression_lib
from cvs.lib import ci_robustness_lib
from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


def _gpu_cleanup_cfg(config_dict):
    """Return the gpu_cleanup config block with sane defaults."""
    cfg = dict(config_dict.get('gpu_cleanup', {}))
    cfg.setdefault('enabled', True)
    cfg.setdefault('kill_gpu_pids', True)
    cfg.setdefault('kill_containers', False)
    cfg.setdefault('use_sudo', False)
    cfg.setdefault('process_patterns', None)  # None -> library defaults
    return cfg


def _do_gpu_cleanup(phdl, config_dict, reason=""):
    """Run stale-GPU cleanup across all nodes if enabled in config."""
    cfg = _gpu_cleanup_cfg(config_dict)
    if not cfg.get('enabled', True):
        return
    log.info("Running stale-GPU cleanup%s", f" ({reason})" if reason else "")
    rccl_lib.cleanup_gpus_on_nodes(
        phdl,
        process_patterns=cfg.get('process_patterns'),
        kill_gpu_pids=cfg.get('kill_gpu_pids', True),
        kill_containers=cfg.get('kill_containers', False),
        use_sudo=cfg.get('use_sudo', False),
    )


# Accumulates per-(collective, dtype, env-combo) A and B run samples across the
# parametrized test invocations, consumed by the final analysis test.
#   ab_runs[group_key] = {"a": [...], "b": [...], "repeats_expected": R, "complete": bool}
#
# "complete" is load-bearing, not bookkeeping. A sweep that dies at repeat 4 of 7
# still leaves 4 usable pairs behind, and the detector's min_repeats=2 will happily
# issue a verdict on them. That verdict is not comparable to a full run's -- fewer
# repeats means wider spread and a much weaker separation gate -- so a truncated
# group must never be scored. The flag is set only after the loop runs to
# completion; any exception mid-loop leaves it False.
ab_runs = {}

# Circuit breaker. The pipeline's worst observed failure was 110 minutes spent
# producing zero measurements: every sweep failed identically, each retry burned
# the full per-collective timeout, and the job kept marching through the matrix
# re-proving the same broken thing. Once N consecutive sweeps fail outright, the
# environment is broken rather than flaky -- stop paying for the rest of the
# matrix and let the analysis step emit an explicit no-verdict report.
_breaker = {"consecutive_failures": 0, "tripped_by": None}


# --------------------------------------------------------------------------- #
# Fixtures (mirror cvs/tests/rccl/rccl_regression.py)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rccl']
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    env_vars = cluster_dict.get("env_vars")
    node_list = list(cluster_dict['node_dict'].keys())
    return Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    node_list = list(cluster_dict['node_dict'].keys())
    env_vars = cluster_dict.get("env_vars")
    head_node = node_list[0]
    return Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)


# --------------------------------------------------------------------------- #
# Parametrization: collectives x data types x NCCL env-combos
# --------------------------------------------------------------------------- #
def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.exists(config_file):
        log.warning(f'Warning: Missing or invalid config file {config_file}')
        return

    with open(config_file) as fp:
        cfg = json.load(fp)
    rccl = cfg.get("rccl", {})
    # The NCCL knob matrix (`regression`) is OPTIONAL for the A/B perf-regression
    # test. Perf regression is a question about collective x dtype x message size
    # under the production env; sweeping NCCL knobs (PXN, channels, ALGO/PROTO,
    # P2P, ...) is extra *coverage*, not a prerequisite for a verdict. When absent
    # we run each collective once under the env_source_script defaults.
    regression = dict(rccl.get("regression", {}))

    # Paired channel handling (min/max kept paired, not Cartesian) - identical to
    # the single-sided regression test. No-op when `regression` is empty.
    has_min = "NCCL_MIN_NCHANNELS" in regression
    has_max = "NCCL_MAX_NCHANNELS" in regression
    if has_min != has_max:
        raise ValueError("NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS must be both present or both absent")
    paired_channels = None
    if has_min and has_max:
        min_vals = regression["NCCL_MIN_NCHANNELS"]
        max_vals = regression["NCCL_MAX_NCHANNELS"]
        if len(min_vals) != len(max_vals):
            raise ValueError("NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS must have equal length")
        paired_channels = list(zip(min_vals, max_vals))
        del regression["NCCL_MIN_NCHANNELS"]
        del regression["NCCL_MAX_NCHANNELS"]

    env_axes = []
    for key in sorted(regression.keys()):
        value = regression[key]
        if isinstance(value, list) and value:
            env_axes.append((key, value))

    if "rccl_collective" in metafunc.fixturenames:
        rccl_collective_list = rccl.get("rccl_collective", ["all_reduce_perf"])
        env_params, env_ids = [], []

        if env_axes:
            # Optional NCCL knob matrix: Cartesian product over each env axis.
            env_fixture_names = [name for name, _ in env_axes]
            env_domains = [dict(env_axes)[name] for name in env_fixture_names]

            channel_fixture_names = []
            if paired_channels is not None:
                channel_fixture_names = ["NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"]
                env_domains.append(paired_channels)

            for env_combo in itertools.product(*env_domains):
                env_dict = dict(zip(env_fixture_names + channel_fixture_names, env_combo))
                if paired_channels is not None:
                    min_ch, max_ch = env_dict.pop("NCCL_MIN_NCHANNELS")
                    env_dict["NCCL_MIN_NCHANNELS"] = min_ch
                    env_dict["NCCL_MAX_NCHANNELS"] = max_ch
                env_params.append(env_dict)
                env_ids.append("|".join(f"{k}={v}" for k, v in env_dict.items()))
        else:
            # Default perf-regression path: one run per collective under the
            # production env (env_source_script), no NCCL knob override.
            env_params = [{}]
            env_ids = ["default"]

        metafunc.parametrize("rccl_collective", rccl_collective_list)
        metafunc.parametrize("regression_params", env_params, ids=env_ids)

        # Optional data-type axis. Each data type runs as its own sweep (rccl-tests -d).
        if "data_type" in metafunc.fixturenames:
            data_type_list = rccl.get("data_types", ["float"])
            metafunc.parametrize("data_type", data_type_list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def derived_thresholds_paths(ab_cfg, out_dir):
    """
    Ordered candidate locations for the calibrated-threshold file.

    The per-run ``out_dir`` USED TO BE the only location, which quietly stopped
    working the moment per-run workspaces landed: ``ws_config_retarget`` points
    ``output_dir`` at a brand-new empty ``runs/<key>/artifacts``, so the file can
    never be there and every run silently fell back to the config's static
    numbers. It went unnoticed only because the static fallback happened to be a
    hand-copy of the derived file. The fix is to read from a stable location that
    outlives any one run; ``out_dir`` is kept last so a calibration run's own
    artifacts still resolve, and so a legacy shared-mode tree keeps working.
    """
    paths = []
    explicit = ab_cfg.get('derived_thresholds_path')
    if explicit:
        paths.append(explicit)
    paths.append(os.path.join(
        os.getenv('RCCL_CI_ROOT', '/it-share/rccl-ci'), 'configs', 'ab_derived_thresholds.json'))
    paths.append(os.path.join(out_dir, 'ab_derived_thresholds.json'))
    # Preserve order, drop duplicates.
    seen, ordered = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _provenance(ab_cfg):
    """
    Everything needed to reproduce or audit this verdict later: which detector
    code ran, which workspace it ran in, and which two libraries it compared.

    The .built_rev stamps are read from the lib dirs themselves rather than from
    the config, because the config records the path we were TOLD to use and the
    stamp records what is actually there -- when a build silently reuses a cache
    entry, only the stamp shows it.
    """
    def _built_rev(side):
        ld = (side or {}).get('ld_library_path') or ''
        first = ld.split(':')[0]
        if not first:
            return None
        try:
            with open(os.path.join(first, '.built_rev')) as fh:
                return fh.read().strip()
        except OSError:
            return None

    ref = ab_cfg.get('reference', {}) or {}
    cand = ab_cfg.get('candidate', {}) or {}
    return {
        "cvs_sha": os.getenv('RCCL_CI_CVS_SHA'),
        "cvs_dir": os.getenv('CVS_DIR'),
        "run_key": os.getenv('RCCL_CI_RUN_KEY'),
        "slurm_job_id": os.getenv('SLURM_JOB_ID'),
        "github_run_id": os.getenv('GITHUB_RUN_ID'),
        "reference": {"lib": (ref.get('ld_library_path') or '').split(':')[0],
                      "built_rev": _built_rev(ref)},
        "candidate": {"lib": (cand.get('ld_library_path') or '').split(':')[0],
                      "built_rev": _built_rev(cand)},
    }


def _sides_identical(ab_cfg):
    """
    Return a description of the collision if reference and candidate resolve to the
    same build, else None. Symlinks are resolved because the two sides are normally
    handed to us as per-side symlinks that a build step points at real dirs -- an
    A=A run shows up as two different names for one inode, not as two equal strings.
    """
    ref = ab_cfg.get('reference', {}) or {}
    cand = ab_cfg.get('candidate', {}) or {}

    def _real(side):
        return tuple(
            os.path.realpath(side[k]) if side.get(k) else None
            for k in ("ld_library_path", "rccl_tests_dir")
        )

    ref_r, cand_r = _real(ref), _real(cand)
    if ref_r == (None, None) and cand_r == (None, None):
        # Neither side overrides anything: both run the ambient library. Same build.
        return "neither side sets ld_library_path or rccl_tests_dir"
    if ref_r == cand_r:
        return ref_r[0] or ref_r[1]

    # Different directories can still hold the same build. That is not exotic: a
    # PR whose merge-base equals its head, or one that changes nothing the build
    # consumes, produces two separate lib dirs stamped with one revision.
    prov = _provenance({"reference": ref, "candidate": cand})
    ref_rev = prov["reference"]["built_rev"]
    cand_rev = prov["candidate"]["built_rev"]
    if ref_rev and cand_rev and ref_rev == cand_rev:
        return f"both sides built from revision {ref_rev}"
    return None


def _publish_derived_thresholds(derived, ab_cfg, out_dir, publish=True):
    """
    Write a calibration to the run's artifacts AND to the stable location that
    later detect runs read from.

    The artifact copy is for the humans reading this run's output; the stable copy
    is the one that does any work. Written via a temp file + os.replace so a
    detect run that reads it concurrently sees either the old file or the new one,
    never a half-written one -- these live on shared NFS with no other locking.
    Publishing is best-effort: a calibration run that cannot write the shared path
    should still finish and still leave its numbers in its own artifacts.

    ``publish=False`` writes only the artifact copy. The caller uses this for a
    control run that did not clear its own checks: the numbers are still worth
    looking at, they are just not fit to widen the gate for every later run.
    """
    local = os.path.join(out_dir, 'ab_derived_thresholds.json')
    with open(local, 'w') as fp:
        json.dump(derived, fp, indent=2)

    if not publish:
        return

    if not ab_cfg.get('publish_derived_thresholds', True):
        log.info("publish_derived_thresholds=false; calibration left in %s only", local)
        return

    stable = derived_thresholds_paths(ab_cfg, out_dir)[0] if ab_cfg.get('derived_thresholds_path') else \
        os.path.join(os.getenv('RCCL_CI_ROOT', '/it-share/rccl-ci'), 'configs', 'ab_derived_thresholds.json')
    try:
        os.makedirs(os.path.dirname(stable), exist_ok=True)
        tmp = f"{stable}.tmp.{os.getpid()}"
        with open(tmp, 'w') as fp:
            json.dump(derived, fp, indent=2)
        os.replace(tmp, stable)
        log.info("Published calibrated thresholds to %s", stable)
    except OSError as exc:
        log.warning("Could not publish calibrated thresholds to %s (%s); "
                    "detect runs will keep using the previous calibration.", stable, exc)


def _resolve_detect_thresholds(detector_overrides, ab_cfg, out_dir):
    """
    In detect mode, override config thresholds with the ones calibrated on THIS
    hardware (``ab_derived_thresholds.json`` written by a prior control run), so the
    gate never silently runs on stale numbers. Pure/testable: returns a (possibly
    new) detector_overrides dict and never raises.

    Prefers a per-collective table when the calibration produced one, since
    collectives differ in noise by more than size tiers do.

    Escape hatch: ``ab_regression.use_derived_thresholds: false`` forces the config
    thresholds. If no derived file is found, config thresholds stand.
    """
    if not ab_cfg.get('use_derived_thresholds', True):
        return {**detector_overrides, "thresholds_source": "config (use_derived_thresholds=false)"}

    cfg_thr = detector_overrides.get("thresholds")
    tried = []
    for derived_path in derived_thresholds_paths(ab_cfg, out_dir):
        tried.append(derived_path)
        try:
            with open(derived_path) as fp:
                doc = json.load(fp) or {}
        except FileNotFoundError:
            continue
        except ValueError as exc:
            log.warning("Could not parse %s (%s); trying next location.", derived_path, exc)
            continue

        derived_thr = doc.get('thresholds_by_collective') or doc.get('thresholds')
        if not derived_thr:
            log.warning("%s has no thresholds; trying next location.", derived_path)
            continue

        shape = 'per-collective' if doc.get('thresholds_by_collective') else 'per-tier'
        log.info("Using calibrated thresholds (%s) from %s: %s (overriding config)",
                 shape, derived_path, derived_thr)
        return {**detector_overrides,
                "thresholds": derived_thr,
                "thresholds_source": f"{derived_path} ({shape})"}

    log.warning("No calibrated thresholds found in any of %s; using config thresholds %s. "
                "Run a control-mode calibration first.", tried, cfg_thr)
    return {**detector_overrides, "thresholds_source": "config (no calibration found)"}


def _side_params(base_rccl_test_params, side_cfg, data_type=None):
    """Build a per-side copy of rccl_test_params with the build's tests dir / lib path."""
    params = copy.deepcopy(base_rccl_test_params)
    if side_cfg.get("rccl_tests_dir"):
        params["rccl_tests_dir"] = side_cfg["rccl_tests_dir"]
    if side_cfg.get("ld_library_path"):
        params["ld_library_path"] = side_cfg["ld_library_path"]
    if data_type:
        params["data_type"] = data_type
    return params


def _run_one_side_once(phdl, shdl, cluster_dict, config_dict, side_cfg, collective, env_overrides, repeat_idx,
                       data_type=None):
    """
    Run a single sweep for one build side and return the parsed result rows.

    Raises on failure so the retry wrapper can act: a sweep "fails" if
    rccl_regression raises, if it recorded errors via fail_test (error_list), or
    if it produced no result rows.
    """
    node_list = list(cluster_dict['node_dict'].keys())
    vpc_node_list = [cluster_dict['node_dict'][n]['vpc_ip'] for n in node_list]

    rccl_test_params = _side_params(config_dict['rccl_test_params'], side_cfg, data_type)

    # Per-side, per-repeat result file so concurrent/sequential runs never clash.
    base_cvs = copy.deepcopy(config_dict['cvs_params'])
    base_file = base_cvs.get('rccl_result_file', '/tmp/rccl_result_output.json')
    stem, ext = os.path.splitext(base_file)
    label = side_cfg.get("label", "side")
    dt = data_type or "na"
    base_cvs['rccl_result_file'] = f'{stem}_{label}_{dt}_r{repeat_idx}{ext}'
    # A/B does its own comparison; disable the single-sided verification entirely.
    base_cvs['verify_bus_bw'] = 'False'
    base_cvs['verify_bw_dip'] = 'False'
    base_cvs['verify_lat_dip'] = 'False'

    # Clean, reproducible per-run log (MPI launch command + rccl-tests output),
    # separate from the verbose parallel-ssh logging. Defaults to a single
    # appended file under the A/B output dir; honour an explicit override.
    if not base_cvs.get('rccl_command_log'):
        ab_cfg = config_dict.get('ab_regression', {})
        out_dir = ab_cfg.get('output_dir') or os.getenv('CVS_OUTPUT_BASE_DIR') or '/tmp'
        base_cvs['rccl_command_log'] = os.path.join(out_dir, 'rccl_runs.log')

    env_overrides = {k: str(v) for k, v in env_overrides.items()}

    # Reset the global failure accumulator so we can detect this sweep's own
    # failures (rccl_regression records via fail_test rather than always raising).
    globals.error_list = []
    rows = rccl_lib.rccl_regression(
        phdl,
        shdl,
        collective,
        config_dict.get('env_source_script', '/dev/null'),
        config_dict['mpi_params'],
        rccl_test_params,
        base_cvs,
        node_list,
        vpc_node_list,
        env_overrides,
    )
    if globals.error_list:
        errs = list(globals.error_list)
        globals.error_list = []
        raise RuntimeError(f"sweep reported failures: {errs}")
    if not rows:
        raise RuntimeError("sweep returned no result rows")
    return rows


def _run_one_side(phdl, shdl, cluster_dict, config_dict, side_cfg, collective, env_overrides, repeat_idx,
                  data_type=None):
    """Run one sweep with retry on transient failures; clean stale GPU state between attempts."""
    retry_cfg = config_dict.get('retry', {})
    max_retries = int(retry_cfg.get('max_retries', 2))
    backoff_sec = float(retry_cfg.get('backoff_sec', 15))

    def attempt():
        return _run_one_side_once(
            phdl, shdl, cluster_dict, config_dict, side_cfg, collective, env_overrides, repeat_idx, data_type
        )

    def on_before_retry(next_attempt):
        # A flaky run can leave orphaned ranks/orted holding GPUs; clear them first.
        _do_gpu_cleanup(phdl, config_dict, reason=f"before retry {next_attempt} of {side_cfg.get('label')}")

    label = f"{collective}/{side_cfg.get('label')}/d={data_type}/r{repeat_idx}"
    try:
        result = ci_robustness_lib.run_with_retries(
            attempt,
            max_retries=max_retries,
            on_before_retry=on_before_retry,
            backoff_sec=backoff_sec,
            log=log,
            label=label,
        )
    except Exception:
        _breaker["consecutive_failures"] += 1
        globals.error_list = []
        raise
    # A sweep that produced rows proves the environment still works; the breaker
    # only cares about an unbroken run of failures.
    _breaker["consecutive_failures"] = 0
    # Ensure no stale errors leak into the test's final pass/fail check.
    globals.error_list = []
    return result


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_00_cleanup_stale_gpu_state(phdl, config_dict):
    """
    Kill stale RCCL/MPI processes (and optionally GPU PIDs / containers) left on
    the allocated nodes by prior or cancelled jobs, before any benchmark runs.
    Runs first by virtue of its position in this module. Best-effort: never fails.
    """
    globals.error_list = []
    _do_gpu_cleanup(phdl, config_dict, reason="pre-run")
    update_test_result()


def test_ab_pair(phdl, shdl, cluster_dict, config_dict, rccl_collective, regression_params, data_type):
    """Run reference (A) and candidate (B) interleaved for R repeats and stash samples."""
    globals.error_list = []

    ab_cfg = config_dict.get('ab_regression', {})

    # Excluded (collective, dtype) combinations. Used for combos that are broken
    # UPSTREAM (e.g. a collective that fails rccl-tests data verification on this
    # ROCm/RCCL revision) — running them would only exhaust retries and then HARD
    # FAIL the whole gate job, masking the real verdict. We pytest.skip them so the
    # gate stays green/meaningful for the working matrix; revisit when upstream
    # fixes the combo. Each entry is a [collective, data_type] pair, e.g.
    #   "skip_keys": [["alltoall_perf", "bfloat16"]]
    skip_keys = {(str(c), str(d)) for c, d in ab_cfg.get('skip_keys', [])}
    if (str(rccl_collective), str(data_type)) in skip_keys:
        pytest.skip(f"{rccl_collective}/{data_type} excluded via ab_regression.skip_keys "
                    f"(known upstream issue; not a gate failure)")

    # Circuit breaker: if the environment has already proven itself broken, skip
    # the rest of the matrix immediately instead of re-discovering it one 30-minute
    # timeout at a time. Skipping (rather than pytest.exit) is deliberate: the
    # analysis test must still run so the job emits a report that says, explicitly,
    # that there is no verdict.
    if _breaker["tripped_by"]:
        pytest.skip(f"circuit breaker tripped by {_breaker['tripped_by']}; "
                    f"environment is broken, not flaky — skipping remaining sweeps")

    repeats = int(ab_cfg.get('repeats', 7))
    control_mode = bool(ab_cfg.get('control_mode', False))
    breaker_budget = int(ab_cfg.get('circuit_breaker_failures', 3))

    reference_cfg = ab_cfg.get('reference', {"label": "ref"})
    candidate_cfg = ab_cfg.get('candidate', {"label": "cand"})
    if control_mode:
        # Both sides use the reference build to characterise noise.
        candidate_cfg = copy.deepcopy(reference_cfg)
        candidate_cfg['label'] = 'cand'
    reference_cfg = {**reference_cfg, "label": reference_cfg.get("label", "ref")}
    candidate_cfg = {**candidate_cfg, "label": candidate_cfg.get("label", "cand")}

    params_str = ' '.join(f'{k}={v}' for k, v in regression_params.items()) or 'default'
    group_key = f'{rccl_collective}-d={data_type}-{params_str}'
    group = ab_runs.setdefault(
        group_key, {"a": [], "b": [], "repeats_expected": repeats, "complete": False})
    group["repeats_expected"] = repeats

    try:
        for r in range(repeats):
            # Interleave A,B per repeat so slow drift affects both sides equally.
            a_rows = _run_one_side(
                phdl, shdl, cluster_dict, config_dict, reference_cfg, rccl_collective, regression_params, r, data_type
            )
            b_rows = _run_one_side(
                phdl, shdl, cluster_dict, config_dict, candidate_cfg, rccl_collective, regression_params, r, data_type
            )
            # Append A and B together, after both succeeded. Appending A before
            # running B would leave an unpaired A sample behind if B then died,
            # which is exactly the asymmetry the detector's balanced-samples guard
            # has to reject -- better not to create it.
            group["a"].append(a_rows)
            group["b"].append(b_rows)
    except Exception:
        if _breaker["consecutive_failures"] >= breaker_budget and not _breaker["tripped_by"]:
            _breaker["tripped_by"] = f"{group_key} ({_breaker['consecutive_failures']} consecutive sweep failures)"
            log.error("Circuit breaker TRIPPED: %s. Remaining sweeps will be skipped.",
                      _breaker["tripped_by"])
        raise

    group["complete"] = True
    update_test_result()


def test_ab_analyze(request, config_dict):
    """Derive thresholds (control mode) and/or run the A/B detector; fail on confirmed regressions."""
    globals.error_list = []
    ab_cfg = config_dict.get('ab_regression', {})
    control_mode = bool(ab_cfg.get('control_mode', False))

    # Effective detector config from the user's ab_regression block.
    detector_overrides = {}
    for k in ("thresholds", "tier_boundaries", "separation_gate", "separation_b_percentile",
              "separation_a_percentile", "adjacency_min_run", "min_bandwidth_floor", "min_repeats",
              "metric", "higher_is_better"):
        if k in ab_cfg:
            detector_overrides[k] = ab_cfg[k]

    out_dir = ab_cfg.get('output_dir') or os.getenv('CVS_OUTPUT_BASE_DIR') or '/tmp'
    os.makedirs(out_dir, exist_ok=True)

    overall_has_regression = False
    all_reports = {}
    thresholds_source = "config"
    # Reasons this run cannot support ANY verdict, pass or fail. Kept separate from
    # the regression list on purpose: "we found no regression" and "we were unable
    # to look" are different statements, and collapsing them into one green check
    # is how a broken gate goes unnoticed for weeks.
    untrustworthy = []

    if _breaker["tripped_by"]:
        untrustworthy.append(f"circuit breaker tripped: {_breaker['tripped_by']}")

    if not control_mode:
        # Detect mode compares two builds. If both sides resolve to the same
        # librccl, the run measures one build against itself and is GUARANTEED to
        # report zero regressions -- a green check that means nothing. That is not
        # hypothetical: any path that decides "RCCL didn't change, skip the build"
        # and then still runs detect lands exactly here, and control_mode is the
        # only legitimate way to ask for A==A.
        same = _sides_identical(ab_cfg)
        if same and ab_cfg.get('allow_identical_sides'):
            log.warning("Both sides resolve to the same build (%s); "
                        "allow_identical_sides is set, so this is a pipeline smoke run "
                        "and its PASS says nothing about any code change.", same)
        elif same:
            untrustworthy.append(
                f"detect mode but both sides resolve to the same build ({same}) — "
                f"this run compares a build against itself and cannot detect anything. "
                f"Use ab_regression.control_mode=true if an A=A noise run was intended.")

    # Groups that did not finish all their repeats are excluded outright.
    scored_runs = {}
    for group_key, runs in ab_runs.items():
        if not runs.get("complete"):
            untrustworthy.append(
                f"{group_key}: only {len(runs['a'])}/{runs.get('repeats_expected', '?')} "
                f"repeats completed — group excluded")
            continue
        scored_runs[group_key] = runs

    if not scored_runs:
        untrustworthy.append("no group completed its full set of repeats")

    # Control mode: derive thresholds from the combined A+B (same build) data.
    pending_publish = None
    if control_mode:
        control_runs = []
        for g in scored_runs.values():
            control_runs.extend(g["a"])
            control_runs.extend(g["b"])
        if control_runs:
            derived = regression_lib.derive_thresholds(
                control_runs,
                config=detector_overrides or None,
                safety_factor=float(ab_cfg.get('safety_factor', 2.0)),
                mad_k=float(ab_cfg.get('mad_k', 3.0)),
            )
            log.info("Derived thresholds from control run: pooled=%s per-collective=%s",
                     derived["thresholds"], derived.get("thresholds_by_collective"))
            log.info("Measured noise: %s", derived["noise"])
            # Publishing is deferred until after the per-group checks below.
            # These numbers go to a shared path that every future detect run
            # reads, so a control run that measured badly must not be allowed to
            # widen the gate for everyone before anyone notices it measured badly.
            pending_publish = derived
            # Apply derived thresholds for the (sanity) detection below.
            #
            # Prefer the per-collective table. Detect mode resolves
            # thresholds_by_collective, so scoring the control run with the
            # pooled per-tier dict checks A=A against numbers production never
            # uses -- looser than production for some collectives (large tier
            # 0.0498 pooled against 0.0300 per-collective) and tighter for
            # others. A control run's whole job is to be a faithful dry run of
            # the gate, and it cannot be one while the two disagree.
            detector_overrides = {
                **detector_overrides,
                "thresholds": (derived.get("thresholds_by_collective")
                               or derived["thresholds"]),
            }
            thresholds_source = "this control run"
    else:
        # Detect mode: prefer thresholds calibrated on THIS hardware (written by a
        # prior control run) over the potentially-stale values baked into the config.
        detector_overrides = _resolve_detect_thresholds(detector_overrides, ab_cfg, out_dir)
        thresholds_source = detector_overrides.pop("thresholds_source", "config")

    for group_key, runs in scored_runs.items():
        report = regression_lib.detect_regressions(runs["a"], runs["b"], config=detector_overrides or None)
        all_reports[group_key] = report
        log.info("[%s]\n%s", group_key, regression_lib.format_report(report))
        summary = report["summary"]
        if summary["has_regression"]:
            overall_has_regression = True
        # detect_regressions decides per group whether its own comparison holds up
        # (keys present on only one side, too many inconclusive keys, nothing
        # compared at all). Propagate that here rather than reading only
        # has_regression, which is False both when the build is clean and when we
        # measured nothing whatsoever.
        if not summary.get("trustworthy", True):
            reasons = []
            if not summary["keys_compared"]:
                reasons.append("no keys compared")
            if summary.get("missing_keys"):
                reasons.append(f"{summary['missing_keys']} key(s) on one side only")
            if summary.get("inconclusive_exceeded"):
                reasons.append(f"{summary['inconclusive_frac'] * 100:.1f}% inconclusive")
            untrustworthy.append(f"{group_key}: {', '.join(reasons) or 'untrustworthy'}")

    # Calibration is only published once the run that produced it has cleared
    # every check. A control run that tripped the breaker, lost a group, or came
    # back mostly inconclusive still writes its numbers to the run's own
    # artifacts for inspection -- it just does not get to hand them to the gate.
    if pending_publish is not None:
        if untrustworthy:
            log.warning("NOT publishing derived thresholds: this control run is untrustworthy (%s). "
                        "They remain in this run's artifacts only.", "; ".join(untrustworthy[:5]))
            _publish_derived_thresholds(pending_publish, ab_cfg, out_dir, publish=False)
        else:
            _publish_derived_thresholds(pending_publish, ab_cfg, out_dir)

    with open(os.path.join(out_dir, 'ab_regression_report.json'), 'w') as fp:
        json.dump({
            "control_mode": control_mode,
            # Which code and which builds produced this verdict. Without it, a
            # report on NFS is just a number: there is no way, after the fact, to
            # tell which detector version scored it or which two libraries it
            # compared -- and both change underneath a shared checkout.
            "provenance": _provenance(ab_cfg),
            "thresholds_source": thresholds_source,
            "trustworthy": not untrustworthy,
            "untrustworthy_reasons": untrustworthy,
            "groups_expected": len(ab_runs),
            "groups_scored": len(scored_runs),
            "reports": all_reports,
        }, fp, indent=2, default=str)

    log.info("Thresholds source: %s", thresholds_source)

    # Order matters: report untrustworthiness FIRST. A run that both could not be
    # trusted and happened to flag something should be reported as the former,
    # because under those conditions the flag is not evidence either.
    if untrustworthy:
        fail_test(f"A/B run produced NO USABLE VERDICT ({len(untrustworthy)} reason(s)): "
                  + "; ".join(untrustworthy[:10])
                  + " — this is neither a PASS nor a regression; see ab_regression_report.json")
    elif control_mode:
        # In control mode a confirmed regression means the detector is NOT stable
        # on this hardware (identical build flagged) - that must fail loudly.
        if overall_has_regression:
            fail_test("Control run (A=B) reported a regression - detector/thresholds are not stable on this hardware")
    else:
        if overall_has_regression:
            regressions = sum(r["summary"]["regressions"] for r in all_reports.values())
            fail_test(f"A/B regression detected: {regressions} confirmed regression(s) - see ab_regression_report.json")

    update_test_result()
