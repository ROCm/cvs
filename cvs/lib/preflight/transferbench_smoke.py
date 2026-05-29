"""IFoE TransferBench smoketest preflight check (AIMVT-181).

This module validates IFoE (Infinity Fabric over Ethernet, a.k.a. XGMI-over-
Ethernet) scale-up connectivity by:

1. Querying ``amd-smi fabric --topology --json`` on every reachable cluster
   node to discover each GPU's physical (``ppod_id``) and virtual / logical
   (``vpod_id``) pod membership. Because TransferBench's data-path smoketest
   exits early (precondition failure) when ranks span multiple virtual pods,
   we enforce that **all reachable nodes report the same singleton vPod** as
   a precondition before invoking the binary.

2. Running the TransferBench candidate-branch ``smoketest`` preset on each
   reachable node (either independently per-node or in multi-rank socket-comm
   mode). The smoketest exercises H2D, D2H, D2D, broadcast, gather, and
   all-to-all traffic on GPU DMA and GPU shader (GFX) executors, which is
   the closest data-path equivalent of "transferbench scale-up full-mesh
   over IFoE" that the candidate branch ships today.

3. Parsing the per-test PASS/FAIL/SKIP markers and any summary line from
   the smoketest's stdout, and reconciling them with the binary's exit code:

   - ``exit_code == 0`` and no ``FAIL`` markers → ``PASS``
   - ``exit_code != 0`` → ``FAIL`` regardless of marker parse success
     (this catches the ERR_FATAL=2 precondition exit emitted by the preset
     when symmetry / pod-membership checks fail inside TransferBench itself).
   - Skipped cells are tolerated up to ``max_skip_pct``; above the budget
     the result is marked ``WARNING``.

The check is **opt-in**: when ``connectivity_check.transferbench.connectivity_mode``
is ``"skip"`` (default) the test records a SKIPPED result and returns without
contacting nodes. Operators flip it to ``"run"`` once TransferBench's
candidate branch is installed cluster-wide.

The smoketest preset itself, and the ``amd-smi fabric`` topology fields, are
both required to run on real IFoE-equipped hardware; this module never
invents them. The unit tests use entirely synthetic fixtures.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cvs.lib.preflight.base import PreflightCheck


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TB_BINARY = 'TransferBench'
DEFAULT_AMD_SMI_PATH = 'amd-smi'
DEFAULT_PRESET = 'smoketest'
DEFAULT_SIZE_LIST = ['1K', '16M']
DEFAULT_NUM_ITERATIONS = 2
DEFAULT_NUM_WARMUPS = 0
DEFAULT_SOCKET_MASTER_PORT = 31337
DEFAULT_SSH_TIMEOUT = 600
DEFAULT_MAX_SKIP_PCT = 25.0

EXIT_CODE_PASS = 0
EXIT_CODE_FATAL_PRECONDITION = 2

# Sentinel suffix appended to every per-node command so we can recover the
# exit code from stdout even when the parallel SSH layer discards it.
EXIT_SENTINEL = '__TB_SMOKE_EXIT__'

# Marker tokens emitted by the candidate-branch SmokeTest preset table.
MARKER_PASS = '*'
MARKER_FAIL = 'F'
MARKER_SKIP = '.'


# ---------------------------------------------------------------------------
# amd-smi fabric topology parsers
# ---------------------------------------------------------------------------


def _coerce_int(value: Any) -> Optional[int]:
    """Return ``int(value)`` or ``None`` for missing/non-numeric values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s or s.upper() in {'N/A', 'NA', 'NONE', 'NULL'}:
            return None
        try:
            return int(s, 0)
        except ValueError:
            try:
                return int(float(s))
            except ValueError:
                return None
    return None


def _walk_gpu_records(payload: Any) -> Iterable[Dict[str, Any]]:
    """Yield dict-shaped GPU records from arbitrary amd-smi JSON shapes.

    amd-smi has historically emitted topology JSON in several flavours:
      - a top-level list of GPU records,
      - a dict with a ``gpu_data`` list (similar to the ECC/PCIe wrappers),
      - a dict keyed by GPU index whose values are GPU records,
      - an envelope dict with ``data`` / ``payload`` lists.
    This walker handles the common cases tolerantly without requiring a
    specific schema version.
    """
    if payload is None:
        return
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                yield entry
        return
    if not isinstance(payload, dict):
        return
    for key in ('gpu_data', 'gpus', 'data', 'payload'):
        inner = payload.get(key)
        if isinstance(inner, list):
            for entry in inner:
                if isinstance(entry, dict):
                    yield entry
            return
    looks_like_gpu_record = any(
        k in payload for k in ('gpu', 'gpu_id', 'bdf', 'vpod_id', 'ppod_id', 'fabric')
    )
    if looks_like_gpu_record:
        yield payload
        return
    for value in payload.values():
        if isinstance(value, dict) and any(
            k in value for k in ('gpu', 'gpu_id', 'bdf', 'vpod_id', 'ppod_id', 'fabric')
        ):
            yield value


def _extract_pod_ids_from_record(record: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """Return ``{ppod_id, vpod_id, ppod_size, vpod_size}`` for one GPU record.

    Looks for the fields at the top level and inside a nested ``fabric`` /
    ``pod`` / ``topology`` dict. Returns ``None`` for any field that is not
    present or cannot be coerced to int.
    """
    keys = ('ppod_id', 'vpod_id', 'ppod_size', 'vpod_size')
    candidates: List[Dict[str, Any]] = [record]
    for nested_key in ('fabric', 'pod', 'topology', 'pod_info', 'membership'):
        nested = record.get(nested_key)
        if isinstance(nested, dict):
            candidates.append(nested)
    out: Dict[str, Optional[int]] = {k: None for k in keys}
    for cand in candidates:
        for key in keys:
            if out[key] is None and key in cand:
                out[key] = _coerce_int(cand[key])
    return out


_PLAINTEXT_KV_RE = re.compile(r'^\s*([A-Z_][A-Z0-9_]*)\s*[:=]\s*(.+?)\s*$', re.IGNORECASE)


def parse_amd_smi_fabric_text(text: str) -> List[Dict[str, Optional[int]]]:
    """Parse the ``amd-smi fabric --topology`` plaintext output.

    Used as a fallback when the cluster's amd-smi build does not honour
    ``--json``. Splits the output into per-GPU blocks (separated by ``GPU``
    headers or blank lines) and extracts ``PPOD_ID`` / ``VPOD_ID`` /
    ``PPOD_SIZE`` / ``VPOD_SIZE`` rows.

    Returns:
        list[dict]: One dict per detected GPU block. Each dict has the same
        shape as ``_extract_pod_ids_from_record`` plus ``gpu`` (index or
        identifier string parsed from the block header).
    """
    if not text:
        return []
    lines = text.splitlines()
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if re.match(r'^\s*(GPU|ACCELERATOR|DEVICE)\b', line, re.IGNORECASE):
            if current:
                blocks.append(current)
            current = [line]
        elif not line.strip():
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    parsed_blocks: List[Dict[str, Optional[int]]] = []
    for block in blocks:
        record: Dict[str, Any] = {}
        header = block[0]
        m = re.search(r'^\s*(?:GPU|ACCELERATOR|DEVICE)\s*[:#]?\s*([A-Za-z0-9_.:-]+)', header, re.IGNORECASE)
        if m:
            record['gpu'] = m.group(1)
        for line in block[1:]:
            kv = _PLAINTEXT_KV_RE.match(line)
            if not kv:
                continue
            key = kv.group(1).strip().lower()
            value = kv.group(2).strip()
            if key in ('ppod_id', 'vpod_id', 'ppod_size', 'vpod_size'):
                record[key] = value
        pod_ids = _extract_pod_ids_from_record(record)
        if any(v is not None for v in pod_ids.values()):
            entry = dict(pod_ids)
            if 'gpu' in record:
                entry['gpu'] = record['gpu']
            parsed_blocks.append(entry)
    return parsed_blocks


def extract_node_pod_membership(node_payload: Any) -> Dict[str, Any]:
    """Reduce an ``amd-smi fabric --topology`` payload to per-node summary.

    Accepts either:
      - the parsed JSON object returned by ``rocm_plib.get_gpu_fabric_info_dict``
        for a single node, or
      - the raw stdout string (used as plaintext-fallback input).

    Returns:
        dict with:
          - ``gpus`` (int): number of GPU records parsed from the payload.
          - ``vpod_ids`` (sorted list[int]): unique vPod IDs observed.
          - ``ppod_ids`` (sorted list[int]): unique pPod IDs observed.
          - ``vpod_size`` (int | None): node-wide majority vpod_size (or None).
          - ``ppod_size`` (int | None): node-wide majority ppod_size (or None).
          - ``errors`` (list[str]): parsing/observation issues.

        A node with a single uniform vPod will have
        ``len(vpod_ids) == 1`` and ``errors == []``.
    """
    out = {
        'gpus': 0,
        'vpod_ids': [],
        'ppod_ids': [],
        'vpod_size': None,
        'ppod_size': None,
        'errors': [],
    }
    records: List[Dict[str, Any]] = []
    if isinstance(node_payload, str):
        plaintext_records = parse_amd_smi_fabric_text(node_payload)
        records.extend(plaintext_records)
    else:
        for entry in _walk_gpu_records(node_payload):
            records.append(entry)

    pod_id_dicts: List[Dict[str, Optional[int]]] = []
    for record in records:
        ids = _extract_pod_ids_from_record(record)
        if any(v is not None for v in ids.values()):
            pod_id_dicts.append(ids)

    out['gpus'] = len(pod_id_dicts)
    if not pod_id_dicts:
        out['errors'].append(
            'amd-smi fabric payload contained no GPU records with vpod_id/ppod_id fields'
        )
        return out

    vpods = sorted({d['vpod_id'] for d in pod_id_dicts if d['vpod_id'] is not None})
    ppods = sorted({d['ppod_id'] for d in pod_id_dicts if d['ppod_id'] is not None})
    out['vpod_ids'] = vpods
    out['ppod_ids'] = ppods

    if not vpods:
        out['errors'].append('No vpod_id values reported by amd-smi fabric topology')
    elif len(vpods) > 1:
        out['errors'].append(
            f'Multiple vPod IDs reported by a single node: {vpods} '
            f'(TransferBench smoketest requires uniform vpod_id across local GPUs)'
        )

    sizes_v = [d['vpod_size'] for d in pod_id_dicts if d['vpod_size'] is not None]
    if sizes_v:
        out['vpod_size'] = max(set(sizes_v), key=sizes_v.count)
    sizes_p = [d['ppod_size'] for d in pod_id_dicts if d['ppod_size'] is not None]
    if sizes_p:
        out['ppod_size'] = max(set(sizes_p), key=sizes_p.count)
    return out


def reconcile_cluster_vpod(
    per_node_membership: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Cross-check pod membership across nodes for a smoketest precondition.

    Args:
        per_node_membership: ``{node: extract_node_pod_membership(...) dict}``.

    Returns:
        dict with:
          - ``status``: ``'PASS'`` if every node reports the same singleton
            vPod ID, ``'FAIL'`` otherwise.
          - ``vpod_id``: the shared vpod_id (``int``) when PASS, else ``None``.
          - ``ppod_id``: the shared ppod_id when PASS-and-all-agree, else
            ``None`` (only informational; the smoketest precondition is on
            vPod).
          - ``per_node``: copy of the input.
          - ``errors``: list of human-readable explanations of any mismatch.
    """
    result: Dict[str, Any] = {
        'status': 'PASS',
        'vpod_id': None,
        'ppod_id': None,
        'per_node': dict(per_node_membership),
        'errors': [],
    }
    if not per_node_membership:
        result['status'] = 'FAIL'
        result['errors'].append('No nodes reported amd-smi fabric topology')
        return result

    nodes_without_vpod = [n for n, m in per_node_membership.items() if not m.get('vpod_ids')]
    if nodes_without_vpod:
        result['status'] = 'FAIL'
        result['errors'].append(
            f'{len(nodes_without_vpod)} node(s) missing vpod_id from amd-smi fabric: '
            + ', '.join(sorted(nodes_without_vpod)[:10])
        )

    multi_vpod_nodes = [
        n for n, m in per_node_membership.items() if len(m.get('vpod_ids') or []) > 1
    ]
    if multi_vpod_nodes:
        result['status'] = 'FAIL'
        result['errors'].append(
            f'{len(multi_vpod_nodes)} node(s) report multiple local vPod IDs: '
            + ', '.join(sorted(multi_vpod_nodes)[:10])
        )

    singleton_pairs = {
        n: m['vpod_ids'][0]
        for n, m in per_node_membership.items()
        if len(m.get('vpod_ids') or []) == 1
    }
    distinct_vpods = sorted(set(singleton_pairs.values()))
    if len(distinct_vpods) > 1:
        result['status'] = 'FAIL'
        groups: Dict[int, List[str]] = {}
        for n, vp in singleton_pairs.items():
            groups.setdefault(vp, []).append(n)
        group_strs = [
            f'vpod_id={vp}: {len(groups[vp])} node(s) ({", ".join(sorted(groups[vp])[:5])}'
            + (', ...' if len(groups[vp]) > 5 else '')
            + ')'
            for vp in distinct_vpods
        ]
        result['errors'].append(
            'Reachable nodes span multiple vPods; TransferBench smoketest requires a single vPod. '
            + ' | '.join(group_strs)
        )

    if result['status'] == 'PASS' and distinct_vpods:
        result['vpod_id'] = distinct_vpods[0]
        ppod_sets = [tuple(m.get('ppod_ids') or []) for m in per_node_membership.values()]
        if ppod_sets and all(len(s) == 1 and s == ppod_sets[0] for s in ppod_sets):
            result['ppod_id'] = ppod_sets[0][0]
    return result


# ---------------------------------------------------------------------------
# Smoketest output parser
# ---------------------------------------------------------------------------


_BRACKET_VERDICT_RE = re.compile(r'\[\s*(PASS|FAIL|SKIP(?:PED)?|ERROR)\s*\]', re.IGNORECASE)
_TEST_LINE_NUMBER_RE = re.compile(r'^\s*(?:Test\s+)?#?\s*(\d+)\b', re.IGNORECASE)
_SUMMARY_LINE_RE = re.compile(
    r'(?P<passed>\d+)\s*/\s*(?P<total>\d+)\s*(?:tests?\s*)?PASS', re.IGNORECASE
)
_SECONDARY_COUNT_RE = re.compile(
    r'(?P<n>\d+)\s+(?P<label>FAIL|SKIP(?:PED)?|ERROR)', re.IGNORECASE
)
_WARNING_LINE_RE = re.compile(r'^\s*(?:WARN(?:ING)?|NOTE|INFO)\s*[:\-]\s*(.+)$', re.IGNORECASE)
_FAIL_KEYWORD_RE = re.compile(r'\b(FAILED|FATAL|ERR_FATAL|FAILURE|ABORTED?)\b', re.IGNORECASE)
_SKIP_KEYWORD_RE = re.compile(r'\b(SKIP(?:PED)?|SKIPPING)\b', re.IGNORECASE)
_MARKER_BLOCK_RE = re.compile(r'^[ \t]*[' + re.escape(MARKER_PASS + MARKER_FAIL + MARKER_SKIP) + r']{4,}\s*$')
_EXIT_SENTINEL_RE = re.compile(
    re.escape(EXIT_SENTINEL) + r'\s*=\s*(?P<code>-?\d+)\s*$', re.MULTILINE
)


class SmoketestParser:
    """Tolerant parser for TransferBench candidate-branch ``smoketest`` output.

    The candidate-branch preset prints a per-test results section followed by
    an aggregate summary. Real-world outputs vary across builds (cell markers
    ``*/F/.`` vs. ``[PASS]/[FAIL]/[SKIP]`` vs. ``Test N: ... PASSED``), so
    the parser accepts all three shapes and treats the binary's exit code as
    the authoritative pass/fail signal. Marker counts are reported as
    diagnostics and used for the optional skip-budget gate.
    """

    @classmethod
    def parse(cls, output: str) -> Dict[str, Any]:
        """Return parsed structured smoketest result.

        Args:
            output: Raw stdout (+stderr if merged) from the TransferBench
                smoketest invocation, optionally including the
                ``__TB_SMOKE_EXIT__=<code>`` sentinel line appended by the
                orchestrator.

        Returns:
            dict with keys:
              - ``exit_code`` (int | None): parsed from the sentinel line; ``None``
                when not present (treated by the orchestrator as FAIL).
              - ``num_pass``, ``num_fail``, ``num_skip`` (int): counts derived from
                whichever marker style was found.
              - ``num_tests`` (int): ``num_pass + num_fail + num_skip``.
              - ``summary_total``, ``summary_pass`` (int | None): values parsed
                from a "N/M PASS" summary line, if present.
              - ``per_test`` (list[dict]): one entry per detected test row with
                ``index`` (str | None), ``status`` (PASS/FAIL/SKIP), and
                ``raw`` (the original line).
              - ``warnings`` (list[str]): WARN / NOTE / SKIPPED reason lines.
              - ``errors`` (list[str]): lines containing FAILED / FATAL /
                ``ERR_FATAL`` and similar fatal-error keywords.
              - ``parse_errors`` (list[str]): notes about ambiguous output
                shapes (empty when parsing succeeded cleanly).
              - ``raw`` (str): stripped raw output (sentinel removed).
        """
        result: Dict[str, Any] = {
            'exit_code': None,
            'num_pass': 0,
            'num_fail': 0,
            'num_skip': 0,
            'num_tests': 0,
            'summary_total': None,
            'summary_pass': None,
            'per_test': [],
            'warnings': [],
            'errors': [],
            'parse_errors': [],
            'raw': '',
        }
        if not output:
            result['parse_errors'].append('Empty smoketest output')
            return result

        text = output
        m = _EXIT_SENTINEL_RE.search(text)
        if m:
            try:
                result['exit_code'] = int(m.group('code'))
            except ValueError:
                pass
            text = _EXIT_SENTINEL_RE.sub('', text).rstrip()
        result['raw'] = text

        cls._collect_per_test_rows(text, result)
        cls._collect_marker_blocks(text, result)
        cls._collect_summary(text, result)
        cls._collect_warnings_and_errors(text, result)

        result['num_tests'] = result['num_pass'] + result['num_fail'] + result['num_skip']
        if result['num_tests'] == 0 and result['summary_total']:
            result['num_tests'] = result['summary_total']
        return result

    @staticmethod
    def _collect_per_test_rows(text: str, result: Dict[str, Any]) -> None:
        """Capture ``Test N: ... [PASS]`` style rows."""
        for line in text.splitlines():
            verdict = _BRACKET_VERDICT_RE.search(line)
            if not verdict:
                continue
            status_token = verdict.group(1).upper()
            if status_token.startswith('SKIP'):
                status = 'SKIP'
            elif status_token == 'FAIL':
                status = 'FAIL'
            elif status_token == 'ERROR':
                status = 'FAIL'
            else:
                status = 'PASS'
            idx_match = _TEST_LINE_NUMBER_RE.match(line)
            index = idx_match.group(1) if idx_match else None
            result['per_test'].append({'index': index, 'status': status, 'raw': line.strip()})
            if status == 'PASS':
                result['num_pass'] += 1
            elif status == 'FAIL':
                result['num_fail'] += 1
            elif status == 'SKIP':
                result['num_skip'] += 1

    @staticmethod
    def _collect_marker_blocks(text: str, result: Dict[str, Any]) -> None:
        """Capture compact marker blocks like ``****F.*.``.

        Only consulted when no bracketed verdict rows were found; otherwise
        the per-test rows already provide the counts we need.
        """
        if result['per_test']:
            return
        for line in text.splitlines():
            if not _MARKER_BLOCK_RE.match(line):
                continue
            for ch in line.strip():
                if ch == MARKER_PASS:
                    result['num_pass'] += 1
                elif ch == MARKER_FAIL:
                    result['num_fail'] += 1
                elif ch == MARKER_SKIP:
                    result['num_skip'] += 1

    @staticmethod
    def _collect_summary(text: str, result: Dict[str, Any]) -> None:
        """Parse aggregate ``N/M PASS, x FAIL, y SKIP`` lines."""
        for line in text.splitlines():
            m = _SUMMARY_LINE_RE.search(line)
            if not m:
                continue
            try:
                summary_pass = int(m.group('passed'))
                summary_total = int(m.group('total'))
            except (TypeError, ValueError):
                continue
            if result['summary_total'] is None:
                result['summary_pass'] = summary_pass
                result['summary_total'] = summary_total
            for sec in _SECONDARY_COUNT_RE.finditer(line):
                try:
                    count = int(sec.group('n'))
                except ValueError:
                    continue
                label = sec.group('label').upper()
                if label.startswith('SKIP'):
                    result['num_skip'] = max(result['num_skip'], count)
                elif label in ('FAIL', 'ERROR'):
                    result['num_fail'] = max(result['num_fail'], count)
            if (
                summary_total > 0
                and not result['per_test']
                and result['num_pass'] + result['num_fail'] + result['num_skip'] == 0
            ):
                result['num_pass'] = summary_pass
                inferred_remainder = max(0, summary_total - summary_pass - result['num_fail'])
                if result['num_skip'] == 0 and inferred_remainder > 0:
                    result['num_fail'] = max(result['num_fail'], inferred_remainder - result['num_skip'])

    @staticmethod
    def _collect_warnings_and_errors(text: str, result: Dict[str, Any]) -> None:
        """Collect WARN/NOTE lines and fatal-keyword error lines."""
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            wm = _WARNING_LINE_RE.match(stripped)
            if wm:
                result['warnings'].append(stripped)
                continue
            if _FAIL_KEYWORD_RE.search(stripped):
                if _BRACKET_VERDICT_RE.search(stripped):
                    continue
                result['errors'].append(stripped)
            elif _SKIP_KEYWORD_RE.search(stripped) and 'reason' in stripped.lower():
                result['warnings'].append(stripped)


# ---------------------------------------------------------------------------
# Smoketest verdict logic
# ---------------------------------------------------------------------------


def evaluate_smoketest(
    parsed: Dict[str, Any],
    max_skip_pct: float = DEFAULT_MAX_SKIP_PCT,
) -> Tuple[str, List[str]]:
    """Derive the per-node PASS/FAIL/WARNING verdict from a parsed result.

    Verdict precedence (first matching rule wins):
      1. ``exit_code is None`` -> ``FAIL`` (no sentinel; orchestration broke
         or the command was killed before printing).
      2. ``exit_code == EXIT_CODE_FATAL_PRECONDITION`` (= 2) -> ``FAIL`` with
         a precondition message (this is how the candidate-branch preset
         signals symmetry / pod-membership failures).
      3. ``exit_code != 0`` -> ``FAIL``.
      4. ``num_fail > 0`` or any ``errors`` -> ``FAIL`` (defence in depth in
         case the binary somehow exited zero with FAIL cells, which would
         itself be a bug worth flagging).
      5. ``num_skip / num_tests > max_skip_pct / 100`` -> ``WARNING``.
      6. otherwise -> ``PASS``.

    Returns:
        Tuple of (``verdict``, ``errors``).
    """
    errors: List[str] = []
    exit_code = parsed.get('exit_code')
    if exit_code is None:
        errors.append(
            'TransferBench smoketest exit code not captured (process may have been killed '
            'before completing or stdout was truncated)'
        )
        return 'FAIL', errors
    if exit_code == EXIT_CODE_FATAL_PRECONDITION:
        errors.append(
            'TransferBench smoketest aborted with ERR_FATAL precondition (exit 2): '
            'pod-membership or executor symmetry check inside the preset failed'
        )
        return 'FAIL', errors
    if exit_code != EXIT_CODE_PASS:
        errors.append(f'TransferBench smoketest non-zero exit code: {exit_code}')
        return 'FAIL', errors
    if parsed.get('num_fail', 0) > 0:
        errors.append(
            f"TransferBench smoketest reported {parsed['num_fail']} FAIL marker(s) despite "
            f"exit code 0 -- treating as FAIL"
        )
        return 'FAIL', errors
    if parsed.get('errors'):
        errors.append('TransferBench stdout contained FAIL/FATAL keyword lines')
        return 'FAIL', errors

    num_tests = parsed.get('num_tests') or 0
    num_skip = parsed.get('num_skip') or 0
    if num_tests > 0 and max_skip_pct >= 0:
        skip_pct = (num_skip / num_tests) * 100.0
        if skip_pct > max_skip_pct + 1e-9:
            errors.append(
                f'TransferBench smoketest skipped {num_skip}/{num_tests} tests '
                f'({skip_pct:.1f}% > max_skip_pct={max_skip_pct}%)'
            )
            return 'WARNING', errors

    return 'PASS', errors


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


RankMode = str  # 'per_node' or 'multi_rank'


class TransferBenchSmokeCheck(PreflightCheck):
    """Run the TransferBench candidate-branch smoketest preset cluster-wide.

    The check is two-phase:

      1. **Precondition** – Query ``amd-smi fabric --topology --json`` on
         every reachable node, then enforce that all nodes report the same
         singleton vPod ID (TransferBench multi-rank smoketest exits with
         ``ERR_FATAL`` when ranks span multiple vPods, and the per-node
         single-rank case is meaningless when the cluster's IFoE fabric was
         intended to span more than one node).
      2. **Smoketest** – Invoke ``TransferBench smoketest`` on each node and
         parse the output. Two orchestration modes are supported:

         - ``per_node`` (default): each node runs the preset independently
           against its local GPUs (``TB_NUM_RANKS=1``). Exercises intra-node
           AID↔MID IFoE hops, but does not traverse the rack IFoE switch.
         - ``multi_rank``: ``N`` reachable nodes coordinate via the preset's
           built-in socket-comm (``TB_NUM_RANKS=N``, ``TB_RANK=0..N-1``,
           ``TB_MASTER_ADDR=<rank0 IP>``). This is the closest thing to a
           full-mesh IFoE scale-up traffic test and is the recommended mode
           when topology supports it; we fall back to ``per_node`` if there
           are fewer than two reachable nodes.

    All TransferBench-specific tuning knobs (preset name, size list,
    iterations, validate / parallel / BDMA flags, master port) are
    configurable so the same module can be repointed at future smoketest
    variants without code changes.
    """

    def __init__(
        self,
        phdl,
        *,
        tb_binary: str = DEFAULT_TB_BINARY,
        rocm_path: Optional[str] = None,
        amd_smi_path: str = DEFAULT_AMD_SMI_PATH,
        use_sudo: bool = False,
        preset: str = DEFAULT_PRESET,
        size_list: Optional[Sequence[str]] = None,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
        num_warmups: int = DEFAULT_NUM_WARMUPS,
        always_validate: bool = True,
        run_parallel: bool = True,
        use_bdma: bool = False,
        force_single_pod: bool = True,
        rank_mode: RankMode = 'per_node',
        socket_master_port: int = DEFAULT_SOCKET_MASTER_PORT,
        master_node: Optional[str] = None,
        max_skip_pct: float = DEFAULT_MAX_SKIP_PCT,
        ssh_timeout: int = DEFAULT_SSH_TIMEOUT,
        extra_env: Optional[Dict[str, str]] = None,
        extra_args: Optional[Sequence[str]] = None,
        skip_pod_check: bool = False,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(phdl, config_dict)
        self.tb_binary = tb_binary
        self.rocm_path = (rocm_path or '').strip()
        self.amd_smi_path = amd_smi_path
        self.use_sudo = bool(use_sudo)
        self.preset = preset
        self.size_list = list(size_list) if size_list else list(DEFAULT_SIZE_LIST)
        self.num_iterations = int(num_iterations)
        self.num_warmups = int(num_warmups)
        self.always_validate = bool(always_validate)
        self.run_parallel = bool(run_parallel)
        self.use_bdma = bool(use_bdma)
        self.force_single_pod = bool(force_single_pod)
        self.socket_master_port = int(socket_master_port)
        self.master_node = master_node
        self.max_skip_pct = float(max_skip_pct)
        self.ssh_timeout = int(ssh_timeout)
        self.extra_env = dict(extra_env or {})
        self.extra_args = list(extra_args or [])
        self.skip_pod_check = bool(skip_pod_check)

        rank_mode_norm = (rank_mode or 'per_node').strip().lower()
        if rank_mode_norm not in ('per_node', 'multi_rank'):
            raise ValueError(
                f"rank_mode must be 'per_node' or 'multi_rank' (got '{rank_mode}')"
            )
        self.rank_mode = rank_mode_norm

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _env_assignments(self, rank: int, num_ranks: int, master_addr: str) -> List[str]:
        """Build ``KEY=VALUE`` assignments for one rank's TransferBench invocation."""
        env: Dict[str, str] = {}
        env['NUM_ITERATIONS'] = str(self.num_iterations)
        env['NUM_WARMUPS'] = str(self.num_warmups)
        env['ALWAYS_VALIDATE'] = '1' if self.always_validate else '0'
        env['USE_REMOTE_READ'] = '1'
        env['BLOCK_BYTES'] = '256'
        if self.run_parallel:
            env['RUN_PARALLEL'] = '1'
        if self.use_bdma:
            env['USE_BDMA'] = '1'
        if self.force_single_pod:
            env['FORCE_SINGLE_POD'] = '1'
        if num_ranks > 1:
            env['TB_NUM_RANKS'] = str(num_ranks)
            env['TB_RANK'] = str(rank)
            env['TB_MASTER_ADDR'] = master_addr
            env['TB_MASTER_PORT'] = str(self.socket_master_port)
        for k, v in self.extra_env.items():
            env[k] = str(v)
        return [f'{k}={shlex.quote(v)}' for k, v in env.items()]

    def _rocm_env_prefix(self) -> str:
        """ROCm PATH/LD_LIBRARY_PATH prefix evaluated inside the inner shell."""
        if not self.rocm_path:
            return ''
        rocm_bin = shlex.quote(f'{self.rocm_path}/bin')
        rocm_lib = shlex.quote(f'{self.rocm_path}/lib')
        return f'PATH={rocm_bin}:$PATH LD_LIBRARY_PATH={rocm_lib}:${{LD_LIBRARY_PATH:-}}'

    def build_command(
        self,
        rank: int,
        num_ranks: int,
        master_addr: str,
        size_list: Optional[Sequence[str]] = None,
    ) -> str:
        """Render a complete shell command for one rank.

        Env-var prefixes (TB_*, NUM_ITERATIONS, etc.) and ROCm PATH/library
        path mods are placed **inside** the inner shell so that, even when
        ``use_sudo=True``, the privileged child process sees them: ``sudo``
        otherwise sanitizes the calling shell's environment. The command
        also appends a sentinel ``__TB_SMOKE_EXIT__=<code>`` line so the
        orchestrator can recover the binary's exit code from stdout when
        the parallel SSH layer discards process exit codes.
        """
        env_parts = self._env_assignments(rank, num_ranks, master_addr)
        binary = shlex.quote(self.tb_binary)
        preset = shlex.quote(self.preset)
        sizes = list(size_list) if size_list is not None else self.size_list
        size_args = ' '.join(shlex.quote(str(s)) for s in sizes)
        extras = ' '.join(shlex.quote(s) for s in self.extra_args)
        env_inline = ' '.join([self._rocm_env_prefix(), *env_parts]).strip()
        binary_invocation = f'{binary} {preset} {size_args} {extras}'.strip()
        if env_inline:
            inner = f'{env_inline} {binary_invocation}'
        else:
            inner = binary_invocation
        inner_with_sentinel = f'{inner}; echo "{EXIT_SENTINEL}=$?"'
        if self.use_sudo:
            return f'sudo bash -c {shlex.quote(inner_with_sentinel)}'
        return f'bash -c {shlex.quote(inner_with_sentinel)}'

    # ------------------------------------------------------------------
    # Precondition: pod membership
    # ------------------------------------------------------------------

    def _amd_smi_fabric_command(self) -> str:
        cmd = f'{self.amd_smi_path} fabric --topology --json'
        if self.use_sudo:
            cmd = 'sudo ' + cmd
        return cmd

    def _query_pod_membership(self) -> Dict[str, Dict[str, Any]]:
        """Run ``amd-smi fabric --topology --json`` on each reachable node.

        Tolerates plaintext (non-JSON) outputs by passing the raw string to
        ``extract_node_pod_membership``, which then routes it through the
        plaintext fallback parser.
        """
        cmd = self._amd_smi_fabric_command()
        self.log_info(f'Querying pod membership: {cmd}')
        out_dict = self.phdl.exec(cmd, timeout=min(60, self.ssh_timeout), print_console=False)
        per_node: Dict[str, Dict[str, Any]] = {}
        for node, output in out_dict.items():
            payload: Any
            if isinstance(output, str):
                stripped = output.strip()
                if stripped.startswith('{') or stripped.startswith('['):
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        payload = output
                else:
                    payload = output
            else:
                payload = output
            membership = extract_node_pod_membership(payload)
            membership['raw_output'] = output if isinstance(output, str) else ''
            per_node[node] = membership
        return per_node

    # ------------------------------------------------------------------
    # Smoketest dispatch
    # ------------------------------------------------------------------

    def _resolve_master_node(self) -> Optional[str]:
        if self.master_node and self.master_node in self.phdl.reachable_hosts:
            return self.master_node
        if self.phdl.reachable_hosts:
            return sorted(self.phdl.reachable_hosts)[0]
        return None

    def _dispatch_per_node(self) -> Dict[str, Dict[str, Any]]:
        """One independent smoketest per node, all in parallel."""
        cmd = self.build_command(rank=0, num_ranks=1, master_addr='127.0.0.1')
        self.log_info(f"Dispatching per-node smoketest: {cmd}")
        out_dict = self.phdl.exec(cmd, timeout=self.ssh_timeout, print_console=False)
        return {node: {'command': cmd, 'output': output} for node, output in out_dict.items()}

    def _dispatch_multi_rank(self) -> Dict[str, Dict[str, Any]]:
        """Multi-rank socket-comm dispatch (one rank per reachable node).

        Each node receives a distinct command (different ``TB_RANK`` value)
        constructed in deterministic hostname-sorted order. Sent as a single
        ``exec_cmd_list`` call so all ranks start in parallel and the
        preset's socket-comm bootstrap can complete.
        """
        hosts = list(self.phdl.reachable_hosts)
        if not hosts:
            return {}
        master = self._resolve_master_node()
        if not master:
            return {}
        ordered = [master] + sorted([h for h in hosts if h != master])
        num_ranks = len(ordered)
        cmd_list: List[str] = []
        per_node: Dict[str, Dict[str, Any]] = {}
        for rank, node in enumerate(ordered):
            cmd = self.build_command(rank=rank, num_ranks=num_ranks, master_addr=master)
            cmd_list.append(cmd)
            per_node[node] = {'command': cmd, 'output': '', 'rank': rank}

        # phdl.exec_cmd_list aligns commands to reachable_hosts ordering. We
        # rebuild the list in that ordering so each host gets its assigned
        # rank's command.
        rank_by_host = {node: rank for rank, node in enumerate(ordered)}
        cmd_by_reachable = [
            self.build_command(
                rank=rank_by_host[h], num_ranks=num_ranks, master_addr=master
            )
            for h in hosts
        ]
        self.log_info(
            f'Dispatching multi-rank smoketest (num_ranks={num_ranks}, master={master})'
        )
        out_dict = self.phdl.exec_cmd_list(
            cmd_by_reachable, timeout=self.ssh_timeout, print_console=False
        )
        for h, output in out_dict.items():
            if h not in per_node:
                per_node[h] = {'command': '', 'output': output}
            per_node[h]['output'] = output
        return per_node

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute pod-membership precondition + smoketest dispatch.

        Returns:
            dict with:
              - ``status``: ``'PASS'`` / ``'FAIL'`` / ``'WARNING'`` for the
                overall check.
              - ``rank_mode``: orchestration mode used (may differ from the
                requested mode if we degraded to ``per_node`` due to too few
                reachable nodes).
              - ``pod_membership``: cross-node reconcile result from
                ``reconcile_cluster_vpod``.
              - ``nodes``: ``{node: {status, errors, command, exit_code,
                parsed, raw_output, rank}}`` for every reachable node.
              - ``totals``: ``{nodes_total, nodes_pass, nodes_warning,
                nodes_fail, tests_pass, tests_fail, tests_skip}``.
              - ``errors``: cluster-level error messages (precondition,
                orchestration failures).
        """
        self.results = {
            'status': 'PASS',
            'rank_mode': self.rank_mode,
            'pod_membership': {},
            'nodes': {},
            'totals': {
                'nodes_total': 0,
                'nodes_pass': 0,
                'nodes_warning': 0,
                'nodes_fail': 0,
                'tests_pass': 0,
                'tests_fail': 0,
                'tests_skip': 0,
            },
            'errors': [],
        }

        hosts = list(self.phdl.reachable_hosts)
        if not hosts:
            self.results['status'] = 'FAIL'
            self.results['errors'].append(
                'No reachable hosts available for TransferBench smoketest'
            )
            return self.results
        self.results['totals']['nodes_total'] = len(hosts)

        if not self.skip_pod_check:
            per_node_membership = self._query_pod_membership()
            reconcile = reconcile_cluster_vpod(per_node_membership)
            self.results['pod_membership'] = reconcile
            if reconcile['status'] != 'PASS':
                self.results['status'] = 'FAIL'
                self.results['errors'].extend(reconcile.get('errors') or [])
                for node in hosts:
                    self.results['nodes'][node] = {
                        'status': 'SKIPPED',
                        'errors': ['pod-membership precondition failed; smoketest not dispatched'],
                        'command': '',
                        'exit_code': None,
                        'parsed': {},
                        'raw_output': '',
                        'rank': None,
                    }
                return self.results
        else:
            self.results['pod_membership'] = {
                'status': 'SKIPPED',
                'errors': [],
                'vpod_id': None,
                'ppod_id': None,
                'per_node': {},
            }
            self.log_warning('skip_pod_check=True; vPod precondition will not be enforced')

        effective_rank_mode = self.rank_mode
        if self.rank_mode == 'multi_rank' and len(hosts) < 2:
            self.log_warning(
                f'multi_rank requested but only {len(hosts)} reachable node(s); '
                f'falling back to per_node'
            )
            effective_rank_mode = 'per_node'
        self.results['rank_mode'] = effective_rank_mode

        try:
            if effective_rank_mode == 'multi_rank':
                dispatch = self._dispatch_multi_rank()
            else:
                dispatch = self._dispatch_per_node()
        except Exception as exc:  # pragma: no cover - defensive
            self.results['status'] = 'FAIL'
            self.results['errors'].append(f'Smoketest dispatch failed: {exc}')
            return self.results

        totals = self.results['totals']
        for node in hosts:
            entry = dispatch.get(node) or {}
            output = entry.get('output') or ''
            command = entry.get('command') or ''
            parsed = SmoketestParser.parse(output if isinstance(output, str) else '')
            verdict, verdict_errors = evaluate_smoketest(parsed, max_skip_pct=self.max_skip_pct)
            self.results['nodes'][node] = {
                'status': verdict,
                'errors': verdict_errors,
                'command': command,
                'exit_code': parsed.get('exit_code'),
                'parsed': parsed,
                'raw_output': output if isinstance(output, str) else '',
                'rank': entry.get('rank'),
            }
            if verdict == 'PASS':
                totals['nodes_pass'] += 1
            elif verdict == 'WARNING':
                totals['nodes_warning'] += 1
            else:
                totals['nodes_fail'] += 1
            totals['tests_pass'] += parsed.get('num_pass', 0)
            totals['tests_fail'] += parsed.get('num_fail', 0)
            totals['tests_skip'] += parsed.get('num_skip', 0)

        if totals['nodes_fail']:
            self.results['status'] = 'FAIL'
        elif totals['nodes_warning']:
            self.results['status'] = 'WARNING'
        else:
            self.results['status'] = 'PASS'
        return self.results


__all__ = [
    'TransferBenchSmokeCheck',
    'SmoketestParser',
    'evaluate_smoketest',
    'extract_node_pod_membership',
    'reconcile_cluster_vpod',
    'parse_amd_smi_fabric_text',
    'EXIT_SENTINEL',
    'EXIT_CODE_PASS',
    'EXIT_CODE_FATAL_PRECONDITION',
    'DEFAULT_SIZE_LIST',
    'DEFAULT_PRESET',
    'DEFAULT_TB_BINARY',
]
