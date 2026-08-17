'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

"""
CI robustness helpers for the RCCL regression pipeline (AIMVT-196):

  1. Retry of transient test failures (run_with_retries / classify_failure).
  2. Killing stale GPU-holding processes / containers before a run
     (build_gpu_cleanup_script / parse_gpu_pids).

The decision logic here is intentionally pure and dependency-free so it can be
unit-tested exhaustively on a login node. The thin cluster-facing wrapper that
actually executes the cleanup over SSH lives in rccl_lib.cleanup_gpus_on_nodes
and reuses the builders below.
"""

import re
import shlex
import time

# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

# Substrings/regexes that indicate a *transient* failure worth retrying.
DEFAULT_RETRIABLE_PATTERNS = [
    r'NCCL ERROR',
    r'unhandled system error',
    r'unhandled (cuda|hip) error',
    r'Test NCCL failure',
    r'ORTE',
    r'PML add procs failed',
    r'MPI_Init',
    r'ompi_mpi_init',
    r'Connection (refused|reset|timed out)',
    r'connection closed',
    r'timed out|timeout',
    r'Hit Exceptions',
    r'no bandwidth numbers',
    r'No route to host',
    r'socket',
    r'Software caused connection abort',
    r'remote process|process exited|exited on signal',
    r'no result rows',
]

# Substrings/regexes that indicate a *real* failure that must NOT be retried
# (retrying would only hide a genuine bug). These take precedence.
DEFAULT_NON_RETRIABLE_PATTERNS = [
    r'SEVERE DATA CORRUPTION',
    r"'#wrong'",
    r'wrong=',
    r'schema validation failed',
]


def classify_failure(message, retriable_patterns=None, non_retriable_patterns=None):
    """
    Decide whether a failure message describes a transient (retriable) error.

    Non-retriable patterns (e.g. data corruption) are checked first and win:
    we never want to paper over a correctness failure by retrying.

    Args:
      message: failure text (str or anything str()-able, e.g. an exception).
      retriable_patterns / non_retriable_patterns: optional override lists.

    Returns:
      bool: True if the failure looks transient and should be retried.
    """
    text = str(message)
    non_retriable = non_retriable_patterns if non_retriable_patterns is not None else DEFAULT_NON_RETRIABLE_PATTERNS
    retriable = retriable_patterns if retriable_patterns is not None else DEFAULT_RETRIABLE_PATTERNS

    for pat in non_retriable:
        if re.search(pat, text, re.IGNORECASE):
            return False
    for pat in retriable:
        if re.search(pat, text, re.IGNORECASE):
            return True
    # Unknown failures: default to retriable. A transient infra blip is the common
    # case in multi-node CI; genuine bugs are caught by the non-retriable list and
    # by the fact that a real regression reproduces on every retry anyway.
    return True


def run_with_retries(
    attempt_fn,
    max_retries=2,
    is_retriable=None,
    on_before_retry=None,
    backoff_sec=0,
    sleep_fn=time.sleep,
    log=None,
    label="task",
):
    """
    Call ``attempt_fn()`` up to ``max_retries + 1`` times.

    ``attempt_fn`` must raise an exception on failure and return a value on
    success. After a failing attempt, if more attempts remain and the failure is
    retriable, ``on_before_retry(next_attempt_index)`` is invoked (e.g. to clean
    up stale GPU state), then we sleep ``backoff_sec * attempt`` (linear backoff)
    and try again.

    Args:
      attempt_fn: zero-arg callable; raises on failure.
      max_retries: number of *extra* attempts after the first (so total = +1).
      is_retriable: callable(exc) -> bool. Defaults to classify_failure(str(exc)).
      on_before_retry: optional callable(next_attempt_index:int) run between attempts.
      backoff_sec: base backoff seconds (linear: backoff_sec * attempt_number).
      sleep_fn: injectable sleep (for tests).
      log: optional logger.
      label: short description for log lines.

    Returns:
      Whatever attempt_fn() returns on the first successful attempt.

    Raises:
      The last exception if all attempts fail or a failure is non-retriable.
    """
    if is_retriable is None:
        is_retriable = lambda exc: classify_failure(exc)
    total_attempts = max(1, max_retries + 1)
    last_exc = None
    for attempt in range(1, total_attempts + 1):
        try:
            return attempt_fn()
        except Exception as exc:  # noqa: BLE001 - we deliberately catch broadly to retry
            last_exc = exc
            retriable = bool(is_retriable(exc))
            more_attempts = attempt < total_attempts
            if log is not None:
                log.warning(
                    "%s attempt %d/%d failed: %r (retriable=%s)",
                    label, attempt, total_attempts, exc, retriable,
                )
            if not (more_attempts and retriable):
                raise
            if on_before_retry is not None:
                try:
                    on_before_retry(attempt + 1)
                except Exception as cleanup_exc:  # noqa: BLE001
                    if log is not None:
                        log.warning("%s on_before_retry hook failed (ignored): %r", label, cleanup_exc)
            if backoff_sec:
                sleep_fn(backoff_sec * attempt)
    # Should be unreachable, but re-raise the last error defensively.
    raise last_exc


# ---------------------------------------------------------------------------
# GPU / container cleanup
# ---------------------------------------------------------------------------

# Default process-name patterns for stale RCCL/MPI leftovers from prior or
# killed jobs. On an exclusive compute node any match is stale by definition.
DEFAULT_GPU_PROCESS_PATTERNS = [
    'all_reduce_perf',
    'all_reduce_bias_perf',
    'all_gather_perf',
    'reduce_scatter_perf',
    'broadcast_perf',
    'alltoall_perf',
    'alltoallv_perf',
    'sendrecv_perf',
    'scatter_perf',
    'gather_perf',
    'reduce_perf',
    'hypercube_perf',
    'mpirun',
    'orted',
    'prted',
]


def parse_gpu_pids(rocm_smi_showpids_output):
    """
    Extract PIDs from the output of ``rocm-smi --showpids``.

    The table rows begin with a numeric PID; header/border lines do not. We pull
    the leading integer from each line and drop obvious non-PIDs (0/1).

    Args:
      rocm_smi_showpids_output: str output of `rocm-smi --showpids`.

    Returns:
      sorted list[int] of unique candidate PIDs.
    """
    pids = set()
    for line in (rocm_smi_showpids_output or "").splitlines():
        m = re.match(r'^\s*(\d+)\b', line)
        if not m:
            continue
        pid = int(m.group(1))
        if pid > 1:
            pids.add(pid)
    return sorted(pids)


def _self_safe_pattern(pattern):
    """
    Rewrite a pkill -f pattern so it cannot match the cleanup command's own
    command line. Wrapping the first alphanumeric character in a regex class
    (e.g. 'orted' -> '[o]rted') matches the target process but not the literal
    pattern text present in our own argv. Classic pgrep/pkill self-match guard.
    """
    for i, ch in enumerate(pattern):
        if ch.isalnum():
            return pattern[:i] + '[' + ch + ']' + pattern[i + 1:]
    return pattern


def build_gpu_cleanup_script(
    process_patterns=None,
    kill_gpu_pids=True,
    kill_containers=False,
    use_sudo=False,
):
    """
    Build a best-effort bash script that clears stale GPU state before a run.

    Steps (all guarded with ``|| true`` so cleanup never fails the job):
      1. pkill -9 -f each known RCCL/MPI process pattern (self-match-safe).
      2. Optionally kill every PID reported by ``rocm-smi --showpids``.
      3. Optionally stop stale GPU containers (docker/podman).

    Args:
      process_patterns: list of process-name substrings (defaults to RCCL/MPI set).
      kill_gpu_pids: also kill PIDs reported by rocm-smi.
      kill_containers: also kill running docker/podman containers.
      use_sudo: prefix kill/pkill/container commands with sudo.

    Returns:
      str: a bash script.
    """
    patterns = process_patterns if process_patterns is not None else DEFAULT_GPU_PROCESS_PATTERNS
    sudo = 'sudo ' if use_sudo else ''
    lines = [
        '#!/usr/bin/env bash',
        '# Auto-generated stale-GPU cleanup (AIMVT-196). Best-effort; never fails.',
        'set +e',
        'echo "[gpu-cleanup] host=$(hostname) start"',
    ]

    for pat in patterns:
        safe = shlex.quote(_self_safe_pattern(pat))
        lines.append(f'{sudo}pkill -9 -f -- {safe} 2>/dev/null || true')

    if kill_gpu_pids:
        lines.append('if command -v rocm-smi >/dev/null 2>&1; then')
        lines.append("  for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do")
        lines.append(f'    {sudo}kill -9 "$p" 2>/dev/null || true')
        lines.append('  done')
        lines.append('fi')

    if kill_containers:
        lines.append('if command -v docker >/dev/null 2>&1; then')
        lines.append(f'  {sudo}docker ps -q 2>/dev/null | xargs -r {sudo}docker kill >/dev/null 2>&1 || true')
        lines.append('fi')
        lines.append('if command -v podman >/dev/null 2>&1; then')
        lines.append(f'  {sudo}podman ps -q 2>/dev/null | xargs -r {sudo}podman kill >/dev/null 2>&1 || true')
        lines.append('fi')

    lines.append('echo "[gpu-cleanup] host=$(hostname) done"')
    lines.append('true')
    return '\n'.join(lines)
