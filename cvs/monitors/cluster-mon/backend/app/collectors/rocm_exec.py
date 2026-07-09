"""
rocm_exec — PATH-fallback helper for amd-smi / rocm-smi commands.

When a command returns a "not found" error on a subset of hosts, this helper
retries those specific hosts with explicit paths, in order:
  1. cd /opt/rocm/bin && ./binary args   (explicit ./ from the ROCm bin dir)
  2. /opt/rocm/bin/binary args            (absolute path, no cd needed)
  3. cd /opt/rocm/bin && PATH=$PWD:$PATH (original PATH trick, for bash -c cmds)

Only the failing hosts are retried; healthy hosts are returned as-is.
"""

from __future__ import annotations

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

_NOT_FOUND_RE = re.compile(
    r"command not found|No such file or directory|not found",
    re.IGNORECASE,
)

ROCM_BIN = "/opt/rocm/bin"


def _is_not_found(output: str) -> bool:
    """True when the output looks like a 'command not found' error."""
    if not output:
        return False
    return bool(_NOT_FOUND_RE.search(output[:300]))


def _build_fallback_commands(command: str) -> list[str]:
    """
    Return fallback commands to try when the primary fails with 'not found'.

    All fallbacks use the same pattern: cd /opt/rocm/bin && ./binary <args>
    This is consistent and explicit — no ambiguity about which binary runs.

    For a direct binary call like 'amd-smi metric --json':
      1. cd /opt/rocm/bin && ./amd-smi metric --json   (explicit ./ from bin dir)

    For a bash -c '...' wrapper (PATH trick — ./ can't be used with bash -c):
      1. cd /opt/rocm/bin && PATH=$PWD:$PATH bash -c '...'
    """
    stripped = command.strip()

    # bash -c '...' — use PATH trick so any binary inside the bash command is found
    if stripped.startswith('bash '):
        return [f"cd {ROCM_BIN} && PATH=$PWD:$PATH {command}"]

    # Direct binary call — cd to bin dir and use ./binary explicitly
    parts = stripped.split(maxsplit=1)
    binary = parts[0]
    args_str = f" {parts[1]}" if len(parts) > 1 else ""

    return [
        f"cd {ROCM_BIN} && ./{binary}{args_str}",
    ]


async def exec_with_rocm_fallback(
    ssh_manager,
    command: str,
    timeout: int = 120,
) -> Dict[str, str]:
    """
    Run *command* on all reachable hosts via ssh_manager.exec_async().

    For any host whose output indicates the binary was not found, retry that
    host with up to three fallback forms (explicit ./, absolute path, PATH
    trick). Returns the merged dict {host: output}.
    """
    output: Dict[str, str] = await ssh_manager.exec_async(command, timeout=timeout)

    failing = {h: out for h, out in output.items() if _is_not_found(out)}
    if not failing:
        return output  # fast path — nothing to retry

    logger.info(f"'{command}' not found on {list(failing.keys())} — trying fallback paths in {ROCM_BIN}")

    fallback_commands = _build_fallback_commands(command)
    merged = dict(output)

    for fb_cmd in fallback_commands:
        # Only retry hosts that are still failing
        still_failing = [h for h in failing if _is_not_found(merged.get(h, ""))]
        if not still_failing:
            break  # all recovered

        logger.debug(f"Fallback attempt: {fb_cmd!r} on {still_failing}")

        try:
            # Run fallback on ALL hosts; then extract results for failing hosts only
            all_out: Dict[str, str] = await ssh_manager.exec_async(fb_cmd, timeout=timeout)
            for host in still_failing:
                if host in all_out:
                    fallback_out = all_out[host]
                    if not _is_not_found(fallback_out):
                        logger.info(f"'{command}' fallback succeeded on {host} via: {fb_cmd!r}")
                        merged[host] = fallback_out
                    # else: still failing, will try next fallback
        except Exception as e:
            logger.warning(f"Fallback command failed with exception: {e}")

    # Log any hosts still not resolved after all fallbacks
    for host in failing:
        if _is_not_found(merged.get(host, "")):
            logger.warning(
                f"'{command}' not found on {host} after all fallbacks "
                f"({ROCM_BIN}/{{./binary, binary, PATH trick}}). "
                f"Check that amd-smi/rocm-smi is installed in {ROCM_BIN}."
            )

    return merged
