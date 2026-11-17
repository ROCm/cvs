"""Stage a read-only dataset onto each node's local storage before training."""

from __future__ import annotations

import copy
import shlex

_MARKER = ".cvs_staged"
_IN_PROGRESS = ".cvs_staging"

# Runs on every host. The source is summarised as file count and total bytes:
# the copy is verified against that summary, and a later run reuses a copy
# whose marker still matches it. A destination without either marker is left
# alone, so a mistyped path cannot delete unrelated data.
#
# The summary walks each second-level directory in parallel (ImageNet's class
# directories): one serial walk of ImageNet on the shared filesystem took 56 s
# per node. Each walk prints a single total line so concurrent writes to the
# pipe cannot interleave mid-line.
_SCRIPT = r"""
set -euo pipefail
src=$1; dst=$2; jobs=$3
count_tree() { find "$@" -type f ! -name __MARKER__ ! -name __IN_PROGRESS__ -printf '%s\n' | awk '{n++; b+=$1} END {printf "%d %d\n", n, b}'; }
export -f count_tree
summarise() {
    { count_tree "$1" -maxdepth 2
      find "$1" -mindepth 2 -maxdepth 2 -type d -print0 | xargs -0 -r -P "$jobs" -n 1 bash -c 'count_tree "$1" -mindepth 1' _
    } | awk '{n+=$1; b+=$2} END {printf "%d %d\n", n, b}'
}
[ -d "$src" ] || { echo "staging source $src is not a directory"; exit 1; }
parent=$(dirname "$dst")
mkdir -p "$parent"
exec 9>"$parent/.$(basename "$dst").lock"
flock 9
read -r count bytes < <(summarise "$src")
signature="$src $count $bytes"
if [ -f "$dst/__MARKER__" ] && [ "$(cat "$dst/__MARKER__")" = "$signature" ]; then
    echo "CVS_STAGE reused $count files, $bytes bytes at $dst"
    exit 0
fi
if [ -d "$dst" ] && [ -n "$(ls -A "$dst")" ] && [ ! -e "$dst/__MARKER__" ] && [ ! -e "$dst/__IN_PROGRESS__" ]; then
    echo "refusing to replace $dst: it was not created by CVS dataset staging"
    exit 1
fi
held=0
[ -d "$dst" ] && held=$(du -sb "$dst" | cut -f1)
avail=$(df -B1 --output=avail "$parent" | tail -1)
if [ $((avail + held)) -lt $((bytes + bytes / 20)) ]; then
    echo "staging needs $bytes bytes plus 5% headroom on $parent; $((avail + held)) available"
    exit 1
fi
rm -rf "$dst"
mkdir -p "$dst"
: > "$dst/__IN_PROGRESS__"
start=$(date +%s)
cd "$src"
find . -mindepth 1 -type d -printf '%P\0' | (cd "$dst" && xargs -0 -r mkdir -p)
find . -type f -printf '%P\0' | xargs -0 -r -P "$jobs" -n 1000 cp --parents -t "$dst"
read -r got_count got_bytes < <(summarise "$dst")
if [ "$got_count $got_bytes" != "$count $bytes" ]; then
    echo "staged copy does not match the source: $got_count files, $got_bytes bytes; expected $count, $bytes"
    exit 1
fi
printf '%s' "$signature" > "$dst/__MARKER__"
rm -f "$dst/__IN_PROGRESS__"
echo "CVS_STAGE copied $count files, $bytes bytes to $dst in $(($(date +%s) - start))s"
""".replace("__MARKER__", _MARKER).replace("__IN_PROGRESS__", _IN_PROGRESS)


_REMOVE_SCRIPT = r"""
set -euo pipefail
dst=$1
[ -d "$dst" ] || exit 0
exec 9>"$(dirname "$dst")/.$(basename "$dst").lock"
flock 9
if [ -e "$dst/__MARKER__" ] || [ -e "$dst/__IN_PROGRESS__" ]; then
    rm -rf "$dst"
    echo "CVS_STAGE removed $dst"
else
    echo "CVS_STAGE kept $dst: it was not created by CVS dataset staging"
fi
""".replace("__MARKER__", _MARKER).replace("__IN_PROGRESS__", _IN_PROGRESS)


def remove_command(staging) -> str:
    """Shell command that deletes a staged copy, and only a staged copy."""
    return "bash -c " + shlex.quote(_REMOVE_SCRIPT) + " cvs-unstage " + shlex.quote(staging.destination.rstrip("/"))


def stage_command(staging) -> str:
    """Shell command that stages ``staging.source`` into ``staging.destination``."""
    args = (staging.source.rstrip("/"), staging.destination.rstrip("/"), str(staging.parallel_copies))
    return "bash -c " + shlex.quote(_SCRIPT) + " cvs-stage " + " ".join(shlex.quote(arg) for arg in args)


def staged_container(container: dict, staging) -> dict:
    """Container spec whose ``staging.source`` volume is mounted from the staged copy."""
    if not staging.enabled:
        return container
    staged = copy.deepcopy(container)
    args = staged.get("runtime", {}).get("args", {})
    source = staging.source.rstrip("/")
    destination = staging.destination.rstrip("/")
    volumes = []
    for volume in args.get("volumes") or []:
        host, sep, rest = str(volume).partition(":")
        volumes.append(f"{destination}{sep}{rest}" if host.rstrip("/") == source else volume)
    args["volumes"] = volumes
    return staged


def stage_dataset(host_handle, hosts, staging) -> dict:
    """Stage on every host and return each host's summary line."""
    return _run_on_hosts(host_handle, hosts, stage_command(staging), staging.timeout_s, "dataset staging")


def remove_staged_dataset(host_handle, hosts, staging) -> dict:
    """Delete the staged copy on every host and return each host's summary line."""
    return _run_on_hosts(host_handle, hosts, remove_command(staging), staging.timeout_s, "staged dataset removal")


def _run_on_hosts(host_handle, hosts, command, timeout_s, action) -> dict:
    result = host_handle.exec(
        command,
        timeout=timeout_s,
        detailed=True,
        print_console=False,
    )
    result = result or {}
    missing = sorted(set(hosts) - set(result))
    failures = {}
    summaries = {}
    for host, host_result in result.items():
        if isinstance(host_result, dict):
            output = host_result.get("output") or host_result.get("stdout") or ""
            exit_code = host_result.get("exit_code", 1)
        else:
            output, exit_code = str(host_result or ""), 1
        if exit_code != 0:
            failures[host] = output[-1500:]
            continue
        summaries[host] = next(
            (line for line in reversed(output.splitlines()) if line.startswith("CVS_STAGE ")),
            output.strip()[-300:],
        )
    if missing or failures:
        raise RuntimeError(f"{action} failed: missing={missing}, failures={failures}")
    return summaries
