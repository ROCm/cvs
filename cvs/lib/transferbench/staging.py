"""Stage a generated TransferBench input file on cluster nodes.

TransferBench reads its transfer list from a file that every rank must be able
to open, so the file is created on each node rather than shipped from the head.
"""

import posixpath
import re
import shlex

# The prefix is interpolated into the mktemp template unquoted, so it is held to
# characters that cannot alter the command.
_PREFIX_RE = re.compile(r'^[A-Za-z0-9_.-]+$')

# exec_cmd_list discards exit codes, so removal reports its own status.
REMOVE_SENTINEL = '__CVS_TB_RM__'

# The staged path is echoed behind a marker because a login shell may print a
# banner or MOTD ahead of the command's own output.
PATH_MARKER = '__CVS_TB_PATH__'


def _validate_prefix(prefix):
    if not isinstance(prefix, str) or not _PREFIX_RE.fullmatch(prefix):
        raise ValueError(f'Staged file prefix must match {_PREFIX_RE.pattern}, got {prefix!r}')
    return prefix


def _validate_path(path, prefix):
    """Reject anything that is not the private file we just asked mktemp for."""
    if not path or '\n' in path:
        return False
    if not posixpath.isabs(path):
        return False
    return posixpath.basename(path).startswith(prefix)


def _extract_path(output):
    """Pull the marked path out of a node's output, ignoring any banner."""
    if not isinstance(output, str):
        return ''
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith(PATH_MARKER + '|'):
            return line[len(PATH_MARKER) + 1 :]
    return ''


def stage_file(orch, content, prefix, timeout=30):
    """Write ``content`` to a private 0600 temp file on every reachable node.

    Returns ``(paths_by_host, failures_by_host)``. A node that fails to stage is
    reported rather than raised so one bad node does not stop the run; the
    caller decides what that means for its results.
    """
    _validate_prefix(prefix)
    if not isinstance(content, str):
        raise ValueError('Staged file content must be a string')

    # umask 077 plus a random name means another user cannot pre-place or
    # redirect the file.
    cmd = (
        '(umask 077 && f=$(mktemp -p "${TMPDIR:-/tmp}" ' + prefix + 'XXXXXXXXXX)'
        f' && printf %s {shlex.quote(content)} > "$f"'
        f' && printf \'{PATH_MARKER}|%s\\n\' "$f")'
    )
    result = orch.all.exec(cmd, timeout=timeout, detailed=True, print_console=False)

    paths = {}
    failures = {}
    for host, res in (result or {}).items():
        if not isinstance(res, dict) or res.get('exit_code') != 0:
            failures[host] = f'could not create staged file: {res!r}'
            continue
        path = _extract_path(res.get('output'))
        if not _validate_path(path, prefix):
            failures[host] = f'staged file path failed validation: {path!r}'
            continue
        paths[host] = path
    return paths, failures


def remove_files(orch, paths_by_host, timeout=30):
    """Delete staged files. Returns the hosts whose removal could not be confirmed."""
    if not paths_by_host:
        return []
    hosts = list(orch.all.reachable_hosts)
    if not hosts:
        # Nothing could be attempted, so every staged file is still out there;
        # saying "nothing outstanding" would lose the paths a retry needs.
        return sorted(paths_by_host)

    # exec_cmd_list pairs commands to reachable_hosts positionally; hosts with
    # nothing staged still need a slot. It also discards exit codes, so each
    # removal echoes its own status.
    cmd_list = []
    for host in hosts:
        path = paths_by_host.get(host)
        cmd_list.append(f'rm -f -- {shlex.quote(path)}; echo "{REMOVE_SENTINEL}=$?"' if path else 'true')

    out_dict = orch.all.exec_cmd_list(cmd_list, timeout=timeout, print_console=False)
    remaining = []
    for host in paths_by_host:
        output = (out_dict or {}).get(host)
        if not isinstance(output, str) or f'{REMOVE_SENTINEL}=0' not in output:
            remaining.append(host)
    return remaining
