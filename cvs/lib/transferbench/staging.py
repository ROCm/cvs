"""Stage a generated TransferBench input file on cluster nodes.

TransferBench reads its transfer list from a file that every rank must be able
to open, so the file is created on each node rather than shipped from the head.
"""

import posixpath
import re
import shlex

# The prefix is interpolated into the mktemp template unquoted, so it is held to
# characters that cannot alter the command. A leading hyphen is excluded too:
# that is not shell syntax, but mktemp would read it as one of its own options.
_PREFIX_RE = re.compile(r'^[A-Za-z0-9_.][A-Za-z0-9_.-]*$')

# exec_cmd_list discards exit codes, so removal and appends report their own.
REMOVE_SENTINEL = '__CVS_TB_RM__'
APPEND_SENTINEL = '__CVS_TB_APPEND__'

# Linux caps one argv entry at MAX_ARG_STRLEN (128 KiB) however large ARG_MAX
# is. Quoting can expand content, so the raw budget leaves room for the worst
# case and for the rest of the command around it.
MAX_CONTENT_BYTES_PER_COMMAND = 16 * 1024

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


def _split_for_argv(content, budget=MAX_CONTENT_BYTES_PER_COMMAND):
    """Split content into pieces small enough to quote into one command.

    Linux caps a single argv entry at 128 KiB regardless of how large ARG_MAX
    is, and the whole command is one ``bash -c`` argument. A cluster-sized
    transfer file passes that cap on its own, so the content is written in
    bounded appends rather than embedded whole.
    """
    if not content:
        return ['']
    return [content[start : start + budget] for start in range(0, len(content), budget)]


def stage_file(orch, content, prefix, timeout=30):
    """Write ``content`` to a private 0600 temp file on every reachable node.

    Returns ``(paths_by_host, failures_by_host)``. A node that fails to stage is
    reported rather than raised so one bad node does not stop the run; the
    caller decides what that means for its results.
    """
    _validate_prefix(prefix)
    if not isinstance(content, str):
        raise ValueError('Staged file content must be a string')

    chunks = _split_for_argv(content)

    # umask 077 plus a random name means another user cannot pre-place or
    # redirect the file. The trap removes it if anything after mktemp fails
    # (e.g. a quota) or the session is dropped (bash also runs an EXIT trap
    # on SIGHUP/SIGTERM/SIGINT; only an unconditional kill bypasses it), and
    # is disarmed only once the file is written and its path reported.
    cmd = (
        '(umask 077 && f=$(mktemp -p "${TMPDIR:-/tmp}" ' + prefix + 'XXXXXXXXXX)'
        ' && trap \'rm -f -- "$f"\' EXIT'
        f' && printf %s {shlex.quote(chunks[0])} > "$f"'
        f' && printf \'{PATH_MARKER}|%s\\n\' "$f"'
        ' && trap - EXIT)'
    )
    result = orch.exec(cmd, timeout=timeout, detailed=True, print_console=False)

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

    for index, chunk in enumerate(chunks[1:], start=2):
        if not paths:
            break
        paths, append_failures = _append_chunk(orch, paths, chunk, index, len(chunks), timeout)
        failures.update(append_failures)
    return paths, failures


def _append_chunk(orch, paths_by_host, chunk, index, total, timeout):
    """Append one chunk to each host's staged file.

    A host whose append fails keeps a half-written file that the first
    command's trap no longer guards, so it is removed here rather than left
    for a caller that has already been told the host failed.
    """
    hosts = list(orch.all.reachable_hosts)
    cmd_list = [
        f'printf %s {shlex.quote(chunk)} >> {shlex.quote(paths_by_host[host])}; echo "{APPEND_SENTINEL}=$?"'
        if host in paths_by_host
        else 'true'
        for host in hosts
    ]
    out_dict = orch.exec_cmd_list(cmd_list, timeout=timeout, print_console=False) or {}

    kept = {}
    failures = {}
    for host, path in paths_by_host.items():
        output = out_dict.get(host)
        if isinstance(output, str) and f'{APPEND_SENTINEL}=0' in output:
            kept[host] = path
            continue
        failures[host] = f'could not write staged file (chunk {index} of {total})'
    if failures:
        leaked = remove_files(orch, {host: paths_by_host[host] for host in failures}, timeout=timeout)
        # Saying only "the write failed" would hide a file still sitting on the
        # node, which is the part a caller may have to clean up by hand.
        for host in leaked:
            failures[host] += f'; the partial file {paths_by_host[host]} could not be removed'
    return kept, failures


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

    out_dict = orch.exec_cmd_list(cmd_list, timeout=timeout, print_console=False)
    remaining = []
    for host in paths_by_host:
        output = (out_dict or {}).get(host)
        if not isinstance(output, str) or f'{REMOVE_SENTINEL}=0' not in output:
            remaining.append(host)
    return remaining
