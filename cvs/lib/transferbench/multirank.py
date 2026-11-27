"""Multi-rank TransferBench invocation shared by preflight and topology.

TransferBench's socket-comm mode needs one process per node, each carrying its
own ``TB_RANK`` and a common ``TB_MASTER_ADDR``. The rank bookkeeping, shell
quoting, sudo wrapping and exit-code recovery are identical for every caller,
so they live here rather than in any one suite.
"""

import os
import re
import shlex

DEFAULT_SOCKET_MASTER_PORT = 31337

# Callers that need a port distinct from the preflight default read it from the
# environment; preflight keeps passing its configured value so its behaviour is
# unchanged by this module.
SOCKET_MASTER_PORT_ENV = 'CVS_TOPO_TB_PORT'

# Appended to every per-node command so the exit code survives a parallel SSH
# layer that discards it.
EXIT_SENTINEL = '__TB_SMOKE_EXIT__'
EXIT_SENTINEL_RE = re.compile(re.escape(EXIT_SENTINEL) + r'\s*=\s*(?P<code>-?\d+)\s*$', re.MULTILINE)

_ENV_VAR_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# The sentinel is echoed inside double quotes and later matched as a regex, so
# it is held to characters that are inert in both.
_SENTINEL_RE = re.compile(r'^[A-Za-z0-9_]+$')


def validate_binary_name(config_key, value):
    """Validate a configured executable name or path."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{config_key} must be a non-empty string')
    return value.strip()


def validate_extra_env(extra_env):
    """Validate runtime environment supplied for the TransferBench process.

    Restricting keys to shell environment variable names prevents a config key
    from changing command syntax before quoting is applied.
    """
    if extra_env is None:
        return {}
    if not isinstance(extra_env, dict):
        raise ValueError('extra_env must be a mapping of environment variable names to values')

    normalized = {}
    for key, value in extra_env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.fullmatch(key):
            raise ValueError(f'Invalid environment variable name in extra_env: {key!r}')
        if value is None:
            raise ValueError(f'extra_env[{key!r}] must not be null')
        normalized[key] = str(value)
    return normalized


def resolve_socket_master_port(configured=None, environ=None):
    """Resolve the socket-comm master port.

    The environment wins so an operator can move the port off a busy one
    without editing config.
    """
    env = os.environ if environ is None else environ
    raw = env.get(SOCKET_MASTER_PORT_ENV)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = configured
    if raw is None:
        return DEFAULT_SOCKET_MASTER_PORT
    try:
        port = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(f'TransferBench master port must be an integer, got {raw!r}')
    if not 1 <= port <= 65535:
        raise ValueError(f'TransferBench master port must be between 1 and 65535, got {port}')
    return port


def sudo_executable_argument(executable):
    """Render an executable argument that remains valid after ``sudo``.

    ``sudo`` may replace PATH with ``secure_path``. A configured path is
    already independent of PATH; a bare executable name is resolved by the
    outer SSH shell instead, while the cluster-file environment is still in
    effect.
    """
    if '/' in executable:
        return shlex.quote(executable)
    return f'"$(command -v {shlex.quote(executable)})"'


def rank_env(rank, num_ranks, master_addr, master_port=DEFAULT_SOCKET_MASTER_PORT):
    """Socket-comm environment for one rank, empty for a single-rank run."""
    if num_ranks <= 1:
        return {}
    return {
        'TB_NUM_RANKS': str(num_ranks),
        'TB_RANK': str(rank),
        'TB_MASTER_ADDR': str(master_addr),
        'TB_MASTER_PORT': str(master_port),
    }


def render_env_assignments(env):
    """Render ``KEY=VALUE`` assignments, quoting every value.

    A name cannot be quoted, because a quoted word is no longer an assignment,
    so an invalid one is rejected here. This repeats the check callers make on
    their own config because this is the point where the text becomes shell
    syntax, and not every caller validates.
    """
    rendered = []
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.fullmatch(key):
            raise ValueError(f'Invalid environment variable name: {key!r}')
        rendered.append(f'{key}={shlex.quote(str(value))}')
    return rendered


def build_command(
    binary,
    args,
    env,
    use_sudo=False,
    runtime_env_prefix='',
    sentinel=EXIT_SENTINEL,
    sudo_argv0='transferbench',
):
    """Render a complete shell command for one TransferBench rank.

    Environment assignments are placed **inside** the inner shell so the
    privileged child sees them; ``sudo`` otherwise sanitizes the calling
    shell's environment. The sentinel line lets the orchestrator recover the
    binary's exit code from stdout.

    Every interpolated value is quoted or validated, except
    ``runtime_env_prefix``, which is shell code by contract: pass only text
    built by ``build_env_prefix``, never a raw configured value.
    """
    if not isinstance(sentinel, str) or not _SENTINEL_RE.fullmatch(sentinel):
        raise ValueError(f'Invalid exit sentinel: {sentinel!r}')
    env_inline = ' '.join(render_env_assignments(env)).strip()

    # A command name must be resolved before sudo applies secure_path, so the
    # resolved path is passed as $1 to the privileged shell.
    invocation_binary = '"$1"' if use_sudo else shlex.quote(binary)
    binary_invocation = ' '.join([invocation_binary] + [shlex.quote(str(arg)) for arg in args]).strip()

    inner = f'{env_inline} {binary_invocation}' if env_inline else binary_invocation
    if runtime_env_prefix:
        inner = f'{runtime_env_prefix}; {inner}'
    inner_with_sentinel = f'{inner}; echo "{sentinel}=$?"'
    if use_sudo:
        # argv0 is quoted like every other interpolation: an unquoted value
        # containing a space would split, pushing the binary out of $1.
        argv0 = shlex.quote(str(sudo_argv0))
        return f'sudo bash -c {shlex.quote(inner_with_sentinel)} {argv0} {sudo_executable_argument(binary)}'
    return f'bash -c {shlex.quote(inner_with_sentinel)}'


def extract_exit_code(output, sentinel_re=EXIT_SENTINEL_RE):
    """Return the exit code echoed by the sentinel, or None when absent."""
    if not isinstance(output, str):
        return None
    match = None
    for match in sentinel_re.finditer(output):
        pass
    if match is None:
        return None
    return int(match.group('code'))


def resolve_master_node(reachable_hosts, preferred=None):
    """Pick the rank-0 node, preferring a caller-supplied one when reachable."""
    hosts = list(reachable_hosts)
    if preferred and preferred in hosts:
        return preferred
    if hosts:
        return sorted(hosts)[0]
    return None


def rank_order(reachable_hosts, master):
    """Order nodes by rank: master is rank 0, the rest follow sorted."""
    return [master] + sorted(host for host in reachable_hosts if host != master)


def dispatch_multi_rank(orch, command_for_rank, timeout=None, preferred_master=None, hosts=None, executor=None):
    """Start one rank per participating node in a single parallel call.

    ``command_for_rank(rank, num_ranks, master_addr)`` renders one rank's shell
    command. All ranks must start together for the socket-comm bootstrap to
    complete, hence a single ``exec_cmd_list``.

    ``hosts`` limits which reachable nodes are given a rank; the rest run a
    no-op and are left out of the result. A caller whose payload addresses
    ranks by number (a TransferBench transfer file does) must pass the same
    node set it numbered against, or the ranks it names resolve to other
    hosts. Omitting it gives every reachable node a rank.

    ``executor`` is the handle the rank commands run through, and defaults to
    ``orch.all`` -- the bare-host handle, which is not container-aware. A
    caller whose discovery and staging go through the orchestrator's public
    interface must pass ``orch`` here, or the ranks run on the host while the
    file they read was staged inside the container. Preflight keeps the host
    handle because the rest of its probes use it too, and splitting the two
    would leave it reading a container's file from outside.

    This returns raw per-rank output; it does not interpret completion or
    success. ``exec_cmd_list`` discards exit codes, so a caller must recover
    completion from a sentinel embedded in the command ``command_for_rank``
    builds (``build_command`` already appends one) and must not treat
    non-empty output as success on its own.
    """
    reachable = list(orch.all.reachable_hosts)
    if not reachable:
        return {}
    participants = [host for host in reachable if host in set(hosts)] if hosts is not None else list(reachable)
    if not participants:
        return {}
    master = resolve_master_node(participants, preferred_master)
    if not master:
        return {}

    ordered = rank_order(participants, master)
    num_ranks = len(ordered)
    per_node = {}
    for rank, node in enumerate(ordered):
        per_node[node] = {
            'command': command_for_rank(rank, num_ranks, master),
            'output': '',
            'rank': rank,
        }

    # exec_cmd_list aligns commands to reachable_hosts positionally, so the
    # list is rebuilt in that order rather than in rank order. A reachable
    # host with no rank still needs its slot, and runs nothing.
    no_rank = [host for host in reachable if host not in per_node]
    cmd_by_reachable = [per_node[host]['command'] if host in per_node else 'true' for host in reachable]
    runner = executor if executor is not None else orch.all
    out_dict = runner.exec_cmd_list(cmd_by_reachable, timeout=timeout, print_console=False)
    for host, output in out_dict.items():
        # A host that ran the no-op has no result to report; one that is absent
        # from reachable_hosts is an unreachable node the layer appended.
        if host in no_rank:
            continue
        if host not in per_node:
            per_node[host] = {'command': '', 'output': output}
        per_node[host]['output'] = output
    return per_node
