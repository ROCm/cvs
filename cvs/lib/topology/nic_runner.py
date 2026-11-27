"""Drive one TransferBench run for the cross-node NIC bandwidth matrix."""

import re

from cvs.lib.env_lib import build_env_prefix
from cvs.lib.topology import nic_matrix
from cvs.lib.transferbench import multirank, staging

# TransferBench takes the per-transfer byte count as its second argument:
# digits with an optional K/M/G suffix. There is no environment variable for
# it, so a value that is not in this form has to be rejected here.
_MESSAGE_SIZE_RE = re.compile(r'^[0-9]+[KMG]?$', re.IGNORECASE)

STAGED_FILE_PREFIX = 'cvs_topo_nic_'

DEFAULT_MESSAGE_SIZE = '64M'
DEFAULT_QUEUE_PAIRS = 4
DEFAULT_ITERATIONS = 10
DEFAULT_WARMUP_ITERATIONS = 3
DEFAULT_TIMEOUT = 1800


def validate_message_size(value):
    """Validate the per-transfer byte count handed to TransferBench."""
    text = str(value).strip()
    if not _MESSAGE_SIZE_RE.fullmatch(text):
        raise ValueError(f'message_size must be digits with an optional K/M/G suffix, got {value!r}')
    return text


def build_settings(config_dict, cluster_dict=None):
    """Read the suite config into the values the run needs, rejecting bad input."""
    config = config_dict if isinstance(config_dict, dict) else {}
    advanced = config.get('advanced') if isinstance(config.get('advanced'), dict) else {}
    cluster_env = (cluster_dict or {}).get('env_vars')

    return {
        'coverage': config.get('coverage', nic_matrix.COVERAGE_AUTO),
        'flag_below_pct_of_median': config.get('flag_below_pct_of_median', nic_matrix.DEFAULT_FLAG_BELOW_PCT_OF_MEDIAN),
        'traffic_class': nic_matrix.resolve_traffic_class(config.get('traffic_class', 'auto'), cluster_env),
        'tb_binary': multirank.validate_binary_name('tb_binary', config.get('tb_binary', 'TransferBench')),
        'use_sudo': bool(config.get('use_sudo', False)),
        'transferbench_env': multirank.validate_extra_env(config.get('transferbench_env')),
        'nic_filter': config.get('nic_filter'),
        'message_size': validate_message_size(advanced.get('message_size', DEFAULT_MESSAGE_SIZE)),
        'queue_pairs': advanced.get('queue_pairs', DEFAULT_QUEUE_PAIRS),
        'iterations': advanced.get('iterations', DEFAULT_ITERATIONS),
        'warmup_iterations': advanced.get('warmup_iterations', DEFAULT_WARMUP_ITERATIONS),
        'timeout': advanced.get('timeout', DEFAULT_TIMEOUT),
        'master_port': multirank.resolve_socket_master_port(advanced.get('master_port')),
    }


def build_run_env(settings):
    """Environment shared by every rank.

    The transfer file and the message size are argv, not environment:
    TransferBench has no environment variable for the byte count, and setting
    one leaves every transfer at its built-in default size.
    """
    env = {
        'NUM_ITERATIONS': str(settings['iterations']),
        'NUM_WARMUPS': str(settings['warmup_iterations']),
        'NIC_TRAFFIC_CLASS': str(settings['traffic_class']),
    }
    if settings.get('nic_filter'):
        env['TB_NIC_FILTER'] = str(settings['nic_filter'])
    return env


def run_matrix(orch, inventory, settings, switch_by_node=None):
    """Measure every cross-node NIC pair and return the matrix.

    Returns ``(matrix, pairs, problems)``. ``problems`` lists nodes that could
    not take part, so one bad node reduces coverage instead of ending the run.
    """
    # Only a node that is both reachable and has addressable NICs can be a
    # rank. The ranks are numbered once, here, and the same map is used to
    # write the transfer file and to launch: numbering them separately lets a
    # node with no backend NICs shift every later rank, so the file would name
    # ranks belonging to other hosts.
    with_nics = {node for node, nics in inventory.items() if nics}
    participants = [host for host in orch.all.reachable_hosts if host in with_nics]
    master = multirank.resolve_master_node(participants)
    if not master:
        return {}, [], []
    node_by_rank = multirank.rank_order(participants, master)
    rank_by_node = {node: rank for rank, node in enumerate(node_by_rank)}

    pairs = nic_matrix.build_pairs(
        {node: inventory[node] for node in node_by_rank},
        coverage=settings['coverage'],
        switch_by_node=switch_by_node,
        rank_by_node=rank_by_node,
    )
    if not pairs:
        return {}, [], []

    transfer_file = nic_matrix.render_transfer_file(pairs, settings['queue_pairs'])
    paths, failures = staging.stage_file(orch, transfer_file, STAGED_FILE_PREFIX, timeout=60)
    problems = [f'{node}: {reason}' for node, reason in sorted(failures.items())]
    if not paths:
        return {}, pairs, problems

    base_env = build_run_env(settings)

    def command_for_rank(rank, num_ranks, master_addr):
        node = node_by_rank[rank]
        if node not in paths:
            raise ValueError(f'no staged transfer file on {node}')
        env = dict(base_env)
        env.update(multirank.rank_env(rank, num_ranks, master_addr, settings['master_port']))
        return multirank.build_command(
            settings['tb_binary'],
            [paths[node], settings['message_size']],
            env,
            use_sudo=settings['use_sudo'],
            runtime_env_prefix=build_env_prefix(settings.get('transferbench_env')),
            sudo_argv0='cvs-topology-nic',
        )

    try:
        per_node = multirank.dispatch_multi_rank(
            orch, command_for_rank, timeout=settings['timeout'], hosts=node_by_rank
        )
    finally:
        staging.remove_files(orch, paths)

    # Rank 0 gathers and prints the results for every rank.
    output = ''
    for record in per_node.values():
        if record.get('rank') == 0:
            output = record.get('output') or ''
            break

    # exec_cmd_list discards exit codes, so dispatch_multi_rank's own contract
    # requires recovering completion from the sentinel; non-empty output is
    # not proof the run finished (see query_nic_table, which checks the same
    # way for the same reason).
    exit_code = multirank.extract_exit_code(output)
    if exit_code is None:
        problems = problems + ['TransferBench did not complete (no exit status recovered from rank 0)']
        return {}, pairs, problems
    if exit_code != 0:
        problems = problems + [f'TransferBench exited {exit_code} on rank 0']
        return {}, pairs, problems

    return nic_matrix.parse_results(output, pairs), pairs, problems
