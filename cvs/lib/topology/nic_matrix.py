"""Cross-node NIC bandwidth matrix.

Measures RDMA bandwidth between every cross-node NIC pair with TransferBench
and flags pairs that fall below a share of the cluster median, so a bad cable,
NIC or uplink shows up as a shape in the matrix rather than a single number.

Same-node pairs are never generated. TransferBench's ``nicp2p`` preset cannot
skip NIC self-loopback, and a loopback run has been implicated in a host-fatal
machine check, so this module builds its own transfer file instead.
"""

import re
import statistics

COVERAGE_FULL = 'full'
COVERAGE_QUICK = 'quick'
COVERAGE_AUTO = 'auto'
COVERAGE_CHOICES = (COVERAGE_FULL, COVERAGE_QUICK, COVERAGE_AUTO)

# Above this many nodes, a full matrix grows past what one run should attempt.
DEFAULT_AUTO_FULL_MAX_NODES = 64
DEFAULT_QUICK_REFERENCE_NODES = 2
DEFAULT_FLAG_BELOW_PCT_OF_MEDIAN = 90

TRAFFIC_CLASS_AUTO = 'auto'
TRAFFIC_CLASS_ENV_VAR = 'NCCL_IB_TC'
DEFAULT_TRAFFIC_CLASS = 0

_RESULT_RE = re.compile(
    r'Executor:\s*Rank\s+(?P<rank>\d+)\s+NIC\s+(?P<nic>\d+)\s*\|\s*(?P<bw>[0-9]+(?:\.[0-9]+)?)\s*GB/s',
    re.IGNORECASE,
)


def resolve_coverage(coverage, node_count, auto_full_max_nodes=DEFAULT_AUTO_FULL_MAX_NODES):
    """Turn the configured coverage into ``full`` or ``quick``."""
    if coverage not in COVERAGE_CHOICES:
        raise ValueError(f'coverage must be one of {COVERAGE_CHOICES}, got {coverage!r}')
    if coverage != COVERAGE_AUTO:
        return coverage
    return COVERAGE_FULL if node_count <= auto_full_max_nodes else COVERAGE_QUICK


def resolve_traffic_class(configured, cluster_env_vars=None):
    """Resolve the RDMA traffic class to pass as ``NIC_TRAFFIC_CLASS``.

    ``auto`` follows the cluster file's NCCL traffic class so the measurement
    runs on the same class as the workload it is meant to predict.
    """
    value = configured
    if isinstance(value, str) and value.strip().lower() == TRAFFIC_CLASS_AUTO:
        env_vars = cluster_env_vars if isinstance(cluster_env_vars, dict) else {}
        value = env_vars.get(TRAFFIC_CLASS_ENV_VAR)
        if value is None or (isinstance(value, str) and not value.strip()):
            return DEFAULT_TRAFFIC_CLASS

    # bool is an int subclass but is never a meaningful traffic class.
    if isinstance(value, bool):
        raise ValueError(f'traffic_class must be {TRAFFIC_CLASS_AUTO!r} or an integer 0-255, got {configured!r}')
    try:
        traffic_class = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f'traffic_class must be {TRAFFIC_CLASS_AUTO!r} or an integer 0-255, got {configured!r}')
    if not 0 <= traffic_class <= 255:
        raise ValueError(f'traffic_class must be between 0 and 255, got {traffic_class}')
    return traffic_class


def select_reference_nodes(nodes, switch_by_node=None, count=DEFAULT_QUICK_REFERENCE_NODES):
    """Pick the source nodes for quick coverage.

    Nodes on different switches are preferred so a quick run still crosses the
    fabric. ``switch_by_node`` is optional; without it the choice is simply
    deterministic.
    """
    ordered = sorted(nodes)
    if count <= 0 or not ordered:
        return []
    switches = switch_by_node if isinstance(switch_by_node, dict) else {}

    selected = []
    seen_switches = set()
    for node in ordered:
        switch = switches.get(node)
        if switch is not None and switch in seen_switches:
            continue
        selected.append(node)
        if switch is not None:
            seen_switches.add(switch)
        if len(selected) == count:
            return selected

    for node in ordered:
        if len(selected) == count:
            break
        if node not in selected:
            selected.append(node)
    return selected


def build_pairs(
    inventory, coverage=COVERAGE_FULL, switch_by_node=None, auto_full_max_nodes=DEFAULT_AUTO_FULL_MAX_NODES
):
    """Build the ordered list of cross-node NIC pairs to measure.

    ``inventory`` maps node to its selected backend NICs, each carrying a
    ``nic`` name and a ``gpu`` index. Rank 0 is the first node in sorted order,
    matching the rank assignment used to launch the ranks.
    """
    nodes = sorted(node for node, nics in inventory.items() if nics)
    rank_by_node = {node: rank for rank, node in enumerate(nodes)}
    effective = resolve_coverage(coverage, len(nodes), auto_full_max_nodes)

    sources = nodes
    if effective == COVERAGE_QUICK:
        sources = select_reference_nodes(nodes, switch_by_node)

    pairs = []
    for src_node in sources:
        for src_index, src in enumerate(inventory[src_node]):
            for dst_node in nodes:
                # Cross-node only: same-node and NIC self-loopback are excluded
                # by construction, not filtered afterwards.
                if dst_node == src_node:
                    continue
                for dst_index, dst in enumerate(inventory[dst_node]):
                    pairs.append(
                        {
                            'src_node': src_node,
                            'src_rank': rank_by_node[src_node],
                            'src_nic': src['nic'],
                            'src_nic_index': src_index,
                            'src_gpu': src['gpu'],
                            'dst_node': dst_node,
                            'dst_rank': rank_by_node[dst_node],
                            'dst_nic': dst['nic'],
                            'dst_nic_index': dst_index,
                            'dst_gpu': dst['gpu'],
                        }
                    )
    return pairs


def render_transfer_file(pairs, queue_pairs):
    """Render the TransferBench transfer file for the given pairs."""
    if not isinstance(queue_pairs, int) or isinstance(queue_pairs, bool) or queue_pairs < 1:
        raise ValueError(f'queue_pairs must be a positive integer, got {queue_pairs!r}')

    lines = []
    for pair in pairs:
        if pair['src_rank'] == pair['dst_rank']:
            raise ValueError(f'refusing to emit a same-rank transfer for {pair["src_node"]}')
        lines.append(
            f'1 {queue_pairs} '
            f'(R{pair["src_rank"]}G{pair["src_gpu"]}->'
            f'R{pair["src_rank"]}I{pair["src_nic_index"]}.{pair["dst_nic_index"]}->'
            f'R{pair["dst_rank"]}G{pair["dst_gpu"]})'
        )
    return '\n'.join(lines) + '\n' if lines else ''


def parse_results(output, pairs):
    """Map TransferBench result lines onto the pairs that produced them.

    Results arrive in transfer-file order, so each line is matched against the
    pair at the same position and its rank/NIC are verified. A mismatch or a
    short read is an error: a partial matrix would silently understate coverage.
    """
    found = _RESULT_RE.findall(output or '')
    if len(found) != len(pairs):
        raise ValueError(f'TransferBench reported {len(found)} results for {len(pairs)} transfers')

    matrix = {}
    for (rank, nic, bandwidth), pair in zip(found, pairs):
        if int(rank) != pair['src_rank'] or int(nic) != pair['src_nic_index']:
            raise ValueError(
                f'result out of order: expected rank {pair["src_rank"]} NIC {pair["src_nic_index"]}, '
                f'got rank {rank} NIC {nic}'
            )
        source = (pair['src_node'], pair['src_nic'])
        matrix.setdefault(source, {})[(pair['dst_node'], pair['dst_nic'])] = float(bandwidth)
    return matrix


def _all_cells(matrix):
    for source, row in matrix.items():
        for destination, value in row.items():
            yield source, destination, value


def flag_outliers(matrix, flag_below_pct_of_median=DEFAULT_FLAG_BELOW_PCT_OF_MEDIAN):
    """Classify low cells into node, NIC and single-pair anomalies.

    A cell is reported once, under the most general shape that explains it: a
    node pair that is low on every NIC points at an uplink or placement issue,
    a NIC low in every direction points at that NIC or its cable, and anything
    left is an individual link.
    """
    cells = list(_all_cells(matrix))
    if not cells:
        return {
            'median': None,
            'threshold': None,
            'threshold_pct': flag_below_pct_of_median,
            'nodes': [],
            'nics': [],
            'pairs': [],
        }

    values = [value for _, _, value in cells]
    median = statistics.median(values)
    threshold = median * flag_below_pct_of_median / 100.0
    low = {(source, destination) for source, destination, value in cells if value < threshold}

    by_node_pair = {}
    by_source_nic = {}
    by_dest_nic = {}
    for source, destination, _ in cells:
        by_node_pair.setdefault((source[0], destination[0]), []).append((source, destination))
        by_source_nic.setdefault(source, []).append((source, destination))
        by_dest_nic.setdefault(destination, []).append((source, destination))

    flagged_nodes = sorted(key for key, members in by_node_pair.items() if all(m in low for m in members))
    explained = {m for key in flagged_nodes for m in by_node_pair[key]}

    flagged_nics = []
    for nic, members in list(by_source_nic.items()) + list(by_dest_nic.items()):
        # A NIC whose every cell is already covered by a flagged node pair adds
        # no information, so the broader shape keeps the report.
        if all(m in low for m in members) and any(m not in explained for m in members) and nic not in flagged_nics:
            flagged_nics.append(nic)
    flagged_nics.sort()
    for nic in flagged_nics:
        explained.update(by_source_nic.get(nic, []))
        explained.update(by_dest_nic.get(nic, []))

    flagged_pairs = sorted(cell for cell in low if cell not in explained)

    return {
        'median': median,
        'threshold': threshold,
        # Heatmap cells are percentages of the median, so the same limit is
        # carried in both units rather than re-derived by the renderer.
        'threshold_pct': flag_below_pct_of_median,
        'nodes': flagged_nodes,
        'nics': flagged_nics,
        'pairs': flagged_pairs,
    }


def describe_flags(flags):
    """Render flags as human-readable failure reasons, most general first."""
    messages = []
    for src_node, dst_node in flags['nodes']:
        messages.append(f'node pair {src_node} -> {dst_node} is below threshold on every NIC (uplink or placement)')
    for node, nic in flags['nics']:
        messages.append(f'NIC {nic} on {node} is below threshold on every pair (NIC or cable)')
    for (src_node, src_nic), (dst_node, dst_nic) in flags['pairs']:
        messages.append(f'NIC pair {src_node}:{src_nic} -> {dst_node}:{dst_nic} is below threshold')
    return messages


def to_topology_json(matrix, flags, meta=None):
    """Serialize the full matrix and its flags."""
    return {
        'meta': dict(meta or {}),
        'median_gbps': flags.get('median'),
        'threshold_gbps': flags.get('threshold'),
        'matrix': [
            {
                'src_node': source[0],
                'src_nic': source[1],
                'dst_node': destination[0],
                'dst_nic': destination[1],
                'bandwidth_gbps': value,
            }
            for source, destination, value in _all_cells(matrix)
        ],
        'flagged': {
            'nodes': [{'src_node': a, 'dst_node': b} for a, b in flags.get('nodes', [])],
            'nics': [{'node': a, 'nic': b} for a, b in flags.get('nics', [])],
            'pairs': [
                {'src_node': s[0], 'src_nic': s[1], 'dst_node': d[0], 'dst_nic': d[1]}
                for s, d in flags.get('pairs', [])
            ],
        },
    }
