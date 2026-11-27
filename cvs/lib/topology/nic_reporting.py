"""Run Deck records for the cross-node NIC bandwidth matrix.

Cells carry bandwidth as a percentage of the cluster median rather than GB/s,
so one threshold reads the same on every cluster and the colour ramp bands mean
the same thing everywhere.
"""

from cvs.lib.transferbench_rundeck import build_node_record, make_meta, record_group

GROUP = 'nic_bandwidth'
CLUSTER_GRID_META_KEY = 'nic_grid'
UNIT = '% of median'

__all__ = [
    'GROUP',
    'CLUSTER_GRID_META_KEY',
    'UNIT',
    'build_heatmap',
    'cell_label',
    'node_heatmap',
    'cluster_heatmap',
    'record_results',
    'make_meta',
]


def cell_label(node, nic):
    return f'{node}:{nic}'


def build_heatmap(name, rows, cols, values, threshold, row_label='', col_label='', scale='graded'):
    """Assemble one heatmap, rejecting a grid whose shape does not line up.

    Grids published through ``_meta`` skip the dataset builder's validation, so
    the shape is checked here instead of silently rendering a ragged table.
    """
    if not rows or not cols:
        return None
    if len(values) != len(rows) or any(not isinstance(row, list) or len(row) != len(cols) for row in values):
        raise ValueError(f'heatmap {name!r} is {len(values)}x? but needs {len(rows)}x{len(cols)}')
    heat = {
        'name': name,
        'unit': UNIT,
        'rows': [str(row) for row in rows],
        'cols': [str(col) for col in cols],
        'values': values,
        'threshold': threshold,
        'direction': 'higher',
        'scale': scale,
    }
    if row_label:
        heat['row_label'] = row_label
    if col_label:
        heat['col_label'] = col_label
    return heat


def _percent(value, median):
    if value is None or not median:
        return None
    return round(value / median * 100.0, 2)


def _sources(matrix):
    return sorted(matrix)


def _destinations(matrix):
    seen = set()
    for row in matrix.values():
        seen.update(row)
    return sorted(seen)


def node_heatmap(node, matrix, median, threshold):
    """That node's NICs against every peer NIC.

    Same-node columns are left empty: cross-node pairs are the only ones
    measured, so there is no value to show.
    """
    rows = [source for source in _sources(matrix) if source[0] == node]
    cols = [dest for dest in _destinations(matrix) if dest[0] != node]
    if not rows or not cols:
        return None
    values = [[_percent(matrix.get(row, {}).get(col), median) for col in cols] for row in rows]
    return build_heatmap(
        'NIC bandwidth',
        [nic for _, nic in rows],
        [cell_label(peer, nic) for peer, nic in cols],
        values,
        threshold,
        row_label='source NIC',
        col_label='destination NIC',
    )


def node_rollup(matrix, median, threshold):
    """Node-to-node view: the worst NIC pair between each ordered node pair."""
    nodes = sorted({source[0] for source in matrix} | {dest[0] for dest in _destinations(matrix)})
    if not nodes:
        return None
    worst = {}
    for source, row in matrix.items():
        for dest, value in row.items():
            key = (source[0], dest[0])
            if key not in worst or value < worst[key]:
                worst[key] = value
    values = [[_percent(worst.get((src, dst)), median) for dst in nodes] for src in nodes]
    return build_heatmap(
        'Node to node (worst NIC pair)',
        nodes,
        nodes,
        values,
        threshold,
        row_label='source node',
        col_label='destination node',
    )


def cluster_heatmap(matrix, median, threshold):
    """Every source NIC against every destination NIC, cluster wide."""
    rows = _sources(matrix)
    cols = _destinations(matrix)
    if not rows or not cols:
        return None
    values = [[_percent(matrix.get(row, {}).get(col), median) for col in cols] for row in rows]
    return build_heatmap(
        'NIC to NIC bandwidth',
        [cell_label(node, nic) for node, nic in rows],
        [cell_label(node, nic) for node, nic in cols],
        values,
        threshold,
        row_label='source NIC',
        col_label='destination NIC',
    )


def _node_status(node, flags):
    for src_node, dst_node in flags.get('nodes', []):
        if node in (src_node, dst_node):
            return 'fail'
    for flagged_node, _ in flags.get('nics', []):
        if flagged_node == node:
            return 'fail'
    for source, dest in flags.get('pairs', []):
        if node in (source[0], dest[0]):
            return 'fail'
    return 'pass'


def record_results(results_store, matrix, flags, meta=None, excluded_by_node=None):
    """Record the matrix as a deck group, one record per node plus the cluster grids.

    A node that was measured, and a node whose every NIC was excluded, both get
    a record: dropping the second would lose the only report of why it was left
    out. The node-to-node rollup is cluster-wide, so it is published once beside
    the cluster grid rather than copied into all N node records, which would put
    N^3 cells in the deck.
    """
    median = flags.get('median')
    threshold_pct = flags.get('threshold_pct')
    measured = {source[0] for source in matrix} | {dest[0] for dest in _destinations(matrix)}
    nodes = sorted(measured | set(excluded_by_node or {}))

    node_records = {}
    for node in nodes:
        heatmaps = [heat for heat in (node_heatmap(node, matrix, median, threshold_pct),) if heat]
        excluded = (excluded_by_node or {}).get(node) or []
        items = [{'name': f'{nic["nic"]} excluded', 'status': 'na', 'detail': nic['reason']} for nic in excluded]
        if node in measured:
            status = _node_status(node, flags)
            summary = f'{len(heatmaps)} heatmap(s)'
        else:
            status = 'na'
            summary = 'not measured: no addressable backend NIC'
        node_records[node] = build_node_record(status, items=items, items_summary=summary, heatmaps=heatmaps)

    record_group(results_store, GROUP, node_records, meta=meta)

    cluster_grids = [
        heat
        for heat in (cluster_heatmap(matrix, median, threshold_pct), node_rollup(matrix, median, threshold_pct))
        if heat
    ]
    store_meta = results_store.setdefault('_meta', {})
    store_meta[CLUSTER_GRID_META_KEY] = {'heatmaps': cluster_grids} if cluster_grids else {}
    return results_store
