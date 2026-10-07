'''
Build the structured results dict consumed by the Run Deck status_matrix
builder from the preflight_results bundle each check already stores.

Shape (consumed by cvs/lib/report/rundeck/dataset_builders/status_matrix.py)::

    {
      "_meta":  {"cluster": ..., "version": ..., "version_label": "ROCm version", "suite": ...},
      "groups": {"<check>": {"nodes": {"<label>": <node record>}}},
    }

A skipped or blocked check is n/a. A node-level PASS does not mark checks
Primus left unscored as passed. Numeric GEMM, HBM, and local RCCL values are
attached when the payload carries them, including values stored on the Tier 1
per-GPU details rather than under tier2.
'''

from cvs.lib.preflight.node_smoke_rows import (
    build_tier1_metric_rows,
    build_tier2_metric_rows,
    build_tier3_metric_rows,
)

_STATUSES = ('pass', 'fail', 'na')

NODE_REACHABILITY = 'node_reachability'
NODE_HEALTH = 'node_health'
ROCM_VERSIONS = 'rocm_versions'
IFOE_L2 = 'ifoe_l2_connectivity'
TRANSFERBENCH = 'transferbench_smoke'
INTERFACE_NAMES = 'interface_names'
GID_CONSISTENCY = 'gid_consistency'
NODE_SMOKE_TIER1 = 'node_smoke_tier1'
NODE_SMOKE_TIER2 = 'node_smoke_tier2'
NODE_SMOKE_TIER3 = 'node_smoke_tier3'
RDMA_CONNECTIVITY = 'rdma_connectivity'

CHECK_IDS = (
    NODE_REACHABILITY,
    NODE_HEALTH,
    ROCM_VERSIONS,
    IFOE_L2,
    TRANSFERBENCH,
    INTERFACE_NAMES,
    GID_CONSISTENCY,
    NODE_SMOKE_TIER1,
    NODE_SMOKE_TIER2,
    NODE_SMOKE_TIER3,
    RDMA_CONNECTIVITY,
)

LIFECYCLE_BY_TEST = {
    'test_node_reachability': NODE_REACHABILITY,
    'test_node_health': NODE_HEALTH,
    'test_rocm_version_consistency': ROCM_VERSIONS,
    'test_ifoe_l2_connectivity': IFOE_L2,
    'test_ifoe_transferbench_smoke': TRANSFERBENCH,
    'test_interface_name_consistency': INTERFACE_NAMES,
    'test_gid_consistency': GID_CONSISTENCY,
    'test_node_smoke_tier1': NODE_SMOKE_TIER1,
    'test_node_smoke_tier2': NODE_SMOKE_TIER2,
    'test_node_smoke_tier3': NODE_SMOKE_TIER3,
    'test_rdma_connectivity': RDMA_CONNECTIVITY,
}


def make_meta(cluster_dict, suite_name, version=None):
    '''Assemble the _meta block for the deck run card.'''
    cluster = (cluster_dict or {}).get('cluster_name') or (cluster_dict or {}).get('name') or '—'
    resolved = version if version not in (None, '') else '—'
    return {
        'cluster': cluster,
        'version': resolved,
        'version_label': 'ROCm version',
        'suite': suite_name or 'preflight_checks',
        'generated_at': '',
    }


def build_preflight_deck(preflight_results, cluster_dict=None, suite_name='preflight_checks'):
    '''Turn a preflight_results bundle into a status_matrix results dict.'''
    results = preflight_results if isinstance(preflight_results, dict) else {}
    nodes = _known_nodes(results, cluster_dict)
    groups = {}
    for check_id in CHECK_IDS:
        records = _BUILDERS[check_id](results, nodes)
        if records:
            groups[check_id] = {'nodes': records}
    return {
        '_meta': make_meta(cluster_dict, suite_name, _rocm_version(results)),
        'groups': groups,
    }


def _record(status, items=None, summary='', metrics=None, series=None):
    normalized = _normalize_status(status)
    body = {
        'status': normalized,
        'items_summary': summary or '',
        'items': list(items or []),
        'errors_json_href': '',
        'log_tarball_href': '',
    }
    if metrics:
        body['metrics'] = metrics
    if series:
        body['series'] = series
    return body


def _normalize_status(status):
    token = str(status or 'na').strip().lower()
    if token in ('warning',):
        return 'pass'
    if token in ('pass', 'passed', 'ok', 'true', 'success'):
        return 'pass'
    if token in ('fail', 'failed', 'error', 'false'):
        return 'fail'
    if token in _STATUSES:
        return token
    return 'na'


def _item(name, status, message=''):
    return {'name': str(name), 'status': _normalize_status(status), 'message': str(message or '')}


def _fail_items(messages, name='check'):
    return [_item(name, 'fail', message) for message in messages or [] if message]


def _summarize(items, fallback=''):
    if not items:
        return fallback or ''
    counts = {}
    for item in items:
        counts[item.get('status') or 'na'] = counts.get(item.get('status') or 'na', 0) + 1
    parts = [f"{counts[status]} {status}" for status in ('fail', 'pass', 'na') if counts.get(status)]
    return ', '.join(parts)


def _status_from_items(items):
    if any(item.get('status') == 'fail' for item in items):
        return 'fail'
    if any(item.get('status') == 'pass' for item in items):
        return 'pass'
    return 'na'


def _na_nodes(nodes, message):
    labels = list(nodes) or ['cluster']
    return {label: _record('na', [], message or 'Not run') for label in labels}


def _not_run(result):
    if not isinstance(result, dict):
        return False
    if result.get('skipped') or result.get('blocked'):
        return True
    return str(result.get('status') or '').upper() in ('SKIPPED', 'BLOCKED')


def _skip_message(result):
    if not isinstance(result, dict):
        return 'Not run'
    return str(result.get('message') or 'Not run')


def _known_nodes(results, cluster_dict):
    nodes = []

    def add(name):
        text = str(name or '').strip()
        if text and text not in nodes:
            nodes.append(text)

    for name in (cluster_dict or {}).get('node_dict') or {}:
        add(name)
    for key in (
        NODE_REACHABILITY,
        NODE_HEALTH,
        IFOE_L2,
        TRANSFERBENCH,
        NODE_SMOKE_TIER1,
        'node_smoke',
        NODE_SMOKE_TIER3,
        'tier3_info',
        RDMA_CONNECTIVITY,
    ):
        block = results.get(key)
        if not isinstance(block, dict):
            continue
        for name in block.get('unreachable_nodes') or []:
            add(name)
        for name in (block.get('coverage') or {}).get('missing_nodes') or []:
            add(name)
        for container in ('nodes', 'node_results', 'node_status'):
            nested = block.get(container)
            if isinstance(nested, dict):
                for name in nested:
                    add(name)
        if key in (ROCM_VERSIONS, INTERFACE_NAMES, GID_CONSISTENCY) or _flat_node_map(block):
            for name in _flat_node_map(block):
                add(name)
    for key in (ROCM_VERSIONS, INTERFACE_NAMES, GID_CONSISTENCY):
        for name in _flat_node_map(results.get(key)):
            add(name)
    return nodes


def _flat_node_map(result):
    '''Per-node maps (ROCm, interfaces, GID) have no summary wrapper.'''
    if not isinstance(result, dict) or _not_run(result):
        return {}
    if any(
        key in result
        for key in (
            'node_results',
            'pair_results',
            'unreachable_nodes',
            'nodes',
            'node_status',
            'coverage',
            'vpod_membership',
        )
    ):
        return {}
    found = {}
    for name, value in result.items():
        if isinstance(value, dict) and any(
            key in value for key in ('status', 'detected_version', 'interfaces', 'errors')
        ):
            found[str(name)] = value
    return found


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(name, value, unit, threshold):
    numeric = _number(value)
    if numeric is None:
        return None
    limit = _number(threshold)
    body = {
        'name': name,
        'value': numeric,
        'unit': unit,
        'direction': 'higher',
        'status': 'fail' if limit is not None and numeric < limit else 'pass',
    }
    if limit is not None:
        body['threshold'] = limit
    return body


def _series(name, points, unit, y_label):
    kept = []
    for point in points:
        y_val = _number(point.get('y'))
        if y_val is None:
            continue
        kept.append({'x': point.get('x'), 'y': y_val})
    if len(kept) < 2:
        return None
    return {'name': name, 'unit': unit, 'x_label': 'GPU', 'y_label': y_label, 'points': kept}


def _rocm_version(results):
    mapping = _flat_node_map(results.get(ROCM_VERSIONS))
    detected = []
    for record in mapping.values():
        version = record.get('detected_version')
        if version and version != 'NOT_FOUND':
            detected.append(str(version))
    if not detected:
        return None
    passing = [
        str(record.get('detected_version'))
        for record in mapping.values()
        if str(record.get('status') or '').upper() == 'PASS'
        and record.get('detected_version') not in (None, '', 'NOT_FOUND')
    ]
    return passing[0] if passing else detected[0]


def _reachability_records(results, nodes):
    block = results.get(NODE_REACHABILITY)
    if not isinstance(block, dict):
        return None
    if _not_run(block):
        return _na_nodes(nodes, _skip_message(block))
    stored = block.get('nodes') if isinstance(block.get('nodes'), dict) else {}
    unreachable = {str(node) for node in block.get('unreachable_nodes') or []}
    labels = list(stored) or [node for node in nodes if node in unreachable or stored]
    if not labels and unreachable:
        labels = sorted(unreachable)
    if not labels and str(block.get('status') or '').upper() == 'PASS':
        labels = list(nodes)
    records = {}
    for node in labels:
        entry = stored.get(node) if isinstance(stored.get(node), dict) else {}
        if entry.get('status'):
            status = _normalize_status(entry.get('status'))
        elif node in unreachable:
            status = 'fail'
        else:
            status = 'pass'
        message = 'unreachable' if status == 'fail' else 'SSH ok'
        items = [_item('ssh', 'fail', 'SSH unreachable')] if status == 'fail' else []
        records[str(node)] = _record(status, items, message)
    return records or _na_nodes(nodes, 'Node reachability did not record any nodes')


def _node_map_records(block, nodes, item_name, summary_for, items_for):
    if not isinstance(block, dict):
        return None
    if _not_run(block):
        return _na_nodes(nodes, _skip_message(block))
    mapping = _flat_node_map(block)
    if not mapping and isinstance(block.get('node_results'), dict):
        mapping = {str(node): value for node, value in block['node_results'].items() if isinstance(value, dict)}
    if not mapping and isinstance(block.get('nodes'), dict):
        mapping = {str(node): value for node, value in block['nodes'].items() if isinstance(value, dict)}
    records = {}
    for node, entry in mapping.items():
        status = _normalize_status(entry.get('status'))
        items = items_for(entry) if status == 'fail' else []
        records[node] = _record(status, items, summary_for(entry))
    missing = (block.get('coverage') or {}).get('missing_nodes') or []
    for node in missing:
        records.setdefault(
            str(node), _record('fail', [_item(item_name, 'fail', 'Check did not run on this node')], 'not run')
        )
    return records


def _health_records(results, nodes):
    block = results.get(NODE_HEALTH)
    if not isinstance(block, dict):
        return None

    def _summary(entry):
        errors = entry.get('errors') or []
        return errors[0] if errors else ('pass' if _normalize_status(entry.get('status')) == 'pass' else '')

    def _items(entry):
        return _fail_items(entry.get('errors'), 'node_health')

    return _node_map_records(block, nodes, 'node_health', _summary, _items)


def _rocm_records(results, nodes):
    block = results.get(ROCM_VERSIONS)
    if not isinstance(block, dict):
        return None

    def _summary(entry):
        detected = entry.get('detected_version') or 'unknown'
        expected = entry.get('expected_version')
        if expected and str(entry.get('status') or '').upper() == 'FAIL':
            return f"{detected} (expected {expected})"
        return str(detected)

    def _items(entry):
        return _fail_items(entry.get('errors'), 'rocm_version')

    return _node_map_records(block, nodes, 'rocm_version', _summary, _items)


def _ifoe_records(results, nodes):
    block = results.get(IFOE_L2)
    if not isinstance(block, dict):
        return None

    def _summary(entry):
        errors = entry.get('errors') or []
        return errors[0] if errors else 'L2 ping pass'

    def _items(entry):
        return _fail_items(entry.get('errors'), 'ifoe_l2')

    return _node_map_records(block, nodes, 'ifoe_l2', _summary, _items)


def _transferbench_records(results, nodes):
    block = results.get(TRANSFERBENCH)
    if not isinstance(block, dict):
        return None

    def _summary(entry):
        errors = entry.get('errors') or []
        if errors:
            return errors[0]
        return str(entry.get('status') or '')

    def _items(entry):
        return _fail_items(entry.get('errors'), 'transferbench')

    return _node_map_records(block, nodes, 'transferbench', _summary, _items)


def _interface_records(results, nodes):
    block = results.get(INTERFACE_NAMES)
    if not isinstance(block, dict):
        return None

    def _summary(entry):
        found = entry.get('found_interfaces') or []
        expected = entry.get('expected_interfaces') or []
        if expected:
            return f"{len(found)}/{len(expected)} interfaces"
        return f"{len(found)} interfaces"

    def _items(entry):
        return _fail_items(entry.get('errors'), 'interface')

    return _node_map_records(block, nodes, 'interface', _summary, _items)


def _gid_records(results, nodes):
    block = results.get(GID_CONSISTENCY)
    if not isinstance(block, dict):
        return None

    def _summary(entry):
        interfaces = entry.get('interfaces') or {}
        ok = sum(1 for iface in interfaces.values() if isinstance(iface, dict) and iface.get('status') == 'OK')
        if interfaces:
            return f"{ok}/{len(interfaces)} GIDs"
        errors = entry.get('errors') or []
        return errors[0] if errors else ''

    def _items(entry):
        items = []
        for name, iface in (entry.get('interfaces') or {}).items():
            if isinstance(iface, dict) and iface.get('status') != 'OK':
                items.append(_item(name, 'fail', iface.get('error') or iface.get('status') or ''))
        if not items:
            items = _fail_items(entry.get('errors'), 'gid')
        return items

    return _node_map_records(block, nodes, 'gid', _summary, _items)


def _smoke_block(results):
    block = results.get(NODE_SMOKE_TIER1) or results.get('node_smoke')
    return block if isinstance(block, dict) else None


def _tier3_block(results):
    block = results.get(NODE_SMOKE_TIER3) or results.get('tier3_info')
    return block if isinstance(block, dict) else None


def _row_items(rows, measurements):
    items = []
    for row in rows:
        token = str(row.get('status') or '').lower()
        if token == 'fail':
            status = 'fail'
        elif token in ('pass', 'record'):
            status = 'pass'
        else:
            status = 'na'
        message = str(row.get('reason') or '')
        measured = measurements.get(row.get('metric')) if measurements else None
        if measured:
            message = measured
        elif row.get('actual') is not None:
            message = f"{row.get('actual')} {row.get('unit') or ''}".strip()
        items.append(_item(row.get('label') or row.get('metric'), status, message))
    return items


def _apply_node_failure(items, node_status, fail_reasons):
    status = _status_from_items(items)
    if _normalize_status(node_status) == 'fail':
        status = 'fail'
        if not any(item.get('status') == 'fail' for item in items):
            items = list(items) + _fail_items(fail_reasons or ['node smoke failed'], 'node')
    return status, items


def _tier1_records(results, nodes):
    block = _smoke_block(results)
    if block is None:
        return None
    if _not_run(block):
        return _na_nodes(nodes, _skip_message(block))
    rows = build_tier1_metric_rows(block)
    node_results = block.get('node_results') or {}
    labels = list(node_results) or [row.get('node') for row in rows if row.get('node')]
    records = {}
    for node in labels:
        node_rows = [row for row in rows if row.get('node') == node]
        entry = node_results.get(node) if isinstance(node_results.get(node), dict) else {}
        items = _row_items(node_rows, None)
        status, items = _apply_node_failure(items, entry.get('status'), entry.get('fail_reasons'))
        records[str(node)] = _record(status, items, _summarize(items, _skip_message(block) if not items else ''))
    return records


def _tier2_lookup(entry, thresholds):
    '''Map a tier2 row metric suffix onto a display string and chart inputs.'''
    payload = entry.get('node_payload') if isinstance(entry.get('node_payload'), dict) else {}
    tier1 = payload.get('tier1') if isinstance(payload.get('tier1'), dict) else {}
    tier2 = payload.get('tier2') if isinstance(payload.get('tier2'), dict) else {}
    texts = {}
    gemm_points = []
    hbm_points = []
    gemm_values = []
    hbm_values = []
    per_gpu = [gpu for gpu in (tier1.get('per_gpu') or []) if isinstance(gpu, dict)]
    tier2_gpus = [gpu for gpu in (tier2.get('per_gpu') or []) if isinstance(gpu, dict)]
    width = max(len(per_gpu), len(tier2_gpus))
    for index in range(width):
        details = {}
        gpu_id = index
        if index < len(per_gpu):
            details = per_gpu[index].get('details') if isinstance(per_gpu[index].get('details'), dict) else {}
            if per_gpu[index].get('gpu') is not None:
                gpu_id = per_gpu[index].get('gpu')
        tier2_entry = tier2_gpus[index] if index < len(tier2_gpus) else {}
        gemm = _number(details.get('gemm_tflops'))
        if gemm is None:
            gemm = _number(tier2_entry.get('gemm_tflops'))
        hbm = _number(details.get('hbm_gbs'))
        if hbm is None:
            hbm = _number(tier2_entry.get('hbm_gbs'))
        label = f"GPU {gpu_id}"
        if gemm is not None:
            texts[f"gpu{index}/large_gemm"] = f"{gemm:g} TFLOPS"
            gemm_points.append({'x': label, 'y': gemm})
            gemm_values.append(gemm)
        if hbm is not None:
            texts[f"gpu{index}/hbm_d2d"] = f"{hbm:g} GB/s"
            hbm_points.append({'x': label, 'y': hbm})
            hbm_values.append(hbm)
    rccl = tier2.get('rccl') if isinstance(tier2.get('rccl'), dict) else {}
    rccl_gbs = _number(rccl.get('gbs'))
    if rccl_gbs is None and isinstance(tier2.get('local_rccl'), dict):
        rccl_gbs = _number(tier2['local_rccl'].get('gbs'))
    metrics = []
    gemm_metric = _metric(
        'large_gemm', min(gemm_values) if gemm_values else None, 'TFLOPS', thresholds.get('gemm_tflops_min')
    )
    hbm_metric = _metric('hbm_d2d', min(hbm_values) if hbm_values else None, 'GB/s', thresholds.get('hbm_gbs_min'))
    rccl_metric = _metric('local_rccl', rccl_gbs, 'GB/s', thresholds.get('rccl_gbs_min'))
    for metric in (gemm_metric, hbm_metric, rccl_metric):
        if metric:
            metrics.append(metric)
    series = []
    for name, points, unit, y_label in (
        ('large_gemm', gemm_points, 'TFLOPS', 'TFLOPS'),
        ('hbm_d2d', hbm_points, 'GB/s', 'GB/s'),
    ):
        built = _series(name, points, unit, y_label)
        if built:
            series.append(built)
    if rccl_gbs is not None:
        texts['local_rccl'] = f"{rccl_gbs:g} GB/s"
    return texts, metrics, series


def _tier2_records(results, nodes):
    block = _smoke_block(results)
    if block is None:
        return None
    if _not_run(block) or not block.get('tier2_perf'):
        if _not_run(block):
            return _na_nodes(nodes, _skip_message(block))
        return None
    rows = build_tier2_metric_rows(block)
    node_results = block.get('node_results') or {}
    thresholds = block.get('tier2_thresholds') or {}
    labels = list(node_results) or [row.get('node') for row in rows if row.get('node')]
    records = {}
    for node in labels:
        entry = node_results.get(node) if isinstance(node_results.get(node), dict) else {}
        texts, metrics, series = _tier2_lookup(entry, thresholds)
        node_rows = [row for row in rows if row.get('node') == node]
        lookup = {}
        for row in node_rows:
            metric = str(row.get('metric') or '')
            for suffix, text in texts.items():
                if metric.endswith(suffix):
                    lookup[metric] = text
        items = _row_items(node_rows, lookup)
        status, items = _apply_node_failure(items, entry.get('status'), entry.get('fail_reasons'))
        records[str(node)] = _record(status, items, _summarize(items), metrics, series)
    return records


def _tier3_records(results, nodes):
    block = _tier3_block(results)
    if block is None:
        return None
    if _not_run(block):
        return _na_nodes(nodes, _skip_message(block))
    node_results = block.get('node_results') or {}
    records = {}
    for node, entry in node_results.items():
        entry = entry if isinstance(entry, dict) else {}
        sliced = {
            'skipped': False,
            'checks': block.get('checks') or entry.get('checks'),
            'failed_nodes': [node]
            if str(entry.get('status') or '').upper() == 'FAIL' or node in (block.get('failed_nodes') or [])
            else [],
            'node_results': {node: entry},
        }
        items = _row_items(build_tier3_metric_rows(sliced), None)
        status, items = _apply_node_failure(items, entry.get('status'), entry.get('fail_reasons'))
        records[str(node)] = _record(status, items, _summarize(items))
    return records


def _rdma_records(results, nodes):
    block = results.get(RDMA_CONNECTIVITY)
    if not isinstance(block, dict):
        return None
    if _not_run(block):
        return _na_nodes(nodes, _skip_message(block))
    node_status = block.get('node_status') if isinstance(block.get('node_status'), dict) else {}
    pairs = block.get('pair_results') if isinstance(block.get('pair_results'), dict) else {}
    involved = set(node_status)
    failed_by_node = {node: [] for node in involved}
    for pair_key, pair in pairs.items():
        if not isinstance(pair, dict):
            continue
        members = [pair.get('server_node'), pair.get('client_node')]
        for member in members:
            if member:
                involved.add(str(member))
                failed_by_node.setdefault(str(member), [])
        if str(pair.get('status') or '').upper() == 'FAIL':
            detail = '; '.join(pair.get('error_details') or []) or pair_key
            for member in members:
                if member:
                    failed_by_node.setdefault(str(member), []).append(_item(pair_key, 'fail', detail))
    if not involved:
        return _na_nodes(nodes, block.get('message') or 'RDMA connectivity produced no node results')
    records = {}
    for node in involved:
        stats = node_status.get(node) if isinstance(node_status.get(node), dict) else {}
        failed = int(stats.get('failed_tests') or 0)
        successful = int(stats.get('successful_tests') or 0)
        items = failed_by_node.get(node) or []
        if failed and not items:
            items = [_item('rdma', 'fail', f"{failed} failed test(s)")]
        status = 'fail' if failed or items else 'pass'
        total = failed + successful
        summary = f"{successful}/{total} pairs" if total else _summarize(items, 'connected')
        records[node] = _record(status, items, summary)
    return records


_BUILDERS = {
    NODE_REACHABILITY: _reachability_records,
    NODE_HEALTH: _health_records,
    ROCM_VERSIONS: _rocm_records,
    IFOE_L2: _ifoe_records,
    TRANSFERBENCH: _transferbench_records,
    INTERFACE_NAMES: _interface_records,
    GID_CONSISTENCY: _gid_records,
    NODE_SMOKE_TIER1: _tier1_records,
    NODE_SMOKE_TIER2: _tier2_records,
    NODE_SMOKE_TIER3: _tier3_records,
    RDMA_CONNECTIVITY: _rdma_records,
}
