'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json

from pssh.exceptions import Timeout

from cvs.lib.utils_lib import *

RCCL_ECC_BLOCKS = (
    'UMC',
    'SDMA',
    'GFX',
    'MMHUB',
    'PCIE_BIF',
    'HDP',
    'XGMI_WAFL',
)
ECC_COUNTER_FIELDS = ('correctable_count', 'uncorrectable_count', 'deferred_count')
ECC_TABLE_COLUMNS = (
    ('correctable_count', 'CE'),
    ('uncorrectable_count', 'UE'),
    ('deferred_count', 'DE'),
)
_ECC_BLOCK_COL_W = 10
_ECC_NUM_COL_W = 9
_ECC_SMI_TIMEOUT = 60


def _amd_smi_json_command(args: str, use_sudo=True, strict=False) -> str:
    """Build a portable amd-smi JSON command for nodes with different install paths.

    use_sudo defaults to True so existing callers keep ``sudo bash -lc``. Host checks
    on a scheduler job pass False: bare ``sudo`` waits for a password on the HTTP
    agent, and firmware JSON is readable without root. A missing binary prints
    ``[]`` and exits 0, so the JSON parser gets an empty payload instead of a
    failed command. With ``strict`` it prints an error and exits 127 with no JSON
    instead, so the caller's JSON parse fails.
    """
    privilege = "sudo " if use_sudo else ""
    missing = "echo \"amd-smi not found\" >&2; exit 127" if strict else "echo \"[]\"; exit 0"
    return (
        f"{privilege}bash -lc '"
        "if command -v amd-smi >/dev/null 2>&1; then AMD_SMI=$(command -v amd-smi); "
        "elif [ -x /opt/rocm/bin/amd-smi ]; then AMD_SMI=/opt/rocm/bin/amd-smi; "
        f"else {missing}; fi; "
        "\"${AMD_SMI}\" "
        f"{args} --json'"
    )


def _iter_amd_smi_gpu_dicts(node_payload):
    if isinstance(node_payload, dict) and 'gpu_data' in node_payload:
        return list(node_payload['gpu_data'])
    if isinstance(node_payload, list):
        return node_payload
    return []


def _to_int_counter(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_amd_smi_json_payload(node, payload):
    if payload is None or payload == '':
        log.warning('ECC_BLOCKS: empty amd-smi output on node=%s; skipping', node)
        return None
    if isinstance(payload, (dict, list)):
        return payload
    try:
        return json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        log.warning('ECC_BLOCKS: failed to parse amd-smi JSON on node=%s; skipping', node)
        return None


def _ecc_blocks_or_default(blocks):
    return blocks if blocks is not None else RCCL_ECC_BLOCKS


def _project_ecc_blocks(raw_blocks, node, gpu_id, log_missing=True, blocks=None):
    projected = {}
    raw_blocks = raw_blocks if isinstance(raw_blocks, dict) else {}
    for block in _ecc_blocks_or_default(blocks):
        raw_fields = raw_blocks.get(block)
        if not isinstance(raw_fields, dict):
            if log_missing:
                log.debug(
                    'ECC_BLOCKS: node=%s gpu=%s block=%s not present in amd-smi output; treating counters as 0',
                    node,
                    gpu_id,
                    block,
                )
            projected[block] = {field: 0 for field in ECC_COUNTER_FIELDS}
            continue
        projected[block] = {field: _to_int_counter(raw_fields.get(field, 0)) for field in ECC_COUNTER_FIELDS}
    return projected


def _format_ecc_block_fields(block_dict, blocks=None):
    parts = []
    for block in _ecc_blocks_or_default(blocks):
        fields = block_dict.get(block, {})
        for field in ECC_COUNTER_FIELDS:
            parts.append(f'{block}.{field}={_to_int_counter(fields.get(field, 0))}')
    return ' '.join(parts)


def _phdl_host_names(phdl):
    for attr in ('reachable_hosts', 'host_list'):
        hosts = getattr(phdl, attr, None)
        if isinstance(hosts, (list, tuple)) and hosts:
            return list(hosts)
    return []


def _mark_ecc_capture_failed(ecc_dict, node, parsed):
    if parsed is not None:
        log.warning('ECC_BLOCKS: amd-smi returned no GPUs on node=%s; capture failed', node)
    ecc_dict[node] = None


def get_amd_smi_ecc_blocks_dict(phdl, blocks=None):
    """Capture named ECC_BLOCKS counters from ``amd-smi metric -g all --json``.

    Does not call fail_test. Missing amd-smi, empty output, bad JSON, and a
    command timeout are logged and stored as None so the report shows NO_DATA.
    That split is intentional: report_ecc_delta is log-only. With
    cluster_snapshot_debug also enabled, get_amd_smi_ras_metrics_dict and
    get_amd_smi_pcie_metrics_dict still parse through convert_phdl_json_to_dict,
    which records a test failure on bad JSON for the same amd-smi command.
    Extra amd-smi blocks outside RCCL_ECC_BLOCKS are ignored. Top-level ``ecc``
    totals are ignored (those are what ``metric --ecc`` reports).

    Args:
        blocks: Optional subset of RCCL_ECC_BLOCKS to capture. Defaults to all blocks.
    """
    ecc_dict = {}
    try:
        raw_by_node = phdl.exec(
            _amd_smi_json_command('metric -g all'),
            timeout=_ECC_SMI_TIMEOUT,
            print_console=False,
        )
    except Timeout:
        hosts = _phdl_host_names(phdl)
        log.warning(
            'ECC_BLOCKS: amd-smi timed out after %ss; capture failed on nodes=%s',
            _ECC_SMI_TIMEOUT,
            hosts,
        )
        return {node: None for node in hosts}
    for node, payload in raw_by_node.items():
        log.debug('amd-smi metric -g all output node=%s:\n%s', node, payload)
        parsed = _parse_amd_smi_json_payload(node, payload)
        gpu_dicts = _iter_amd_smi_gpu_dicts(parsed) if parsed is not None else []
        if not gpu_dicts:
            _mark_ecc_capture_failed(ecc_dict, node, parsed)
            continue
        ecc_dict[node] = {}
        for gpu_dict in gpu_dicts:
            if not isinstance(gpu_dict, dict):
                continue
            gpu_id = gpu_dict.get('gpu')
            if gpu_id is None:
                log.warning('ECC_BLOCKS: node=%s gpu_data entry missing gpu id; skipping', node)
                continue
            if gpu_id in ecc_dict[node]:
                log.warning('ECC_BLOCKS: node=%s duplicate gpu id=%s; keeping first', node, gpu_id)
                continue
            ecc_dict[node][gpu_id] = _project_ecc_blocks(gpu_dict.get('ecc_blocks'), node, gpu_id, blocks=blocks)
        if not ecc_dict[node]:
            _mark_ecc_capture_failed(ecc_dict, node, parsed)
    return ecc_dict


def log_ecc_blocks_snapshot(label, ecc_dict, blocks=None):
    for node in sorted(ecc_dict.keys(), key=str):
        node_gpus = ecc_dict[node]
        if not isinstance(node_gpus, dict):
            log.debug('ECC_BLOCKS %s: node=%s capture failed', label, node)
            continue
        for gpu_id in sorted(node_gpus.keys(), key=str):
            log.debug(
                'ECC_BLOCKS %s: node=%s gpu=%s %s',
                label,
                node,
                gpu_id,
                _format_ecc_block_fields(node_gpus[gpu_id], blocks=blocks),
            )


def _gpu_id_map(node_gpus):
    if not isinstance(node_gpus, dict):
        return {}
    return {str(gpu_id): gpu_id for gpu_id in node_gpus}


def _snapshot_capture_failed(snapshot, node):
    return node in snapshot and snapshot[node] is None


def _ecc_delta_result(n_increased, n_decreased):
    if n_increased and n_decreased:
        return 'MIXED'
    if n_increased:
        return 'INCREASED'
    if n_decreased:
        return 'DECREASED'
    return 'CLEAN'


def _missing_gpu_snapshot_result(before_present, after_present):
    if before_present and not after_present:
        return 'MISSING_AFTER'
    if after_present and not before_present:
        return 'MISSING_BEFORE'
    return None


def _format_signed_delta(delta):
    if delta > 0:
        return f'+{delta}'
    return str(delta)


def _ecc_table_header():
    parts = [f'{"block":<{_ECC_BLOCK_COL_W}}']
    for _field, prefix in ECC_TABLE_COLUMNS:
        for suffix in ('before', 'after', 'Delta'):
            parts.append(f'{f"{prefix} {suffix}":>{_ECC_NUM_COL_W}}')
    return '  '.join(parts)


def _ecc_table_row(block, values):
    parts = [f'{block:<{_ECC_BLOCK_COL_W}}']
    for value in values:
        parts.append(f'{value:>{_ECC_NUM_COL_W}}')
    return '  '.join(parts)


def _gpu_ecc_rows(before_blocks, after_blocks, blocks=None):
    rows = []
    n_increased = 0
    n_decreased = 0
    changes = []
    for block in _ecc_blocks_or_default(blocks):
        cells = []
        for field, _prefix in ECC_TABLE_COLUMNS:
            before_val = _to_int_counter(before_blocks.get(block, {}).get(field, 0))
            after_val = _to_int_counter(after_blocks.get(block, {}).get(field, 0))
            delta = after_val - before_val
            cells.extend([str(before_val), str(after_val), _format_signed_delta(delta)])
            if delta != 0:
                changes.append(
                    {
                        'block': block,
                        'field': field,
                        'before': before_val,
                        'after': after_val,
                        'delta': delta,
                    }
                )
            if delta > 0:
                n_increased += 1
            elif delta < 0:
                n_decreased += 1
        rows.append(_ecc_table_row(block, cells))
    return rows, n_increased, n_decreased, changes


def _ecc_snapshot_comparisons(before, after, collective=None, blocks=None):
    """Walk each node and GPU once.

    The report table and the warning list both come from this result, so a
    later change to delta handling stays in one place.
    """
    comparisons = []
    nodes = sorted(set(before) | set(after), key=str)
    for node in nodes:
        before_gpus = _gpu_id_map(before.get(node, {}))
        after_gpus = _gpu_id_map(after.get(node, {}))
        gpu_keys = sorted(set(before_gpus) | set(after_gpus), key=str)
        before_failed = _snapshot_capture_failed(before, node)
        after_failed = _snapshot_capture_failed(after, node)
        gpu_sections = []
        findings = []
        node_increased = 0
        node_decreased = 0
        node_missing = False
        for gpu_key in gpu_keys:
            before_present = gpu_key in before_gpus
            after_present = gpu_key in after_gpus
            gpu_id = after_gpus.get(gpu_key, before_gpus.get(gpu_key))
            missing_result = _missing_gpu_snapshot_result(before_present, after_present)
            if missing_result:
                gpu_sections.append(f'  gpu={gpu_id}  result={missing_result}')
                findings.append(('missing', {'node': node, 'gpu': gpu_id, 'result': missing_result}))
                node_missing = True
                continue
            before_blocks = before.get(node, {}).get(before_gpus[gpu_key], {})
            after_blocks = after.get(node, {}).get(after_gpus[gpu_key], {})
            rows, n_increased, n_decreased, changes = _gpu_ecc_rows(before_blocks, after_blocks, blocks=blocks)
            node_increased += n_increased
            node_decreased += n_decreased
            for change in changes:
                found = {'node': node, 'gpu': gpu_id, **change}
                if change['delta'] > 0:
                    findings.append(('increase', found))
                elif change['delta'] < 0:
                    findings.append(('decrease', found))
            gpu_result = _ecc_delta_result(n_increased, n_decreased)
            gpu_lines = [f'  gpu={gpu_id}  result={gpu_result}', f'    {_ecc_table_header()}']
            gpu_lines.extend(f'    {row}' for row in rows)
            gpu_sections.append('\n'.join(gpu_lines))
        if not gpu_keys and (before_failed or after_failed):
            node_result = 'NO_DATA'
        elif node_missing:
            node_result = 'MISSING_DATA'
        else:
            node_result = _ecc_delta_result(node_increased, node_decreased)
        header = 'ECC_BLOCKS'
        if collective:
            header += f'  collective={collective}'
        header += f'  node={node}  result={node_result}  increased={node_increased}  decreased={node_decreased}'
        comparisons.append(
            {
                'node': node,
                'section': header + '\n' + '\n'.join(gpu_sections),
                'findings': findings,
                'before_failed': before_failed,
                'after_failed': after_failed,
            }
        )
    return comparisons


def _format_ecc_report_sections(comparisons):
    return '\n'.join(item['section'] for item in comparisons)


def _log_ecc_finding(kind, found):
    if kind == 'missing':
        log.warning(
            'ECC_BLOCKS missing snapshot data: node=%s gpu=%s result=%s',
            found['node'],
            found['gpu'],
            found['result'],
        )
        return
    message = 'ECC_BLOCKS increased: node=%s gpu=%s block=%s field=%s before=%s after=%s delta=%s'
    if kind == 'decrease':
        message = 'ECC_BLOCKS decreased: node=%s gpu=%s block=%s field=%s before=%s after=%s delta=%s'
    log.warning(
        message,
        found['node'],
        found['gpu'],
        found['block'],
        found['field'],
        found['before'],
        found['after'],
        found['delta'],
    )


def compare_ecc_blocks_snapshots(before, after, collective=None, blocks=None):
    """Log a post-test ECC_BLOCKS table and warn on counter increases.

    INFO is one aligned table per node (CE/UE/DE before, after, Delta). Capture
    one-liners stay at DEBUG. A GPU present in only one snapshot is warned as
    missing data and is not compared as a counter change. A node whose capture
    failed on both sides is reported as NO_DATA. Never calls fail_test.
    Returns a list of increase dicts for unit tests.
    """
    comparisons = _ecc_snapshot_comparisons(before, after, collective=collective, blocks=blocks)
    increases = []
    for item in comparisons:
        if item['before_failed'] or item['after_failed']:
            log.warning(
                'ECC_BLOCKS capture failed: node=%s before_snapshot=%s after_snapshot=%s',
                item['node'],
                'failed' if item['before_failed'] else 'ok',
                'failed' if item['after_failed'] else 'ok',
            )
        for kind, found in item['findings']:
            _log_ecc_finding(kind, found)
            if kind == 'increase':
                increases.append(found)
    report = _format_ecc_report_sections(comparisons)
    if report:
        log.info('%s', f'======== ECC_BLOCKS report ========\n{report}')
    return increases


def ecc_delta_check_enabled(config_dict):
    return bool(re.search('True', str(config_dict.get('cvs_params', {}).get('report_ecc_delta', 'False')), re.I))


def skipped_sudo_work(config_dict):
    """Work the RCCL sudo gate skips. ECC capture is included only when report_ecc_delta is on."""
    parts = ['dmesg markers/verification']
    if ecc_delta_check_enabled(config_dict):
        parts.append('ECC_BLOCKS capture')
    parts.append('sudo-only snapshots')
    return ', '.join(parts)


_RCCL_ECC_BLOCKS_BY_UPPER = {block.upper(): block for block in RCCL_ECC_BLOCKS}


def resolve_ecc_blocks(config_dict):
    """Return the ECC block subset from cvs_params.report_ecc_blocks.

    Missing or empty means all blocks. An unknown name or a non-list value
    calls fail_test and does not substitute the full block list.
    """
    raw = config_dict.get('cvs_params', {}).get('report_ecc_blocks')
    if not raw:
        return RCCL_ECC_BLOCKS
    if isinstance(raw, str):
        raw = [entry.strip() for entry in raw.split(',') if entry.strip()]
    allowed = ', '.join(RCCL_ECC_BLOCKS)
    if not isinstance(raw, (list, tuple)):
        fail_test(f'ECC_BLOCKS: report_ecc_blocks must be a list, got {type(raw).__name__}. Allowed blocks: {allowed}')
        return ()
    resolved = []
    unknown = []
    for entry in raw:
        name = str(entry).strip().upper()
        if not name:
            continue
        canonical = _RCCL_ECC_BLOCKS_BY_UPPER.get(name)
        if canonical is None:
            unknown.append(entry)
            continue
        if canonical not in resolved:
            resolved.append(canonical)
    if unknown:
        fail_test(f'ECC_BLOCKS: unknown report_ecc_blocks entries {unknown}. Allowed blocks: {allowed}')
    if not resolved:
        if not unknown:
            fail_test(f'ECC_BLOCKS: report_ecc_blocks has no valid block names. Allowed blocks: {allowed}')
        return ()
    return tuple(resolved)


def capture_ecc_blocks_snapshot(phdl, label, blocks=None):
    ecc_dict = get_amd_smi_ecc_blocks_dict(phdl, blocks=blocks)
    log_ecc_blocks_snapshot(label, ecc_dict, blocks=blocks)
    return ecc_dict


def get_rocm_smi_dict(phdl):
    rocm_smi_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi -a --json'))
    return rocm_smi_dict


def get_gpu_partition_dict(phdl):
    amd_part_dict = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('partition')))
    return amd_part_dict


def get_gpu_process_dict(phdl):
    amd_proc_dict = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('process')))
    return amd_proc_dict


def get_amd_smi_metric_dict(phdl):
    amd_metric_dict = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('metric')))
    return amd_metric_dict


def get_amd_smi_fw_dict(phdl, use_sudo=True):
    firmware_dict = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('firmware', use_sudo=use_sudo)))
    return firmware_dict


def get_amd_smi_ras_metrics_dict(phdl):
    ras_dict = {}
    ras_dict_t = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('metric --ecc')))
    log.info("%s", ras_dict_t)
    for node in ras_dict_t.keys():
        ras_dict[node] = {}
        log.info('^^^^^')
        log.info("%s", ras_dict_t[node])
        for gpu_dict in _iter_amd_smi_gpu_dicts(ras_dict_t[node]):
            ras_dict[node][gpu_dict['gpu']] = gpu_dict['ecc']

    return ras_dict


def get_amd_smi_pcie_metrics_dict(phdl):
    pcie_dict = {}
    pcie_dict_t = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('metric --pcie')))
    for node in pcie_dict_t.keys():
        pcie_dict[node] = {}
        for gpu_dict in _iter_amd_smi_gpu_dicts(pcie_dict_t[node]):
            pcie_dict[node][gpu_dict['gpu']] = gpu_dict['pcie']
    return pcie_dict


def get_gpu_mem_use_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showmemuse --json'))
    return d_dict


def get_gpu_use_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showuse --json'))
    return d_dict


def get_gpu_metrics_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showmetric --json'))
    return d_dict


def get_gpu_fw_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showfwinfo --json'))
    return d_dict


def get_gpu_pcie_bus_dict(phdl, use_sudo=True):
    """Return ``{node: {'card<N>': {'PCI Bus': <bdf>}}}`` from ``amd-smi list``.

    amd-smi is found on PATH or at /opt/rocm/bin. With sudo a node that has neither
    fails the JSON parse, as rocm-smi did. Without sudo it maps to ``{}``, so callers
    walking every card never KeyError on it.
    """
    d_dict = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('list', use_sudo, strict=use_sudo)))
    for node, payload in d_dict.items():
        gpus = [g for g in _iter_amd_smi_gpu_dicts(payload) if isinstance(g, dict) and 'gpu' in g and 'bdf' in g]
        d_dict[node] = {f"card{gpu['gpu']}": {'PCI Bus': gpu['bdf']} for gpu in gpus}
    return d_dict


def get_gpu_model_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showproductname --json'))
    return d_dict


def get_gpu_temp_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showtemp --json'))
    return d_dict


def get_gpu_fabric_info_dict(phdl, use_sudo=True, amd_smi_path='amd-smi'):
    """Return ``amd-smi fabric --topology --json`` output per cluster node.

    Parsed via ``convert_phdl_json_to_dict``. The structure is amd-smi's JSON
    topology payload (GPU records with fabric port and pod-membership fields).
    Used by the AIMVT-181 IFoE TransferBench preflight check to detect pPod
    (physical pod) and vPod (virtual / logical pod) membership before invoking
    the TransferBench smoketest preset.

    Args:
        phdl: Parallel SSH handle for cluster nodes.
        use_sudo: When True (default), prefix the command with ``sudo``.
        amd_smi_path: Override for the ``amd-smi`` binary (e.g. an absolute
            path). Defaults to PATH-resolved ``amd-smi``.

    Returns:
        dict[str, Any]: ``{node: parsed_amd_smi_topology_json | str}``.
        When ``amd-smi`` output is not valid JSON on a node, the raw string is
        returned for that node so callers can degrade gracefully.
    """
    cmd = f'{amd_smi_path} fabric --topology --json'
    if use_sudo:
        cmd = 'sudo ' + cmd
    return convert_phdl_json_to_dict(phdl.exec(cmd))
