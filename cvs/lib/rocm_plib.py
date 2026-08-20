'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json

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


def _amd_smi_json_command(args: str) -> str:
    """Build a portable amd-smi JSON command for nodes with different install paths."""
    return (
        "sudo bash -lc '"
        "if command -v amd-smi >/dev/null 2>&1; then AMD_SMI=$(command -v amd-smi); "
        "elif [ -x /opt/rocm/bin/amd-smi ]; then AMD_SMI=/opt/rocm/bin/amd-smi; "
        "else echo \"[]\"; exit 0; fi; "
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


def get_amd_smi_ecc_blocks_dict(phdl, blocks=None):
    """Capture named ECC_BLOCKS counters from ``amd-smi metric -g all --json``.

    Does not call fail_test: missing amd-smi or bad JSON is logged and skipped.
    Extra amd-smi blocks outside RCCL_ECC_BLOCKS are ignored. Top-level ``ecc``
    totals are ignored (those are what ``metric --ecc`` reports).

    Args:
        blocks: Optional subset of RCCL_ECC_BLOCKS to capture. Defaults to all blocks.
    """
    ecc_dict = {}
    raw_by_node = phdl.exec(_amd_smi_json_command('metric -g all'), print_console=False)
    for node, payload in raw_by_node.items():
        log.debug('amd-smi metric -g all output node=%s:\n%s', node, payload)
        parsed = _parse_amd_smi_json_payload(node, payload)
        ecc_dict[node] = {}
        if parsed is None:
            continue
        for gpu_dict in _iter_amd_smi_gpu_dicts(parsed):
            if not isinstance(gpu_dict, dict):
                continue
            gpu_id = gpu_dict.get('gpu', 0)
            ecc_dict[node][gpu_id] = _project_ecc_blocks(gpu_dict.get('ecc_blocks'), node, gpu_id, blocks=blocks)
    return ecc_dict


def log_ecc_blocks_snapshot(label, ecc_dict, blocks=None):
    for node in sorted(ecc_dict.keys(), key=str):
        for gpu_id in sorted(ecc_dict[node].keys(), key=str):
            log.debug(
                'ECC_BLOCKS %s: node=%s gpu=%s %s',
                label,
                node,
                gpu_id,
                _format_ecc_block_fields(ecc_dict[node][gpu_id], blocks=blocks),
            )


def _gpu_id_map(node_gpus):
    return {str(gpu_id): gpu_id for gpu_id in node_gpus}


def _ecc_delta_result(n_increased, n_decreased):
    if n_increased and n_decreased:
        return 'MIXED'
    if n_increased:
        return 'INCREASED'
    if n_decreased:
        return 'DECREASED'
    return 'CLEAN'


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


def format_ecc_blocks_report(before, after, collective=None, blocks=None):
    """Build the post-test ECC_BLOCKS INFO table (one section per node)."""
    sections = []
    nodes = sorted(set(before) | set(after), key=str)
    for node in nodes:
        before_gpus = _gpu_id_map(before.get(node, {}))
        after_gpus = _gpu_id_map(after.get(node, {}))
        gpu_keys = sorted(set(before_gpus) | set(after_gpus), key=str)
        gpu_sections = []
        node_increased = 0
        node_decreased = 0
        for gpu_key in gpu_keys:
            gpu_id = after_gpus.get(gpu_key, before_gpus.get(gpu_key))
            before_blocks = before.get(node, {}).get(before_gpus.get(gpu_key, gpu_id), {})
            after_blocks = after.get(node, {}).get(after_gpus.get(gpu_key, gpu_id), {})
            rows, n_increased, n_decreased, _changes = _gpu_ecc_rows(before_blocks, after_blocks, blocks=blocks)
            node_increased += n_increased
            node_decreased += n_decreased
            gpu_result = _ecc_delta_result(n_increased, n_decreased)
            gpu_lines = [f'  gpu={gpu_id}  result={gpu_result}', f'    {_ecc_table_header()}']
            gpu_lines.extend(f'    {row}' for row in rows)
            gpu_sections.append('\n'.join(gpu_lines))
        header = 'ECC_BLOCKS'
        if collective:
            header += f'  collective={collective}'
        header += (
            f'  node={node}  result={_ecc_delta_result(node_increased, node_decreased)}'
            f'  increased={node_increased}  decreased={node_decreased}'
        )
        sections.append(header + '\n' + '\n'.join(gpu_sections))
    return '\n'.join(sections)


def compare_ecc_blocks_snapshots(before, after, collective=None, blocks=None):
    """Log a post-test ECC_BLOCKS table and warn on counter increases.

    INFO is one aligned table per node (CE/UE/DE before, after, Delta). Capture
    one-liners stay at DEBUG. Never calls fail_test.
    Returns a list of increase dicts for unit tests.
    """
    increases = []
    nodes = sorted(set(before) | set(after), key=str)
    for node in nodes:
        before_gpus = _gpu_id_map(before.get(node, {}))
        after_gpus = _gpu_id_map(after.get(node, {}))
        gpu_keys = sorted(set(before_gpus) | set(after_gpus), key=str)
        for gpu_key in gpu_keys:
            gpu_id = after_gpus.get(gpu_key, before_gpus.get(gpu_key))
            before_blocks = before.get(node, {}).get(before_gpus.get(gpu_key, gpu_id), {})
            after_blocks = after.get(node, {}).get(after_gpus.get(gpu_key, gpu_id), {})
            _rows, _n_inc, _n_dec, changes = _gpu_ecc_rows(before_blocks, after_blocks, blocks=blocks)
            for change in changes:
                if change['delta'] > 0:
                    log.warning(
                        'ECC_BLOCKS increased: node=%s gpu=%s block=%s field=%s before=%s after=%s delta=%s',
                        node,
                        gpu_id,
                        change['block'],
                        change['field'],
                        change['before'],
                        change['after'],
                        change['delta'],
                    )
                    increases.append({'node': node, 'gpu': gpu_id, **change})
                elif change['delta'] < 0:
                    log.warning(
                        'ECC_BLOCKS decreased: node=%s gpu=%s block=%s field=%s before=%s after=%s delta=%s',
                        node,
                        gpu_id,
                        change['block'],
                        change['field'],
                        change['before'],
                        change['after'],
                        change['delta'],
                    )
    report = format_ecc_blocks_report(before, after, collective=collective, blocks=blocks)
    if report:
        log.info('%s', f'======== ECC_BLOCKS report ========\n{report}')
    return increases


def ecc_delta_check_enabled(config_dict):
    return bool(re.search('True', str(config_dict.get('cvs_params', {}).get('verify_ecc_delta', 'False')), re.I))


_RCCL_ECC_BLOCKS_BY_UPPER = {block.upper(): block for block in RCCL_ECC_BLOCKS}


def resolve_ecc_blocks(config_dict):
    """Return the ECC block subset to capture/compare from cvs_params.verify_ecc_blocks."""
    raw = config_dict.get('cvs_params', {}).get('verify_ecc_blocks')
    if not raw:
        return RCCL_ECC_BLOCKS
    if isinstance(raw, str):
        raw = [entry.strip() for entry in raw.split(',') if entry.strip()]
    if not isinstance(raw, (list, tuple)):
        log.warning('ECC_BLOCKS: verify_ecc_blocks must be a list; using all blocks')
        return RCCL_ECC_BLOCKS
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
        log.warning('ECC_BLOCKS: unknown verify_ecc_blocks entries %s; ignoring', unknown)
    if not resolved:
        log.warning('ECC_BLOCKS: no valid verify_ecc_blocks entries; using all blocks')
        return RCCL_ECC_BLOCKS
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


def get_amd_smi_fw_dict(phdl):
    firmware_dict = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('firmware')))
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


def get_gpu_pcie_bus_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showbus --json'))
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
