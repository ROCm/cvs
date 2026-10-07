'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Build the structured ``host_res_dict`` consumed by the Run Deck status_matrix
builder from the host-config command output CVS already collects per node.

Each group is one ``host_configs_cvs`` check. The cell verdict uses the same
match that check uses to call ``fail_test``. Parsed rows are drill-down only.

Shape (consumed by cvs/lib/report/rundeck/dataset_builders/status_matrix.py)::

    {
      "_meta":  {"cluster": ..., "version": ..., "version_label": "ROCm version", "suite": ...},
      "groups": {"<check>": {"nodes": {"<label>": <node record>}}},
    }
'''

import re

_STATUSES = ('pass', 'fail', 'na')

# Timed-stage labels in host_configs_cvs. The deck profile lists the same ids.
OS_RELEASE = 'os_release'
KERNEL_VERSION = 'kernel_version'
BIOS_VERSION = 'bios_version'
ROCM_VERSION = 'rocm_version'
GPU_FW = 'gpu_fw'
PCI_REALLOC = 'pci_realloc'
IOMMU_PT = 'iommu_pt'
NUMA_BALANCING = 'numa_balancing'
ONLINE_MEMORY = 'online_memory'
PCI_ACCELERATORS = 'pci_accelerators'
GPU_PCIE = 'gpu_pcie'
NIC_PCIE = 'nic_pcie'
PCI_ACS = 'pci_acs'
DMESG_DRIVER = 'dmesg_driver'
DMESG_RESET = 'dmesg_reset'
CHECK_IDS = (
    OS_RELEASE,
    KERNEL_VERSION,
    BIOS_VERSION,
    ROCM_VERSION,
    GPU_FW,
    PCI_REALLOC,
    IOMMU_PT,
    NUMA_BALANCING,
    ONLINE_MEMORY,
    PCI_ACCELERATORS,
    GPU_PCIE,
    NIC_PCIE,
    PCI_ACS,
    DMESG_DRIVER,
    DMESG_RESET,
)

_OS_ACTUAL_RE = re.compile(r'VERSION="(([0-9.\-_A-Z]+)\s+)', re.I)
_KERNEL_ACTUAL_RE = re.compile(r'([0-9.\-_]+generic)', re.I)
_BIOS_ACTUAL_RE = re.compile(r'([a-z0-9_.\-]+)', re.I)
_ROCM_ACTUAL_RE = re.compile(r'ROCm version:\s+([0-9.]+)', re.I)
_MEMORY_ACTUAL_RE = re.compile(r'Total online memory:\s+([0-9.A-Za-z]+)')
_ACCEL_RE = re.compile(r'accelerators:\s+Advanced', re.I)


def make_meta(cluster_dict, suite_name, version=None):
    '''Assemble the _meta block for the deck run card.'''
    cluster = (cluster_dict or {}).get('cluster_name') or (cluster_dict or {}).get('name') or '—'
    resolved = version if version not in (None, '') else '—'
    return {
        'cluster': cluster,
        'version': resolved,
        'version_label': 'ROCm version',
        'suite': suite_name or 'host_configs_cvs',
        'generated_at': '',
    }


def build_node_record(status, items=None, items_summary=''):
    '''Assemble one node record for a host-config check.'''
    normalized = str(status or 'na').lower()
    if normalized not in _STATUSES:
        normalized = 'na'
    return {
        'status': normalized,
        'items_summary': items_summary or '',
        'items': list(items or []),
        'errors_json_href': '',
        'log_tarball_href': '',
    }


def record_group(res_dict, group, node_records, meta=None):
    '''
    Merge one check's per-node records into the results dict.

    Idempotent per (group, node): a re-run of the same check overwrites.
    A later real ROCm version replaces a placeholder left by an earlier check.
    '''
    if meta:
        existing = res_dict.get('_meta')
        if not existing:
            res_dict['_meta'] = dict(meta)
        elif meta.get('version') not in (None, '', '—') and existing.get('version') in (None, '', '—'):
            existing['version'] = meta['version']
    groups = res_dict.setdefault('groups', {})
    group_entry = groups.setdefault(str(group), {'nodes': {}})
    group_entry['nodes'].update(node_records or {})
    return res_dict


def _fail_item(name, message):
    return {'name': name, 'status': 'fail', 'message': message}


def _pass_record(summary):
    return build_node_record('pass', [], summary)


def _fail_record(name, message, summary):
    return build_node_record('fail', [_fail_item(name, message)], summary)


def _version_mismatch(out_dict, expected, actual_re, item_name, message_for):
    '''Fail a node when ``expected`` is absent. ``actual_re`` must match on that path.'''
    records = {}
    messages = []
    for node, output in out_dict.items():
        if re.search(f'{expected}', output, re.I):
            records[str(node)] = _pass_record(str(expected))
            continue
        actual = actual_re.search(output).group(1)
        message = message_for(node, actual)
        messages.append(message)
        records[str(node)] = _fail_record(item_name, message, actual.strip())
    return records, messages


def _must_match(out_dict, pattern, item_name, message_for, pass_summary):
    '''Fail a node when ``pattern`` is absent.'''
    records = {}
    messages = []
    for node, output in out_dict.items():
        if re.search(pattern, output, re.I):
            records[str(node)] = _pass_record(pass_summary)
            continue
        message = message_for(node)
        messages.append(message)
        records[str(node)] = _fail_record(item_name, message, 'missing')
    return records, messages


def _take_denied(out_dict, token, item_name, message_for):
    '''Split nodes whose output contains ``token`` into failures. Return (records, messages, rest).'''
    records = {}
    messages = []
    rest = {}
    for node, output in out_dict.items():
        if token in str(output):
            message = message_for(node)
            messages.append(message)
            records[str(node)] = _fail_record(item_name, message, 'denied')
        else:
            rest[node] = output
    return records, messages, rest


def _must_not_match(out_dict, pattern, item_name, message_for, pass_summary):
    '''Fail a node when ``pattern`` is present.'''
    records = {}
    messages = []
    for node, output in out_dict.items():
        if not re.search(pattern, output, re.I):
            records[str(node)] = _pass_record(pass_summary)
            continue
        message = message_for(node)
        messages.append(message)
        records[str(node)] = _fail_record(item_name, message, 'seen')
    return records, messages


def eval_os_release(out_dict, expected):
    '''Return (node_records, fail_messages) for /etc/os-release.'''

    def _message(node, actual):
        return f'Installed OS Version {actual} not matching expected version {expected} on node {node}'

    return _version_mismatch(out_dict, expected, _OS_ACTUAL_RE, 'os_version', _message)


def eval_kernel_version(out_dict, expected):
    '''Return (node_records, fail_messages) for uname.'''

    def _message(node, actual):
        return f'Installed Kernel Version {actual} not matching expected version {expected} on node {node}'

    return _version_mismatch(out_dict, expected, _KERNEL_ACTUAL_RE, 'kernel_version', _message)


def eval_bios_version(out_dict, expected):
    '''Return (node_records, fail_messages) for dmidecode bios-version.'''

    def _message(node, actual):
        return f'Installed BIOS Version {actual} not matching expected version {expected} on node {node}'

    return _version_mismatch(out_dict, expected, _BIOS_ACTUAL_RE, 'bios_version', _message)


def eval_rocm_version(out_dict, expected):
    '''
    Return (node_records, fail_messages, detected_version) for amd-smi version.

    ``detected_version`` is the first ``ROCm version:`` token, or ``expected``
    when every node passed and the banner regex did not match.
    '''
    records, messages = _version_mismatch(
        out_dict,
        expected,
        _ROCM_ACTUAL_RE,
        'rocm_version',
        lambda node, actual: (
            f'Installed rocm version {actual} not matching expected version {expected} on node {node}'
        ),
    )
    detected = ''
    for output in out_dict.values():
        match = _ROCM_ACTUAL_RE.search(output)
        if match:
            detected = match.group(1)
            break
    if not detected and not messages:
        detected = str(expected)
    return records, messages, detected


def eval_firmware(out_dict, fw_dict):
    '''Return (node_records, fail_messages) for per-GPU firmware versions.'''
    records = {}
    messages = []
    for node, gpus in out_dict.items():
        items = []
        for gpu_dict in gpus:
            gpu_no = gpu_dict['gpu']
            for fw_list_dict in gpu_dict['fw_list']:
                fw_key = fw_list_dict['fw_id']
                actual = fw_list_dict['fw_version']
                expected = fw_dict[fw_key]
                if actual == expected:
                    continue
                message = (
                    f'For Firmware {fw_key} actual FW version {actual} for gpu {gpu_no} '
                    f'on node {node} is not matching expected FW version {expected}'
                )
                messages.append(message)
                items.append(_fail_item(f'{fw_key} gpu {gpu_no}', message))
        if items:
            records[str(node)] = build_node_record('fail', items, f'{len(items)} firmware mismatch(es)')
        else:
            records[str(node)] = _pass_record('firmware matched')
    return records, messages


def eval_pci_realloc(out_dict, expected):
    '''Return (node_records, fail_messages) for pci=realloc on the kernel command line.'''
    return _must_match(
        out_dict,
        f'pci=realloc={expected}',
        'pci_realloc',
        lambda node: f'PCI realloc flag not set to {expected} on node {node}',
        f'pci=realloc={expected}',
    )


def eval_iommu_pt(out_dict):
    '''Return (node_records, fail_messages) for iommu=pt.'''
    return _must_match(
        out_dict,
        'iommu=pt',
        'iommu',
        lambda node: f'IOMMU not set to pt on node {node}',
        'iommu=pt',
    )


def eval_numa_balancing(out_dict):
    '''Return (node_records, fail_messages) for kernel.numa_balancing.'''
    return _must_match(
        out_dict,
        '=0|= 0',
        'numa_balancing',
        lambda node: f'NUMA balancing not disabled on node {node}',
        'disabled',
    )


def eval_online_memory(out_dict, expected):
    '''Return (node_records, fail_messages) for lsmem total online memory.'''
    records = {}
    messages = []
    for node, output in out_dict.items():
        if re.search(f'Total online memory:\\s+{expected}', output, re.I):
            records[str(node)] = _pass_record(str(expected))
            continue
        actual = _MEMORY_ACTUAL_RE.search(output).group(1)
        message = f'Total online memory {actual} not matching expected online mem {expected} on node {node}'
        messages.append(message)
        records[str(node)] = _fail_record('online_memory', message, actual)
    return records, messages


def eval_pci_accelerators(out_dict, gpu_count):
    '''Return (node_records, fail_messages) for lspci accelerator count.'''
    expected = int(gpu_count)
    records = {}
    messages = []
    for node, output in out_dict.items():
        actual = len(_ACCEL_RE.findall(output))
        if expected == actual:
            records[str(node)] = _pass_record(str(actual))
            continue
        message = f'Expected GPU count in PCI {gpu_count} not matching actual GPU count {actual} on node {node}'
        messages.append(message)
        records[str(node)] = _fail_record('gpu_count', message, str(actual))
    return records, messages


def _ensure_link_record(records, node, speed, width):
    label = str(node)
    record = records.get(label)
    if record is None:
        record = _pass_record(f'Speed {speed}GT Width x{width}')
        records[label] = record
    return record


def _add_link_failures(record, failures):
    if not failures:
        return
    record['status'] = 'fail'
    record['items'].extend(_fail_item(name, message) for name, message in failures)
    record['items_summary'] = f"{len(record['items'])} link check(s) failed"


def _gpu_link_failures(output, speed, width, bus_no, node):
    failures = []
    if not re.search(f'Speed {speed}GT', output):
        failures.append(
            (
                f'{bus_no} speed',
                f'PCIe speed not matching for bus {bus_no} on node {node}, expected {speed}GT/s but got {output}',
            )
        )
    if not re.search(f'Width x{width}', output):
        failures.append(
            (
                f'{bus_no} width',
                f'PCIe width not matching for bus {bus_no} on node {node}, expected {width} but got {output}',
            )
        )
    if re.search('downgrade', output):
        failures.append((f'{bus_no} downgrade', f'PCIe in downgraded state for bus {bus_no} on node {node}'))
    return failures


def _nic_link_failures(output, speed, width, nic_bdf, node):
    failures = []
    if not re.search(f'Speed {speed}GT', output):
        failures.append(
            (
                f'{nic_bdf} speed',
                f'NIC PCIe speed mismatch for {nic_bdf} on {node}: expected {speed}GT/s, got {output}',
            )
        )
    if not re.search(f'Width x{width}', output):
        failures.append(
            (
                f'{nic_bdf} width',
                f'NIC PCIe width mismatch for {nic_bdf} on {node}: expected x{width}, got {output}',
            )
        )
    if re.search('downgrade', output):
        failures.append((f'{nic_bdf} downgrade', f'NIC PCIe in downgraded state for {nic_bdf} on {node}'))
    return failures


def _absorb_links(records, pci_dict, bus_for, speed, width, failures_for):
    messages = []
    for node, output in pci_dict.items():
        bus = bus_for(node)
        failures = failures_for(output, bus, node)
        _add_link_failures(_ensure_link_record(records, node, speed, width), failures)
        messages.extend(message for _name, message in failures)
    return messages


def absorb_gpu_pcie(records, pci_dict, bus_dict, card_no, speed, width):
    '''Merge one GPU's LnkSta results. Return the fail messages for this card.'''
    return _absorb_links(
        records,
        pci_dict,
        lambda node: bus_dict[node][card_no]['PCI Bus'],
        speed,
        width,
        lambda output, bus, node: _gpu_link_failures(output, speed, width, bus, node),
    )


def absorb_nic_pcie(records, pci_dict, bus_dict, card_no, speed, width):
    '''Merge one backend NIC's LnkSta results. Return the fail messages for this card.'''
    return _absorb_links(
        records,
        pci_dict,
        lambda node: bus_dict[node][card_no]['nic_bdf'],
        speed,
        width,
        lambda output, bus, node: _nic_link_failures(output, speed, width, bus, node),
    )


def eval_pci_acs(out_dict):
    '''Return (node_records, fail_messages) for PCIe ACS. ACSCtl means ACS is enabled.'''
    denied_records, denied_messages, rest = _take_denied(
        out_dict,
        'CVS_CMD_DENIED',
        'acs',
        lambda node: f'PCIe ACS check could not run lspci on node {node}',
    )
    records, messages = _must_not_match(
        rest,
        'ACSCtl:',
        'acs',
        lambda node: f'PCIe ACS not disabled on node {node}',
        'ACS disabled',
    )
    records.update(denied_records)
    messages.extend(denied_messages)
    return records, messages


def eval_dmesg_driver(out_dict):
    '''Return (node_records, fail_messages) for amdgpu fail/error lines.'''
    denied_records, denied_messages, rest = _take_denied(
        out_dict,
        'CVS_CMD_DENIED',
        'amdgpu',
        lambda node: f'Dmesg check could not run on node {node}',
    )
    records, messages = _must_not_match(
        rest,
        'fail|error',
        'amdgpu',
        lambda node: f'Dmesg has amdgpu driver errors on node {node}',
        'clean',
    )
    records.update(denied_records)
    messages.extend(denied_messages)
    return records, messages


def eval_dmesg_reset(out_dict):
    '''Return (node_records, fail_messages) for amdgpu reset/hang lines.'''
    denied_records, denied_messages, rest = _take_denied(
        out_dict,
        'CVS_CMD_DENIED',
        'amdgpu',
        lambda node: f'Dmesg check could not run on node {node}',
    )
    records, messages = _must_not_match(
        rest,
        'reset|hang',
        'amdgpu',
        lambda node: f'Dmesg has amdgpu reset/hang errors on node {node}',
        'clean',
    )
    records.update(denied_records)
    messages.extend(denied_messages)
    return records, messages
