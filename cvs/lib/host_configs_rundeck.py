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
import shlex

from cvs.lib import dmesg_noise

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
NIC_LINK = 'nic_link'
PCI_ACS = 'pci_acs'
DMESG_DRIVER = 'dmesg_driver'
DMESG_RESET = 'dmesg_reset'
IFOE_FW = 'ifoe_fw'
IFOE_KMOD = 'ifoe_kmod'
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
    NIC_LINK,
    PCI_ACS,
    DMESG_DRIVER,
    DMESG_RESET,
    IFOE_FW,
    IFOE_KMOD,
)

_OS_ACTUAL_RE = re.compile(r'VERSION="(([0-9.\-_A-Z]+)\s+)', re.I)
_KERNEL_ACTUAL_RE = re.compile(r'([0-9.\-_]+generic)', re.I)
_BIOS_ACTUAL_RE = re.compile(r'([a-z0-9_.\-]+)', re.I)
_ROCM_ACTUAL_RE = re.compile(r'ROCm version:\s+([0-9.]+)', re.I)
_MEMORY_ACTUAL_RE = re.compile(r'Total online memory:\s+([0-9.A-Za-z]+)')
_ACCEL_RE = re.compile(r'accelerators:\s+Advanced', re.I)
DEFAULT_NIC_LINK_SPEED = 400000
# The kernel reports SPEED_UNKNOWN as -1 or unsigned 0xFFFFFFFF when it cannot determine link speed.
_SPEED_UNKNOWN = (-1, 4294967295)
_NIC_LINK_RE = re.compile(r'^\s*(\S+)[ \t]+speed=(\S*)[ \t]+operstate=(\S*)[ \t]+flags=(\S*)\s*$')
_IFF_UP = 0x1

# Emitted by the host_configs_cvs IFoE commands when the ifoe kernel module is not
# loaded, so a non-MI4XX node reports NA instead of failing.
CVS_IFOE_ABSENT = 'CVS_IFOE_ABSENT'
# Emitted when the module is loaded (an IFoE node) but the reader binary (afmctl /
# modinfo) could not be resolved. This is a read failure, not an absent module, so
# the check fails as unreadable rather than silently recording NA.
CVS_IFOE_READER_MISSING = 'CVS_IFOE_READER_MISSING'
# ``afmctl show device`` prints one ``BDF : X`` header then one ``Firmware version``
# per device block. The BDF count is the device count: every reported device must
# carry a firmware version, so a block missing its version fails rather than passing
# on the other blocks' values alone.
_IFOE_BDF_RE = re.compile(r'^\s*BDF\s*:\s*\S+', re.I | re.M)
# Line-anchored so a banner or tool-version line that merely mentions the words
# cannot be counted as a device and trip the divergence check.
_IFOE_FW_RE = re.compile(r'^\s*Firmware\s+version\s*:\s*(\S+)', re.I | re.M)
# ``modinfo ifoe`` prints a ``version:`` line (dotted/dash build string).
_IFOE_KMOD_RE = re.compile(r'^version:\s*(\S+)', re.I | re.M)
# A well-formed firmware version is dotted digits (e.g. 0.14.15); reject garbled data.
_IFOE_FW_FORMAT_RE = re.compile(r'^\d+(?:\.\d+)+$')
# A well-formed kmod version token: alphanumerics with dots/dashes/underscores, no spaces.
_IFOE_KMOD_FORMAT_RE = re.compile(r'^[0-9A-Za-z][0-9A-Za-z._-]*$')


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


def config_error_records(nodes, item_name, message):
    '''Fail every node with the same config error, so the deck shows why the check did not run.'''
    return {str(node): _fail_record(item_name, message, 'config error') for node in nodes}


def _version_mismatch(out_dict, expected, actual_re, item_name, message_for):
    '''Fail a node when ``expected`` is absent. A regex miss is a read failure, not an exception.'''
    records = {}
    messages = []
    for node, output in out_dict.items():
        if re.search(f'{expected}', output, re.I):
            records[str(node)] = _pass_record(str(expected))
            continue
        match = actual_re.search(output or '')
        if match is None:
            message = f'{item_name} could not be read on node {node}'
            messages.append(message)
            records[str(node)] = _fail_record(item_name, message, 'unavailable')
            continue
        actual = match.group(1)
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


def _must_not_match_with_denial(out_dict, token, pattern, item_name, denied_message_for, message_for, pass_summary):
    '''Apply _must_not_match, and fail nodes whose output contains ``token`` first.

    Denial is recorded after the pattern results so a later change to merge order
    lives in this one helper.
    '''
    denied_records, denied_messages, rest = _take_denied(out_dict, token, item_name, denied_message_for)
    records, messages = _must_not_match(rest, pattern, item_name, message_for, pass_summary)
    records.update(denied_records)
    messages.extend(denied_messages)
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
    '''Return (node_records, fail_messages) for the BIOS version read.'''

    def _message(node, actual):
        return f'Installed BIOS Version {actual} not matching expected version {expected} on node {node}'

    denied_records, denied_messages, rest = _take_denied(
        out_dict,
        'CVS_CMD_DENIED',
        'bios_version',
        lambda node: f'BIOS version could not be read on node {node}',
    )
    records, messages = _version_mismatch(rest, expected, _BIOS_ACTUAL_RE, 'bios_version', _message)
    records.update(denied_records)
    messages.extend(denied_messages)
    return records, messages


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


def _na_record(summary='not applicable'):
    return build_node_record('na', [], summary)


def _eval_ifoe_version(out_dict, expected, detect_re, format_re, item_name, label, count_re=None):
    '''Shared IFoE firmware/kmod evaluation.

    Per node:
      - absent sentinel in output -> ``na`` record, no fail message.
      - denial token in output -> fail (the privileged read was refused).
      - reader-missing marker in output -> fail (module loaded but reader not found).
      - ``count_re`` set and more devices than detected versions -> fail (a reported
        device is missing its version).
      - ``expected`` set -> every detected value must equal it, else fail.
      - ``expected`` blank -> inventory mode: pass only when every detected value is a
        well-formed version (``format_re``); a missing or garbled value fails.

    ``afmctl show device`` reports one version per device, so every match on a node
    is checked and a divergence between devices fails rather than passing on the
    first value alone. When ``count_re`` matches the per-device header, a device whose
    version line is absent fails rather than passing on the other devices' values.
    '''
    records = {}
    messages = []
    want = str(expected).strip() if expected not in (None, '') else ''
    for node, output in out_dict.items():
        text = str(output or '')
        if CVS_IFOE_ABSENT in text:
            records[str(node)] = _na_record()
            continue
        if 'CVS_CMD_DENIED' in text:
            message = f'{label} could not be read on node {node}'
            messages.append(message)
            records[str(node)] = _fail_record(item_name, message, 'denied')
            continue
        if CVS_IFOE_READER_MISSING in text:
            message = f'{label} reader not found on node {node}'
            messages.append(message)
            records[str(node)] = _fail_record(item_name, message, 'unavailable')
            continue
        detected = detect_re.findall(text)
        if not detected:
            message = f'{label} could not be read on node {node}'
            messages.append(message)
            records[str(node)] = _fail_record(item_name, message, 'unavailable')
            continue
        if count_re is not None:
            device_count = len(count_re.findall(text))
            if device_count > len(detected):
                message = (
                    f'{label} missing for {device_count - len(detected)} of {device_count} device(s) on node {node}'
                )
                messages.append(message)
                records[str(node)] = _fail_record(item_name, message, 'incomplete')
                continue
        unique = list(dict.fromkeys(detected))
        if len(unique) > 1:
            joined = ', '.join(unique)
            message = f'{label} differs across devices on node {node}: {joined}'
            messages.append(message)
            records[str(node)] = _fail_record(item_name, message, 'divergent')
            continue
        value = unique[0]
        if want:
            if value == want:
                records[str(node)] = _pass_record(value)
            else:
                message = f'Installed {label} {value} not matching expected version {want} on node {node}'
                messages.append(message)
                records[str(node)] = _fail_record(item_name, message, value)
        elif format_re.match(value):
            records[str(node)] = _pass_record(value)
        else:
            message = f'{label} {value!r} on node {node} is not a well-formed version'
            messages.append(message)
            records[str(node)] = _fail_record(item_name, message, 'malformed')
    return records, messages


def eval_ifoe_firmware(out_dict, expected):
    '''Return (node_records, fail_messages) for the IFoE/AFM device firmware version.'''
    return _eval_ifoe_version(
        out_dict,
        expected,
        _IFOE_FW_RE,
        _IFOE_FW_FORMAT_RE,
        'ifoe_fw_version',
        'IFoE firmware version',
        count_re=_IFOE_BDF_RE,
    )


def eval_ifoe_kmod(out_dict, expected):
    '''Return (node_records, fail_messages) for the ifoe kernel module version.'''
    return _eval_ifoe_version(
        out_dict, expected, _IFOE_KMOD_RE, _IFOE_KMOD_FORMAT_RE, 'ifoe_kmod_version', 'IFoE kmod version'
    )


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


def pcie_card_list(out_dict):
    '''Card ids from the first node that actually reported a bus map.'''
    for node_map in out_dict.values():
        if isinstance(node_map, dict) and node_map:
            return list(node_map.keys())
    return []


def partition_pcie_cards(out_dict, card_no, field):
    '''Split nodes into those with ``field`` for ``card_no`` and those without.

    A node whose rocm-smi payload is missing or not a card dict must not be indexed
    with another node's card id. Callers still emit one command per node so
    exec_cmd_list stays aligned with the host list.
    '''
    present = {}
    missing = []
    for node, node_map in out_dict.items():
        card = node_map.get(card_no) if isinstance(node_map, dict) else None
        bus = card.get(field) if isinstance(card, dict) else None
        if bus:
            present[node] = bus
        else:
            missing.append(node)
    return present, missing


def note_missing_link(records, node, message):
    '''Record one missing-bus failure. Return False when this message is already stored.'''
    label = str(node)
    item = {'name': 'bus', 'status': 'fail', 'message': message}
    record = records.get(label)
    if record is None:
        records[label] = build_node_record('fail', [item], 'missing')
        return True
    if any(existing.get('message') == message for existing in record.get('items') or []):
        return False
    record['status'] = 'fail'
    record.setdefault('items', []).append(item)
    record['items_summary'] = 'missing'
    return True


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


def parse_nic_link_speed_setting(value):
    '''Return the configured expected link speed in Mb/s as a positive integer.'''
    if value is None:
        return DEFAULT_NIC_LINK_SPEED
    try:
        speed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f'nic_link_speed must be a positive integer in Mb/s (e.g. 400000), got {value!r}') from exc
    if speed <= 0:
        raise ValueError(f'nic_link_speed must be a positive integer in Mb/s (e.g. 400000), got {value!r}')
    return speed


def nic_link_speed_cmd(interfaces):
    '''Build a command printing each interface's sysfs link speed, operstate and flags.'''
    names = ' '.join(shlex.quote(name) for name in sorted(set(interfaces)))
    if not names:
        return 'true'
    return (
        f'for i in {names}; do d="/sys/class/net/$i"; '
        'echo "$i speed=$(cat "$d/speed" 2>/dev/null) operstate=$(cat "$d/operstate" 2>/dev/null) '
        'flags=$(cat "$d/flags" 2>/dev/null)"; done'
    )


def parse_nic_links(output):
    '''Map interface names to their sysfs speed, operstate and flags. Unreadable fields are None.'''
    links = {}
    for line in str(output).splitlines():
        match = _NIC_LINK_RE.match(line)
        if not match:
            continue
        speed, operstate, flags = match.group(2, 3, 4)
        links[match.group(1)] = {
            'speed': int(speed) if re.fullmatch(r'-?\d+', speed) else None,
            'operstate': operstate or None,
            'flags': int(flags, 16) if re.fullmatch(r'0x[0-9a-f]+', flags, re.I) else None,
        }
    return links


def _nic_link_problem(iface, link, expected, node):
    '''Return the failure message for one interface, or None when it is at the expected speed.'''
    want = f'expected {expected} Mb/s on node {node}'
    if link is None:
        return f'Backend NIC {iface} link speed unreadable from /sys/class/net/{iface}/speed, {want}'
    speed, operstate, flags = link['speed'], link['operstate'], link['flags']
    state = f' (operstate {operstate})' if operstate else ''
    if speed is None:
        if operstate is None and flags is None:
            return f'Backend NIC {iface} not found in /sys/class/net, {want}'
        # operstate reads "down" both when admin-down and when up without carrier; only IFF_UP tells them apart.
        if flags is not None and not flags & _IFF_UP:
            return f'Backend NIC {iface} is administratively down, link speed unavailable, {want}'
        return f'Backend NIC {iface} link speed unreadable from /sys/class/net/{iface}/speed{state}, {want}'
    if speed in _SPEED_UNKNOWN:
        return f'Backend NIC {iface} link speed unknown ({speed}), link may be down{state}, {want}'
    if speed != expected:
        return f'Backend NIC {iface} link speed {speed} Mb/s not matching expected {expected} Mb/s on node {node}'
    return None


def eval_nic_link_speed(out_dict, nics_by_node, expected, compare_counts=False):
    '''
    Return (node_records, fail_messages) for backend NIC network link speed.

    With compare_counts, a node with fewer NICs than the most found on any node
    fails. Auto-detection only sees NICs that have a netdev and an RDMA device,
    so a NIC whose driver failed to bind would otherwise go unchecked.
    '''
    records = {}
    messages = []
    nodes = list(out_dict) + [node for node in nics_by_node if node not in out_dict]
    peak = max((len(set(nics or [])) for nics in nics_by_node.values()), default=0) if compare_counts else 0
    for node in nodes:
        interfaces = list(dict.fromkeys(nics_by_node.get(node) or []))
        if not interfaces:
            message = f'No backend NICs found on node {node} to check link speed'
            records[str(node)] = _fail_record('backend_nics', message, 'no NICs')
            messages.append(message)
            continue

        count = len(interfaces)
        items = []
        summary = []
        if count < peak:
            message = (
                f'Only {count} backend NIC(s) detected on node {node}, fewer than the {peak} detected on '
                'another node; a backend NIC may be missing its netdev or RDMA device'
            )
            items.append(_fail_item('backend_nics', message))
            messages.append(message)
            summary.append(f'{count} of {peak} NIC(s) detected')

        links = parse_nic_links(out_dict.get(node, ''))
        slow = 0
        for iface in interfaces:
            message = _nic_link_problem(iface, links.get(iface), expected, node)
            if message is None:
                continue
            slow += 1
            items.append(_fail_item(f'{iface} speed', message))
            messages.append(message)
        if slow:
            summary.append(f'{slow} of {count} NIC(s) not at {expected} Mb/s')

        records[str(node)] = (
            build_node_record('fail', items, '; '.join(summary))
            if items
            else _pass_record(f'{count} x {expected} Mb/s')
        )
    return records, messages


def eval_pci_acs(out_dict):
    '''Return (node_records, fail_messages) for PCIe ACS. ACSCtl means ACS is enabled.'''
    return _must_not_match_with_denial(
        out_dict,
        'CVS_CMD_DENIED',
        'ACSCtl:',
        'acs',
        lambda node: f'PCIe ACS check could not run lspci on node {node}',
        lambda node: f'PCIe ACS not disabled on node {node}',
        'ACS disabled',
    )


def _actionable_amdgpu_blob(event):
    '''Return event text for an amdgpu line or an SW_DRIVER event.

    Same acceptance rule as ``verify_driver_errors``: the event line names
    amdgpu, or node-scraper tagged the event ``SW_DRIVER``.
    '''
    # Imported here so the rest of this module does not load the analyzer.
    from cvs.lib.node_scraper_adapter import event_match_lines, event_match_text

    if dmesg_noise.dmesg_line_is_benign(event_match_text(event)):
        return ''
    lines = event_match_lines([event])
    blob = lines[0] if lines else ''
    if 'amdgpu' not in blob.lower() and event.get('category') != 'SW_DRIVER':
        return ''
    return blob


def _hit_record(hits, message):
    if not hits:
        return _pass_record('clean'), None
    return _fail_record('amdgpu', message, 'seen'), message


def eval_dmesg_node_events(events_by_node, denied_nodes=None):
    '''Return driver and reset verdicts from one pass over node-scraper events.

    Returns ``(driver_records, driver_messages, reset_records, reset_messages)``.
    Each event's line is built and benign-checked once, then used for both checks.
    '''
    driver_records = {}
    driver_messages = []
    reset_records = {}
    reset_messages = []
    for node, events in events_by_node.items():
        driver_hits = []
        reset_hits = []
        for event in events or []:
            blob = _actionable_amdgpu_blob(event)
            if not blob:
                continue
            # SW_DRIVER is already a driver fault; it does not have to say "error".
            if event.get('category') == 'SW_DRIVER' or dmesg_noise.line_indicates_driver_error(blob):
                driver_hits.append(blob)
            if re.search(r'reset|hang|traceback', blob, re.I):
                reset_hits.append(blob)
        driver_message = f'Dmesg has amdgpu driver errors on node {node}'
        reset_message = f'Dmesg has amdgpu reset/hang errors on node {node}'
        driver_record, driver_fail = _hit_record(driver_hits, driver_message)
        reset_record, reset_fail = _hit_record(reset_hits, reset_message)
        driver_records[str(node)] = driver_record
        reset_records[str(node)] = reset_record
        if driver_fail:
            driver_messages.append(driver_fail)
        if reset_fail:
            reset_messages.append(reset_fail)
    for node in denied_nodes or []:
        message = f'Dmesg check could not run on node {node}'
        driver_records[str(node)] = _fail_record('amdgpu', message, 'denied')
        reset_records[str(node)] = _fail_record('amdgpu', message, 'denied')
        driver_messages.append(message)
        reset_messages.append(message)
    return driver_records, driver_messages, reset_records, reset_messages


def eval_dmesg_driver_events(events_by_node):
    '''Return (node_records, fail_messages) from node-scraper dmesg events.'''
    records, messages, _, _ = eval_dmesg_node_events(events_by_node)
    return records, messages


def eval_dmesg_reset_events(events_by_node):
    '''Return (node_records, fail_messages) for reset/hang node-scraper events.'''
    _, _, records, messages = eval_dmesg_node_events(events_by_node)
    return records, messages
