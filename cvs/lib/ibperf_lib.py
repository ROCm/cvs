'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import logging
import math
import re
import shlex
import time
import xlsxwriter
from collections import Counter
from xlsxwriter.utility import xl_col_to_name

from cvs.lib.preflight.scaleup_fabric import parse_afmctl_device_json
from cvs.lib.utils_lib import *
from cvs.lib.rdma_gid_lib import (
    DEFAULT_GID_TYPE,
    GID_OK,
    build_gid_probe_cmd,
    build_gid_table_cmd,
    check_gid_entry,
    is_auto_gid_index,
    normalize_gid_index,
    parse_gid_probe_output,
    parse_gid_table_output,
    select_common_gid_index,
)
from cvs.lib.utils.polling import poll_until

log = logging.getLogger(__name__)

IBPERF_PAIRING_MODES = ('sequential', 'intra_vpod', 'inter_vpod')
IBPERF_VPOD_SOURCES = ('afm', 'cluster_file')
_AFM_ROOT_ERROR_RE = re.compile(r'requires root privileges', re.I)
_AFM_SUDO_ERROR_MARKER = 'CVS_ERROR: ibperf.vpod_source afm needs passwordless sudo for afmctl'

DMABUF_SUPPORTED = 'supported'
DMABUF_UNSUPPORTED = 'unsupported'
DMABUF_PROBE_FAILED = 'probe_failed'
PERFTEST_DMABUF_BUILD_HINT = 'rebuild perftest with ROCm DMA-BUF support (configure --enable-rocm --enable-rocm-dmabuf)'

# Cached one-time ROCm path probe results for the current process.
_rocm_path_cache = {}
_GPU_VISIBILITY_ENV_VARS = ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
_GPU_ENV_TAG = '__CVS_GPU_ENV__'
# Job schedulers set these to a subset or to values ROCm can't parse. perftest runs with them cleared,
# and the HIP-ordinal probe clears them too so KFD order matches the ordinals perftest sees.
_GPU_ENV_UNSET = 'env ' + ' '.join(f'-u {var}' for var in _GPU_VISIBILITY_ENV_VARS)
_HIP_DEVICE_PROBE_SCRIPT = (
    'grep -HE "^(simd_count|location_id|domain) " /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null; '
    f"printf '{_GPU_ENV_TAG} %s=%s\\n' " + ' '.join(f'{var} "${{{var}-}}"' for var in _GPU_VISIBILITY_ENV_VARS)
)
_HIP_DEVICE_PROBE_CMD = f'{_GPU_ENV_UNSET} sh -c {shlex.quote(_HIP_DEVICE_PROBE_SCRIPT)}'

DEFAULT_BW_TESTS = ['ib_write_bw', 'ib_read_bw', 'ib_send_bw']
DEFAULT_LAT_TESTS = ['ib_write_lat', 'ib_send_lat']
SUPPORTED_BW_TESTS = ('ib_write_bw', 'ib_read_bw', 'ib_send_bw')
SUPPORTED_LAT_TESTS = ('ib_write_lat', 'ib_read_lat', 'ib_send_lat')

# Added to the BW ``duration``: covers connection setup, the client's launch delay, and perftest's warm-up margins.
PERFTEST_EXIT_SLACK_S = 60
# Latency tests have no ``-D`` duration, so this bounds the whole run.
PERFTEST_LAT_EXIT_TIMEOUT_S = 60
PERFTEST_EXIT_POLL_S = 5
PERFTEST_RESULT_POLL_S = 10
PERFTEST_BW_RESULT_TIMEOUT_S = 80
PERFTEST_LAT_RESULT_TIMEOUT_S = 20
# The command file needs bash because a node's agent can run commands under dash, which has no ``source``.
PERFTEST_LAUNCH_CMD = f'{_GPU_ENV_UNSET} bash /tmp/ib_cmds_file.txt'


def build_ibperf_node_pairs(node_list, pairing_mode='sequential', vpod_map=None):
    """Return ordered (server, client) pairs and unpaired nodes in input order."""
    if pairing_mode not in IBPERF_PAIRING_MODES:
        raise ValueError(f'Invalid ibperf pairing_mode {pairing_mode!r}; allowed values: {IBPERF_PAIRING_MODES}')
    if pairing_mode == 'sequential':
        return (
            [(node_list[i], node_list[i + 1]) for i in range(0, len(node_list) - 1, 2)],
            list(node_list[len(node_list) // 2 * 2 :]),
        )
    if vpod_map is None:
        raise ValueError(f'vpod_map is required for ibperf pairing_mode {pairing_mode}')

    groups = {}
    for node in node_list:
        key = vpod_map.get(node)
        if key is not None:
            groups.setdefault(key, []).append(node)

    pairs = []
    if pairing_mode == 'intra_vpod':
        for group in groups.values():
            pairs.extend((group[i], group[i + 1]) for i in range(0, len(group) - 1, 2))
    else:
        remaining = [list(group) for group in groups.values()]
        while sum(bool(group) for group in remaining) >= 2:
            first, second = sorted(
                (index for index, group in enumerate(remaining) if group),
                key=lambda index: (-len(remaining[index]), index),
            )[:2]
            server_group, client_group = sorted((first, second))
            pairs.append((remaining[server_group].pop(0), remaining[client_group].pop(0)))

    paired_nodes = {node for pair in pairs for node in pair}
    return pairs, [node for node in node_list if node not in paired_nodes]


def get_cluster_file_vpod_map(cluster_dict):
    """Read node vPOD labels from a cluster configuration."""
    vpod_map = {}
    errors = []
    for node, details in cluster_dict['node_dict'].items():
        value = details.get('vpod_id')
        label = str(value).strip() if value is not None else ''
        if label:
            vpod_map[node] = label
        else:
            errors.append(f'{node}: node_dict entry has no vpod_id (required when ibperf.vpod_source is cluster_file)')
    return vpod_map, errors


def _afm_pod_id(section):
    if isinstance(section, dict) and section.get('id') is not None:
        return str(section['id']).strip()
    return ''


def extract_afm_pod_keys(output):
    """Read keys unique across racks because virtual pod IDs are scoped by physical pod IDs."""
    try:
        document = json.loads(output)
    except (TypeError, ValueError):
        return []
    records = document.get('device', document.get('devices')) if isinstance(document, dict) else document
    if not isinstance(records, list):
        return []
    keys = []
    for record in records:
        if not isinstance(record, dict):
            continue
        spec = record['spec'] if isinstance(record.get('spec'), dict) else record
        fabric = spec.get('scaleup_fabric')
        if not isinstance(fabric, dict):
            fabric = {}
        physical = _afm_pod_id(fabric.get('physical_pod'))
        virtual = _afm_pod_id(fabric.get('virtual_pod'))
        keys.append((physical, virtual) if physical and virtual else None)
    return keys


def parse_afm_vpod_membership(out_dict):
    """Parse per-node AFM output into vPOD keys and discovery errors."""
    pod_keyed = {}
    fallback = {}
    accels = {}
    errors = []
    for node, output in out_dict.items():
        if _AFM_SUDO_ERROR_MARKER in (output or ''):
            errors.append(f'{node}: ibperf.vpod_source afm needs passwordless sudo for afmctl')
            continue
        root_match = _AFM_ROOT_ERROR_RE.search(output or '')
        if root_match:
            line = next(line.strip() for line in output.splitlines() if root_match.group(0) in line)
            errors.append(
                f'{node}: afmctl needs root ({line}); CVS runs "sudo -n afmctl show device --json", '
                'so grant the SSH user passwordless sudo for afmctl'
            )
            continue
        devices, parse_errors = parse_afmctl_device_json(output)
        if parse_errors or not devices:
            errors.append(f'{node}: AFM vPOD discovery failed: {"; ".join(parse_errors) or "no devices"}')
            continue
        accels[node] = {d['accelerator_id'] for d in devices if d.get('accelerator_id') is not None}
        keys = extract_afm_pod_keys(output)
        present = set(keys) - {None}
        if len(present) > 1:
            errors.append(f'{node}: AFM devices report multiple vPODs: {sorted(present)}')
        elif present and None in keys:
            errors.append(f'{node}: AFM reports scaleup_fabric physical_pod/virtual_pod IDs on only some devices')
        elif present:
            pod_keyed[node] = next(iter(present))
        else:
            sets = {tuple(sorted(d.get('vpod_accelerators') or [])) for d in devices} - {()}
            if not sets:
                errors.append(f'{node}: AFM reports no vPOD membership')
            elif len(sets) > 1:
                errors.append(f'{node}: AFM devices report multiple vPOD memberships: {sorted(sets)}')
            else:
                fallback[node] = next(iter(sets))

    if pod_keyed:
        for node in fallback:
            errors.append(
                f'{node}: AFM does not report scaleup_fabric physical_pod/virtual_pod IDs while other nodes do; '
                'cannot compare vPOD membership across AFM versions'
            )
        vpod_map = dict(pod_keyed)
        log.info('AFM vPOD key: scaleup_fabric (physical_pod.id, virtual_pod.id) for %d node(s)', len(pod_keyed))
    elif fallback:
        log.warning(
            'AFM on %s reports no scaleup_fabric physical_pod/virtual_pod IDs; keying vPODs by '
            'vpod_accelerators, which is only unique within one physical pod',
            sorted(fallback),
        )
        vpod_map = dict(fallback)
        members_by_key = {}
        for node, key in vpod_map.items():
            for other in members_by_key.get(key, []):
                if accels[node] & accels[other]:
                    raise ValueError(
                        f'AFM vPOD {list(key)} is reported by nodes with overlapping accelerator IDs ({other}, {node}); '
                        'AFM accelerator IDs are only unique within one scale-up domain. Set ibperf.vpod_source to '
                        '"cluster_file" and label each node_dict entry with vpod_id.'
                    )
            members_by_key.setdefault(key, []).append(node)
    else:
        vpod_map = {}
    return vpod_map, errors


def _build_afmctl_device_command(afmctl_path):
    """Check sudo per host and resolve relative afmctl names outside sudo's secure_path."""
    afmctl_path = str(afmctl_path or 'afmctl')
    arguments = 'show device --json'
    executable = shlex.quote(afmctl_path)
    sudo_error = shlex.quote(_AFM_SUDO_ERROR_MARKER)
    sudo_check = f'if ! sudo -n true >/dev/null 2>&1; then printf \'%s\\n\' {sudo_error} >&2; exit 1; fi; '
    if afmctl_path.startswith('/'):
        command = f'{sudo_check}sudo -n {executable} {arguments}'
    else:
        missing_command = shlex.quote(f'Unable to find {afmctl_path} in PATH')
        command = (
            f'{sudo_check}_cvs_afmctl="$(command -v {executable})" || '
            f'{{ printf \'%s\\n\' {missing_command} >&2; exit 127; }}; '
            'case "$_cvs_afmctl" in /*) ;; *) '
            "printf '%s\\n' 'Resolved afmctl path is not absolute' >&2; exit 127 ;; esac; "
            'sudo -n "$_cvs_afmctl" show device --json'
        )
    return f'{{ {command}; }} 2>&1'


def discover_afm_vpod_map(phdl, afmctl_path='afmctl'):
    """Query AFM on every node in a parallel handle and parse vPOD membership."""
    cmd = _build_afmctl_device_command(afmctl_path)
    return parse_afm_vpod_membership(phdl.exec(cmd, print_console=False))


def resolve_ibperf_node_pairing(node_list, config_dict, cluster_dict, phdl_factory=None):
    """Resolve pairing configuration and return pairs, leftovers, and discovery errors."""
    mode = str(config_dict.get('pairing_mode', 'sequential')).strip().lower()
    source = str(config_dict.get('vpod_source', 'afm')).strip().lower()
    if mode not in IBPERF_PAIRING_MODES:
        raise ValueError(f'Invalid ibperf pairing_mode {mode!r}; allowed values: {IBPERF_PAIRING_MODES}')
    if source not in IBPERF_VPOD_SOURCES:
        raise ValueError(f'Invalid ibperf vpod_source {source!r}; allowed values: {IBPERF_VPOD_SOURCES}')

    vpod_map = None
    errors = []
    if mode != 'sequential':
        if source == 'cluster_file':
            vpod_map, errors = get_cluster_file_vpod_map(cluster_dict)
        else:
            if phdl_factory is None:
                raise ValueError('phdl_factory is required for ibperf.vpod_source afm')
            hdl = phdl_factory(node_list)
            try:
                vpod_map, errors = discover_afm_vpod_map(hdl, config_dict.get('afmctl_path', 'afmctl'))
            finally:
                hdl.destroy_clients()
        log.info('ibperf vPOD map (vpod_source=%s): %s', source, vpod_map)

    pairs, unpaired = build_ibperf_node_pairs(node_list, mode, vpod_map)
    for err in errors:
        log.error('%s', err)
    if not pairs:
        reason = {
            'inter_vpod': 'inter_vpod needs nodes in at least two vPODs',
            'intra_vpod': 'intra_vpod needs at least two nodes in one vPOD',
            'sequential': 'at least 2 nodes are required',
        }[mode]
        raise ValueError(
            f'No ibperf node pairs could be formed (pairing_mode={mode}, nodes={len(node_list)}); '
            + ('; '.join(errors) if errors else reason)
        )
    log.info('ibperf pairing_mode=%s: %d pair(s): %s', mode, len(pairs), pairs)
    if unpaired:
        if mode == 'sequential':
            log.info(
                'Odd number of nodes (%d); excluding last node %s to form server/client pairs',
                len(node_list),
                unpaired[-1],
            )
        else:
            log.warning('Nodes left unpaired by %s pairing: %s', mode, unpaired)
    return {'pairing_mode': mode, 'vpod_source': source, 'pairs': pairs, 'unpaired': unpaired, 'errors': errors}


def _assign_pair_roles(node_pairs, nodes):
    """Return active node roles in handle order and nodes lacking a partner."""
    assignments = {}
    for server, client in node_pairs:
        if server in assignments or client in assignments or server == client:
            raise ValueError(f'A node appears in more than one ibperf pair: {(server, client)}')
        assignments[server] = None
        assignments[client] = server
    present = set(nodes)
    active = {
        node for server, client in node_pairs if server in present and client in present for node in (server, client)
    }
    roles = {node: assignments[node] for node in nodes if node in active}
    orphans = [node for node in nodes if node not in active]
    return roles, orphans


def get_node_gpu_cards(gpu_nic_dict, node):
    """Return card names in GPU index order, including card2 before card10."""
    return sorted(gpu_nic_dict.get(node, {}), key=_card_index)


def _prepare_pair_roles(node_pairs, nodes, gpu_nic_dict, gpu_numa_dict, hip_dev_dict, test_name):
    """Keep only complete pairs with the same discovered GPU card list and command inputs."""
    roles, orphans = _assign_pair_roles(node_pairs, nodes)
    if orphans:
        fail_test(f'ibperf partner node unreachable or unpaired for {orphans}; excluding them from {test_name}')
    pair_cards = {}
    for client, server in roles.items():
        if server is None:
            continue
        server_cards = get_node_gpu_cards(gpu_nic_dict, server)
        client_cards = get_node_gpu_cards(gpu_nic_dict, client)
        if not server_cards or server_cards != client_cards:
            fail_test(
                f'ibperf pair ({server}, {client}) has different or missing GPU cards '
                f'({server}: {server_cards}, {client}: {client_cards}); excluding the pair from {test_name}'
            )
            continue
        missing = []
        for node in (server, client):
            for card in server_cards:
                if not gpu_nic_dict[node][card].get('rdma_dev'):
                    missing.append(f'{node}/{card} RDMA device')
                if not gpu_numa_dict.get(node, {}).get(card, {}).get('local_cpulist'):
                    missing.append(f'{node}/{card} NUMA CPU list')
                if hip_dev_dict.get(node, {}).get(card) is None:
                    missing.append(f'{node}/{card} HIP device')
        if missing:
            fail_test(
                f'ibperf pair ({server}, {client}) has incomplete GPU mapping: {missing}; '
                f'excluding the pair from {test_name}'
            )
            continue
        pair_cards[(server, client)] = tuple(server_cards)
    common = Counter(pair_cards.values()).most_common(1)[0][0] if pair_cards else ()
    for (server, client), pair in pair_cards.items():
        if pair != common:
            fail_test(
                f'ibperf pair ({server}, {client}) has GPU cards {list(pair)}, but other pairs use '
                f'{list(common)}; excluding the pair from {test_name}'
            )
    kept = {node for pair, cards in pair_cards.items() if cards == common for node in pair}
    roles = {node: server for node, server in roles.items() if node in kept}
    excluded = [node for node in nodes if node not in roles]
    return roles, list(common), excluded


def _configured_tests(config, key, defaults, supported):
    """Return configured perftest binaries, or defaults when the key is absent.

    Raise ValueError for a non-list value, unsupported test, or duplicate.
    """
    if key not in config:
        log.info('%s not set in ibperf config; using defaults %s', key, defaults)
        return list(defaults)

    value = config[key]
    if not isinstance(value, list):
        raise ValueError(f'{key} must be a list of test names, got {value!r}')

    unknown = [test for test in value if not isinstance(test, str) or test not in supported]
    if unknown:
        raise ValueError(f'{key} has unsupported tests {unknown}; supported: {list(supported)}')

    duplicates = sorted({test for test in value if value.count(test) > 1})
    if duplicates:
        raise ValueError(f'{key} lists tests more than once: {duplicates}; supported: {list(supported)}')

    return list(value)


def configured_bw_tests(config):
    """Return the bandwidth tests requested by the ibperf config dictionary."""
    return _configured_tests(config, 'ib_bw_test_list', DEFAULT_BW_TESTS, SUPPORTED_BW_TESTS)


def configured_lat_tests(config):
    """Return the latency tests requested by the ibperf config dictionary."""
    return _configured_tests(config, 'ib_lat_test_list', DEFAULT_LAT_TESTS, SUPPORTED_LAT_TESTS)


def _log_bw_summary(msg_size, res_dict, instance_no=None):
    if not res_dict:
        return
    bws = []
    for node in sorted(res_dict):
        bw = res_dict[node]['bw']
        pps = res_dict[node]['pps']
        bws.append(float(bw))
        log.debug('Node %s gpu=%s BW=%s MPPS=%s', node, instance_no, bw, pps)
    if instance_no is not None and bws:
        log.debug(
            'BW msg_size=%s gpu=%s: avg=%.2f Gbps across %d nodes',
            msg_size,
            instance_no,
            sum(bws) / len(bws),
            len(bws),
        )


def _log_lat_summary(msg_size, res_dict, instance_no=None):
    if not res_dict:
        return
    avgs = []
    parts = []
    for node in sorted(res_dict):
        lat = res_dict[node]['t_avg']
        avgs.append(float(lat))
        parts.append(f'{node}={lat}')
        log.debug('Node %s msg_size %s: avg latency=%s us', node, msg_size, lat)
    if not avgs:
        return
    inst_label = f' gpu={instance_no}' if instance_no is not None else ''
    log.debug(
        'Latency msg_size=%s%s: avg=%.2f us [%s]',
        msg_size,
        inst_label,
        sum(avgs) / len(avgs),
        ', '.join(parts),
    )


def _log_bw_run_done(bw_test, msg_size, qp_count, result_dict):
    all_bws = []
    for node in result_dict:
        for inst in result_dict.get(node, {}):
            all_bws.append(float(result_dict[node][inst]['bw']))
    if all_bws:
        log.info(
            '%s: msg_size=%s qp=%s done, BW avg=%.2f Gbps (%d GPUs)',
            bw_test,
            msg_size,
            qp_count,
            sum(all_bws) / len(all_bws),
            len(all_bws),
        )


def _log_lat_run_done(lat_test, msg_size, result_dict):
    all_lats = []
    for node in result_dict:
        for inst in result_dict.get(node, {}):
            all_lats.append(float(result_dict[node][inst]['t_avg']))
    if all_lats:
        log.info(
            '%s: msg_size=%s done, latency avg=%.2f us (%d GPUs)',
            lat_test,
            msg_size,
            sum(all_lats) / len(all_lats),
            len(all_lats),
        )


def detect_rocm_path(phdl, config_rocm_path):
    """
    Detect the ROCm installation path, supporting both old (/opt/rocm) and new
    (/opt/rocm/core-X.Y) layouts.

    Args:
        phdl: Parallel SSH handle
        config_rocm_path (str): Configured ROCm path from config file
                                (empty string or '<changeme>' for auto-detect)

    Returns:
        str: Detected ROCm path
    """
    cache_key = config_rocm_path if config_rocm_path else '<auto>'
    if cache_key in _rocm_path_cache:
        log.debug('Using cached ROCm path for %s: %s', cache_key, _rocm_path_cache[cache_key])
        return _rocm_path_cache[cache_key]

    if config_rocm_path and config_rocm_path != '<changeme>':
        out_dict = phdl.exec(
            f'test -d {config_rocm_path}/lib && ls {config_rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1',
            print_console=False,
        )
        for node, output in out_dict.items():
            if output.strip() and 'libamdhip64.so' in output:
                log.info('ROCm path: %s (configured)', config_rocm_path)
                _rocm_path_cache[cache_key] = config_rocm_path
                return config_rocm_path
            else:
                log.warning(
                    f'Configured ROCm path {config_rocm_path} does not contain required libraries, will auto-detect'
                )

    # Try new ROCm 7.x structure first (/opt/rocm/core-X.Y)
    out_dict = phdl.exec('ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1', print_console=False)
    for node, output in out_dict.items():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            validate_dict = phdl.exec(
                f'test -d {rocm_path}/lib && ls {rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1',
                print_console=False,
            )
            for _, lib_output in validate_dict.items():
                if lib_output.strip() and 'libamdhip64.so' in lib_output:
                    log.info('ROCm path: %s (auto-detected, new layout)', rocm_path)
                    _rocm_path_cache[cache_key] = rocm_path
                    return rocm_path

    # Fall back to legacy /opt/rocm
    out_dict = phdl.exec(
        'test -d /opt/rocm/lib && ls /opt/rocm/lib/libamdhip64.so* 2>/dev/null | head -1',
        print_console=False,
    )
    for node, output in out_dict.items():
        if output.strip() and 'libamdhip64.so' in output:
            log.info('ROCm path: /opt/rocm (auto-detected, legacy layout)')
            _rocm_path_cache[cache_key] = '/opt/rocm'
            return '/opt/rocm'

    log.warning('Could not detect ROCm path with required libraries, defaulting to /opt/rocm')
    _rocm_path_cache[cache_key] = '/opt/rocm'
    return '/opt/rocm'


def is_dmabuf_required(config_dict):
    """Require DMA-BUF by default, including for unrecognized config values."""
    value = config_dict.get('require_dmabuf', True)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('true', '1', 'yes', 'on'):
            return True
        if normalized in ('false', '0', 'no', 'off'):
            return False
    log.warning('Unrecognized require_dmabuf value %r; treating as True', value)
    return True


def probe_perftest_dmabuf_support(phdl, binary_path, rocm_path=''):
    """Probe each node's binary, using the launch ROCm lib path to distinguish load failures from missing DMA-BUF."""
    ld_prefix = f'LD_LIBRARY_PATH={rocm_path}/lib:$LD_LIBRARY_PATH ' if rocm_path else ''
    cmd = (
        f'help_out=$({ld_prefix}{binary_path} --help 2>&1); '
        'case "$help_out" in '
        '*use_rocm_dmabuf*) echo "dmabuf=supported" ;; '
        '*use_rocm*) echo "dmabuf=unsupported" ;; '
        '*) echo "dmabuf=probe_failed" ;; '
        'esac'
    )
    out_dict = phdl.exec(cmd, print_console=False)
    status_dict = {}
    for node, output in out_dict.items():
        match = re.search(r'dmabuf=(supported|unsupported|probe_failed)', output)
        status_dict[node] = match.group(1) if match else DMABUF_PROBE_FAILED
        log.debug('Perftest DMA-BUF probe on %s: %s', node, status_dict[node])
    return status_dict


def check_perftest_dmabuf(phdl, binary_path, rocm_path='', require_dmabuf=True):
    """Return whether DMA-BUF can be used on every node and report per-node failures."""
    status_dict = probe_perftest_dmabuf_support(phdl, binary_path, rocm_path)
    if not status_dict:
        fail_test(f'Could not probe perftest binary {binary_path} on any test node: no probe output')
    broken = sorted(node for node, status in status_dict.items() if status == DMABUF_PROBE_FAILED)
    missing = sorted(node for node, status in status_dict.items() if status == DMABUF_UNSUPPORTED)
    for node in broken:
        fail_test(
            f'Could not probe perftest binary {binary_path} on node {node}: '
            'binary missing, not loadable (check rocm_dir), or built without --enable-rocm'
        )
    if missing and require_dmabuf:
        for node in missing:
            fail_test(
                f'perftest binary {binary_path} on node {node} lacks ROCm DMA-BUF support '
                f'(--use_rocm_dmabuf not in --help); {PERFTEST_DMABUF_BUILD_HINT}, '
                'or set "require_dmabuf": "False" in the ibperf config to run without DMA-BUF'
            )
    elif missing:
        log.warning(
            '%s lacks DMA-BUF support on %d node(s) %s; running without --use_rocm_dmabuf on all nodes',
            binary_path,
            len(missing),
            ', '.join(missing),
        )
    use_dmabuf = bool(status_dict) and not missing and not broken
    log.info(
        '%s: DMA-BUF %s (%d nodes checked)',
        binary_path,
        'enabled' if use_dmabuf else 'disabled',
        len(status_dict),
    )
    return use_dmabuf


def get_perftest_configure_cmd(install_dir, rocm_path, rocm_dmabuf=True):
    """
    Build the shell command that configures a perftest source checkout for ROCm.

    Args:
        install_dir (str): Directory containing the perftest checkout; also used as
                           the base of the --prefix.
        rocm_path (str): ROCm installation path passed to --with-rocm.
        rocm_dmabuf (bool): Append --enable-rocm-dmabuf so perftest builds the
                            --use_rocm_dmabuf option.

    Returns:
        str: Command to run on the build node.
    """
    cmd = (
        f'cd {install_dir}/perftest; ./configure --prefix={install_dir}/perftest --with-rocm={rocm_path} --enable-rocm'
    )
    if rocm_dmabuf:
        cmd += ' --enable-rocm-dmabuf'
    return cmd


def configure_perftest(shdl, install_dir, rocm_path):
    """
    Configure perftest with ROCm DMA-BUF support, falling back to a ROCm-only
    configure when the DMA-BUF configure fails (e.g. a ROCm without
    hsa_amd_portable_export_dmabuf).

    Args:
        shdl: Pssh handle for the build (head) node.
        install_dir (str): Directory containing the perftest checkout.
        rocm_path (str): ROCm installation path.

    Returns:
        bool: True if the DMA-BUF configure succeeded, False if it fell back.
    """
    cmd = get_perftest_configure_cmd(install_dir, rocm_path, rocm_dmabuf=True)
    log.info('Configuring perftest: %s', cmd)
    out_dict = shdl.exec(cmd, timeout=200, print_console=False, detailed=True)
    failed = [node for node, res in out_dict.items() if res.get('exit_code') != 0]
    if not failed:
        log.info('perftest configured with ROCm DMA-BUF support')
        return True

    log.warning(
        'perftest configure with --enable-rocm-dmabuf failed on %s; retrying without ROCm DMA-BUF',
        ', '.join(failed),
    )
    for node in failed:
        tail = ' | '.join(out_dict[node].get('output', '').splitlines()[-5:])
        log.debug('%s configure output: %s', node, tail)

    cmd = get_perftest_configure_cmd(install_dir, rocm_path, rocm_dmabuf=False)
    log.info('Configuring perftest: %s', cmd)
    out_dict = shdl.exec(cmd, timeout=200, print_console=False, detailed=True)
    for node, res in out_dict.items():
        if res.get('exit_code') != 0:
            tail = ' | '.join(res.get('output', '').splitlines()[-5:])
            fail_test(f'perftest configure failed on node {node}: {tail}')
    return False


def parse_kfd_gpu_bdf_list(kfd_output):
    """Return lowercase GPU BDFs in HIP device ordinal order from KFD properties output."""
    props = {}
    for line in kfd_output.splitlines():
        match = re.match(r'/sys/class/kfd/kfd/topology/nodes/(\d+)/properties:(\w+)\s+(\d+)\s*$', line.strip())
        if match:
            node, key, value = match.groups()
            props.setdefault(int(node), {})[key] = int(value)

    # ROCr enumerates GPU agents in KFD node order, which differs from rocm-smi card order.
    bdfs = []
    for node in sorted(props):
        if props[node].get('simd_count', 0) > 0:
            loc = props[node].get('location_id', 0)
            dom = props[node].get('domain', 0)
            bdfs.append(f'{dom:04x}:{(loc >> 8) & 0xFF:02x}:{(loc >> 3) & 0x1F:02x}.{loc & 0x7:x}')
    return bdfs


def parse_gpu_visibility_env(probe_output):
    """Return non-empty GPU visibility variables found on tagged probe lines."""
    visible_env = {}
    for line in probe_output.splitlines():
        match = re.match(rf'{_GPU_ENV_TAG} (\w+)=(.*)$', line.strip())
        if match:
            var, value = match.groups()
            value = value.strip()
            if value and var in _GPU_VISIBILITY_ENV_VARS:
                visible_env[var] = value
    return visible_env


def _card_index(card):
    return int(re.sub(r'\D', '', card))


def get_hip_device_dict(phdl, gpu_nic_dict):
    """Return each node's card-to-HIP-ordinal map, failing and falling back to card index when unknown."""
    out_dict = phdl.exec(_HIP_DEVICE_PROBE_CMD, print_console=False)
    hip_dev_dict = {}
    for node in gpu_nic_dict:
        node_out = out_dict.get(node, '')
        hip_dev_dict[node] = {}
        visible_env = parse_gpu_visibility_env(node_out)
        if visible_env:
            fail_test(
                f'GPU visibility env set on node {node}: '
                + ', '.join(f'{var}={value}' for var, value in visible_env.items())
                + '; HIP ordinal cannot be derived from KFD topology when device visibility is remapped, '
                'falling back to card index for --use_rocm'
            )
            for card in gpu_nic_dict[node]:
                hip_dev_dict[node][card] = _card_index(card)
        else:
            hip_bdfs = parse_kfd_gpu_bdf_list(node_out)
            for card in gpu_nic_dict[node]:
                bdf = str(gpu_nic_dict[node][card].get('gpu_bdf', '')).lower()
                if bdf in hip_bdfs:
                    hip_dev_dict[node][card] = hip_bdfs.index(bdf)
                else:
                    fail_test(
                        f'Could not map {card} (GPU {bdf}) to a HIP device on node {node}; KFD GPU BDFs: {hip_bdfs}'
                    )
                    hip_dev_dict[node][card] = _card_index(card)
        log.info(
            'HIP device map for %s: %s',
            node,
            {
                card: (
                    gpu_nic_dict[node][card].get('rdma_dev'),
                    gpu_nic_dict[node][card].get('gpu_bdf'),
                    hip_dev_dict[node][card],
                )
                for card in gpu_nic_dict[node]
            },
        )
    return hip_dev_dict


def get_ibperf_rdma_devices(gpu_nic_dict, bck_nic_dict, gpu_count=None):
    """Mirror the NIC selection in the bandwidth and latency perftest runners."""
    return {
        node: sorted(
            {
                device
                for card in (
                    get_node_gpu_cards(gpu_nic_dict, node)
                    if gpu_count is None
                    else (f'card{gpu}' for gpu in range(gpu_count))
                )
                if (device := gpu_nic_dict.get(node, {}).get(card, {}).get('rdma_dev')) is not None
            }
        )
        for node in bck_nic_dict
    }


def resolve_gid_index(phdl, node_dev_dict, gid_index, expected_gid_type=DEFAULT_GID_TYPE):
    """Select or validate a GID index before launching perftest."""
    devices = sorted({device for devs in node_dev_dict.values() for device in devs})
    if is_auto_gid_index(gid_index):
        output = phdl.exec(build_gid_table_cmd(devices), print_console=False)
        tables = {node: parse_gid_table_output(output.get(node, '')) for node in node_dev_dict}
        index, errors = select_common_gid_index(tables, node_dev_dict, expected_gid_type)
        if errors:
            for error in errors:
                fail_test(error)
            return None
        log.info(
            'Auto-selected GID index %s for %d nodes (%s IPv4-mapped on RoCE, lowest populated on InfiniBand)',
            index,
            len(node_dev_dict),
            expected_gid_type,
        )
    else:
        try:
            index = normalize_gid_index(gid_index)
        except ValueError as exc:
            fail_test(f'ibperf gid_index is invalid: {exc}')
            return None

    output = phdl.exec(build_gid_probe_cmd(index, devices), print_console=False)
    failed = False
    for node, node_devices in node_dev_dict.items():
        entries = parse_gid_probe_output(output.get(node, ''))
        for device in node_devices:
            status, message = check_gid_entry(device, entries.get(device), index, expected_gid_type)
            if status != GID_OK:
                fail_test(f'Node {node} NIC {device}: {message}')
                failed = True
    if failed:
        return None
    log.info(
        'GID index %s validated on %d NICs across %d nodes (expected type %s, not enforced on InfiniBand)',
        index,
        sum(map(len, node_dev_dict.values())),
        len(node_dev_dict),
        expected_gid_type,
    )
    return index


def wait_for_perftest_exit(phdl, app_name, timeout):
    """Poll every node until ``pgrep`` confirms no ``app_name`` process is left or ``timeout`` seconds pass.

    Only a ``pgrep`` no-match counts as exited, so a node whose poll errored keeps the wait going.
    ``pgrep -x`` matches the kernel's 15-character process name, so ``app_name`` is truncated to it.
    Returns the nodes not confirmed exited when the wait ended; empty when all exited.
    """
    cmd = f'pgrep -x {app_name[:15]} > /dev/null; echo "pgrep_rc=$?"'

    def unconfirmed():
        out_dict = phdl.exec(cmd, print_console=False)
        return [node for node, out in out_dict.items() if not re.search(r'\bpgrep_rc=1\b', out)]

    nodes, _ = poll_until(unconfirmed, lambda pending: not pending, timeout, PERFTEST_EXIT_POLL_S)
    return nodes


def get_ib_bw_pps(phdl, msg_size, cmd, instance_no=None, running=None):
    """Read BW/PPS from each node's perftest log.

    Only nodes in ``running`` (default: every node) are polled for a missing result. A node
    whose perftest pgrep confirmed exited has its final log, so it is read once.
    """
    res_dict = {}

    # Run some of the standard verifications
    out_dict = phdl.exec(cmd, print_console=False)
    err_pattern = (
        "Couldn't initialize ROCm device|Failed to init|Unable to open file descriptor|ERROR|FAIL|Segmentation fault"
    )
    for node in out_dict.keys():
        if not re.search('bytes of GPU buffer', out_dict[node], re.I):
            fail_test(f'GPU Buffer allocation failed or Connection not setup for IB Test on node {node}')

        if re.search(err_pattern, out_dict[node], re.I):
            fail_test(f'IB Test failed - Error patterns seen on node {node}')

    # Collect the BW, PPS numbers
    pattern = r"{}\s+\d+\s+[0-9\.]+\s+([0-9\.]+)\s+([0-9\.]+)".format(msg_size)
    pending = list(out_dict.keys())
    running = set(pending if running is None else running)

    def collect():
        out_dict = phdl.exec(cmd, print_console=False)
        for node in list(pending):
            match = re.search(pattern, out_dict.get(node, ''))
            if match:
                res_dict[node] = {'bw': match.group(1), 'pps': match.group(2)}
                pending.remove(node)
        if pending:
            log.debug('BW results for msg_size %s not ready on %s', msg_size, pending)
        return pending

    poll_until(
        collect, lambda nodes: not running.intersection(nodes), PERFTEST_BW_RESULT_TIMEOUT_S, PERFTEST_RESULT_POLL_S
    )
    for node in pending:
        why = (
            f'no result within {PERFTEST_BW_RESULT_TIMEOUT_S}s'
            if node in running
            else 'perftest exited without a result'
        )
        fail_test(f'ERROR !!! on node {node} Client did not complete for msg size {msg_size}: {why}')
        fail_test(f'ERROR !!! pls check log file for errors on node {node}')
    _log_bw_summary(msg_size, res_dict, instance_no)
    return res_dict


def get_ib_lat_numb(phdl, msg_size, cmd, instance_no=None, result_nodes=None, running=None):
    """Read latency from each node's perftest log; ``running`` works as in ``get_ib_bw_pps``."""
    res_dict = {}

    out_dict = phdl.exec(cmd, print_console=False)
    err_pattern = (
        "Couldn't initialize ROCm device|Failed to init|Unable to open file descriptor|ERROR|FAIL|Segmentation fault"
    )
    required_nodes = list(result_nodes) if result_nodes is not None else list(out_dict)
    for node in out_dict:
        if not re.search('bytes of GPU buffer', out_dict[node], re.I):
            fail_test(f'GPU Buffer allocation failed or Connection not setup for IB Test on node {node}')
        if re.search(err_pattern, out_dict[node], re.I):
            fail_test(f'IB Test failed - Error patterns seen on node {node}')

    pattern = r"{}\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)".format(
        msg_size
    )
    keys = ('t_min', 't_max', 't_typical', 't_avg', 't_stdev', 't_99_pct', 't_99_9_pct')
    pending = list(required_nodes)
    running = set(pending if running is None else running)

    def collect():
        out_dict = phdl.exec(cmd, print_console=False)
        for node in list(pending):
            match = re.search(pattern, out_dict.get(node, ''))
            if match:
                res_dict[node] = dict(zip(keys, match.groups()))
                pending.remove(node)
        if pending:
            log.debug('Latency results for msg_size %s not ready on %s', msg_size, pending)
        return pending

    poll_until(
        collect, lambda nodes: not running.intersection(nodes), PERFTEST_LAT_RESULT_TIMEOUT_S, PERFTEST_RESULT_POLL_S
    )
    for node in pending:
        fail_test(f'ERROR !!! on node {node} Client did not complete for msg size {msg_size}')
    _log_lat_summary(msg_size, res_dict, instance_no)
    return res_dict


def verify_expected_bw(bw_test, msg_size, qp_count, res_dict, expected_res):
    """Fail the test for every GPU instance whose bandwidth is below the configured threshold.

    Args:
        bw_test (str): Perftest binary name, such as 'ib_write_bw'.
        msg_size (int | str): Message size from msg_size_list.
        qp_count (int | str): QP count from qp_count_list.
        res_dict (dict): Results keyed by node, then GPU instance, with 'bw' values.
        expected_res (dict): Configured thresholds keyed by test, message size, and QP count.

    Returns:
        None if no threshold applies, True if every instance meets it, False otherwise.
    """
    test_thresholds = (expected_res or {}).get(bw_test)
    if test_thresholds is None:
        _log_not_gated(bw_test, msg_size, qp_count)
        return None
    if not isinstance(test_thresholds, dict):
        fail_test(f'expected_results.{bw_test} must map message size to QP thresholds, got {test_thresholds!r}')
        return False

    # JSON keys are strings, while sweep lists may contain integers.
    msg_thresholds = test_thresholds.get(str(msg_size))
    if msg_thresholds is None:
        _log_not_gated(bw_test, msg_size, qp_count)
        return None
    if not isinstance(msg_thresholds, dict):
        fail_test(f'expected_results.{bw_test}.{msg_size} must map QP count to Gbps, got {msg_thresholds!r}')
        return False

    raw = msg_thresholds.get(str(qp_count))
    if raw is None:
        _log_not_gated(bw_test, msg_size, qp_count)
        return None
    try:
        threshold = float(raw)
    except (TypeError, ValueError):
        threshold = math.nan
    # NaN compares false against everything, so it would silently pass every GPU.
    if not math.isfinite(threshold):
        fail_test(f'Invalid expected BW {raw!r} for {bw_test} msg_size={msg_size} qp={qp_count}')
        return False

    log.info('Verifying expected BW: %s msg_size=%s qp=%s threshold=%.2f Gbps', bw_test, msg_size, qp_count, threshold)
    if not res_dict:
        fail_test(
            f'No BW results for {bw_test} msg_size={msg_size} qp={qp_count}; cannot verify expected BW {threshold} Gbps'
        )
        return False
    passed = True
    count = 0
    for node in sorted(res_dict):
        if not res_dict[node]:
            fail_test(
                f'No BW results for {bw_test} msg_size={msg_size} qp={qp_count} on node {node}; '
                f'cannot verify expected BW {threshold} Gbps'
            )
            passed = False
            continue
        for inst in sorted(res_dict[node]):
            bw = res_dict[node][inst].get('bw')
            if bw is None:
                fail_test(f'No BW result for {bw_test} msg_size={msg_size} qp={qp_count} on node {node} gpu={inst}')
                passed = False
                continue
            try:
                actual_bw = float(bw)
            except (TypeError, ValueError):
                fail_test(
                    f'Invalid actual BW {bw!r} for {bw_test} msg_size={msg_size} qp={qp_count} on node {node} gpu={inst}'
                )
                passed = False
                continue
            count += 1
            if actual_bw < threshold:
                fail_test(
                    f'Actual BW {bw} Gbps less than the expected BW {threshold} Gbps for test '
                    f'{bw_test} msg_size={msg_size} qp={qp_count} on node {node} gpu={inst}'
                )
                passed = False
    if passed:
        log.info(
            'Expected BW met: %s msg_size=%s qp=%s, all %d GPU instances >= %.2f Gbps',
            bw_test,
            msg_size,
            qp_count,
            count,
            threshold,
        )
    return passed


def verify_expected_lat(lat_test, msg_size, res_dict, expected_res):
    """Fail the test for every GPU instance whose average latency exceeds the configured threshold.

    Args:
        lat_test (str): Perftest binary name, such as 'ib_write_lat'.
        msg_size (int | str): Message size from msg_size_list.
        res_dict (dict): Results keyed by node, then GPU instance, with 't_avg' values.
        expected_res (dict): Configured thresholds keyed by test and message size.

    Returns:
        None if no threshold applies, True if every instance meets it, False otherwise.
    """
    test_thresholds = (expected_res or {}).get(lat_test)
    if test_thresholds is None:
        _log_not_gated(lat_test, msg_size)
        return None
    if not isinstance(test_thresholds, dict):
        fail_test(f'expected_results.{lat_test} must map message size to latency, got {test_thresholds!r}')
        return False

    # JSON keys are strings, while sweep lists may contain integers.
    raw = test_thresholds.get(str(msg_size))
    if raw is None:
        _log_not_gated(lat_test, msg_size)
        return None
    if isinstance(raw, dict):
        fail_test(f'expected_results.{lat_test}.{msg_size} must be a latency in usec, got {raw!r}')
        return False
    try:
        threshold = float(raw)
    except (TypeError, ValueError):
        threshold = math.nan
    if not math.isfinite(threshold):
        fail_test(f'Invalid expected latency {raw!r} for {lat_test} msg_size={msg_size}')
        return False

    log.info('Verifying expected latency: %s msg_size=%s threshold=%.2f us', lat_test, msg_size, threshold)
    if not res_dict:
        fail_test(
            f'No latency results for {lat_test} msg_size={msg_size}; cannot verify expected latency {threshold} us'
        )
        return False
    passed = True
    count = 0
    for node in sorted(res_dict):
        if not res_dict[node]:
            if lat_test == 'ib_read_lat':
                continue  # passive/server side of an RDMA read has no results by design
            fail_test(
                f'No latency results for {lat_test} msg_size={msg_size} on node {node}; '
                f'cannot verify expected latency {threshold} us'
            )
            passed = False
            continue
        for inst in sorted(res_dict[node]):
            t_avg = res_dict[node][inst].get('t_avg')
            if t_avg is None:
                fail_test(f'No latency result for {lat_test} msg_size={msg_size} on node {node} gpu={inst}')
                passed = False
                continue
            try:
                actual_lat = float(t_avg)
            except (TypeError, ValueError):
                fail_test(
                    f'Invalid actual latency {t_avg!r} for {lat_test} msg_size={msg_size} on node {node} gpu={inst}'
                )
                passed = False
                continue
            count += 1
            if actual_lat > threshold:
                fail_test(
                    f'Actual latency {t_avg} us greater than the expected latency {threshold} us for test '
                    f'{lat_test} msg_size={msg_size} on node {node} gpu={inst}'
                )
                passed = False
    if passed:
        log.info(
            'Expected latency met: %s msg_size=%s, all %d GPU instances <= %.2f us',
            lat_test,
            msg_size,
            count,
            threshold,
        )
    return passed


def _log_not_gated(test_name, msg_size, qp_count=None):
    """Log a skipped threshold separately from a passing threshold."""
    qp_label = f' qp={qp_count}' if qp_count is not None else ''
    log.info('No expected result configured for %s msg_size=%s%s; not gated', test_name, msg_size, qp_label)


def find_unmatched_thresholds(expected_res, msg_size_list, qp_count_list):
    """Return dotted paths of thresholds the configured sweep cannot evaluate.

    Bandwidth entries map message size to QP counts. Latency entries map message
    size directly to a threshold. Entries outside the message size sweep report
    the test and size; QP entries outside the QP sweep also report the QP count.

    Returns:
        list of str: Unmatched paths in config iteration order, or an empty list.
    """
    msg_keys = {str(msg_size) for msg_size in msg_size_list}
    qp_keys = {str(qp_count) for qp_count in qp_count_list}
    unmatched = []
    for test, by_msg in (expected_res or {}).items():
        if not isinstance(by_msg, dict):
            continue
        for msg_key, value in by_msg.items():
            if msg_key not in msg_keys:
                unmatched.append(f'{test}.{msg_key}')
                continue
            if isinstance(value, dict):
                for qp_key in value:
                    if qp_key not in qp_keys:
                        unmatched.append(f'{test}.{msg_key}.{qp_key}')
    return unmatched


def run_ib_perf_bw_test(
    phdl,
    bw_test,
    gpu_numa_dict,
    gpu_nic_dict,
    bck_nic_dict,
    app_path,
    msg_size,
    gid_index,
    qp_count=8,
    port_no=1516,
    duration=60,
    rocm_path='',
    use_dmabuf=True,
    hip_dev_dict=None,
    node_pairs=None,
):
    log.info(
        '%s: msg_size=%s qp=%s start (%ss, %d nodes)',
        bw_test,
        msg_size,
        qp_count,
        duration,
        len(bck_nic_dict),
    )
    if hip_dev_dict is None:
        hip_dev_dict = get_hip_device_dict(phdl, gpu_nic_dict)
    app_port = port_no
    result_dict = {}
    cmd_dict = {}
    phdl.exec('rm -rf /tmp/ib_cmds_file.txt', print_console=False)
    phdl.exec('rm -rf /tmp/ib_perf*', print_console=False)
    phdl.exec('touch /tmp/ib_cmds_file.txt', print_console=False)
    if rocm_path:
        log.debug('Setting LD_LIBRARY_PATH to %s/lib for perftest binaries', rocm_path)
        phdl.exec(
            f'echo "export LD_LIBRARY_PATH={rocm_path}/lib:$LD_LIBRARY_PATH" >> /tmp/ib_cmds_file.txt',
            print_console=False,
        )
    nodes = list(bck_nic_dict.keys())
    if node_pairs is None:
        node_pairs = build_ibperf_node_pairs(nodes)[0]
    roles, cards, excluded = _prepare_pair_roles(node_pairs, nodes, gpu_nic_dict, gpu_numa_dict, hip_dev_dict, bw_test)
    if excluded:
        phdl.prune_nodes(excluded)
        for node in excluded:
            bck_nic_dict.pop(node, None)
    if not roles:
        return result_dict
    for node, server_addr in roles.items():
        result_dict[node] = {}
        cmd_dict[node] = []
        if server_addr is None:
            # server
            cmd = 'echo "sleep 1" >> /tmp/ib_cmds_file.txt'
            cmd_dict[node].append(cmd)
            inst_count = 0
            port_no = app_port
            for card_no in cards:
                rdma_dev = gpu_nic_dict[node][card_no]['rdma_dev']
                cmd_parts = [
                    "numactl",
                    f'--physcpubind={gpu_numa_dict[node][card_no]["local_cpulist"]}',
                    "--localalloc",
                    f"{app_path}/{bw_test}",
                    "-d",
                    rdma_dev,
                    f"--use_rocm={hip_dev_dict[node][card_no]}",
                    *(["--use_rocm_dmabuf"] if use_dmabuf else []),
                    "-x",
                    str(gid_index),
                    "--report_gbits",
                    "-b",
                    "-F",
                    "-D",
                    str(duration),
                    "-p",
                    str(port_no),
                    "-s",
                    str(msg_size),
                    "-q",
                    str(qp_count),
                ]
                cmd = " ".join(cmd_parts) + f" > /tmp/ib_perf_{inst_count}_logs 2>&1 &"
                cmd_dict[node].append(f'echo "{cmd}" >> /tmp/ib_cmds_file.txt')
                inst_count = inst_count + 1
                port_no = port_no + 1
        else:
            # client
            cmd = 'echo "sleep 5" >> /tmp/ib_cmds_file.txt'
            cmd_dict[node].append(cmd)
            inst_count = 0
            port_no = app_port
            for card_no in cards:
                rdma_dev = gpu_nic_dict[node][card_no]['rdma_dev']
                cmd_parts = [
                    "numactl",
                    f'--physcpubind={gpu_numa_dict[node][card_no]["local_cpulist"]}',
                    "--localalloc",
                    f"{app_path}/{bw_test}",
                    "-d",
                    rdma_dev,
                    f"--use_rocm={hip_dev_dict[node][card_no]}",
                    *(["--use_rocm_dmabuf"] if use_dmabuf else []),
                    "-x",
                    str(gid_index),
                    "--report_gbits",
                    "-b",
                    "-F",
                    "-D",
                    str(duration),
                    "-p",
                    str(port_no),
                    "-s",
                    str(msg_size),
                    "-q",
                    str(qp_count),
                    server_addr,
                ]
                cmd = " ".join(cmd_parts) + f" > /tmp/ib_perf_{inst_count}_logs 2>&1 &"
                cmd_dict[node].append(f'echo "{cmd}" >> /tmp/ib_cmds_file.txt')
                inst_count = inst_count + 1
                port_no = port_no + 1

    # Get number of commands in cmd_dict - should be same for all nodes
    node_list = list(cmd_dict.keys())
    first_node_cmd_list = cmd_dict[node_list[0]]

    for j in range(0, len(first_node_cmd_list)):
        cmd_list = []
        for node in cmd_dict.keys():
            cmd_list.append(cmd_dict[node][j])
        # print(cmd_list)
        phdl.exec_cmd_list(cmd_list, print_console=False)

    # Clean up any stale instances from earlier runs (ignore if none running)
    log.debug('Killing stale %s processes before starting test', bw_test)
    phdl.exec(f'killall {bw_test} 2>/dev/null || true', print_console=False)
    time.sleep(2)
    phdl.exec(PERFTEST_LAUNCH_CMD, print_console=False)

    exit_timeout = duration + PERFTEST_EXIT_SLACK_S
    unconfirmed = wait_for_perftest_exit(phdl, bw_test, exit_timeout)
    if unconfirmed:
        log.warning('%s exit not confirmed on %s after %ss; reading logs anyway', bw_test, unconfirmed, exit_timeout)

    for instance_no in range(0, inst_count):
        try:
            bw_pps_dict = get_ib_bw_pps(
                phdl, msg_size, f'cat /tmp/ib_perf_{instance_no}_logs', instance_no=instance_no, running=unconfirmed
            )
            for node in bw_pps_dict.keys():
                result_dict[node][instance_no] = {}
                result_dict[node][instance_no]['pps'] = bw_pps_dict[node]['pps']
                result_dict[node][instance_no]['bw'] = bw_pps_dict[node]['bw']
        except Exception:
            log.error(
                'Failed to collect BW/PPS for msg_size=%s qp_count=%s instance=%d',
                msg_size,
                qp_count,
                instance_no,
                exc_info=True,
            )

    _log_bw_run_done(bw_test, msg_size, qp_count, result_dict)
    return result_dict


def run_ib_perf_lat_test(
    phdl,
    lat_test,
    gpu_numa_dict,
    gpu_nic_dict,
    bck_nic_dict,
    app_path,
    msg_size,
    gid_index,
    port_no=1516,
    rocm_path='',
    use_dmabuf=True,
    hip_dev_dict=None,
    node_pairs=None,
):
    log.info('%s: msg_size=%s start (%d nodes)', lat_test, msg_size, len(bck_nic_dict))
    if hip_dev_dict is None:
        hip_dev_dict = get_hip_device_dict(phdl, gpu_nic_dict)
    app_port = port_no
    result_dict = {}
    cmd_dict = {}
    phdl.exec('rm -rf /tmp/ib_cmds_file.txt', print_console=False)
    phdl.exec('rm -rf /tmp/ib_perf*', print_console=False)
    phdl.exec('touch /tmp/ib_cmds_file.txt', print_console=False)
    if rocm_path:
        log.debug('Setting LD_LIBRARY_PATH to %s/lib for perftest binaries', rocm_path)
        phdl.exec(
            f'echo "export LD_LIBRARY_PATH={rocm_path}/lib:$LD_LIBRARY_PATH" >> /tmp/ib_cmds_file.txt',
            print_console=False,
        )
    nodes = list(bck_nic_dict.keys())
    if node_pairs is None:
        node_pairs = build_ibperf_node_pairs(nodes)[0]
    roles, cards, excluded = _prepare_pair_roles(node_pairs, nodes, gpu_nic_dict, gpu_numa_dict, hip_dev_dict, lat_test)
    if excluded:
        phdl.prune_nodes(excluded)
        for node in excluded:
            bck_nic_dict.pop(node, None)
    if not roles:
        return result_dict
    for node, server_addr in roles.items():
        result_dict[node] = {}
        cmd_dict[node] = []
        if server_addr is None:
            # server
            cmd = 'echo "sleep 1" >> /tmp/ib_cmds_file.txt'
            cmd_dict[node].append(cmd)
            inst_count = 0
            port_no = app_port
            for card_no in cards:
                rdma_dev = gpu_nic_dict[node][card_no]['rdma_dev']
                cmd_parts = [
                    "numactl",
                    f'--physcpubind={gpu_numa_dict[node][card_no]["local_cpulist"]}',
                    "--localalloc",
                    f"{app_path}/{lat_test}",
                    "-d",
                    rdma_dev,
                    f"--use_rocm={hip_dev_dict[node][card_no]}",
                    *(["--use_rocm_dmabuf"] if use_dmabuf else []),
                    "-x",
                    str(gid_index),
                    "-F",
                    "-p",
                    str(port_no),
                    "-s",
                    str(msg_size),
                ]
                cmd = " ".join(cmd_parts) + f" > /tmp/ib_perf_{inst_count}_logs 2>&1 &"
                cmd_dict[node].append(f'echo "{cmd}" >> /tmp/ib_cmds_file.txt')
                inst_count = inst_count + 1
                port_no = port_no + 1
        else:
            # client
            cmd = 'echo "sleep 5" >> /tmp/ib_cmds_file.txt'
            cmd_dict[node].append(cmd)
            inst_count = 0
            port_no = app_port
            for card_no in cards:
                rdma_dev = gpu_nic_dict[node][card_no]['rdma_dev']
                cmd_parts = [
                    "numactl",
                    f'--physcpubind={gpu_numa_dict[node][card_no]["local_cpulist"]}',
                    "--localalloc",
                    f"{app_path}/{lat_test}",
                    "-d",
                    rdma_dev,
                    f"--use_rocm={hip_dev_dict[node][card_no]}",
                    *(["--use_rocm_dmabuf"] if use_dmabuf else []),
                    "-x",
                    str(gid_index),
                    "-F",
                    "-p",
                    str(port_no),
                    "-s",
                    str(msg_size),
                    server_addr,
                ]
                cmd = " ".join(cmd_parts) + f" > /tmp/ib_perf_{inst_count}_logs 2>&1 &"
                cmd_dict[node].append(f'echo "{cmd}" >> /tmp/ib_cmds_file.txt')
                inst_count = inst_count + 1
                port_no = port_no + 1

    # Get number of commands in cmd_dict - should be same for all nodes
    node_list = list(cmd_dict.keys())
    first_node_cmd_list = cmd_dict[node_list[0]]

    for j in range(0, len(first_node_cmd_list)):
        cmd_list = []
        for node in cmd_dict.keys():
            cmd_list.append(cmd_dict[node][j])
        # print(cmd_list)
        phdl.exec_cmd_list(cmd_list, print_console=False)

    # Clean up any stale instances from earlier runs (ignore if none running)
    log.debug('Killing stale %s processes before starting test', lat_test)
    phdl.exec(f'killall {lat_test} 2>/dev/null || true', print_console=False)
    time.sleep(2)
    phdl.exec(PERFTEST_LAUNCH_CMD, print_console=False)

    unconfirmed = wait_for_perftest_exit(phdl, lat_test, PERFTEST_LAT_EXIT_TIMEOUT_S)
    if unconfirmed:
        log.warning(
            '%s exit not confirmed on %s after %ss; reading logs anyway',
            lat_test,
            unconfirmed,
            PERFTEST_LAT_EXIT_TIMEOUT_S,
        )

    result_nodes = [node for node, server in roles.items() if server is not None] if lat_test == 'ib_read_lat' else None
    for instance_no in range(0, inst_count):
        try:
            lat_dict = get_ib_lat_numb(
                phdl,
                msg_size,
                f'cat /tmp/ib_perf_{instance_no}_logs',
                instance_no=instance_no,
                result_nodes=result_nodes,
                running=unconfirmed,
            )
            for node, latency in lat_dict.items():
                result_dict[node][instance_no] = latency
        except Exception:
            log.error(
                'Failed to collect latency for msg_size=%s instance=%d',
                msg_size,
                instance_no,
                exc_info=True,
            )

    _log_lat_run_done(lat_test, msg_size, result_dict)
    return result_dict


def split_list_into_n_chunks(original_list, n):
    """
    Splits a list into n approximately equal chunks.
    """
    if not original_list:
        return [[] for _ in range(n)]

    avg_chunk_size = len(original_list) // n
    remainder = len(original_list) % n
    result = []
    current_index = 0

    for i in range(n):
        chunk_size = avg_chunk_size + (1 if i < remainder else 0)
        result.append(original_list[current_index : current_index + chunk_size])
        current_index += chunk_size
    return result


def _to_float(x):
    """Convert ints/floats/strings to float. Accepts '1,234.5' and spaces."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace("_", "")
        # allow trailing percent via uncomment lines below if you want:
        # if s.endswith("%"):
        #     return float(s[:-1]) / 100.0
        return float(s)  # will raise ValueError if not numeric
    raise TypeError(f"Unsupported type {type(x)} for value {x!r}")


def average_of_lists(lists):
    """
    Element-wise average of a list of equal-length lists.
    Normalizes all values to float first.
    """
    if not lists:
        return []
    # normalize
    norm = [[_to_float(v) for v in row] for row in lists]
    # length check
    n = len(norm[0])
    if any(len(row) != n for row in norm):
        raise ValueError("All inner lists must be the same length")
    # average column-wise
    return [sum(col) / len(norm) for col in zip(*norm)]


def round_vals(list_a):
    rounded_vals = [round(float(x), 2) for x in list_a]
    return rounded_vals


# Sample res_dict
# {'ib_write_bw': {2: {'8': {'x.x.x.x': {0: {'pps': '6.229094', 'bw': '0.099666'}, 1: {'pps': '6.391922', 'bw': '0.102271'}, .... 7: { 'pps': '6.79', 'bw': '0.10227' }}
# Top level - application like ib_write_bw, ib_read_bw
# 2 - msg_size
# 8 - Number of QPs
# x.x.x.x - Node IP
# 0 - NIC Instance (will have 0-7)
# pps - packets per sec
# BW - Bandwidth in Gbps


def _dmabuf_heading_suffix(dmabuf_dict, app_name):
    if not dmabuf_dict or app_name not in dmabuf_dict:
        return ''
    return ' (DMA-BUF on)' if dmabuf_dict[app_name] else ' (DMA-BUF off)'


def _result_gpu_list(node_results):
    """Return GPU instance indexes through the largest result index."""
    instances = [index for results in node_results for index in results]
    return list(range(max(instances) + 1)) if instances else []


def generate_ibperf_bw_chart(res_dict, excel_file='ib_bw_pps_perf.xlsx', dmabuf_dict=None):
    workbook = xlsxwriter.Workbook(excel_file)

    app_list = list(res_dict.keys())

    merge_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "bg_color": "yellow",
        }
    )
    node_merge_format = workbook.add_format(
        {
            "bold": 1,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )

    node_list = []
    msg_size_list = []
    qp_count_list = []

    for app_name in app_list:
        for msg_size in res_dict[app_name].keys():
            if msg_size not in msg_size_list:
                msg_size_list.append(msg_size)

            for qp_count in res_dict[app_name][msg_size].keys():
                if qp_count not in qp_count_list:
                    qp_count_list.append(qp_count)

                for node in res_dict[app_name][msg_size][qp_count].keys():
                    if node not in node_list:
                        node_list.append(node)

    gpu_no_list = _result_gpu_list(
        res_dict[a][m][q][n] for a in app_list for m in res_dict[a] for q in res_dict[a][m] for n in res_dict[a][m][q]
    )

    for app_name in app_list:
        for qp_count in qp_count_list:
            sheet_name = app_name + "_qp_" + str(qp_count)
            worksheet = workbook.add_worksheet(sheet_name)
            bold = workbook.add_format({"bold": 1})

            heading = "Test {} - BW, MPPS Numbers for {} QPs{}".format(
                app_name, qp_count, _dmabuf_heading_suffix(dmabuf_dict, app_name)
            )
            worksheet.merge_range(f'A1:{xl_col_to_name(3 + 2 * len(gpu_no_list))}1', heading, merge_format)

            d_msg_size_list = []
            d_bw_gpu = {g: [] for g in gpu_no_list}
            d_pps_gpu = {g: [] for g in gpu_no_list}
            d_pps_total_list = []
            d_bw_total_list = []

            for node in node_list:
                d_msg_size_list.extend(msg_size_list)
                for msg_size in msg_size_list:
                    tot_pps = 0.0
                    tot_bw = 0.0
                    for gpu_no in gpu_no_list:
                        data = res_dict[app_name][msg_size][qp_count][node][gpu_no]
                        d_bw_gpu[gpu_no].append(data['bw'])
                        d_pps_gpu[gpu_no].append(data['pps'])
                        tot_pps = tot_pps + float(data['pps'])
                        tot_bw = tot_bw + float(data['bw'])
                    d_bw_total_list.append(tot_bw)
                    d_pps_total_list.append(tot_pps)

            split_bw_list = split_list_into_n_chunks(d_bw_total_list, len(node_list))
            split_pps_list = split_list_into_n_chunks(d_pps_total_list, len(node_list))
            avg_bw_data = average_of_lists(split_bw_list)
            avg_pps_data = average_of_lists(split_pps_list)
            avg_bw_gpu = {
                g: average_of_lists(split_list_into_n_chunks(d_bw_gpu[g], len(node_list))) for g in gpu_no_list
            }
            avg_pps_gpu = {
                g: average_of_lists(split_list_into_n_chunks(d_pps_gpu[g], len(node_list))) for g in gpu_no_list
            }

            # Merge cols for Node IP
            worksheet.merge_range("A2:A3", "Node IP", bold)
            worksheet.merge_range("B2:B3", "Msg Size", bold)
            # Merge 2 cols for every GPU and Total
            worksheet.merge_range("C2:D2", "Total", bold)
            for pos, g in enumerate(gpu_no_list):
                worksheet.merge_range(f'{xl_col_to_name(4 + 2 * pos)}2:{xl_col_to_name(5 + 2 * pos)}2', f'GPU{g}', bold)

            headings = ['BW', 'PPS'] * (len(gpu_no_list) + 1)
            # Write headings for BW, PPS for total and every GPU
            worksheet.write_row("C3", headings, bold)

            cur_row = 4
            msg_list_len = len(msg_size_list)
            for node_ip in node_list:
                worksheet.merge_range(f"A{cur_row}:A{cur_row + msg_list_len - 1}", node_ip, node_merge_format)
                cur_row = cur_row + msg_list_len

            # For the cluster Avg
            worksheet.merge_range(f"A{cur_row}:A{cur_row + msg_list_len - 1}", "Cluster Avg")

            worksheet.write_column("B4", round_vals(d_msg_size_list))
            worksheet.write_column("C4", round_vals(d_bw_total_list))
            worksheet.write_column("D4", round_vals(d_pps_total_list))
            for pos, g in enumerate(gpu_no_list):
                worksheet.write_column(f'{xl_col_to_name(4 + 2 * pos)}4', round_vals(d_bw_gpu[g]))
                worksheet.write_column(f'{xl_col_to_name(5 + 2 * pos)}4', round_vals(d_pps_gpu[g]))

            x_chart_index = len(d_msg_size_list) + 4

            worksheet.write_column(f"B{x_chart_index}", msg_size_list)
            worksheet.write_column(f"C{x_chart_index}", round_vals(avg_bw_data))
            worksheet.write_column(f"D{x_chart_index}", round_vals(avg_pps_data))
            for pos, g in enumerate(gpu_no_list):
                worksheet.write_column(f'{xl_col_to_name(4 + 2 * pos)}{x_chart_index}', round_vals(avg_bw_gpu[g]))
                worksheet.write_column(f'{xl_col_to_name(5 + 2 * pos)}{x_chart_index}', round_vals(avg_pps_gpu[g]))

            x_chart_index = len(d_msg_size_list) + len(msg_size_list) + 7
            y_chart_index = 2
            chart = workbook.add_chart({"type": "column"})

            row_end = 4 + len(d_msg_size_list)

            chart.add_series(
                {
                    "name": "={}!$B$2".format(sheet_name),
                    "categories": "={}!$B${}:$B${}".format(sheet_name, row_end, row_end + len(msg_size_list)),
                    "values": "={}!$C${}:$C${}".format(sheet_name, row_end, row_end + len(msg_size_list)),
                    "data_labels": {
                        "value": True,
                        "font": {
                            "rotation": -45,
                        },
                    },
                }
            )

            chart.set_size({'width': 1200, 'height': 720})
            chart.set_title({"name": "Bandwidth in Gbps"})
            chart.set_x_axis({"name": "Msg Size"})
            chart.set_y_axis({"name": "Bandwidth in Gbps"})
            chart.set_style(11)
            worksheet.insert_chart(x_chart_index, y_chart_index, chart)

    workbook.close()
    log.info('BW performance chart saved: %s', excel_file)


def generate_ibperf_lat_chart(res_dict, excel_file='ib_lat_perf.xlsx', dmabuf_dict=None):
    """Write latency measurements for nodes and GPU instances with available rows."""
    metrics = ('t_min', 't_max', 't_avg', 't_stdev', 't_99_pct')
    for app_name, sizes in res_dict.items():
        if not any(instances for nodes in sizes.values() for instances in nodes.values()):
            raise ValueError(f'No latency results for {app_name}')

    workbook = xlsxwriter.Workbook(excel_file)
    merge_format = workbook.add_format(
        {'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'bg_color': 'yellow'}
    )
    node_merge_format = workbook.add_format({'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
    bold = workbook.add_format({'bold': 1})

    for app_name, sizes in res_dict.items():
        gpu_no_list = _result_gpu_list(instances for nodes in sizes.values() for instances in nodes.values())
        sheet_name = app_name + '_lat'
        worksheet = workbook.add_worksheet(sheet_name)
        worksheet.merge_range(
            'A1:Z1', f'Test {app_name} - latency results{_dmabuf_heading_suffix(dmabuf_dict, app_name)}', merge_format
        )
        worksheet.merge_range('A2:A3', 'Node IP', bold)
        worksheet.merge_range('B2:B3', 'Msg Size', bold)
        worksheet.merge_range('C2:G2', 'Node Avg ', bold)
        for gpu_no in gpu_no_list:
            first_col = 7 + gpu_no * len(metrics)
            worksheet.merge_range(1, first_col, 1, first_col + len(metrics) - 1, f'GPU{gpu_no}', bold)
        worksheet.write_row(2, 2, ['Tmin', 'Tmax', 'TAvg', 'TStdev', 'T99pct'] * (len(gpu_no_list) + 1), bold)

        nodes = list(dict.fromkeys(node for size_nodes in sizes.values() for node in size_nodes))
        msg_sizes = [size for size, size_nodes in sizes.items() if any(size_nodes.values())]
        row = 3
        for node in nodes:
            node_rows = [(size, sizes[size].get(node, {})) for size in msg_sizes if sizes[size].get(node)]
            if not node_rows:
                continue
            if len(node_rows) > 1:
                worksheet.merge_range(row, 0, row + len(node_rows) - 1, 0, node, node_merge_format)
            else:
                worksheet.write(row, 0, node, node_merge_format)
            for size, instances in node_rows:
                worksheet.write(row, 1, size)
                for col, metric in enumerate(metrics, 2):
                    worksheet.write_number(
                        row, col, round(sum(float(data[metric]) for data in instances.values()) / len(instances), 2)
                    )
                for gpu_no, data in instances.items():
                    for col, metric in enumerate(metrics):
                        worksheet.write_number(row, 7 + gpu_no * len(metrics) + col, round(float(data[metric]), 2))
                row += 1

        cluster_row = row + 1
        if len(msg_sizes) > 1:
            worksheet.merge_range(cluster_row, 0, cluster_row + len(msg_sizes) - 1, 0, 'Cluster Avg')
        else:
            worksheet.write(cluster_row, 0, 'Cluster Avg')
        for offset, size in enumerate(msg_sizes):
            instances = [data for node_instances in sizes[size].values() for data in node_instances.values()]
            worksheet.write(cluster_row + offset, 1, size)
            for col, metric in enumerate(metrics, 2):
                worksheet.write_number(
                    cluster_row + offset, col, round(sum(float(data[metric]) for data in instances) / len(instances), 2)
                )
            for gpu_no in gpu_no_list:
                gpu_rows = [
                    node_instances[gpu_no] for node_instances in sizes[size].values() if gpu_no in node_instances
                ]
                if gpu_rows:
                    for col, metric in enumerate(metrics):
                        worksheet.write_number(
                            cluster_row + offset,
                            7 + gpu_no * len(metrics) + col,
                            round(sum(float(data[metric]) for data in gpu_rows) / len(gpu_rows), 2),
                        )

        chart = workbook.add_chart({'type': 'column'})
        chart.add_series(
            {
                'name': f'={sheet_name}!$B$2',
                'categories': [sheet_name, cluster_row, 1, cluster_row + len(msg_sizes) - 1, 1],
                'values': [sheet_name, cluster_row, 4, cluster_row + len(msg_sizes) - 1, 4],
                'data_labels': {'value': True, 'font': {'rotation': -45}},
            }
        )
        chart.set_size({'width': 1200, 'height': 720})
        chart.set_title({'name': 'Latency in us'})
        chart.set_x_axis({'name': 'Msg Size'})
        chart.set_y_axis({'name': 'Latency in us'})
        chart.set_style(11)
        worksheet.insert_chart(cluster_row + len(msg_sizes) + 3, 2, chart)

    workbook.close()
    log.info('Latency performance chart saved: %s', excel_file)
