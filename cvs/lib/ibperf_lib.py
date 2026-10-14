'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import logging
import math
import re
import time
import xlsxwriter

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

log = logging.getLogger(__name__)

DMABUF_SUPPORTED = 'supported'
DMABUF_UNSUPPORTED = 'unsupported'
DMABUF_PROBE_FAILED = 'probe_failed'
PERFTEST_DMABUF_BUILD_HINT = 'rebuild perftest with ROCm DMA-BUF support (configure --enable-rocm --enable-rocm-dmabuf)'

# Cached one-time ROCm path probe results for the current process.
_rocm_path_cache = {}
_GPU_VISIBILITY_ENV_VARS = ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
_GPU_ENV_TAG = '__CVS_GPU_ENV__'
_HIP_DEVICE_PROBE_CMD = (
    'grep -HE "^(simd_count|location_id|domain) " /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null; '
    f"printf '{_GPU_ENV_TAG} %s=%s\\n' " + ' '.join(f'{var} "${{{var}-}}"' for var in _GPU_VISIBILITY_ENV_VARS)
)

DEFAULT_BW_TESTS = ['ib_write_bw', 'ib_read_bw', 'ib_send_bw']
DEFAULT_LAT_TESTS = ['ib_write_lat', 'ib_send_lat']
SUPPORTED_BW_TESTS = ('ib_write_bw', 'ib_read_bw', 'ib_send_bw')
SUPPORTED_LAT_TESTS = ('ib_write_lat', 'ib_read_lat', 'ib_send_lat')


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


def get_ibperf_rdma_devices(gpu_nic_dict, bck_nic_dict, gpu_count=8):
    """Mirror the NIC selection in the bandwidth and latency perftest runners."""
    return {
        node: sorted(
            {
                device
                for gpu in range(gpu_count)
                if (device := gpu_nic_dict.get(node, {}).get(f'card{gpu}', {}).get('rdma_dev')) is not None
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


def get_ib_bw_pps(phdl, msg_size, cmd, instance_no=None):
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
    for i in range(1, 10):
        log.debug('BW collection iteration %d for msg_size %s', i, msg_size)
        out_dict = phdl.exec(cmd, print_console=False)
        for node in out_dict.keys():
            res_dict[node] = {}
            pattern = r"{}\s+\d+\s+[0-9\.]+\s+([0-9\.]+)\s+([0-9\.]+)".format(msg_size)
            if re.search(pattern, out_dict[node]):
                match = re.search(pattern, out_dict[node])
                res_dict[node]['bw'] = match.group(1)
                res_dict[node]['pps'] = match.group(2)
                continue
            else:
                log.debug('Node %s: results not ready, sleeping 10s (iteration %d)', node, i)
                time.sleep(10)
            fail_test(
                f'ERROR !!! on node {node} Client did not complete even after max iterations for msg size {msg_size}'
            )
            fail_test(f'ERROR !!! pls check log file for errors on node {node}')
    _log_bw_summary(msg_size, res_dict, instance_no)
    return res_dict


def get_ib_lat_numb(phdl, msg_size, cmd, instance_no=None, result_nodes=None):
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
    for i in range(1, 4):
        log.debug('Latency collection iteration %d for msg_size %s', i, msg_size)
        out_dict = phdl.exec(cmd, print_console=False)
        for node in required_nodes:
            match = re.search(pattern, out_dict.get(node, ''))
            if match:
                res_dict[node] = {
                    't_min': match.group(1),
                    't_max': match.group(2),
                    't_typical': match.group(3),
                    't_avg': match.group(4),
                    't_stdev': match.group(5),
                    't_99_pct': match.group(6),
                    't_99_9_pct': match.group(7),
                }
            else:
                log.debug('Node %s: latency results not ready, sleeping 10s (iteration %d)', node, i)
                time.sleep(10)
    for node in required_nodes:
        if node not in res_dict:
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
    i = 0
    cmd_dict = {}
    phdl.exec('sudo rm -rf /tmp/ib_cmds_file.txt', print_console=False)
    phdl.exec('sudo rm -rf /tmp/ib_perf*', print_console=False)
    phdl.exec('touch /tmp/ib_cmds_file.txt', print_console=False)
    if rocm_path:
        log.debug('Setting LD_LIBRARY_PATH to %s/lib for perftest binaries', rocm_path)
        phdl.exec(
            f'echo "export LD_LIBRARY_PATH={rocm_path}/lib:$LD_LIBRARY_PATH" >> /tmp/ib_cmds_file.txt',
            print_console=False,
        )
    server_addr = None
    for node in bck_nic_dict.keys():
        result_dict[node] = {}
        cmd_dict[node] = []
        # even nodes make it server and odd as clients
        if i % 2 == 0:
            # server
            server_addr = node
            cmd = 'echo "sleep 1" >> /tmp/ib_cmds_file.txt'
            cmd_dict[node].append(cmd)
            inst_count = 0
            port_no = app_port
            for gpu_no in range(0, 8):
                card_no = 'card' + str(gpu_no)
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
            for gpu_no in range(0, 8):
                card_no = 'card' + str(gpu_no)
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
        i = i + 1

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
    phdl.exec('source /tmp/ib_cmds_file.txt', print_console=False)

    time.sleep(20)

    for instance_no in range(0, inst_count):
        try:
            bw_pps_dict = get_ib_bw_pps(phdl, msg_size, f'cat /tmp/ib_perf_{instance_no}_logs', instance_no=instance_no)
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
):
    log.info('%s: msg_size=%s start (%d nodes)', lat_test, msg_size, len(bck_nic_dict))
    if hip_dev_dict is None:
        hip_dev_dict = get_hip_device_dict(phdl, gpu_nic_dict)
    app_port = port_no
    result_dict = {}
    i = 0
    cmd_dict = {}
    phdl.exec('sudo rm -rf /tmp/ib_cmds_file.txt', print_console=False)
    phdl.exec('sudo rm -rf /tmp/ib_perf*', print_console=False)
    phdl.exec('touch /tmp/ib_cmds_file.txt', print_console=False)
    if rocm_path:
        log.debug('Setting LD_LIBRARY_PATH to %s/lib for perftest binaries', rocm_path)
        phdl.exec(
            f'echo "export LD_LIBRARY_PATH={rocm_path}/lib:$LD_LIBRARY_PATH" >> /tmp/ib_cmds_file.txt',
            print_console=False,
        )
    server_addr = None
    for node in bck_nic_dict.keys():
        result_dict[node] = {}
        cmd_dict[node] = []
        # even nodes make it server and odd as clients
        if i % 2 == 0:
            # server
            server_addr = node
            cmd = 'echo "sleep 1" >> /tmp/ib_cmds_file.txt'
            cmd_dict[node].append(cmd)
            inst_count = 0
            port_no = app_port
            for gpu_no in range(0, 8):
                card_no = 'card' + str(gpu_no)
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
            for gpu_no in range(0, 8):
                card_no = 'card' + str(gpu_no)
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
        i = i + 1

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
    phdl.exec('source /tmp/ib_cmds_file.txt', print_console=False)

    # wait for the duration of test
    time.sleep(20)

    result_nodes = list(bck_nic_dict)[1::2] if lat_test == 'ib_read_lat' else None
    for instance_no in range(0, inst_count):
        try:
            lat_dict = get_ib_lat_numb(
                phdl,
                msg_size,
                f'cat /tmp/ib_perf_{instance_no}_logs',
                instance_no=instance_no,
                result_nodes=result_nodes,
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
    gpu_no_list = [0, 1, 2, 3, 4, 5, 6, 7]

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

    for app_name in app_list:
        for qp_count in qp_count_list:
            sheet_name = app_name + "_qp_" + str(qp_count)
            worksheet = workbook.add_worksheet(sheet_name)
            bold = workbook.add_format({"bold": 1})

            heading = "Test {} - BW, MPPS Numbers for {} QPs{}".format(
                app_name, qp_count, _dmabuf_heading_suffix(dmabuf_dict, app_name)
            )
            worksheet.merge_range("A1:T1", heading, merge_format)

            # Init data lists
            d_msg_size_list = []
            d_bw_gpu_0_list = []
            d_bw_gpu_1_list = []
            d_bw_gpu_2_list = []
            d_bw_gpu_3_list = []
            d_bw_gpu_4_list = []
            d_bw_gpu_5_list = []
            d_bw_gpu_6_list = []
            d_bw_gpu_7_list = []
            d_pps_gpu_0_list = []
            d_pps_gpu_1_list = []
            d_pps_gpu_2_list = []
            d_pps_gpu_3_list = []
            d_pps_gpu_4_list = []
            d_pps_gpu_5_list = []
            d_pps_gpu_6_list = []
            d_pps_gpu_7_list = []
            d_pps_total_list = []
            d_bw_total_list = []

            for node in node_list:
                d_msg_size_list.extend(msg_size_list)
                for msg_size in msg_size_list:
                    tot_pps = 0.0
                    tot_bw = 0.0
                    for gpu_no in gpu_no_list:
                        if gpu_no == 0:
                            d_bw_gpu_0_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_0_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 1:
                            d_bw_gpu_1_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_1_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 2:
                            d_bw_gpu_2_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_2_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 3:
                            d_bw_gpu_3_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_3_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 4:
                            d_bw_gpu_4_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_4_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 5:
                            d_bw_gpu_5_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_5_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 6:
                            d_bw_gpu_6_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_6_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                        elif gpu_no == 7:
                            d_bw_gpu_7_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                            d_pps_gpu_7_list.append(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_pps = tot_pps + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['pps'])
                            tot_bw = tot_bw + float(res_dict[app_name][msg_size][qp_count][node][gpu_no]['bw'])
                    d_bw_total_list.append(tot_bw)
                    d_pps_total_list.append(tot_pps)

            split_bw_list = split_list_into_n_chunks(d_bw_total_list, len(node_list))
            split_pps_list = split_list_into_n_chunks(d_pps_total_list, len(node_list))
            split_bw_gpu0_list = split_list_into_n_chunks(d_bw_gpu_0_list, len(node_list))
            split_bw_gpu1_list = split_list_into_n_chunks(d_bw_gpu_1_list, len(node_list))
            split_bw_gpu2_list = split_list_into_n_chunks(d_bw_gpu_2_list, len(node_list))
            split_bw_gpu3_list = split_list_into_n_chunks(d_bw_gpu_3_list, len(node_list))
            split_bw_gpu4_list = split_list_into_n_chunks(d_bw_gpu_4_list, len(node_list))
            split_bw_gpu5_list = split_list_into_n_chunks(d_bw_gpu_5_list, len(node_list))
            split_bw_gpu6_list = split_list_into_n_chunks(d_bw_gpu_6_list, len(node_list))
            split_bw_gpu7_list = split_list_into_n_chunks(d_bw_gpu_7_list, len(node_list))
            split_pps_gpu0_list = split_list_into_n_chunks(d_pps_gpu_0_list, len(node_list))
            split_pps_gpu1_list = split_list_into_n_chunks(d_pps_gpu_1_list, len(node_list))
            split_pps_gpu2_list = split_list_into_n_chunks(d_pps_gpu_2_list, len(node_list))
            split_pps_gpu3_list = split_list_into_n_chunks(d_pps_gpu_3_list, len(node_list))
            split_pps_gpu4_list = split_list_into_n_chunks(d_pps_gpu_4_list, len(node_list))
            split_pps_gpu5_list = split_list_into_n_chunks(d_pps_gpu_5_list, len(node_list))
            split_pps_gpu6_list = split_list_into_n_chunks(d_pps_gpu_6_list, len(node_list))
            split_pps_gpu7_list = split_list_into_n_chunks(d_pps_gpu_7_list, len(node_list))

            avg_bw_data = average_of_lists(split_bw_list)
            avg_pps_data = average_of_lists(split_pps_list)
            avg_bw_gpu0_data = average_of_lists(split_bw_gpu0_list)
            avg_bw_gpu1_data = average_of_lists(split_bw_gpu1_list)
            avg_bw_gpu2_data = average_of_lists(split_bw_gpu2_list)
            avg_bw_gpu3_data = average_of_lists(split_bw_gpu3_list)
            avg_bw_gpu4_data = average_of_lists(split_bw_gpu4_list)
            avg_bw_gpu5_data = average_of_lists(split_bw_gpu5_list)
            avg_bw_gpu6_data = average_of_lists(split_bw_gpu6_list)
            avg_bw_gpu7_data = average_of_lists(split_bw_gpu7_list)

            avg_pps_gpu0_data = average_of_lists(split_pps_gpu0_list)
            avg_pps_gpu1_data = average_of_lists(split_pps_gpu1_list)
            avg_pps_gpu2_data = average_of_lists(split_pps_gpu2_list)
            avg_pps_gpu3_data = average_of_lists(split_pps_gpu3_list)
            avg_pps_gpu4_data = average_of_lists(split_pps_gpu4_list)
            avg_pps_gpu5_data = average_of_lists(split_pps_gpu5_list)
            avg_pps_gpu6_data = average_of_lists(split_pps_gpu6_list)
            avg_pps_gpu7_data = average_of_lists(split_pps_gpu7_list)

            # Merge cols for Node IP
            worksheet.merge_range("A2:A3", "Node IP", bold)
            worksheet.merge_range("B2:B3", "Msg Size", bold)
            # Merge 2 cols for every GPU and Total
            worksheet.merge_range("C2:D2", "Total", bold)
            worksheet.merge_range("E2:F2", "GPU0", bold)
            worksheet.merge_range("G2:H2", "GPU1", bold)
            worksheet.merge_range("I2:J2", "GPU2", bold)
            worksheet.merge_range("K2:L2", "GPU3", bold)
            worksheet.merge_range("M2:N2", "GPU4", bold)
            worksheet.merge_range("O2:P2", "GPU5", bold)
            worksheet.merge_range("Q2:R2", "GPU6", bold)
            worksheet.merge_range("S2:T2", "GPU7", bold)

            headings = []
            for i in range(0, 9):
                headings.append("BW")
                headings.append("PPS")
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
            worksheet.write_column("E4", round_vals(d_bw_gpu_0_list))
            worksheet.write_column("F4", round_vals(d_pps_gpu_0_list))
            worksheet.write_column("G4", round_vals(d_bw_gpu_1_list))
            worksheet.write_column("H4", round_vals(d_pps_gpu_1_list))
            worksheet.write_column("I4", round_vals(d_bw_gpu_2_list))
            worksheet.write_column("J4", round_vals(d_pps_gpu_2_list))
            worksheet.write_column("K4", round_vals(d_bw_gpu_3_list))
            worksheet.write_column("L4", round_vals(d_pps_gpu_3_list))
            worksheet.write_column("M4", round_vals(d_bw_gpu_4_list))
            worksheet.write_column("N4", round_vals(d_pps_gpu_4_list))
            worksheet.write_column("O4", round_vals(d_bw_gpu_5_list))
            worksheet.write_column("P4", round_vals(d_pps_gpu_5_list))
            worksheet.write_column("Q4", round_vals(d_bw_gpu_6_list))
            worksheet.write_column("R4", round_vals(d_pps_gpu_6_list))
            worksheet.write_column("S4", round_vals(d_bw_gpu_7_list))
            worksheet.write_column("T4", round_vals(d_pps_gpu_7_list))

            x_chart_index = len(d_msg_size_list) + 4

            worksheet.write_column(f"B{x_chart_index}", msg_size_list)
            worksheet.write_column(f"C{x_chart_index}", round_vals(avg_bw_data))
            worksheet.write_column(f"D{x_chart_index}", round_vals(avg_pps_data))
            worksheet.write_column(f"E{x_chart_index}", round_vals(avg_bw_gpu0_data))
            worksheet.write_column(f"F{x_chart_index}", round_vals(avg_pps_gpu0_data))
            worksheet.write_column(f"G{x_chart_index}", round_vals(avg_bw_gpu1_data))
            worksheet.write_column(f"H{x_chart_index}", round_vals(avg_pps_gpu1_data))
            worksheet.write_column(f"I{x_chart_index}", round_vals(avg_bw_gpu2_data))
            worksheet.write_column(f"J{x_chart_index}", round_vals(avg_pps_gpu2_data))
            worksheet.write_column(f"K{x_chart_index}", round_vals(avg_bw_gpu3_data))
            worksheet.write_column(f"L{x_chart_index}", round_vals(avg_pps_gpu3_data))
            worksheet.write_column(f"M{x_chart_index}", round_vals(avg_bw_gpu4_data))
            worksheet.write_column(f"N{x_chart_index}", round_vals(avg_pps_gpu4_data))
            worksheet.write_column(f"O{x_chart_index}", round_vals(avg_bw_gpu5_data))
            worksheet.write_column(f"P{x_chart_index}", round_vals(avg_pps_gpu5_data))
            worksheet.write_column(f"Q{x_chart_index}", round_vals(avg_bw_gpu6_data))
            worksheet.write_column(f"R{x_chart_index}", round_vals(avg_pps_gpu6_data))
            worksheet.write_column(f"S{x_chart_index}", round_vals(avg_bw_gpu7_data))
            worksheet.write_column(f"T{x_chart_index}", round_vals(avg_pps_gpu7_data))

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
        sheet_name = app_name + '_lat'
        worksheet = workbook.add_worksheet(sheet_name)
        worksheet.merge_range(
            'A1:Z1', f'Test {app_name} - latency results{_dmabuf_heading_suffix(dmabuf_dict, app_name)}', merge_format
        )
        worksheet.merge_range('A2:A3', 'Node IP', bold)
        worksheet.merge_range('B2:B3', 'Msg Size', bold)
        worksheet.merge_range('C2:G2', 'Node Avg ', bold)
        for gpu_no in range(8):
            first_col = 7 + gpu_no * len(metrics)
            worksheet.merge_range(1, first_col, 1, first_col + len(metrics) - 1, f'GPU{gpu_no}', bold)
        worksheet.write_row(2, 2, ['Tmin', 'Tmax', 'TAvg', 'TStdev', 'T99pct'] * 9, bold)

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
            for gpu_no in range(8):
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
