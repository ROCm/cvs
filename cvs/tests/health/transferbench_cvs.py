'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import json
import re
import shlex

from cvs.lib.env_lib import build_env_prefix
from cvs.lib.utils_lib import *
from cvs.lib import globals

log = globals.log

_ENV_VAR_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_LINUX_ID_LIST_PART_RE = re.compile(r'^(\d+)(?:-(\d+))?$')
_DETECT_NUM_CPU_DEVICES_CMD = (
    "c=0; "
    "for f in /sys/devices/system/node/node*/cpulist; do "
    "[ -f \"$f\" ] || continue; "
    "grep -q '[0-9]' \"$f\" || continue; "
    "c=$((c+1)); "
    "done; "
    "if [ \"$c\" -gt 0 ]; then printf '%s\\n' \"$c\"; "
    "else awk '/^Mems_allowed_list:/ {print $2}' /proc/self/status; fi"
)
_TB_NUMA_ABORT_HINT = (
    "If the run aborted enumerating sparse NUMA CPU endpoints, set "
    "transferbench.num_cpu_devices or transferbench.env.NUM_CPU_DEVICES to the "
    "populated CPU NUMA count (Helios-R / Venice: 2)."
)


def count_linux_id_list(spec):
    """Count IDs in a Linux cpuset-style list such as ``0-1,4,6-8``."""
    text = str(spec or '').strip()
    if not text:
        raise ValueError('linux id list is empty')
    total = 0
    for part in text.split(','):
        part = part.strip()
        match = _LINUX_ID_LIST_PART_RE.fullmatch(part)
        if not match:
            raise ValueError(f'invalid linux id list: {spec!r}')
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            raise ValueError(f'invalid linux id list: {spec!r}')
        total += end - start + 1
    return total


def parse_cpu_device_count(value):
    """Accept an int, a numeric string, or a Linux cpuset list (``0-1`` -> 2)."""
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        raise ValueError(f'invalid CPU device count: {value!r}')
    if isinstance(value, int):
        if value < 1:
            raise ValueError('num_cpu_devices must be >= 1')
        return value
    text = str(value).strip()
    if re.fullmatch(r'\d+', text):
        parsed = int(text)
        if parsed < 1:
            raise ValueError('num_cpu_devices must be >= 1')
        return parsed
    return count_linux_id_list(text)


def _validate_tb_env(env_vars):
    if env_vars is None:
        return {}
    if not isinstance(env_vars, dict):
        raise ValueError('transferbench.env must be a mapping of environment variable names to values')
    normalized = {}
    for key, value in env_vars.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.fullmatch(key):
            raise ValueError(f'Invalid environment variable name in transferbench.env: {key!r}')
        if value is None:
            raise ValueError(f'transferbench.env[{key!r}] must not be null')
        normalized[key] = str(value)
    return normalized


def resolve_configured_tb_env(config_dict):
    """Return TransferBench env from health config (no node probing)."""
    extra = _validate_tb_env(config_dict.get('env') or config_dict.get('env_vars'))
    raw_count = config_dict.get('num_cpu_devices')
    if raw_count not in (None, '', '<changeme>'):
        extra['NUM_CPU_DEVICES'] = str(parse_cpu_device_count(raw_count))
    return extra


def detect_num_cpu_devices(orch):
    """Probe populated CPU NUMA nodes; return a count only when all hosts agree."""
    out_dict = orch.exec(_DETECT_NUM_CPU_DEVICES_CMD, timeout=30)
    counts = {}
    for node, output in (out_dict or {}).items():
        text = output if isinstance(output, str) else (output or {}).get('output', '')
        line = next((ln.strip() for ln in str(text).splitlines() if ln.strip()), '')
        try:
            counts[node] = parse_cpu_device_count(line)
        except (TypeError, ValueError):
            log.warning('Could not parse CPU NUMA count from %s: %r', node, line)
    values = {count for count in counts.values() if count}
    if len(values) == 1:
        return next(iter(values))
    if len(values) > 1:
        log.warning('NUMA CPU device counts disagree across nodes: %s', counts)
    return None


def resolve_runtime_tb_env(orch, config_dict):
    """Merge config env with auto-detected NUM_CPU_DEVICES when unset."""
    extra = resolve_configured_tb_env(config_dict)
    auto = config_dict.get('auto_num_cpu_devices', True)
    if isinstance(auto, str):
        auto = auto.strip().lower() in ('1', 'true', 'yes', 'on')
    if 'NUM_CPU_DEVICES' not in extra and auto:
        detected = detect_num_cpu_devices(orch)
        if detected:
            log.info('Auto-scoped TransferBench NUM_CPU_DEVICES=%s from populated CPU NUMA nodes', detected)
            extra['NUM_CPU_DEVICES'] = str(detected)
    return extra


def build_transferbench_command(path, rocm_path, preset, extra_env=None):
    """Build a ``sudo bash -c`` TransferBench invocation with env inside the inner shell.

    Cluster/PSSH ``env_vars`` do not survive these ``sudo bash -c`` wrappers, so
    NUM_CPU_DEVICES and LD_LIBRARY_PATH must be exported in the inner script.
    """
    env = {'LD_LIBRARY_PATH': f'{rocm_path}/lib:$LD_LIBRARY_PATH'}
    if extra_env:
        env.update(extra_env)
    exports = build_env_prefix(env)
    inner = f'{exports} && echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" && {path}/TransferBench {preset}'
    return f'sudo bash -c {shlex.quote(inner)}'


def run_transferbench(orch, config_dict, preset, timeout, extra_env=None):
    """Resolve ROCm/env and execute one TransferBench preset on all orch hosts."""
    path = config_dict['path']
    rocm_path = detect_rocm_path(orch, config_dict.get('rocm_path', ''))
    env = resolve_runtime_tb_env(orch, config_dict)
    if extra_env:
        env.update(extra_env)
    cmd = build_transferbench_command(path, rocm_path, preset, env)
    log.info('TransferBench command: %s', cmd)
    return orch.exec(cmd, timeout=timeout)


def _tb_output_excerpt(out, tail=6000):
    if len(out) <= tail:
        return out
    return f"...[truncated {len(out) - tail} chars]...\n{out[-tail:]}"


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


# Importing the cluster and cofig files to script to access node, switch, test config params
@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['transferbench']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)

    log.info("%s", config_dict)
    return config_dict


def detect_rocm_path(orch, config_rocm_path):
    """
    Detect the ROCm installation path, supporting both old (/opt/rocm) and new (/opt/rocm/core-X.Y) layouts.
    Args:
        orch: Orchestrator instance
        config_rocm_path (str): Configured ROCm path from config file (empty string for auto-detect)
    Returns:
        str: Detected ROCm path
    """
    # If rocm_path is explicitly configured, validate and use it. The validation
    # is a glob+pipe pipeline (does {path}/lib exist AND does it contain a
    # libamdhip64.so*?), wrapped in explicit `bash -c` because
    # ContainerOrchestrator's docker-exec transport does not spawn a shell.
    if config_rocm_path and config_rocm_path != '<changeme>':
        out_dict = orch.exec(
            f"bash -c 'test -d {config_rocm_path}/lib"
            f" && ls {config_rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1'"
        )
        for node, output in out_dict.items():
            if output.strip() and 'libamdhip64.so' in output:
                log.info(f'Using configured ROCm path: {config_rocm_path} (validated)')
                return config_rocm_path
            else:
                log.warning(
                    f'Configured ROCm path {config_rocm_path} does not contain required libraries, will auto-detect'
                )

    # Auto-detect ROCm path
    log.info('Auto-detecting ROCm path...')

    # Try new ROCm 7.x structure first (/opt/rocm/core-X.Y). Glob+pipe wrapped
    # in explicit `bash -c`; same rationale as the configured-path probe above.
    out_dict = orch.exec("bash -c 'ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1'")
    for node, output in out_dict.items():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            # Validate it has the library
            validate_dict = orch.exec(
                f"bash -c 'test -d {rocm_path}/lib && ls {rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1'"
            )
            for _, lib_output in validate_dict.items():
                if lib_output.strip() and 'libamdhip64.so' in lib_output:
                    log.info(f'Detected ROCm path (new layout): {rocm_path}')
                    return rocm_path

    # Fall back to legacy /opt/rocm
    out_dict = orch.exec("bash -c 'test -d /opt/rocm/lib && ls /opt/rocm/lib/libamdhip64.so* 2>/dev/null | head -1'")
    for node, output in out_dict.items():
        if output.strip() and 'libamdhip64.so' in output:
            log.info('Detected ROCm path (legacy layout): /opt/rocm')
            return '/opt/rocm'

    # If nothing found, default to /opt/rocm (will fail gracefully later)
    log.warning('Could not detect ROCm path with required libraries, defaulting to /opt/rocm')
    return '/opt/rocm'


def parse_tb_a2a_bw(out_dict, exp_dict):
    for node in out_dict.keys():
        log.info("%s", exp_dict)
        rtotal_list = re.findall(
            r'(?:│\s+)?RTotal\s+(?:│\s+)?([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s*',
            out_dict[node],
        )
        if not rtotal_list:
            fail_test(f"RTotal row not found in TransferBench a2a output on node {node}")
            continue
        for gpu_bw in list(rtotal_list[0]):
            if float(gpu_bw) < float(exp_dict['gpu_to_gpu_a2a_rtotal']):
                fail_test(
                    f"Actual GPU a2a bandwidth {gpu_bw} in transferbench a2a test lower than expected {exp_dict['gpu_to_gpu_a2a_rtotal']} on node {node}"
                )


def parse_tb_p2p_bw(out_dict, exp_dict):
    for node in out_dict.keys():
        out = out_dict[node]
        match = re.search(r'Averages\s+\(During\s+UniDir\):\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+([0-9\.]+)', out, re.I)
        if not match:
            fail_test(
                f"TransferBench p2p UniDir averages not found on node {node}. {_TB_NUMA_ABORT_HINT} "
                f"Output: {_tb_output_excerpt(out)}"
            )
            continue
        avg_unidir = float(match.group(1))
        if float(avg_unidir) < float(exp_dict['avg_gpu_to_gpu_p2p_unidir_bw']):
            fail_test(
                f"Actual Avg UniDir GPU to GPU bandwidth {avg_unidir} is less than expected {exp_dict['avg_gpu_to_gpu_p2p_unidir_bw']} on node {node}"
            )
        match = re.search(r'Averages\s+\(During\s+BiDir\):\s+[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+([0-9\.]+)', out, re.I)
        if not match:
            fail_test(
                f"TransferBench p2p BiDir averages not found on node {node}. {_TB_NUMA_ABORT_HINT} "
                f"Output: {_tb_output_excerpt(out)}"
            )
            continue
        avg_bidir = float(match.group(1))
        if float(avg_bidir) < float(exp_dict['avg_gpu_to_gpu_p2p_bidir_bw']):
            fail_test(
                f"Actual Avg BiDir GPU to GPU bandwidth {avg_bidir} is less than expected {exp_dict['avg_gpu_to_gpu_p2p_bidir_bw']} on node {node}"
            )


def parse_tb_scaling_bw(out_dict, exp_dict):
    for node in out_dict.keys():
        log.info(f"^^^^^ {out_dict[node]} ^^^^^^")
        # Best summary row: skip CPU00/CPU01 columns, capture GPU00 peak bandwidth.
        # CU counts in parentheses may be padded (e.g. "(  8)" vs "( 32)").
        match = re.search(
            r'(?m)^[ \t]*Best\s+'
            r'(?:[0-9\.]+\(\s*[0-9]+\)\s+){2}'
            r'([0-9\.]+)',
            out_dict[node],
        )
        if not match:
            out = out_dict[node]
            tail = 6000
            if len(out) > tail:
                out = f"...[truncated {len(out) - tail} chars; Best row is usually near the end]...\n{out[-tail:]}"
            fail_test(f"TransferBench scaling Best row GPU00 bandwidth not found on node {node}. Output: {out}")
            continue
        gpu0_bw = float(match.group(1))
        if float(gpu0_bw) < float(exp_dict['best_gpu0_bw']):
            fail_test(
                f"Actual Best BW from GPU0 in scaling test is lower than expected {exp_dict['best_gpu0_bw']} on node {node}"
            )


def parse_tb_schmoo_bw(out_dict, exp_dict):
    for node in out_dict.keys():
        # Line-anchored schmoo data row: leading indent + optional |/│, then whole 32 (not 132/320).
        # Use (?<![0-9])32(?![0-9]) instead of \b32\b so indented lines (spaces are non-word) still match.
        match = re.search(
            r'(?m)^[ \t]*(?:[|│][ \t]*)*(?<![0-9])32(?![0-9])(?:[ \t]*[|│])?[ \t]+'
            r'([0-9\.]+)[ \t]+([0-9\.]+)[ \t]+([0-9\.]+)[ \t]+([0-9\.]+)[ \t]+([0-9\.]+)[ \t]+([0-9\.]+)',
            out_dict[node],
        )
        if not match:
            out = out_dict[node]
            tail = 6000
            if len(out) > tail:
                out = f"...[truncated {len(out_dict[node]) - tail} chars; schmoo table is usually near the end]...\n{out_dict[node][-tail:]}"
            fail_test(f"TransferBench schmoo row for 32 CU not found on node {node}. Output: {out}")
            continue
        local_read = float(match.group(1))
        local_write = float(match.group(2))
        local_copy = float(match.group(3))
        remote_read = float(match.group(4))
        remote_write = float(match.group(5))
        remote_copy = float(match.group(6))
        if float(local_read) < float(exp_dict['32_cu_local_read']):
            fail_test(
                f"Actual local read for 32 CU {local_read} is less than expected {exp_dict['32_cu_local_read']} for node {node}"
            )
        if float(local_write) < float(exp_dict['32_cu_local_write']):
            fail_test(
                f"Actual local write for 32 CU {local_write} is less than expected {exp_dict['32_cu_local_write']} for node {node}"
            )
        if float(local_copy) < float(exp_dict['32_cu_local_copy']):
            fail_test(
                f"Actual local copy for 32 CU {local_copy} is less than expected {exp_dict['32_cu_local_copy']} for node {node}"
            )
        if float(remote_read) < float(exp_dict['32_cu_rem_read']):
            fail_test(
                f"Actual remote read for 32 CU {remote_read} is less than expected {exp_dict['32_cu_rem_read']} for node {node}"
            )
        if float(remote_write) < float(exp_dict['32_cu_rem_write']):
            fail_test(
                f"Actual remote write for 32 CU {remote_write} is less than expected {exp_dict['32_cu_rem_write']} for node {node}"
            )
        if float(remote_copy) < float(exp_dict['32_cu_rem_copy']):
            fail_test(
                f"Actual remote copy for 32 CU {remote_copy} is less than expected {exp_dict['32_cu_rem_copy']} for node {node}"
            )


def test_transfer_bench_a2a(
    orch,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run Transferbench a2a')
    preset = config_dict.get('a2a_preset') or 'a2a'
    out_dict = run_transferbench(orch, config_dict, preset, timeout=(60 * 5))
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    parse_tb_a2a_bw(out_dict, config_dict['results'])
    scan_test_results(out_dict)
    update_test_result()


def test_transfer_bench_p2p(
    orch,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run Transferbench p2p')
    preset = config_dict.get('p2p_preset') or 'p2p'
    out_dict = run_transferbench(orch, config_dict, preset, timeout=(60 * 5))
    print_test_output(log, out_dict)
    parse_tb_p2p_bw(out_dict, config_dict['results'])
    scan_test_results(out_dict)
    update_test_result()


def test_transfer_bench_healthcheck(
    orch,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run TransferBench healthcheck')
    preset = config_dict.get('healthcheck_preset') or 'healthcheck'
    out_dict = run_transferbench(orch, config_dict, preset, timeout=(60 * 3))
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    update_test_result()


def test_transfer_bench_a2asweep(
    orch,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run TransferBench a2asweep')
    preset = config_dict.get('a2asweep_preset') or 'a2asweep'
    out_dict = run_transferbench(orch, config_dict, preset, timeout=(60 * 10))
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    update_test_result()


def test_transfer_bench_scaling(
    orch,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run TransferBench scaling')
    preset = config_dict.get('scaling_preset') or 'scaling'
    out_dict = run_transferbench(
        orch,
        config_dict,
        preset,
        timeout=(60 * 10),
        extra_env={'GFX_TEMPORAL': '3', 'GFX_UNROLL': '32'},
    )
    print_test_output(log, out_dict)
    parse_tb_scaling_bw(out_dict, config_dict['results'])
    scan_test_results(out_dict)
    update_test_result()


def test_transfer_bench_schmoo(
    orch,
    config_dict,
):
    globals.error_list = []
    log.info('Testcase Run TransferBench schmoo')
    preset = config_dict.get('schmoo_preset') or 'schmoo'
    out_dict = run_transferbench(
        orch,
        config_dict,
        preset,
        timeout=(60 * 5),
        extra_env={'GFX_UNROLL': '32', 'SWEEP_MIN': '32'},
    )
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    parse_tb_schmoo_bw(out_dict, config_dict['results'])
    update_test_result()
