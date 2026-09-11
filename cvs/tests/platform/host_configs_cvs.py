'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import json

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib.rocm_plib import *
from cvs.lib import globals, linux_utils
from cvs.lib.platform.host_inventory import (
    normalize_output,
    parse_gpu_count,
    parse_iommu,
    parse_kernel_version,
    parse_numa_balancing,
    parse_online_memory,
    parse_os_release,
    parse_pci_acs,
    parse_pci_realloc,
    parse_pcie_link,
    parse_rocm_version,
    record_gpu_firmware,
    record_indexed_node_facts,
    record_node_facts,
)

log = globals.log


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Return the path to the cluster configuration file provided via pytest CLI.

    Expects pytest to be invoked with:
      --cluster_file <path>
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Return the path to the test configuration file provided via pytest CLI.

    Expects pytest to be invoked with:
      --config_file <path>
    """
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and return the cluster definition as a dictionary.

    Behavior:
      - Opens the JSON file specified by cluster_file.
      - Logs and returns the parsed content (assumed to include node_dict, username, priv_key_file).

    Notes:
      - Ensure the JSON schema matches what downstream fixtures/functions expect.
    """
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    """
    Load and return the host-level test configuration sub-dictionary.

    Behavior:
      - Opens the JSON file specified by config_file.
      - Extracts and returns the 'host' sub-dictionary (config values used by tests).

    Notes:
      - The top-level JSON is expected to include a 'host' key.
      - Adjust if your configuration schema changes.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['host']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="session")
def cvs_results_dict():
    return {"nodes": {}, "firmware": {}}


@pytest.fixture(scope="session")
def host_inventory_metadata(cvs_results_dict):
    return cvs_results_dict


# Main Test cases start from here ..


def test_check_os_release(
    orch,
    config_dict,
    cvs_results_dict,
):
    """
    Validate that each node's OS release matches the expected version.

    This test:
      - Reads the expected OS version from config_dict['os_version'].
      - Executes 'cat /etc/os-release' on all nodes via phdl.
      - Fails the test if any node's /etc/os-release content does not contain
        the expected version string.
      - Extracts and reports the actual detected version (best-effort) on failure.
      - Calls update_test_result() at the end to report pass/fail.

    Args:
        phdl: Remote execution handle. Expected to provide exec(cmd: str) -> Dict[node, str].
        config_dict: Configuration dict containing 'os_version' (string to search for).

    Notes:
        - The regex used to extract actual version assumes a line like VERSION="..."
          and may need adjustment for distro-specific variations.
        - globals.error_list is reset at test start; fail_test() should append errors.
    """
    globals.error_list = []  # Reset error accumulator before running this test
    log.info('Testcase check OS Version')
    os_version = config_dict['os_version']  # Expected version substring/pattern
    out_dict = orch.all.exec('cat /etc/os-release')
    record_node_facts(cvs_results_dict, "os_release", out_dict, parse_os_release)
    for node in out_dict.keys():
        # If expected version is not present, extract the actual version and fail
        if not re.search(f'{os_version}', out_dict[node], re.I):
            match = re.search('VERSION="(([0-9\.\-\_A-Z]+)\s+)', out_dict[node], re.I)
            actual_ver = match.group(1)
            fail_test(f'Installed OS Version {actual_ver} not matching expected version {os_version} on node {node}')
    # Consolidate and record the test result
    update_test_result()


def test_check_kernel_version(orch, config_dict, cvs_results_dict):
    """
    Validate that each node's kernel version matches the expected version.

    This test:
      - Reads the expected kernel version from config_dict['kernel_version'].
      - Executes 'uname -a' on all nodes via phdl.
      - Fails the test if the output does not include the expected kernel version.
      - Extracts and reports the actual detected kernel version (best-effort) on failure.
      - Calls update_test_result() at the end to report pass/fail.

    Args:
        phdl: Remote execution handle. Expected to provide exec(cmd: str) -> Dict[node, str].
        config_dict: Configuration dict containing 'kernel_version' (string to search for).

    Notes:
        - The extraction regex targets versions ending with 'generic' (Ubuntu-style).
          Adjust for other kernel package naming conventions if needed.
        - globals.error_list is reset at test start; fail_test() should append errors.
    """

    globals.error_list = []
    log.info('Testcase check Kernel Version')
    kernel_version = config_dict['kernel_version']
    out_dict = orch.all.exec('uname -a')
    record_node_facts(cvs_results_dict, "kernel", out_dict, parse_kernel_version)
    for node in out_dict.keys():
        # If expected version is not present, extract the actual version and fail
        if not re.search(f'{kernel_version}', out_dict[node], re.I):
            match = re.search('([0-9\.\-\_]+generic)', out_dict[node], re.I)
            actual_ver = match.group(1)
            fail_test(
                f'Installed Kernel Version {actual_ver} not matching expected version {kernel_version} on node {node}'
            )
    # Consolidate and record the test result
    update_test_result()


def test_check_bios_version(orch, config_dict, cvs_results_dict):
    """
    Verify that each node's BIOS/firmware version matches the expected value.

    This test:
      - Reads the expected BIOS version from config_dict['bios_version'].
      - Executes 'sudo dmidecode -s bios-version' on all nodes via orch.all.
      - Fails the test for any node whose output does not contain the expected version.
      - Attempts to extract and report the actual BIOS version when a mismatch is found.
      - Calls update_test_result() at the end to record pass/fail.

    Args:
        phdl: Remote execution handle with exec(cmd: str) -> Dict[node, str].
        config_dict: Configuration containing:
            - 'bios_version': Expected BIOS/firmware version string (substring/pattern).

    Notes:
        - globals.error_list is reset at test start; fail_test() should record failures there.
    """
    globals.error_list = []
    log.info('Testcase check BIOS Version')
    bios_version = config_dict['bios_version']
    out_dict = orch.all.exec('sudo dmidecode -s bios-version')
    record_node_facts(cvs_results_dict, "bios", out_dict, normalize_output)
    for node in out_dict.keys():
        if not re.search(f'{bios_version}', out_dict[node], re.I):
            match = re.search('([a-z0-9\_\.\-]+)', out_dict[node], re.I)
            act_bios_ver = match.group(1)
            fail_test(
                f'Installed BIOS Version {act_bios_ver} not matching expected version {bios_version} on node {node}'
            )
    update_test_result()


def test_check_rocm_version(orch, config_dict, cvs_results_dict):
    """
    Verify that each node's ROCm version matches the expected value.

    This test:
      - Reads the expected ROCm version from config_dict['rocm_version'].
      - Executes 'amd-smi version' on all nodes via phdl.
      - Fails the test for any node whose output does not contain the expected version.
      - Attempts to extract and report the actual ROCm version from the tool output.
      - Calls update_test_result() at the end to record pass/fail.

    Args:
        phdl: Remote execution handle with exec(cmd: str) -> Dict[node, str].
        config_dict: Configuration containing:
            - 'rocm_version': Expected ROCm version string (substring/pattern).

    Notes:
        - The extraction regex specifically looks for a line like 'ROCm version: X.Y.Z'.
          If amd-smi?s format differs across versions, the regex may require adjustment.
        - globals.error_list is reset at test start; fail_test() should record failures there.
    """

    globals.error_list = []
    log.info('Testcase check rocm version')
    rocm_version = config_dict['rocm_version']
    out_dict = orch.all.exec('amd-smi version')
    record_node_facts(cvs_results_dict, "rocm", out_dict, parse_rocm_version)
    for node in out_dict.keys():
        if not re.search(f'{rocm_version}', out_dict[node], re.I):
            match = re.search('ROCm version:\s+([0-9\.]+)', out_dict[node], re.I)
            actual_rocm_version = match.group(1)
            fail_test(
                f'Installed rocm version {actual_rocm_version} not matching expected version {rocm_version} on node {node}'
            )
    update_test_result()


def test_check_gpu_fw_version(orch, config_dict, cvs_results_dict):
    """
    Validate GPU firmware versions on each node against expected versions.

    This test:
      - Reads expected firmware versions from config_dict['fw_dict'] as a mapping of
        {<fw_id>: <expected_version>}.
      - Uses get_amd_smi_fw_dict(phdl) to collect per-node GPU firmware data with the
        following assumed structure:
          {
            "<node>": [
              {
                "gpu": "<gpu_index_or_id>",
                "fw_list": [
                  {"fw_id": "<firmware_key>", "fw_version": "<version_str>"},
                  ...
                ]
              },
              ...
            ]
          }
      - Compares each reported firmware version with the expected version for the matching fw_id.
      - Calls fail_test() if any mismatch is detected, then update_test_result() to record status.

    Args:
        phdl: Remote execution/transport handle used by get_amd_smi_fw_dict.
        config_dict: Configuration dict containing:
            - fw_dict (dict): Expected firmware versions keyed by firmware identifier.

    Notes:
        - globals.error_list is reset at test start; fail_test() should append errors there.
        - Assumes fw_id keys in fw_list appear in config_dict['fw_dict'].
    """

    globals.error_list = []
    log.info('Testcase check GPU Firmware versions')
    fw_dict = config_dict['fw_dict']
    out_dict = get_amd_smi_fw_dict(orch.all)
    record_gpu_firmware(cvs_results_dict, out_dict)
    for node in out_dict.keys():
        for gpu_dict in out_dict[node]:
            gpu_no = gpu_dict['gpu']
            for fw_list_dict in gpu_dict['fw_list']:
                fw_key = fw_list_dict['fw_id']
                if fw_list_dict['fw_version'] != fw_dict[fw_key]:
                    fail_test(
                        f"For Firmware {fw_key} actual FW version {fw_list_dict['fw_version']} for gpu {gpu_no} on node {node} is not matching expected FW version {fw_dict[fw_key]}"
                    )
    update_test_result()


def test_check_pci_realloc(orch, config_dict, cvs_results_dict):
    """
    Verify that the kernel command line contains the expected PCI realloc flag.

    This test:
      - Reads the desired PCI realloc setting from config_dict['pci_realloc'] (e.g., 'off' or 'on').
      - Executes 'cat /proc/cmdline' on all nodes via phdl.
      - Ensures 'pci=realloc=<value>' is present; fails if not found.
      - Calls update_test_result() to record pass/fail.

    Args:
        phdl: Remote execution handle capable of exec(cmd: str) -> Dict[node, str].
        config_dict: Configuration dict containing:
            - pci_realloc (str): Expected realloc value (e.g., "off", "on").

    Notes:
        - globals.error_list is reset at test start; fail_test() should record any failures.
    """
    globals.error_list = []
    log.info('Testcase check pci realloc')
    pci_realloc = config_dict['pci_realloc']
    out_dict = orch.all.exec('cat /proc/cmdline')
    record_node_facts(cvs_results_dict, "pci_realloc", out_dict, parse_pci_realloc)
    for node in out_dict.keys():
        if not re.search(f'pci=realloc={pci_realloc}', out_dict[node], re.I):
            fail_test(f'PCI realloc flag not set to {pci_realloc} on node {node}')
    update_test_result()


def test_check_iommu_pt(orch, config_dict, cvs_results_dict):
    """
    Verify that IOMMU is configured in pass-through mode (iommu=pt) on all nodes.

    This test:
      - Executes 'cat /proc/cmdline' on all nodes via phdl.
      - Ensures 'iommu=pt' is present on the kernel command line; fails if not found.
      - Calls update_test_result() to record pass/fail.

    Args:
        phdl: Remote execution handle capable of exec(cmd: str) -> Dict[node, str].
        config_dict: Unused in this test (kept for consistent test function signature).

    Notes:
        - globals.error_list is reset at test start; fail_test() should record failures.
    """

    globals.error_list = []
    log.info('Testcase check IOMMU PT')
    out_dict = orch.all.exec('cat /proc/cmdline')
    record_node_facts(cvs_results_dict, "iommu", out_dict, parse_iommu)
    for node in out_dict.keys():
        if not re.search('iommu=pt', out_dict[node], re.I):
            fail_test(f'IOMMU not set to pt on node {node}')
    update_test_result()


def test_check_numa_balancing(orch, config_dict, cvs_results_dict):
    """
    Verify that automatic NUMA balancing is disabled across all nodes.

    This test:
      - Runs 'sudo sysctl kernel.numa_balancing' on each node via orch.all.
      - Checks that the reported value is 0 (disabled). Accepts either '=0' or '= 0'.
      - Records a failure if any node does not report a disabled state.
      - Calls update_test_result() at the end to record pass/fail.

    Args:
        phdl: Remote execution handle. Expected to provide exec(cmd: str) -> Dict[node, str].
        config_dict: Included for consistency with other tests (not used here).

    Notes:
        - globals.error_list is reset at test start; fail_test() should append errors.
        - This test relies on sysctl output format; if localized/altered, the regex may need adjustment.
    """
    globals.error_list = []
    log.info('Testcase check NUMA balancing')
    out_dict = orch.all.exec('sudo sysctl kernel.numa_balancing')
    record_node_facts(cvs_results_dict, "numa_balancing", out_dict, parse_numa_balancing)
    for node in out_dict.keys():
        if not re.search('=0|= 0', out_dict[node], re.I):
            fail_test(f'NUMA balancing not disabled on node {node}')
    update_test_result()


def test_check_online_memory(orch, config_dict, cvs_results_dict):
    """
    Validate that the total online memory matches the expected value on each node.

    This test:
      - Reads the expected value from config_dict['online_memory'] (e.g., "512G").
      - Runs 'lsmem' on each node via phdl and searches for the "Total online memory" line.
      - Compares the actual reported value to the expected; fails if there is a mismatch.
      - Calls update_test_result() at the end to record pass/fail.

    Args:
        phdl: Remote execution handle. Expected to provide exec(cmd: str) -> Dict[node, str].
        config_dict: Must include:
            - 'online_memory' (str): Expected memory string as reported by lsmem (units included).

    Notes:
        - globals.error_list is reset at test start; fail_test() should append errors.
        - The regex extracts "Total online memory: <value>" as a single token; ensure units match
          (e.g., MB/GB/GiB) with what lsmem reports on your systems.
    """
    globals.error_list = []
    log.info('Testcase check online memory')
    online_mem = config_dict['online_memory']
    out_dict = orch.all.exec('lsmem')
    record_node_facts(cvs_results_dict, "online_memory", out_dict, parse_online_memory)
    for node in out_dict.keys():
        if not re.search(f'Total online memory:\s+{online_mem}', out_dict[node], re.I):
            match = re.search('Total online memory:\s+([0-9\.A-Za-z]+)', out_dict[node])
            actual_mem = match.group(1)
            fail_test(f'Total online memory {actual_mem} not matching expected online mem {online_mem} on node {node}')
    update_test_result()


def test_check_pci_accelerators(orch, config_dict, cvs_results_dict):
    """
    Confirm that the expected number of GPUs (accelerators) are enumerated on PCIe.

    This test:
      - Reads the expected GPU count from config_dict['gpu_count'].
      - Executes 'lspci | grep "accelerators" --color=never' on each node via phdl.
      - Counts the number of lines matching 'accelerators: Advanced' and compares to expected.
      - Calls update_test_result() at the end to record pass/fail.

    Args:
        phdl: Remote execution handle. Expected to provide exec(cmd: str) -> Dict[node, str].
        config_dict: Must include:
            - 'gpu_count' (int or str): Expected number of accelerators reported by lspci.

    Notes:
        - globals.error_list is reset at test start; fail_test() should append errors.
        - The pattern 'accelerators: Advanced' is vendor/driver specific; adjust the regex if
          your platform reports accelerators differently in lspci output.
    """

    globals.error_list = []
    log.info('Testcase check online GPUs in pcie')
    gpu_count = config_dict['gpu_count']
    out_dict = orch.all.exec('lspci | grep "accelerators" --color=never')
    record_node_facts(cvs_results_dict, "gpu_count", out_dict, parse_gpu_count)
    for node in out_dict.keys():
        match_list = re.findall('accelerators:\s+Advanced', out_dict[node], re.I)
        actual_gpu_count = len(match_list)
        if int(gpu_count) != actual_gpu_count:
            fail_test(
                f'Expected GPU count in PCI {gpu_count} not matching actual GPU count {actual_gpu_count} on node {node}'
            )
    update_test_result()


def test_check_gpu_pcie_speed_width(orch, config_dict, cvs_results_dict):
    """
    Verify PCIe link speed and width for each GPU on all nodes.

    This test:
      - Reads expected PCIe speed and width from config_dict:
          - gpu_pcie_speed (e.g., "32" for 32 GT/s)
          - gpu_pcie_width (e.g., "16" for x16)
      - Uses get_gpu_pcie_bus_dict(phdl) to collect GPU PCI bus IDs per node.
      - Assumes a homogeneous cluster (same set/order of GPUs on every node) and
        builds a command list per card index to run in parallel across nodes:
          sudo lspci -vvv -s <bus> | grep "LnkSta:"
      - Checks each node's LnkSta line for:
          - Speed <gpu_pcie_speed>GT
          - Width x<gpu_pcie_width>
          - Not in a downgrade state
      - Calls update_test_result() at the end to record pass/fail.

    Args:
      phdl: Remote execution handle; must provide:
            - exec_cmd_list(list[str]) -> dict[node, str]
      config_dict: Must include:
            - 'gpu_pcie_speed': expected GT/s as string (e.g., "32")
            - 'gpu_pcie_width': expected width as string (e.g., "16")

    Notes:
      - globals.error_list is reset at test start; fail_test() should accumulate failures.
      - Assumes get_gpu_pcie_bus_dict returns:
          { node: { card_index: {"PCI Bus": "<domain:bus:slot.func>"} } }
      - The variable bus_no is taken from the earlier loop; in failure messages it may
        not correspond to the specific p_node when iterating pci_dict (kept as-is).
    """

    globals.error_list = []
    log.info('Testcase check online GPUs in pcie')
    gpu_pcie_speed = config_dict['gpu_pcie_speed']
    gpu_pcie_width = config_dict['gpu_pcie_width']
    out_dict = get_gpu_pcie_bus_dict(orch.all)
    cmd_list = []
    node_0 = list(out_dict.keys())[0]
    card_list = list(out_dict[node_0].keys())

    # We are making an assumption that it is a homogenous cluster
    # and all nodes have same PCI Bus number
    for card_no in card_list:
        cmd_list = []
        for node in out_dict.keys():
            bus_no = out_dict[node][card_no]['PCI Bus']
            cmd_list.append(f'sudo lspci -vvv -s {bus_no} | grep "LnkSta:" --color=never')
        pci_dict = orch.all.exec_cmd_list(cmd_list)
        record_indexed_node_facts(cvs_results_dict, "gpu_pcie", card_no, pci_dict, parse_pcie_link)
        for p_node in pci_dict.keys():
            bus_no = out_dict[p_node][card_no]['PCI Bus']
            if not re.search(f'Speed {gpu_pcie_speed}GT', pci_dict[p_node]):
                fail_test(
                    f'PCIe speed not matching for bus {bus_no} on node {p_node}, expected {gpu_pcie_speed}GT/s but got {pci_dict[p_node]}'
                )
            if not re.search(f'Width x{gpu_pcie_width}', pci_dict[p_node]):
                fail_test(
                    f'PCIe width not matching for bus {bus_no} on node {p_node}, expected {gpu_pcie_width} but got {pci_dict[p_node]}'
                )
            if re.search('downgrade', pci_dict[p_node]):
                fail_test(f'PCIe in downgraded state for bus {bus_no} on node {p_node}')
    update_test_result()


def test_check_be_nic_pcie_speed_width(orch, config_dict, cvs_results_dict):
    """
    Verify PCIe link speed and width for each Backend NIC on all nodes.

    Reads 'nic_pcie_speed' and 'nic_pcie_width' from config_dict.
    Uses get_gpu_nic_mapping_dict(phdl) to get NIC BDF per card per node.
    Runs 'sudo lspci -vvv -s <nic_bdf> | grep LnkSta:' across nodes in parallel.
    Checks Speed, Width, and absence of 'downgrade' in the output.
    """
    globals.error_list = []
    log.info('Testcase check backend NIC PCIe speed and width')

    # if nic on the Scale Out network is connected via UAlink (example, MI450) insteaad
    # of PCIe then skip this test
    if 'nic_pcie_speed' not in config_dict or 'nic_pcie_width' not in config_dict:
        log.info('nic_pcie_speed or nic_pcie_width not in config_dict, skipping test')
        return

    nic_pcie_speed = config_dict['nic_pcie_speed']
    nic_pcie_width = config_dict['nic_pcie_width']

    out_dict = linux_utils.get_gpu_nic_mapping_dict(orch.all)
    node_0 = list(out_dict.keys())[0]
    card_list = list(out_dict[node_0].keys())

    for card_no in card_list:
        cmd_list = []
        for node in out_dict:
            nic_bdf = out_dict[node][card_no]['nic_bdf']
            cmd_list.append(f'sudo lspci -vvv -s {nic_bdf} | grep "LnkSta:" --color=never')
        pci_dict = orch.all.exec_cmd_list(cmd_list)
        record_indexed_node_facts(cvs_results_dict, "nic_pcie", card_no, pci_dict, parse_pcie_link)
        for p_node in pci_dict:
            output = pci_dict[p_node]
            nic_bdf = out_dict[p_node][card_no]['nic_bdf']
            if not re.search(f'Speed {nic_pcie_speed}GT', output):
                fail_test(
                    f'NIC PCIe speed mismatch for {nic_bdf} on {p_node}: expected {nic_pcie_speed}GT/s, got {output}'
                )
            if not re.search(f'Width x{nic_pcie_width}', output):
                fail_test(
                    f'NIC PCIe width mismatch for {nic_bdf} on {p_node}: expected x{nic_pcie_width}, got {output}'
                )
            if re.search('downgrade', output):
                fail_test(f'NIC PCIe in downgraded state for {nic_bdf} on {p_node}')

    update_test_result()


def test_check_pci_acs(orch, config_dict, cvs_results_dict):
    """
    Verify PCIe ACS is disabled on all nodes.

    This test:
      - Runs 'sudo lspci -vv | grep ACSCtl | grep SrcValid+ --color=never' on each node.
      - If 'ACSCtl:' appears in output, flags a failure (indicates ACS is enabled).
      - Calls update_test_result() to record pass/fail.

    Args:
      phdl: Remote execution handle with exec(cmd: str) -> dict[node, str].
      config_dict: Unused; kept for consistent test signature.

    Notes:
      - globals.error_list is reset at test start; fail_test() records failures.
      - Command pipeline assumes lspci is present and accessible with required privileges.
    """

    globals.error_list = []
    out_dict = orch.all.exec('sudo lspci -vv | grep ACSCtl | grep SrcValid+ --color=never')
    record_node_facts(cvs_results_dict, "pci_acs", out_dict, parse_pci_acs)
    for node in out_dict.keys():
        if re.search('ACSCtl:', out_dict[node], re.I):
            fail_test(f'PCIe ACS not disabled on node {node}')
    update_test_result()


def test_check_dmesg_driver_errors(orch, config_dict):
    """
    Check dmesg for AMDGPU driver errors on each node.

    This test:
      - Runs 'sudo dmesg -T | grep -i amdgpu | egrep -i "fail|error"' on each node.
      - Flags a failure if any 'fail' or 'error' appears in the filtered output.
      - Calls update_test_result() to record pass/fail.

    Args:
      phdl: Remote execution handle with exec(cmd: str) -> dict[node, str].
      config_dict: Unused; kept for consistent test signature.

    Notes:
      - globals.error_list is reset at test start; fail_test() accumulates failures.
      - dmesg -T requires a relatively recent kernel; content depends on ring buffer.
      - Grep may miss issues if log levels or formats differ; adjust patterns as needed.
    """

    globals.error_list = []
    out_dict = orch.all.exec("sudo dmesg -T | grep -i amdgpu  | egrep -i 'fail|error' --color=never")
    for node in out_dict.keys():
        if re.search('fail|error', out_dict[node], re.I):
            fail_test(f'Dmesg has amdgpu driver errors on node {node}')
    update_test_result()
    out_dict = orch.all.exec("sudo dmesg -T | grep -i amdgpu  | egrep -i 'reset|hang|traceback' --color=never")
    for node in out_dict.keys():
        if re.search('reset|hang', out_dict[node], re.I):
            fail_test(f'Dmesg has amdgpu reset/hang errors on node {node}')
    update_test_result()
