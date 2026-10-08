'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json

import pytest

from cvs.lib import globals, host_configs_rundeck, linux_utils
from cvs.lib.report.health_lifecycle import HealthLifecycle, timed_stage
from cvs.lib.rocm_plib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

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
      - Keys whose names start with '_' (_example_*, _comment*) are documentation
        and are removed before placeholder checks and test comparisons.
      - Adjust if your configuration schema changes.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = omit_doc_config_keys(config_dict_t['host'])

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def lifecycle():
    """Wall-clock of each host check, bound as the deck lifecycle source."""
    return HealthLifecycle()


@pytest.fixture(scope="module")
def host_res_dict():
    """
    Module-scoped structured host-check results for the Run Deck status matrix.

    Each check merges its per-node verdict into this dict. The host_configs_cvs
    profile names this fixture in sources.results, so session binding captures
    it at module teardown.
    """
    return {}


def _capture_host_rundeck(host_res_dict, cluster_dict, group, records, version=None):
    """Best-effort: a reporting problem must not change the host-check pass/fail."""
    if host_res_dict is None:
        return
    try:
        meta = host_configs_rundeck.make_meta(cluster_dict, 'host_configs_cvs', version)
        host_configs_rundeck.record_group(host_res_dict, group, records, meta=meta)
    except Exception as exc:
        log.warning("Host configs '%s': could not capture Run Deck results: %s", group, exc)


def _fail_messages(messages):
    for message in messages:
        fail_test(message)


# Main Test cases start from here ..


def test_check_os_release(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.OS_RELEASE):
        out_dict = orch.all.exec('cat /etc/os-release')
    records, messages = host_configs_rundeck.eval_os_release(out_dict, os_version)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.OS_RELEASE, records)
    update_test_result()


def test_check_kernel_version(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.KERNEL_VERSION):
        out_dict = orch.all.exec('uname -a')
    records, messages = host_configs_rundeck.eval_kernel_version(out_dict, kernel_version)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.KERNEL_VERSION, records)
    update_test_result()


def test_check_bios_version(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.BIOS_VERSION):
        out_dict = orch.all.exec('sudo dmidecode -s bios-version')
    records, messages = host_configs_rundeck.eval_bios_version(out_dict, bios_version)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.BIOS_VERSION, records)
    update_test_result()


def test_check_rocm_version(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.ROCM_VERSION):
        out_dict = orch.all.exec('amd-smi version')
    records, messages, detected = host_configs_rundeck.eval_rocm_version(out_dict, rocm_version)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.ROCM_VERSION, records, version=detected)
    update_test_result()


def test_check_gpu_fw_version(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.GPU_FW):
        out_dict = get_amd_smi_fw_dict(orch.all)
    records, messages = host_configs_rundeck.eval_firmware(out_dict, fw_dict)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.GPU_FW, records)
    update_test_result()


def test_check_pci_realloc(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.PCI_REALLOC):
        out_dict = orch.all.exec('cat /proc/cmdline')
    records, messages = host_configs_rundeck.eval_pci_realloc(out_dict, pci_realloc)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.PCI_REALLOC, records)
    update_test_result()


def test_check_iommu_pt(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.IOMMU_PT):
        out_dict = orch.all.exec('cat /proc/cmdline')
    records, messages = host_configs_rundeck.eval_iommu_pt(out_dict)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.IOMMU_PT, records)
    update_test_result()


def test_check_numa_balancing(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.NUMA_BALANCING):
        out_dict = orch.all.exec('sudo sysctl kernel.numa_balancing')
    records, messages = host_configs_rundeck.eval_numa_balancing(out_dict)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.NUMA_BALANCING, records)
    update_test_result()


def test_check_online_memory(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.ONLINE_MEMORY):
        out_dict = orch.all.exec('lsmem')
    records, messages = host_configs_rundeck.eval_online_memory(out_dict, online_mem)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.ONLINE_MEMORY, records)
    update_test_result()


def test_check_pci_accelerators(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.PCI_ACCELERATORS):
        out_dict = orch.all.exec('lspci | grep "accelerators" --color=never')
    records, messages = host_configs_rundeck.eval_pci_accelerators(out_dict, gpu_count)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.PCI_ACCELERATORS, records)
    update_test_result()


def test_check_gpu_pcie_speed_width(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    records = {}

    # We are making an assumption that it is a homogenous cluster
    # and all nodes have same PCI Bus number
    with timed_stage(lifecycle, host_configs_rundeck.GPU_PCIE):
        out_dict = get_gpu_pcie_bus_dict(orch.all)
        node_0 = list(out_dict.keys())[0]
        card_list = list(out_dict[node_0].keys())
        for card_no in card_list:
            cmd_list = []
            for node in out_dict.keys():
                bus_no = out_dict[node][card_no]['PCI Bus']
                cmd_list.append(f'sudo lspci -vvv -s {bus_no} | grep "LnkSta:" --color=never')
            pci_dict = orch.all.exec_cmd_list(cmd_list)
            _fail_messages(
                host_configs_rundeck.absorb_gpu_pcie(
                    records, pci_dict, out_dict, card_no, gpu_pcie_speed, gpu_pcie_width
                )
            )
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.GPU_PCIE, records)
    update_test_result()


def test_check_be_nic_pcie_speed_width(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    records = {}

    with timed_stage(lifecycle, host_configs_rundeck.NIC_PCIE):
        out_dict = linux_utils.get_gpu_nic_mapping_dict(orch.all)
        node_0 = list(out_dict.keys())[0]
        card_list = list(out_dict[node_0].keys())
        for card_no in card_list:
            cmd_list = []
            for node in out_dict:
                nic_bdf = out_dict[node][card_no]['nic_bdf']
                cmd_list.append(f'sudo lspci -vvv -s {nic_bdf} | grep "LnkSta:" --color=never')
            pci_dict = orch.all.exec_cmd_list(cmd_list)
            _fail_messages(
                host_configs_rundeck.absorb_nic_pcie(
                    records, pci_dict, out_dict, card_no, nic_pcie_speed, nic_pcie_width
                )
            )
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.NIC_PCIE, records)
    update_test_result()


def test_check_be_nic_link_speed(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
    """
    Verify the network link speed of every backend NIC on all nodes.

    Reads 'nic_link_speed' (Mb/s, default 400000) and 'nic_link_interfaces'
    (backend netdev names) from config_dict. Without 'nic_link_interfaces' the
    backend NICs are auto-detected per node. Reads /sys/class/net/<iface>/speed
    and fails for any interface that is unreadable or not at the expected speed.
    """
    globals.error_list = []
    log.info('Testcase check backend NIC link speed')
    try:
        expected = host_configs_rundeck.parse_nic_link_speed_setting(config_dict.get('nic_link_speed'))
    except ValueError as exc:
        fail_test(str(exc))
        update_test_result()
        return

    configured = config_dict.get('nic_link_interfaces')
    configured = [configured] if isinstance(configured, str) else configured
    configured = configured or []
    with timed_stage(lifecycle, host_configs_rundeck.NIC_LINK):
        if configured:
            out_dict = orch.all.exec(host_configs_rundeck.nic_link_speed_cmd(configured))
            nics_by_node = {node: list(configured) for node in orch.hosts}
        else:
            log.info('nic_link_interfaces not set, auto-detecting backend NICs')
            nics_by_node = linux_utils.get_backend_nic_dict(orch.all)
            for node in orch.hosts:
                nics_by_node.setdefault(node, [])
            union = [nic for nics in nics_by_node.values() for nic in nics]
            out_dict = orch.all.exec(host_configs_rundeck.nic_link_speed_cmd(union))
    records, messages = host_configs_rundeck.eval_nic_link_speed(out_dict, nics_by_node, expected)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.NIC_LINK, records)
    update_test_result()


def test_check_pci_acs(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.PCI_ACS):
        out_dict = orch.all.exec('sudo lspci -vv | grep ACSCtl | grep SrcValid+ --color=never')
    records, messages = host_configs_rundeck.eval_pci_acs(out_dict)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.PCI_ACS, records)
    update_test_result()


def test_check_dmesg_driver_errors(orch, config_dict, host_res_dict, cluster_dict, lifecycle):
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
    with timed_stage(lifecycle, host_configs_rundeck.DMESG_DRIVER):
        out_dict = orch.all.exec("sudo dmesg -T | grep -i amdgpu  | egrep -i 'fail|error' --color=never")
    records, messages = host_configs_rundeck.eval_dmesg_driver(out_dict)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.DMESG_DRIVER, records)
    update_test_result()
    with timed_stage(lifecycle, host_configs_rundeck.DMESG_RESET):
        out_dict = orch.all.exec("sudo dmesg -T | grep -i amdgpu  | egrep -i 'reset|hang|traceback' --color=never")
    records, messages = host_configs_rundeck.eval_dmesg_reset(out_dict)
    _fail_messages(messages)
    _capture_host_rundeck(host_res_dict, cluster_dict, host_configs_rundeck.DMESG_RESET, records)
    update_test_result()
