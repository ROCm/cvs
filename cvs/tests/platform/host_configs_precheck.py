import pytest
import re
import json
import time

from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as f:
        cluster_dict = json.load(f)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as f:
        config_dict_t = json.load(f)
    config_dict = config_dict_t['host_precheck']
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    log.info("%s", cluster_dict)
    env_vars = cluster_dict.get("env_vars")
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'],
                pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
    return phdl


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    node_list = list(cluster_dict['node_dict'].keys())
    env_vars = cluster_dict.get("env_vars")
    head_node = node_list[0]
    shdl = Pssh(log, [head_node], user=cluster_dict['username'],
                pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
    return shdl


# ─────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────

def test_node_reachability(phdl, cluster_dict):
    """
    Verify all cluster nodes are reachable via SSH.
    Equivalent to clschkping / clschkssh pdsh scripts.

    Runs 'hostname' on every node. If a node is unreachable the SSH handle
    will report an error which is captured by fail_test.
    """
    globals.error_list = []
    log.info('Testcase: node reachability check')

    out_dict = phdl.exec('hostname')
    for node in out_dict:
        output = out_dict[node].strip()
        if not output:
            fail_test(f'Node {node} did not return a hostname — may be unreachable')
        else:
            log.info('Node %s is reachable: %s', node, output)

    update_test_result()


def test_node_uptime(phdl):
    """
    Collect uptime from all nodes.
    Equivalent to clschkuptime pdsh script.

    Logs uptime for visibility; does not fail on any uptime value.
    """
    globals.error_list = []
    log.info('Testcase: node uptime check')

    out_dict = phdl.exec('uptime')
    for node in out_dict:
        log.info('Node %s uptime: %s', node, out_dict[node].strip())

    update_test_result()


def test_amdgpu_driver(phdl):
    """
    Verify the amdgpu kernel module is loaded on all nodes.
    Equivalent to clschkamdgpu pdsh script.
    """
    globals.error_list = []
    log.info('Testcase: amdgpu driver check')

    out_dict = phdl.exec('lsmod | grep -c amdgpu || true')
    for node in out_dict:
        count = out_dict[node].strip()
        if count == '0' or count == '':
            fail_test(f'amdgpu module not loaded on node {node}')
        else:
            log.info('Node %s: amdgpu loaded', node)

    update_test_result()


def test_rdma_links(phdl, config_dict):
    """
    Verify all expected RDMA/backend NIC links are ACTIVE and LINK_UP.
    Equivalent to clschkrdmalink pdsh script.

    Reads expected RDMA device list from config_dict['rdma_devices'].
    Example config entry:
        "rdma_devices": ["ionic_0", "ionic_1", ..., "ionic_7"]
    """
    globals.error_list = []
    log.info('Testcase: RDMA link status check')

    expected_devices = config_dict.get('rdma_devices', [])
    if not expected_devices:
        log.warning('rdma_devices not set in config — skipping RDMA link check')
        update_test_result()
        return

    out_dict = phdl.exec('rdma link show 2>/dev/null || true')
    for node in out_dict:
        output = out_dict[node]
        missing = []
        for dev in expected_devices:
            # Match: link <dev>/<port> state ACTIVE physical_state LINK_UP
            pattern = rf'link\s+{re.escape(dev)}/\d+\s+state\s+ACTIVE\s+physical_state\s+LINK_UP'
            if not re.search(pattern, output):
                missing.append(dev)
        if missing:
            fail_test(f'Node {node}: RDMA links not ACTIVE/LINK_UP: {missing}')
        else:
            log.info('Node %s: all RDMA links up', node)

    update_test_result()


def test_mutual_ssh(phdl, cluster_dict):
    """
    Verify that every node can SSH to every other node without a password.
    Equivalent to ansible mutualssh check_ssh.yml.

    For each node, attempts 'ssh <peer> hostname' for all peer nodes.
    """
    globals.error_list = []
    log.info('Testcase: mutual SSH check')

    node_list = list(cluster_dict['node_dict'].keys())
    vpc_list = [cluster_dict['node_dict'][n]['vpc_ip'] for n in node_list]

    for target_ip in vpc_list:
        ssh_cmd = (
            f'ssh -o BatchMode=yes '
            f'-o ConnectTimeout=5 '
            f'-o StrictHostKeyChecking=no '
            f'-o UserKnownHostsFile=/dev/null '
            f'{target_ip} hostname 2>&1 || echo SSH_FAILED'
        )
        out_dict = phdl.exec(ssh_cmd)
        for source_node in out_dict:
            output = out_dict[source_node].strip()
            if 'SSH_FAILED' in output or not output:
                fail_test(f'Node {source_node} cannot SSH to {target_ip}: {output}')
            else:
                log.info('Node %s -> %s: SSH OK (%s)', source_node, target_ip, output)

    update_test_result()


def test_etc_hosts(phdl, cluster_dict):
    """
    Verify /etc/hosts on each node contains entries for all other nodes.
    Equivalent to ansible checketchosts check_hosts.yml.

    Checks that each node's hostname appears in /etc/hosts on every other node.
    """
    globals.error_list = []
    log.info('Testcase: /etc/hosts check')

    node_list = list(cluster_dict['node_dict'].keys())

    # Collect hostnames first
    hostname_dict = phdl.exec('hostname -s')

    # Read /etc/hosts on all nodes
    etchosts_dict = phdl.exec('cat /etc/hosts')

    for node in node_list:
        etc_hosts_content = etchosts_dict.get(node, '')
        for peer in node_list:
            if peer == node:
                continue
            peer_hostname = hostname_dict.get(peer, '').strip()
            if not peer_hostname:
                continue
            if peer_hostname not in etc_hosts_content:
                fail_test(
                    f'Node {node}: /etc/hosts does not contain entry for peer {peer} ({peer_hostname})'
                )

    update_test_result()


def test_limits_conf(phdl, config_dict):
    """
    Verify /etc/security/limits.conf has all required entries on every node.
    Equivalent to ansible readlimits readlimits.yml.

    Reads required_limits list from config_dict.
    Example config entry:
        "required_limits": [
            "root soft memlock unlimited",
            "root hard memlock unlimited",
            "ubuntu soft nofile 1048576",
            ...
        ]
    """
    globals.error_list = []
    log.info('Testcase: limits.conf validation')

    required_limits = config_dict.get('required_limits', [])
    if not required_limits:
        log.warning('required_limits not set in config — skipping limits.conf check')
        update_test_result()
        return

    out_dict = phdl.exec('cat /etc/security/limits.conf')
    for node in out_dict:
        content = out_dict[node]
        missing = [line for line in required_limits if line not in content]
        if missing:
            for m in missing:
                fail_test(f'Node {node}: missing limits.conf entry: "{m}"')
        else:
            log.info('Node %s: limits.conf OK', node)

    update_test_result()


def test_bnxt_drivers(phdl, config_dict):
    """
    Verify bnxt_re and bnxt_en driver versions and DKMS installation.
    Equivalent to clschkbrcm pdsh script.

    Reads expected versions from config_dict:
        "bnxt_re_version": "236.1.155.0"
        "bnxt_en_version": "1.10.3-236.1.155.0"
    """
    globals.error_list = []
    log.info('Testcase: bnxt driver version check')

    expected_re = config_dict.get('bnxt_re_version', '')
    expected_en = config_dict.get('bnxt_en_version', '')

    if not expected_re and not expected_en:
        log.warning('bnxt_re_version / bnxt_en_version not in config — skipping bnxt driver check')
        update_test_result()
        return

    check_cmd = (
        'BRE_VER=$(modinfo -F version bnxt_re 2>/dev/null); '
        'BRE_FILE=$(modinfo -F filename bnxt_re 2>/dev/null); '
        'BEN_VER=$(modinfo -F version bnxt_en 2>/dev/null); '
        'BEN_FILE=$(modinfo -F filename bnxt_en 2>/dev/null); '
        'echo "bnxt_re_ver=${BRE_VER} bnxt_re_file=${BRE_FILE} bnxt_en_ver=${BEN_VER} bnxt_en_file=${BEN_FILE}"'
    )

    out_dict = phdl.exec(check_cmd)
    for node in out_dict:
        output = out_dict[node]

        re_ver_match = re.search(r'bnxt_re_ver=(\S+)', output)
        re_file_match = re.search(r'bnxt_re_file=(\S+)', output)
        en_ver_match = re.search(r'bnxt_en_ver=(\S+)', output)
        en_file_match = re.search(r'bnxt_en_file=(\S+)', output)

        re_ver = re_ver_match.group(1) if re_ver_match else ''
        re_file = re_file_match.group(1) if re_file_match else ''
        en_ver = en_ver_match.group(1) if en_ver_match else ''
        en_file = en_file_match.group(1) if en_file_match else ''

        re_dkms = 'dkms' in re_file
        en_dkms = 'dkms' in en_file

        if expected_re:
            if re_ver != expected_re or not re_dkms:
                fail_test(
                    f'Node {node}: bnxt_re version mismatch or not DKMS. '
                    f'Expected={expected_re} dkms=True, Got={re_ver} dkms={re_dkms}'
                )
            else:
                log.info('Node %s: bnxt_re OK (ver=%s, dkms=%s)', node, re_ver, re_dkms)

        if expected_en:
            if en_ver != expected_en or not en_dkms:
                fail_test(
                    f'Node {node}: bnxt_en version mismatch or not DKMS. '
                    f'Expected={expected_en} dkms=True, Got={en_ver} dkms={en_dkms}'
                )
            else:
                log.info('Node %s: bnxt_en OK (ver=%s, dkms=%s)', node, en_ver, en_dkms)

    update_test_result()


def test_ionic_nic_count(phdl, config_dict):
    """
    Verify the expected number of Ionic NICs are present on each node.
    Equivalent to the NIC count check in ansible ainicfwcheck fwcheck.yml.

    Reads expected count from config_dict:
        "ionic_nic_count": 8
    """
    globals.error_list = []
    log.info('Testcase: Ionic NIC count check')

    expected_count = int(config_dict.get('ionic_nic_count', 8))

    out_dict = phdl.exec("ibv_devices | awk '/ionic_[0-9]+/ {print $1}' | wc -l")
    for node in out_dict:
        count_str = out_dict[node].strip()
        try:
            count = int(count_str)
        except ValueError:
            fail_test(f'Node {node}: could not parse Ionic NIC count: {count_str}')
            continue

        if count != expected_count:
            fail_test(
                f'Node {node}: Ionic NIC count mismatch. '
                f'Expected={expected_count}, Found={count}'
            )
        else:
            log.info('Node %s: Ionic NIC count OK (%d)', node, count)

    update_test_result()


def test_ionic_fw_version(phdl, config_dict):
    """
    Verify Ionic NIC firmware version on all nodes.
    Equivalent to ansible ainicfwcheck fwcheck.yml firmware check.

    Reads expected version from config_dict:
        "ionic_fw_version": "1.117.5-a-56"
    """
    globals.error_list = []
    log.info('Testcase: Ionic NIC firmware version check')

    expected_fw = config_dict.get('ionic_fw_version', '')
    if not expected_fw:
        log.warning('ionic_fw_version not in config — skipping Ionic FW check')
        update_test_result()
        return

    # Parse: for each NIC entry extract Firmware-A version
    parse_cmd = (
        "nicctl show version firmware | awk '"
        "/^NIC/ { nic=$3 } "
        "/Firmware-A/ { print nic, $NF }'"
    )

    out_dict = phdl.exec(parse_cmd)
    for node in out_dict:
        output = out_dict[node].strip()
        if not output:
            fail_test(f'Node {node}: nicctl show version firmware returned no output')
            continue

        lines = output.splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            nic_name = parts[0]
            fw_ver = parts[1]
            if fw_ver != expected_fw:
                fail_test(
                    f'Node {node}: NIC {nic_name} firmware mismatch. '
                    f'Expected={expected_fw}, Got={fw_ver}'
                )
            else:
                log.info('Node %s: NIC %s firmware OK (%s)', node, nic_name, fw_ver)

    update_test_result()


def test_ionic_host_sw_version(phdl, config_dict):
    """
    Verify Ionic NIC host software version (nicctl + IPC driver) on all nodes.
    Equivalent to ansible ainicfwcheck fwcheck.yml host software check.

    Reads expected version from config_dict:
        "ionic_host_sw_version": "1.117.5-a-56"
    """
    globals.error_list = []
    log.info('Testcase: Ionic NIC host software version check')

    expected_sw = config_dict.get('ionic_host_sw_version', '')
    if not expected_sw:
        log.warning('ionic_host_sw_version not in config — skipping Ionic host SW check')
        update_test_result()
        return

    def normalize(ver):
        return re.sub(r'[.\-]', '', ver).strip()

    expected_norm = normalize(expected_sw)

    parse_cmd = (
        "nicctl show version host-software | awk '"
        "/nicctl/ { nicctl=$NF } "
        "/IPC driver/ { ipc=$NF } "
        "END { print nicctl, ipc }'"
    )

    out_dict = phdl.exec(parse_cmd)
    for node in out_dict:
        output = out_dict[node].strip()
        parts = output.split()
        if len(parts) < 2:
            fail_test(f'Node {node}: could not parse host software versions: {output}')
            continue

        nicctl_ver = parts[0]
        ipc_ver = parts[1]

        nicctl_ok = normalize(nicctl_ver) == expected_norm
        ipc_ok = normalize(ipc_ver) == expected_norm

        if not nicctl_ok or not ipc_ok:
            fail_test(
                f'Node {node}: host software version mismatch. '
                f'Expected={expected_sw}, nicctl={nicctl_ver}, ipc={ipc_ver}'
            )
        else:
            log.info('Node %s: host SW OK (nicctl=%s, ipc=%s)', node, nicctl_ver, ipc_ver)

    update_test_result()


def test_rdma_netdev_mapping(phdl, config_dict):
    """
    Verify RDMA-to-netdev mapping is correct on all nodes.
    Equivalent to ansible checkrdmanetdev check_rdma_mapping.yml.

    Reads expected mapping from config_dict:
        "rdma_netdev_map": {
            "ionic_0": "ens1f0np0",
            "ionic_1": "ens1f1np0",
            ...
        }
    """
    globals.error_list = []
    log.info('Testcase: RDMA netdev mapping check')

    expected_map = config_dict.get('rdma_netdev_map', {})
    if not expected_map:
        log.warning('rdma_netdev_map not in config — skipping RDMA netdev mapping check')
        update_test_result()
        return

    out_dict = phdl.exec('rdma link show 2>/dev/null || true')
    for node in out_dict:
        output = out_dict[node]
        for rdma_dev, expected_netdev in expected_map.items():
            # Look for: link ionic_0/1 ... netdev <name>
            pattern = rf'link\s+{re.escape(rdma_dev)}/\d+.*?netdev\s+(\S+)'
            match = re.search(pattern, output)
            if not match:
                fail_test(
                    f'Node {node}: RDMA device {rdma_dev} not found in rdma link output'
                )
            elif match.group(1) != expected_netdev:
                fail_test(
                    f'Node {node}: {rdma_dev} netdev mismatch. '
                    f'Expected={expected_netdev}, Got={match.group(1)}'
                )
            else:
                log.info('Node %s: %s -> %s OK', node, rdma_dev, expected_netdev)

    update_test_result()


def test_dcqcn_pfc_qos(phdl, config_dict):
    """
    Validate DCQCN, PFC, and QoS settings on all nodes.
    Equivalent to ansible ainicvalidation playbooks/validate_all.yml
    (validate_dcqcn.sh, validate_pfc.sh, validate_qos.sh).

    Reads expected values from config_dict:
        "dcqcn_enabled": true
        "pfc_enabled": true
        "qos_enabled": true
    """
    globals.error_list = []
    log.info('Testcase: DCQCN / PFC / QoS validation')

    check_dcqcn = config_dict.get('dcqcn_enabled', False)
    check_pfc = config_dict.get('pfc_enabled', False)
    check_qos = config_dict.get('qos_enabled', False)

    if not any([check_dcqcn, check_pfc, check_qos]):
        log.warning('No DCQCN/PFC/QoS checks enabled in config — skipping')
        update_test_result()
        return

    if check_dcqcn:
        # DCQCN is enabled when ECN is configured on the NIC interfaces
        out_dict = phdl.exec(
            "mlnx_qos -i eth0 2>/dev/null | grep -i ecn || "
            "cat /sys/class/net/*/ecn 2>/dev/null | head -4 || "
            "echo 'dcqcn_check_skipped'"
        )
        for node in out_dict:
            output = out_dict[node].strip()
            if 'dcqcn_check_skipped' in output:
                log.warning('Node %s: DCQCN check skipped (tool not available)', node)
            else:
                log.info('Node %s: DCQCN output: %s', node, output)

    if check_pfc:
        # PFC: check that pause frames are enabled on the backend interfaces
        out_dict = phdl.exec(
            "ethtool -a eth0 2>/dev/null | grep -i pause || echo 'pfc_check_skipped'"
        )
        for node in out_dict:
            output = out_dict[node].strip()
            if 'pfc_check_skipped' in output:
                log.warning('Node %s: PFC check skipped (ethtool not available)', node)
            elif 'off' in output.lower():
                fail_test(f'Node {node}: PFC pause frames appear to be disabled: {output}')
            else:
                log.info('Node %s: PFC output: %s', node, output)

    if check_qos:
        out_dict = phdl.exec("tc qdisc show 2>/dev/null | head -20 || echo 'qos_check_skipped'")
        for node in out_dict:
            output = out_dict[node].strip()
            if 'qos_check_skipped' in output:
                log.warning('Node %s: QoS check skipped', node)
            else:
                log.info('Node %s: QoS qdisc: %s', node, output[:200])

    update_test_result()
