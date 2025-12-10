'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
from cvs.lib import rocm_plib
from cvs.lib.utils_lib import *


def get_lshw_network_dict(phdl):
    """
    Parse `lshw -class network -businfo` output (per node) into a nested dictionary.

    This function executes `sudo lshw -class network -businfo` via the provided phdl
    handle (which is expected to run the command on one or more nodes and return a
    mapping of node -> command output). It then parses each line of output to extract:
      - The PCI bus identifier (e.g., 0000:03:00.0)
      - The OS device name (e.g., enp3s0), when present
      - The device description (e.g., RTL8111/8168/8411 PCI Express Gigabit Ethernet Controller)

    It returns a structure like:
      {
        "nodeA": {
          "enp3s0": {
            "pci_bus": "0000:03:00.0",
            "description": "RTL8111/8168/8411 PCI Express Gigabit Ethernet Controller"
          },
          "virtio": {
            "pci_bus": "0000:00:03.0",
            "description": "Virtio network device"
          }
        },
        "nodeB": { ... }
      }

    Notes:
    - The parsing relies on regex patterns tailored for PCI bus entries. If the device
      name is missing on a line, the key 'virtio' is used as a fallback, which can
      overwrite entries if multiple unnamed devices are present.
    - The function currently matches only 'pci@...' lines. Non-PCI buses (e.g., usb@)
      will be ignored by these patterns.

    Args:
        phdl: Parallel ssh handle with a method `exec(cmd)` that returns a dict-like
              object mapping node identifiers to the command's text output.

    Returns:
        dict: Nested dictionary of parsed network devices per node.
    """

    lshw_dict = {}

    # Execute lshw on all nodes via the provided handle. The expectation is that
    # out_dict is like: { node_name: "command stdout as string", ... }

    out_dict = phdl.exec('sudo lshw -class network -businfo')
    for node in out_dict.keys():
        lshw_dict[node] = {}
        # Process the output line-by-line. split("\n") assumes LF newlines from lshw.
        for line in out_dict[node].split("\n"):
            # Pattern 1: lines with pci bus, device name, then the 'network' class and description
            # Example: "pci@0000:03:00.0 enp3s0 network 8411 PCI Express Gigabit Ethernet Controller"
            pattern = r"pci\@([0-9a-f\:\.]+)\s+([a-z0-9\-\.]+)\s+network\s+([a-z0-9\s\[\]\/\-\_]+)"

            # Pattern 2: lines with pci bus and 'network' class/description but without a device name
            # Example: "pci@0000:00:03.0 network Virtio network device"
            pattern_else = r"pci\@([0-9a-f\:\.]+)\s+network\s+([a-z0-9\s\[\]\/\-\_]+)"

            if re.search(pattern, line, re.I):
                match = re.search(pattern, line, re.I)
                # Extract fields from the matched groups
                pci_bus = match.group(1)  # e.g., "0000:03:00.0"
                dev_name = match.group(2)  # e.g., "enp3s0"
                dev_descr = match.group(3)  # e.g., AMD PCI Express ..
                lshw_dict[node][dev_name] = {}
                lshw_dict[node][dev_name]['pci_bus'] = pci_bus
                lshw_dict[node][dev_name]['description'] = dev_descr

            # If no device name is present, use the fallback pattern
            elif re.search(pattern_else, line, re.I):
                match = re.search(pattern_else, line, re.I)
                pci_bus = match.group(1)  # e.g., "0000:00:03.0"
                dev_name = 'virtio'  # Fallback key when the device name is missing
                dev_descr = match.group(2)  # e.g., "Virtio network device"
                lshw_dict[node][dev_name] = {}
                lshw_dict[node][dev_name]['pci_bus'] = pci_bus
                lshw_dict[node][dev_name]['description'] = dev_descr

    return lshw_dict


def get_ip_addr_dict(phdl):
    """
    Parse `ip addr show` output (per node) into a structured dictionary.

    This function executes:
        sudo ip addr show | grep -A 5 mtu --color=never
    via the provided phdl handle (which should run the command on one or more nodes and
    return a mapping of node -> command output). It then parses each node's output to
    extract interface-level details:
      - Interface name
      - Flags
      - MTU
      - Administrative/operational state
      - MAC address
      - IPv4 addresses (list)
      - IPv6 addresses (list)

    Returns data in the form:
      {
        "node1": {
          "eth0": {
            "mtu": "1500",
            "state": "UP",
            "mac_addr": "aa:bb:cc:dd:ee:ff",
            "ipv4_addr_list": ["192.168.1.10/24"],
            "ipv6_addr_list": ["fe80::abcd/64"]
          },
          ...
        },
        "node2": { ... }
      }

    Notes:
    - The parser depends on the typical formatting of `ip addr show`. If the output
      format varies across distros or versions, matches may fail for some fields.
    - The grep context (`-A 5`) limits the lines inspected per interface; if an address
      appears beyond that window, it may not be captured.
    - If multiple interfaces are unnamed or if parsing fails to find an interface line
      before property lines, subsequent property captures could raise KeyError. The
      current code assumes the interface-identifying line appears before its details.
    """

    ip_dict = {}

    # Execute the command on one or more nodes; expect dict[node] -> output string
    out_dict = phdl.exec('sudo ip addr show | grep -A 5 mtu --color=never')
    int_nam = None
    for node in out_dict.keys():
        ip_dict[node] = {}
        for line in out_dict[node].split("\n"):
            # Match interface header line:
            #   <idx>: <ifname>: <FLAGS>
            # Example:
            #   2: enp3s0: <BROADCAST,MULTICAST,UP,LOWER_UP>
            pattern = r"[0-9]+\:\s+([0-9a-z\.\_\-\/]+):\s+([\<\>\,A-Z0-9]+)"
            if re.search(pattern, line):
                match = re.search(pattern, line)
                int_nam = match.group(1)
                # Initialize interface dict with lists for multiple addresses
                ip_dict[node][int_nam] = {}
                ip_dict[node][int_nam]['ipv4_addr_list'] = []
                ip_dict[node][int_nam]['ipv6_addr_list'] = []
                ip_dict[node][int_nam]['flags'] = match.group(2)

            # Capture MTU value, e.g., "mtu 1500"
            if re.search('mtu ([0-9]+)', line):
                match = re.search('mtu ([0-9]+)', line)
                ip_dict[node][int_nam]['mtu'] = match.group(1)

            # Capture state, e.g., "state UP"
            if re.search('state ([A-Z]+)', line):
                match = re.search('state ([A-Z]+)', line)
                ip_dict[node][int_nam]['state'] = match.group(1)

            # Capture MAC address line, e.g., "link/ether aa:bb:cc:dd:ee:ff"
            pattern = r"link\/ether\s+([a-f0-9\:]+)"
            if re.search(pattern, line):
                match = re.search(pattern, line)
                ip_dict[node][int_nam]['mac_addr'] = match.group(1)

            # Capture IPv4 addresses, e.g., "inet 192.168.1.10/24 ..."
            pattern = r"inet\s+([0-9\.\/]+)"
            if re.search(pattern, line):
                match = re.search(pattern, line)
                ip_dict[node][int_nam]['ipv4_addr_list'].append(match.group(1))

            # Capture IPv6 addresses, e.g., "inet6 fe80::abcd/64 ..."
            pattern = r"inet6\s+([a-f0-9\:\/]+)"
            if re.search(pattern, line):
                match = re.search(pattern, line)
                ip_dict[node][int_nam]['ipv6_addr_list'].append(match.group(1))

    return ip_dict


def get_rdma_nic_dict(phdl):
    """
    Execute `rdma link` on one or more nodes via the provided handle and parse the
    output into a nested dictionary of RDMA NIC information.

    Expected behavior:
      - phdl.exec('sudo rdma link') should return a dict mapping each node identifier
        to the command's stdout as a single multiline string.
      - For each node, the function parses lines starting with 'link' that look like:
          link <rdma_dev>/<port> state <STATE> physical_state <PHYS_STATE> netdev <NETDEV>
        Example:
          link mlx5_0/1 state ACTIVE physical_state LinkUp netdev eth0

    Returned structure:
      {
        "<node>": {
          "<rdma_dev>": {
            "port": "<port_number_as_string>",
            "device_status": "<STATE>",       # e.g., ACTIVE, DOWN
            "link_status": "<PHYS_STATE>",    # e.g., LinkUp, Disabled
            "eth_device": "<NETDEV>"          # e.g., eth0
          },
          ...
        },
        ...
      }

    Notes:
      - Only lines beginning with 'link' are considered.
      - This parser assumes the canonical `rdma link` output format; if the output
        varies (driver/version differences), the regex may fail to match.
      - If multiple ports exist for the same RDMA device, later matches will overwrite
        the same device key (since the device name is used as the key). If per-port
        granularity is required, consider using a composite key (e.g., "<dev>/<port>").
    """

    rdma_dict = {}

    # Run 'rdma link' across node(s); expect a dict: { node_name: "<multiline stdout>", ... }
    out_dict = phdl.exec('sudo rdma link')
    # gid_dict_t = phdl.exec('sudo show_gids | grep -i v2 --color=never')
    for node in out_dict.keys():
        rdma_dict[node] = {}
        for line in out_dict[node].split("\n"):
            if re.search('^link', line):
                pattern = r"link\s+([a-zA-Z0-9\_\-\.]+)\/([0-9]+)\s+state\s+([A-Za-z]+)\s+physical_state\s+([A-Za-z\_]+)\s+netdev\s+([a-z0-9A-Z\.]+)"
                match = re.search(pattern, line)
                dev = match.group(1)
                rdma_dict[node][dev] = {}
                rdma_dict[node][dev]['port'] = match.group(2)  # Port number (string)
                rdma_dict[node][dev]['device_status'] = match.group(3)  # Device state (e.g., ACTIVE)
                rdma_dict[node][dev]['link_status'] = match.group(4)  # Physical link state (e.g., LinkUp)
                rdma_dict[node][dev]['eth_device'] = match.group(5)  # Associated netdev (e.g., eth0)
    return rdma_dict


def get_active_rdma_nic_dict(phdl):
    """
    Execute `rdma link` on one or more nodes via the provided handle and parse the
    output into a nested dictionary of RDMA NIC information and build only for
    ACTIVE Interfaces

    Expected behavior:
      - phdl.exec('sudo rdma link') should return a dict mapping each node identifier
        to the command's stdout as a single multiline string.
      - For each node, the function parses lines starting with 'link' that look like:
          link <rdma_dev>/<port> state <STATE> physical_state <PHYS_STATE> netdev <NETDEV>
        Example:
          link mlx5_0/1 state ACTIVE physical_state LinkUp netdev eth0

    Returned structure:
      {
        "<node>": {
          "<rdma_dev>": {
            "port": "<port_number_as_string>",
            "device_status": "<STATE>",       # e.g., if device status is Active
            "link_status": "<PHYS_STATE>",    # e.g., LinkUp, Disabled
            "eth_device": "<NETDEV>"          # e.g., eth0
          },
          ...
        },
        ...
      }

    Notes:
      - Only lines beginning with 'link' are considered.
      - This parser assumes the canonical `rdma link` output format; if the output
        varies (driver/version differences), the regex may fail to match.
      - If multiple ports exist for the same RDMA device, later matches will overwrite
        the same device key (since the device name is used as the key). If per-port
        granularity is required, consider using a composite key (e.g., "<dev>/<port>").
    """

    rdma_dict = {}
    out_dict = phdl.exec('sudo rdma link')
    # gid_dict_t = phdl.exec('sudo show_gids | grep -i v2 --color=never')
    for node in out_dict.keys():
        rdma_dict[node] = {}
        for line in out_dict[node].split("\n"):
            if re.search('^link', line):
                pattern = r"link\s+([a-zA-Z0-9\_\-\.]+)\/([0-9]+)\s+state\s+([A-Za-z]+)\s+physical_state\s+([A-Za-z\_]+)\s+netdev\s+([a-z0-9A-Z\.]+)"
                match = re.search(pattern, line)
                dev = match.group(1)
                status = match.group(3)
                if re.search('ACTIVE', status, re.I):
                    rdma_dict[node][dev] = {}
                    rdma_dict[node][dev]['port'] = match.group(2)
                    rdma_dict[node][dev]['device_status'] = status
                    rdma_dict[node][dev]['link_status'] = match.group(4)
                    rdma_dict[node][dev]['eth_device'] = match.group(5)
    return rdma_dict


def get_backend_nic_dict(phdl):
    lshw_dict = get_lshw_network_dict(phdl)
    bck_net_dict = {}
    for node in lshw_dict.keys():
        bck_net_dict[node] = []
        print(lshw_dict[node])
        # we make a crude assumption that number of backend nics are more than front end ..
        # if we have to pass this from Input file, we can do that.
        list_a = []
        list_b = []
        for intf_name in lshw_dict[node].keys():
            if len(list_a) == 0:
                list_a.append(intf_name)
            else:
                if lshw_dict[node][intf_name]['description'] == lshw_dict[node][list_a[0]]['description']:
                    list_a.append(intf_name)
                else:
                    list_b.append(intf_name)
        if len(list_a) > len(list_b):
            bck_net_dict[node] = list_a
        else:
            bck_net_dict[node] = list_b
    return bck_net_dict


def get_backend_rdma_nic_dict(phdl):
    """
    Build a per-node dictionary of RDMA devices that map to backend NICs.

    This function cross-references:
      - All active RDMA NICs (via get_active_rdma_nic_dict)
      - The list of backend NICs (via get_backend_nic_dict)

    For each node, it filters RDMA devices to only those whose associated Ethernet
    device (eth_device) is present in the backend NIC list for that node.

    Args:
        phdl: Execution/handle/context object passed through to helper functions.

    Returns:
        dict: Nested structure of the form:
          {
            "<node>": {
              "<rdma_dev>": {
                ... same fields as returned by get_active_rdma_nic_dict() for that device ...
                # Must include at least 'eth_device' used for filtering
              },
              ...
            },
            ...
          }
    """
    bck_rdma_nic_dict = {}

    # Gather backend NICs per node (e.g., NICs designated for backend traffic)
    bck_nic_dict = get_backend_nic_dict(phdl)

    # Gather active RDMA NIC information per node
    rdma_nic_dict = get_active_rdma_nic_dict(phdl)

    # Iterate over nodes that have RDMA NIC info
    for node in rdma_nic_dict.keys():
        # Backend NIC list for this node (e.g., ["eth0", "eth1"])
        bck_rdma_nic_dict[node] = {}
        bck_nic_list = bck_nic_dict[node]

        # For each RDMA device on this node, keep it only if its associated
        # Ethernet device is in the backend list
        for rdma_dev in rdma_nic_dict[node].keys():
            if rdma_nic_dict[node][rdma_dev]['eth_device'] in bck_nic_list:
                bck_rdma_nic_dict[node][rdma_dev] = rdma_nic_dict[node][rdma_dev]
    return bck_rdma_nic_dict


def convert_ethtool_out_to_dict(ethtool_out, vendor=None):
    """
    Parse ethtool -S output into a flat dictionary of total (non per-queue) stats.

    This function scans the provided ethtool statistics output and extracts key:value
    pairs where keys are lowercase, underscore, or hyphen separated tokens and values
    are non-negative integers. It intentionally ignores per-queue stats and focuses
    only on "total" counters that match the simple pattern "<name>: <number>".

    Args:
        ethtool_out (str): Raw string output from a command like `ethtool -S <iface>`.
        vendor (str, optional): Vendor hint (currently unused). Reserved for future
            vendor-specific parsing adjustments.

    Returns:
        dict: Mapping of stat name -> value (as strings), e.g.:
              {"rx_packets": "12345", "tx_errors": "0", ...}

    Notes:
        - The regex matches only keys composed of [a-z_-] followed by ": <digits>".
          Mixed-case keys or keys with other characters will be ignored.
        - Values are kept as strings to preserve original behavior.
        - The function prints the intermediate match list for debugging purposes.
          Consider removing or replacing with logging in production.
    """

    out_dict = {}
    # For now, let us ignore the per queue stats and just collect total stats
    pattern = r"([a-z\_\-]+\:\s+[0-9]+)"
    match_list = re.findall(pattern, ethtool_out, re.I)

    # print(match_list)

    # Convert each "<name>: <number>" into dict entries
    for match_item in match_list:
        pattern = r"([a-z\_\-]+)\:\s+([0-9]+)"
        match = re.search(pattern, match_item, re.I)
        out_dict[match.group(1)] = match.group(2)
    return out_dict


# stats_dict will be indexed by node_ip, followed by interface name of backend NICs


def get_nic_ethtool_stats_dict(phdl, vendor=None):
    """
    Collect per-interface ethtool statistics from backend RDMA NICs across nodes.

    This function:
      1) Retrieves per-node backend RDMA NICs using get_backend_rdma_nic_dict(phdl).
      2) Builds a set of ethtool -S commands per NIC index so that execution can be done
         in parallel across nodes even if interface names differ by node.
      3) Executes those commands in batches via phdl.exec_cmd_list.
      4) Parses each ethtool output with convert_ethtool_out_to_dict, keyed by interface name.
      5) Emits warnings to stdout if any error-like counters (err|discard|drop|crc|fcs|reset)
         are greater than zero.

    Args:
        phdl: A handle that can execute commands on multiple nodes. Must support:
              - exec(...) returning dict[node] -> output string
              - exec_cmd_list(list_of_cmds) returning dict[node] -> output string
        vendor (str, optional): Hint for vendor-specific parsing that is forwarded to
              convert_ethtool_out_to_dict. Currently optional/unused depending on parser.

    Returns:
        dict: Nested mapping of the form:
          {
            "<node>": {
              "<interface_name>": {
                "<stat_key>": "<stat_value_str>",
                ...
              },
              ...
            },
            ...
          }

    Important assumptions and behavior:
        - Assumes each node in bck_nic_dict has the same number of backend RDMA NICs,
          and that NICs are accessed by a consistent index across nodes when batching.
        - Uses grep -v "[" to skip per-queue stats (bracketed) and focus on aggregate totals.
        - Prints warnings (stdout) for non-zero error-like counters; does not raise.
        - Values parsed from ethtool output are kept as strings to preserve the parser behavior.
    """

    stats_dict = {}

    # Gather backend RDMA NICs per node:
    # bck_nic_dict structure example:
    #   { node: { rdma_dev: { 'eth_device': '<iface>' , ... }, ... }, ... }

    bck_nic_dict = get_backend_rdma_nic_dict(phdl)

    # Derive node list and infer NIC count from the first node
    node_list = list(bck_nic_dict.keys())
    node_0 = node_list[0]
    no_of_nics = len(bck_nic_dict[node_0])

    # Initialize the final per-node stats dict
    for node in node_list:
        stats_dict[node] = {}

    # Build a list of list of cmds with the assumptions that NICs can be with different
    # interface names across nodes and we still want to do parallel execution ..
    # cmd_dict is a dict with key as nodes and value as list of cmds

    # Prepare parallel command batches:
    # We build a list of interface names per node and then create batches by NIC index.
    cmd_dict = {}
    eth_dev_dict = {}
    print(node_list)
    print(bck_nic_dict)

    # Build map of nodes to ordered lists of backend interface names
    for node in node_list:
        node_nic_dict = bck_nic_dict[node]
        eth_dev_dict[node] = []
        for dev_name in list(node_nic_dict.keys()):
            intf_name = node_nic_dict[dev_name]['eth_device']
            eth_dev_dict[node].append(intf_name)

    # For each NIC index, generate one command per node so they can be executed in parallel
    for i in range(0, no_of_nics):
        cmd_dict[i] = []
        for node in node_list:
            intf_nam = eth_dev_dict[node][i]
            cmd_dict[i].append(f'sudo ethtool -S {intf_nam} | grep -v "\[" --color=never')

    # Execute each batch of commands and parse results into stats_dict
    for i in range(0, no_of_nics):
        cmd_list = cmd_dict[i]
        stats_dict_out = phdl.exec_cmd_list(cmd_list)
        for node in stats_dict_out:
            intf_nam = eth_dev_dict[node][i]
            stats_dict[node][intf_nam] = convert_ethtool_out_to_dict(stats_dict_out[node], vendor)

    # Emit warnings for any non-zero error-like counters
    for node in stats_dict.keys():
        for intf in stats_dict[node].keys():
            for counter in stats_dict[node][intf].keys():
                if re.search('err|discard|drop|crc|fcs|reset', counter, re.I):
                    if int(stats_dict[node][intf][counter]) > 0:
                        print(f'WARN !! {node} {intf} {counter} {stats_dict[node][intf][counter]}')

    return stats_dict


def get_lldp_dict(phdl):
    """
    Retrieve LLDP neighbor information from one or more nodes and return it as a dict.

    This function runs:
        sudo lldpcli show neighbors -f json
    on each node via the provided phdl handle, which is expected to return a mapping of
    node identifiers to the command's raw JSON output string. Each node's JSON output is
    then converted to a Python dict via json_to_dict.

    Args:
        phdl: An execution handle/context that exposes `exec(cmd: str) -> Dict[str, str]`,
              returning { "<node>": "<lldp-json-output>", ... }.

    Returns:
        dict: A mapping of node -> parsed LLDP neighbors data (as Python dict), e.g.:
              {
                "<node>": { ... parsed JSON from lldpcli ... },
                ...
              }

    Notes:
        - Requires lldpcli to be installed and accessible with sudo privileges.
        - Assumes a helper function `json_to_dict` is available to parse/normalize the
          LLDP JSON output; behavior depends on that function?s implementation.
        - Output structure mirrors lldpcli JSON; fields may vary by lldpcli version.
    """

    lldp_dict = {}
    lldp_installed = True
    out_dict = phdl.exec('which lldpcli')
    for node in out_dict.keys():
        if not re.search('lldpcli', out_dict[node]):
            lldp_installed = False
    if lldp_installed is not True:
        print('Cannot get LLDP Dict as lldpcli is missing')
        return {}
        # try:
        #    phdl.exec('sudo apt update -y')
        #    phdl.exec('sudo DEBIAN_FRONTEND=noninteractive apt install -yq lldpd')
        # except Exception as e:
        #    print('Error installing LLDP with apt install - {}'.format(e))
        #    return lldp_dict

    print('Get LLDP dict')

    # Execute lldpcli across nodes; expected shape: { node_name: "<json string>", ... }
    out_dict = phdl.exec('sudo lldpcli show neighbors -f json')

    # Convert each node's JSON string into a Python dictionary
    for node in out_dict.keys():
        lldp_dict[node] = json_to_dict(out_dict[node])
    return lldp_dict


def get_dns_dict(phdl):
    dns_dict = {}
    out_dict = phdl.exec('sudo resolvectl status | head -7')
    for node in out_dict.keys():
        dns_dict[node] = {}
        for line in out_dict[node].split("\n"):
            if re.search('Protocols', line, re.I):
                print('')
            elif re.search('Protocols', line, re.I):
                print('')
            elif re.search('Current DNS Server', line, re.I):
                print('')
            elif re.search('DNS Servers', line, re.I):
                print('')
            elif re.search('DNS Domain', line, re.I):
                print('')
    return dns_dict


def get_rdma_stats_dict(phdl):
    """
    Collect per-node RDMA statistics (JSON) and return only those for backend RDMA NICs.

    This function:
      - Retrieves the per-node list/dict of backend RDMA NICs via get_backend_rdma_nic_dict(phdl).
      - Executes `sudo rdma statistic --json` across nodes using phdl.exec, which is
        expected to return a mapping of node -> raw JSON output string.
      - Parses each node's JSON output with json_to_dict into a list of RDMA stat dicts.
      - Filters the stats to include only entries whose 'ifname' matches a backend NIC
        for that node.

    Args:
        phdl: Execution handle/context capable of running commands on multiple nodes.
              Must provide:
                - exec(cmd: str) -> Dict[str, str], mapping node to command stdout.

    Returns:
        dict: Nested mapping of the form:
          {
            "<node>": {
              "<ifname>": { ... raw RDMA stats for that interface ... },
              ...
            },
            ...
          }

    Notes:
        - Assumes get_backend_rdma_nic_dict returns a per-node mapping where membership
          checks like `if device_name in bck_nic_list` are meaningful (e.g., list of
          interface names or a dict whose keys are interface names).
        - Assumes json_to_dict returns an iterable (e.g., list) of dicts, each containing
          at least the key 'ifname'.
        - Prints the backend NIC list per node for debugging/tracing purposes.
        - No explicit error handling is included; missing keys or unexpected structures
          may raise exceptions.
    """

    rdma_stats_dict = {}

    # Determine which RDMA NICs are considered "backend" per node
    bck_nic_dict = get_backend_rdma_nic_dict(phdl)

    # Fetch RDMA statistics in JSON for all nodes
    out_dict = phdl.exec('sudo rdma statistic --json')

    # Process each node's output
    for node in out_dict.keys():
        bck_nic_list = bck_nic_dict[node]
        print(bck_nic_list)
        rdma_stats_dict[node] = {}

        # Convert the node's JSON output into Python objects (expected to be a list of dicts)
        rdma_dict_list = json_to_dict(out_dict[node])

        # Keep only stats for backend NICs, keyed by interface name
        for rdma_dict in rdma_dict_list:
            device_name = rdma_dict['ifname']
            if device_name in bck_nic_list:
                rdma_stats_dict[node][device_name] = rdma_dict
    return rdma_stats_dict


def get_linux_perf_tuning_dict(phdl):
    """
    Collect key Linux performance-related settings and return them in a dictionary.

    This function uses the provided execution handle (phdl) to run a set of commands
    commonly reviewed for performance tuning and platform characterization:
      - BIOS/Firmware version (via dmidecode)
      - Kernel NUMA balancing (sysctl)
      - NMI watchdog status (/proc)
      - Transparent Huge Pages (THP) mode (/sys)
      - CPU power/driver information (cpupower)

    Assumptions and behavior:
      - phdl.exec(cmd: str) executes the command (often via sudo) and returns its stdout
        as a string (or per your phdl implementation).
      - The target system provides the required tools and paths:
          dmidecode, sysctl, cpupower; /proc/sys and /sys/kernel are mounted.
      - Many of these commands require elevated privileges. Ensure sudo access is
        configured in your execution environment.
      - Outputs are kept as raw strings (unparsed) to preserve original behavior; consumers
        may parse them further to derive structured values (e.g., booleans/ints).
      - Note: This function currently lacks an explicit `return out_dict`. If the intention
        is to return the collected data, add `return out_dict` at the end.

    Returns:
      None (as written; see note above if a dictionary return is intended).
    """

    out_dict = {}

    # BIOS/FW version string (requires dmidecode; typically needs root privileges)
    out_dict['bios_version'] = phdl.exec('sudo dmidecode -s bios-version')

    # Kernel auto-NUMA balancing setting (e.g., "kernel.numa_balancing = 0/1")
    out_dict['numa_balancing'] = phdl.exec('sudo sysctl kernel.numa_balancing')

    # NMI watchdog status (0 disabled, >0 enabled mode; read directly from /proc)
    out_dict['nmi_watchdog'] = phdl.exec('sudo cat /proc/sys/kernel/nmi_watchdog')

    # Transparent Huge Pages configuration (e.g., "[always] madvise never")
    out_dict['huge_pages'] = phdl.exec('sudo cat /sys/kernel/mm/transparent_hugepage/enabled')

    # CPU driver/governor/boost information (cpupower info dump)
    out_dict['cpu_power_profile'] = phdl.exec('sudo cpupower info')


def get_lshw_backend_nic_dict(phdl):
    lshw_bck_nic_dict = {}
    lshw_dict = get_lshw_network_dict(phdl)
    rdma_nic_dict = get_backend_rdma_nic_dict(phdl)
    for node in rdma_nic_dict.keys():
        lshw_bck_nic_dict[node] = {}
        for rdma_dev in rdma_nic_dict[node].keys():
            eth_dev = rdma_nic_dict[node][rdma_dev]['eth_device']
            lshw_bck_nic_dict[node][eth_dev] = {}
            lshw_bck_nic_dict[node][eth_dev]['pci_bus'] = lshw_dict[node][eth_dev]['pci_bus']
            lshw_bck_nic_dict[node][eth_dev]['description'] = lshw_dict[node][eth_dev]['description']
            lshw_bck_nic_dict[node][eth_dev]['rdma_dev'] = rdma_dev
    return lshw_bck_nic_dict


def get_nearest_bus_no(target_hex: str, candidates: list[str]) -> str:
    """
    Return the nearest matching hex value (as one of the candidate strings).
    - Inputs are hex strings like '0x1f', '1F', '-0x10' (case-insensitive).
    - Tie-breaker: picks the smaller numeric value.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    t = int(target_hex, 16)
    return min(candidates, key=lambda s: (abs(int(s, 16) - t), int(s, 16)))


def get_gpu_nic_mapping_dict(
    phdl,
):
    gpu_nic_dict = {}
    gpu_pcie_dict = rocm_plib.get_gpu_pcie_bus_dict(phdl)
    lshw_dict = get_lshw_backend_nic_dict(phdl)

    nic_bus_dict = {}
    for node in lshw_dict.keys():
        nic_bus_dict[node] = []
        for eth_dev in lshw_dict[node].keys():
            nic_pci = lshw_dict[node][eth_dev]['pci_bus']
            match = re.search('[0-9a-f]+\:([0-9a-f]+)\:[0-9a-f]+\.[0-9a-f]', nic_pci, re.I)
            nic_bus_no = match.group(1)
            nic_bus_dict[node].append(nic_bus_no)

    for node in gpu_pcie_dict.keys():
        gpu_nic_dict[node] = {}
        for card in gpu_pcie_dict[node].keys():
            gpu_nic_dict[node][card] = {}
            gpu_bdf = gpu_pcie_dict[node][card]['PCI Bus']
            gpu_nic_dict[node][card]['gpu_bdf'] = gpu_bdf
            match = re.search('[0-9a-f]+\:([0-9a-f]+)\:[0-9a-f]+\.[0-9a-f]', gpu_bdf, re.I)
            bus_no = match.group(1)

            # find nearest nic bus no.
            nic_bus_list = nic_bus_dict[node]
            print(f'nic_bus_list = {nic_bus_list}')
            print(f'bus_no = {bus_no}')

            nearest_nic_bus_no = get_nearest_bus_no(bus_no, nic_bus_list)

            print(f'nic_bus_list = {nic_bus_list}')
            print(f'nearest_nic_bus_no = {nearest_nic_bus_no}')
            for eth_dev in lshw_dict[node].keys():
                match = re.search(
                    '[0-9a-f]+\:([0-9a-f]+)\:[0-9a-f]+\.[0-9a-f]', lshw_dict[node][eth_dev]['pci_bus'], re.I
                )
                lshw_bus_no = match.group(1)
                if hex(int(nearest_nic_bus_no, 16)) == hex(int(lshw_bus_no, 16)):
                    gpu_nic_dict[node][card]['eth_dev'] = eth_dev
                    gpu_nic_dict[node][card]['rdma_dev'] = lshw_dict[node][eth_dev]['rdma_dev']
                    gpu_nic_dict[node][card]['nic_bdf'] = lshw_dict[node][eth_dev]['pci_bus']
                    continue
    print(gpu_nic_dict)
    return gpu_nic_dict


def get_gpu_numa_dict(phdl):
    gpu_numa_dict = {}
    gpu_pcie_dict = rocm_plib.get_gpu_pcie_bus_dict(phdl)

    first_node = list(gpu_pcie_dict.keys())[0]
    card_list = list(gpu_pcie_dict[first_node].keys())
    for node in gpu_pcie_dict.keys():
        gpu_numa_dict[node] = {}
        for card in card_list:
            gpu_numa_dict[node][card] = {}

    # Build cmd list
    for card in card_list:
        cmd_list = []
        for node in gpu_pcie_dict.keys():
            gpu_bdf = str(gpu_pcie_dict[node][card]['PCI Bus']).lower()
            cmd_list.append(f'cat /sys/bus/pci/devices/{gpu_bdf}/local_cpulist')

        out_dict = phdl.exec_cmd_list(cmd_list)
        for node in out_dict.keys():
            gpu_numa_dict[node][card]['local_cpulist'] = str(out_dict[node]).rstrip('\n').rstrip('\r')

    # Build cmd list
    for card in card_list:
        cmd_list = []
        for node in gpu_pcie_dict.keys():
            gpu_bdf = str(gpu_pcie_dict[node][card]['PCI Bus']).lower()
            cmd_list.append(f'cat /sys/bus/pci/devices/{gpu_bdf}/numa_node')

        out_dict = phdl.exec_cmd_list(cmd_list)
        for node in out_dict.keys():
            gpu_numa_dict[node][card]['numa_node'] = str(out_dict[node]).rstrip('\n').rstrip('\r')

    print(gpu_numa_dict)
    return gpu_numa_dict
    """Detect Linux distro and return distro type"""
    out_dict = phdl.exec('cat /etc/os-release')
    for node, output in out_dict.items():
        if re.search('ubuntu|debian', output, re.I):
            return 'debian'
        elif re.search('rhel|centos|fedora|rocky|alma', output, re.I):
            return 'rhel'
        elif re.search('sles|suse', output, re.I):
            return 'suse'
    return 'debian'  # Default fallback

def get_package_manager_cmd(distro, action='install'):
    #based on distro find  correct cmds
    commands = {
        'debian': {
            'update': 'sudo apt-get update -y',
            'install': 'sudo apt-get install -y',
            'remove': 'sudo apt-get remove -y',
        },
        'rhel': {
            'update': 'sudo dnf check-update || true',
            'install': 'sudo dnf install -y',
            'remove': 'sudo dnf remove -y',
        },
        'suse': {
            'update': 'sudo zypper refresh',
            'install': 'sudo zypper install -y',
            'remove': 'sudo zypper remove -y',
        }
    }
    return commands.get(distro, commands['debian']).get(action, '')

def translate_package_name(package, distro):
    #some pkgs differ in naming across distro. use right ones
    if distro == 'debian':
        return package
    
    # Package name mappings for non-Debian distros
    package_map = {
        'rhel': {
            'libgtest-dev': 'gtest-devel',
            'libpci-dev': 'pciutils-devel',
            'libpci3': 'pciutils',
            'libyaml-cpp-dev': 'yaml-cpp-devel',
            'libibverbs-dev': 'libibverbs-devel',
            'librdmacm-dev': 'librdmacm-devel',
            'libibumad-dev': 'libibumad-devel',
            'openmpi-bin': 'openmpi',
            'openmpi-common': 'openmpi',
            'libopenmpi-dev': 'openmpi-devel',
            'hipblaslt-dev': 'hipblaslt-devel',
            'ibverbs-providers': 'rdma-core',
            'build-essential': ['gcc', 'gcc-c++', 'make'],  # Install separately as multiple packages
            'apt-transport-https': None,  # Not needed
            'software-properties-common': None,  # Not needed
        },
        'suse': {
            'libgtest-dev': 'gtest',
            'libpci-dev': 'pciutils-devel',
            'libpci3': 'pciutils',
            'libyaml-cpp-dev': 'libyaml-cpp-devel',
            'openmpi-bin': 'openmpi',
            'libopenmpi-dev': 'openmpi-devel',
            'build-essential': ['gcc', 'gcc-c++', 'make']
        }
    }
    
    map_dict = package_map.get(distro, {})
    return map_dict.get(package, package)

def map_packages(distro, packages):
    #update package names and flatten if needed
    result = []
    for pkg in packages:
        translated = translate_package_name(pkg, distro)
        
        if translated is None:
            # Skip packages not needed on this distro
            continue
        elif isinstance(translated, list):
            # Package expands to multiple packages (e.g., build-essential)
            result.extend(translated)
        else:
            # Single package
            result.append(translated)
    
    return result

def install_package(hdl, package, distro=None, timeout=200):
    # Install a package using the appropriate package manager for the detected distro.
    if distro is None:
        distro = detect_distro(hdl)
    
    # Translate package name for the distro
    translated_pkg = translate_package_name(package, distro)
    
    # Skip if package is not needed on this distro
    if translated_pkg is None:
        log.info(f'Package {package} not needed on {distro}, skipping')
        return {}
    
    # Get the install command
    install_cmd = get_package_manager_cmd(distro, 'install')
    
    # Execute and return out_dict for error checking
    out_dict = hdl.exec(f'{install_cmd} {translated_pkg}', timeout=timeout)
    return out_dict

def update_package_cache(hdl, distro=None, timeout=600):
    """
    Update package manager cache and return out_dict for error checking.
    
    Returns:
        out_dict: Command output dictionary for error checking
    """
    if distro is None:
        distro = detect_distro(hdl)
    
    update_cmd = get_package_manager_cmd(distro, 'update')
    out_dict = hdl.exec(update_cmd, timeout=timeout)
    return out_dict

def install_build_tools(hdl, distro, timeout=200):
    """Install build tools appropriate for the distro"""
    if distro == 'debian':
        return install_package(hdl, 'build-essential', distro, timeout)
    elif distro in ['rhel', 'suse']:
        results = {}
        for pkg in ['gcc', 'gcc-c++', 'make']:
            out_dict = install_package(hdl, pkg, distro, timeout)
            results.update(out_dict)
        return results

#linux distro discovery and package management
def detect_distro(phdl):
    """Detect Linux distro and return distro type"""
    out_dict = phdl.exec('cat /etc/os-release')
    for node, output in out_dict.items():
        if re.search('ubuntu|debian', output, re.I):
            return 'debian'
        elif re.search('rhel|centos|fedora|rocky|alma', output, re.I):
            return 'rhel'
        elif re.search('sles|suse', output, re.I):
            return 'suse'
    return 'debian'  # Default fallback
    """Detect Linux distro and return distro type"""
    out_dict = phdl.exec('cat /etc/os-release')
    for node, output in out_dict.items():
        if re.search('ubuntu|debian', output, re.I):
            return 'debian'
        elif re.search('rhel|centos|fedora|rocky|alma', output, re.I):
            return 'rhel'
        elif re.search('sles|suse', output, re.I):
            return 'suse'
    return 'debian'  # Default fallback

def get_package_manager_cmd(distro, action='install'):
    #based on distro find  correct cmds
    commands = {
        'debian': {
            'update': 'sudo apt-get update -y',
            'install': 'sudo apt-get install -y',
            'remove': 'sudo apt-get remove -y',
        },
        'rhel': {
            'update': 'sudo dnf check-update || true',
            'install': 'sudo dnf install -y',
            'remove': 'sudo dnf remove -y',
        },
        'suse': {
            'update': 'sudo zypper refresh',
            'install': 'sudo zypper install -y',
            'remove': 'sudo zypper remove -y',
        }
    }
    return commands.get(distro, commands['debian']).get(action, '')

def translate_package_name(package, distro):
    #some pkgs differ in naming across distro. use right ones
    if distro == 'debian':
        return package
    
    # Package name mappings for non-Debian distros
    package_map = {
        'rhel': {
            'libgtest-dev': 'gtest-devel',
            'libpci-dev': 'pciutils-devel',
            'libpci3': 'pciutils',
            'libyaml-cpp-dev': 'yaml-cpp-devel',
            'libibverbs-dev': 'libibverbs-devel',
            'librdmacm-dev': 'librdmacm-devel',
            'libibumad-dev': 'libibumad-devel',
            'openmpi-bin': 'openmpi',
            'openmpi-common': 'openmpi',
            'libopenmpi-dev': 'openmpi-devel',
            'hipblaslt-dev': 'hipblaslt-devel',
            'ibverbs-providers': 'rdma-core',
            'build-essential': ['gcc', 'gcc-c++', 'make'],  # Install separately as multiple packages
            'apt-transport-https': None,  # Not needed
            'software-properties-common': None,  # Not needed
        },
        'suse': {
            'libgtest-dev': 'gtest',
            'libpci-dev': 'pciutils-devel',
            'libpci3': 'pciutils',
            'libyaml-cpp-dev': 'libyaml-cpp-devel',
            'openmpi-bin': 'openmpi',
            'libopenmpi-dev': 'openmpi-devel',
            'build-essential': ['gcc', 'gcc-c++', 'make']
        }
    }
    
    map_dict = package_map.get(distro, {})
    return map_dict.get(package, package)

def map_packages(distro, packages):
    #update package names and flatten if needed
    result = []
    for pkg in packages:
        translated = translate_package_name(pkg, distro)
        
        if translated is None:
            # Skip packages not needed on this distro
            continue
        elif isinstance(translated, list):
            # Package expands to multiple packages (e.g., build-essential)
            result.extend(translated)
        else:
            # Single package
            result.append(translated)
    
    return result

def install_package(hdl, package, distro=None, timeout=200):
    # Install a package using the appropriate package manager for the detected distro.
    if distro is None:
        distro = detect_distro(hdl)
    
    # Translate package name for the distro
    translated_pkg = translate_package_name(package, distro)
    
    # Skip if package is not needed on this distro
    if translated_pkg is None:
        log.info(f'Package {package} not needed on {distro}, skipping')
        return {}
    
    # Get the install command
    install_cmd = get_package_manager_cmd(distro, 'install')
    
    # Execute and return out_dict for error checking
    out_dict = hdl.exec(f'{install_cmd} {translated_pkg}', timeout=timeout)
    return out_dict

def update_package_cache(hdl, distro=None, timeout=600):
    """
    Update package manager cache and return out_dict for error checking.
    
    Returns:
        out_dict: Command output dictionary for error checking
    """
    if distro is None:
        distro = detect_distro(hdl)
    
    update_cmd = get_package_manager_cmd(distro, 'update')
    out_dict = hdl.exec(update_cmd, timeout=timeout)
    return out_dict

def install_build_tools(hdl, distro, timeout=200):
    """Install build tools appropriate for the distro"""
    if distro == 'debian':
        return install_package(hdl, 'build-essential', distro, timeout)
    elif distro in ['rhel', 'suse']:
        results = {}
        for pkg in ['gcc', 'gcc-c++', 'make']:
            out_dict = install_package(hdl, pkg, distro, timeout)
            results.update(out_dict)
        return results