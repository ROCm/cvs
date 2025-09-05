import os
import re
import sys

from utils_lib import *
from rocm_plib import *
import linux_utils


err_patterns_dict = {

    'gpu_reset': 'GPU reset begin|GPU hang|cp might be in an unrecoverable state|fence wait loop timeout expired',
    'crash': 'crashed|Traceback|cut here|Bug:|Call Trace|RIP:|end trace|amdgpu: Fatal error|show_stack|dump_stack|fault ',
    'test_fail': 'Test failure',
    'fault': 'no-retry page fault|Illegal register access|PROTECTION_FAULT_STATUS',
    'driver': 'Queue preemption failed for queue|Failed to evict process queues|Runlist is getting oversubscribed|No more SDMA queue to allocate|Expect reduced ROCm performance|amdgpu: process pid',
    'hardware': 'hardware error|hardware fail|ras error|uncorrectable|correctable err',
    'network': 'NIC Link is Down|link is down|ib_uverb'

}


err_stats_pattern = 'err|drop|discard|overflow|fcs|nak|uncorrect|loss'
warn_stats_pattern = 'retry|timeout|exceeded|ooo|retransmit'
threshold_stats_pattern = 'cnp|ecn'
threshold_counter_val=1000



def verify_gpu_pcie_bus_width( phdl, expected_cards=8, gpu_pcie_speed=32, gpu_pcie_width=16):
    """
    Verify that all GPUs across nodes are operating at the expected PCIe link speed and width.

    Parameters:
      phdl: parallel ssh handle to execute on all nodes:
            - exec_cmd_list(list_of_shell_cmds) -> dict mapping node to command output.
              used when the commands to run can vary from node to node
      expected_cards (int): Expected number of GPUs per node to validate.
      gpu_pcie_speed (int): Expected PCIe link speed in GT/s (e.g., 32 for PCIe Gen5).
      gpu_pcie_width (int): Expected PCIe link width (e.g., 16 for x16).

    Behavior:
      - Retrieves a per-node, per-GPU PCIe bus mapping via get_gpu_pcie_bus_dict(phdl).
      - Ensures each node has the expected number of GPUs.
      - For each GPU index (card_no), runs lspci on each node to read the link status (LnkSta).
      - Validates the reported link Speed (e.g., "Speed 32GT/s") and Width (e.g., "Width x16").
      - Fails the test (via fail_test) if any node/GPU does not meet the expected link attributes or is downgraded.

    Assumptions:
      - get_gpu_pcie_bus_dict returns a structure like:
        {
          "nodeA": {
            "card0": {"PCI Bus": "0000:0b:00.0", ...},
            "card1": {"PCI Bus": "0000:0c:00.0", ...},
            ...
          },
          "nodeB": { ... }
        }
      - The host has lspci installed and sudo permission to run it without password prompts.
      - fail_test is available in scope and raises/aborts on failure.
    """

    err_dict = {}
    # Query per-node GPU PCIe bus mapping (node -> card -> PCI attributes)
    out_dict = get_gpu_pcie_bus_dict( phdl )
    for node in out_dict.keys():
        err_dict[node] = []
    cmd_list = []

    # Use first node to derive an initial card list (structure validation)
    node_0 = list(out_dict.keys())[0]
    card_list = list(out_dict[node_0].keys())

    # Check each node has the expected number of GPUs
    for node in out_dict.keys():
        card_list = out_dict[node].keys()
        if len(card_list)!= expected_cards:
            fail_test(f'ERROR !! Number of cards not matching expected no {expected_cards} on node {node}')

    # Let us take the last card_list for further checks ..
    # Iterate over the (last seen) card indices and validate link for each across all nodes
    # Note: This assumes all nodes expose the same set of card indices/keys.
    for card_no in card_list:
        cmd_list = []

        # Build lspci commands for the same card index across all nodes
        for node in out_dict.keys(): 
            bus_no = out_dict[node][card_no]['PCI Bus']
            cmd_list.append(f'sudo lspci -vvv -s {bus_no} | grep "LnkSta:" --color=never')

        # Execute all commands; expect dict mapping node -> command output text
        pci_dict = phdl.exec_cmd_list( cmd_list )

        # Validate each node's output for speed, width, and downgraded status
        for p_node in pci_dict.keys():
            # Validate negotiated speed (e.g., "Speed 32GT/s")
            if not re.search( f'Speed {gpu_pcie_speed}GT', pci_dict[p_node] ):
                msg = f'ERROR !! PCIe speed not matching for bus {bus_no} on node {p_node}, expected {gpu_pcie_speed}GT/s but got {pci_dict[p_node]}'
                fail_test(msg)
                err_dict[node].append(msg)
            # Validate negotiated width (e.g., "Width x16")
            if not re.search( f'Width x{gpu_pcie_width}', pci_dict[p_node] ):
                msg = f'ERROR !! PCIe width not matching for bus {bus_no} on node {p_node}, expected {gpu_pcie_width} but got {pci_dict[p_node]}'
                fail_test(msg)
                err_dict[node].append(msg)
            # Check for link downgrade indications
            if re.search( 'downgrade', pci_dict[p_node] ):
                msg = f'ERROR !! PCIe in downgraded state for bus {bus_no} on node {p_node}'
                fail_test(f'ERROR !! PCIe in downgraded state for bus {bus_no} on node {p_node}')
                err_dict[node].append(msg)
    return err_dict




def verify_gpu_pcie_errors( phdl ):
    """
    Check PCIe error-related GPU metrics across nodes and fail if thresholds are exceeded.

    Parameters:
      phdl: Process/host handle used by helper utilities (works with get_gpu_metrics_dict).

    Behavior:
      - Retrieves per-node, per-GPU PCIe-related counters via get_gpu_metrics_dict(phdl).
      - Validates three accumulated counters for each GPU:
          * 'pcie_l0_to_recov_count_acc (Count)'   -> L0 to Recovery transitions
          * 'pcie_nak_sent_count_acc (Count)'      -> NAK packets sent
          * 'pcie_nak_rcvd_count_acc (Count)'      -> NAK packets received
      - If any counter exceeds 100, fail_test is invoked immediately with details.

    Assumptions:
      - get_gpu_metrics_dict returns a nested dict of strings for counts, which are cast to int.
      - fail_test is available in scope and will raise/abort the test.
      - Threshold is hardcoded at 100; adjust as needed for environment.
    """

    err_dict = {}
    # Query a nested dict of GPU metrics per node:
    # { node: { card: { 'metric_name': 'value_str', ... }, ... }, ... }
    metrics_dict = get_gpu_metrics_dict( phdl )
    for node in metrics_dict.keys():
        err_dict[node] = []

     # Iterate through each node's metrics
    for node in metrics_dict.keys():
        d_dict = metrics_dict[node]

        # Check each GPU's (card) PCIe-related counters
        for card in d_dict.keys():

            # Count of transitions from L0 to Recovery; high counts may indicate link instability
            if int(d_dict[card]['pcie_l0_to_recov_count_acc (Count)']) > 100:
                msg = f"ERROR !! Node {node} card {card} having higher L0 to recovery counter - \
                    {d_dict[card]['pcie_l0_to_recov_count_acc (Count)']}"
                fail_test(msg)
                err_dict[node].append(msg)

            # Number of NAKs sent; persistent/excessive values may indicate PCIe issues
            if int(d_dict[card]['pcie_nak_sent_count_acc (Count)']) > 100:
                msg = f"ERROR !! Node {node} card {card} having PCIe NAK Sent counter - \
                    {d_dict[card]['pcie_nak_sent_count_acc (Count)']}"
                fail_test(msg)
                err_dict[node].append(msg)

            # Number of NAKs received; persistent/excessive values may indicate PCIe issues
            if int(d_dict[card]['pcie_nak_rcvd_count_acc (Count)']) > 100:
                msg = f"ERROR !! Node {node} card {card} having PCIe NAK Recv counter - \
                    {d_dict[card]['pcie_nak_rcvd_count_acc (Count)']}"
                fail_test(msg)
                err_dict[node].append(msg)

    return err_dict




def verify_dmesg_for_errors(phdl, start_time_dict, end_time_dict, till_end_flag=True ):
   
    """
    Scan kernel logs (dmesg) between given start and end timestamps across nodes
    and fail if any known error patterns are detected.

    Parameters:
      phdl: pssh handle that can execute remote shell commands via .exec(cmd) -> dict.
      start_time_dict (dict): Mapping of node -> start timestamp string (human-readable, e.g., 'Mon Jan  2 03:04:05').
      end_time_dict (dict): Mapping of node -> end timestamp string (same format as start).

    Behavior:
      - Extracts a human-readable timestamp prefix (e.g., 'Mon Jan  2 03:04:05') from provided times.
      - Uses dmesg -T (human-readable timestamps) piped to awk to slice the log from start to end.
      - Filters out lines containing 'ALLOWED' or 'DENIED' (non-fatal/noisy) via egrep -v.
      - Scans each line against a set of known error regex patterns (err_patterns_dict).
      - Immediately fails the test via fail_test if any error pattern is seen.

    Assumptions:
      - err_patterns_dict is defined in scope: {name: regex_pattern, ...}.
      - phdl.exec(cmd) returns a dict: { node: stdout_str }.
      - Input timestamps contain a prefix matching the regex used here.
      - sudo is available and does not prompt for a password when running dmesg.

    Notes:
      - This function fails fast on the first detected error to shorten feedback cycles.
      - If start/end times are not aligned with dmesg -T formatting, the awk range may be empty.
      - Consider handling cases where regex extraction fails (no match) to avoid attribute errors.
    """
 
    print('scan dmesg')

    err_dict = {}

    # Use the first node key to derive the time window to search .. assume cluster has NTP
    node0 = list(start_time_dict.keys())[0]
    start_time = start_time_dict[node0].rstrip("\n")
    end_time = end_time_dict[node0].rstrip("\n")

    # Extract the "Mon Jan  2 03:04:05" style prefix from the provided timestamps
    pattern = r"([a-zA-Z]+\s+[a-zA-Z]+\s+[0-9]+\s+[0-9]+\:[0-9]+)"
    match = re.search( pattern, start_time)
    start_pattern = match.group(1)

    # Extract end timestamp prefix similarly
    pattern = r"([a-zA-Z]+\s+[a-zA-Z]+\s+[0-9]+\s+[0-9]+\:[0-9]+)"
    match = re.search( pattern, end_time)
    end_pattern = match.group(1)

    # Pull human-readable dmesg and slice the lines between start and end timestamps.
    # Filter out allowed/denied lines to reduce noise. Return is a dict keyed by node.
    if till_end_flag:
        output_dict = phdl.exec(f"sudo dmesg -T | sed -n '/{start_pattern}/,$p' | egrep -v 'ALLOWED|DENIED' --color=never")
    else:
        output_dict = phdl.exec(f"sudo dmesg -T | awk '/{start_pattern}.*/,/{end_pattern}.*/' | egrep -v 'ALLOWED|DENIED' --color=never")
    #print(output_dict)
    for node in output_dict.keys():
        err_dict[node] = []

    # Iterate through each node's sliced dmesg and scan for known error patterns
    for node in output_dict.keys():
        for line in output_dict[node].split("\n"):
            for err_key in err_patterns_dict.keys():
                if re.search( f'{err_patterns_dict[err_key]}', line, re.I ):
                    fail_test(f'ERROR - Failue pattern ** {line} ** seen in Dmesg')
                    err_dict[node].append(line)

    return err_dict





def verify_nic_link_flap( phdl ):

    """
    Verify NIC health by checking link flap/reset/down counters from ethtool stats
    and scanning dmesg for link-down events across all nodes.

    Parameters:
      phdl: A pssh handle that supports:
            - exec(cmd: str) -> dict[node: str, output: str]
            Used here to run dmesg on each node.

    Behavior:
      - Queries per-node NIC statistics via linux_utils.get_nic_ethtool_stats_dict(phdl).
      - For each interface, looks for counters whose names contain 'reset', 'down', or 'flap'
        (case-insensitive). If any such counter has a value > 0, fails the test.
      - Scans kernel logs (dmesg) on each node for "NIC Link is down" messages,
        failing the test if found.

    Assumptions:
      - linux_utils.get_nic_ethtool_stats_dict returns a nested dictionary:
          {
            node1: {
              "eth0": {"some_counter": "0", "rx_reset_count": "1", ...},
              "eth1": {...}
            },
            node2: {...}
          }
        Counter values are strings that can be cast to int.
      - fail_test is available in scope to signal a test failure (raises/exits).
      - sudo dmesg is allowed without prompting.
    """

    err_dict = {}

    # Gather NIC stats per node and per interface using ethtool
    nic_stats_dict = linux_utils.get_nic_ethtool_stats_dict( phdl )
    for node in nic_stats_dict:
        err_dict[node] = []

    # Iterate through nodes, interfaces, and counters
    for node in nic_stats_dict:
        for intf in nic_stats_dict[node].keys():
            for counter in nic_stats_dict[node][intf].keys():
                # Consider counters indicating resets, link down, or flaps
                if re.search( 'reset|down|flap', counter, re.I ):
                    # Fail if any of those counters is non-zero
                    if int(nic_stats_dict[node][intf][counter]) > 0:
                        msg = f'ERROR !! {node} {intf} {counter} {nic_stats_dict[node][intf][counter]}'
                        fail_test(msg)
                        err_dict[node].append(msg)

    # Scan dmesg for link-related messages across nodes
    nic_dmesg_dict = phdl.exec( 'sudo dmesg -T | grep -i link --color=never')

    # If the kernel logs show link down messages, fail the test
    for node in nic_dmesg_dict.keys():
        if re.search( 'NIC Link is down', nic_dmesg_dict[node], re.I ):
            msg = f'ERROR !! NIC Link down dmesg logs seen on node {node}'
            fail_test(msg)
            err_dict[node].append(msg)
    return err_dict




def verify_host_lspci( phdl, pcie_speed=32, pcie_width=16 ):

    """
    Verify host-side PCIe link attributes for AMD GPUs against expected speed/width and
    check for PCIe error indications using lspci.

    Parameters:
      phdl: Process/host handle abstraction that supports:
            - exec(cmd: str) -> dict[node -> stdout]
            - exec_cmd_list(cmds: list[str]) -> dict[node -> stdout]
      pcie_speed (int): Expected PCIe link speed in GT/s (e.g., 32 for PCIe Gen4).
      pcie_width (int): Expected PCIe link width (e.g., 16 for x16).

    Behavior:
      - Uses `amd-smi list` to collect GPU BDFs per node.
      - For each BDF, runs `lspci -vvv -s <BDF>` and inspects "LnkSta" line for:
          * Negotiated Speed (e.g., "Speed 32GT/s")
          * Negotiated Width (e.g., "Width x16")
      - Also scans for strings indicating PCIe error flags in the lspci output.
      - Calls fail_test on the first mismatch or error condition detected.

    Notes:
      - Assumes `amd-smi` and `lspci` are available and `sudo` can run non-interactively.
      - This function currently indexes `bdf_list` derived from the last processed node
        when iterating all nodes; it assumes all nodes share the same BDF list. If nodes
        differ, logic should be adjusted to iterate per-node BDFs.
      - The final regex/error check appears inverted: it fails when NO error keywords
        are found, while the message states there ARE error indications. Adjust logic
        if needed.
    """

    err_dict = {}
    # Gather BDFs (Bus:Device.Function) for AMD GPUs on each node
    out_dict = phdl.exec('sudo amd-smi list | grep BDF --color=never')
    for node in out_dict.keys():
        err_dict[node] = []

    bdf_dict = {}
    for node in out_dict.keys():
        bdf_list_out =  out_dict[node]
        pattern = r"BDF:\s+([0-9a-f\:\.]+)"
        # Extract all BDF tokens from the amd-smi output for this node
        bdf_list = re.findall( pattern, out_dict[node], re.I )
        bdf_dict[node] = bdf_list

    # Iterate over BDFs using the most recently assigned bdf_list
    # Note: This assumes all nodes share the same BDF indices; see note above.
    for i in range(0,len(bdf_list)):
        cmd_list = []

        # Build lspci queries (one per node) for the current BDF slot
        for node in out_dict.keys():
            cmd_list.append(f'sudo lspci -vvv -s {bdf_list[i]} | grep Sta: --color=never')

        # Execute the list of commands across nodes; returns mapping of node -> lspci output
        lspci_dict = phdl.exec_cmd_list(cmd_list)

        # Validate lspci-reported link speed, width, and error indicators
        for lnode in lspci_dict.keys():
            # Check negotiated link speed (e.g., "LnkSta: Speed 32GT/s")
            pattern = r"LnkSta:\s+Speed\s+" + str(pcie_speed) + "GT"
            if not re.search( pattern, lspci_dict[lnode], re.I ):
                msg = f'ERROR !! PCIe Link speed not matching with expected output on node {lnode} - expected {pcie_speed}'
                fail_test(msg)
                err_dict[node].append(msg)

            # Check negotiated link width (e.g., "Width x16")
            pattern = r"Width\s+x" + str(pcie_width)
            if not re.search( pattern, lspci_dict[lnode], re.I ):
                msg = f'ERROR !! PCIe Link width not matching with expected output on node {lnode} - expected {pcie_width}'
                fail_test(msg)
                err_dict[node].append(msg)

            # Check for PCIe error indications (correctable/uncorrectable, etc.)
            # NOTE: Logic seems inverted: this fails when none of these tokens are present.
            # If the intent is to fail when ANY are present, remove the 'not'.
            if not re.search( 'CorrErr+|FatalErr+|RxErr+|BadTLP+|BadDLLP+|DLP+|SDES+|ExOF+|TLP+|MalfTLP+', lspci_dict[lnode], re.I ):
                msg = f'ERROR !! PCIe corretable or uncorrectable error indications on Host side on node {lnode}'
                fail_test(msg)
                err_dict[node].append(msg)

    return err_dict



def full_journalctl_scan( phdl ):

    """
    Scan kernel logs via journalctl across nodes for GPU/interrupt/error-related issues
    and fail if any known error patterns are detected.

    Parameters:
      phdl: Host/process handle abstraction that supports:
            - exec(cmd: str) -> dict[node: str, output: str]
              Executes the given command on all relevant nodes and returns a mapping
              of node identifier to the command's stdout.

    Behavior:
      - Runs 'journalctl -k' (kernel messages) filtered through egrep for high-signal
        keywords: amdgpu, interrupt, error, fail, timeout, fault.
      - Iterates over each node?s output and checks each line against regex patterns
        defined in err_patterns_dict.
      - If any line matches a known failure pattern, calls fail_test immediately
        with a descriptive message including the offending line and node.

    Assumptions:
      - err_patterns_dict is defined in scope and maps keys to regex patterns:
          { "pattern_name": "regex", ... }
      - fail_test(...) is available in scope to signal test failure (raise/exit).
      - sudo can run journalctl without interactive password prompts.
      - journalctl stores kernel logs (equivalent to dmesg -k), and system uses journald.

    Notes:
      - This function fails fast on the first matching error; consider aggregating
        all matches per node if you prefer a single consolidated report.
      - The initial egrep reduces volume; ensure it doesn?t hide relevant lines
        not containing those keywords if broader scanning is desired.
    """

    err_dict = {}
    # Fetch kernel logs filtered for likely error indicators across nodes
    out_dict = phdl.exec( 'sudo journalctl -k | egrep "amdgpu|interrupt|error|fail|timeout|fault"')
    for node in out_dict.keys():
        err_dict[node] = []

    # For each node, scan each line against known error regex patterns
    for node in out_dict.keys():
        for line in out_dict[node].split("\n"):
            for err_key in err_patterns_dict.keys():
                # Case-insensitive match against the configured error patterns
                if re.search( f'{err_patterns_dict[err_key]}', line, re.I ):
                    msg = f'ERROR - Failure pattern *** {line} *** seen in Dmesg on node {node}'
                    fail_test(msg)
                    err_dict[node].append(line)
    return err_dict






def full_dmesg_scan(phdl,):
    """
    Scan dmesg across nodes for known error patterns and fail on first match.

    Parameters:
      phdl: Host/process handle abstraction that supports:
            - exec(cmd: str) -> dict[node: str]
              Executes the given command on all target nodes and returns a map of node to output.

    Behavior:
      - Runs `dmesg -T` to get human-readable timestamps from kernel logs on each node.
      - Filters out noisy lines:
          * Excludes lines containing 'initialized'
          * Excludes lines containing 'ALLOWED' or 'DENIED' (case-sensitive as written)
      - Iterates through each remaining line and checks for matches against regex patterns
        defined in err_patterns_dict.
      - On the first match, invokes fail_test with a message indicating the failing line and node.

    Assumptions:
      - err_patterns_dict is available in scope and maps labels to regex patterns:
          { "pattern_name": "regex", ... }
      - fail_test(...) is available in scope and raises/exits on failure.
      - sudo can run dmesg without interactive password prompts.
      - phdl.exec runs the command on all relevant nodes and returns their outputs.

    Notes:
      - This function fails fast. If you need a consolidated report of all matches,
        consider accumulating them and failing once at the end.
      - Current filters are simple grep/egrep; adjust if they exclude useful diagnostics.
      - Consider adding case-insensitive filtering (e.g., grep -i) where appropriate.
    """

    print('scan dmesg')

    err_dict = {}

    # Pull human-readable kernel logs and filter out common noise
    output_dict = phdl.exec(f"sudo dmesg -T | grep -v initialized | egrep -v 'ALLOWED|DENIED' --color=never")
    for node in output_dict.keys():
        err_dict[node] = []

    # Iterate node-by-node through the filtered dmesg output
    for node in output_dict.keys():
        # Examine each line for any known failure pattern 
        for line in output_dict[node].split("\n"):
            for err_key in err_patterns_dict.keys():
                # Case-insensitive match against configured error regexes
                if re.search( f'{err_patterns_dict[err_key]}', line, re.I ):
                    msg = f'ERROR - Failure pattern *** {line} *** seen in Dmesg on node {node}'
                    fail_test(msg)
                    err_dict[node].append(line)
    return err_dict





def verify_driver_errors( phdl ):
    """
    Scan dmesg for AMDGPU driver-related errors across nodes and fail on detection.

    Parameters:
      phdl: Host/process handle abstraction that supports:
            - exec(cmd: str) -> dict[node: str]
              Runs the given command across relevant nodes and returns a mapping
              from node identifier to the command's stdout.

    Behavior:
      - Executes `dmesg -T` (human-readable timestamps) and filters for lines containing
        'amdgpu' and any of the error indicators: fail, error, reset, hang, traceback.
      - If any node's output contains 'fail' or 'error' (case-insensitive), triggers a
        test failure via fail_test.

    Assumptions:
      - sudo can run dmesg without interactive password prompts.
      - fail_test(...) is available in scope and raises/aborts on failure.
      - re (regex module) is imported and available.
      - phdl.exec returns a dict mapping node -> stdout string.

    Notes:
      - This function currently only checks for 'fail' or 'error' in the aggregated filtered output,
        not the other keywords ('reset', 'hang', 'traceback') even though they are included in the grep;
        consider aligning the regex check with the grep filter.
      - The fail_test message is truncated in the original code snippet. Consider completing it
        to include node identification and perhaps sample matching lines for context.
      - The function fails fast (first detection); if you want a comprehensive report across nodes,
        aggregate matches and report them collectively before failing.
    """

    print('Scan for AMD GPU driver errors')

    err_dict = {}
    # Collect AMDGPU-related kernel messages filtered for likely error terms
    out_dict = phdl.exec("sudo dmesg -T | grep -i amdgpu  | egrep -i 'fail|error|reset|hang|traceback' --color=never")
    for node in out_dict.keys():
        err_dict[node] = []


    # For each node, if the filtered output contains 'fail' or 'error', mark test as failed.
    for node in out_dict.keys():
        for line in out_dict[node].split("\n"):
            if re.search( 'fail|error', line, re.I ):
                msg = f'ERROR !! amdgpu driver errors detected in dmesg on node {node}: {line}'
                fail_test(msg)
                err_dict[node].append(line)
    return err_dict




def create_cluster_metrics_snapshot( phdl, ):
    """
    Collect a point-in-time snapshot of key cluster metrics across nodes.

    Parameters:
      phdl: Host/process handle abstraction used by utility functions to execute
            commands remotely and gather per-node metrics.

    Returns:
      dict: A dictionary containing multiple categories of metrics aggregated
            across the cluster:
        - 'eth_stats': Per-node NIC ethtool statistics
        - 'rdma_stats': Per-node RDMA device statistics
        - 'gpu_ras_stats': Per-node AMD SMI RAS (Reliability, Availability, Serviceability) metrics
        - 'gpu_pcie_stats': Per-node AMD SMI PCIe-related metrics
        # - 'gpu_stats': (optional) Per-node general GPU metrics (currently commented out)
    """

    s_dict = {}

    # Gather Ethernet interface statistics via ethtool across all nodes
    s_dict['eth_stats'] = linux_utils.get_nic_ethtool_stats_dict( phdl )

    # Gather RDMA device statistics (e.g., counters per HCA/NIC) across all nodes
    s_dict['rdma_stats'] = linux_utils.get_rdma_stats_dict( phdl )

    # Gather GPU RAS metrics from amd-smi (e.g., ECC, XGMI, memory error counters)
    s_dict['gpu_ras_stats'] = get_amd_smi_ras_metrics_dict( phdl )

    # Gather GPU PCIe metrics from amd-smi (e.g., link speed/width, errors)
    s_dict['gpu_pcie_stats'] = get_amd_smi_pcie_metrics_dict( phdl )

    #s_dict['gpu_stats'] = get_gpu_metrics_dict( phdl )
    return s_dict





def get_metrics_snapshot_diff_dict( s_dict_before, s_dict_after ):

    """
    Compute a nested dictionary of deltas between two cluster metrics snapshots.

    Parameters:
      s_dict_before (dict): Snapshot taken "before", structured as:
                            { category: { node: { device: { stat_name: value, ... } } } }
      s_dict_after  (dict): Snapshot taken "after", with the same structure/keys as s_dict_before.

    Returns:
      dict: diff_dict with the same nested keys (category -> node -> device -> stat_name),
            where each stat_name holds the numeric difference (after - before) for:
              - integer stats
              - numeric strings (e.g., "123"), while skipping textual strings
              - lists are ignored (no diffs computed)

    Notes/Assumptions:
      - Both snapshots have identical nested keys and shapes.
      - Values can be ints, numeric strings, or textual strings; lists are skipped.
      - Textual strings are detected via regex and excluded from diffing.
      - This function does not handle missing keys or type mismatches gracefully.
    """

    diff_dict = {}
    # Pre-initialize the nested structure of diff_dict to mirror s_dict_before
    # key_nam will be like rdma_stats, ethtool_stats etc.

    for key_nam in s_dict_before.keys():
        diff_dict[key_nam] = {}
        for node in s_dict_before[key_nam].keys():
            diff_dict[key_nam][node] = {}
            for dev_nam in s_dict_before[key_nam][node].keys():
                diff_dict[key_nam][node][dev_nam] = {}

    # Walk through all stats and compute numeric deltas where applicable
    for key_nam in s_dict_before.keys():
        for node in s_dict_before[key_nam].keys():
            for dev_nam in s_dict_before[key_nam][node].keys():
                for stat_nam in s_dict_before[key_nam][node][dev_nam].keys():

                    # Skip lists entirely (no delta computed for list-type stats)
                    if not isinstance( s_dict_before[key_nam][node][dev_nam][stat_nam], list ):

                        # If value is a string, only diff when it looks numeric (not textual)
                        if isinstance( s_dict_before[key_nam][node][dev_nam][stat_nam], str ):
                            # Pattern matches alphabetic/._- tokens; if found, treat as textual and skip
                            pattern = r"[a-z\.\_\-]+"
                            if not re.search( pattern, s_dict_before[key_nam][node][dev_nam][stat_nam], re.I ):
                                # Treat as numeric string; compute integer delta
                                diff_dict[key_nam][node][dev_nam][stat_nam] =  \
                                       int(s_dict_after[key_nam][node][dev_nam][stat_nam]) - \
                                       int(s_dict_before[key_nam][node][dev_nam][stat_nam])
                        # If value is an integer, compute direct delta
                        elif isinstance( s_dict_before[key_nam][node][dev_nam][stat_nam], int ): 
                            diff_dict[key_nam][node][dev_nam][stat_nam] =  \
                                s_dict_after[key_nam][node][dev_nam][stat_nam] - \
                                s_dict_before[key_nam][node][dev_nam][stat_nam]

    return diff_dict
                      
            


# top level key_nam is type like eth_stats, rdma_stats
def compare_cluster_metrics_snapshots( s_dict_before, s_dict_after ):
    """
    Compare two cluster metrics snapshots and report counters that increased,
    classifying them as warnings, errors, or threshold-based warnings.

    Parameters:
      s_dict_before (dict): "Before" snapshot with nested structure:
                            { category: { node: { device: { stat_name: value, ... } } } }
      s_dict_after  (dict): "After" snapshot with identical structure/keys to s_dict_before.

    Behavior:
      - Computes a diff dictionary (after - before) via get_metrics_snapshot_diff_dict.
      - Iterates all stats and:
          * If stat name matches warn_stats_pattern and delta > 0: logs/prints a WARN.
          * If stat name matches err_stats_pattern and delta > 0: logs/prints an ERROR.
          * If stat name matches threshold_stats_pattern and delta > threshold_counter_val:
              logs/prints a WARN (labeled as threshold warn).
      - Prints start and completion messages to track progress.

    Assumptions:
      - get_metrics_snapshot_diff_dict is available and returns numeric deltas for comparable stats.
      - warn_stats_pattern, err_stats_pattern, threshold_stats_pattern are valid regex strings
        available in scope and intended to match stat names.
      - threshold_counter_val is an int threshold available in scope for threshold-based warnings.
      - log is a logger with warn/error methods; re (regex) is imported.
      - All diff values for matched stats are numeric or numeric strings (castable to int).
    """

    print('Compare 2 cluster snapshots')
    # err_dict will capture the ERROR, WARN log messages at a node level which have seen 
    # increment in values for any error or warning counters. The same can be obtained
    # from the complete test log by doing a grep on ERROR|WARN and snapshot
    err_dict = {}

    # err_stats_diff_dict is a collection of node, device level error/warning metrics which 
    # have actually incremented during the snapshot period with the values of before and
    # after for each of those metrics. This will be used to display the values before and
    # after in the snaphot Diff tables for GPU and NIC metrics
    err_stats_diff_dict = {}

    # Compute per-stat deltas between the two snapshots:
    # diff_dict mirrors the structure {category -> node -> device -> stat_name -> delta}.
    diff_dict = get_metrics_snapshot_diff_dict( s_dict_before, s_dict_after )

    # Walk the nested structure and evaluate each stat's delta against patterns/thresholds.
    for key_nam in diff_dict.keys():                                      # category (e.g., eth_stats, rdma_stats)
        err_dict[key_nam] = {}
        err_stats_diff_dict[key_nam] = {}
        for node in diff_dict[key_nam].keys():                            # node identifier
            err_dict[key_nam][node] = []
            err_stats_diff_dict[key_nam][node] = {}
            for dev_nam in diff_dict[key_nam][node].keys():               # device/interface identifier
                err_stats_diff_dict[key_nam][node][dev_nam] = {}
                for stat_nam in diff_dict[key_nam][node][dev_nam].keys(): # metric/statistic identifier

                    # Warning counters: log when any positive increase occurs
                    if re.search( f'{warn_stats_pattern}', stat_nam, re.I ):
                        if int(diff_dict[key_nam][node][dev_nam][stat_nam]) > 0:
                            msg = f'WARN !! cluster snapshot showing some warning counters going up - {key_nam} {node} {dev_nam} {stat_nam} have incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}'
                            log.warn(msg)
                            print(msg)
                            err_dict[key_nam][node].append(msg)
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam] = {}
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'] = \
                               s_dict_before[key_nam][node][dev_nam][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after'] = \
                               s_dict_after[key_nam][node][dev_nam][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['diff'] = \
                               int(err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after']) - \
                               int(err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'])


                    # Error counters: log when any positive increase occurs
                    elif re.search( f'{err_stats_pattern}', stat_nam, re.I ):
                        if int(diff_dict[key_nam][node][dev_nam][stat_nam]) > 0:
                            msg = f'ERROR !! cluster snapshot showing some error counters going up - {key_nam} {node} {dev_nam} {stat_nam} have incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}'
                            log.error(msg)
                            print(msg)
                            err_dict[key_nam][node].append(msg)
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam] = {}
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'] = \
                            s_dict_before[key_nam][node][dev_nam][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after'] = \
                            s_dict_after[key_nam][node][dev_nam][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['diff'] = \
                            int(err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after']) - \
                            int(err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'])

                    # Threshold-based warning counters: require delta to exceed a configured threshold
                    elif re.search( f'{threshold_stats_pattern}', stat_nam, re.I ):
                        if int(diff_dict[key_nam][node][dev_nam][stat_nam]) > threshold_counter_val:
                            msg = f'WARN !! cluster snapshot showing some threshold warn counters going up - {key_nam} {node} {dev_nam} {stat_nam} have incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}'
                            log.warn(msg)
                            print(msg)
                            err_dict[key_nam][node].append(msg)
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam] = {}
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'] = \
                            s_dict_before[key_nam][node][dev_nam][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after'] = \
                            s_dict_after[key_nam][node][dev_nam][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['diff'] = \
                            int(err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after']) - \
                            int(err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'])

    print('Completed comparing the cluster snapshots')
    return err_dict, err_stats_diff_dict



