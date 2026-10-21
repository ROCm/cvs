'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import datetime
import re

from cvs.lib.utils_lib import *
from cvs.lib.rocm_plib import *
from cvs.lib import dmesg_noise
from cvs.lib import linux_utils
from cvs.lib import node_scraper_adapter


err_patterns_dict = {
    'gpu_reset': 'GPU reset begin|GPU hang|cp might be in an unrecoverable state|fence wait loop timeout expired',
    'crash': r'crashed|Traceback|cut here|Bug:|Call Trace|RIP:|end trace|amdgpu: Fatal error|segfault|show_stack|dump_stack|\bfault\b',
    'test_fail': 'Test failure',
    # A zero PROTECTION_FAULT_STATUS is a cleared register, not a fault.
    'fault': r'no-retry page fault|Illegal register access|PROTECTION_FAULT_STATUS(?!\s*:\s*0x0+\b)',
    # Note: amdgpu oversubscription messages ('Runlist is getting oversubscribed',
    # 'Expect reduced ROCm performance') are perf-degrading but not test failures —
    # they're matched as warnings via warn_patterns_dict below. See AMD docs:
    # https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/oversubscription.html
    'driver': 'Queue preemption failed for queue|Failed to evict process queues|No more SDMA queue to allocate|amdgpu: process pid',
    'hardware': r'hardware error|hardware fail|ras error|(?:uncorrectable|correctable err)(?!.*initialized)'
    r'|EDAC (?:MC\d+: \d+ [CU]E |DEVICE\d+: [CU]E: )',
    'network': (
        'NIC Link is Down|link is down|ib_uverb|error cqe|cqe_dump:|'
        r'(?<!dump )CQE(?!: syndrome 0x0)|queue catastrophic|CQ error'
    ),
}


# warn_patterns_dict captures kernel events that indicate the GPU entered a
# perf-degrading state but the workload itself still completes correctly. Matches
# emit a WARN to the run log (no fail_test) so reviewers know the perf numbers
# from this run were measured under a degraded HW state and should not be trusted
# blindly for regression comparisons.
warn_patterns_dict = {
    # GPU runlist / VM context oversubscription. Per AMD docs, when this fires the
    # HW scheduler is round-robining queues and inactive queues can block the GPU
    # for millisecond-scale windows. Triggers: >24 user-mode compute queues per
    # GPU, >11 VM context slots, or >1 process using cooperative workgroups.
    # https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/oversubscription.html
    'oversubscribed': 'Runlist is getting oversubscribed|Expect reduced ROCm performance',
}


err_stats_pattern = 'err|drop|discard|overflow|fcs|nak|uncorrect|loss'
warn_stats_pattern = 'retry|timeout|exceeded|ooo|retransmit'
threshold_stats_pattern = 'cnp|ecn'
threshold_counter_val = 1000


# Map CVS's historical dmesg categories (err_patterns_dict) to node-scraper
# event categories, so the legacy patterns can be preserved as analyzer
# extensions on top of node-scraper's built-in pattern table.
_CVS_DMESG_CATEGORY_MAP = {
    'gpu_reset': 'RAS',
    'crash': 'SW_DRIVER',
    'test_fail': 'APPLICATION',
    'fault': 'SW_DRIVER',
    'driver': 'SW_DRIVER',
    'hardware': 'RAS',
    'network': 'NETWORK',
}


def cvs_dmesg_error_regex():
    """Return CVS's err_patterns_dict as node-scraper error_regex extensions.

    Each legacy pattern is wrapped as a case-insensitive node-scraper ErrorRegex
    dict so the node-scraper path still flags everything the historical regex
    caught, in addition to node-scraper's built-in pattern table.
    """
    return [
        {
            'regex': f'(?i){pattern}',
            'message': f'CVS {name} pattern',
            'event_category': _CVS_DMESG_CATEGORY_MAP.get(name, 'UNKNOWN'),
        }
        for name, pattern in err_patterns_dict.items()
    ]


# node-scraper reports every kernel err/crit/alert/emerg line it cannot classify as
# an "Unknown dmesg error", so this list is a deliberate allowlist of known-harmless
# messages. Keep each rule anchored to one specific message; never widen it to a
# whole category or priority, which would hide real lockups and I/O errors.
_BENIGN_UNKNOWN_DMESG_RULES = [
    {'message': 'Unknown dmesg error', 'match_regex': r'IPVS: .*no destination available'},
]


def cvs_dmesg_analysis_args():
    """Return the node-scraper analyzer args shared by every CVS dmesg scan."""
    return {
        'error_regex': cvs_dmesg_error_regex(),
        'ignore_match_rules': list(_BENIGN_UNKNOWN_DMESG_RULES),
    }


_NON_FATAL_DMESG_PRIORITIES = frozenset({'WARNING', 'WARN', 'INFO', 'NOTICE', 'DEBUG'})
_ERROR_DMESG_PRIORITIES = frozenset({'ERROR', 'ERR', 'CRITICAL', 'CRIT', 'ALERT', 'EMERGENCY', 'EMERG', 'PANIC'})


def _node_scraper_event_fails_test(event):
    """Return True when a node-scraper event should call fail_test."""
    priority = (event.get('priority') or '').strip().upper()
    if priority in _NON_FATAL_DMESG_PRIORITIES:
        return False
    if not priority or priority in _ERROR_DMESG_PRIORITIES:
        return True
    return False


def _parse_cvs_time(time_str):
    """Convert a `date +"%a %b %e %H:%M"` or `date +"%a %b %e %H:%M:%S"` string
    to a tz-aware datetime.

    Returns None if the string cannot be parsed. The year is assumed to be the
    current year and the timezone the local timezone (CVS already assumes the
    cluster is NTP-synced with the head node). Seconds default to 0 when the
    input string is minute-precision only, for backward compatibility with
    callers that haven't been updated to capture seconds.
    """
    if not time_str:
        return None
    match = re.search(
        r'([A-Za-z]{3})\s+([A-Za-z]{3})\s+(\d+)\s+(\d{1,2}):(\d{2})(?::(\d{2}))?',
        time_str.strip(),
    )
    if not match:
        return None
    try:
        month = datetime.datetime.strptime(match.group(2), '%b').month
    except ValueError:
        return None
    now_local = datetime.datetime.now().astimezone()
    return datetime.datetime(
        now_local.year,
        month,
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
        int(match.group(6) or 0),
        tzinfo=now_local.tzinfo,
    )


def _drop_corrected_mces(output_dict, source_label='Dmesg'):
    """Remove clean corrected host-CPU MCE reports and log one WARN per affected node.

    The kernel already corrected these ("no action required"); uncorrected,
    deferred, poisoned and TCC reports remain in place.
    """
    filtered = {}
    for node, text in output_dict.items():
        kept, status_lines = dmesg_noise.drop_corrected_mce_blocks(text)
        if status_lines:
            log.warning(
                f'WARN - {len(status_lines)} corrected host-CPU MCE(s) in {source_label} on node {node} '
                f'not treated as failures (kernel: no action required); first: {status_lines[0]}'
            )
        filtered[node] = kept
    return filtered


def _log_degraded_gpu_lines(output_dict):
    """Log each warn_patterns_dict match; node-scraper does not report these lines."""
    for node, text in output_dict.items():
        for line in (text or '').splitlines():
            for warn_key, pattern in warn_patterns_dict.items():
                if re.search(pattern, line, re.I):
                    log.warning(
                        f'WARN - GPU in degraded state ({warn_key}) on {node}: {line.strip()} '
                        f'-- perf numbers from this run may not be trustworthy'
                    )


def _node_scraper_scan(output_dict, analysis_args=None, source_label='Dmesg'):
    """Scan per-node log text with node-scraper and report each detected error.

    Args:
        output_dict: {node: raw_log_text} as returned by phdl.exec.
        analysis_args: optional dict of DmesgAnalyzerArgs fields (e.g.
            error_regex, analysis_range_start/analysis_range_end).
        source_label: label used in the failure message (e.g. 'Dmesg').

    Returns:
        {node: [matched lines]}; calls fail_test for each detected error.
        Corrected MCE blocks are removed first and logged as WARN. Uncorrected
        and deferred status rows are reported verbatim. warn_patterns_dict
        matches are logged as WARN and do not fail.
    """
    range_start = (analysis_args or {}).get('analysis_range_start')
    range_end = (analysis_args or {}).get('analysis_range_end')
    if range_start or range_end:
        # Keep the WARN count and uncorrected-MCE report within the analysis window.
        output_dict = {
            node: node_scraper_adapter.filter_dmesg_range(text, range_start, range_end)
            for node, text in output_dict.items()
        }
    output_dict = _drop_corrected_mces(output_dict, source_label)
    _log_degraded_gpu_lines(output_dict)
    err_dict = {}
    for node in output_dict.keys():
        err_dict[node] = []
        events = node_scraper_adapter.parse_dmesg(output_dict[node], node_name=node, analysis_args=analysis_args)
        for event in events:
            if not _node_scraper_event_fails_test(event):
                priority = (event.get('priority') or '').strip().upper()
                if priority in ('WARNING', 'WARN'):
                    lines = node_scraper_adapter.event_match_lines([event])
                    if lines:
                        log.warning(
                            f'WARN - {lines[0]} seen in {source_label} on node {node} '
                            f'(warning-level kernel event; not failing test)'
                        )
                continue
            if dmesg_noise.dmesg_line_is_benign(node_scraper_adapter.event_match_text(event)):
                continue
            lines = node_scraper_adapter.event_match_lines([event])
            line = lines[0] if lines else ''
            if not line:
                continue
            msg = f'ERROR - Failure pattern *** {line} *** seen in {source_label} on node {node}'
            fail_test(msg)
            err_dict[node].append(line)
        for row in dmesg_noise.uncorrected_mce_status_lines(output_dict[node]):
            fail_test(
                f'ERROR - Failure pattern *** Uncorrected host-CPU MCE: {row} *** seen in {source_label} on node {node}'
            )
            err_dict[node].append(row)
    return err_dict


def parse_dmesg_nodes(output_dict, analysis_args=None):
    '''Parse per-node dmesg text with node-scraper. Does not call fail_test.'''
    if analysis_args is None:
        analysis_args = {'error_regex': cvs_dmesg_error_regex()}
    parsed = {}
    for node, text in output_dict.items():
        parsed[node] = node_scraper_adapter.parse_dmesg(
            text or '',
            node_name=str(node),
            analysis_args=analysis_args,
        )
    return parsed


def _collect_kernel_log(phdl, cmd):
    """Fetch kernel-log text without echoing it into the run log."""
    # A historical backlog can contain millions of lines, and scans run repeatedly.
    return phdl.exec(cmd, print_console=False)


def _log_kernel_scan_summary(scan_name, output_dict, err_dict):
    """Log host, fetched-line, and reported-match counts for one scan."""
    line_count = sum(len(str(text or '').splitlines()) for text in output_dict.values())
    match_count = sum(len(lines) for lines in err_dict.values())
    log.info(
        'kernel-log scan %s: %d host(s), %d line(s) scanned, %d match(es)',
        scan_name,
        len(output_dict),
        line_count,
        match_count,
    )


def verify_gpu_pcie_bus_width(phdl, expected_cards=None, gpu_pcie_speed=32, gpu_pcie_width=16):
    """
    Verify that all GPUs across nodes are operating at the expected PCIe link speed and width.

    Parameters:
      phdl: parallel ssh handle to execute on all nodes:
            - exec_cmd_list(list_of_shell_cmds) -> dict mapping node to command output.
              used when the commands to run can vary from node to node
      expected_cards (int, optional): Expected number of GPUs per node to validate.
              GPU count per node varies by platform (e.g. 4 on a Helios-R tray,
              8 on other platforms), so this has no fixed default. If None,
              it is auto-detected from the first node's actual GPU count and
              every other node is checked for agreement with that count.
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
    out_dict = get_gpu_pcie_bus_dict(phdl)
    for node in out_dict.keys():
        err_dict[node] = []
    cmd_list = []

    # Use first node to derive an initial card list (structure validation)
    node_0 = list(out_dict.keys())[0]
    card_list = list(out_dict[node_0].keys())

    if expected_cards is None:
        expected_cards = len(card_list)

    # Check each node has the expected number of GPUs
    for node in out_dict.keys():
        card_list = out_dict[node].keys()
        if len(card_list) != expected_cards:
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
        pci_dict = phdl.exec_cmd_list(cmd_list)

        # Validate each node's output for speed, width, and downgraded status
        for p_node in pci_dict.keys():
            # Validate negotiated speed (e.g., "Speed 32GT/s")
            if not re.search(f'Speed {gpu_pcie_speed}GT', pci_dict[p_node]):
                msg = f'ERROR !! PCIe speed not matching for bus {bus_no} on node {p_node}, expected {gpu_pcie_speed}GT/s but got {pci_dict[p_node]}'
                fail_test(msg)
                err_dict[node].append(msg)
            # Validate negotiated width (e.g., "Width x16")
            if not re.search(f'Width x{gpu_pcie_width}', pci_dict[p_node]):
                msg = f'ERROR !! PCIe width not matching for bus {bus_no} on node {p_node}, expected {gpu_pcie_width} but got {pci_dict[p_node]}'
                fail_test(msg)
                err_dict[node].append(msg)
            # Check for link downgrade indications
            if re.search('downgrade', pci_dict[p_node]):
                msg = f'ERROR !! PCIe in downgraded state for bus {bus_no} on node {p_node}'
                fail_test(f'ERROR !! PCIe in downgraded state for bus {bus_no} on node {p_node}')
                err_dict[node].append(msg)
    return err_dict


def verify_gpu_pcie_errors(phdl):
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
    metrics_dict = get_gpu_metrics_dict(phdl)
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


def verify_dmesg_for_errors(phdl, start_time_dict, end_time_dict, till_end_flag=True):
    """
    Scan kernel logs (dmesg) between given start and end timestamps across nodes
    and fail if any known error patterns are detected. Lines matching warn-only
    patterns are logged via log.warning but do not fail the test.

    Parameters:
      phdl: pssh handle supporting exec(cmd, print_console=False) -> {node: str}.
      start_time_dict (dict): Mapping of node -> start timestamp string
        (`date +"%a %b %e %H:%M:%S"`, e.g. 'Mon Jan  2 03:04:05'; minute precision also accepted).
      end_time_dict (dict): Mapping of node -> end timestamp string (same format as start).
      till_end_flag (bool): When True, scan from the start time to the end of the buffer.

    Behavior:
      - Collects dmesg with ISO timestamps and filters out 'ALLOWED' / 'DENIED' lines.
      - node-scraper keeps only lines in [start, end) (or [start, end of buffer)) and reports
        its built-in patterns plus err_patterns_dict; each match calls fail_test.
      - Clean corrected host-CPU MCE reports are removed before scanning and logged as WARN.
      - warn_patterns_dict matches are logged as WARN; the test still passes, but perf numbers
        from the run should not be trusted blindly.

    Assumptions:
      - The first node's timestamps bound every node (the cluster is NTP-synced).
      - sudo is available and does not prompt for a password when running dmesg.

    Notes:
      - Logs one 'kernel-log scan' INFO summary line; the fetched log text itself is not logged.
    """

    node0 = list(start_time_dict.keys())[0]
    analysis_args = cvs_dmesg_analysis_args()
    start_dt = _parse_cvs_time(start_time_dict[node0])
    if start_dt:
        analysis_args['analysis_range_start'] = start_dt
    if not till_end_flag:
        end_dt = _parse_cvs_time(end_time_dict[node0])
        if end_dt:
            analysis_args['analysis_range_end'] = end_dt
    output_dict = _collect_kernel_log(phdl, "sudo dmesg --time-format iso -x | egrep -v 'ALLOWED|DENIED' --color=never")
    err_dict = _node_scraper_scan(output_dict, analysis_args=analysis_args, source_label='Dmesg')
    _log_kernel_scan_summary('verify_dmesg_for_errors', output_dict, err_dict)
    return err_dict


def verify_nic_link_flap(phdl):
    """
    Verify NIC health by checking link flap/reset/down counters from ethtool stats
    and scanning dmesg for link-down events across all nodes.

    Parameters:
      phdl: A pssh handle that supports:
            - exec(cmd, print_console=False) -> {node: str}
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
      - Logs one 'kernel-log scan' INFO summary line; the fetched log text itself is not logged.
    """

    err_dict = {}

    # Gather NIC stats per node and per interface using ethtool
    nic_stats_dict = linux_utils.get_nic_ethtool_stats_dict(phdl)
    for node in nic_stats_dict:
        err_dict[node] = []

    # Iterate through nodes, interfaces, and counters
    for node in nic_stats_dict:
        for intf in nic_stats_dict[node].keys():
            for counter in nic_stats_dict[node][intf].keys():
                # Consider counters indicating resets, link down, or flaps
                if re.search('reset|down|flap', counter, re.I):
                    # Fail if any of those counters is non-zero
                    if int(nic_stats_dict[node][intf][counter]) > 0:
                        msg = f'ERROR !! {node} {intf} {counter} {nic_stats_dict[node][intf][counter]}'
                        fail_test(msg)
                        err_dict[node].append(msg)

    # Scan dmesg for link-related messages across nodes
    nic_dmesg_dict = _collect_kernel_log(phdl, 'sudo dmesg -T | grep -i link --color=never')

    # If the kernel logs show link down messages, fail the test
    # err_dict also holds the ethtool counter failures above; the kernel-log
    # summary must count only what the dmesg scan itself found.
    dmesg_err_dict = {node: [] for node in nic_dmesg_dict}
    for node in nic_dmesg_dict.keys():
        if re.search('NIC Link is down', nic_dmesg_dict[node], re.I):
            msg = f'ERROR !! NIC Link down dmesg logs seen on node {node}'
            fail_test(msg)
            err_dict[node].append(msg)
            dmesg_err_dict[node].append(msg)
    _log_kernel_scan_summary('verify_nic_link_flap', nic_dmesg_dict, dmesg_err_dict)
    return err_dict


def verify_host_lspci(phdl, pcie_speed=32, pcie_width=16):
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
        pattern = r"BDF:\s+([0-9a-f\:\.]+)"
        # Extract all BDF tokens from the amd-smi output for this node
        bdf_list = re.findall(pattern, out_dict[node], re.I)
        bdf_dict[node] = bdf_list

    # Iterate over BDFs using the most recently assigned bdf_list
    # Note: This assumes all nodes share the same BDF indices; see note above.
    for i in range(0, len(bdf_list)):
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
            if not re.search(pattern, lspci_dict[lnode], re.I):
                msg = f'ERROR !! PCIe Link speed not matching with expected output on node {lnode} - expected {pcie_speed}'
                fail_test(msg)
                err_dict[node].append(msg)

            # Check negotiated link width (e.g., "Width x16")
            pattern = r"Width\s+x" + str(pcie_width)
            if not re.search(pattern, lspci_dict[lnode], re.I):
                msg = f'ERROR !! PCIe Link width not matching with expected output on node {lnode} - expected {pcie_width}'
                fail_test(msg)
                err_dict[node].append(msg)

            # Check for PCIe error indications (correctable/uncorrectable, etc.)
            # NOTE: Logic seems inverted: this fails when none of these tokens are present.
            # If the intent is to fail when ANY are present, remove the 'not'.
            if not re.search(
                'CorrErr+|FatalErr+|RxErr+|BadTLP+|BadDLLP+|DLP+|SDES+|ExOF+|TLP+|MalfTLP+', lspci_dict[lnode], re.I
            ):
                msg = f'ERROR !! PCIe corretable or uncorrectable error indications on Host side on node {lnode}'
                fail_test(msg)
                err_dict[node].append(msg)

    return err_dict


def full_journalctl_scan(phdl, start_time_dict=None):
    """
    Scan kernel logs via journalctl across nodes for GPU/interrupt/error-related issues
    and fail if any known error patterns are detected.

    Parameters:
      phdl: Host/process handle abstraction that supports:
            - exec(cmd, print_console=False) -> {node: str}
              Executes the given command on all relevant nodes and returns a mapping
              of node identifier to the command's stdout.
      start_time_dict: Optional {node: cvs_time_str} (same format produced by
            `date +"%a %b %e %H:%M"` / `...%H:%M:%S`, e.g. from the caller's own
            start-of-run timestamp). When provided, journalctl is asked to only
            return entries `--since` that time (derived from the first node's
            entry), instead of collecting the entire kernel journal.

    Behavior:
      - Reads the kernel journal ('journalctl -k -o short-iso'); --since is applied on the
        node so the whole historic journal is not transferred.
      - node-scraper reports its built-in patterns plus err_patterns_dict; each match calls fail_test.
      - Clean corrected host-CPU MCE reports are removed before scanning and logged as WARN.

    Assumptions:
      - sudo can run journalctl without interactive password prompts.
      - journalctl stores kernel logs (equivalent to dmesg -k), and system uses journald.

    Notes:
      - Logs one 'kernel-log scan' INFO summary line; the fetched log text itself is not logged.
    """

    since_clause = ''
    if start_time_dict:
        node0 = list(start_time_dict.keys())[0]
        start_dt = _parse_cvs_time(start_time_dict[node0])
        if start_dt:
            since_clause = f' --since="{start_dt.strftime("%Y-%m-%d %H:%M:%S")}"'

    output_dict = _collect_kernel_log(phdl, f'sudo journalctl -k -o short-iso{since_clause}')
    err_dict = _node_scraper_scan(
        output_dict,
        analysis_args=cvs_dmesg_analysis_args(),
        source_label='journalctl',
    )
    _log_kernel_scan_summary('full_journalctl_scan', output_dict, err_dict)
    return err_dict


def full_dmesg_scan(
    phdl,
):
    """
    Scan the whole dmesg buffer across nodes for known error patterns and fail on each match.

    Parameters:
      phdl: Host/process handle abstraction that supports:
            - exec(cmd, print_console=False) -> {node: str}
              Executes the given command on all target nodes and returns a map of node to output.

    Behavior:
      - Collects dmesg with ISO timestamps and the decoded level prefix ('--time-format iso -x'),
        dropping lines that contain 'initialized', 'ALLOWED', or 'DENIED'.
      - node-scraper reports its built-in patterns plus err_patterns_dict; each match calls fail_test.
      - Clean corrected host-CPU MCE reports are removed before scanning and logged as WARN.

    Assumptions:
      - sudo can run dmesg without interactive password prompts.

    Notes:
      - Logs one 'kernel-log scan' INFO summary line; the fetched log text itself is not logged.
    """

    output_dict = _collect_kernel_log(
        phdl, "sudo dmesg --time-format iso -x | grep -v initialized | egrep -v 'ALLOWED|DENIED' --color=never"
    )
    err_dict = _node_scraper_scan(
        output_dict,
        analysis_args=cvs_dmesg_analysis_args(),
        source_label='Dmesg',
    )
    _log_kernel_scan_summary('full_dmesg_scan', output_dict, err_dict)
    return err_dict


def verify_driver_errors(phdl):
    """
    Scan dmesg for AMDGPU driver-related errors across nodes and fail on detection.

    Parameters:
      phdl: Host/process handle abstraction that supports:
            - exec(cmd, print_console=False) -> {node: str}
              Runs the given command across relevant nodes and returns a mapping
              from node identifier to the command's stdout.

    Behavior:
      - Parses full dmesg with node-scraper and reports only driver-related events: the
        matched line mentions amdgpu, or node-scraper classifies the event as SW_DRIVER.
      - Each reported event calls fail_test.

    Assumptions:
      - sudo can run dmesg without interactive password prompts.

    Notes:
      - Logs one 'kernel-log scan' INFO summary line; the fetched log text itself is not logged.
    """

    err_dict = {}
    output_dict = _collect_kernel_log(phdl, "sudo dmesg --time-format iso -x | egrep -v 'ALLOWED|DENIED' --color=never")
    for node in output_dict.keys():
        err_dict[node] = []
        events = node_scraper_adapter.parse_dmesg(
            output_dict[node],
            node_name=node,
            analysis_args=cvs_dmesg_analysis_args(),
        )
        for event in events:
            text = node_scraper_adapter.event_match_text(event)
            if dmesg_noise.dmesg_line_is_benign(text):
                continue
            lines = node_scraper_adapter.event_match_lines([event])
            line = lines[0] if lines else ''
            if 'amdgpu' not in line.lower() and event.get('category') != 'SW_DRIVER':
                continue
            priority = (event.get('priority') or '').strip().upper()
            should_fail = _node_scraper_event_fails_test(event) or (
                priority in ('WARNING', 'WARN') and dmesg_noise.line_indicates_driver_error(text)
            )
            if should_fail:
                msg = f'ERROR !! amdgpu driver errors detected in dmesg on node {node}: {line}'
                fail_test(msg)
                err_dict[node].append(line)
    _log_kernel_scan_summary('verify_driver_errors', output_dict, err_dict)
    return err_dict


def create_cluster_metrics_snapshot(
    phdl,
):
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
    s_dict['eth_stats'] = linux_utils.get_nic_ethtool_stats_dict(phdl)

    # Gather RDMA device statistics (e.g., counters per HCA/NIC) across all nodes
    s_dict['rdma_stats'] = linux_utils.get_rdma_stats_dict(phdl)

    # Gather GPU RAS metrics from amd-smi (e.g., ECC, XGMI, memory error counters)
    s_dict['gpu_ras_stats'] = get_amd_smi_ras_metrics_dict(phdl)

    # Gather GPU PCIe metrics from amd-smi (e.g., link speed/width, errors)
    s_dict['gpu_pcie_stats'] = get_amd_smi_pcie_metrics_dict(phdl)

    # s_dict['gpu_stats'] = get_gpu_metrics_dict( phdl )
    return s_dict


def get_metrics_snapshot_diff_dict(s_dict_before, s_dict_after):
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
                    if not isinstance(s_dict_before[key_nam][node][dev_nam][stat_nam], list):
                        # If value is a string, only diff when it looks numeric (not textual)
                        if isinstance(s_dict_before[key_nam][node][dev_nam][stat_nam], str):
                            # Pattern matches alphabetic/._- tokens; if found, treat as textual and skip
                            pattern = r"[a-z\.\_\-]+"
                            if not re.search(pattern, s_dict_before[key_nam][node][dev_nam][stat_nam], re.I):
                                # Treat as numeric string; compute integer delta
                                diff_dict[key_nam][node][dev_nam][stat_nam] = int(
                                    s_dict_after[key_nam][node][dev_nam][stat_nam]
                                ) - int(s_dict_before[key_nam][node][dev_nam][stat_nam])
                        # If value is an integer, compute direct delta
                        elif isinstance(s_dict_before[key_nam][node][dev_nam][stat_nam], int):
                            diff_dict[key_nam][node][dev_nam][stat_nam] = (
                                s_dict_after[key_nam][node][dev_nam][stat_nam]
                                - s_dict_before[key_nam][node][dev_nam][stat_nam]
                            )

    return diff_dict


# top level key_nam is type like eth_stats, rdma_stats
def compare_cluster_metrics_snapshots(s_dict_before, s_dict_after):
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

    log.info('Compare 2 cluster snapshots')
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
    diff_dict = get_metrics_snapshot_diff_dict(s_dict_before, s_dict_after)

    # Walk the nested structure and evaluate each stat's delta against patterns/thresholds.
    for key_nam in diff_dict.keys():  # category (e.g., eth_stats, rdma_stats)
        err_dict[key_nam] = {}
        err_stats_diff_dict[key_nam] = {}
        for node in diff_dict[key_nam].keys():  # node identifier
            err_dict[key_nam][node] = []
            err_stats_diff_dict[key_nam][node] = {}
            for dev_nam in diff_dict[key_nam][node].keys():  # device/interface identifier
                err_stats_diff_dict[key_nam][node][dev_nam] = {}
                for stat_nam in diff_dict[key_nam][node][dev_nam].keys():  # metric/statistic identifier
                    # Warning counters: log when any positive increase occurs
                    if re.search(f'{warn_stats_pattern}', stat_nam, re.I):
                        if int(diff_dict[key_nam][node][dev_nam][stat_nam]) > 0:
                            msg = f'WARN !! cluster snapshot showing some warning counters going up - {key_nam} {node} {dev_nam} {stat_nam} have incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}'
                            log.warning("%s", msg)
                            err_dict[key_nam][node].append(msg)
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam] = {}
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'] = s_dict_before[key_nam][node][
                            dev_nam
                        ][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after'] = s_dict_after[key_nam][node][
                            dev_nam
                        ][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['diff'] = int(
                            err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after']
                        ) - int(err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'])

                    # Error counters: log when any positive increase occurs
                    elif re.search(f'{err_stats_pattern}', stat_nam, re.I):
                        if int(diff_dict[key_nam][node][dev_nam][stat_nam]) > 0:
                            msg = f'ERROR !! cluster snapshot showing some error counters going up - {key_nam} {node} {dev_nam} {stat_nam} have incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}'
                            log.error("%s", msg)
                            err_dict[key_nam][node].append(msg)
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam] = {}
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'] = s_dict_before[key_nam][node][
                            dev_nam
                        ][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after'] = s_dict_after[key_nam][node][
                            dev_nam
                        ][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['diff'] = int(
                            err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after']
                        ) - int(err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'])

                    # Threshold-based warning counters: require delta to exceed a configured threshold
                    elif re.search(f'{threshold_stats_pattern}', stat_nam, re.I):
                        if int(diff_dict[key_nam][node][dev_nam][stat_nam]) > threshold_counter_val:
                            msg = f'WARN !! cluster snapshot showing some threshold warn counters going up - {key_nam} {node} {dev_nam} {stat_nam} have incremented by {diff_dict[key_nam][node][dev_nam][stat_nam]} Before = {s_dict_before[key_nam][node][dev_nam][stat_nam]} After = {s_dict_after[key_nam][node][dev_nam][stat_nam]}'
                            log.warning("%s", msg)
                            err_dict[key_nam][node].append(msg)
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam] = {}
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'] = s_dict_before[key_nam][node][
                            dev_nam
                        ][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after'] = s_dict_after[key_nam][node][
                            dev_nam
                        ][stat_nam]
                        err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['diff'] = int(
                            err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['after']
                        ) - int(err_stats_diff_dict[key_nam][node][dev_nam][stat_nam]['before'])

    log.info('Completed comparing the cluster snapshots')
    return err_dict, err_stats_diff_dict
