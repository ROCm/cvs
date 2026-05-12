'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import sys
import json
import logging
import argparse
import time

from cvs.monitors.base import MonitorPlugin
from cvs.lib import parallel_ssh_lib
from cvs.lib import verify_lib
from cvs.lib import linux_utils
from cvs.lib import rocm_plib
from cvs.lib import html_lib
from cvs.lib import utils_lib

log = logging.getLogger()


def load_cluster_file(cluster_file_path):
    """
    Load and validate a CVS cluster file for the cluster health monitor.

    Parses the JSON cluster file used by ``cvs run`` / ``cvs exec`` and
    extracts the SSH credentials and node list. Path placeholders such as
    ``{user-id}`` are resolved via :func:`utils_lib.resolve_cluster_config_placeholders`.

    Args:
        cluster_file_path (str): Path to the cluster JSON file.

    Returns:
        tuple: ``(node_list, username, priv_key_file)`` where ``node_list``
        is the list of host IPs / hostnames (the keys of ``node_dict``).

    Raises:
        FileNotFoundError: If the cluster file does not exist.
        ValueError: If the cluster file is missing required keys, has no
            nodes, or contains invalid JSON.
    """
    try:
        with open(cluster_file_path, 'r') as f:
            cluster = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in cluster file '{cluster_file_path}': {e}") from e

    cluster = utils_lib.resolve_cluster_config_placeholders(cluster)

    username = cluster.get('username')
    if not username:
        raise ValueError(f"'username' missing from cluster file '{cluster_file_path}'")

    priv_key_file = cluster.get('priv_key_file')
    if not priv_key_file:
        raise ValueError(f"'priv_key_file' missing from cluster file '{cluster_file_path}'")

    node_list = list((cluster.get('node_dict') or {}).keys())
    if not node_list:
        raise ValueError(f"'node_dict' is empty or missing in cluster file '{cluster_file_path}'")

    return node_list, username, priv_key_file


def general_health_checks(
    phdl,
):
    health_dict = {}
    print('Verify General Health Checks')
    # Check PCIe Bus and Width
    try:
        health_dict['gpu_pcie_link'] = verify_lib.verify_gpu_pcie_bus_width(phdl)
    except Exception as e:
        print(f'ERROR running verify_gpu_pcie_bus_width, due to exception {e}')
    # Check Dmesg for errors
    try:
        health_dict['dmesg_scan'] = verify_lib.full_dmesg_scan(phdl)
    except Exception as e:
        print(f'ERROR running full_dmesg_scan, due to exception {e}')
    # Check Dmesg for AMD GPU driver errors
    try:
        health_dict['driver_errors'] = verify_lib.verify_driver_errors(phdl)
    except Exception as e:
        print(f'ERROR running verify_driver_errors, due to exception {e}')
    # journlctl scan
    try:
        health_dict['journlctl_scan'] = verify_lib.full_journalctl_scan(phdl)
    except Exception as e:
        print(f'ERROR running full_journalctl_scan, due to exception {e}')
    # Check for any link flap evidence
    try:
        health_dict['nic_link_flap'] = verify_lib.verify_nic_link_flap(phdl)
    except Exception as e:
        print(f'ERROR running verify_nic_link_flap, due to exception {e}')
    # Check for GPU PCIe errors from amd-smi commands
    try:
        health_dict['gpu_pcie_errors'] = verify_lib.verify_gpu_pcie_errors(phdl)
    except Exception as e:
        print(f'ERROR running verify_gpu_pcie_errors, due to exception {e}')
    # Verify PCIe status from Host OS side ..
    try:
        health_dict['host_pcie'] = verify_lib.verify_host_lspci(phdl)
    except Exception as e:
        print(f'ERROR running verify_host_lspci, due to exception {e}')
    return health_dict


def build_html_report(phdl, html_file, gen_health_dict, start_time, snapshot_err_dict, snapshot_err_stats_dict):
    # stats collection
    try:
        lshw_dict = linux_utils.get_lshw_network_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_lshw_network_dict, due to exception {e}')
    try:
        rdma_nic_dict = linux_utils.get_rdma_nic_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_rdma_nic_dict, due to exception {e}')

    try:
        ip_dict = linux_utils.get_ip_addr_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_ip_addr_dict, due to exception {e}')

    try:
        model_dict = rocm_plib.get_gpu_model_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_gpu_model_dict, due to exception {e}')

    try:
        fw_dict = rocm_plib.get_gpu_fw_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_gpu_fw_dict, due to exception {e}')

    try:
        use_dict = rocm_plib.get_gpu_use_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_gpu_use_dict, due to exception {e}')

    try:
        mem_dict = rocm_plib.get_gpu_mem_use_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_gpu_mem_use_dict, due to exception {e}')

    try:
        metrics_dict = rocm_plib.get_gpu_metrics_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_gpu_mem_metrics_dict, due to exception {e}')

    try:
        amd_dict = rocm_plib.get_amd_smi_metric_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_amd_smi_metric_dict, due to exception {e}')

    try:
        lldp_dict = linux_utils.get_lldp_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_lldp_dict, due to exception {e}')

    try:
        rdma_stats_dict = linux_utils.get_rdma_stats_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_rdma_stats_dict, due to exception {e}')

    try:
        ethtool_stats_dict = linux_utils.get_nic_ethtool_stats_dict(phdl)
    except Exception as e:
        print(f'ERROR running get_nic_ethtool_stats_dict, due to exception {e}')

    # Html headers
    html_lib.build_html_page_header(html_file)

    # LLDP Table
    try:
        html_lib.build_lldp_table(html_file, lldp_dict)
    except Exception as e:
        print(f'ERROR running build_lldp_table, due to exception {e}')

    # GPU Info tables
    try:
        html_lib.build_html_cluster_product_table(html_file, model_dict, fw_dict)
    except Exception as e:
        print(f'ERROR running build_html_cluster_product_table, due to exception {e}')

    try:
        html_lib.build_html_gpu_utilization_table(html_file, use_dict)
    except Exception as e:
        print(f'ERROR running build_html_gpu_utilization_table, due to exception {e}')

    try:
        html_lib.build_html_mem_utilization_table(html_file, mem_dict, amd_dict)
    except Exception as e:
        print(f'ERROR running build_html_mem_utilization_table, due to exception {e}')

    try:
        html_lib.build_html_pcie_xgmi_metrics_table(html_file, metrics_dict, amd_dict)
    except Exception as e:
        print(f'ERROR running build_html_pcie_xgmi_metrics_table, due to exception {e}')

    try:
        html_lib.build_html_error_table(html_file, metrics_dict, amd_dict)
    except Exception as e:
        print(f'ERROR running build_html_error_table, due to exception {e}')

    # NIC Info tables
    try:
        html_lib.build_html_nic_table(html_file, rdma_nic_dict, lshw_dict, ip_dict)
    except Exception as e:
        print(f'ERROR running build_html_nic_table, due to exception {e}')

    try:
        html_lib.build_rdma_stats_table(html_file, rdma_stats_dict)
    except Exception as e:
        print(f'ERROR running build_rdma_stats_table, due to exception {e}')

    try:
        html_lib.build_ethtool_stats_table(html_file, ethtool_stats_dict)
    except Exception as e:
        print(f'ERROR running build_ethtool_stats_table, due to exception {e}')

    # Historic Info Tables
    try:
        html_lib.build_err_log_table(
            html_file, gen_health_dict['dmesg_scan'], 'Dmesg Error Table', 'dmesgerrtable', 'dmesgerrid'
        )
    except Exception as e:
        print(f'ERROR running build_err_log_table for dmesg, due to exception {e}')

    try:
        html_lib.build_err_log_table(
            html_file, gen_health_dict['driver_errors'], 'GPU Driver Error Table', 'gpudrivererrtable', 'gpudrivererrid'
        )
    except Exception as e:
        print(f'ERROR running build_err_log_table, due to exception {e}')

    try:
        html_lib.build_err_log_table(
            html_file, gen_health_dict['journlctl_scan'], 'Journlctl Error Table', 'journlctlerrtable', 'journlctlerrid'
        )
    except Exception as e:
        print(f'ERROR running build_err_log_table, due to exception {e}')

    try:
        html_lib.build_err_log_table(
            html_file, gen_health_dict['gpu_pcie_errors'], 'GPU PCIE Errors Table', 'gpupcieerrtable', 'gpupcieerrid'
        )
    except Exception as e:
        print(f'ERROR running build_err_log_table, due to exception {e}')

    try:
        html_lib.build_err_log_table(
            html_file,
            gen_health_dict['gpu_pcie_link'],
            'GPU PCIE Link Status Errors',
            'gpupcielinktable',
            'gpupcielinkid',
        )
    except Exception as e:
        print(f'ERROR running build_err_log_table, due to exception {e}')

    try:
        html_lib.build_err_log_table(
            html_file,
            gen_health_dict['host_pcie'],
            'Host Side PCIE Status Errors',
            'hostpcielinktable',
            'hostpcielinkid',
        )
    except Exception as e:
        print(f'ERROR running build_err_log_table, due to exception {e}')

    try:
        html_lib.build_err_log_table(
            html_file, gen_health_dict['nic_link_flap'], 'NIC Link Flap Logs Table', 'niclinkflaptable', 'niclinkflapid'
        )
    except Exception as e:
        print(f'ERROR running build_err_log_table, due to exception {e}')

    # Snapshot tables
    # Scan Dmesgs to see any new errors while running the passive health check
    end_time = phdl.exec('date')
    dmesg_diff_dict = verify_lib.verify_dmesg_for_errors(phdl, start_time, end_time)
    html_lib.build_err_log_table(
        html_file, dmesg_diff_dict, 'New Dmesg Errors during snapshotting', 'snapdmesgtable', 'snapdmesgid'
    )

    # Compare the snapshots and use the diff of metrics to see any new errors occurred
    # for GPU or NIC
    html_lib.build_err_log_table(
        html_file,
        snapshot_err_dict['eth_stats'],
        'Snapshot diff logs of any new ethstats errors incrementing across snapshots',
        'snaperrlogsethtable',
        'snaperrlogsethid',
    )

    html_lib.build_err_log_table(
        html_file,
        snapshot_err_dict['gpu_pcie_stats'],
        'Snapshot diff logs of any new GPU PCIe errors incrementing across snapshots',
        'snaperrlogspcietable',
        'snaperrlogspcieid',
    )

    html_lib.build_err_log_table(
        html_file,
        snapshot_err_dict['gpu_ras_stats'],
        'Snapshot diff logs of any new GPU RAS errors incrementing across snapshots',
        'snaperrlogsrastable',
        'snaperrlogsrasid',
    )

    html_lib.build_err_log_table(
        html_file,
        snapshot_err_dict['rdma_stats'],
        'Snapshot diff logs of any new RDMA errors incrementing across snapshots',
        'snaperrlogsrdmatable',
        'snaperrlogsrdmaid',
    )

    html_lib.build_snapshot_stats_diff_table(
        html_file,
        snapshot_err_stats_dict['rdma_stats'],
        'New RDMA errors during snapshotting',
        'snaprdmastatstable',
        'snaprdmastatsid',
    )

    html_lib.build_snapshot_stats_diff_table(
        html_file,
        snapshot_err_stats_dict['eth_stats'],
        'New Eth errors during snapshotting',
        'snapethstatstable',
        'snapethstatsid',
    )

    html_lib.build_snapshot_stats_diff_table(
        html_file,
        snapshot_err_stats_dict['gpu_pcie_stats'],
        'New PCIe errors during snapshotting',
        'snappcieerrtable',
        'snappcieerrid',
    )

    html_lib.build_snapshot_stats_diff_table(
        html_file,
        snapshot_err_stats_dict['gpu_ras_stats'],
        'New GPU RAS errors during snapshotting',
        'snaprastatstable',
        'snaprastatsid',
    )

    # Snapshot Tables..

    # Html footers
    html_lib.build_html_page_footer(html_file)


# Things to do
# Html table comparison of error counters ..
# check and add Pytorch, tensorflow error patterns
# Add NCCL errors
# NIC CLI check - niccli -i 1 pcie -counters
# Congestion checker


class CheckClusterHealthMonitor(MonitorPlugin):
    """Monitor for checking cluster health and generating reports."""

    def get_name(self):
        return "check_cluster_health"

    def get_description(self):
        return "Generate a health report of your cluster by collecting various logs and metrics"

    def get_parser(self):
        parser = argparse.ArgumentParser(description="Check Cluster Health")
        # Source of hosts: either a CVS cluster file (preferred) or a legacy
        # one-IP-per-line hosts file. Exactly one must be provided.
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--cluster_file",
            help=(
                "Path to a CVS cluster JSON file "
                "(see cvs/input/cluster_file/cluster.json). "
                "Provides node list, username, and SSH key. Recommended."
            ),
        )
        source.add_argument(
            "--hosts_file",
            help="[DEPRECATED] File with one host IP/hostname per line. Use --cluster_file instead.",
        )
        # Credentials - only required when --hosts_file is used.
        parser.add_argument("--username", help="SSH username (required with --hosts_file)")
        parser.add_argument("--password", help="SSH password (only valid with --hosts_file)")
        parser.add_argument("--key_file", help="SSH private key file (only valid with --hosts_file)")
        parser.add_argument("--iterations", type=int, default=2, help="Number of iterations to run the checks")
        parser.add_argument(
            "--time_between_iters", type=int, default=60, help="Time duration to sleep between iterations"
        )
        parser.add_argument("--report_file", default="./cluster_report.html", help="Output HTML report file path")
        return parser

    def _resolve_connection(self, args):
        """Return ``(node_list, username, pkey, password)`` based on parsed args.

        Centralizes the cluster_file vs. hosts_file argument handling so the
        ``monitor`` method stays linear and the logic stays unit-testable.
        """
        if args.cluster_file:
            if args.username or args.password or args.key_file:
                print(
                    "ERROR: --username/--password/--key_file are not allowed with --cluster_file; "
                    "credentials come from the cluster file. Aborting."
                )
                sys.exit(1)
            node_list, username, pkey = load_cluster_file(args.cluster_file)
            return node_list, username, pkey, None

        # Legacy --hosts_file path.
        print(
            "WARNING: --hosts_file is deprecated. Switch to --cluster_file (see cvs/input/cluster_file/cluster.json)."
        )
        if not args.username:
            print("ERROR: --username is required when using --hosts_file. Aborting.")
            sys.exit(1)
        if bool(args.password) == bool(args.key_file):
            print("ERROR: exactly one of --password or --key_file is required with --hosts_file. Aborting.")
            sys.exit(1)
        with open(args.hosts_file, "r") as f:
            node_list = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
        if not node_list:
            print(f"ERROR: no hosts found in '{args.hosts_file}'. Aborting.")
            sys.exit(1)
        return node_list, args.username, args.key_file, args.password

    def monitor(self, args):
        """Execute the cluster health check monitoring."""
        node_list, username, pkey, password = self._resolve_connection(args)

        html_report_file = args.report_file

        # Connect to all hosts in the cluster.
        if pkey:
            phdl = parallel_ssh_lib.Pssh(log, node_list, user=username, pkey=pkey)
        else:
            phdl = parallel_ssh_lib.Pssh(log, node_list, user=username, password=password)

        start_time = phdl.exec('date')

        # Run general health checks and scan historic errors
        gen_health_dict = general_health_checks(phdl)

        # Take cluster metrics snapshot before iterations
        snapshot_dict_before = verify_lib.create_cluster_metrics_snapshot(phdl)

        snapshot_iters_dict = {}
        for i in range(1, int(args.iterations) + 1):
            print('#------------------------------------------------------------#')
            print(f'Starting Iteration - {i}')
            print('#------------------------------------------------------------#')
            snapshot_iters_dict[i] = verify_lib.create_cluster_metrics_snapshot(phdl)
            print('#............................................................#')
            print(f'Waiting for {args.time_between_iters} for time between iterations - Iteration {i}')
            print('#............................................................#')
            time.sleep(int(args.time_between_iters))

        print('Completed all iterations, taking final snapshot for comparison')

        snapshot_dict_after = verify_lib.create_cluster_metrics_snapshot(phdl)
        (snapshot_err_logs_dict, snapshot_err_stats_dict) = verify_lib.compare_cluster_metrics_snapshots(
            snapshot_dict_before, snapshot_dict_after
        )

        # Build cluster html report
        build_html_report(
            phdl, html_report_file, gen_health_dict, start_time, snapshot_err_logs_dict, snapshot_err_stats_dict
        )

        print(gen_health_dict)

        print('#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#')
        print('Completed all iterations, script completed, scan logs for ERROR, WARN')
        print('#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#')


# Keep backward compatibility with direct script execution
def main():
    monitor = CheckClusterHealthMonitor()
    parser = monitor.get_parser()
    args = parser.parse_args()
    monitor.monitor(args)


if __name__ == "__main__":
    main()
