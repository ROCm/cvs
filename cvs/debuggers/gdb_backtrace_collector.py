'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import sys
import logging
import argparse
import re
import json
from collections import defaultdict, Counter

from cvs.debuggers.base import DebugPlugin
from cvs.lib import parallel_ssh_lib
from cvs.lib.utils_lib import resolve_cluster_config_placeholders

log = logging.getLogger()


class GdbBacktraceCollectorDebugger(DebugPlugin):
    """Debugger for collecting GDB backtraces from hung processes across cluster nodes."""

    PROCESS_NOT_FOUND_MSG = "Process not found"

    def get_name(self):
        return "gdb_backtrace_collector"

    def get_description(self):
        return "Collect GDB backtraces from hung processes to diagnose cluster-wide hangs"

    def get_parser(self):
        parser = argparse.ArgumentParser(description="Collect GDB backtraces from hung processes")
        parser.add_argument("--cluster_file", required=True, help="Path to cluster configuration JSON file")
        parser.add_argument(
            "--process", required=True, help="Process name pattern to filter (e.g., 'mpirun|all_reduce_perf')"
        )
        parser.add_argument("--pid", type=int, help="Specific PID to debug (requires --node)")
        parser.add_argument("--node", help="Specific node IP to debug (required with --pid)")
        parser.add_argument(
            "--output_format", choices=["console", "json"], default="console", help="Output format (default: console)"
        )
        parser.add_argument("--timeout", type=int, default=30, help="GDB attach timeout in seconds (default: 30)")
        return parser

    def debug(self, args):
        """Execute GDB backtrace collection."""
        cluster_dict = self._load_cluster_config(args.cluster_file)
        node_list, username, priv_key_file, password = self._extract_cluster_info(cluster_dict)
        phdl = self._initialize_ssh_handler(node_list, username, priv_key_file, password, args.timeout)

        print(f"Collecting GDB backtraces from {len(node_list)} nodes...")

        if args.pid:
            if not args.node:
                print("ERROR: --node is required when using --pid (PIDs are unique per node)")
                sys.exit(1)
            results = self._collect_backtrace_by_pid(phdl, args.pid, args.timeout, node_ip=args.node)
        else:
            results = self._collect_backtrace_by_process(phdl, args.process, args.timeout)

        if args.output_format == "console":
            self._print_console_report(results, node_list)
        elif args.output_format == "json":
            self._print_json_report(results)

    def _load_cluster_config(self, cluster_file):
        """Load and resolve cluster configuration."""
        with open(cluster_file) as json_file:
            cluster_dict = json.load(json_file)
        cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
        return cluster_dict

    def _extract_cluster_info(self, cluster_dict):
        """Extract node list and authentication info from cluster config."""
        node_dict = cluster_dict['node_dict']
        # Handle different cluster config formats
        if node_dict and isinstance(list(node_dict.values())[0], dict):
            # Format: {"hostname": {"vpc_ip": "ip", "bmc_ip": "bmc"}}
            node_list = [node_info.get('vpc_ip', hostname) for hostname, node_info in node_dict.items()]
        else:
            # Format: {"hostname": "ip"}
            node_list = list(node_dict.values())

        username = cluster_dict['username']
        priv_key_file = cluster_dict.get('priv_key_file')
        password = cluster_dict.get('password')

        if not node_list:
            print("ERROR: No nodes found in cluster configuration")
            sys.exit(1)

        if not username:
            print("ERROR: Username not found in cluster configuration")
            sys.exit(1)

        return node_list, username, priv_key_file, password

    def _initialize_ssh_handler(self, node_list, username, priv_key_file, password, timeout=30):
        """Initialize parallel SSH handler with error tolerance for unreachable hosts."""
        if priv_key_file:
            phdl = parallel_ssh_lib.Pssh(
                log, node_list, user=username, pkey=priv_key_file, stop_on_errors=False, timeout=timeout
            )
        elif password:
            phdl = parallel_ssh_lib.Pssh(
                log, node_list, user=username, password=password, stop_on_errors=False, timeout=timeout
            )
        else:
            print("ERROR: No authentication method (priv_key_file or password) found in cluster configuration")
            sys.exit(1)
        return phdl

    def _collect_backtrace_by_pid(self, phdl, pid, timeout, node_ip):
        """Collect backtrace for a specific PID on a specific node.

        Args:
            phdl: Parallel SSH handler
            pid: Process ID to debug
            timeout: GDB attach timeout in seconds
            node_ip: Specific node IP to target
        """
        # Validate node_ip is in reachable hosts
        if node_ip not in phdl.reachable_hosts:
            print(f"ERROR: Node {node_ip} is not in the list of reachable hosts")
            sys.exit(1)

        # Build command for GDB execution on the target host
        gdb_cmd = f"timeout {timeout} sudo gdb -p {pid} --batch -ex 'thread apply all bt' 2>&1 || echo 'GDB_ERROR'"

        # Create a temporary Pssh instance for just the target node
        if phdl.password:
            target_phdl = parallel_ssh_lib.Pssh(
                log, [node_ip], user=phdl.user, password=phdl.password, stop_on_errors=False, timeout=timeout
            )
        else:
            target_phdl = parallel_ssh_lib.Pssh(
                log, [node_ip], user=phdl.user, pkey=phdl.pkey, stop_on_errors=False, timeout=timeout
            )

        # Run GDB on the target node
        output = target_phdl.exec_cmd_list([gdb_cmd])

        results = {}
        stdout = output[node_ip][0]  # Get the output from the single command
        error = 'GDB_ERROR' in stdout or 'No such process' in stdout
        results[node_ip] = {
            'pids': [pid],
            'backtraces': [
                {
                    'pid': pid,
                    'backtrace': stdout,
                    'error': error,
                }
            ],
        }
        return results

    def _collect_backtrace_by_process(self, phdl, process_filter, timeout):
        """Collect backtraces for processes matching the filter."""
        pid_map = self._find_process_pids(phdl, process_filter)
        return self._collect_backtraces_from_pids(phdl, pid_map, timeout)

    def _find_process_pids(self, phdl, process_filter):
        """Find PIDs of processes matching the filter on each host."""
        ps_cmd = f"pgrep -f '{process_filter}'"
        ps_output = phdl.exec(ps_cmd, timeout=10)
        pid_map = {}
        for host in phdl.reachable_hosts:
            pid_strs = ps_output.get(host, '').strip().split('\n')
            pids = []
            for pid_str in pid_strs:
                pid_str = pid_str.strip()
                if pid_str and pid_str.isdigit():
                    pids.append(int(pid_str))
            pid_map[host] = pids if pids else []
        return pid_map

    def _collect_backtraces_from_pids(self, phdl, pid_map, timeout):
        """Collect backtraces for given PIDs on each host."""
        cmd_list = []
        sep = '===GDB_SEP==='

        for host, pids in pid_map.items():
            if pids:
                cmds = []
                for pid in pids:
                    gdb_cmd = f"timeout {timeout} sudo gdb -p {pid} --batch -ex 'thread apply all bt' 2>&1 || echo 'GDB_ERROR'"
                    cmds.append(gdb_cmd)
                # Chain commands with separator
                full_cmd = f'echo "{sep}"; '.join(cmds) + f'; echo "{sep}"'
                cmd_list.append(full_cmd)
            else:
                cmd_list.append(f"echo '{self.PROCESS_NOT_FOUND_MSG}'")

        gdb_output = phdl.exec_cmd_list(cmd_list, timeout=10)

        results = {}
        for i, host in enumerate(phdl.reachable_hosts):
            pids = pid_map[host]
            output = gdb_output[host]
            if not pids:
                # No processes found
                results[host] = {'pids': [], 'backtraces': [], 'error': True, 'message': self.PROCESS_NOT_FOUND_MSG}
            else:
                # Split output by separator
                parts = output.split(f'{sep}\n')
                # Remove empty parts and the trailing sep
                backtrace_strs = [part.strip() for part in parts if part.strip() and part.strip() != sep]
                has_error = any('GDB_ERROR' in bt or 'No such process' in bt for bt in backtrace_strs)
                # Create list of dicts
                bt_list = []
                for pid, bt_str in zip(pids, backtrace_strs):
                    top_function = self._extract_top_function(bt_str)
                    bt_list.append({'pid': pid, 'backtrace': bt_str, 'top_function': top_function})
                results[host] = {'pids': pids, 'backtraces': bt_list, 'error': has_error}
        return results

    def _print_console_report(self, results, node_list):
        """Print color-coded console report with aggregation."""
        hung_functions = defaultdict(list)
        error_nodes = []
        no_process_nodes = []

        backtraces_by_node = defaultdict(list)

        for host, data in results.items():
            if data.get('error', False):
                if data.get('message', '').strip() == self.PROCESS_NOT_FOUND_MSG:
                    no_process_nodes.append(host)
                else:
                    error_nodes.append(host)
            else:
                # Process all backtraces for this host
                backtraces_by_node[host] = data['backtraces']
                for bt in data['backtraces']:
                    top_function = bt['top_function']
                    hung_functions[top_function].append((host, bt['pid']))

        print("\n" + "=" * 80)
        print("GDB BACKTRACE COLLECTION REPORT")
        print("=" * 80)

        total_nodes = len(node_list)
        analyzed_nodes = len([h for h in results if not results[h].get('error', False)])
        total_processes = sum(len(data.get('backtraces', [])) for data in results.values() if not data.get('error', True))

        print(f"\nTotal nodes: {total_nodes}")
        print(f"Analyzed nodes: {analyzed_nodes}")
        print(f"Total processes analyzed: {total_processes}")
        print(f"Error nodes: {len(error_nodes)}")
        print(f"No process found: {len(no_process_nodes)}")

        if hung_functions:
            print("\n\033[1;33mHUNG FUNCTION SUMMARY:\033[0m")
            for func, host_pid_list in sorted(hung_functions.items(), key=lambda x: len(x[1]), reverse=True):
                hosts = [h for h, p in host_pid_list]
                node_count = len(set(hosts))  # Unique hosts
                process_count = len(host_pid_list)  # Total process instances
                process_percentage = (process_count / total_processes) * 100 if total_processes > 0 else 0
                color = "\033[1;31m" if process_percentage > 10 else "\033[1;33m"
                print(
                    f"{color}{process_count} processes ({process_percentage:.1f}% of total, {node_count} nodes) hung in: {func}\033[0m"
                )
                if node_count <= 5:
                    unique_hosts = list(set(hosts))
                    print(f"  Nodes: {', '.join(unique_hosts)}")

        # Analysis section
        if backtraces_by_node:
            print("\n\033[1;36mHANG ANALYSIS AND EDUCATED GUESSES:\033[0m")

            # Collect all functions
            all_funcs = []
            for bt_list in backtraces_by_node.values():
                all_funcs.extend([bt['top_function'] for bt in bt_list])

            func_counts = Counter(all_funcs)
            if func_counts:
                most_common_func, most_common_count = func_counts.most_common(1)[0]
                print(f"- Most common hang function: {most_common_func} ({most_common_count} processes)")

                # Identify potential root cause nodes
                potential_causes = []
                for node, bt_list in backtraces_by_node.items():
                    node_funcs = [bt['top_function'] for bt in bt_list]
                    if most_common_func not in node_funcs or node_funcs.count(most_common_func) < len(node_funcs):
                        potential_causes.append(node)

                if potential_causes:
                    print(f"- Potential root cause nodes (processes not all in {most_common_func}): {', '.join(potential_causes)}")
                    for node in potential_causes:
                        bt_list = backtraces_by_node[node]
                        funcs = [bt['top_function'] for bt in bt_list]
                        unique_funcs = set(funcs)
                        func_counts_node = Counter(funcs)
                        print(f"  - {node}: {dict(func_counts_node)}")
                    print("  Recommendation: Investigate these nodes first, as they have processes hung in different functions, potentially indicating the source of the hang.")
                else:
                    print("- All processes across all nodes are hung in the same function. This suggests a cluster-wide issue:")
                    print("  - Possible deadlock in collective communication (e.g., NCCL)")
                    print("  - Network connectivity or hardware failure")
                    print("  - Shared resource contention or synchronization bug")
                    print("  Recommendation: Check network logs, RCCL version compatibility, and hardware status.")

                # Additional insights based on function names
                if any('nccl' in func.lower() or 'NCCL' in func for func in func_counts.keys()):
                    print("- Hang involves NCCL functions: Likely a communication layer issue. Check RCCL logs, network topology, and process ranks.")
                elif any('poll' in func.lower() or 'wait' in func.lower() for func in func_counts.keys()):
                    print("- Processes are waiting on I/O or synchronization. Check for blocking operations or deadlocks in user code.")
                elif any('sched_yield' in func for func in func_counts.keys()):
                    print("- Processes are yielding CPU. Possible busy-wait loops or resource starvation.")
                else:
                    print("- Unknown hang pattern. Manual inspection of full backtraces recommended for deeper analysis.")
            else:
                print("- No functions extracted from backtraces.")

        if error_nodes:
            print(f"\n\033[1;31mERROR NODES ({len(error_nodes)}):\033[0m")
            for host in error_nodes[:10]:
                print(f"  {host}")
            if len(error_nodes) > 10:
                print(f"  ... and {len(error_nodes) - 10} more")

        print("\n" + "=" * 80)

    def _print_json_report(self, results):
        """Print results in JSON format."""
        import json

        print(json.dumps(results, indent=2))

    def _extract_top_function(self, backtrace):
        """Extract the top-most meaningful function from backtrace."""
        lines = backtrace.split('\n')
        for line in lines:
            match = re.search(r'#\d+\s+.*\s+in\s+(\S+)', line)
            if match:
                func_name = match.group(1)
                if not func_name.startswith('0x'):
                    return func_name
        return "unknown_function"


def main():
    debugger = GdbBacktraceCollectorDebugger()
    parser = debugger.get_parser()
    args = parser.parse_args()
    debugger.debug(args)


if __name__ == "__main__":
    main()
