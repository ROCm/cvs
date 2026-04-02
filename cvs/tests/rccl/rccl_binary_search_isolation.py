'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest
import time
import copy
import os
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from cvs.lib import globals
from cvs.lib.utils_lib import *
from cvs.lib import rccl_lib
from cvs.lib.parallel_ssh_lib import Pssh

log = globals.log


@dataclass
class ClusterPartition:
    """Represents a partition of cluster nodes for binary search testing."""

    partition_id: str
    nodes: List[str]
    parent_id: Optional[str] = None
    test_output: Optional[str] = None
    meets_threshold: bool = False
    bus_bandwidth: Optional[float] = None
    alg_bandwidth: Optional[float] = None
    test_results: Optional[List[Dict]] = None


@dataclass
class BinarySearchNode:
    """Represents a node in the binary search tree for performance isolation."""

    partition: ClusterPartition
    level: int
    parent: Optional['BinarySearchNode'] = None
    left_child: Optional['BinarySearchNode'] = None
    right_child: Optional['BinarySearchNode'] = None
    is_problematic: bool = False

    def add_children(self, left_partition: ClusterPartition, right_partition: ClusterPartition):
        """Add child nodes for binary search tree."""
        self.left_child = BinarySearchNode(left_partition, self.level + 1, parent=self)
        self.right_child = BinarySearchNode(right_partition, self.level + 1, parent=self)
        return self.left_child, self.right_child


class RCCLBinarySearchIsolation:
    """Main class for RCCL binary search performance isolation."""

    def __init__(self, cluster_dict: Dict, config_dict: Dict, phdl, shdl):
        self.cluster_dict = cluster_dict
        self.config_dict = config_dict
        self.phdl = phdl
        self.shdl = shdl

        # Configuration
        self.max_recursion_depth = 10
        self.min_partition_size = 1
        self.test_collective = "all_reduce_perf"  # Hardcoded as per user confirmation

        # Extract performance thresholds from existing config
        self.performance_thresholds = self._extract_performance_thresholds()

        # Log the extracted thresholds for transparency
        log.info(f"Binary search will use collective: {self.test_collective}")
        for size, threshold in self.performance_thresholds.items():
            size_gb = int(size) / (1024**3)  # Convert bytes to GB
            log.info(f"  Message size {size_gb:.1f}GB: Expected bus_bw >= {threshold['bus_bw']} GB/s")

        # Results tracking
        self.search_tree_root = None
        self.problematic_nodes = []
        self.good_partitions = []
        self.test_results = []
        self.fabric_bottlenecks = []
        self.parallel_tests_count = 0  # Track parallel testing usage

    def _extract_performance_thresholds(self) -> Dict:
        """Extract performance thresholds from existing RCCL config."""
        try:
            results_config = self.config_dict.get('results', {})
            collective_config = results_config.get(self.test_collective, {})
            bus_bw_thresholds = collective_config.get('bus_bw', {})

            if not bus_bw_thresholds:
                log.warning(f"No bus_bw thresholds found for {self.test_collective}, using defaults")
                return {
                    '8589934592': {'bus_bw': '300.00'},  # 8GB
                    '17179869184': {'bus_bw': '320.00'},  # 16GB
                }

            # Convert to the format expected by binary search
            thresholds = {}
            for size_bytes, threshold_gbps in bus_bw_thresholds.items():
                thresholds[size_bytes] = {'bus_bw': threshold_gbps}

            return thresholds

        except Exception as e:
            log.error(f"Error extracting performance thresholds: {e}")
            # Fallback to reasonable defaults
            return {
                '8589934592': {'bus_bw': '300.00'},  # 8GB
                '17179869184': {'bus_bw': '320.00'},  # 16GB
            }

    def _create_partition_cluster_dict(self, partition: ClusterPartition) -> Dict:
        """Create a cluster dictionary for a specific partition."""
        partition_cluster_dict = copy.deepcopy(self.cluster_dict)

        # Filter nodes to only include partition nodes
        filtered_nodes = {}
        for node in partition.nodes:
            if node in self.cluster_dict['node_dict']:
                filtered_nodes[node] = self.cluster_dict['node_dict'][node]
            else:
                log.warning(f"Node {node} not found in cluster_dict")

        partition_cluster_dict['node_dict'] = filtered_nodes

        # Set the first node as head node for this partition
        if partition.nodes:
            partition_cluster_dict['head_node_dict'] = {'mgmt_ip': partition.nodes[0]}

            return partition_cluster_dict

    def _create_optimized_test_config(self) -> Dict:
        """
        Create optimized RCCL config for binary search with faster execution.
        Only tests 8GB and 16GB message sizes with reduced iterations.
        """
        # Extract RCCL configuration from the nested config structure
        rccl_config = self.config_dict.get("rccl", {})
        if not rccl_config:
            raise ValueError("RCCL configuration not found in config file. Expected 'rccl' section.")

        optimized_config = copy.deepcopy(rccl_config)

        # Optimize for speed: only test the message sizes we care about
        optimized_config['start_msg_size'] = "8589934592"  # 8GB in bytes
        optimized_config['end_msg_size'] = "17179869184"  # 16GB in bytes
        optimized_config['step_function'] = "2"  # 8GB -> 16GB (only 2 sizes)

        # Reduce iterations for faster execution
        optimized_config['warmup_iterations'] = "3"  # Reduced from default
        optimized_config['no_of_iterations'] = "5"  # Reduced from default
        optimized_config['no_of_cycles'] = "1"  # Single cycle

        return optimized_config

    def _evaluate_partition_performance(self, partition: ClusterPartition) -> bool:
        """
        Test a partition's performance and determine if it meets thresholds.
        Returns True if partition meets performance requirements.
        """
        log.info(
            f"Testing partition {partition.partition_id} with {len(partition.nodes)} nodes: {', '.join(partition.nodes)}"
        )

        try:
            # Create partition-specific cluster and config
            partition_cluster_dict = self._create_partition_cluster_dict(partition)
            optimized_config = self._create_optimized_test_config()

            # Calculate ranks for this partition
            ranks_per_node = int(optimized_config.get('ranks_per_node', 8))
            total_ranks = len(partition.nodes) * ranks_per_node
            optimized_config['no_of_global_ranks'] = str(total_ranks)

            # SSH connectivity assumed to be pre-configured

            # Run RCCL test on this partition - only mandatory args + optimized overrides
            test_results = rccl_lib.rccl_cluster_test_default(
                # Mandatory positional parameters
                self.phdl,
                self.shdl,
                self.test_collective,
                partition.nodes,
                partition.nodes,
                partition_cluster_dict['username'],
                optimized_config['ib_hca_list'],
                optimized_config['net_dev_list'],
                optimized_config['oob_port'],
                optimized_config['no_of_global_ranks'],
                optimized_config.get('rocm_path_var', '/opt/rocm'),
                optimized_config.get('mpi_dir', '/mnt/scratch1/amd/ichristo/mpi'),
                optimized_config.get('mpi_path_var', '/mnt/scratch1/amd/ichristo/mpi'),
                optimized_config.get('rccl_dir', '/opt/rocm'),
                optimized_config.get('rccl_path_var', '/opt/rocm'),
                optimized_config.get('rccl_tests_dir', '/mnt/scratch1/amd/ichristo/rccl_tests'),
                # Only override specific parameters for fast binary search
                nccl_socket_ifname=optimized_config.get('nccl_socket_ifname', ''),
                gid_index=int(optimized_config.get('gid_index', 1)),
                start_msg_size=8589934592,  # 8GB
                end_msg_size=17179869184,   # 16GB  
                warmup_iterations=3,        # Fast warmup
                no_of_iterations=5,         # Fast iterations
                data_types=['float'],       # Only float
                rccl_result_file='/tmp/rccl_result_file_float.json',
                debug_level=optimized_config.get('debug_level', 'INFO'),  # From config
                mpi_pml=optimized_config.get('mpi_pml', 'auto'),
                nic_model=optimized_config.get('nic_model', 'ainic'),  # From config
                exp_results_dict=optimized_config.get('results'),     # Expected results from config
                env_source_script=optimized_config.get('env_source_script'),
                verify_bus_bw="False",      # Skip for speed (string, not boolean)
                verify_bw_dip="False",      # Skip for speed (string, not boolean)
                verify_lat_dip="False",     # Skip for speed (string, not boolean)
            )

            # CRITICAL: Validate that the test actually ran with the expected number of ranks
            expected_nodes = len(partition.nodes)
            expected_ranks_per_node = int(self.config_dict.get('ranks_per_node', 8))
            expected_total_ranks = expected_nodes * expected_ranks_per_node

            if test_results and len(test_results) > 0:
                actual_nodes = test_results[0].get('nodes', 0)
                actual_total_ranks = test_results[0].get('ranks', 0)

                if actual_total_ranks != expected_total_ranks:
                    log.error(f"🚨 RCCL TEST VALIDATION FAILED for partition {partition.partition_id}")
                    log.error(
                        f"   Expected total ranks: {expected_total_ranks} ({expected_nodes} nodes × {expected_ranks_per_node} ranks/node)"
                    )
                    log.error(f"   Actual total ranks: {actual_total_ranks}")
                    log.error("   This indicates the multi-node test failed")
                    log.error("   These results are INVALID for binary search analysis")
                    partition.test_output = (
                        f"RCCL test failed: expected {expected_total_ranks} ranks, got {actual_total_ranks} ranks"
                    )
                    partition.meets_threshold = False
                    return False

                if actual_nodes != expected_nodes:
                    log.warning(f"⚠️  RCCL library bug detected for partition {partition.partition_id}")
                    log.warning(
                        f"   Reports {actual_nodes} nodes instead of {expected_nodes}, but rank count ({actual_total_ranks}) is correct"
                    )
                    log.warning("   Continuing with performance validation...")

            # Store test results
            partition.test_results = test_results

            # Check performance against thresholds
            meets_all_thresholds = True
            bandwidth_results = {}

            for test_result in test_results:
                for size_bytes, threshold_dict in self.performance_thresholds.items():
                    expected_bus_bw = float(threshold_dict['bus_bw'])

                    # Find matching result for this message size
                    size_results = test_result.get('results', {})
                    if size_bytes in size_results:
                        actual_bus_bw = float(size_results[size_bytes]['bus_bw'])
                        bandwidth_results[size_bytes] = actual_bus_bw

                        # Apply 5% tolerance for threshold comparison
                        threshold_with_tolerance = expected_bus_bw * 0.95

                        if actual_bus_bw < threshold_with_tolerance:
                            meets_all_thresholds = False
                            log.warning(
                                f"Partition {partition.partition_id} failed threshold for {int(size_bytes) / (1024**3):.1f}GB: "
                                f"{actual_bus_bw:.2f} < {expected_bus_bw:.2f} GB/s"
                            )

            # Store the best bandwidth result (typically from largest message size)
            if bandwidth_results:
                partition.bus_bandwidth = max(bandwidth_results.values())

            partition.meets_threshold = meets_all_thresholds

            if meets_all_thresholds:
                log.info(
                    f"✅ Partition {partition.partition_id} PASSED performance test with {expected_total_ranks} ranks on {expected_nodes} nodes"
                )
                self.good_partitions.append(partition)
            else:
                log.warning(
                    f"❌ Partition {partition.partition_id} FAILED performance test with {expected_total_ranks} ranks on {expected_nodes} nodes"
                )

            return meets_all_thresholds

        except Exception as e:
            log.error(f"Error testing partition {partition.partition_id}: {e}")
            partition.test_output = f"Test failed with error: {e}"
            partition.meets_threshold = False
            return False

    def _split_partition(self, partition: ClusterPartition) -> Tuple[ClusterPartition, ClusterPartition]:
        """Split a partition into two roughly equal parts."""
        nodes = partition.nodes
        mid_point = len(nodes) // 2

        left_nodes = nodes[:mid_point]
        right_nodes = nodes[mid_point:]

        left_partition = ClusterPartition(
            partition_id=f"{partition.partition_id}_L", nodes=left_nodes, parent_id=partition.partition_id
        )

        right_partition = ClusterPartition(
            partition_id=f"{partition.partition_id}_R", nodes=right_nodes, parent_id=partition.partition_id
        )

        return left_partition, right_partition

    def _binary_search_recursive(self, partition: ClusterPartition, depth: int = 0) -> List[ClusterPartition]:
        """
        Recursively perform binary search to isolate problematic nodes.
        Returns list of problematic partitions found.
        """
        problematic_partitions = []

        # Base cases
        if depth >= self.max_recursion_depth:
            log.warning(
                f"Reached maximum recursion depth {self.max_recursion_depth} for partition {partition.partition_id}"
            )
            return problematic_partitions

        if len(partition.nodes) < self.min_partition_size:
            log.info(
                f"Partition {partition.partition_id} has {len(partition.nodes)} nodes, below minimum size {self.min_partition_size}"
            )
            return problematic_partitions

        # Test current partition
        log.info(
            f"Binary search at depth {depth}: Partition {partition.partition_id}: {len(partition.nodes)} nodes ({', '.join(partition.nodes[:3])}{'...' if len(partition.nodes) > 3 else ''})"
        )

        # Record this test for detailed results
        test_record = {
            'partition_id': partition.partition_id,
            'depth': depth,
            'nodes': partition.nodes.copy(),
            'meets_threshold': False,
            'bus_bandwidth': None,
        }

        meets_threshold = self._evaluate_partition_performance(partition)
        test_record['meets_threshold'] = meets_threshold
        test_record['bus_bandwidth'] = partition.bus_bandwidth
        self.test_results.append(test_record)

        if meets_threshold:
            log.info(
                f"✅ Partition {partition.partition_id} meets performance thresholds - no further subdivision needed"
            )
            return problematic_partitions

        # If partition fails and is at minimum size, all nodes are problematic
        if len(partition.nodes) <= self.min_partition_size:
            log.warning(f"❌ Partition {partition.partition_id} failed and cannot be subdivided further")
            self.problematic_nodes.extend(partition.nodes)
            problematic_partitions.append(partition)
            return problematic_partitions

        # Split and test recursively
        left_partition, right_partition = self._split_partition(partition)

        # Test partitions in parallel for efficiency
        parallel_start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(self._binary_search_recursive, left_partition, depth + 1)
            right_future = executor.submit(self._binary_search_recursive, right_partition, depth + 1)

            left_problematic = left_future.result()
            right_problematic = right_future.result()

        parallel_end_time = time.time()
        parallel_duration = parallel_end_time - parallel_start_time
        self.parallel_tests_count += 1
        log.info(f"Parallel testing completed in {parallel_duration:.1f} seconds")

        # Check for fabric bottlenecks (both partitions work individually but not together)
        left_meets_threshold = any(
            p.partition_id == left_partition.partition_id and p.meets_threshold for p in self.test_results[-10:]
        )
        right_meets_threshold = any(
            p.partition_id == right_partition.partition_id and p.meets_threshold for p in self.test_results[-10:]
        )

        if left_meets_threshold and right_meets_threshold and not meets_threshold:
            log.warning(f"🚨 FABRIC BOTTLENECK detected: {partition.partition_id}")
            log.warning(f"   Left partition ({left_partition.partition_id}) works: {left_meets_threshold}")
            log.warning(f"   Right partition ({right_partition.partition_id}) works: {right_meets_threshold}")
            log.warning("   Combined partition fails - indicates network fabric limitation")

            # Calculate bandwidth loss
            combined_bw = partition.bus_bandwidth or 0
            left_bw = next(
                (p.bus_bandwidth for p in self.test_results if p['partition_id'] == left_partition.partition_id), 0
            )
            right_bw = next(
                (p.bus_bandwidth for p in self.test_results if p['partition_id'] == right_partition.partition_id), 0
            )
            avg_individual_bw = (left_bw + right_bw) / 2 if left_bw and right_bw else 0

            if avg_individual_bw > 0:
                bandwidth_loss_pct = ((avg_individual_bw - combined_bw) / avg_individual_bw) * 100
                log.warning(f"   Bandwidth loss when combined: {bandwidth_loss_pct:.1f}%")

                self.fabric_bottlenecks.append(
                    {
                        'combined_partition': partition.partition_id,
                        'left_partition': left_partition.partition_id,
                        'right_partition': right_partition.partition_id,
                        'combined_bandwidth': combined_bw,
                        'left_bandwidth': left_bw,
                        'right_bandwidth': right_bw,
                        'bandwidth_loss_percent': bandwidth_loss_pct,
                    }
                )

        # Combine results
        problematic_partitions.extend(left_problematic)
        problematic_partitions.extend(right_problematic)

        return problematic_partitions

    def _run_final_validation_test(self, good_nodes: List[str]) -> Dict:
        """Run a final validation test with only the good nodes identified."""
        if not good_nodes:
            return {
                'reason': 'no_good_nodes',
                'good_nodes': [],
                'performance_passed': False,
                'message': 'No good nodes available for final validation',
            }

        log.info(f"Running final validation test with {len(good_nodes)} good nodes")

        # Create final test partition
        final_partition = ClusterPartition(partition_id="FINAL_VALIDATION", nodes=good_nodes)

        # Test the pruned cluster
        performance_passed = self._evaluate_partition_performance(final_partition)

        return {
            'reason': 'final_validation',
            'good_nodes': good_nodes,
            'excluded_nodes': self.problematic_nodes,
            'performance_passed': performance_passed,
            'meets_threshold': final_partition.meets_threshold,
            'bus_bandwidth': final_partition.bus_bandwidth,
            'message': f'Final validation with {len(good_nodes)} nodes: {"PASSED" if performance_passed else "FAILED"}',
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []

        # Fabric bottleneck recommendations
        if self.fabric_bottlenecks:
            recommendations.append(
                "🚨 NETWORK FABRIC BOTTLENECKS DETECTED - This is likely the primary performance issue"
            )
            recommendations.append(
                f"Found {len(self.fabric_bottlenecks)} fabric bottleneck(s) where smaller partitions work but larger ones fail"
            )
            recommendations.append("IMMEDIATE ACTIONS:")
            recommendations.append("  1. Check spine switch utilization and upgrade if oversubscribed")
            recommendations.append("  2. Verify inter-rack/inter-pod bandwidth capacity")
            recommendations.append("  3. Review RDMA routing configuration and subnet manager settings")
            recommendations.append(
                "  4. Consider running workloads on smaller, well-performing partitions until fabric is upgraded"
            )

            for bottleneck in self.fabric_bottlenecks:
                recommendations.append(
                    f"  - Partition {bottleneck['combined_partition']}: {bottleneck['bandwidth_loss_percent']:.1f}% bandwidth loss when combining groups"
                )

        # Node-specific recommendations
        if self.problematic_nodes:
            recommendations.append(
                f"Remove the following {len(self.problematic_nodes)} problematic nodes from your cluster: {self.problematic_nodes}"
            )
            recommendations.append(
                f"Consider running workloads on the {len([p for p in self.good_partitions])} nodes in well-performing partitions"
            )
        else:
            recommendations.append("No problematic individual nodes identified - issues appear to be fabric-related")

        return recommendations

    def run_binary_search(self) -> Dict:
        """
        Main entry point for binary search performance isolation.
        Returns comprehensive analysis results.
        """
        start_time = time.time()
        log.info("Starting RCCL Binary Search Performance Isolation")

        # Get all cluster nodes
        all_nodes = list(self.cluster_dict['node_dict'].keys())
        log.info(f"Cluster size: {len(all_nodes)} nodes")
        log.info(f"Test collective: {self.test_collective}")
        log.info(f"Performance thresholds: {self.performance_thresholds}")

        # Create root partition
        root_partition = ClusterPartition(partition_id="ROOT", nodes=all_nodes)

        # Start binary search
        self._binary_search_recursive(root_partition)

        end_time = time.time()

        # Run final validation test with good nodes only
        good_nodes = [node for node in all_nodes if node not in self.problematic_nodes]
        final_test_result = None

        if good_nodes:
            # Add delay before final validation to allow proper cleanup of resources
            log.info("Waiting 10 seconds for resource cleanup before final validation test...")
            time.sleep(10)
            final_test_result = self._run_final_validation_test(good_nodes)
        else:
            final_test_result = {
                'reason': 'no_good_nodes',
                'good_nodes': [],
                'performance_passed': False,
                'message': 'Entire cluster performs well - no pruning needed',
            }

        # Generate comprehensive report
        report = {
            'summary': {
                'total_nodes': len(all_nodes),
                'problematic_nodes': list(set(self.problematic_nodes)),  # Remove duplicates
                'good_partitions': [
                    {'id': p.partition_id, 'nodes': p.nodes, 'bandwidth': p.bus_bandwidth} for p in self.good_partitions
                ],
                'fabric_bottlenecks': self.fabric_bottlenecks,
                'total_tests_run': len(self.test_results),
                'execution_time_seconds': round(end_time - start_time, 2),
                'max_depth_reached': max([r['depth'] for r in self.test_results]) if self.test_results else 0,
            },
            'detailed_results': self.test_results,
            'final_validation': final_test_result,
            'recommendations': self._generate_recommendations(),
        }

        log.info("Binary Search Isolation Complete")
        log.info(f"Problematic nodes identified: {report['summary']['problematic_nodes']}")
        log.info(f"Network fabric bottlenecks detected: {len(report['summary']['fabric_bottlenecks'])}")
        log.info(f"Total tests run: {report['summary']['total_tests_run']}")
        log.info(f"Parallel test levels executed: {self.parallel_tests_count}")
        log.info(f"Execution time: {report['summary']['execution_time_seconds']} seconds")

        return report


# Pytest fixtures for cluster and config setup
@pytest.fixture(scope="session")
def cluster_dict(pytestconfig):
    """Load cluster configuration from CLI argument."""
    cluster_file = pytestconfig.getoption("cluster_file")
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="session")
def config_dict(pytestconfig):
    """Load RCCL configuration from CLI argument."""
    config_file = pytestconfig.getoption("config_file")
    with open(config_file) as json_file:
        config_dict = json.load(json_file)
    return config_dict


@pytest.fixture(scope="session")
def phdl(cluster_dict):
    """Create parallel SSH handle for cluster operations."""
    all_nodes = list(cluster_dict['node_dict'].keys())
    return Pssh(log, all_nodes, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])


@pytest.fixture(scope="session")
def shdl(cluster_dict):
    """Create single SSH handle for head node operations."""
    head_node = cluster_dict['head_node_dict']['mgmt_ip']
    return Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])


def _generate_binary_search_html_report(results: Dict) -> str:
    """Generate HTML content for the binary search isolation report."""

    summary = results['summary']
    detailed_results = results['detailed_results']
    recommendations = results['recommendations']

    html_content = f"""
    <div class="binary-search-report">
        <h2>Binary Search Performance Isolation Results</h2>
        
        <div class="summary-section">
            <h3>Summary</h3>
            <table class="summary-table">
                <tr><td><strong>Total Nodes Analyzed:</strong></td><td>{summary['total_nodes']}</td></tr>
                <tr><td><strong>Problematic Nodes Found:</strong></td><td>{len(summary['problematic_nodes'])}</td></tr>
                <tr><td><strong>Network Fabric Bottlenecks:</strong></td><td>{len(summary.get('fabric_bottlenecks', []))}</td></tr>
                <tr><td><strong>Total Tests Executed:</strong></td><td>{summary['total_tests_run']}</td></tr>
                <tr><td><strong>Execution Time:</strong></td><td>{summary['execution_time_seconds']} seconds</td></tr>
                <tr><td><strong>Maximum Search Depth:</strong></td><td>{summary['max_depth_reached']}</td></tr>
            </table>
        </div>
        
        <div class="fabric-bottlenecks-section">
            <h3>Network Fabric Bottlenecks</h3>
            {_generate_fabric_bottlenecks_html(summary.get('fabric_bottlenecks', []))}
        </div>
        
        <div class="problematic-nodes-section">
            <h3>Problematic Nodes</h3>
            {_generate_problematic_nodes_html(summary['problematic_nodes'])}
        </div>
        
        <div class="good-partitions-section">
            <h3>Well-Performing Partitions</h3>
            {_generate_good_partitions_html(summary['good_partitions'])}
        </div>
        
        <div class="final-validation-section">
            <h3>Final Validation Test (Pruned Cluster)</h3>
            {_generate_final_validation_html(results.get('final_validation', {}))}
        </div>
        
        <div class="detailed-results-section">
            <h3>Detailed Test Results</h3>
            {_generate_detailed_results_table(detailed_results)}
        </div>
        
        <div class="recommendations-section">
            <h3>Recommendations</h3>
            {_generate_recommendations_html(recommendations)}
        </div>
    </div>
    
    <style>
        .binary-search-report {{ font-family: Arial, sans-serif; }}
        .summary-table {{ border-collapse: collapse; width: 100%; }}
        .summary-table td {{ border: 1px solid #ddd; padding: 8px; }}
        .summary-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .problematic-node {{ color: #d32f2f; font-weight: bold; }}
        .good-partition {{ color: #388e3c; }}
        .performance-pass {{ background-color: #c8e6c9; }}
        .performance-fail {{ background-color: #ffcdd2; }}
        .recommendations-section ul {{ list-style-type: disc; margin-left: 20px; }}
    </style>
    """

    return html_content


def _generate_fabric_bottlenecks_html(fabric_bottlenecks: List[Dict]) -> str:
    """Generate HTML for fabric bottlenecks section."""
    if not fabric_bottlenecks:
        return "<p>No network fabric bottlenecks detected.</p>"

    html = f"<div class='fabric-bottlenecks'><p>🚨 <strong>{len(fabric_bottlenecks)} Network Fabric Bottleneck(s) Detected</strong></p>"
    html += "<p>These indicate network infrastructure issues rather than individual node problems:</p>"

    for i, bottleneck in enumerate(fabric_bottlenecks, 1):
        combined_bw = bottleneck.get('combined_bandwidth', 0)
        left_bw = bottleneck.get('left_bandwidth', 0)
        right_bw = bottleneck.get('right_bandwidth', 0)
        loss_pct = bottleneck.get('bandwidth_loss_percent', 0)

        html += "<div class='bottleneck-item' style='border: 2px solid #ff9800; padding: 10px; margin: 10px 0;'>"
        html += f"<h4>Bottleneck #{i}: {bottleneck['combined_partition']}</h4>"
        html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += f"<tr><td><strong>Combined Partition:</strong></td><td>{bottleneck['combined_partition']} ({len(bottleneck.get('nodes', []))} nodes) - {combined_bw:.2f} GB/s ❌</td></tr>"
        html += f"<tr><td><strong>Left Partition:</strong></td><td>{bottleneck['left_partition']} ({len(bottleneck.get('left_nodes', []))} nodes) - {left_bw:.2f} GB/s ✅</td></tr>"
        html += f"<tr><td><strong>Right Partition:</strong></td><td>{bottleneck['right_partition']} ({len(bottleneck.get('right_nodes', []))} nodes) - {right_bw:.2f} GB/s ✅</td></tr>"
        html += f"<tr><td><strong>Bandwidth Loss:</strong></td><td>{loss_pct:.1f}% when groups are combined</td></tr>"
        html += "</table></div>"

    html += "<p><strong>Recommended Actions:</strong></p>"
    html += "<ul>"
    html += "<li>Check spine switch utilization and capacity</li>"
    html += "<li>Verify inter-rack/inter-pod bandwidth limits</li>"
    html += "<li>Review RDMA subnet manager configuration</li>"
    html += "<li>Consider network topology optimization</li>"
    html += "</ul>"
    html += "</div>"

    return html


def _generate_problematic_nodes_html(problematic_nodes: List[str]) -> str:
    """Generate HTML for problematic nodes section."""
    if not problematic_nodes:
        return "<p>No problematic nodes identified. Cluster performance is acceptable.</p>"

    html = "<div class='problematic-nodes'>"
    html += f"<p>The following {len(problematic_nodes)} nodes are causing performance issues:</p>"
    html += "<ul>"
    for node in problematic_nodes:
        html += f"<li class='problematic-node'>{node}</li>"
    html += "</ul>"
    html += "</div>"

    return html


def _generate_good_partitions_html(good_partitions: List[Dict]) -> str:
    """Generate HTML for good partitions section."""
    if not good_partitions:
        return "<p>No well-performing partitions identified.</p>"

    html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    html += "<tr><th>Partition ID</th><th>Node Count</th><th>Nodes</th><th>Bus Bandwidth (GB/s)</th></tr>"

    for partition in good_partitions:
        nodes_str = ', '.join(partition['nodes'])  # Show ALL nodes

        bandwidth = partition.get('bandwidth', 'N/A')
        bandwidth_str = f"{bandwidth:.2f}" if isinstance(bandwidth, (int, float)) else str(bandwidth)

        html += f"""
        <tr class='good-partition'>
            <td>{partition['id']}</td>
            <td>{len(partition['nodes'])}</td>
            <td>{nodes_str}</td>
            <td>{bandwidth_str}</td>
        </tr>
        """

    html += "</table>"
    return html


def _generate_detailed_results_table(detailed_results: List[Dict]) -> str:
    """Generate HTML table for detailed binary search results with full node lists."""
    if not detailed_results:
        return "<p>No detailed test results available.</p>"

    html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    html += "<tr><th>Partition ID</th><th>Depth</th><th>Node Count</th><th>Performance</th><th>Bus BW (GB/s)</th><th>Nodes</th></tr>"

    for result in detailed_results:
        partition_id = result.get('partition_id', 'Unknown')
        depth = result.get('depth', 0)
        nodes = result.get('nodes', [])
        node_count = len(nodes)

        # Performance status
        meets_threshold = result.get('meets_threshold', False)
        performance_status = "PASS" if meets_threshold else "FAIL"
        performance_class = "performance-pass" if meets_threshold else "performance-fail"

        # Bandwidth
        bandwidth = result.get('bus_bandwidth', 'N/A')
        bandwidth_str = f"{bandwidth:.2f}" if isinstance(bandwidth, (int, float)) else str(bandwidth)

        # Show ALL nodes without truncation
        nodes_str = ', '.join(nodes) if nodes else 'No nodes'

        html += f"""
        <tr class='{performance_class}'>
            <td>{partition_id}</td>
            <td>{depth}</td>
            <td>{node_count}</td>
            <td>{performance_status}</td>
            <td>{bandwidth_str}</td>
            <td>{nodes_str}</td>
        </tr>
        """

    html += "</table>"
    return html


def _generate_recommendations_html(recommendations: List[str]) -> str:
    """Generate HTML for recommendations section."""
    if not recommendations:
        return "<p>No specific recommendations at this time.</p>"

    html = "<ul>"
    for recommendation in recommendations:
        html += f"<li>{recommendation}</li>"
    html += "</ul>"

    return html


def _generate_final_validation_html(final_validation: Dict) -> str:
    """Generate HTML for final validation test results with full node lists."""
    if not final_validation:
        return "<p>No final validation test performed.</p>"

    # Generate results table
    status_class = "performance-pass" if final_validation.get('performance_passed') else "performance-fail"
    status_text = "PASSED ✅" if final_validation.get('performance_passed') else "FAILED ❌"

    bandwidth = final_validation.get('bus_bandwidth', 'N/A')
    bandwidth_str = f"{bandwidth:.2f}" if isinstance(bandwidth, (int, float)) else str(bandwidth)

    good_nodes = final_validation.get('good_nodes', [])
    excluded_nodes = final_validation.get('excluded_nodes', [])

    html = f"""
    <div class="final-validation-results">
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr><td><strong>Test Status:</strong></td><td class="{status_class}">{status_text}</td></tr>
            <tr><td><strong>Nodes Tested:</strong></td><td>{len(good_nodes)} good nodes</td></tr>
            <tr><td><strong>Nodes Excluded:</strong></td><td>{len(excluded_nodes)} problematic nodes</td></tr>
            <tr><td><strong>Bus Bandwidth:</strong></td><td>{bandwidth_str} GB/s</td></tr>
            <tr><td><strong>Meets Threshold:</strong></td><td>{"Yes" if final_validation.get('meets_threshold') else "No"}</td></tr>
        </table>
        
        <h4>Good Nodes Used in Final Test:</h4>
        <p>{', '.join(good_nodes) if good_nodes else 'None'}</p>
        
        <h4>Excluded Problematic Nodes:</h4>
        <p>{', '.join(excluded_nodes) if excluded_nodes else 'None'}</p>
    </div>
    """

    return html


# Global variable to store binary search results for HTML report generation
binary_search_results = None


def test_binary_search_isolation(phdl, shdl, cluster_dict, config_dict):
    """
    Main test function that performs binary search isolation of problematic nodes.

    This test automatically identifies nodes causing poor RCCL performance by:
    1. Testing the full cluster performance
    2. If performance is poor, recursively splitting the cluster in half
    3. Testing each partition until problematic nodes are isolated
    4. Generating a comprehensive report with recommendations
    """
    global binary_search_results

    log.info("=== RCCL Binary Search Performance Isolation Test ===")

    try:
        # Initialize the binary search isolation system
        isolation_system = RCCLBinarySearchIsolation(cluster_dict, config_dict, phdl, shdl)

        # Run the binary search analysis
        results = isolation_system.run_binary_search()

        # Store results globally for HTML report generation
        binary_search_results = results

        # Log summary results
        summary = results['summary']
        log.info(
            f"Analysis complete - tested {summary['total_tests_run']} partitions in {summary['execution_time_seconds']} seconds"
        )

        if summary['problematic_nodes']:
            log.warning(
                f"Identified {len(summary['problematic_nodes'])} problematic nodes: {summary['problematic_nodes']}"
            )
            for recommendation in results['recommendations']:
                log.info(f"RECOMMENDATION: {recommendation}")
        else:
            log.info("No problematic nodes identified - cluster performance is acceptable")

        # The test always passes - this is an analysis tool, not a pass/fail test
        log.info("Binary search isolation analysis completed successfully")

    except Exception as e:
        log.error(f"Binary search analysis failed with exception: {e}")
        log.error(f"Exception type: {type(e).__name__}")
        import traceback

        log.error(f"Traceback: {traceback.format_exc()}")

        # Store empty results so report generation doesn't fail
        binary_search_results = {
            'summary': {'problematic_nodes': [], 'total_tests_run': 0, 'execution_time_seconds': 0},
            'recommendations': [f"Binary search failed due to: {e}"],
            'good_partitions': [],
            'fabric_bottlenecks': [],
            'final_validation': None,
            'detailed_results': [],  # Add missing detailed_results key
        }

        # Re-raise the exception so pytest shows it as failed with details
        raise


def test_generate_binary_search_report(config_dict, request):
    """
    Generate HTML report for binary search isolation results.
    This follows the same pattern as other RCCL test suites.
    """
    global binary_search_results

    if not binary_search_results:
        log.warning("No binary search results available for report generation")
        return

    log.info("Generating binary search isolation HTML report")

    # Create temporary HTML file
    proc_id = os.getpid()
    html_file = f'/tmp/rccl_binary_search_report_{proc_id}.html'

    # Generate HTML content
    html_content = _generate_binary_search_html_report(binary_search_results)

    # Write HTML file
    with open(html_file, 'w') as f:
        f.write(html_content)

    # HTML report file generated successfully - pytest integration not needed
    # The standalone HTML file provides complete reporting functionality

    log.info(f"Binary search HTML report generated: {html_file}")
