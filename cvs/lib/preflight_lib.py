'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


# check_gid_consistency moved to cvs.lib.preflight.gid_consistency


# Function body removed - now in cvs.lib.preflight.gid_consistency


# check_rocm_versions moved to cvs.lib.preflight.version_check


# check_interface_names moved to cvs.lib.preflight.interface_consistency


# generate_node_pairs moved to cvs.lib.preflight.rdma_connectivity


# Removed generate_full_mesh_batches - replaced with parallel group-based algorithm


# partition_nodes_into_groups moved to cvs.lib.preflight.base


# calculate_resource_requirements moved to cvs.lib.preflight.base


# find_host_group moved to cvs.lib.preflight.base


# check_rdma_connectivity moved to cvs.lib.preflight.rdma_connectivity


# Import functions from the new modular structure - moved to top of file


# execute_round_with_script_coordination moved to cvs.lib.preflight.rdma_connectivity


# _calculate_test_port_assignments moved to cvs.lib.preflight.rdma_connectivity


# generate_server_commands_for_round moved to cvs.lib.preflight.rdma_connectivity


# generate_client_commands_for_round moved to cvs.lib.preflight.rdma_connectivity


# create_ibv_result_collection_script moved to cvs.lib.preflight.rdma_connectivity


# collect_test_results_with_scriptlet moved to cvs.lib.preflight.rdma_connectivity


def collect_test_results(phdl, test_groups, round_type, port_start, expected_interfaces, gid_index):
    """
    Collect and analyze test results using ScriptLet for optimal performance.

    Eliminates exponential SSH calls by using ScriptLet's minimal output approach:
    - One script per node analyzes all logs locally
    - Only failed tests reported in key=value format
    - Dramatic reduction in network traffic and execution time
    """
    # Build test metadata from groups
    test_metadata = []
    port_offset = 0

    if round_type == "intra_group":
        for group_id, group_nodes in test_groups.items():
            for server_node in group_nodes:
                for client_node in group_nodes:
                    if server_node != client_node:
                        for server_iface in expected_interfaces:
                            for client_iface in expected_interfaces:
                                port = port_start + port_offset
                                combo_name = (
                                    f"{server_iface}→{client_iface}" if server_iface != client_iface else server_iface
                                )
                                pair_key = f"{server_node} ↔ {client_node} ({combo_name})"

                                test_metadata.append(
                                    {
                                        'pair_key': pair_key,
                                        'server_node': server_node,
                                        'client_node': client_node,
                                        'server_iface': server_iface,
                                        'client_iface': client_iface,
                                        'combo_name': combo_name,
                                        'port': port,
                                        'client_log_path': f"{temp_dir}/client_{client_iface}_{port}.log",
                                        'server_log_path': f"{temp_dir}/server_{server_iface}_{port}.log",
                                    }
                                )
                                port_offset += 1

    elif round_type == "inter_group":
        for test_name, test_config in test_groups.items():
            group1_nodes = test_config['group1']
            group2_nodes = test_config['group2']

            for server_node in group1_nodes:
                for client_node in group2_nodes:
                    for server_iface in expected_interfaces:
                        for client_iface in expected_interfaces:
                            port = port_start + port_offset
                            combo_name = (
                                f"{server_iface}→{client_iface}" if server_iface != client_iface else server_iface
                            )
                            pair_key = f"{server_node} ↔ {client_node} ({combo_name})"

                            test_metadata.append(
                                {
                                    'pair_key': pair_key,
                                    'server_node': server_node,
                                    'client_node': client_node,
                                    'server_iface': server_iface,
                                    'client_iface': client_iface,
                                    'combo_name': combo_name,
                                    'port': port,
                                    'client_log_path': f"{temp_dir}/client_{client_iface}_{port}.log",
                                    'server_log_path': f"{temp_dir}/server_{server_iface}_{port}.log",
                                }
                            )
                            port_offset += 1

    # Use ScriptLet for efficient result collection
    return collect_test_results_with_scriptlet(phdl, test_metadata, expected_interfaces, gid_index)


# _run_ibv_rc_pingpong_batch moved to cvs.lib.preflight.rdma_connectivity


# _analyze_ibv_rc_pingpong_output moved to cvs.lib.preflight.rdma_connectivity


# _extract_ibv_rc_pingpong_errors moved to cvs.lib.preflight.rdma_connectivity


# generate_preflight_summary moved to cvs.lib.preflight.report


# _summarize_gid_results moved to cvs.lib.preflight.report


# _summarize_connectivity_results moved to cvs.lib.preflight.report


# _summarize_rocm_results moved to cvs.lib.preflight.report


# _summarize_interface_results moved to cvs.lib.preflight.report


# _summarize_reachability_results moved to cvs.lib.preflight.report


# _summarize_ssh_connectivity_results moved to cvs.lib.preflight.report


# generate_html_report moved to cvs.lib.preflight.report


# _generate_html_content moved to cvs.lib.preflight.report


# _get_html_styles moved to cvs.lib.preflight.report
# _generate_executive_summary_html moved to cvs.lib.preflight.report
# _generate_gid_consistency_html moved to cvs.lib.preflight.report
# (removed) directed RDMA root-cause / hotspot aggregation was in report; HTML now lists detailed rows only
# _generate_connectivity_html moved to cvs.lib.preflight.report
# _generate_ssh_connectivity_html moved to cvs.lib.preflight.report
# _generate_rocm_versions_html moved to cvs.lib.preflight.report
# _generate_interface_names_html moved to cvs.lib.preflight.report
# _generate_configuration_html moved to cvs.lib.preflight.report
# _generate_recommendations_html moved to cvs.lib.preflight.report
