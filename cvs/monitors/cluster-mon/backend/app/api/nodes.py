"""
Node-level API endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from app.api.cluster import check_node_health

router = APIRouter()


@router.get("")
async def list_nodes() -> List[Dict[str, Any]]:
    """
    List all cluster nodes with basic status.

    Returns:
        [
            {
                "hostname": "node1",
                "status": "healthy|unhealthy|unreachable",
                "gpu_count": 8,
                "avg_gpu_util": 75.5,
                "avg_gpu_temp": 68.2,
                "health_issues": []
            },
            ...
        ]
    """
    from app.main import app_state

    if not app_state.ssh_manager:
        return []

    nodes_list = []
    gpu_data = app_state.latest_metrics.get("gpu", {})
    util_data = gpu_data.get("utilization", {})
    temp_data = gpu_data.get("temperature", {})

    # Process all configured nodes (use stable status from app_state)
    all_nodes = app_state.ssh_manager.host_list

    for node in all_nodes:
        # Get stable status (requires 5 consecutive failures to change)
        stable_status = app_state.node_health_status.get(node, 'healthy')

        # Check for hardware health issues
        has_metrics_error = node in util_data and isinstance(util_data[node], dict) and 'error' in util_data[node]
        health_status, health_issues = check_node_health(node, gpu_data, has_metrics_error)

        # Use stable status if it's unreachable
        if stable_status == 'unreachable':
            health_status = 'unreachable'
            health_issues = [f'Unreachable after {app_state.node_failure_count.get(node, 0)} consecutive failures']

        node_info = {
            "hostname": node,
            "status": health_status,
            "gpu_count": 0,
            "avg_gpu_util": 0.0,
            "avg_gpu_temp": 0.0,
            "health_issues": health_issues,
        }

        # Calculate GPU stats - handle both list and dict formats
        if node in util_data:
            node_util = util_data[node]

            # Handle list format (new amd-smi format)
            if isinstance(node_util, list):
                node_info["gpu_count"] = len(node_util)
                total_util = 0
                for gpu_entry in node_util:
                    if isinstance(gpu_entry, dict):
                        # Try different field names for utilization
                        usage = gpu_entry.get("usage", {})
                        if isinstance(usage, dict):
                            gfx_activity = usage.get("gfx_activity", {})
                            if isinstance(gfx_activity, dict):
                                util = gfx_activity.get("value", 0)
                            else:
                                util = 0
                        else:
                            util = gpu_entry.get("GPU use (%)", 0)
                        total_util += float(util) if util else 0

                node_info["avg_gpu_util"] = (
                    round(total_util / node_info["gpu_count"], 2) if node_info["gpu_count"] > 0 else 0
                )

            # Handle dict format (old rocm-smi format)
            elif isinstance(node_util, dict) and 'error' not in node_util:
                node_info["gpu_count"] = len(node_util)
                total_util = 0
                for gpu_metrics in node_util.values():
                    if isinstance(gpu_metrics, dict):
                        util = gpu_metrics.get("GPU use (%)", 0)
                        total_util += float(util) if util else 0

                node_info["avg_gpu_util"] = (
                    round(total_util / node_info["gpu_count"], 2) if node_info["gpu_count"] > 0 else 0
                )

        if node in temp_data:
            node_temp = temp_data[node]
            total_temp = 0
            temp_count = 0

            # Handle list format (new amd-smi format)
            if isinstance(node_temp, list):
                for gpu_entry in node_temp:
                    if isinstance(gpu_entry, dict):
                        # Try to get temperature from new format
                        temp_data_field = gpu_entry.get("temperature", {})
                        if isinstance(temp_data_field, dict):
                            # Try different sensor names
                            temp = temp_data_field.get(
                                "hotspot", temp_data_field.get("edge", temp_data_field.get("junction", 0))
                            )
                            if isinstance(temp, dict):
                                temp = temp.get("value", 0)
                        else:
                            temp = 0
                        if temp:
                            try:
                                total_temp += float(temp)
                                temp_count += 1
                            except (ValueError, TypeError):
                                pass

            # Handle dict format (old rocm-smi format)
            elif isinstance(node_temp, dict) and 'error' not in node_temp:
                for gpu_temp in node_temp.values():
                    if isinstance(gpu_temp, dict):
                        # Try junction (hotspot) first, then edge, then memory
                        temp = gpu_temp.get(
                            "Temperature (Sensor junction) (C)",
                            gpu_temp.get(
                                "Temperature (Sensor edge) (C)", gpu_temp.get("Temperature (Sensor memory) (C)", 0)
                            ),
                        )
                        if temp:
                            try:
                                total_temp += float(temp)
                                temp_count += 1
                            except (ValueError, TypeError):
                                pass

            node_info["avg_gpu_temp"] = round(total_temp / temp_count, 2) if temp_count > 0 else 0

        nodes_list.append(node_info)

    return nodes_list


@router.get("/{node_id}")
async def get_node_details(node_id: str) -> Dict[str, Any]:
    """
    Get detailed information for a specific node.

    Args:
        node_id: Node hostname or IP

    Returns:
        {
            "hostname": "node1",
            "status": "reachable",
            "gpus": [...],
            "nics": [...],
            "last_update": "2025-02-11T12:00:00Z"
        }
    """
    from app.main import app_state

    if not app_state.ssh_manager:
        raise HTTPException(status_code=503, detail="SSH manager not initialized")

    # Check if node exists
    all_nodes = app_state.ssh_manager.host_list
    if node_id not in all_nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    # Check if node is reachable
    is_reachable = node_id in app_state.ssh_manager.reachable_hosts

    node_details = {
        "hostname": node_id,
        "status": "reachable" if is_reachable else "unreachable",
        "last_update": app_state.latest_metrics.get("timestamp"),
        "gpus": [],
        "nics": [],
    }

    if not is_reachable:
        return node_details

    # Extract GPU data
    gpu_data = app_state.latest_metrics.get("gpu", {})
    util_data = gpu_data.get("utilization", {}).get(node_id, {})
    mem_data = gpu_data.get("memory", {}).get(node_id, {})
    temp_data = gpu_data.get("temperature", {}).get(node_id, {})
    power_data = gpu_data.get("power", {}).get(node_id, {})

    # Build GPU list
    if isinstance(util_data, dict):
        for gpu_id in util_data.keys():
            gpu_info = {
                "id": gpu_id,
                "utilization": 0.0,
                "memory_used_mb": 0,
                "memory_total_mb": 0,
                "memory_util_percent": 0.0,
                "temperature_c": 0.0,
                "power_w": 0.0,
            }

            # Utilization
            if gpu_id in util_data and isinstance(util_data[gpu_id], dict):
                util = util_data[gpu_id].get("GPU use (%)", 0)
                gpu_info["utilization"] = float(util) if util else 0.0

            # Memory
            if gpu_id in mem_data and isinstance(mem_data[gpu_id], dict):
                mem_used = mem_data[gpu_id].get("VRAM Total Used Memory (B)", 0)
                mem_total = mem_data[gpu_id].get("VRAM Total Memory (B)", 0)
                gpu_info["memory_used_mb"] = int(mem_used) // (1024 * 1024) if mem_used else 0
                gpu_info["memory_total_mb"] = int(mem_total) // (1024 * 1024) if mem_total else 0

                if gpu_info["memory_total_mb"] > 0:
                    gpu_info["memory_util_percent"] = round(
                        (gpu_info["memory_used_mb"] / gpu_info["memory_total_mb"]) * 100, 2
                    )

            # Temperature - try junction (hotspot) first, then edge, then memory
            if gpu_id in temp_data and isinstance(temp_data[gpu_id], dict):
                temp = temp_data[gpu_id].get(
                    "Temperature (Sensor junction) (C)",
                    temp_data[gpu_id].get(
                        "Temperature (Sensor edge) (C)", temp_data[gpu_id].get("Temperature (Sensor memory) (C)", 0)
                    ),
                )
                try:
                    gpu_info["temperature_c"] = float(temp) if temp else 0.0
                except (ValueError, TypeError):
                    gpu_info["temperature_c"] = 0.0

            # Power (from amd-smi format)
            if isinstance(power_data, dict) and "gpu_data" in power_data:
                for gpu_entry in power_data["gpu_data"]:
                    if str(gpu_entry.get("gpu")) == gpu_id.replace("card", ""):
                        power_str = gpu_entry.get("power", {}).get("socket_power", "0 W")
                        try:
                            gpu_info["power_w"] = float(power_str.split()[0])
                        except:
                            pass

            node_details["gpus"].append(gpu_info)

    # Extract NIC data
    nic_data = app_state.latest_metrics.get("nic", {})
    ip_data = nic_data.get("ip_addr", {}).get(node_id, {})
    rdma_data = nic_data.get("rdma_links", {}).get(node_id, {})

    if isinstance(ip_data, dict):
        for nic_name, nic_info in ip_data.items():
            if isinstance(nic_info, dict) and "error" not in nic_info:
                nic_details = {
                    "name": nic_name,
                    "state": nic_info.get("state", "UNKNOWN"),
                    "mtu": nic_info.get("mtu", ""),
                    "mac_addr": nic_info.get("mac_addr", ""),
                    "ipv4_addrs": nic_info.get("ipv4_addr_list", []),
                    "rdma": None,
                }

                # Add RDMA info if available
                if isinstance(rdma_data, dict):
                    for rdma_dev, rdma_info in rdma_data.items():
                        if rdma_info.get("netdev") == nic_name:
                            nic_details["rdma"] = {
                                "device": rdma_dev,
                                "state": rdma_info.get("state"),
                                "physical_state": rdma_info.get("physical_state"),
                            }

                node_details["nics"].append(nic_details)

    return node_details
