"""
Cluster-level API endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()


def check_node_health(node: str, gpu_data: dict, metrics_error: bool = False) -> tuple[str, list]:
    """
    Check if a node is healthy based on error metrics.
    Uses stable status from app_state (requires 5 consecutive failures).

    Returns:
        Tuple of (status, issues_list)
        Status can be: "healthy", "unhealthy", or "unreachable"
    """
    from app.main import app_state

    issues = []

    # If metrics collection had an error, this indicates connectivity issue
    if metrics_error:
        # Don't immediately mark as unreachable, use stable status
        stable_status = app_state.node_health_status.get(node, 'healthy')
        if stable_status == 'unreachable':
            return 'unreachable', ['Node unreachable via SSH']
        # Otherwise check for hardware errors below
        status = stable_status
    else:
        status = "healthy"

    # Check RAS errors
    ras_data = gpu_data.get("ras_errors", {}).get(node, {})
    if isinstance(ras_data, dict) and "gpu_data" in ras_data:
        for gpu in ras_data["gpu_data"]:
            ecc = gpu.get("ecc", {})
            uncorrectable = ecc.get("total_uncorrectable", 0)
            correctable = ecc.get("total_correctable", 0)

            if uncorrectable > 0:
                status = "unhealthy"
                issues.append(f"GPU {gpu['gpu']}: {uncorrectable} uncorrectable ECC errors")
            elif correctable > 10:  # Threshold for correctable errors
                status = "unhealthy"
                issues.append(f"GPU {gpu['gpu']}: {correctable} correctable ECC errors")

    # Check PCIe errors
    pcie_data = gpu_data.get("pcie", {}).get(node, {})
    if isinstance(pcie_data, dict) and "gpu_data" in pcie_data:
        for gpu in pcie_data["gpu_data"]:
            pcie = gpu.get("pcie", {})
            replay_count = pcie.get("replay_count", 0)
            nak_count = pcie.get("nak_count", 0)

            # Convert to int if string
            try:
                replay_count = int(replay_count) if replay_count else 0
                nak_count = int(nak_count) if nak_count else 0
            except (ValueError, TypeError):
                replay_count = 0
                nak_count = 0

            if replay_count > 100 or nak_count > 100:  # Error thresholds
                status = "unhealthy"
                issues.append(f"GPU {gpu['gpu']}: PCIe errors (replay: {replay_count}, nak: {nak_count})")

    # Check XGMI errors
    xgmi_data = gpu_data.get("xgmi", {}).get(node, {})
    if isinstance(xgmi_data, dict) and "gpu_data" in xgmi_data:
        for gpu in xgmi_data["gpu_data"]:
            xgmi = gpu.get("xgmi", {})
            error_count = xgmi.get("error_count", 0)

            if error_count > 10:  # XGMI error threshold
                status = "unhealthy"
                issues.append(f"GPU {gpu['gpu']}: {error_count} XGMI errors")

    # Check temperature (warning, not critical)
    temp_data = gpu_data.get("temperature", {}).get(node, {})
    if isinstance(temp_data, dict):
        for gpu_id, gpu_temp in temp_data.items():
            if isinstance(gpu_temp, dict):
                temp = gpu_temp.get("Temperature (Sensor edge) (c)")
                if temp and float(temp) > 85:
                    if status == "healthy":
                        status = "unhealthy"
                    issues.append(f"{gpu_id}: High temperature {temp}°C")

    return status, issues


@router.get("/status")
async def get_cluster_status() -> Dict[str, Any]:
    """
    Get overall cluster status.

    Returns:
        {
            "total_nodes": 10,
            "healthy_nodes": 8,
            "unhealthy_nodes": 1,
            "unreachable_nodes": 1,
            "total_gpus": 80,
            "avg_gpu_utilization": 65.5,
            "status": "healthy|degraded|critical"
        }
    """
    from app.main import app_state

    if not app_state.latest_metrics:
        return {
            "total_nodes": 0,
            "healthy_nodes": 0,
            "unhealthy_nodes": 0,
            "unreachable_nodes": 0,
            "total_gpus": 0,
            "status": "no_data",
        }

    # Extract metrics
    gpu_data = app_state.latest_metrics.get("gpu", {})
    util_data = gpu_data.get("utilization", {})
    memory_data = gpu_data.get("memory", {})
    temp_data = gpu_data.get("temperature", {})

    # Calculate cluster stats
    if not app_state.ssh_manager:
        return {
            "total_nodes": 0,
            "healthy_nodes": 0,
            "unhealthy_nodes": 0,
            "unreachable_nodes": 0,
            "total_gpus": 0,
            "status": "no_ssh_manager",
        }

    total_nodes = len(app_state.ssh_manager.host_list)
    unreachable_nodes = len(app_state.ssh_manager.unreachable_hosts)

    # Check health of each reachable node
    healthy_nodes = 0
    unhealthy_nodes = 0

    for node in app_state.ssh_manager.reachable_hosts:
        node_status, _ = check_node_health(node, gpu_data)
        if node_status == "healthy":
            healthy_nodes += 1
        else:
            unhealthy_nodes += 1

    total_gpus = 0
    total_util = 0
    total_memory_util = 0
    total_temp = 0
    gpu_count = 0
    memory_count = 0
    temp_count = 0

    for node, node_data in util_data.items():
        if isinstance(node_data, dict) and "error" not in node_data:
            gpu_cards = len(node_data.keys())
            total_gpus += gpu_cards

            # Calculate average utilization
            for gpu_id, gpu_metrics in node_data.items():
                if isinstance(gpu_metrics, dict):
                    util = gpu_metrics.get("GPU use (%)", 0)
                    if util:
                        total_util += float(util)
                        gpu_count += 1

    # Calculate average memory utilization
    for node, node_data in memory_data.items():
        if isinstance(node_data, dict) and "error" not in node_data:
            for gpu_id, gpu_metrics in node_data.items():
                if isinstance(gpu_metrics, dict):
                    mem_used = gpu_metrics.get("VRAM Total Used Memory (B)", 0)
                    mem_total = gpu_metrics.get("VRAM Total Memory (B)", 0)
                    if mem_total and int(mem_total) > 0:
                        mem_percent = (int(mem_used) / int(mem_total)) * 100
                        total_memory_util += mem_percent
                        memory_count += 1

    # Calculate average temperature
    for node, node_data in temp_data.items():
        if isinstance(node_data, dict) and "error" not in node_data:
            for gpu_id, gpu_metrics in node_data.items():
                if isinstance(gpu_metrics, dict):
                    temp = gpu_metrics.get("Temperature (Sensor junction) (C)") or gpu_metrics.get(
                        "Temperature (Sensor edge) (C)"
                    )
                    if temp:
                        total_temp += float(temp)
                        temp_count += 1

    avg_util = total_util / gpu_count if gpu_count > 0 else 0
    avg_memory_util = total_memory_util / memory_count if memory_count > 0 else 0
    avg_temp = total_temp / temp_count if temp_count > 0 else 0

    # Determine overall cluster status
    if unreachable_nodes > 0:
        cluster_status = "critical"
    elif unhealthy_nodes > 0:
        cluster_status = "degraded"
    else:
        cluster_status = "healthy"

    return {
        "total_nodes": total_nodes,
        "healthy_nodes": healthy_nodes,
        "unhealthy_nodes": unhealthy_nodes,
        "unreachable_nodes": unreachable_nodes,
        "total_gpus": total_gpus,
        "avg_gpu_utilization": round(avg_util, 2),
        "avg_gpu_memory_utilization": round(avg_memory_util, 2),
        "avg_gpu_temperature": round(avg_temp, 1),
        "status": cluster_status,
        "last_update": app_state.latest_metrics.get("timestamp"),
    }


@router.get("/health")
async def get_cluster_health() -> Dict[str, Any]:
    """
    Get detailed cluster health information.

    Returns health status of all nodes including errors and alerts.
    """
    from app.main import app_state

    if not app_state.latest_metrics:
        raise HTTPException(status_code=503, detail="No metrics available yet")

    health_data = {
        "timestamp": app_state.latest_metrics.get("timestamp"),
        "nodes": {},
        "alerts": [],
    }

    # Check GPU health
    gpu_data = app_state.latest_metrics.get("gpu", {})

    # Check all reachable nodes
    for node in app_state.ssh_manager.reachable_hosts:
        node_status, issues = check_node_health(node, gpu_data)
        health_data["nodes"][node] = {
            "status": node_status,
            "issues": issues,
        }

        # Create alerts for unhealthy nodes
        if node_status == "unhealthy":
            for issue in issues:
                health_data["alerts"].append({"severity": "warning", "node": node, "message": issue})

    # Check unreachable nodes
    for node in app_state.ssh_manager.unreachable_hosts:
        health_data["nodes"][node] = {
            "status": "unreachable",
            "issues": ["Node is unreachable via SSH"],
        }
        health_data["alerts"].append({"severity": "critical", "node": node, "message": f"Node {node} is unreachable"})

    return health_data
