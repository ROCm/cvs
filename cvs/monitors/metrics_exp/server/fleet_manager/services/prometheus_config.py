"""Dynamic Prometheus configuration management."""

import os
import json
import logging
import httpx
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")
PROMETHEUS_TARGETS_PATH = os.environ.get("PROMETHEUS_TARGETS_PATH", "/etc/prometheus/targets")


class PrometheusConfigManager:
    """Manages Prometheus target configuration for dynamic service discovery."""

    def __init__(self, targets_path: str = PROMETHEUS_TARGETS_PATH):
        self.targets_path = Path(targets_path)
        self.targets_path.mkdir(parents=True, exist_ok=True)

    def _get_gpu_targets_file(self, node_group_name: str) -> Path:
        """Get path to GPU metrics targets file for a node group."""
        safe_name = "".join(c if c.isalnum() else "_" for c in node_group_name)
        return self.targets_path / f"gpu_{safe_name}.json"

    def _get_node_targets_file(self, node_group_name: str) -> Path:
        """Get path to node exporter targets file for a node group."""
        safe_name = "".join(c if c.isalnum() else "_" for c in node_group_name)
        return self.targets_path / f"node_{safe_name}.json"

    def _get_rdma_targets_file(self, node_group_name: str) -> Path:
        """Get path to RDMA exporter targets file for a node group."""
        safe_name = "".join(c if c.isalnum() else "_" for c in node_group_name)
        return self.targets_path / f"rdma_{safe_name}.json"

    def _get_user_activity_targets_file(self, node_group_name: str) -> Path:
        """Get path to user activity exporter targets file for a node group."""
        safe_name = "".join(c if c.isalnum() else "_" for c in node_group_name)
        return self.targets_path / f"user_activity_{safe_name}.json"

    def update_node_group_targets(
        self,
        node_group_name: str,
        nodes: List[Dict[str, Any]],
    ) -> bool:
        """
        Update Prometheus targets for a node group.

        Args:
            node_group_name: Name of the node group
            nodes: List of node dicts with 'ip', 'hostname', 'gpu_port', 'node_port', 'status'

        Returns:
            True if successful
        """
        try:
            # Filter active nodes only
            active_nodes = [n for n in nodes if n.get("status") == "active"]

            # GPU metrics targets
            gpu_targets = []
            if active_nodes:
                gpu_targets = [
                    {
                        "targets": [f"{n['ip']}:{n.get('gpu_port', 5000)}" for n in active_nodes],
                        "labels": {
                            "node_group": node_group_name,
                            "job": "amd_gpu_metrics",
                        },
                    }
                ]

                # Add per-node labels for hostname
                gpu_targets = []
                for node in active_nodes:
                    gpu_targets.append(
                        {
                            "targets": [f"{node['ip']}:{node.get('gpu_port', 5000)}"],
                            "labels": {
                                "node_group": node_group_name,
                                "hostname": node.get("hostname", node["ip"]),
                                "job": "amd_gpu_metrics",
                            },
                        }
                    )

            gpu_file = self._get_gpu_targets_file(node_group_name)
            with open(gpu_file, "w") as f:
                json.dump(gpu_targets, f, indent=2)

            # Node exporter targets
            node_targets = []
            for node in active_nodes:
                node_targets.append(
                    {
                        "targets": [f"{node['ip']}:{node.get('node_port', 9100)}"],
                        "labels": {
                            "node_group": node_group_name,
                            "hostname": node.get("hostname", node["ip"]),
                            "job": "node_exporter",
                        },
                    }
                )

            node_file = self._get_node_targets_file(node_group_name)
            with open(node_file, "w") as f:
                json.dump(node_targets, f, indent=2)

            # RDMA exporter targets
            rdma_targets = []
            for node in active_nodes:
                rdma_targets.append(
                    {
                        "targets": [f"{node['ip']}:{node.get('rdma_port', 9417)}"],
                        "labels": {
                            "node_group": node_group_name,
                            "hostname": node.get("hostname", node["ip"]),
                            "job": "rdma_metrics",
                        },
                    }
                )

            rdma_file = self._get_rdma_targets_file(node_group_name)
            with open(rdma_file, "w") as f:
                json.dump(rdma_targets, f, indent=2)

            # User activity exporter targets (port 9420)
            user_activity_targets = []
            for node in active_nodes:
                user_activity_targets.append(
                    {
                        "targets": [f"{node['ip']}:9420"],
                        "labels": {
                            "node_group": node_group_name,
                            "hostname": node.get("hostname", node["ip"]),
                            "job": "user_activity",
                        },
                    }
                )
            user_activity_file = self._get_user_activity_targets_file(node_group_name)
            with open(user_activity_file, "w") as f:
                json.dump(user_activity_targets, f, indent=2)

            logger.info(f"Updated targets for node group '{node_group_name}': {len(active_nodes)} nodes")
            return True

        except Exception as e:
            logger.error(f"Failed to update targets for '{node_group_name}': {e}")
            return False

    def remove_node_group_targets(self, node_group_name: str) -> bool:
        """Remove all targets for a node group."""
        try:
            gpu_file = self._get_gpu_targets_file(node_group_name)
            node_file = self._get_node_targets_file(node_group_name)
            rdma_file = self._get_rdma_targets_file(node_group_name)
            user_activity_file = self._get_user_activity_targets_file(node_group_name)

            for f in [gpu_file, node_file, rdma_file, user_activity_file]:
                if f.exists():
                    f.unlink()

            logger.info(f"Removed targets for node group '{node_group_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to remove targets for '{node_group_name}': {e}")
            return False

    def get_all_targets(self) -> Dict[str, List[Dict]]:
        """Get all configured targets."""
        targets = {"gpu": [], "node": [], "rdma": []}

        for file in self.targets_path.glob("gpu_*.json"):
            try:
                with open(file) as f:
                    targets["gpu"].extend(json.load(f))
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")

        for file in self.targets_path.glob("node_*.json"):
            try:
                with open(file) as f:
                    targets["node"].extend(json.load(f))
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")

        for file in self.targets_path.glob("rdma_*.json"):
            try:
                with open(file) as f:
                    targets["rdma"].extend(json.load(f))
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")

        return targets

    async def reload_prometheus(self) -> bool:
        """Trigger Prometheus configuration reload."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{PROMETHEUS_URL}/-/reload")
                if response.status_code == 200:
                    logger.info("Prometheus configuration reloaded")
                    return True
                else:
                    logger.error(f"Prometheus reload failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Failed to reload Prometheus: {e}")
            return False

    async def check_prometheus_health(self) -> bool:
        """Check if Prometheus is healthy."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{PROMETHEUS_URL}/-/healthy")
                return response.status_code == 200
        except Exception:
            return False

    async def query_prometheus(self, query: str) -> Dict[str, Any]:
        """Execute a PromQL query."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
                if response.status_code == 200:
                    return response.json()
                return {"status": "error", "error": response.text}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ----------------------------------------------------------------
    # Control Node Group target management
    # ----------------------------------------------------------------

    def _get_control_node_targets_file(self, group_name: str) -> Path:
        """Path to node_exporter targets file for a control node group."""
        safe_name = "".join(c if c.isalnum() else "_" for c in group_name)
        return self.targets_path / f"control_node_{safe_name}.json"

    def _get_slurm_targets_file(self, group_name: str) -> Path:
        safe_name = "".join(c if c.isalnum() else "_" for c in group_name)
        return self.targets_path / f"slurm_{safe_name}.json"

    def _get_k8s_targets_file(self, group_name: str) -> Path:
        safe_name = "".join(c if c.isalnum() else "_" for c in group_name)
        return self.targets_path / f"k8s_{safe_name}.json"

    def update_control_node_group_targets(
        self,
        group_name: str,
        control_type: str,
        nodes: List[Dict[str, Any]],
    ) -> bool:
        """
        Update Prometheus targets for a control node group.

        Args:
            group_name:   Name of the control node group
            control_type: "slurm" or "kubernetes"
            nodes:        List of node dicts with keys:
                          ip, hostname, node_port, exporter_port, status
        Returns:
            True if successful
        """
        try:
            active_nodes = [n for n in nodes if n.get("status") == "active"]

            # ---- node_exporter targets (control_node_exporter job) ----
            node_targets = []
            for node in active_nodes:
                node_targets.append(
                    {
                        "targets": [f"{node['ip']}:{node.get('node_port', 9100)}"],
                        "labels": {
                            "control_node_group": group_name,
                            "hostname": node.get("hostname", node["ip"]),
                            "job": "control_node_exporter",
                        },
                    }
                )
            control_node_file = self._get_control_node_targets_file(group_name)
            with open(control_node_file, "w") as f:
                json.dump(node_targets, f, indent=2)

            # ---- Custom exporter targets (slurm_metrics or k8s_control_plane job) ----
            default_exporter_port = 9418 if control_type == "slurm" else 9419
            job_name = "slurm_metrics" if control_type == "slurm" else "k8s_control_plane"

            exporter_targets = []
            for node in active_nodes:
                exporter_targets.append(
                    {
                        "targets": [f"{node['ip']}:{node.get('exporter_port', default_exporter_port)}"],
                        "labels": {
                            "control_node_group": group_name,
                            "hostname": node.get("hostname", node["ip"]),
                            "job": job_name,
                        },
                    }
                )

            if control_type == "slurm":
                target_file = self._get_slurm_targets_file(group_name)
            else:
                target_file = self._get_k8s_targets_file(group_name)

            with open(target_file, "w") as f:
                json.dump(exporter_targets, f, indent=2)

            logger.info(
                f"Updated control node group targets '{group_name}' ({control_type}): {len(active_nodes)} active nodes"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update control node targets for '{group_name}': {e}")
            return False

    def remove_control_node_group_targets(self, group_name: str, control_type: str) -> bool:
        """Remove all Prometheus target files for a control node group."""
        try:
            files_to_remove = [self._get_control_node_targets_file(group_name)]
            if control_type == "slurm":
                files_to_remove.append(self._get_slurm_targets_file(group_name))
            else:
                files_to_remove.append(self._get_k8s_targets_file(group_name))

            for f in files_to_remove:
                if f.exists():
                    f.unlink()

            logger.info(f"Removed control node group targets for '{group_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to remove control node targets for '{group_name}': {e}")
            return False
