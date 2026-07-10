"""Grafana dashboard provisioning and management."""

import os
import logging
import base64
import httpx
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://grafana:3000")
GRAFANA_API_KEY = os.environ.get("GRAFANA_API_KEY", "")
GRAFANA_ADMIN_USER = os.environ.get("GRAFANA_ADMIN_USER", "admin")
GRAFANA_ADMIN_PASSWORD = os.environ.get("GRAFANA_ADMIN_PASSWORD", "admin")


class GrafanaProvisioner:
    """Manages Grafana dashboards and folders for node groups."""

    def __init__(self, base_url: str = GRAFANA_URL, api_key: str = GRAFANA_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        else:
            # Use admin credentials from environment variables
            credentials = f"{GRAFANA_ADMIN_USER}:{GRAFANA_ADMIN_PASSWORD}"
            encoded = base64.b64encode(credentials.encode()).decode()
            self._headers["Authorization"] = f"Basic {encoded}"

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an API request to Grafana."""
        url = f"{self.base_url}{endpoint}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=self._headers,
                    json=data,
                    timeout=30.0,
                )
                if response.status_code >= 400:
                    logger.error(f"Grafana API error: {response.status_code} - {response.text}")
                    return {"error": response.text, "status_code": response.status_code}
                return response.json() if response.text else {}
        except Exception as e:
            logger.error(f"Grafana request failed: {e}")
            return {"error": str(e)}

    async def check_health(self) -> bool:
        """Check if Grafana is healthy."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/health")
                return response.status_code == 200
        except Exception:
            return False

    async def create_folder(self, title: str, uid: Optional[str] = None) -> Dict[str, Any]:
        """Create a Grafana folder for organizing dashboards."""
        data = {"title": title}
        if uid:
            data["uid"] = uid

        result = await self._request("POST", "/api/folders", data)
        if "error" not in result:
            logger.info(f"Created Grafana folder: {title}")
        return result

    async def get_or_create_folder(self, title: str, uid: str) -> str:
        """Get existing folder or create new one."""
        # Try to get existing folder
        result = await self._request("GET", f"/api/folders/{uid}")
        if "error" not in result and "uid" in result:
            return result["uid"]

        # Create new folder
        result = await self.create_folder(title, uid)
        return result.get("uid", uid)

    async def create_dashboard(
        self,
        dashboard: Dict[str, Any],
        folder_uid: Optional[str] = None,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """Create or update a Grafana dashboard."""
        payload = {
            "dashboard": dashboard,
            "overwrite": overwrite,
        }
        if folder_uid:
            payload["folderUid"] = folder_uid

        result = await self._request("POST", "/api/dashboards/db", payload)
        if "error" not in result:
            logger.info(f"Created/updated dashboard: {dashboard.get('title', 'Unknown')}")
        return result

    async def get_dashboard(self, uid: str) -> Optional[Dict[str, Any]]:
        """Get a dashboard by UID."""
        result = await self._request("GET", f"/api/dashboards/uid/{uid}")
        if "error" in result:
            return None
        return result.get("dashboard")

    async def delete_dashboard(self, uid: str) -> bool:
        """Delete a dashboard by UID."""
        result = await self._request("DELETE", f"/api/dashboards/uid/{uid}")
        return "error" not in result

    async def list_dashboards(self, folder_uid: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all dashboards, optionally filtered by folder."""
        endpoint = "/api/search?type=dash-db"
        if folder_uid:
            endpoint += f"&folderUIDs={folder_uid}"

        result = await self._request("GET", endpoint)
        if isinstance(result, list):
            return result
        return []

    async def setup_alert_contact_point(
        self,
        name: str,
        type: str = "email",
        settings: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Set up an alert contact point."""
        data = {
            "name": name,
            "type": type,
            "settings": settings or {},
        }
        return await self._request("POST", "/api/v1/provisioning/contact-points", data)

    async def create_alert_rule(
        self,
        title: str,
        condition: str,
        folder_uid: str,
        labels: Optional[Dict] = None,
        annotations: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create an alert rule."""
        data = {
            "title": title,
            "ruleGroup": "fleet-alerts",
            "folderUid": folder_uid,
            "condition": condition,
            "labels": labels or {},
            "annotations": annotations or {},
        }
        return await self._request("POST", "/api/v1/provisioning/alert-rules", data)

    def generate_node_group_dashboard(
        self,
        node_group_name: str,
        include_gpu: bool = True,
        include_cpu: bool = True,
        include_logs: bool = True,
    ) -> Dict[str, Any]:
        """Generate a dashboard template for a node group."""
        safe_name = "".join(c if c.isalnum() else "_" for c in node_group_name)
        uid = f"ng_{safe_name}"[:40]

        panels = []
        panel_id = 1
        y_pos = 0

        # Row: Overview
        panels.append(
            {
                "id": panel_id,
                "type": "row",
                "title": "Overview",
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
            }
        )
        panel_id += 1
        y_pos += 1

        # Active nodes count
        panels.append(
            {
                "id": panel_id,
                "type": "stat",
                "title": "Active Nodes",
                "gridPos": {"h": 4, "w": 4, "x": 0, "y": y_pos},
                "targets": [
                    {
                        "expr": f'count(up{{job="amd_gpu_metrics", node_group="{node_group_name}"}} == 1)',
                        "legendFormat": "Active",
                    }
                ],
                "options": {"colorMode": "value", "graphMode": "none"},
            }
        )
        panel_id += 1

        # Total GPUs
        panels.append(
            {
                "id": panel_id,
                "type": "stat",
                "title": "Total GPUs",
                "gridPos": {"h": 4, "w": 4, "x": 4, "y": y_pos},
                "targets": [
                    {
                        "expr": f'count(gpu_health{{node_group="{node_group_name}"}})',
                        "legendFormat": "GPUs",
                    }
                ],
                "options": {"colorMode": "value", "graphMode": "none"},
            }
        )
        panel_id += 1

        if include_gpu:
            # GPU Health
            panels.append(
                {
                    "id": panel_id,
                    "type": "stat",
                    "title": "GPU Health",
                    "gridPos": {"h": 4, "w": 4, "x": 8, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'avg(gpu_health{{node_group="{node_group_name}"}})',
                            "legendFormat": "Health",
                        }
                    ],
                    "options": {"colorMode": "value", "graphMode": "none"},
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "red", "value": 0},
                                    {"color": "green", "value": 1},
                                ],
                            }
                        }
                    },
                }
            )
            panel_id += 1

            # Avg GPU Temperature
            panels.append(
                {
                    "id": panel_id,
                    "type": "gauge",
                    "title": "Avg GPU Temperature",
                    "gridPos": {"h": 4, "w": 4, "x": 12, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'avg(gpu_junction_temperature{{node_group="{node_group_name}"}})',
                            "legendFormat": "Temp",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "celsius",
                            "max": 100,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 70},
                                    {"color": "red", "value": 85},
                                ],
                            },
                        }
                    },
                }
            )
            panel_id += 1

            # Avg GPU Power
            panels.append(
                {
                    "id": panel_id,
                    "type": "gauge",
                    "title": "Avg GPU Power",
                    "gridPos": {"h": 4, "w": 4, "x": 16, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'avg(gpu_power_usage{{node_group="{node_group_name}"}})',
                            "legendFormat": "Power",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "watt",
                            "max": 700,
                            "thresholds": {
                                "mode": "percentage",
                                "steps": [
                                    {"color": "green", "value": 0},
                                    {"color": "yellow", "value": 70},
                                    {"color": "red", "value": 90},
                                ],
                            },
                        }
                    },
                }
            )
            panel_id += 1

            # GPU Utilization
            panels.append(
                {
                    "id": panel_id,
                    "type": "gauge",
                    "title": "Avg GPU Utilization",
                    "gridPos": {"h": 4, "w": 4, "x": 20, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'avg(gpu_gfx_activity{{node_group="{node_group_name}"}})',
                            "legendFormat": "Utilization",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "max": 100,
                        }
                    },
                }
            )
            panel_id += 1
            y_pos += 4

            # GPU Metrics Row
            panels.append(
                {
                    "id": panel_id,
                    "type": "row",
                    "title": "GPU Metrics",
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
                }
            )
            panel_id += 1
            y_pos += 1

            # GPU Temperature over time
            panels.append(
                {
                    "id": panel_id,
                    "type": "timeseries",
                    "title": "GPU Temperature",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'gpu_junction_temperature{{node_group="{node_group_name}"}}',
                            "legendFormat": "{{hostname}} GPU{{gpu}}",
                        }
                    ],
                    "fieldConfig": {"defaults": {"unit": "celsius"}},
                }
            )
            panel_id += 1

            # GPU Power over time
            panels.append(
                {
                    "id": panel_id,
                    "type": "timeseries",
                    "title": "GPU Power Usage",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'gpu_power_usage{{node_group="{node_group_name}"}}',
                            "legendFormat": "{{hostname}} GPU{{gpu}}",
                        }
                    ],
                    "fieldConfig": {"defaults": {"unit": "watt"}},
                }
            )
            panel_id += 1
            y_pos += 8

            # GPU Utilization over time
            panels.append(
                {
                    "id": panel_id,
                    "type": "timeseries",
                    "title": "GPU Utilization",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'gpu_gfx_activity{{node_group="{node_group_name}"}}',
                            "legendFormat": "{{hostname}} GPU{{gpu}}",
                        }
                    ],
                    "fieldConfig": {"defaults": {"unit": "percent", "max": 100}},
                }
            )
            panel_id += 1

            # GPU Memory Usage
            panels.append(
                {
                    "id": panel_id,
                    "type": "timeseries",
                    "title": "GPU Memory Usage",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'gpu_used_vram{{node_group="{node_group_name}"}} / gpu_total_vram{{node_group="{node_group_name}"}} * 100',
                            "legendFormat": "{{hostname}} GPU{{gpu}}",
                        }
                    ],
                    "fieldConfig": {"defaults": {"unit": "percent", "max": 100}},
                }
            )
            panel_id += 1
            y_pos += 8

        if include_cpu:
            # CPU Row
            panels.append(
                {
                    "id": panel_id,
                    "type": "row",
                    "title": "System Metrics",
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
                }
            )
            panel_id += 1
            y_pos += 1

            # CPU Usage
            panels.append(
                {
                    "id": panel_id,
                    "type": "timeseries",
                    "title": "CPU Usage",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'100 - (avg by (hostname) (irate(node_cpu_seconds_total{{mode="idle", node_group="{node_group_name}"}}[5m])) * 100)',
                            "legendFormat": "{{hostname}}",
                        }
                    ],
                    "fieldConfig": {"defaults": {"unit": "percent", "max": 100}},
                }
            )
            panel_id += 1

            # Memory Usage
            panels.append(
                {
                    "id": panel_id,
                    "type": "timeseries",
                    "title": "Memory Usage",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'(1 - node_memory_MemAvailable_bytes{{node_group="{node_group_name}"}} / node_memory_MemTotal_bytes{{node_group="{node_group_name}"}}) * 100',
                            "legendFormat": "{{hostname}}",
                        }
                    ],
                    "fieldConfig": {"defaults": {"unit": "percent", "max": 100}},
                }
            )
            panel_id += 1
            y_pos += 8

        if include_logs:
            # Logs Row
            panels.append(
                {
                    "id": panel_id,
                    "type": "row",
                    "title": "Logs",
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
                }
            )
            panel_id += 1
            y_pos += 1

            # Error Logs
            panels.append(
                {
                    "id": panel_id,
                    "type": "logs",
                    "title": "Critical Logs (GPU Errors, ECC, RAS)",
                    "gridPos": {"h": 10, "w": 24, "x": 0, "y": y_pos},
                    "targets": [
                        {
                            "expr": f'{{node_group="{node_group_name}"}} |~ "(?i)(error|fail|ecc|ras|xgmi|gpu.*hang|amdgpu.*timeout|hardware error|mce|thermal)"',
                            "refId": "A",
                        }
                    ],
                    "datasource": {"type": "loki", "uid": "loki"},
                    "options": {
                        "showTime": True,
                        "showLabels": True,
                        "wrapLogMessage": True,
                        "enableLogDetails": True,
                    },
                }
            )
            panel_id += 1
            y_pos += 10

        # ── User Activity & KFD GPU Process Tracking ─────────────────────
        # This section is always added — it shows data when the user_activity_exporter
        # is installed (Force Reinstall deploys it automatically).
        def _ua_table(pid, title, expr, col_map, x, y, w=24, h=10, desc="", sort_col=None, sort_desc=False):
            exclude = {"Time": True, "Value": True, "__name__": True, "job": True, "instance": True, "node_group": True}
            sort_by = [{"displayName": sort_col, "desc": sort_desc}] if sort_col else []
            return {
                "id": pid,
                "type": "table",
                "title": title,
                "description": desc,
                "gridPos": {"h": h, "w": w, "x": x, "y": y},
                "targets": [{"expr": expr, "instant": True, "legendFormat": "", "format": "table"}],
                "transformations": [
                    {"id": "merge", "options": {}},
                    {
                        "id": "organize",
                        "options": {
                            "excludeByName": exclude,
                            "renameByName": {k: v for k, v in col_map.items()},
                        },
                    },
                ],
                "options": {"sortBy": sort_by},
                "fieldConfig": {"defaults": {"custom": {"displayMode": "auto", "filterable": True}}},
            }

        ng = node_group_name

        panels.append(
            {
                "id": panel_id,
                "type": "row",
                "title": "User Activity & GPU Process Tracking",
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
            }
        )
        panel_id += 1
        y_pos += 1

        # Stat row: KFD process count, logged-in count, unique GPU users
        panels.append(
            {
                "id": panel_id,
                "type": "stat",
                "title": "Active GPU Processes (KFD)",
                "description": "Processes currently launching GPU kernels. 0 = idle or KFD not loaded.",
                "gridPos": {"h": 4, "w": 8, "x": 0, "y": y_pos},
                "targets": [{"expr": f'sum(node_kfd_process_count{{node_group="{ng}"}})', "instant": True}],
                "fieldConfig": {"defaults": {"unit": "short"}},
                "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
            }
        )
        panel_id += 1
        panels.append(
            {
                "id": panel_id,
                "type": "stat",
                "title": "Logged-In Users",
                "gridPos": {"h": 4, "w": 8, "x": 8, "y": y_pos},
                "targets": [{"expr": f'sum(node_logged_in_users_count{{node_group="{ng}"}})', "instant": True}],
                "fieldConfig": {"defaults": {"unit": "short"}},
                "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
            }
        )
        panel_id += 1
        panels.append(
            {
                "id": panel_id,
                "type": "stat",
                "title": "Unique GPU Users (KFD)",
                "gridPos": {"h": 4, "w": 8, "x": 16, "y": y_pos},
                "targets": [
                    {"expr": f'count(count by (user) (node_kfd_process_info{{node_group="{ng}"}}))', "instant": True}
                ],
                "fieldConfig": {"defaults": {"unit": "short"}},
                "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
            }
        )
        panel_id += 1
        y_pos += 4

        # KFD GPU processes table
        panels.append(
            _ua_table(
                panel_id,
                "KFD GPU Processes (Active GPU Kernel Users)",
                f'node_kfd_process_info{{node_group="{ng}"}} == 1',
                {"hostname": "Node", "user": "User", "pid": "PID", "cmd": "Command", "gpu_mem_mb": "GPU Mem (MB)"},
                0,
                y_pos,
                desc="Processes using the GPU right now via KFD. Empty when no GPU workloads are running.",
                sort_col="GPU Mem (MB)",
                sort_desc=True,
            )
        )
        panel_id += 1
        y_pos += 10

        # Currently logged in + process count side by side
        panels.append(
            _ua_table(
                panel_id,
                "Currently Logged-In Users",
                f'node_logged_in_user_info{{node_group="{ng}"}} == 1',
                {
                    "hostname": "Node",
                    "user": "User",
                    "tty": "Terminal",
                    "from_host": "From",
                    "login_time": "Login Time",
                },
                0,
                y_pos,
                w=12,
            )
        )
        panel_id += 1
        panels.append(
            _ua_table(
                panel_id,
                "User Process Summary",
                f'node_user_process_count{{node_group="{ng}"}} > 0',
                {"hostname": "Node", "user": "User", "Value": "Process Count"},
                12,
                y_pos,
                w=12,
                sort_col="Process Count",
                sort_desc=True,
            )
        )
        panel_id += 1
        y_pos += 10

        # Notable commands
        panels.append(
            _ua_table(
                panel_id,
                "Notable User Commands (GPU/ML workloads prioritised)",
                f'node_user_top_process_info{{node_group="{ng}"}} == 1',
                {"hostname": "Node", "user": "User", "cmd": "Command"},
                0,
                y_pos,
            )
        )
        panel_id += 1
        y_pos += 10

        # Recent login history
        panels.append(
            _ua_table(
                panel_id,
                "Recent Login History",
                f'node_recent_login_info{{node_group="{ng}"}} == 1',
                {"hostname": "Node", "user": "User", "tty": "Terminal", "from_host": "From", "date": "Date/Time"},
                0,
                y_pos,
                h=12,
            )
        )
        panel_id += 1
        y_pos += 12

        # Auth logs
        panels.append(
            {
                "id": panel_id,
                "type": "logs",
                "title": "Auth Logs (SSH Logins & sudo Events)",
                "description": "Live from /var/log/auth.log or /var/log/secure. Use the search bar to filter.",
                "datasource": {"type": "loki", "uid": "loki"},
                "gridPos": {"h": 12, "w": 24, "x": 0, "y": y_pos},
                "targets": [
                    {"datasource": {"type": "loki", "uid": "loki"}, "expr": f'{{node_group="{ng}", job="auth"}}'}
                ],
                "options": {"showTime": True, "showLabels": True, "wrapLogMessage": False, "dedupStrategy": "none"},
            }
        )
        panel_id += 1
        y_pos += 12

        dashboard = {
            "uid": uid,
            "title": f"Node Group: {node_group_name}",
            "tags": ["gpu-fleet", "node-group", node_group_name],
            "timezone": "browser",
            "refresh": "30s",
            "schemaVersion": 38,
            "panels": panels,
            "time": {"from": "now-1h", "to": "now"},
            "templating": {
                "list": [
                    {
                        "name": "hostname",
                        "type": "query",
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "query": f'label_values(up{{node_group="{node_group_name}"}}, hostname)',
                        "refresh": 2,
                        "multi": True,
                        "includeAll": True,
                    }
                ]
            },
        }

        return dashboard

    async def provision_node_group_dashboard(
        self,
        node_group_name: str,
        folder_uid: str = "gpu-fleet",
    ) -> Optional[str]:
        """Create a dashboard for a node group."""
        # Ensure folder exists
        await self.get_or_create_folder("GPU Fleet Monitoring", folder_uid)

        # Generate and create dashboard
        dashboard = self.generate_node_group_dashboard(node_group_name)
        result = await self.create_dashboard(dashboard, folder_uid)

        if "error" not in result:
            return result.get("uid") or dashboard["uid"]
        return None

    async def remove_node_group_dashboard(self, node_group_name: str) -> bool:
        """Remove the dashboard for a node group."""
        safe_name = "".join(c if c.isalnum() else "_" for c in node_group_name)
        uid = f"ng_{safe_name}"[:40]
        return await self.delete_dashboard(uid)

    def get_default_dashboards(self) -> List[Dict[str, Any]]:
        """Get all default dashboards for the GPU Fleet Monitoring system."""
        dashboards = []

        # Fleet Overview Dashboard
        dashboards.append(
            {
                "uid": "fleet-overview",
                "title": "GPU Fleet Overview",
                "tags": ["gpu-fleet", "amd", "fleet"],
                "timezone": "browser",
                "schemaVersion": 38,
                "refresh": "30s",
                "time": {"from": "now-1h", "to": "now"},
                "templating": {
                    "list": [
                        {
                            "name": "nodegroup",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(gpu_gfx_activity, node_group)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        }
                    ]
                },
                "panels": [
                    {
                        "id": 1,
                        "type": "stat",
                        "title": "Total GPUs",
                        "gridPos": {"h": 4, "w": 4, "x": 0, "y": 0},
                        "targets": [
                            {"expr": "count(gpu_gfx_activity{node_group=~\"$nodegroup\"})", "legendFormat": "GPUs"}
                        ],
                        "options": {"colorMode": "value"},
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {"mode": "absolute", "steps": [{"color": "blue", "value": None}]}
                            }
                        },
                    },
                    {
                        "id": 2,
                        "type": "stat",
                        "title": "Avg GPU Utilization",
                        "gridPos": {"h": 4, "w": 4, "x": 4, "y": 0},
                        "targets": [
                            {"expr": "avg(gpu_gfx_activity{node_group=~\"$nodegroup\"})", "legendFormat": "Avg"}
                        ],
                        "options": {"colorMode": "value"},
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percent",
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "blue", "value": None},
                                        {"color": "green", "value": 20},
                                        {"color": "yellow", "value": 80},
                                        {"color": "red", "value": 95},
                                    ],
                                },
                            }
                        },
                    },
                    {
                        "id": 3,
                        "type": "timeseries",
                        "title": "GPU Utilization Over Time",
                        "gridPos": {"h": 8, "w": 16, "x": 8, "y": 0},
                        "targets": [
                            {
                                "expr": "gpu_gfx_activity{node_group=~\"$nodegroup\"}",
                                "legendFormat": "{{hostname}} GPU{{gpu}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100}},
                    },
                ],
            }
        )

        # GPU Utilization Dashboard
        dashboards.append(
            {
                "uid": "gpu-utilization",
                "title": "GPU Utilization",
                "tags": ["gpu-fleet", "utilization"],
                "timezone": "browser",
                "schemaVersion": 38,
                "refresh": "30s",
                "time": {"from": "now-1h", "to": "now"},
                "templating": {
                    "list": [
                        {
                            "name": "nodegroup",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(gpu_gfx_activity, node_group)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        },
                        {
                            "name": "hostname",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(gpu_gfx_activity{node_group=~\"$nodegroup\"}, hostname)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        },
                    ]
                },
                "panels": [
                    {
                        "id": 1,
                        "type": "timeseries",
                        "title": "GPU Utilization",
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "gpu_gfx_activity{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
                                "legendFormat": "{{hostname}} GPU{{gpu}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100}},
                    },
                    {
                        "id": 2,
                        "type": "timeseries",
                        "title": "Memory Controller Activity",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "gpu_umc_activity{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
                                "legendFormat": "{{hostname}} GPU{{gpu}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100}},
                    },
                    {
                        "id": 3,
                        "type": "timeseries",
                        "title": "VCN Activity",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "gpu_vcn_busy_instantaneous{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
                                "legendFormat": "{{hostname}} GPU{{gpu}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100}},
                    },
                ],
            }
        )

        # Thermal & Power Dashboard
        dashboards.append(
            {
                "uid": "thermal-power",
                "title": "Thermal & Power",
                "tags": ["gpu-fleet", "thermal", "power"],
                "timezone": "browser",
                "schemaVersion": 38,
                "refresh": "30s",
                "time": {"from": "now-1h", "to": "now"},
                "templating": {
                    "list": [
                        {
                            "name": "nodegroup",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(gpu_junction_temperature, node_group)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        },
                        {
                            "name": "hostname",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(gpu_junction_temperature{node_group=~\"$nodegroup\"}, hostname)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        },
                    ]
                },
                "panels": [
                    {
                        "id": 1,
                        "type": "timeseries",
                        "title": "GPU Temperature",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "gpu_junction_temperature{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
                                "legendFormat": "{{hostname}} GPU{{gpu}}",
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "unit": "celsius",
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "yellow", "value": 75},
                                        {"color": "red", "value": 90},
                                    ],
                                },
                            }
                        },
                    },
                    {
                        "id": 2,
                        "type": "timeseries",
                        "title": "GPU Power",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "gpu_power_usage{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
                                "legendFormat": "{{hostname}} GPU{{gpu}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "watt"}},
                    },
                ],
            }
        )

        # GPU Health Dashboard
        dashboards.append(
            {
                "uid": "gpu-health",
                "title": "GPU Health & Errors",
                "tags": ["gpu-fleet", "health", "ecc", "ras"],
                "timezone": "browser",
                "schemaVersion": 38,
                "refresh": "30s",
                "time": {"from": "now-1h", "to": "now"},
                "templating": {
                    "list": [
                        {
                            "name": "nodegroup",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(gpu_gfx_activity, node_group)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        }
                    ]
                },
                "panels": [
                    {
                        "id": 1,
                        "type": "stat",
                        "title": "ECC Correctable Errors",
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "sum(gpu_ecc_correct_total{node_group=~\"$nodegroup\"}) or vector(0)",
                                "legendFormat": "Correctable",
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "yellow", "value": 1},
                                        {"color": "red", "value": 10},
                                    ],
                                }
                            }
                        },
                    },
                    {
                        "id": 2,
                        "type": "stat",
                        "title": "ECC Uncorrectable Errors",
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0},
                        "targets": [
                            {
                                "expr": "sum(gpu_ecc_uncorrect_total{node_group=~\"$nodegroup\"}) or vector(0)",
                                "legendFormat": "Uncorrectable",
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}],
                                }
                            }
                        },
                    },
                ],
            }
        )

        # CPU & System Dashboard
        dashboards.append(
            {
                "uid": "cpu-system",
                "title": "CPU & System",
                "tags": ["gpu-fleet", "cpu", "system", "memory"],
                "timezone": "browser",
                "schemaVersion": 38,
                "refresh": "30s",
                "time": {"from": "now-1h", "to": "now"},
                "templating": {
                    "list": [
                        {
                            "name": "nodegroup",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(node_cpu_seconds_total, node_group)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        },
                        {
                            "name": "hostname",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(node_cpu_seconds_total{node_group=~\"$nodegroup\"}, hostname)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        },
                    ]
                },
                "panels": [
                    {
                        "id": 1,
                        "type": "timeseries",
                        "title": "CPU Usage",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "100 - (avg by(hostname)(rate(node_cpu_seconds_total{mode=\"idle\", node_group=~\"$nodegroup\", hostname=~\"$hostname\"}[5m])) * 100)",
                                "legendFormat": "{{hostname}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100}},
                    },
                    {
                        "id": 2,
                        "type": "timeseries",
                        "title": "Memory Usage",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "(1 - node_memory_MemAvailable_bytes{node_group=~\"$nodegroup\", hostname=~\"$hostname\"} / node_memory_MemTotal_bytes{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}) * 100",
                                "legendFormat": "{{hostname}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100}},
                    },
                ],
            }
        )

        # RDMA Network Dashboard
        dashboards.append(
            {
                "uid": "rdma-network",
                "title": "RDMA Network",
                "description": "RDMA/RoCE network metrics including traffic, congestion (PFC/DCQCN), and errors",
                "tags": ["gpu-fleet", "rdma", "network", "roce", "pfc", "dcqcn"],
                "timezone": "browser",
                "schemaVersion": 38,
                "refresh": "30s",
                "time": {"from": "now-1h", "to": "now"},
                "templating": {
                    "list": [
                        {
                            "name": "nodegroup",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(rdma_link_state, node_group)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        },
                        {
                            "name": "hostname",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(rdma_link_state{node_group=~\"$nodegroup\"}, hostname)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        },
                        {
                            "name": "device",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(rdma_link_state{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}, device)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        },
                    ]
                },
                "panels": [
                    # Link Status Row
                    {
                        "id": 1,
                        "type": "row",
                        "title": "Link Status Overview",
                        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0},
                    },
                    {
                        "id": 2,
                        "type": "stat",
                        "title": "Total RDMA Ports",
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 1},
                        "targets": [
                            {
                                "expr": "count(rdma_link_state{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"})",
                                "legendFormat": "Total",
                            }
                        ],
                        "options": {"colorMode": "value"},
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {"mode": "absolute", "steps": [{"color": "blue", "value": None}]}
                            }
                        },
                    },
                    {
                        "id": 3,
                        "type": "stat",
                        "title": "Ports Up",
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 1},
                        "targets": [
                            {
                                "expr": "count(rdma_link_physical_state{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"} == 1)",
                                "legendFormat": "Up",
                            }
                        ],
                        "options": {"colorMode": "value"},
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}
                            }
                        },
                    },
                    {
                        "id": 4,
                        "type": "stat",
                        "title": "Ports Down",
                        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 1},
                        "targets": [
                            {
                                "expr": "count(rdma_link_physical_state{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"} == 0) or vector(0)",
                                "legendFormat": "Down",
                            }
                        ],
                        "options": {"colorMode": "value"},
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}],
                                }
                            }
                        },
                    },
                    # Traffic Row
                    {"id": 10, "type": "row", "title": "Traffic - TX/RX", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 5}},
                    {
                        "id": 11,
                        "type": "timeseries",
                        "title": "TX Bytes Rate",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_tx_bytes{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}}/{{port}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "Bps", "custom": {"lineWidth": 1, "fillOpacity": 10}}},
                    },
                    {
                        "id": 12,
                        "type": "timeseries",
                        "title": "RX Bytes Rate",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_rx_bytes{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}}/{{port}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "Bps", "custom": {"lineWidth": 1, "fillOpacity": 10}}},
                    },
                    # Congestion - CNP Row
                    {
                        "id": 50,
                        "type": "row",
                        "title": "Congestion Control - CNP (DCQCN)",
                        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 14},
                    },
                    {
                        "id": 51,
                        "type": "timeseries",
                        "title": "CNP Packets Sent",
                        "description": "Congestion Notification Packets sent (TX CNP)",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 15},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_tx_cnp_pkts{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} tx_cnp",
                            },
                        ],
                        "fieldConfig": {"defaults": {"unit": "pps"}},
                    },
                    {
                        "id": 52,
                        "type": "timeseries",
                        "title": "CNP Packets Received / ECN Marked",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 15},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_rx_cnp_pkts{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} rx_cnp",
                            },
                            {
                                "expr": "rate(rdma_stat_rx_ecn_marked_pkts{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} ecn_marked",
                            },
                        ],
                        "fieldConfig": {"defaults": {"unit": "pps"}},
                    },
                    # Congestion - PFC Row
                    {
                        "id": 60,
                        "type": "row",
                        "title": "Congestion Control - PFC",
                        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 23},
                    },
                    {
                        "id": 61,
                        "type": "timeseries",
                        "title": "DCQCN Pacing Events",
                        "description": "DCQCN rate adjustment pacing events (PFC counters not available via rdma statistic on Broadcom bnxt_re)",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_pacing_alerts{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} pacing_alerts",
                            },
                            {
                                "expr": "rate(rdma_stat_pacing_reschedule{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} pacing_reschedule",
                            },
                        ],
                        "fieldConfig": {"defaults": {"unit": "short"}},
                    },
                    {
                        "id": 62,
                        "type": "timeseries",
                        "title": "DCQCN Pacing Complete",
                        "description": "DCQCN pacing completions",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_pacing_complete{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} pacing_complete",
                            },
                        ],
                        "fieldConfig": {"defaults": {"unit": "short"}},
                    },
                    # Error Statistics Row
                    {
                        "id": 70,
                        "type": "row",
                        "title": "Error Statistics",
                        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 32},
                    },
                    {
                        "id": 71,
                        "type": "timeseries",
                        "title": "Packet Sequence Errors",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 33},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_seq_err_naks_rcvd{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} seq_err_naks",
                            },
                            {
                                "expr": "rate(rdma_stat_oos_drop_count{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} oos_drops",
                            },
                        ],
                        "fieldConfig": {"defaults": {"unit": "pps"}},
                    },
                    {
                        "id": 72,
                        "type": "timeseries",
                        "title": "ICRC & Retry Errors",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 33},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_rx_roce_errors{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} rx_roce_err",
                            },
                            {
                                "expr": "rate(rdma_stat_rnr_naks_rcvd{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} rnr_naks",
                            },
                            {
                                "expr": "rate(rdma_stat_to_retransmits{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} timeout_retransmits",
                            },
                        ],
                        "fieldConfig": {"defaults": {"unit": "pps"}},
                    },
                    # Resources Row
                    {"id": 80, "type": "row", "title": "Resources", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 41}},
                    {
                        "id": 81,
                        "type": "bargauge",
                        "title": "Queue Pairs by Host",
                        "gridPos": {"h": 6, "w": 8, "x": 0, "y": 42},
                        "targets": [
                            {
                                "expr": "rdma_resource_qp_total{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
                                "legendFormat": "{{hostname}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "short"}},
                        "options": {"orientation": "horizontal", "displayMode": "gradient"},
                    },
                    {
                        "id": 82,
                        "type": "bargauge",
                        "title": "Completion Queues by Host",
                        "gridPos": {"h": 6, "w": 8, "x": 8, "y": 42},
                        "targets": [
                            {
                                "expr": "rdma_resource_cq_total{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
                                "legendFormat": "{{hostname}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "short"}},
                        "options": {"orientation": "horizontal", "displayMode": "gradient"},
                    },
                    {
                        "id": 83,
                        "type": "bargauge",
                        "title": "Memory Regions by Host",
                        "gridPos": {"h": 6, "w": 8, "x": 16, "y": 42},
                        "targets": [
                            {
                                "expr": "rdma_resource_mr_total{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
                                "legendFormat": "{{hostname}}",
                            }
                        ],
                        "fieldConfig": {"defaults": {"unit": "short"}},
                        "options": {"orientation": "horizontal", "displayMode": "gradient"},
                    },
                ],
            }
        )

        # Logs Analysis Dashboard
        dashboards.append(
            {
                "uid": "logs-analysis",
                "title": "Logs Analysis",
                "tags": ["gpu-fleet", "logs", "dmesg", "errors"],
                "timezone": "browser",
                "schemaVersion": 38,
                "refresh": "30s",
                "time": {"from": "now-1h", "to": "now"},
                "panels": [
                    {
                        "id": 1,
                        "type": "logs",
                        "title": "System Logs",
                        "gridPos": {"h": 12, "w": 24, "x": 0, "y": 0},
                        "targets": [
                            {
                                "datasource": {"type": "loki", "uid": "loki"},
                                "expr": "{job=~\"systemd-journal|syslog|dmesg\"}",
                            }
                        ],
                        "options": {"showTime": True, "showLabels": True, "wrapLogMessage": True},
                    }
                ],
            }
        )

        return dashboards

    async def provision_default_dashboards(self, folder_uid: str = "gpu-fleet") -> Dict[str, bool]:
        """Provision all default dashboards to Grafana.

        Returns a dict mapping dashboard uid to success status.
        """
        results = {}

        # Ensure folder exists
        await self.get_or_create_folder("GPU Fleet Monitoring", folder_uid)

        dashboards = self.get_default_dashboards()
        for dashboard in dashboards:
            uid = dashboard.get("uid", "unknown")
            result = await self.create_dashboard(dashboard, folder_uid)
            results[uid] = "error" not in result
            if "error" in result:
                logger.error(f"Failed to provision dashboard {uid}: {result.get('error')}")
            else:
                logger.info(f"Provisioned dashboard: {dashboard.get('title')}")

        return results

    # ----------------------------------------------------------------
    # Control Node Group Dashboards
    # ----------------------------------------------------------------

    def _cng_safe_name(self, group_name: str) -> str:
        """Return safe UID component for a control node group name."""
        return ("".join(c if c.isalnum() else "_" for c in group_name))[:35]

    def generate_slurm_dashboard(self, group_name: str) -> Dict[str, Any]:
        """Generate a comprehensive Grafana dashboard for a Slurm control node group."""
        safe_name = self._cng_safe_name(group_name)
        uid = f"cng_{safe_name}"
        ng = group_name

        def ts(panel_id, title, expr_list, x, y, w=12, h=8, unit="short"):
            targets = [{"expr": e, "legendFormat": lf} for e, lf in expr_list]
            return {
                "id": panel_id,
                "type": "timeseries",
                "title": title,
                "gridPos": {"h": h, "w": w, "x": x, "y": y},
                "targets": targets,
                "fieldConfig": {"defaults": {"unit": unit}},
            }

        def stat(panel_id, title, expr, x, y, w=4, h=4, unit="short", thresholds=None, description=""):
            p = {
                "id": panel_id,
                "type": "stat",
                "title": title,
                "description": description,
                "gridPos": {"h": h, "w": w, "x": x, "y": y},
                "targets": [{"expr": expr, "instant": True}],
                "fieldConfig": {"defaults": {"unit": unit}},
                "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
            }
            if thresholds:
                p["fieldConfig"]["defaults"]["thresholds"] = thresholds
            return p

        def row(panel_id, title, y):
            return {"id": panel_id, "type": "row", "title": title, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}}

        def table(panel_id, title, expr, col_map, x, y, w=24, h=12, sort_col=None, desc=False):
            """
            Table panel using format=table so Prometheus returns all rows directly.
            The merge transformation combines multiple frames into one scrollable table.
            h=12 gives enough height to show ~8 rows with a scrollbar for overflow.
            """
            exclude = {
                "Time": True,
                "Value": True,
                "__name__": True,
                "job": True,
                "instance": True,
                "control_node_group": True,
            }
            sort_by = [{"displayName": sort_col, "desc": desc}] if sort_col else []
            return {
                "id": panel_id,
                "type": "table",
                "title": title,
                "gridPos": {"h": h, "w": w, "x": x, "y": y},
                "targets": [{"expr": expr, "instant": True, "legendFormat": "", "format": "table"}],
                "transformations": [
                    {"id": "merge", "options": {}},
                    {
                        "id": "organize",
                        "options": {
                            "excludeByName": exclude,
                            "renameByName": {k: v for k, v in col_map.items()},
                        },
                    },
                ],
                "options": {"sortBy": sort_by},
                "fieldConfig": {
                    "defaults": {
                        "custom": {
                            "displayMode": "auto",
                            "filterable": True,
                        }
                    }
                },
            }

        up_thresh = {"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "green", "value": 1}]}

        p = 1
        y = 0
        panels = []

        # ---- Row 1: Cluster Compute Overview ----
        # These stats reflect the compute nodes (not the head node itself)
        panels.append(row(p, "Cluster Compute Overview (69 compute nodes, not the head node)", y))
        p += 1
        y += 1
        panels.append(
            stat(
                p,
                "Total Compute Nodes",
                f'slurm_nodes_total{{control_node_group="{ng}"}}',
                0,
                y,
                description="Total compute nodes managed by this Slurm cluster",
            )
        )
        p += 1
        panels.append(stat(p, "Running Jobs", f'slurm_jobs_state{{control_node_group="{ng}",state="running"}}', 4, y))
        p += 1
        panels.append(stat(p, "Pending Jobs", f'slurm_jobs_state{{control_node_group="{ng}",state="pending"}}', 8, y))
        p += 1
        panels.append(
            stat(
                p,
                "Compute CPU Util %",
                f'100 * slurm_running_cpus{{control_node_group="{ng}"}} '
                f'/ clamp_min(slurm_cpus_total{{control_node_group="{ng}"}}, 1)',
                12,
                y,
                unit="percent",
                description="% of COMPUTE NODE CPUs used by actively running jobs. "
                "This is cluster-wide utilization across all compute nodes, "
                "NOT the head node CPU (head node is always near 0%).",
                thresholds={
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 70},
                        {"color": "red", "value": 90},
                    ],
                },
            )
        )
        p += 1
        panels.append(
            stat(
                p,
                "Running CPUs",
                f'slurm_running_cpus{{control_node_group="{ng}"}}',
                16,
                y,
                description="Total CPUs actively used by running jobs across all compute nodes",
            )
        )
        p += 1
        panels.append(
            stat(
                p,
                "Total Cluster CPUs",
                f'slurm_cpus_total{{control_node_group="{ng}"}}',
                20,
                y,
                description="Total CPUs available across all compute nodes in the cluster",
            )
        )
        p += 1
        y += 4

        # ---- Row 2: Head Node Health ----
        # These stats reflect the head/login node where slurmctld runs
        panels.append(row(p, "Head Node Health (the slurmctld server — always near 0% CPU, that is normal)", y))
        p += 1
        y += 1
        panels.append(
            stat(
                p,
                "slurmctld",
                f'slurm_slurmctld_up{{control_node_group="{ng}"}}',
                0,
                y,
                description="slurmctld reachability via scontrol ping",
                thresholds=up_thresh,
            )
        )
        p += 1
        panels.append(
            stat(
                p,
                "slurmdbd",
                f'slurm_slurmdbd_up{{control_node_group="{ng}"}}',
                4,
                y,
                description="slurmdbd reachability via sacctmgr",
                thresholds=up_thresh,
            )
        )
        p += 1
        panels.append(
            stat(
                p,
                "Head Node CPU %",
                f'100 - (avg(irate(node_cpu_seconds_total{{control_node_group="{ng}",mode="idle"}}[5m])) * 100)',
                8,
                y,
                unit="percent",
                description="Head node's own CPU usage. Expected to be near 0% — "
                "the head node only runs slurmctld/slurmdbd, not compute jobs.",
            )
        )
        p += 1
        panels.append(
            stat(
                p,
                "Head Node Memory %",
                f'100 * (1 - node_memory_MemAvailable_bytes{{control_node_group="{ng}"}}'
                f' / node_memory_MemTotal_bytes{{control_node_group="{ng}"}})',
                12,
                y,
                unit="percent",
                description="Head node's own memory usage",
                thresholds={
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 75},
                        {"color": "red", "value": 90},
                    ],
                },
            )
        )
        p += 1
        panels.append(
            stat(
                p,
                "Head Node Disk %",
                f'max(100 * (1 - node_filesystem_avail_bytes{{control_node_group="{ng}",'
                f'fstype!~"tmpfs|devtmpfs|squashfs"}}'
                f' / node_filesystem_size_bytes{{control_node_group="{ng}",'
                f'fstype!~"tmpfs|devtmpfs|squashfs"}}))',
                16,
                y,
                unit="percent",
                description="Head node's most-used filesystem",
                thresholds={
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 75},
                        {"color": "red", "value": 90},
                    ],
                },
            )
        )
        p += 1
        y += 4

        # ---- Row 2: Node states + CPU ----
        panels.append(row(p, "Node States & CPU", y))
        p += 1
        y += 1
        panels.append(
            ts(
                p,
                "Node States Over Time",
                [
                    (f'slurm_nodes_state{{control_node_group="{ng}",state="idle"}}', "Idle"),
                    (f'slurm_nodes_state{{control_node_group="{ng}",state="allocated"}}', "Allocated"),
                    (f'slurm_nodes_state{{control_node_group="{ng}",state="mixed"}}', "Mixed"),
                    (f'slurm_nodes_state{{control_node_group="{ng}",state="planned"}}', "Planned"),
                    (f'slurm_nodes_state{{control_node_group="{ng}",state="drained"}}', "Drained"),
                    (f'slurm_nodes_state{{control_node_group="{ng}",state="down"}}', "Down"),
                ],
                0,
                y,
            )
        )
        p += 1
        panels.append(
            ts(
                p,
                "CPU — Running vs Total",
                [
                    (f'slurm_cpus_total{{control_node_group="{ng}"}}', "Total CPUs"),
                    (f'slurm_running_cpus{{control_node_group="{ng}"}}', "Running CPUs"),
                    (f'slurm_cpus_allocated{{control_node_group="{ng}"}}', "Allocated (alloc+mixed nodes)"),
                ],
                12,
                y,
            )
        )
        p += 1
        y += 8

        # ---- Row 3: Jobs + Memory ----
        panels.append(row(p, "Jobs & Memory", y))
        p += 1
        y += 1
        panels.append(
            ts(
                p,
                "Jobs Over Time",
                [
                    (f'slurm_jobs_state{{control_node_group="{ng}",state="running"}}', "Running"),
                    (f'slurm_jobs_state{{control_node_group="{ng}",state="pending"}}', "Pending"),
                    (f'slurm_jobs_state{{control_node_group="{ng}",state="completing"}}', "Completing"),
                ],
                0,
                y,
            )
        )
        p += 1
        panels.append(
            ts(
                p,
                "Slurm Memory Allocated vs Total (MB)",
                [
                    (f'slurm_memory_total_mb{{control_node_group="{ng}"}}', "Total"),
                    (f'slurm_memory_allocated_mb{{control_node_group="{ng}"}}', "Allocated"),
                ],
                12,
                y,
                unit="decmbytes",
            )
        )
        p += 1
        y += 8

        # ---- Row 4: Scheduler health ----
        panels.append(row(p, "Scheduler Health", y))
        p += 1
        y += 1
        panels.append(
            ts(
                p,
                "Backfill Cycle Time (s)",
                [
                    (f'slurm_scheduler_backfill_cycle_last_seconds{{control_node_group="{ng}"}}', "Last"),
                    (f'slurm_scheduler_backfill_cycle_mean_seconds{{control_node_group="{ng}"}}', "Mean"),
                ],
                0,
                y,
                unit="s",
            )
        )
        p += 1
        panels.append(
            ts(
                p,
                "Active Scheduler Threads",
                [
                    (f'slurm_scheduler_threads_active{{control_node_group="{ng}"}}', "Threads"),
                ],
                12,
                y,
            )
        )
        p += 1
        y += 8
        panels.append(
            ts(
                p,
                "DBD Agent Queue Size",
                [
                    (f'slurm_scheduler_dbd_agent_queue_size{{control_node_group="{ng}"}}', "Queue Size"),
                ],
                0,
                y,
                w=24,
            )
        )
        p += 1
        y += 8

        # ---- Row 5: Head node system resources ----
        panels.append(row(p, "Head Node System Resources (node_exporter)", y))
        p += 1
        y += 1
        panels.append(
            ts(
                p,
                "Head Node Memory Usage %",
                [
                    (
                        f'100 * (1 - node_memory_MemAvailable_bytes{{control_node_group="{ng}"}}'
                        f' / node_memory_MemTotal_bytes{{control_node_group="{ng}"}})',
                        "{{hostname}} Memory %",
                    ),
                ],
                0,
                y,
                unit="percent",
            )
        )
        p += 1
        panels.append(
            ts(
                p,
                "Head Node Disk Usage %",
                [
                    (
                        f'100 * (1 - node_filesystem_avail_bytes{{control_node_group="{ng}",'
                        f'fstype!~"tmpfs|devtmpfs|squashfs"}}'
                        f' / node_filesystem_size_bytes{{control_node_group="{ng}",'
                        f'fstype!~"tmpfs|devtmpfs|squashfs"}})',
                        "{{hostname}} {{mountpoint}}",
                    ),
                ],
                12,
                y,
                unit="percent",
            )
        )
        p += 1
        y += 8
        panels.append(
            ts(
                p,
                "Head Node CPU Usage %",
                [
                    (
                        f'100 - (avg by (hostname) (irate(node_cpu_seconds_total{{'
                        f'control_node_group="{ng}",mode="idle"}}[5m])) * 100)',
                        "{{hostname}} CPU %",
                    ),
                ],
                0,
                y,
                w=24,
                unit="percent",
            )
        )
        p += 1
        y += 8

        # ---- Row 6: Node state table ----
        panels.append(row(p, "Node Details", y))
        p += 1
        y += 1
        panels.append(
            table(
                p,
                "Node State Table",
                f'slurm_node_info{{control_node_group="{ng}"}} == 1',
                {
                    "node": "Node",
                    "partition": "Partition",
                    "state": "State",
                    "cpus": "CPUs",
                    "memory": "Memory (MB)",
                    "reason": "Reason",
                },
                0,
                y,
                sort_col="Node",
            )
        )
        p += 1
        y += 12

        # ---- Row 7: Job queue table ----
        panels.append(row(p, "Job Queue", y))
        p += 1
        y += 1
        panels.append(
            table(
                p,
                "Active Job Queue",
                f'slurm_job_info{{control_node_group="{ng}"}} == 1',
                {
                    "job_id": "Job ID",
                    "name": "Name",
                    "user": "User",
                    "partition": "Partition",
                    "state": "State",
                    "cpus": "CPUs",
                    "time_limit": "Time Limit",
                    "reason": "Reason (if pending)",
                },
                0,
                y,
                sort_col="Job ID",
                desc=True,
            )
        )
        p += 1
        y += 12

        # ---- Row 8: Top CPU consumers (pie chart + table) ----
        panels.append(row(p, "Top CPU Consumers", y))
        p += 1
        y += 1

        # Pie chart: CPU share by user
        panels.append(
            {
                "id": p,
                "type": "piechart",
                "title": "CPU Distribution by User (Running Jobs)",
                "description": "Share of cluster CPUs consumed by each user's running jobs",
                "gridPos": {"h": 10, "w": 24, "x": 0, "y": y},
                "targets": [
                    {
                        "expr": f'sum by (user) (slurm_job_cpu_count{{control_node_group="{ng}",state="running"}})',
                        "legendFormat": "{{user}}",
                        "instant": True,
                    }
                ],
                "options": {
                    "pieType": "donut",
                    "displayLabels": ["name", "percent"],
                    "legend": {
                        "displayMode": "table",
                        "placement": "right",
                        "values": ["value", "percent"],
                    },
                    "tooltip": {"mode": "single"},
                },
                "fieldConfig": {"defaults": {"unit": "short"}},
            }
        )
        p += 1
        y += 10

        panels.append(
            table(
                p,
                "Top CPU Consumers — Job Detail (Running Jobs)",
                f'topk(20, slurm_job_cpu_count{{control_node_group="{ng}",state="running"}})',
                {"job_id": "Job ID", "name": "Job Name", "user": "User", "partition": "Partition", "Value": "CPUs"},
                0,
                y,
                sort_col="CPUs",
                desc=True,
            )
        )
        p += 1
        y += 12

        # ---- Row 9: Partition table ----
        panels.append(row(p, "Partitions", y))
        p += 1
        y += 1
        panels.append(
            table(
                p,
                "Partition Summary",
                f'slurm_partition_info{{control_node_group="{ng}"}} == 1',
                {
                    "partition": "Partition",
                    "state": "State",
                    "nodes": "Nodes",
                    "cpus_total": "CPUs Total",
                    "cpus_alloc": "CPUs Allocated",
                },
                0,
                y,
                h=8,
            )
        )
        p += 1
        y += 8

        # ---- Row 10: Recent jobs table ----
        panels.append(row(p, "Recent Completed Jobs (last 24h)", y))
        p += 1
        y += 1
        panels.append(
            table(
                p,
                "Recent Jobs",
                f'slurm_recent_job_info{{control_node_group="{ng}"}} == 1',
                {
                    "job_id": "Job ID",
                    "name": "Name",
                    "user": "User",
                    "state": "State",
                    "elapsed": "Elapsed",
                    "exit_code": "Exit Code",
                },
                0,
                y,
                sort_col="Job ID",
                desc=True,
            )
        )
        p += 1
        y += 12

        # ---- Row 11: Logs ----
        panels.append(row(p, "Logs", y))
        p += 1
        y += 1
        panels.append(
            {
                "id": p,
                "type": "logs",
                "title": "slurmctld Logs",
                "datasource": {"type": "loki", "uid": "loki"},
                "gridPos": {"h": 10, "w": 24, "x": 0, "y": y},
                "targets": [
                    {
                        "datasource": {"type": "loki", "uid": "loki"},
                        "expr": f'{{control_node_group="{ng}"}}',
                    }
                ],
                "options": {"showTime": True, "showLabels": True, "wrapLogMessage": True},
            }
        )

        return {
            "uid": uid,
            "title": f"Slurm Control Plane — {group_name}",
            "tags": ["control-plane", "slurm"],
            "timezone": "browser",
            "schemaVersion": 38,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "templating": {"list": []},
            "panels": panels,
        }

    def generate_k8s_dashboard(self, group_name: str) -> Dict[str, Any]:
        """Generate a comprehensive Grafana dashboard for a Kubernetes control node group."""
        safe_name = self._cng_safe_name(group_name)
        uid = f"cng_{safe_name}"
        ng = group_name

        def ts(panel_id, title, expr_list, x, y, w=12, h=8, unit="short"):
            targets = [{"expr": e, "legendFormat": lf} for e, lf in expr_list]
            return {
                "id": panel_id,
                "type": "timeseries",
                "title": title,
                "gridPos": {"h": h, "w": w, "x": x, "y": y},
                "targets": targets,
                "fieldConfig": {"defaults": {"unit": unit}},
            }

        def stat(panel_id, title, expr, x, y, w=6, h=4, unit="short", thresholds=None):
            p = {
                "id": panel_id,
                "type": "stat",
                "title": title,
                "gridPos": {"h": h, "w": w, "x": x, "y": y},
                "targets": [{"expr": expr, "instant": True}],
                "fieldConfig": {"defaults": {"unit": unit}},
                "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
            }
            if thresholds:
                p["fieldConfig"]["defaults"]["thresholds"] = thresholds
            return p

        def row(panel_id, title, y):
            return {"id": panel_id, "type": "row", "title": title, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}}

        def table(panel_id, title, expr, col_map, x, y, w=24, h=12, sort_col=None, desc=False):
            exclude = {
                "Time": True,
                "Value": True,
                "__name__": True,
                "job": True,
                "instance": True,
                "control_node_group": True,
            }
            sort_by = [{"displayName": sort_col, "desc": desc}] if sort_col else []
            return {
                "id": panel_id,
                "type": "table",
                "title": title,
                "gridPos": {"h": h, "w": w, "x": x, "y": y},
                "targets": [{"expr": expr, "instant": True, "legendFormat": "", "format": "table"}],
                "transformations": [
                    {"id": "merge", "options": {}},
                    {
                        "id": "organize",
                        "options": {
                            "excludeByName": exclude,
                            "renameByName": {k: v for k, v in col_map.items()},
                        },
                    },
                ],
                "options": {"sortBy": sort_by},
                "fieldConfig": {
                    "defaults": {
                        "custom": {
                            "displayMode": "auto",
                            "filterable": True,
                        }
                    }
                },
            }

        up_thresholds = {
            "mode": "absolute",
            "steps": [{"color": "red", "value": None}, {"color": "green", "value": 1}],
        }

        p = 1
        y = 0
        panels = []

        # Row 1: Health overview stats
        panels.append(row(p, "Control Plane Health", y))
        p += 1
        y += 1
        panels.append(
            stat(p, "API Server", f'k8s_apiserver_up{{control_node_group="{ng}"}}', 0, y, thresholds=up_thresholds)
        )
        p += 1
        panels.append(stat(p, "etcd", f'k8s_etcd_up{{control_node_group="{ng}"}}', 6, y, thresholds=up_thresholds))
        p += 1
        panels.append(
            stat(p, "Scheduler", f'k8s_scheduler_up{{control_node_group="{ng}"}}', 12, y, thresholds=up_thresholds)
        )
        p += 1
        panels.append(
            stat(
                p,
                "Controller Manager",
                f'k8s_controller_manager_up{{control_node_group="{ng}"}}',
                18,
                y,
                thresholds=up_thresholds,
            )
        )
        p += 1
        y += 4

        # Row 2: API Server
        panels.append(row(p, "API Server", y))
        p += 1
        y += 1
        panels.append(
            ts(
                p,
                "API Request Rate (by verb)",
                [
                    (f'sum by (verb) (k8s_apiserver_request_rate{{control_node_group="{ng}"}})', "{{verb}}"),
                ],
                0,
                y,
                unit="reqps",
            )
        )
        p += 1
        panels.append(
            ts(
                p,
                "API P99 Latency (s)",
                [
                    (f'k8s_apiserver_request_duration_p99_seconds{{control_node_group="{ng}"}}', "{{verb}}"),
                ],
                12,
                y,
                unit="s",
            )
        )
        p += 1
        y += 8
        panels.append(
            ts(
                p,
                "API Error Rate (5xx)",
                [
                    (
                        f'sum by (verb) (k8s_apiserver_request_rate{{control_node_group="{ng}",code=~"5.."}})',
                        "{{verb}}",
                    ),
                ],
                0,
                y,
                w=24,
            )
        )
        p += 1
        y += 8

        # Row 3: etcd
        panels.append(row(p, "etcd", y))
        p += 1
        y += 1
        panels.append(
            stat(
                p,
                "etcd Has Leader",
                f'k8s_etcd_has_leader{{control_node_group="{ng}"}}',
                0,
                y,
                w=4,
                thresholds=up_thresholds,
            )
        )
        p += 1
        panels.append(
            ts(
                p,
                "etcd Leader Changes",
                [
                    (f'k8s_etcd_leader_changes_total{{control_node_group="{ng}"}}', "Leader Changes"),
                ],
                4,
                y,
                w=10,
            )
        )
        p += 1
        panels.append(
            ts(
                p,
                "etcd WAL Fsync P99 (s)",
                [
                    (f'k8s_etcd_wal_fsync_duration_p99_seconds{{control_node_group="{ng}"}}', "P99 Fsync"),
                ],
                14,
                y,
                w=10,
                unit="s",
            )
        )
        p += 1
        y += 8
        panels.append(
            ts(
                p,
                "etcd Proposals Failed",
                [
                    (f'k8s_etcd_proposals_failed_total{{control_node_group="{ng}"}}', "Proposals Failed"),
                ],
                0,
                y,
                w=24,
            )
        )
        p += 1
        y += 8

        # Row 4: Scheduler
        panels.append(row(p, "Scheduler", y))
        p += 1
        y += 1
        panels.append(
            ts(
                p,
                "Pending Pods by Queue",
                [
                    (f'k8s_scheduler_pending_pods{{control_node_group="{ng}",queue="active"}}', "Active"),
                    (f'k8s_scheduler_pending_pods{{control_node_group="{ng}",queue="backoff"}}', "Backoff"),
                    (f'k8s_scheduler_pending_pods{{control_node_group="{ng}",queue="unschedulable"}}', "Unschedulable"),
                    (f'k8s_scheduler_pending_pods{{control_node_group="{ng}",queue="gated"}}', "Gated"),
                ],
                0,
                y,
            )
        )
        p += 1
        panels.append(
            ts(
                p,
                "Scheduling Attempt Failures",
                [
                    (f'k8s_scheduler_schedule_attempts_total{{control_node_group="{ng}",result="error"}}', "Error"),
                    (
                        f'k8s_scheduler_schedule_attempts_total{{control_node_group="{ng}",result="unschedulable"}}',
                        "Unschedulable",
                    ),
                ],
                12,
                y,
            )
        )
        p += 1
        y += 8

        # Row 5: Controller Manager
        panels.append(row(p, "Controller Manager", y))
        p += 1
        y += 1
        panels.append(
            ts(
                p,
                "Work Queue Depth",
                [
                    (f'k8s_controller_manager_workqueue_depth{{control_node_group="{ng}"}}', "{{queue_name}}"),
                ],
                0,
                y,
                w=24,
            )
        )
        p += 1
        y += 8

        # Row 6: System (node_exporter)
        panels.append(row(p, "Control Plane Node System", y))
        p += 1
        y += 1
        panels.append(
            ts(
                p,
                "CPU Usage %",
                [
                    (
                        f'100 - (avg by (hostname) (irate(node_cpu_seconds_total{{control_node_group="{ng}",mode="idle"}}[5m])) * 100)',
                        "{{hostname}} CPU",
                    ),
                ],
                0,
                y,
                unit="percent",
            )
        )
        p += 1
        panels.append(
            ts(
                p,
                "Memory Usage %",
                [
                    (
                        f'100 * (1 - node_memory_MemAvailable_bytes{{control_node_group="{ng}"}} / node_memory_MemTotal_bytes{{control_node_group="{ng}"}})',
                        "{{hostname}} Memory",
                    ),
                ],
                12,
                y,
                unit="percent",
            )
        )
        p += 1
        y += 8

        # Row 7: Node table
        panels.append(row(p, "Cluster Nodes", y))
        p += 1
        y += 1
        panels.append(
            table(
                p,
                "Node Status",
                f'k8s_node_info{{control_node_group="{ng}"}} == 1',
                {"node": "Node", "role": "Role", "status": "Status", "version": "K8s Version"},
                0,
                y,
                h=6,
            )
        )
        p += 1
        y += 6

        # Row 8: Pending/Failed pods
        panels.append(row(p, "Pending & Failed Pods", y))
        p += 1
        y += 1
        panels.append(
            table(
                p,
                "Pending / Failed Pods",
                f'k8s_pod_info{{control_node_group="{ng}"}} == 1',
                {"pod": "Pod", "namespace": "Namespace", "phase": "Phase", "reason": "Reason", "node": "Node"},
                0,
                y,
            )
        )
        p += 1
        y += 8

        # Row 9: Component health table
        panels.append(row(p, "Component Health", y))
        p += 1
        y += 1
        panels.append(
            table(
                p,
                "Component Health Check",
                f'k8s_component_health{{control_node_group="{ng}"}}',
                {"component": "Component", "endpoint": "Endpoint"},
                0,
                y,
                h=6,
            )
        )
        p += 1
        y += 6

        # Row 10: Logs
        panels.append(row(p, "Logs", y))
        p += 1
        y += 1
        panels.append(
            {
                "id": p,
                "type": "logs",
                "title": "Control Plane Logs",
                "datasource": {"type": "loki", "uid": "loki"},
                "gridPos": {"h": 10, "w": 24, "x": 0, "y": y},
                "targets": [
                    {
                        "datasource": {"type": "loki", "uid": "loki"},
                        "expr": f'{{control_node_group="{ng}"}}',
                    }
                ],
                "options": {"showTime": True, "showLabels": True, "wrapLogMessage": True},
            }
        )

        return {
            "uid": uid,
            "title": f"Kubernetes Control Plane — {group_name}",
            "tags": ["control-plane", "kubernetes"],
            "timezone": "browser",
            "schemaVersion": 38,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "templating": {"list": []},
            "panels": panels,
        }

    async def provision_control_node_group_dashboard(
        self,
        group_name: str,
        control_type: str,
    ) -> Optional[str]:
        """Create or update the Grafana dashboard for a control node group.

        Returns the dashboard UID on success, None on failure.
        """
        folder_uid = "control-plane"
        await self.get_or_create_folder("Control Plane Monitoring", folder_uid)

        if control_type == "slurm":
            dashboard = self.generate_slurm_dashboard(group_name)
        else:
            dashboard = self.generate_k8s_dashboard(group_name)

        result = await self.create_dashboard(dashboard, folder_uid=folder_uid, overwrite=True)
        if "error" in result:
            logger.error(f"Failed to provision control node group dashboard for '{group_name}': {result.get('error')}")
            return None

        uid = dashboard["uid"]
        logger.info(f"Provisioned control node group dashboard: {dashboard['title']} (uid={uid})")
        return uid

    async def remove_control_node_group_dashboard(self, group_name: str) -> bool:
        """Delete the Grafana dashboard for a control node group."""
        safe_name = self._cng_safe_name(group_name)
        uid = f"cng_{safe_name}"
        success = await self.delete_dashboard(uid)
        if success:
            logger.info(f"Removed control node group dashboard: {uid}")
        return success
