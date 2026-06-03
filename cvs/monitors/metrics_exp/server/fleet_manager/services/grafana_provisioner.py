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
                        "title": "MMA Activity",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "gpu_mm_activity{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
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
                            "query": "label_values(gpu_temperature_celsius, node_group)",
                            "refresh": 2,
                            "multi": True,
                            "includeAll": True,
                        },
                        {
                            "name": "hostname",
                            "type": "query",
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "query": "label_values(gpu_temperature_celsius{node_group=~\"$nodegroup\"}, hostname)",
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
                                "expr": "gpu_temperature_celsius{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
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
                                "expr": "gpu_power_watts{node_group=~\"$nodegroup\", hostname=~\"$hostname\"}",
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
                                "expr": "sum(gpu_ecc_correctable_total{node_group=~\"$nodegroup\"}) or vector(0)",
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
                                "expr": "sum(gpu_ecc_uncorrectable_total{node_group=~\"$nodegroup\"}) or vector(0)",
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
                        "description": "Congestion Notification Packets sent",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 15},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_np_cnp_sent{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} np_cnp_sent",
                            },
                            {
                                "expr": "rate(rdma_stat_cnp_sent{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} cnp_sent",
                            },
                        ],
                        "fieldConfig": {"defaults": {"unit": "pps"}},
                    },
                    {
                        "id": 52,
                        "type": "timeseries",
                        "title": "CNP Packets Received",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 15},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_rp_cnp_handled{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} rp_handled",
                            },
                            {
                                "expr": "rate(rdma_stat_cnp_received{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} cnp_rcvd",
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
                        "title": "PFC TX Pause Frames",
                        "description": "PFC pause frames transmitted",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_tx_pause{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} tx_pause",
                            },
                            {
                                "expr": "rate(rdma_stat_tx_pfc_frames_prio3{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} tx_pfc_prio3",
                            },
                        ],
                        "fieldConfig": {"defaults": {"unit": "pps"}},
                    },
                    {
                        "id": 62,
                        "type": "timeseries",
                        "title": "PFC RX Pause Frames",
                        "description": "PFC pause frames received",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
                        "targets": [
                            {
                                "expr": "rate(rdma_stat_rx_pause{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} rx_pause",
                            },
                            {
                                "expr": "rate(rdma_stat_rx_pfc_frames_prio3{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} rx_pfc_prio3",
                            },
                        ],
                        "fieldConfig": {"defaults": {"unit": "pps"}},
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
                                "expr": "rate(rdma_stat_packet_seq_err{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} seq_err",
                            },
                            {
                                "expr": "rate(rdma_stat_out_of_sequence{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} out_of_seq",
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
                                "expr": "rate(rdma_stat_rx_icrc_errors{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} icrc_err",
                            },
                            {
                                "expr": "rate(rdma_stat_rnr_nak_retry_err{node_group=~\"$nodegroup\", hostname=~\"$hostname\", device=~\"$device\"}[1m])",
                                "legendFormat": "{{hostname}} {{device}} rnr_nak",
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
