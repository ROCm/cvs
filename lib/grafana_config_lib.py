"""
Grafana configuration and provisioning library
"""
import os
import json
import logging

log = logging.getLogger(__name__)


def setup_grafana_provisioning(monitoring_dir="/tmp/grafana_provisioning"):
    """
    Setup Grafana provisioning configs for datasources and dashboards
    """
    os.makedirs(f"{monitoring_dir}/datasources", exist_ok=True)
    os.makedirs(f"{monitoring_dir}/dashboards", exist_ok=True)
    os.makedirs(f"{monitoring_dir}/dashboard_files", exist_ok=True)
    
    # Datasource config
    datasource_config = """apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: false
    jsonData:
      timeInterval: "5s"
"""
    
    with open(f"{monitoring_dir}/datasources/prometheus.yml", 'w') as f:
        f.write(datasource_config)
    
    # Dashboard provisioning config
    dashboard_config = """apiVersion: 1

providers:
  - name: 'Default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
"""
    
    with open(f"{monitoring_dir}/dashboards/default.yml", 'w') as f:
        f.write(dashboard_config)
    
    log.info(f"Grafana provisioning configs created in {monitoring_dir}")
    return monitoring_dir


def create_gpu_dashboard(output_file="/tmp/grafana_provisioning/dashboard_files/gpu-metrics.json"):
    """
    Create GPU metrics dashboard JSON
    """
    dashboard = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": [
            {
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "gradientMode": "none",
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "showPoints": "never"
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 85}
                            ]
                        },
                        "unit": "celsius"
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "id": 1,
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi"}
                },
                "targets": [
                    {
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "expr": "gpu_temp_degrees",
                        "legendFormat": "{{instance}} - GPU {{gpu_index}}",
                        "refId": "A"
                    }
                ],
                "title": "GPU Temperature",
                "type": "timeseries"
            },
            {
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisPlacement": "auto",
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5
                        },
                        "mappings": [],
                        "max": 100,
                        "min": 0,
                        "unit": "percent"
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "id": 2,
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi"}
                },
                "targets": [
                    {
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "expr": "gpu_utilization_percent",
                        "legendFormat": "{{instance}} - GPU {{gpu_index}}",
                        "refId": "A"
                    }
                ],
                "title": "GPU Utilization %",
                "type": "timeseries"
            },
            {
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisPlacement": "auto",
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "lineInterpolation": "linear",
                            "lineWidth": 1
                        },
                        "mappings": [],
                        "unit": "bytes"
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "id": 3,
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi"}
                },
                "targets": [
                    {
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "expr": "gpu_memory_used_bytes",
                        "legendFormat": "{{instance}} - GPU {{gpu_index}} Used",
                        "refId": "A"
                    },
                    {
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "expr": "gpu_memory_total_bytes",
                        "legendFormat": "{{instance}} - GPU {{gpu_index}} Total",
                        "refId": "B"
                    }
                ],
                "title": "GPU Memory Usage",
                "type": "timeseries"
            },
            {
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisPlacement": "auto",
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "lineInterpolation": "linear",
                            "lineWidth": 1
                        },
                        "mappings": [],
                        "unit": "watt"
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "id": 4,
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi"}
                },
                "targets": [
                    {
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "expr": "gpu_power_watts",
                        "legendFormat": "{{instance}} - GPU {{gpu_index}}",
                        "refId": "A"
                    }
                ],
                "title": "GPU Power Consumption",
                "type": "timeseries"
            }
        ],
        "refresh": "5s",
        "schemaVersion": 38,
        "style": "dark",
        "tags": ["gpu", "amd", "rocm"],
        "templating": {"list": []},
        "time": {"from": "now-15m", "to": "now"},
        "timepicker": {},
        "timezone": "",
        "title": "AMD GPU Metrics Dashboard",
        "uid": "amd-gpu-metrics",
        "version": 1
    }
    
    with open(output_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    log.info(f"GPU dashboard created: {output_file}")
    return output_file
