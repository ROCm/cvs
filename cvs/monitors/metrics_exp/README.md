# GPU Fleet Monitor

A comprehensive, web-based monitoring solution for AMD GPU clusters at scale. Deploy monitoring infrastructure, manage GPU nodes, and visualize metrics through an intuitive UI with pre-built Grafana dashboards.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
- [Quick Start](#quick-start)
- [Web UI Guide](#web-ui-guide)
- [Collected Metrics](#collected-metrics)
- [Pre-built Dashboards](#pre-built-dashboards)
- [Configuration](#configuration)
- [Debugging Guide](#debugging-guide)
- [SSH Tunnel for Blocked Ports](#ssh-tunnel-for-blocked-ports)
- [API Reference](#api-reference)
- [Requirements](#requirements)

## Overview

GPU Fleet Monitor provides end-to-end monitoring for AMD GPU clusters with:

- **Web-based Management UI**: Configure monitoring servers, node groups, and metric collection
- **Automatic Exporter Installation**: Deploy AMD GPU, Node, RDMA, and log exporters via SSH
- **Pre-built Grafana Dashboards**: GPU utilization, thermal/power, health, RDMA network, and more
- **Log Aggregation**: Capture dmesg/journalctl for GPU errors, ECC events, and critical failures
- **Scalable Design**: Built for 1000+ nodes with configurable retention

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CORPORATE NETWORK                                   │
│                                                                                  │
│  ┌──────────────────────┐                                                       │
│  │  Web Application     │     SSH (via Jump Host)                               │
│  │  Server              │─────────────────────────────────┐                     │
│  │  (xsjvenksrin50)     │                                 │                     │
│  │                      │                                 ▼                     │
│  │  ┌────────────────┐  │                    ┌─────────────────────┐            │
│  │  │ Fleet Monitor  │  │                    │    Jump Host        │            │
│  │  │ Web UI :30080  │  │                    │  (if required)      │            │
│  │  │                │  │                    └──────────┬──────────┘            │
│  │  │ - Node Groups  │  │                               │                       │
│  │  │ - Mon. Servers │  │                               │ SSH                   │
│  │  │ - Metrics Cfg  │  │                               │                       │
│  │  └────────────────┘  │                               ▼                       │
│  └──────────────────────┘              ┌─────────────────────────────────────┐  │
│                                        │         Monitoring Server            │  │
│                                        │                                      │  │
│                                        │                                      │  │
│                                        │  ┌───────────┐ ┌───────────┐        │  │
│                                        │  │Prometheus │ │  Grafana  │        │  │
│                                        │  │  :30090   │ │  :30030   │        │  │
│                                        │  └───────────┘ └───────────┘        │  │
│                                        │  ┌───────────┐                      │  │
│                                        │  │   Loki    │                      │  │
│                                        │  │  :30100   │                      │  │
│                                        │  └───────────┘                      │  │
│                                        └──────────────┬──────────────────────┘  │
│                                                       │                         │
│                            HTTP Scrape (Prometheus) ──┼── Log Push (Promtail)   │
│                                                       │                         │
│         ┌─────────────────────────────────────────────┼─────────────────────┐   │
│         │                                             │                     │   │
│         ▼                                             ▼                     ▼   │
│  ┌─────────────────┐                         ┌─────────────────┐    ┌──────────┐│
│  │   GPU Node 1    │                         │   GPU Node 2    │    │  GPU     ││
│  │                 │                         │                 │    │  Node N  ││
│  │ AMD Exporter    │                         │ AMD Exporter    │    │          ││
│  │   :5000         │                         │   :5000         │    │  ...     ││
│  │ Node Exporter   │                         │ Node Exporter   │    │          ││
│  │   :9100         │                         │   :9100         │    │          ││
│  │ RDMA Exporter   │                         │ RDMA Exporter   │    │          ││
│  │   :9417         │                         │   :9417         │    │          ││
│  │ Promtail        │                         │ Promtail        │    │          ││
│  └─────────────────┘                         └─────────────────┘    └──────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Web Application Server

**Role**: Hosts the GPU Fleet Monitor web UI and orchestrates deployments.

**Services**:
- **Fleet Monitor UI** (port 30080): Web interface for managing monitoring infrastructure
- **PostgreSQL**: Stores configuration for node groups, monitoring servers, and nodes

**Responsibilities**:
- Manage monitoring server configurations
- Define and organize node groups
- Install/uninstall exporters on GPU nodes via SSH
- Sync Prometheus targets to monitoring servers

### 2. Monitoring Server

**Role**: Collects, stores, and visualizes metrics from GPU nodes.

**Services**:
- **Prometheus** (port 30090): Time-series database for metrics
- **Grafana** (port 30030): Visualization and dashboards
- **Loki** (port 30100): Log aggregation

**Responsibilities**:
- Scrape metrics from GPU nodes (GPU, Node, RDMA exporters)
- Store metrics with configurable retention
- Provide pre-built dashboards for GPU monitoring
- Aggregate logs from GPU nodes

### 3. GPU Nodes

**Role**: Run GPU workloads and expose metrics for collection.

**Installed Exporters**:
- **AMD Device Metrics Exporter** (port 5000): GPU metrics (utilization, temperature, power, ECC, etc.)
- **Node Exporter** (port 9100): System metrics (CPU, memory, disk, network)
- **RDMA Exporter** (port 9417): RDMA/RoCE network metrics (link status, traffic, congestion, errors)
- **Promtail**: Log collector for dmesg/journalctl

### 4. Jump Host (Optional)

**Role**: Provides SSH access to GPU nodes in isolated networks.

**Use Case**: When GPU nodes are not directly accessible from the web application server, configure a jump host for SSH tunneling.

## Quick Start

### 1. Deploy Web Application Server

```bash
git clone <repo-url>
cd metrics_proj

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Build and start
docker compose build --no-cache
docker compose up -d
```

### 2. Access the Web UI

Open http://your-server:30080 in your browser.

### 3. Configure Monitoring Server

1. Go to **Monitoring Servers** → **Add Server**
2. Enter the monitoring server details (IP, ports, SSH credentials)
3. Click **Install Stack** to deploy Prometheus, Grafana, and Loki

### 4. Add GPU Nodes

1. Go to **Node Groups** → **Add Node Group**
2. Enter node IPs and associate with a monitoring server
3. Upload SSH key for authentication
4. Click **Install** to deploy exporters on all nodes

### 5. View Dashboards

Access Grafana at http://monitoring-server:30030 (default: admin/admin)

## Web UI Guide

### Dashboard

The main dashboard provides an overview of:
- Total GPU nodes and their status
- Active monitoring servers
- Node group summary
- Recent activity

### Monitoring Servers

Configure where metrics are stored and visualized.

| Field | Description |
|-------|-------------|
| Name | Friendly name for the monitoring server |
| Server IP | IP address of the monitoring server |
| Prometheus Port | Port for Prometheus (default: 30090) |
| Grafana Port | Port for Grafana (default: 30030) |
| Loki Port | Port for Loki (default: 30100) |
| SSH User | Username for SSH access |
| SSH Auth Type | Key or password authentication |
| Use Jump Host | Enable if monitoring server requires jump host access |

**Actions**:
- **Test Connection**: Verify connectivity to monitoring services
- **Check Services**: See what's installed on the server
- **Install Stack**: Deploy Prometheus, Grafana, Loki with all dashboards
- **Sync Targets**: Update Prometheus targets for all associated node groups

### Metric Groups

Define custom metric collection configurations (optional).

| Field | Description |
|-------|-------------|
| Name | Configuration name |
| Scrape Interval | How often to collect metrics |
| Enabled Metrics | Which metric types to collect |

### Node Groups

Organize GPU nodes into logical groups.

| Field | Description |
|-------|-------------|
| Name | Group name (e.g., "cluster-a", "training-nodes") |
| Monitoring Server | Which monitoring server collects metrics |
| SSH User | Username for node access |
| SSH Auth Type | Key or password authentication |
| Node IPs | List of GPU node IP addresses |

**Actions**:
- **Install**: Deploy all exporters (AMD GPU, Node, RDMA, Promtail)
- **Uninstall**: Remove all exporters from nodes
- **Health Check**: Verify exporter status on all nodes
- **Sync Targets**: Update Prometheus targets

## Collected Metrics

### GPU Metrics (AMD Device Metrics Exporter - Port 5000)

| Category | Metrics | Description |
|----------|---------|-------------|
| Utilization | `gpu_gfx_activity`, `gpu_umc_activity`, `gpu_mm_activity` | GPU compute, memory controller, multimedia activity |
| Temperature | `gpu_temperature_celsius`, `gpu_junction_temperature`, `gpu_memory_temperature` | Various temperature sensors |
| Power | `gpu_power_watts`, `gpu_package_power` | Power consumption |
| Memory | `gpu_total_vram`, `gpu_used_vram`, `gpu_free_vram` | VRAM usage |
| PCIe | `pcie_speed`, `pcie_bandwidth`, `pcie_replay_count` | PCIe link health |
| ECC | `gpu_ecc_correctable_total`, `gpu_ecc_uncorrectable_total` | Error correction counts |
| Health | `gpu_health` | Overall GPU health status |

### System Metrics (Node Exporter - Port 9100)

- CPU usage, load average, and frequency
- Memory utilization and swap
- Disk I/O and space usage
- Network interface statistics

### RDMA Network Metrics (RDMA Exporter - Port 9417)

| Category | Metrics | Description |
|----------|---------|-------------|
| Link Status | `rdma_link_state`, `rdma_link_physical_state` | Port up/down status |
| Traffic | `rdma_stat_tx_bytes`, `rdma_stat_rx_bytes` | Bytes transmitted/received |
| Congestion (DCQCN) | `rdma_stat_np_cnp_sent`, `rdma_stat_rp_cnp_handled` | Congestion notification packets |
| Congestion (PFC) | `rdma_stat_tx_pause`, `rdma_stat_rx_pause` | Priority flow control frames |
| Errors | `rdma_stat_packet_seq_err`, `rdma_stat_rx_icrc_errors` | Packet sequence and CRC errors |
| Resources | `rdma_resource_qp_total`, `rdma_resource_cq_total` | Queue pairs, completion queues |

### Log Collection (Promtail → Loki)

- Kernel messages (dmesg)
- Systemd journal entries
- Critical patterns: ECC errors, RAS events, GPU hangs, thermal throttling

## Pre-built Dashboards

1. **GPU Fleet Overview**: High-level fleet status, total GPUs, average utilization
2. **GPU Utilization**: Detailed compute and memory activity over time
3. **Thermal & Power**: Temperature trends and power consumption
4. **GPU Health & Errors**: ECC/RAS monitoring with error counts
5. **CPU & System**: Host CPU, memory, and system metrics
6. **RDMA Network**: Link status, traffic, congestion (PFC/DCQCN), errors
7. **Logs Analysis**: Critical log pattern detection from dmesg/journalctl

## Configuration

### Environment Variables (.env)

```bash
# PostgreSQL
POSTGRES_PASSWORD=your_secure_password

# Fleet Monitor
SECRET_KEY=your-secret-key-here

# Default ports (can be customized per monitoring server)
PROMETHEUS_PORT=30090
GRAFANA_PORT=30030
LOKI_PORT=30100
```

### Scaling for Large Deployments

For 1000+ nodes:

1. **Increase scrape interval**: 30s → 60s for reduced load
2. **Configure retention**: Adjust based on storage capacity
3. **Enable Loki S3 storage**: For long-term log persistence
4. **Horizontal scaling**: Multiple monitoring servers for different node groups

## Debugging Guide

### Web Application Server (Fleet Monitor)

#### Check Container Status

```bash
# List all containers
docker ps -a | grep fleet
```

**Expected output:**
```
CONTAINER ID   IMAGE                    STATUS          NAMES
a1b2c3d4e5f6   fleet-manager:latest     Up 2 hours      fleet-manager
b2c3d4e5f6a7   postgres:15-alpine       Up 2 hours      fleet-postgres
```

```bash
# Check fleet-manager logs
docker logs fleet-manager --tail 100

# Follow logs in real-time
docker logs -f fleet-manager
```

**Expected output (healthy):**
```
INFO:     Started server process [1]
INFO:     Uvicorn running on http://0.0.0.0:8080
INFO:     Application startup complete.
INFO:     10.x.x.x:54321 - "GET /health HTTP/1.1" 200 OK
INFO:     10.x.x.x:54322 - "GET /api/v1/nodegroups HTTP/1.1" 200 OK
```

#### Check Application Health

```bash
# Health endpoint
curl http://localhost:30080/health
```

**Expected output:**
```json
{"status":"healthy","version":"1.0.0","database":"connected"}
```

#### Check Database

```bash
# Connect to PostgreSQL
docker exec -it fleet-postgres psql -U fleet -d fleet_manager

# List tables
\dt
```

**Expected output:**
```
              List of relations
 Schema |       Name        | Type  | Owner
--------+-------------------+-------+-------
 public | monitoring_servers| table | fleet
 public | node_groups       | table | fleet
 public | nodes             | table | fleet
```

### GPU Nodes

#### Check AMD GPU Exporter

```bash
# Container status
docker ps | grep device-metrics-exporter
```

**Expected output:**
```
CONTAINER ID   IMAGE                                    STATUS         NAMES
c3d4e5f6a7b8   rocm/device-metrics-exporter:v1.5.0     Up 3 hours     device-metrics-exporter
```

```bash
# Verify metrics endpoint
curl -s http://localhost:5000/metrics | head -20
```

**Expected output:**
```
# HELP gpu_gfx_activity GPU graphics activity percentage
# TYPE gpu_gfx_activity gauge
gpu_gfx_activity{gpu="0",hostname="gpu-node-01"} 45.2
gpu_gfx_activity{gpu="1",hostname="gpu-node-01"} 78.5
# HELP gpu_temperature_celsius GPU temperature in Celsius
# TYPE gpu_temperature_celsius gauge
gpu_temperature_celsius{gpu="0",hostname="gpu-node-01",sensor="edge"} 52
gpu_temperature_celsius{gpu="1",hostname="gpu-node-01",sensor="edge"} 58
```

```bash
# Check for specific GPU metrics
curl -s http://localhost:5000/metrics | grep gpu_gfx_activity
```

**Expected output:**
```
gpu_gfx_activity{gpu="0",hostname="gpu-node-01"} 45.2
gpu_gfx_activity{gpu="1",hostname="gpu-node-01"} 78.5
gpu_gfx_activity{gpu="2",hostname="gpu-node-01"} 92.1
gpu_gfx_activity{gpu="3",hostname="gpu-node-01"} 88.7
```

#### Check Node Exporter

```bash
# Service status
systemctl status node_exporter
```

**Expected output:**
```
● node_exporter.service - Prometheus Node Exporter
     Loaded: loaded (/etc/systemd/system/node_exporter.service; enabled)
     Active: active (running) since Mon 2026-06-02 10:00:00 UTC; 5h ago
   Main PID: 12345 (node_exporter)
      Tasks: 5 (limit: 629145)
     Memory: 15.2M
```

```bash
# Verify metrics endpoint
curl -s http://localhost:9100/metrics | grep node_cpu_seconds
```

**Expected output:**
```
# HELP node_cpu_seconds_total Seconds the CPUs spent in each mode.
# TYPE node_cpu_seconds_total counter
node_cpu_seconds_total{cpu="0",mode="idle"} 1.52345678e+06
node_cpu_seconds_total{cpu="0",mode="system"} 12345.67
node_cpu_seconds_total{cpu="0",mode="user"} 45678.90
```

#### Check RDMA Exporter

```bash
# Service status
systemctl status rdma-exporter
```

**Expected output:**
```
● rdma-exporter.service - RDMA Metrics Exporter for Prometheus
     Loaded: loaded (/etc/systemd/system/rdma-exporter.service; enabled)
     Active: active (running) since Mon 2026-06-02 10:00:00 UTC; 5h ago
   Main PID: 12346 (python3)
      Tasks: 2 (limit: 629145)
     Memory: 10.8M
```

```bash
# Verify metrics endpoint
curl -s http://localhost:9417/metrics | head -20
```

**Expected output:**
```
rdma_link_physical_state{hostname="gpu-node-01",device="mlx5_0",port="1",netdev="enp1s0"} 1
rdma_link_state{hostname="gpu-node-01",device="mlx5_0",port="1",netdev="enp1s0"} 1
rdma_link_info{hostname="gpu-node-01",device="mlx5_0",port="1",netdev="enp1s0",link_layer="Ethernet"} 1
rdma_stat_tx_bytes{hostname="gpu-node-01",device="mlx5_0",port="1"} 1234567890
rdma_stat_rx_bytes{hostname="gpu-node-01",device="mlx5_0",port="1"} 9876543210
rdma_stat_tx_packets{hostname="gpu-node-01",device="mlx5_0",port="1"} 12345678
rdma_stat_rx_packets{hostname="gpu-node-01",device="mlx5_0",port="1"} 87654321
```

```bash
# Check for RDMA devices
rdma link
```

**Expected output:**
```
link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev enp1s0
link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev enp2s0
```

#### Check Promtail

```bash
# Service status
systemctl status promtail
```

**Expected output:**
```
● promtail.service - Promtail Log Collector
     Loaded: loaded (/etc/systemd/system/promtail.service; enabled)
     Active: active (running) since Mon 2026-06-02 10:00:00 UTC; 5h ago
   Main PID: 12347 (promtail)
      Tasks: 7 (limit: 629145)
     Memory: 25.4M
```

### Monitoring Server

#### Check Container Status

```bash
# List monitoring containers
docker ps | grep fleet-
```

**Expected output:**
```
CONTAINER ID   IMAGE                      STATUS         NAMES
d4e5f6a7b8c9   prom/prometheus:v2.51.0    Up 4 hours     fleet-prometheus
e5f6a7b8c9d0   grafana/grafana:10.4.1     Up 4 hours     fleet-grafana
f6a7b8c9d0e1   grafana/loki:2.9.5         Up 4 hours     fleet-loki
```

#### Check Prometheus

```bash
# Health check
curl http://localhost:30090/-/healthy
```

**Expected output:**
```
Prometheus Server is Healthy.
```

```bash
# Check targets
curl -s "http://localhost:30090/api/v1/targets" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for t in data.get('data', {}).get('activeTargets', []):
    print(f\"{t['labels'].get('job')}: {t['labels'].get('instance')} - {t['health']}\")"
```

**Expected output (all healthy):**
```
prometheus: localhost:30090 - up
amd_gpu_metrics: 10.x.x.101:5000 - up
amd_gpu_metrics: 10.x.x.102:5000 - up
node_exporter: 10.x.x.101:9100 - up
node_exporter: 10.x.x.102:9100 - up
rdma_metrics: 10.x.x.101:9417 - up
rdma_metrics: 10.x.x.102:9417 - up
```

**Troubleshooting - if targets show "down":**
```
amd_gpu_metrics: 10.x.x.103:5000 - down
```
This indicates the exporter is not running or not reachable. Check firewall and exporter status on that node.

```bash
# Query GPU metrics
curl -s "http://localhost:30090/api/v1/query?query=gpu_gfx_activity" | python3 -c "
import json, sys
data = json.load(sys.stdin)
results = data.get('data', {}).get('result', [])
print(f'Found {len(results)} GPU metrics')
for r in results[:5]:
    labels = r.get('metric', {})
    value = r.get('value', [0, '0'])[1]
    print(f\"  {labels.get('hostname')} GPU{labels.get('gpu')}: {value}%\")"
```

**Expected output:**
```
Found 16 GPU metrics
  gpu-node-01 GPU0: 45.2%
  gpu-node-01 GPU1: 78.5%
  gpu-node-02 GPU0: 92.1%
  gpu-node-02 GPU1: 88.7%
  gpu-node-03 GPU0: 12.3%
```

```bash
# Check RDMA metrics exist
curl -s "http://localhost:30090/api/v1/label/__name__/values" | grep -i rdma
```

**Expected output:**
```
rdma_link_info
rdma_link_physical_state
rdma_link_state
rdma_resource_cq_total
rdma_resource_mr_total
rdma_resource_qp_total
rdma_scrape_success
rdma_stat_rx_bytes
rdma_stat_tx_bytes
...
```

**If no output:** RDMA targets may not be configured. Run "Sync Targets" from the UI.

#### Check Grafana

```bash
# Health check
curl http://localhost:30030/api/health
```

**Expected output:**
```json
{"commit":"abc123","database":"ok","version":"10.4.1"}
```

```bash
# List dashboards
curl -s -u admin:admin "http://localhost:30030/api/search?type=dash-db" | python3 -c "
import json, sys
for d in json.load(sys.stdin):
    print(f\"{d['title']} (uid: {d['uid']})\")"
```

**Expected output:**
```
CPU & System (uid: cpu-system)
GPU Fleet Overview (uid: fleet-overview)
GPU Health & Errors (uid: gpu-health)
GPU Utilization (uid: gpu-utilization)
Logs Analysis (uid: logs-analysis)
RDMA Network (uid: rdma-network)
Thermal & Power (uid: thermal-power)
```

```bash
# Check Grafana logs for errors
docker logs fleet-grafana --tail 50 | grep -i error
```

**Expected output (no errors):**
```
(no output means no errors)
```

**Problematic output (dashboard provisioning issue):**
```
level=warn msg="the same UID is used more than once" uid=fleet-overview
level=warn msg="dashboards provisioning provider has no database write permissions"
```
**Fix:** Remove duplicate dashboard files and restart Grafana.

#### Check Loki

```bash
# Health check
curl http://localhost:30100/ready
```

**Expected output:**
```
ready
```

#### Check Prometheus Targets Directory

```bash
# List target files
ls -la ~/fleet-monitoring/prometheus/targets/
```

**Expected output:**
```
total 24
drwxrwxr-x 2 user user 4096 Jun  2 10:00 .
drwxrwxr-x 3 user user 4096 Jun  2 10:00 ..
-rw-rw-r-- 1 user user  405 Jun  2 10:00 gpu_cluster-a.json
-rw-rw-r-- 1 user user  333 Jun  2 10:00 node_cluster-a.json
-rw-rw-r-- 1 user user  333 Jun  2 10:00 rdma_cluster-a.json
```

```bash
# View GPU targets
cat ~/fleet-monitoring/prometheus/targets/gpu_*.json
```

**Expected output:**
```json
[
  {
    "targets": ["10.x.x.101:5000"],
    "labels": {
      "job": "amd_gpu_metrics",
      "node_group": "cluster-a",
      "hostname": "gpu-node-01"
    }
  },
  {
    "targets": ["10.x.x.102:5000"],
    "labels": {
      "job": "amd_gpu_metrics",
      "node_group": "cluster-a",
      "hostname": "gpu-node-02"
    }
  }
]
```

```bash
# View RDMA targets
cat ~/fleet-monitoring/prometheus/targets/rdma_*.json
```

**Expected output:**
```json
[
  {
    "targets": ["10.x.x.101:9417"],
    "labels": {
      "job": "rdma_metrics",
      "node_group": "cluster-a",
      "hostname": "gpu-node-01"
    }
  }
]
```

**If RDMA targets file is missing:** Click "Sync Targets" in the Fleet Monitor UI for the monitoring server.

## SSH Tunnel for Blocked Ports

If ports are blocked by firewall, use SSH tunneling to access services via localhost.

### Tunnel to Monitoring Server

```bash
# Single tunnel for Grafana
ssh -L 3030:localhost:30030 user@monitoring-server

# Multiple tunnels (Prometheus, Grafana, Loki)
ssh -L 9090:localhost:30090 \
    -L 3030:localhost:30030 \
    -L 3100:localhost:30100 \
    user@monitoring-server
```

Then access:
- Grafana: http://localhost:3030
- Prometheus: http://localhost:9090
- Loki: http://localhost:3100

### Tunnel via Jump Host

```bash
# If monitoring server requires jump host
ssh -J user@jump-host \
    -L 3030:localhost:30030 \
    -L 9090:localhost:30090 \
    user@monitoring-server
```

### Tunnel to Web Application Server

```bash
# Access Fleet Monitor UI
ssh -L 8080:localhost:30080 user@web-app-server
```

Then access Fleet Monitor at http://localhost:8080

### Persistent Tunnel with autossh

```bash
# Install autossh
sudo apt install autossh

# Create persistent tunnel that auto-reconnects
autossh -M 0 -f -N \
    -L 3030:localhost:30030 \
    -L 9090:localhost:30090 \
    -o "ServerAliveInterval 30" \
    -o "ServerAliveCountMax 3" \
    user@monitoring-server
```

## API Reference

The Fleet Monitor API is available at `http://localhost:30080/docs` (Swagger UI).

### Key Endpoints

#### Node Groups
- `GET /api/v1/nodegroups` - List all node groups
- `POST /api/v1/nodegroups` - Create a node group
- `POST /api/v1/nodegroups/{id}/ssh-key` - Upload SSH key
- `POST /api/v1/nodegroups/{id}/install` - Install exporters
- `POST /api/v1/nodegroups/{id}/uninstall` - Uninstall exporters
- `POST /api/v1/nodegroups/{id}/health-check` - Check exporter health

#### Monitoring Servers
- `GET /api/v1/monitoring-servers` - List monitoring servers
- `POST /api/v1/monitoring-servers` - Create monitoring server
- `POST /api/v1/monitoring-servers/{id}/install-stack` - Install monitoring stack
- `POST /api/v1/monitoring-servers/{id}/sync-targets` - Sync Prometheus targets
- `POST /api/v1/monitoring-servers/{id}/test-connection` - Test connectivity

#### Stats
- `GET /api/v1/stats` - Get fleet statistics
- `GET /health` - Application health check

## Requirements

### Web Application Server
- Docker 20.10+
- Docker Compose v2+
- 2GB RAM minimum
- Network access to monitoring server and GPU nodes (SSH)

### Monitoring Server
- Docker 20.10+
- Docker Compose v2+
- 8GB RAM minimum (16GB+ for 500+ nodes)
- Storage: 100GB+ for metrics retention
- Network access to GPU nodes on ports 5000, 9100, 9417

### GPU Nodes
- AMD GPU with ROCm driver installed
- Docker (for AMD Device Metrics Exporter)
- Python 3 (for RDMA Exporter)
- Systemd for service management
- SSH access from web application server
- RDMA-capable NICs (optional, for RDMA metrics)

## License

Apache 2.0

## Support

For issues and feature requests, please open a GitHub issue.
