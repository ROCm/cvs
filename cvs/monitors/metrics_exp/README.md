# GPU Fleet & Control Plane Monitor

A comprehensive, web-based monitoring solution for AMD GPU clusters and HPC/Kubernetes control plane infrastructure. Deploy monitoring, manage node groups, and visualize metrics through an intuitive UI with pre-built Grafana dashboards.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Software Installed Per Node Type](#software-installed-per-node-type)
- [Components](#components)
- [Quick Start](#quick-start)
- [Web UI Guide](#web-ui-guide)
- [Collected Metrics](#collected-metrics)
- [Storage Planning](#storage-planning)
- [Pre-built Dashboards](#pre-built-dashboards)
- [Configuration](#configuration)
- [Debugging Guide](#debugging-guide)
- [SSH Tunnel for Blocked Ports](#ssh-tunnel-for-blocked-ports)
- [API Reference](#api-reference)
- [Requirements](#requirements)

## Overview

GPU Fleet Monitor provides end-to-end monitoring for AMD GPU clusters and control plane infrastructure with:

- **Web-based Management UI**: Configure monitoring servers, GPU node groups, and control plane node groups
- **Automatic Exporter Installation**: Deploy exporters via SSH — no manual setup on nodes
- **GPU Node Monitoring**: AMD GPU metrics, system metrics, RDMA network, and logs
- **Slurm Control Plane Monitoring**: Job queues, node states, scheduler health, backfill stats
- **Kubernetes Control Plane Monitoring**: API server, etcd, scheduler, controller-manager, pod health
- **Pre-built Grafana Dashboards**: 9 dashboards covering GPU fleet, RDMA network, Slurm, and K8s
- **Log Aggregation**: Capture dmesg/journalctl/slurmctld logs via Promtail → Loki
- **Scalable Design**: Built for 1000+ nodes with configurable retention

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CORPORATE NETWORK                                   │
│                                                                                  │
│  ┌──────────────────────┐                                                       │
│  │  Web Application     │     SSH (direct or via Jump Host)                     │
│  │  Server              │─────────────────────────────────┐                     │
│  │  Fleet Monitor UI    │                                 │                     │
│  │  :30080              │                                 ▼                     │
│  │                      │                    ┌─────────────────────┐            │
│  │  Manages:            │                    │    Jump Host        │            │
│  │  - Monitoring Servers│                    │  (if required)      │            │
│  │  - GPU Node Groups   │                    └──────────┬──────────┘            │
│  │  - Control Node Grps │                               │ SSH                   │
│  └──────────────────────┘                               │                       │
│                                                         ▼                       │
│                                        ┌─────────────────────────────────────┐  │
│                                        │         Monitoring Server            │  │
│                                        │  Prometheus :30090                   │  │
│                                        │  Grafana    :30030                   │  │
│                                        │  Loki       :30100                   │  │
│                                        └──────┬──────────────┬────────────────┘  │
│                              HTTP scrape ─────┘              └──── Log push      │
│                                         │                          │             │
│         ┌───────────────────────────────┼──────────────────────────┼──────────┐  │
│         ▼                               ▼                          ▼          │  │
│  ┌─────────────┐              ┌─────────────────┐        ┌──────────────────┐ │  │
│  │  GPU Node   │              │  Slurm Head Node│        │  K8s Control     │ │  │
│  │  :5000 GPU  │              │  :9418 Slurm    │        │  Plane Node      │ │  │
│  │  :9100 Node │              │  :9100 Node     │        │  :9419 K8s       │ │  │
│  │  :9417 RDMA │              │  Promtail       │        │  :9100 Node      │ │  │
│  │  Promtail   │              └─────────────────┘        │  Promtail        │ │  │
│  └─────────────┘                                         └──────────────────┘ │  │
│         GPU Nodes (1000+)         Control Plane Nodes                          │  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Software Installed Per Node Type

This section describes exactly what gets installed on each type of node, which ports are used, and what firewall rules are required.

---

### 1. Monitoring Server

The monitoring server runs the full observability stack as Docker containers. This is the same server that runs the Fleet Monitor UI (or a separate dedicated server).

| Service | Container Name | Port | Purpose |
|---|---|---|---|
| Fleet Monitor UI + API | `fleet-manager` | **30080** | Web UI, REST API, SSH orchestration |
| PostgreSQL | `fleet-postgres` | 5432 (internal) | Configuration database |
| Prometheus | `fleet-prometheus` | **30090** | Metrics time-series database |
| Grafana | `fleet-grafana` | **30030** | Dashboards and visualization |
| Loki | `fleet-loki` | **30100** | Log aggregation |

**Inbound ports to open** on the monitoring server:
- `30080` — Fleet Monitor UI (from your browser/admin workstation)
- `30090` — Prometheus (optional, for direct PromQL access)
- `30030` — Grafana (from your browser)
- `30100` — Loki (optional, for direct log queries)

**Outbound access required** from the monitoring server:
- SSH port `22` to all GPU nodes, Slurm head nodes, and K8s control plane nodes (or via jump host)
- HTTP to GPU nodes: `5000`, `9100`, `9417`, `9420`
- HTTP to Slurm head nodes: `9418`, `9100`
- HTTP to K8s control plane nodes: `9419`, `9100`

---

### 2. GPU Compute Nodes

Installed automatically by Fleet Monitor via SSH when you click **Install** on a Node Group.

| Component | Binary/Container | Port | Managed By | Purpose |
|---|---|---|---|---|
| AMD Device Metrics Exporter | Docker container `device-metrics-exporter` | **5000** | Docker (restart: always) | GPU metrics — utilization, temperature, power, VRAM, ECC, RAS |
| Node Exporter | `/usr/local/bin/node_exporter` | **9100** | systemd `node_exporter.service` | Host system — CPU, memory, disk, network |
| RDMA Exporter | `/usr/local/bin/rdma-exporter.py` | **9417** | systemd `rdma-exporter.service` | RDMA/RoCE — link status, traffic, congestion, errors |
| User Activity Exporter | `/usr/local/bin/user-activity-exporter.py` | **9420** | systemd `user-activity-exporter.service` | User logins, KFD GPU processes (who is running GPU workloads), process counts |
| Promtail | `/usr/local/bin/promtail` | 9080 (internal) | systemd `promtail.service` | Ships dmesg/journalctl/auth logs to Loki |

**Inbound ports to open** on GPU nodes (from monitoring server):
- `5000` — AMD GPU Exporter
- `9100` — Node Exporter
- `9417` — RDMA Exporter
- `9420` — User Activity Exporter

**Prerequisites on GPU nodes**:
- AMD GPU with ROCm driver (`/dev/kfd`, `/dev/dri` present)
- Docker or Podman (for AMD Device Metrics Exporter)
- Python 3.x (for RDMA and User Activity exporters)
- `systemd` for service management
- SSH access from Fleet Monitor server
- RDMA-capable NICs (optional, for RDMA metrics)

---

### 3. Slurm Head Node / Login Node

Installed automatically by Fleet Monitor via SSH when you click **Install** on a Control Node Group (type: Slurm). The exporter can run on the Slurm head node OR a login node — both work since Slurm CLI commands (`sinfo`, `squeue`, `sdiag`, `sacct`) are available on both.

| Component | Binary | Port | Managed By | Purpose |
|---|---|---|---|---|
| Slurm Exporter | `/usr/local/bin/slurm-exporter.py` | **9418** | systemd `slurm-exporter.service` | Slurm cluster metrics — node states, job queue, scheduler health, CPU/memory allocation |
| Node Exporter | `/usr/local/bin/node_exporter` | **9100** | systemd `node_exporter.service` | Head node system metrics — CPU, memory, disk |
| Promtail | `/usr/local/bin/promtail` | 9080 (internal) | systemd `promtail.service` | Ships slurmctld/system logs to Loki |

**Inbound ports to open** on Slurm head/login node (from monitoring server):
- `9418` — Slurm Exporter
- `9100` — Node Exporter

**Prerequisites on Slurm head/login node**:
- Slurm CLI tools on PATH: `sinfo`, `squeue`, `sdiag`, `sacct`, `scontrol`
- Python 3.x
- `systemd`
- SSH access from Fleet Monitor server

> **Note on CPU utilization**: The "Compute CPU Util %" in the dashboard shows **cluster-wide compute node** utilization (CPUs used by running jobs across all compute nodes), not the head node's own CPU. The head node's CPU is always near 0% because it only runs the Slurm scheduler — this is normal and expected.

> **Note on slurmctld/slurmdbd indicators**: These are checked via `scontrol ping` and `sacctmgr show stats` respectively, which work correctly from both the head node and login nodes without requiring local daemon processes.

---

### 4. Kubernetes Control Plane Node

Installed automatically by Fleet Monitor via SSH when you click **Install** on a Control Node Group (type: Kubernetes). Must be installed on an **actual control plane node** (not a worker node) for full metrics access.

| Component | Binary | Port | Managed By | Purpose |
|---|---|---|---|---|
| K8s Control Plane Exporter | `/usr/local/bin/k8s-cp-exporter.py` | **9419** | systemd `k8s-cp-exporter.service` | K8s component metrics — API server, etcd, scheduler, controller-manager, node/pod status |
| Node Exporter | `/usr/local/bin/node_exporter` | **9100** | systemd `node_exporter.service` | Control plane node system metrics — CPU, memory, disk |
| Promtail | `/usr/local/bin/promtail` | 9080 (internal) | systemd `promtail.service` | Ships kube-apiserver/etcd/system logs to Loki |

**Inbound ports to open** on K8s control plane node (from monitoring server):
- `9419` — K8s Control Plane Exporter
- `9100` — Node Exporter

> **Cilium users**: If the cluster uses Cilium CNI with host firewall enabled, standard `iptables` rules may be bypassed. Apply a `CiliumClusterWideNetworkPolicy` or use `CiliumNetworkPolicy` to allow ingress on ports 9419 and 9100 from the monitoring server.

**Prerequisites on K8s control plane node**:
- `kubectl` on PATH with access to `/etc/kubernetes/admin.conf` (auto-detected)
- Python 3.x
- `systemd`
- SSH access from Fleet Monitor server
- Read access to `/etc/kubernetes/pki/` for etcd TLS certificates (for etcd metrics)

**Kubeconfig options** (configured via the UI when creating the group):

| Option | Description |
|---|---|
| **Auto-detect** (default) | Exporter looks for `/etc/kubernetes/admin.conf` then `~/.kube/config` on the node |
| **Path on node** | Specify the full path of an existing kubeconfig on the K8s node |
| **Upload file** | Upload a kubeconfig from your workstation via the UI; deployed to `/etc/k8s-cp-exporter/kubeconfig` on the node during Install |

**Authentication strategy used by the K8s exporter** (in priority order):
1. **API Server & kubectl-based metrics**: Uses `kubectl get --raw /metrics` which reads the configured kubeconfig automatically
2. **etcd metrics**: Auto-detects TLS certs from `/etc/kubernetes/pki/etcd/` — tries `healthcheck-client.crt` → `peer.crt` → `server.crt`. Override with `--etcd-ca-cert`, `--etcd-client-cert`, `--etcd-client-key` flags if certs are at a non-standard path
3. **Scheduler (10259) & Controller Manager (10257)**: Uses `apiserver-kubelet-client.crt` for mTLS (skips server cert verification for localhost connections)

---

### 5. Jump Host (Optional)

No software is installed on the jump host. Fleet Monitor uses it purely as an SSH relay to reach nodes in isolated networks.

**Requirements**:
- SSH access from Fleet Monitor server to jump host
- SSH access from jump host to target nodes
- The key for target nodes can be either on the Fleet Monitor server (relayed through jump host) or stored on the jump host itself

---

## Components

### 1. Web Application Server

**Role**: Hosts the Fleet Monitor web UI and orchestrates all deployments.

**Services**:
- **Fleet Monitor UI** (port 30080): Web interface for managing monitoring infrastructure
- **PostgreSQL**: Stores configuration for node groups, monitoring servers, and nodes

**Responsibilities**:
- Manage monitoring server configurations
- Define and organize GPU node groups and control node groups
- Install/uninstall exporters on nodes via SSH
- Sync Prometheus targets to monitoring servers
- Provision Grafana dashboards automatically

### 2. Monitoring Server

**Role**: Collects, stores, and visualizes metrics from all node types.

**Services**:
- **Prometheus** (port 30090): Time-series database — scrapes all exporters
- **Grafana** (port 30030): Visualization, dashboards, and alerting
- **Loki** (port 30100): Log aggregation from all node types

**Prometheus scrape jobs configured**:
| Job | Port | Source | Label |
|---|---|---|---|
| `amd_gpu_metrics` | 5000 | GPU nodes | `node_group` |
| `node_exporter` | 9100 | GPU nodes | `node_group` |
| `rdma_metrics` | 9417 | GPU nodes | `node_group` |
| `control_node_exporter` | 9100 | Control plane nodes | `control_node_group` |
| `slurm_metrics` | 9418 | Slurm head/login nodes | `control_node_group` |
| `k8s_control_plane` | 9419 | K8s control plane nodes | `control_node_group` |
| `user_activity` | 9420 | GPU nodes | `node_group` |

### 3. GPU Nodes

**Role**: Run GPU workloads and expose metrics for collection.

**Installed by**: Fleet Monitor → Node Groups → Install

### 4. Slurm Head / Login Nodes

**Role**: Run Slurm scheduler daemons and expose cluster-level metrics.

**Installed by**: Fleet Monitor → Control Node Groups (type: Slurm) → Install

### 5. Kubernetes Control Plane Nodes

**Role**: Run K8s control plane components and expose component-level metrics.

**Installed by**: Fleet Monitor → Control Node Groups (type: Kubernetes) → Install

### 6. Jump Host (Optional)

**Role**: SSH relay for nodes in isolated networks.

## Quick Start

### 1. Deploy the Web Application + Monitoring Stack

```bash
git clone <repo-url>
cd metrics_exp

# Configure environment
cp .env.example .env
# Edit .env with your passwords and settings

# Build and start
docker compose build --no-cache
docker compose up -d

# Verify all containers are healthy
docker compose ps
```

### 2. Access the Web UI

Open `http://your-server:30080` in your browser.

### 3. Configure a Monitoring Server

1. Go to **Monitoring Servers** → **Add Server**
2. Enter the monitoring server IP and SSH credentials
3. Click **Install Stack** to deploy Prometheus, Grafana, and Loki
4. All 9 dashboards are provisioned automatically

### 4. Monitor GPU Nodes

1. Go to **Node Groups** → **Add Node Group**
2. Enter node IPs and associate with a monitoring server
3. Upload SSH key for authentication
4. Click **Verify Connectivity** → then **Install**
5. Exporters installed: AMD GPU Exporter, Node Exporter, RDMA Exporter, Promtail

### 5. Monitor Slurm Control Plane

1. Go to **Control Node Groups** → **Add Control Group**
2. Select type: **Slurm**
3. Enter the Slurm head node or login node IP
4. Associate with a monitoring server
5. Upload SSH key → **Verify Connectivity** → **Install**
6. Exporters installed: Slurm Exporter (9418), Node Exporter (9100), Promtail
7. A Slurm Control Plane dashboard is created automatically in Grafana

### 6. Monitor Kubernetes Control Plane

1. Go to **Control Node Groups** → **Add Control Group**
2. Select type: **Kubernetes**
3. Enter a K8s control plane node IP (must be a master node, not a worker)
4. Associate with a monitoring server
5. Upload SSH key → **Verify Connectivity** → **Install**
6. Exporters installed: K8s CP Exporter (9419), Node Exporter (9100), Promtail
7. A Kubernetes Control Plane dashboard is created automatically in Grafana

### 7. View Dashboards

Access Grafana at `http://monitoring-server:30030` (default: admin/admin)

Dashboards are organized in two folders:
- **GPU Fleet Monitoring**: GPU utilization, thermal/power, GPU health, CPU/system, RDMA network, logs
- **Control Plane Monitoring**: Slurm control plane, Kubernetes control plane

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
|---|---|
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
|---|---|
| Name | Configuration name |
| Scrape Interval | How often to collect metrics |
| Enabled Metrics | Which metric types to collect |

### Node Groups (GPU Nodes)

Organize AMD GPU compute nodes into logical groups.

| Field | Description |
|---|---|
| Name | Group name (e.g., "cluster-a", "training-nodes") |
| Monitoring Server | Which monitoring server collects metrics |
| SSH User | Username for node access |
| SSH Auth Type | Key or password authentication |
| Use Jump Host | Enable if nodes require SSH relay |
| Node IPs | List of GPU node IP addresses |

**Actions**:
- **Verify Connectivity**: Test SSH to all nodes
- **Install**: Deploy AMD GPU Exporter, Node Exporter, RDMA Exporter, User Activity Exporter, Promtail (with auth log shipping)
- **Force Reinstall**: Reinstall even if already active (use after upgrades or exporter script updates)
- **Uninstall**: Remove all exporters from nodes
- **Sync Targets**: Update Prometheus targets

### Control Node Groups (Slurm / Kubernetes)

Monitor control plane infrastructure alongside your GPU fleet.

| Field | Description |
|---|---|
| Name | Group name (e.g., "hpc-cluster-head", "k8s-prod-cp") |
| Control Plane Type | **Slurm** or **Kubernetes** |
| Monitoring Server | Uses the same monitoring stack as GPU node groups |
| SSH User | Username for control node access |
| SSH Auth Type | Key or password authentication |
| Use Jump Host | Enable if nodes require SSH relay |
| Node IPs | IP address(es) of head node(s) |
| Custom Exporter Port | Override default port (9418 for Slurm, 9419 for K8s) |
| Kubeconfig Source | **Kubernetes only**: Auto-detect / Path on node / Upload file |
| Kubeconfig Path | Path on the K8s node (when source = "Path on node") |

**Actions**:
- **Verify Connectivity**: Test SSH to control nodes
- **Install**: Deploy control-plane-specific exporter, Node Exporter, Promtail
- **Upload Kubeconfig**: (Kubernetes groups) Upload kubeconfig file — deployed to node on next Install
- **Force Reinstall**: Reinstall (use after exporter script updates or kubeconfig changes)
- **Refresh Targets**: Update Prometheus target files

## Collected Metrics

> **Maintainer note**: When adding new metrics (new exporter, new metric category, or new Prometheus job), update this section and the [Storage Planning](#storage-planning) section to keep series counts accurate.

The table below summarises every metric collected, organised by exporter, category, and Prometheus label structure. Series counts assume 8 GPUs per GPU node and 2 RDMA NICs per GPU node; adjust proportionally for your hardware.

---

### 1. AMD GPU Metrics — Port 5000 · Scrape interval: 30s · Label: `node_group`

**Installed on**: GPU compute nodes via AMD Device Metrics Exporter (Docker container).

| Category | Metric Names | Labels | Series/node (8 GPUs) | Description |
|---|---|---|---|---|
| **Compute Utilization** | `gpu_gfx_activity` | `gpu`, `hostname` | 8 | Graphics engine utilisation % |
| | `gpu_umc_activity` | `gpu`, `hostname` | 8 | Unified memory controller activity % |
| | `gpu_mm_activity` | `gpu`, `hostname` | 8 | Multimedia engine activity % |
| | `gpu_vcn_busy_instantaneous` | `gpu`, `hostname` | 8 | Video codec engine busy % |
| **Temperature** | `gpu_edge_temperature` | `gpu`, `hostname` | 8 | Edge (die) temperature °C |
| | `gpu_junction_temperature` | `gpu`, `hostname` | 8 | Junction (hotspot) temperature °C |
| | `gpu_memory_temperature` | `gpu`, `hostname` | 8 | HBM memory temperature °C |
| **Power** | `gpu_power_usage` | `gpu`, `hostname` | 8 | Socket power draw (W) |
| | `gpu_package_power` | `gpu`, `hostname` | 8 | Package total power (W) |
| **VRAM** | `gpu_total_vram` | `gpu`, `hostname` | 8 | Total HBM capacity (bytes) |
| | `gpu_used_vram` | `gpu`, `hostname` | 8 | Used VRAM (bytes) |
| | `gpu_free_vram` | `gpu`, `hostname` | 8 | Free VRAM (bytes) |
| **PCIe** | `pcie_speed` | `gpu`, `hostname` | 8 | PCIe link speed (GT/s) |
| | `pcie_bandwidth` | `gpu`, `hostname` | 8 | PCIe bandwidth (bytes/s) |
| | `pcie_replay_count` | `gpu`, `hostname` | 8 | PCIe replay error count |
| **ECC / RAS** | `gpu_ecc_correctable_total` | `gpu`, `hostname`, `block` | ~40 | Correctable (single-bit) errors per memory block |
| | `gpu_ecc_uncorrectable_total` | `gpu`, `hostname`, `block` | ~40 | Uncorrectable (multi-bit) errors per memory block |
| | `gpu_ras_error_count` | `gpu`, `hostname`, `block` | ~16 | RAS error events |
| **Clocks** | `gpu_sclk_frequency` | `gpu`, `hostname` | 8 | GPU shader clock (MHz) |
| | `gpu_mclk_frequency` | `gpu`, `hostname` | 8 | Memory clock (MHz) |
| **Fan** | `gpu_fan_speed` | `gpu`, `hostname` | 8 | Fan speed (RPM) |
| **Health** | `gpu_health` | `gpu`, `hostname` | 8 | Overall GPU health (1=healthy, 0=error) |
| **Subtotal** | | | **~280 series/node** | |

---

### 2. System Metrics — Port 9100 · Scrape interval: 15s · Label: `node_group` or `control_node_group`

**Installed on**: GPU nodes, Slurm head nodes, K8s control plane nodes.

| Category | Metric Names | Labels | Series/node (128-core server) | Description |
|---|---|---|---|---|
| **CPU** | `node_cpu_seconds_total` | `cpu`, `mode` (idle/user/system/iowait/irq/softirq/steal/guest) | ~1,024 | Per-core CPU time by mode — **largest contributor to storage** |
| | `node_load1`, `node_load5`, `node_load15` | — | 3 | System load averages |
| | `node_context_switches_total` | — | 1 | Context switches/sec |
| **Memory** | `node_memory_MemTotal_bytes` | — | 1 | Total RAM |
| | `node_memory_MemAvailable_bytes` | — | 1 | Available RAM |
| | `node_memory_MemFree_bytes` | — | 1 | Free RAM |
| | `node_memory_Buffers_bytes`, `node_memory_Cached_bytes` | — | 2 | Buffers and cache |
| | `node_memory_SwapTotal_bytes`, `node_memory_SwapFree_bytes` | — | 2 | Swap space |
| | ~20 other memory counters | — | ~20 | Huge pages, vmalloc, slab, etc. |
| **Disk I/O** | `node_disk_read_bytes_total`, `node_disk_written_bytes_total` | `device` | ~8 (4 disks) | Bytes read/written per device |
| | `node_disk_io_time_seconds_total`, `node_disk_reads_completed_total` | `device` | ~16 | IOPS and latency per device |
| **Filesystem** | `node_filesystem_size_bytes`, `node_filesystem_avail_bytes`, `node_filesystem_free_bytes` | `device`, `mountpoint`, `fstype` | ~15 (3 mounts) | Disk space per filesystem |
| **Network** | `node_network_receive_bytes_total`, `node_network_transmit_bytes_total` | `device` | ~8 (4 NICs) | Traffic per interface |
| | `node_network_receive_packets_total`, `node_network_transmit_packets_total` | `device` | ~8 | Packets per interface |
| | `node_network_receive_errs_total`, `node_network_transmit_errs_total` | `device` | ~8 | Errors per interface |
| **Hardware misc** | `node_hwmon_temp_celsius` | `chip`, `sensor` | ~20 | Hardware temperature sensors |
| | `node_pressure_cpu_waiting_seconds_total` | — | ~3 | PSI (pressure stall information) |
| **Subtotal** | | | **~1,150 series/node** | |

> **Tip**: `node_cpu_seconds_total` alone accounts for ~88% of node_exporter series on high-core-count servers. Reduce its impact by increasing the scrape interval to 30s in `prometheus.yml`.

---

### 3. RDMA Network Metrics — Port 9417 · Scrape interval: 15s · Label: `node_group`

**Installed on**: GPU nodes (Python script `rdma-exporter.py`).

Assumes 2 RDMA NICs × 2 ports each = 4 device/port combinations per node.

| Category | Metric Names | Labels | Series/node | Description |
|---|---|---|---|---|
| **Link State** | `rdma_link_state` | `hostname`, `device`, `port`, `netdev` | 4 | Logical link up/down (1=active) |
| | `rdma_link_physical_state` | `hostname`, `device`, `port`, `netdev` | 4 | Physical layer state (1=link_up) |
| | `rdma_link_info` | `hostname`, `device`, `port`, `netdev`, `link_layer` | 4 | Link info label metric |
| **Traffic** | `rdma_stat_tx_bytes` | `hostname`, `device`, `port` | 4 | Bytes transmitted |
| | `rdma_stat_rx_bytes` | `hostname`, `device`, `port` | 4 | Bytes received |
| | `rdma_stat_tx_packets` | `hostname`, `device`, `port` | 4 | Packets transmitted |
| | `rdma_stat_rx_packets` | `hostname`, `device`, `port` | 4 | Packets received |
| **Congestion (DCQCN)** | `rdma_stat_np_cnp_sent` | `hostname`, `device`, `port` | 4 | Notification CNPs sent (reaction point) |
| | `rdma_stat_rp_cnp_handled` | `hostname`, `device`, `port` | 4 | CNPs handled (reaction point) |
| | `rdma_stat_pacing_alerts` | `hostname`, `device`, `port` | 4 | DCQCN pacing alert events |
| | `rdma_stat_pacing_reschedule` | `hostname`, `device`, `port` | 4 | DCQCN pacing reschedule events |
| | `rdma_stat_pacing_complete` | `hostname`, `device`, `port` | 4 | DCQCN pacing completions |
| **Congestion (PFC)** | `rdma_stat_rx_cnp_pkts` | `hostname`, `device`, `port` | 4 | PFC/CNP packets received |
| | `rdma_stat_rx_ecn_marked_pkts` | `hostname`, `device`, `port` | 4 | ECN-marked packets |
| **Errors** | `rdma_stat_packet_seq_err` | `hostname`, `device`, `port` | 4 | Packet sequence errors |
| | `rdma_stat_rx_icrc_errors` | `hostname`, `device`, `port` | 4 | CRC errors |
| | `rdma_stat_out_of_sequence` | `hostname`, `device`, `port` | 4 | Out-of-sequence packets |
| | `rdma_stat_local_ack_timeout_err` | `hostname`, `device`, `port` | 4 | ACK timeout errors |
| **Resources** | `rdma_resource_qp_total` | `hostname` | 1 | Total queue pairs |
| | `rdma_resource_cq_total` | `hostname` | 1 | Completion queues |
| | `rdma_resource_mr_total` | `hostname` | 1 | Memory regions |
| | `rdma_resource_pd_total` | `hostname` | 1 | Protection domains |
| | `rdma_resource_srq_total` | `hostname` | 1 | Shared receive queues |
| **Scrape** | `rdma_scrape_success` | `hostname` | 1 | 1 if RDMA devices found |
| **Subtotal** | | | **~85 series/node** | |

---

### 4. Slurm Control Plane Metrics — Port 9418 · Scrape interval: 30s · Label: `control_node_group`

**Installed on**: Slurm head node or login node (Python script `slurm-exporter.py`). One instance per cluster.

| Category | Metric Names | Series | Source CLI | Description |
|---|---|---|---|---|
| **Cluster State** | `slurm_nodes_total` | 1 | `sinfo --json` | Total compute nodes in cluster |
| | `slurm_nodes_state{state}` | ~6 | `sinfo --json` | Node count per state: idle, allocated, mixed, drained, down, planned |
| **CPU** | `slurm_cpus_total` | 1 | `sinfo --json` | Total CPUs across all compute nodes |
| | `slurm_cpus_allocated` | 1 | `sinfo --json` | CPUs allocated (alloc+mixed nodes only) |
| | `slurm_cpus_idle` | 1 | `sinfo --json` | Idle CPUs |
| | `slurm_running_cpus` | 1 | `squeue --json` | CPUs actively used by running jobs (most accurate utilisation metric) |
| **Memory** | `slurm_memory_total_mb` | 1 | `sinfo --json` | Total RAM across compute nodes (MB) |
| | `slurm_memory_allocated_mb` | 1 | `sinfo --json` | Allocated RAM (MB) |
| **Jobs** | `slurm_jobs_state{state}` | ~5 | `squeue --json` | Job count per state: running, pending, completing, failed, cancelled |
| **Scheduler** | `slurm_scheduler_backfill_cycle_last_seconds` | 1 | `sdiag` | Last backfill cycle duration |
| | `slurm_scheduler_backfill_cycle_mean_seconds` | 1 | `sdiag` | Mean backfill cycle duration |
| | `slurm_scheduler_backfill_jobs_total` | 1 | `sdiag` | Jobs started via backfill |
| | `slurm_scheduler_threads_active` | 1 | `sdiag` | Active slurmctld threads |
| | `slurm_scheduler_dbd_agent_queue_size` | 1 | `sdiag` | DBD message queue depth |
| **Daemons** | `slurm_slurmctld_up` | 1 | `scontrol ping` | slurmctld reachability (1=up) |
| | `slurm_slurmdbd_up` | 1 | `sacctmgr` | slurmdbd reachability (1=up) |
| **Table metrics** | `slurm_node_info{node,partition,state,cpus,memory,reason}` | 1 per compute node | `sinfo -N` | Per-node detail for Grafana table panel |
| | `slurm_job_info{job_id,name,user,partition,state,cpus,...}` | 1 per active job | `squeue --json` | Per-job detail for Grafana table panel |
| | `slurm_job_cpu_count{job_id,name,user,partition,state}` | 1 per running job | `squeue --json` | Numeric CPU count for sorting top consumers |
| | `slurm_partition_info{partition,state,nodes,cpus_total,cpus_alloc}` | 1 per partition | `sinfo -h` | Per-partition summary |
| | `slurm_recent_job_info{job_id,name,user,state,elapsed,exit_code}` | up to 200 | `sacct` | Completed jobs in last 24h |
| **Subtotal** | | **~30 fixed + variable table rows** | | Fixed series count is small; table metrics scale with cluster size |

---

### 5. Kubernetes Control Plane Metrics — Port 9419 · Scrape interval: 30s · Label: `control_node_group`

**Installed on**: K8s control plane node (Python script `k8s-cp-exporter.py`). One instance per cluster.

| Category | Metric Names | Series | Source | Description |
|---|---|---|---|---|
| **API Server** | `k8s_apiserver_up` | 1 | `/metrics` port 6443 | API server reachability (1=up) |
| | `k8s_apiserver_request_rate{verb,code}` | ~50 | `/metrics` port 6443 | Request rate by HTTP verb and response code |
| | `k8s_apiserver_request_duration_p99_seconds{verb}` | ~15 | `/metrics` port 6443 | P99 request latency by verb |
| | `k8s_apiserver_storage_objects{resource}` | ~40 | `/metrics` port 6443 | etcd object count by resource type |
| **etcd** | `k8s_etcd_up` | 1 | `/metrics` port 2379/2381 | etcd reachability |
| | `k8s_etcd_has_leader` | 1 | `/metrics` port 2379 | Leader elected (1) or not (0) |
| | `k8s_etcd_leader_changes_total` | 1 | `/metrics` port 2379 | Leader election change count |
| | `k8s_etcd_wal_fsync_duration_p99_seconds` | 1 | `/metrics` port 2379 | WAL fsync P99 latency (should be <10ms) |
| | `k8s_etcd_proposals_failed_total` | 1 | `/metrics` port 2379 | Failed Raft proposals |
| **Scheduler** | `k8s_scheduler_up` | 1 | `/metrics` port 10259 | Scheduler reachability |
| | `k8s_scheduler_pending_pods{queue}` | ~4 | `/metrics` port 10259 | Pending pods by queue: active, backoff, unschedulable, gated |
| | `k8s_scheduler_schedule_attempts_total{result}` | ~3 | `/metrics` port 10259 | Scheduling attempts by result |
| **Controller Manager** | `k8s_controller_manager_up` | 1 | `/metrics` port 10257 | Controller-manager reachability |
| | `k8s_controller_manager_workqueue_depth{queue_name}` | ~20 | `/metrics` port 10257 | Work queue depth per controller |
| **Table metrics** | `k8s_node_info{node,role,status,version}` | 1 per K8s node | `kubectl get nodes` | Node table for Grafana panel |
| | `k8s_pod_info{pod,namespace,phase,reason,node}` | 1 per non-running pod | `kubectl get pods` | Pending/failed pods table |
| | `k8s_component_health{component,endpoint}` | 4 | `/readyz` endpoints | Component health table |
| **Subtotal** | | **~150 fixed + variable table rows** | | |

---

### 6. User Activity Metrics — Port 9420 · Scrape interval: 30s · Label: `node_group`

**Installed on**: GPU compute nodes (Python script `user-activity-exporter.py`).

All data sources are world-readable (`/proc`, `/sys`, `who`, `last`) — no root access required for data collection. The systemd service runs as root (consistent with all other exporters) for future flexibility.

| Category | Metric Names | Labels | Notes |
|---|---|---|---|
| **Login count** | `node_logged_in_users_count` | — | Total active SSH/interactive sessions on the node |
| **Current sessions** | `node_logged_in_user_info` | `user`, `tty`, `from_host`, `login_time` | One series per active session (value always 1) — for Grafana table display |
| **Login history** | `node_recent_login_info` | `user`, `tty`, `from_host`, `date` | Last 30 login events from `last` (value always 1) — for history table |
| **KFD GPU process count** | `node_kfd_process_count` | — | Processes actively launching GPU kernels via KFD. **0 if KFD not loaded or no active GPU workloads** — handled gracefully |
| **KFD GPU process detail** | `node_kfd_process_info` | `user`, `pid`, `cmd`, `gpu_mem_mb` | Per-process GPU usage from `/sys/class/kfd/kfd/proc/` (value always 1) — shows who is running GPU workloads right now |
| **Process count per user** | `node_user_process_count` | `user` | Number of running processes per non-system user |
| **Top process per user** | `node_user_top_process_info` | `user`, `cmd` | Most notable command per user; GPU/ML workload keywords prioritised (value always 1) |
| **Subtotal** | | | **~25 fixed series + variable (scales with active users and GPU processes)** |

> **KFD data source**: `/sys/class/kfd/kfd/proc/<pid>/` — populated by the kernel for every process that opens `/dev/kfd`. No root required; world-readable sysfs.

---

### 7. Log Collection — Promtail → Loki · Push · Port 30100 on monitoring server

**Installed on**: All node types. Logs are pushed by Promtail running on each node.

| Node Type | Loki Stream Label | Log Sources | Typical Rate |
|---|---|---|---|
| GPU Nodes | `{node_group="<name>"}` | `/var/log/dmesg`, systemd journal, `/var/log/syslog`, `/var/log/auth.log` (Ubuntu) or `/var/log/secure` (RHEL) — ECC, RAS, GPU hang, thermal events, user logins, sudo | 50–500 lines/min/node |
| Slurm Nodes | `{control_node_group="<name>"}` | systemd journal, `/var/log/syslog`, `/var/log/auth.log` or `/var/log/secure` — slurmctld, slurmdbd, user logins | 10–100 lines/min |
| K8s Nodes | `{control_node_group="<name>"}` | systemd journal, `/var/log/syslog`, `/var/log/auth.log` or `/var/log/secure` — kube-apiserver, etcd, kubelet, user logins | 50–200 lines/min |

Loki compresses log data approximately 5:1 for typical system log text.

---

## Storage Planning

> **Maintainer note**: Update this section when new exporters or metric categories are added. Recalculate using the formula: `storage_GB = (retention_days × 86400 × samples_per_second × 1.5) / 1e9`

### Series Count Summary Per Node

| Exporter | Port | Scrape Interval | Series/node | % of Total |
|---|---|---|---|---|
| AMD GPU Exporter | 5000 | 30s | ~280 | 17% |
| Node Exporter | 9100 | 15s | ~1,150 | **69%** |
| RDMA Exporter | 9417 | 15s | ~85 | 5% |
| User Activity Exporter | 9420 | 30s | ~25 fixed + variable | ~2% |
| Slurm Exporter | 9418 | 30s | ~30 fixed | <1% |
| K8s Exporter | 9419 | 30s | ~150 fixed | <1% |
| **Total (GPU node)** | | | **~1,540/node** | |
| **Total (Slurm head)** | | | **~1,180/node** | |
| **Total (K8s control)** | | | **~1,300/node** | |

> **Node Exporter dominates** (~69%) due to high-frequency (15s) per-core CPU series. On a 128-core server, `node_cpu_seconds_total` alone generates 1,024 series (128 cores × 8 modes).
> **User Activity Exporter** has variable cardinality: fixed series (~15) plus 1 series per active user session, logged-in user, and GPU process. On a busy node with 10 active users and 20 GPU processes this adds ~50 extra series — negligible.

### Ingestion Rate Formula

```
samples/sec = Σ (series_per_node × node_count / scrape_interval_seconds)

Example — 128 GPU nodes:
  GPU:   128 × 280  ÷ 30  =  1,195 samples/sec
  Node:  128 × 1150 ÷ 15  = 9,813 samples/sec
  RDMA:  128 × 85   ÷ 15  =   725 samples/sec
  ──────────────────────────────────────────────
  Total:                   ≈ 11,733 samples/sec
```

### Prometheus TSDB Storage Formula

```
prometheus_GB = retention_days × 86,400 × samples_per_second × 1.5 bytes / 1,000,000,000
             × 1.15  (WAL overhead)

Example — 128 GPU nodes, 15 days:
  = 15 × 86,400 × 11,733 × 1.5 / 1e9 × 1.15
  ≈ 24.7 GB
```

### Loki Log Storage Formula

```
loki_GB = nodes × avg_lines_per_min × 60 × 24 × retention_days × avg_bytes_per_line
        ÷ compression_ratio / 1,000,000,000

Example — 128 GPU nodes, 200 lines/min avg, 150 bytes/line, 5:1 compression, 15 days:
  = 128 × 200 × 60 × 24 × 15 × 150 / 5 / 1e9
  ≈ 19.9 GB
```

### Scaling Reference Table

Assumes: 8 GPUs/node, 128 cores/node, 2 RDMA NICs/node, 15-day retention, all metrics enabled.

| GPU Nodes | Active Series | Ingestion Rate | Prometheus 15d | Loki 15d | **Total Recommended** |
|---|---|---|---|---|---|
| 16 | ~24K | ~1,500/s | ~3 GB | ~4 GB | **~10 GB** |
| 32 | ~48K | ~3,000/s | ~6 GB | ~8 GB | **~20 GB** |
| 64 | ~97K | ~6,000/s | ~12 GB | ~15 GB | **~35 GB** |
| **128** | **~194K** | **~12,000/s** | **~25 GB** | **~25 GB** | **~60 GB** |
| 256 | ~388K | ~24,000/s | ~50 GB | ~50 GB | **~115 GB** |
| 512 | ~776K | ~47,000/s | ~100 GB | ~100 GB | **~215 GB** |
| 1,000 | ~1.5M | ~92,000/s | ~195 GB | ~195 GB | **~420 GB** |

Add 20% headroom to all figures. For 500+ nodes, consider running multiple monitoring servers (one per 256 nodes) or use Thanos/Cortex for federated long-term storage.

### Storage Reduction Options

| Option | Method | Saving |
|---|---|---|
| Increase Node Exporter scrape interval | Change `15s` → `30s` in `prometheus.yml` job `node_exporter` | ~35% total Prometheus storage |
| Increase GPU scrape interval | Change `30s` → `60s` in job `amd_gpu_metrics` | ~8% total Prometheus storage |
| Reduce CPU mode tracking | Use recording rules to aggregate CPU modes before storage | ~20% total Prometheus storage |
| Shorter log retention | Set `loki_retention_days` in `.env` to e.g. 7 | 53% Loki storage |
| Disable RDMA exporter | Remove `rdma_metrics` job or don't install RDMA exporter | ~5% total Prometheus storage |

### Default Configuration Limits

The default `.env` values are:
```bash
PROMETHEUS_RETENTION_TIME=15d
PROMETHEUS_RETENTION_SIZE=50GB   # ← size cap (whichever hits first)
```

At 128 GPU nodes you use ~25 GB of the 50 GB cap — comfortable headroom.
At 256 nodes you approach the cap — increase `PROMETHEUS_RETENTION_SIZE=100GB`.

## Pre-built Dashboards

### GPU Fleet Monitoring Folder

1. **GPU Fleet Overview**: High-level fleet status, total GPUs, average utilization
2. **GPU Utilization**: Detailed compute and memory activity over time, plus **User Activity section** at the bottom — KFD GPU processes (who is running GPU workloads), active sessions, login history, auth logs
3. **Thermal & Power**: Temperature trends and power consumption
4. **GPU Health & Errors**: ECC/RAS monitoring with error counts
5. **CPU & System**: Host CPU, memory, and system metrics
6. **RDMA Network**: Link status, traffic, congestion (PFC/DCQCN), errors, resources
7. **Logs Analysis**: Critical log pattern detection from dmesg/journalctl

### Control Plane Monitoring Folder

8. **Slurm Control Plane**: Cluster compute overview, node states, job queue, top CPU consumers (pie chart + table), scheduler health, partition summary, recent jobs, head node system resources, logs
9. **Kubernetes Control Plane**: Component health stats, API server metrics, etcd health, scheduler queues, controller-manager workqueues, node/pod/component health tables, control plane system resources, logs

> **Per-group dashboards**: When you install a Control Node Group, a group-specific dashboard is automatically created in Grafana (pre-filtered to that group's data, no dropdown needed). The static dashboards above use a `$control_node_group` dropdown to switch between groups.

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

# Prometheus retention
PROMETHEUS_RETENTION_TIME=15d
PROMETHEUS_RETENTION_SIZE=50GB
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
docker compose ps
docker logs fleet-manager --tail 100
```

#### Check Application Health

```bash
curl http://localhost:30080/health
```

**Expected:**
```json
{"status":"healthy","version":"1.0.0","services":[{"name":"prometheus","status":"healthy"},{"name":"grafana","status":"healthy"},{"name":"database","status":"healthy"}]}
```

#### Check Database Tables

```bash
docker exec -it fleet-postgres psql -U fleet -d fleet_monitor -c "\dt"
```

**Expected tables include**: `monitoring_servers`, `node_groups`, `nodes`, `control_node_groups`, `control_nodes`, `metric_groups`, `installation_logs`

---

### GPU Nodes

#### Check AMD GPU Exporter

```bash
docker ps | grep device-metrics-exporter
curl -s http://localhost:5000/metrics | grep gpu_gfx_activity | head -5
```

#### Check Node Exporter

```bash
systemctl status node_exporter
curl -s http://localhost:9100/metrics | grep -c "^node_"
```

#### Check RDMA Exporter

```bash
systemctl status rdma-exporter
curl -s http://localhost:9417/metrics | head -10
```

#### Check Promtail

```bash
systemctl status promtail
```

#### Check User Activity Exporter

```bash
systemctl status user-activity-exporter
curl -s http://localhost:9420/health              # should return: OK

# Check key metrics
curl -s http://localhost:9420/metrics | grep -E "^node_(logged_in_users_count|kfd_process_count)"

# Verify KFD GPU processes are tracked (non-zero when GPU workloads are running)
curl -s http://localhost:9420/metrics | grep "^node_kfd_process_info" | head -5

# Verify current logins are tracked
curl -s http://localhost:9420/metrics | grep "^node_logged_in_user_info" | head -5
```

**Expected when GPU jobs are running:**
```
node_logged_in_users_count 3
node_kfd_process_count 8
node_kfd_process_info{user="alice",pid="12345",cmd="python3 train.py",gpu_mem_mb="24576"} 1
```

**Expected when cluster is idle:**
```
node_logged_in_users_count 0
node_kfd_process_count 0
```

---

### Slurm Head / Login Node

#### Check Slurm Exporter

```bash
systemctl status slurm-exporter
curl -s http://localhost:9418/health              # should return: OK
curl -s http://localhost:9418/metrics | grep -E "^slurm_(nodes_total|jobs_state|slurmctld_up|running_cpus)"
```

**Expected:**
```
slurm_nodes_total 69
slurm_jobs_state{state="running"} 21
slurm_slurmctld_up 1
slurm_running_cpus 23810
```

#### Verify Slurm CLI is available (required by exporter)

```bash
which sinfo squeue sdiag scontrol sacct
sinfo --json | python3 -c "import sys,json; d=json.load(sys.stdin); print('Keys:', list(d.keys())); items=d.get('sinfo',d.get('nodes',[])); print('Nodes:', len(items))"
```

#### Check service logs if exporter is not working

```bash
journalctl -u slurm-exporter -n 50 --no-pager
```

---

### Kubernetes Control Plane Node

#### Check K8s Exporter

```bash
systemctl status k8s-cp-exporter
curl -s http://localhost:9419/health              # should return: OK
curl -s http://localhost:9419/metrics | grep -E "^k8s_(apiserver|etcd|scheduler|controller_manager)_up"
```

**Expected (all components reachable):**
```
k8s_apiserver_up 1
k8s_etcd_up 1
k8s_scheduler_up 1
k8s_controller_manager_up 1
```

#### If etcd metrics show 0

The exporter auto-detects etcd certs. Check what it found at startup:
```bash
journalctl -u k8s-cp-exporter -n 30 --no-pager | grep -i "etcd\|cert\|kubeconfig"
ls /etc/kubernetes/pki/etcd/
```

If the certs are at a non-standard path, override via the systemd service:
```bash
sudo systemctl edit k8s-cp-exporter
# Add:
# [Service]
# ExecStart=
# ExecStart=/usr/bin/python3 /usr/local/bin/k8s-cp-exporter.py --port 9419 \
#   --etcd-ca-cert /path/to/etcd/ca.crt \
#   --etcd-client-cert /path/to/etcd/client.crt \
#   --etcd-client-key /path/to/etcd/client.key
```

#### If ports are blocked by Cilium CNI

```bash
# Temporary fix (does not survive reboot)
sudo iptables -I INPUT 1 -p tcp --dport 9419 -j ACCEPT
sudo iptables -I INPUT 1 -p tcp --dport 9100 -j ACCEPT

# Permanent fix — apply CiliumNetworkPolicy
kubectl apply -f - <<'EOF'
apiVersion: cilium.io/v2
kind: CiliumClusterWideNetworkPolicy
metadata:
  name: allow-monitoring-exporter
spec:
  nodeSelector:
    matchLabels:
      node-role.kubernetes.io/control-plane: ''
  ingress:
  - fromEntities:
    - world
    toPorts:
    - ports:
      - port: '9419'
        protocol: TCP
      - port: '9100'
        protocol: TCP
EOF
```

---

### Monitoring Server

#### Check Prometheus Targets

```bash
# Inside the container
docker exec fleet-prometheus ls /etc/prometheus/targets/

# Check all targets are UP
curl -s http://localhost:30090/api/v1/targets | python3 -c "
import json,sys
data=json.load(sys.stdin)
for t in data.get('data',{}).get('activeTargets',[]):
    job=t.get('labels',{}).get('job','')
    print(job,'|',t['health'],'|',t['scrapeUrl'],'|',t.get('lastError','')[:60])"
```

**Expected (GPU + Slurm + K8s):**
```
amd_gpu_metrics | up | http://10.x.x.x:5000/metrics |
node_exporter | up | http://10.x.x.x:9100/metrics |
rdma_metrics | up | http://10.x.x.x:9417/metrics |
control_node_exporter | up | http://10.x.x.x:9100/metrics |
slurm_metrics | up | http://10.x.x.x:9418/metrics |
k8s_control_plane | up | http://10.x.x.x:9419/metrics |
```

#### Check Prometheus Target Files

```bash
docker exec fleet-manager ls /prometheus-targets/
```

**Expected (with GPU + Slurm + K8s groups):**
```
gpu_my-cluster.json
node_my-cluster.json
rdma_my-cluster.json
control_node_slurm-head.json
slurm_slurm-head.json
k8s_k8s-prod.json
control_node_k8s-prod.json
```

#### Check Grafana Dashboards

```bash
curl -s -u admin:admin "http://localhost:30030/api/search?type=dash-db" | python3 -c "
import json,sys
for d in json.load(sys.stdin):
    print(f\"{d.get('folderTitle','General')}/{d['title']} (uid:{d['uid']})\")"
```

**Expected:**
```
GPU Fleet Monitoring/CPU & System (uid:cpu-system)
GPU Fleet Monitoring/GPU Fleet Overview (uid:fleet-overview)
GPU Fleet Monitoring/GPU Health & Errors (uid:gpu-health)
GPU Fleet Monitoring/GPU Utilization (uid:gpu-utilization)
GPU Fleet Monitoring/Logs Analysis (uid:logs-analysis)
GPU Fleet Monitoring/RDMA Network (uid:rdma-network)
GPU Fleet Monitoring/Thermal & Power (uid:thermal-power)
Control Plane Monitoring/Slurm Control Plane (uid:slurm-control-plane)
Control Plane Monitoring/Kubernetes Control Plane (uid:k8s-control-plane)
```

#### Check Loki Logs are Arriving

```bash
# Check Slurm head node logs are in Loki
curl -s -G 'http://localhost:30100/loki/api/v1/query_range' \
  --data-urlencode 'query={control_node_group="your-slurm-group-name"}' \
  --data-urlencode 'limit=5' | python3 -c "
import json,sys; d=json.load(sys.stdin)
streams=d.get('data',{}).get('result',[])
print('Streams found:', len(streams))
if streams: print('Sample:', streams[0].get('values',[['']])[0][1][:100])"
```

## SSH Tunnel for Blocked Ports

If ports are blocked by firewall, use SSH tunneling to access services via localhost.

### Tunnel to Monitoring Server

```bash
# Multiple tunnels (Prometheus, Grafana, Loki, Fleet Monitor)
ssh -L 9090:localhost:30090 \
    -L 3030:localhost:30030 \
    -L 3100:localhost:30100 \
    -L 8080:localhost:30080 \
    user@monitoring-server
```

Then access:
- Fleet Monitor: http://localhost:8080
- Grafana: http://localhost:3030
- Prometheus: http://localhost:9090

### Tunnel via Jump Host

```bash
ssh -J user@jump-host \
    -L 3030:localhost:30030 \
    -L 9090:localhost:30090 \
    user@monitoring-server
```

### Persistent Tunnel with autossh

```bash
autossh -M 0 -f -N \
    -L 3030:localhost:30030 \
    -L 9090:localhost:30090 \
    -o "ServerAliveInterval 30" \
    -o "ServerAliveCountMax 3" \
    user@monitoring-server
```

## API Reference

The Fleet Monitor API is available at `http://localhost:30080/docs` (Swagger UI).

### GPU Node Groups
- `GET /api/v1/nodegroups` — List all GPU node groups
- `POST /api/v1/nodegroups/with-nodes` — Create node group with initial nodes
- `POST /api/v1/nodegroups/{id}/ssh-key` — Upload SSH key
- `POST /api/v1/nodegroups/{id}/verify` — Test SSH connectivity
- `POST /api/v1/nodegroups/{id}/install` — Install GPU exporters
- `POST /api/v1/nodegroups/{id}/install` (force=true) — Force reinstall
- `POST /api/v1/nodegroups/{id}/refresh-targets` — Sync Prometheus targets

### Control Node Groups (Slurm / Kubernetes)
- `GET /api/v1/control-nodegroups` — List all control node groups
- `POST /api/v1/control-nodegroups/with-nodes` — Create control node group with initial nodes
- `PATCH /api/v1/control-nodegroups/{id}` — Update group settings (including kubeconfig source/path)
- `DELETE /api/v1/control-nodegroups/{id}` — Delete group and all its resources
- `POST /api/v1/control-nodegroups/{id}/ssh-key` — Upload SSH key for direct connection
- `POST /api/v1/control-nodegroups/{id}/jump-key` — Upload SSH key for jump host
- `POST /api/v1/control-nodegroups/{id}/kubeconfig` — Upload kubeconfig file (Kubernetes groups only)
- `POST /api/v1/control-nodegroups/{id}/verify` — Test SSH connectivity
- `POST /api/v1/control-nodegroups/{id}/install` — Install control plane exporter
- `POST /api/v1/control-nodegroups/{id}/install` (force=true) — Force reinstall
- `POST /api/v1/control-nodegroups/{id}/refresh-targets` — Sync Prometheus targets
- `GET /api/v1/control-nodegroups/{id}/nodes` — List nodes
- `POST /api/v1/control-nodegroups/{id}/nodes/bulk` — Bulk add nodes
- `DELETE /api/v1/control-nodegroups/{id}/nodes/{node_id}` — Remove a node

### Monitoring Servers
- `GET /api/v1/monitoring-servers` — List monitoring servers
- `POST /api/v1/monitoring-servers` — Create monitoring server
- `POST /api/v1/monitoring-servers/{id}/install-stack` — Install monitoring stack
- `POST /api/v1/monitoring-servers/{id}/sync-targets` — Sync all targets
- `POST /api/v1/monitoring-servers/{id}/test-connection` — Test connectivity

### Stats & Health
- `GET /api/v1/stats` — Get fleet statistics
- `GET /health` — Application health check

## Requirements

### Web Application Server
- Docker 20.10+
- Docker Compose v2+
- 2GB RAM minimum
- Network access to monitoring server and all nodes (SSH on port 22)

### Monitoring Server
- Docker 20.10+
- Docker Compose v2+
- 8GB RAM minimum (16GB+ for 500+ nodes)
- Storage: 100GB+ for metrics retention
- Network access to:
  - GPU nodes on ports `5000`, `9100`, `9417`, `9420`
  - Slurm nodes on ports `9418`, `9100`
  - K8s control plane nodes on ports `9419`, `9100`

### GPU Nodes
- AMD GPU with ROCm driver installed (`/dev/kfd`, `/dev/dri` present)
- Docker or Podman (for AMD Device Metrics Exporter)
- Python 3.x (for RDMA Exporter and User Activity Exporter)
- `systemd`
- SSH access from Fleet Monitor server
- Firewall: open inbound `5000`, `9100`, `9417`, `9420`
- RDMA-capable NICs (optional)

### Slurm Head / Login Nodes
- Slurm CLI tools: `sinfo`, `squeue`, `sdiag`, `sacct`, `scontrol`, `sacctmgr`
- Python 3.x
- `systemd`
- SSH access from Fleet Monitor server
- Firewall: open inbound `9418`, `9100`

### Kubernetes Control Plane Nodes
- `kubectl` on PATH with kubeconfig access (auto-detected, or specify path/upload via UI)
- Python 3.x
- `systemd`
- SSH access from Fleet Monitor server
- Firewall: open inbound `9419`, `9100` (see Cilium note above)
- `/etc/kubernetes/pki/` readable by root (for etcd TLS auto-detection; optional if etcd certs are elsewhere)

## License

Apache 2.0

## Support

For issues and feature requests, please open a GitHub issue.
