# CVS Cluster Monitor

Real-time GPU cluster monitoring dashboard for AMD MI300/MI325 GPUs with intelligent host reachability detection, automatic recovery, and comprehensive log analysis.

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

### 🚀 **High-Performance Architecture**
- **TCP Probe-Based Reachability** (NEW): 5-second TCP probes detect reachable hosts before SSH attempts
  - Avoids 60+ second timeouts on unreachable nodes
  - 279x faster command execution (< 1s vs 198s for amd-smi on 392 nodes)
  - Periodic re-probing every 5 minutes with automatic SSH client recreation
- **Optimized SSH Operations**: Pool size of 50, supports 150-600+ nodes
- **Thread-Safe Execution**: Global lock prevents concurrent SSH crashes
- **Auto-Recovery**: Automatic SSH reconnection on failures, self-recovering from crashes

### 📊 **GPU Monitoring**
- **Real-time Metrics**: Utilization, temperature, power, memory usage (auto-updates every 60s)
- **PCIe Monitoring**: Link speed, bandwidth, replay count, NAK count
- **ECC Error Tracking**: Correctable and uncorrectable memory errors
- **XGMI Errors**: AMD Infinity Fabric interconnect monitoring
- **GPU Software Info**: ROCm version, amdgpu driver, firmware versions

### 🌐 **Network Monitoring**
- **RDMA Statistics**: Port data (rx/tx bytes), congestion control metrics (ECN, CNP)
- **RDMA Resources**: Protection domains, completion queues, queue pairs
- **LLDP Topology**: Network neighbor discovery and visualization
- **NIC Software Info**: Firmware versions, driver info for AMD AINIC, Broadcom Thor2, Mellanox CX7
- **Congestion Control**: Real-time PFC, ECN, and CNP packet monitoring

### 📝 **Advanced Log Analysis**
- **AMD Hardware Logs**: Filtered dmesg for PCIe, XGMI, amdgpu, CPU, NIC, Link errors
- **System Error Logs**: Critical kernel errors (emerg, alert, crit, err levels)
- **Userspace Errors**: OOM kills, segfaults, ML framework errors (PyTorch, TensorFlow, etc.)
- **Custom Grep Search** (NEW): Powerful search with validated grep/egrep pipe commands
  - Example: `grep -i 'error' | grep -v 'vital buffer'`
  - Security validated (only grep/egrep allowed)
  - First 5 lines per node displayed

### 🔧 **Configuration & Management**
- **Web-based Configuration**: No manual file editing required
- **SSH Key Upload**: Secure key upload via browser
- **Auto-Initialization**: Automatic startup from config files after restart/crash
- **Jump Host Support**: Full support for bastion/jump host architectures
- **LLDP Package Installation**: One-click cluster-wide package deployment
- **Health Tracking**: Stability-based status (requires 5 consecutive failures)

### 💻 **User Interface**
- **Dashboard**: Cluster overview with heatmaps (GPU utilization, memory, temperature, NIC traffic)
- **GPU Metrics**: Detailed per-GPU statistics, PCIe link status, ECC errors
- **NIC Metrics**: RDMA statistics, link status, congestion control
- **GPU Software**: ROCm, drivers, firmware versions
- **NIC Software**: NIC firmware and driver information
- **Topology**: Network topology visualization with LLDP neighbors
- **Logs**: Filterable system logs with clickable navigation
- **WebSocket Updates**: Real-time data push (no polling from browser)

## Quick Start

### Prerequisites

- Docker and Docker Compose v2
- SSH access to cluster nodes (direct or via jump host)
- AMD MI300/MI325 GPUs on cluster nodes (with amd-smi installed)
- RDMA-capable NICs (optional, for network monitoring)

### 1. Clone Repository

```bash
git clone <repository-url>
cd cvs-cluster-monitor
```

### 2. Prepare Configuration Files

Create configuration files in the `config/` directory:

**config/nodes.txt** (one node per line):
```
node1.cluster.local
node2.cluster.local
node3.cluster.local
```

**config/cluster.yaml**:
```yaml
cluster:
  nodes_file: config/nodes.txt
  ssh:
    username: your_ssh_username
    key_file: /root/.ssh/id_rsa
    timeout: 30
    jump_host:
      enabled: false  # Set to true if using jump host
      host: jumphost.example.com
      username: jump_username
      password: jump_password  # Or use key_file instead
  polling:
    interval: 60  # Metrics collection interval (seconds)
    failure_threshold: 5  # Consecutive failures before marking unreachable
```

### 3. Build and Deploy

Run the **full-rebuild.sh** script for complete build and deployment:

```bash
./full-rebuild.sh
```

**What full-rebuild.sh does:**
1. ✅ Stops any existing containers
2. ✅ Removes old containers and images
3. ✅ Builds fresh Docker image (multi-stage build):
   - Builds React frontend (optimized production bundle)
   - Installs Python backend dependencies
   - Creates single production image with both frontend + backend
4. ✅ Starts new container on port 8005
5. ✅ **Auto-initializes monitoring** from config files
6. ✅ Runs TCP probe to detect reachable hosts
7. ✅ Starts metrics collection automatically

**Build time:** ~2-5 minutes (depending on system)

### 4. Access Dashboard

Open browser: **http://<your-server-ip>:8005**

The monitoring starts automatically! No manual configuration needed if config files are set up.

### 5. Optional: Configure via Web UI

If you need to change configuration without rebuilding:

1. Go to **Configuration** tab
2. **Upload SSH Keys** (if not using password auth)
3. **Modify settings** (nodes, jump host, etc.)
4. Click **"Save Configuration and Start Monitoring"**

### 6. Monitor Your Cluster

**Dashboard Features:**
- **Cluster Overview**: Total nodes, healthy nodes, average GPU utilization, temperature
- **GPU Heatmaps**: Visual representation of utilization, memory, temperature across cluster
- **NIC Heatmaps**: RDMA traffic visualization for 8 GPU-mapped NICs

**Available Pages:**
- **Dashboard**: Overview + heatmaps
- **GPU Metrics**: Detailed GPU utilization, memory, temperature, power, PCIe, ECC, XGMI
- **NIC Metrics**: RDMA links, statistics, congestion control, IP addresses
- **GPU Software**: ROCm versions, amdgpu driver, firmware
- **NIC Software**: NIC firmware, drivers (AMD AINIC, Broadcom, Mellanox)
- **Topology**: LLDP network topology visualization
- **Logs**: AMD hardware logs, system errors, userspace errors, custom grep search
- **Configuration**: Cluster settings, SSH keys, package installation

## Performance & Scalability

### TCP Probe Optimization

**Before optimization:**
- 392 nodes with 1 unreachable: 198 seconds for amd-smi command
- Unreachable nodes timeout after 60+ seconds each

**After TCP probe optimization:**
- Initial probe: ~5 seconds for all 392 nodes (100 concurrent threads)
- amd-smi command: **0.71 seconds** (279x faster!)
- Unreachable nodes: Never attempted via SSH (instant skip)

### Supported Scale

- **Tested**: 150-617 nodes
- **TCP Probe**: Handles 600+ nodes in ~10 seconds
- **SSH Pool Size**: 50 concurrent connections (configurable)
- **Thread-Safe**: Global lock prevents crashes from concurrent operations
- **Auto-Recovery**: Periodic re-probe every 5 minutes, jump host auto-reconnection

### Resource Usage

- **Memory**: ~3-5GB for 600 nodes (depends on metrics volume)
- **CPU**: Low (mostly I/O wait during SSH)
- **Network**: Minimal (compressed SSH traffic)

## Testing

### Run Comprehensive Test Suite

```bash
python3 test_cluster_monitor.py
python3 test_cluster_monitor.py --hosts config/test_nodes.txt --config config/cluster.yaml
python3 test_cluster_monitor.py --verbose
```

**Test Coverage:**
- ✅ TCP probe functionality (5 tests)
- ✅ SSH connectivity (4 tests)
- ✅ API endpoints (9 tests)
- ✅ Feature validation (3 tests)
- ✅ Performance characteristics (2 tests)

**Output includes:**
- Pass/Fail counts with color coding
- Execution time per test
- Detailed error messages for failures
- Overall test duration

## Configuration Files

### cluster.yaml

Located at `config/cluster.yaml` (excluded from git):

```yaml
cluster:
  ssh:
    username: your_username
    key_file: /root/.ssh/id_rsa  # Path in container
    timeout: 30
    jump_host:
      enabled: true  # Set to false for direct SSH
      host: jumphost.example.com
      username: jump_user
      password: jump_password  # Or use key_file
      node_username: cluster_username
      node_key_file: /home/user/.ssh/cluster_key  # Path on jump host
```

### nodes.txt

Located at `config/nodes.txt` (excluded from git):

```
# One node per line
node1.cluster.local
node2.cluster.local
```

## Operational Details

### Auto-Initialization on Startup

The system automatically initializes monitoring when the container starts:
1. Loads configuration from `config/cluster.yaml` and `config/nodes.txt`
2. Runs TCP probe to detect reachable hosts (~5-10 seconds)
3. Initializes SSH connections (Pssh or JumpHostPssh based on config)
4. Starts metrics collection loop (every 60 seconds by default)
5. Starts periodic probe task (every 5 minutes)

**After crash/restart**: Monitoring resumes automatically without manual intervention!

### Periodic Host Probing

Every 5 minutes, the system:
- Re-probes all configured hosts via TCP
- Detects newly reachable nodes (nodes that came back online)
- Detects newly unreachable nodes (nodes that went offline)
- Recreates SSH client with updated reachable host list
- Logs all reachability changes

### Mid-Execution Recovery

If connection errors occur during metrics collection:
- Triggers immediate re-probe (doesn't wait 5 minutes)
- Updates reachable/unreachable lists
- Recreates SSH client
- Continues with next metrics collection cycle

### Jump Host Auto-Reconnection

For jump host scenarios:
- Checks jump host connection health before each operation
- Automatically reconnects if connection is lost
- Re-probes cluster nodes after reconnection
- Ensures continuous monitoring even if jump host has network issues

### Command Timeouts

All SSH commands have timeouts to prevent hanging:
- **GPU commands** (amd-smi): 120 seconds
- **NIC commands** (rdma, ip): 60 seconds
- **Log searches**: 60 seconds
- **Software queries**: 60-120 seconds

## Helper Scripts

### full-rebuild.sh

Complete rebuild and deployment:
```bash
bash full-rebuild.sh
```

Automatically:
- Stops existing containers
- Builds fresh Docker image
- Starts container
- Loads configuration
- Tests endpoints


## Architecture

- **Backend**: FastAPI (Python 3.10+)
  - SSH connection management via parallel-ssh
  - Metrics collection from AMD GPUs
  - WebSocket for real-time updates
  - Serves static frontend in production

- **Frontend**: React + TypeScript + Vite
  - Real-time dashboard with WebSocket
  - Configuration management UI
  - Responsive design with Tailwind CSS

- **SSH Management**:
  - Jump host support via paramiko
  - Parallel SSH execution to cluster nodes
  - Configurable parallelism (default: 5 concurrent)

## Security Notes

- **Passwords**: Jump host passwords stored in cluster.yaml (development) or memory only (production)
- **SSH Keys**: Uploaded via web UI or copied using refresh-ssh-keys.sh
- **Configuration**: cluster.yaml and nodes.txt excluded from git (.gitignore)
- **Secrets**: Never commit cluster.yaml or nodes.txt to version control

## Troubleshooting

### Container won't start

Check logs:
```bash
sudo docker logs cvs-cluster-monitor
sudo docker exec cvs-cluster-monitor tail -100 /app/backend.log
```

### Container crashes with memory errors

**Symptoms:** `munmap_chunk(): invalid pointer` or libssh2 assertion failures

**Cause:** Concurrent SSH operations causing thread-safety issues

**Fix:** System now has global lock to prevent this. If still occurs, reduce:
- `pool_size: 50 → 30` in cvs_parallel_ssh_reliable.py
- `max_workers: 100 → 50` in host_probe.py

### SSH connection failures

1. Verify SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
2. Test manual SSH: `ssh -i /root/.ssh/id_rsa user@node`
3. Check jump host configuration if using bastion
4. Review backend.log for connection errors:
   ```bash
   sudo docker exec cvs-cluster-monitor grep -i "error connecting" /app/backend.log | tail -20
   ```

### No metrics collected / Stale data

**Symptoms:** Dashboard shows old timestamps, no updates

**Possible causes:**
1. Commands hanging without timeout
2. SSH manager not initialized
3. Metrics collection loop crashed

**Debug steps:**
```bash
# Check if metrics collection is running
sudo docker exec cvs-cluster-monitor grep "Collecting metrics\|Metrics collected" /app/backend.log | tail -10

# Check for errors
sudo docker exec cvs-cluster-monitor grep -i error /app/backend.log | tail -20

# Restart container
sudo docker restart cvs-cluster-monitor
# (Will auto-initialize from config)
```

### Nodes showing as unreachable but are actually reachable

**Cause:** TCP probe timeout too short for slow nodes

**Fix:** Increase TCP probe timeout in `backend/app/core/host_probe.py`:
- Change `timeout: int = 5` to `timeout: int = 10` or `15`

### LLDP neighbors not showing

**Symptoms:** Topology page is empty

**Cause:** lldpd not installed on nodes

**Fix:**
1. Go to Configuration tab
2. Click "Check Status" for lldp
3. Click "Install on All Nodes"
4. Wait for installation to complete
5. Refresh Topology page

### Logs/Software tabs not loading

**Symptoms:** Tabs show loading spinner forever

**Cause:** Commands hanging due to lock contention or missing timeouts

**Check:**
```bash
# See what command is currently running
sudo docker exec cvs-cluster-monitor ps aux | grep python

# Check recent log activity
sudo docker exec cvs-cluster-monitor tail -50 /app/backend.log
```

### Performance degradation with large clusters (500+ nodes)

**Recommendations:**
1. Increase polling interval: `interval: 60 → 300` in cluster.yaml
2. Reduce pool size for stability: `pool_size: 50 → 30`
3. Reduce TCP probe workers: `max_workers: 100 → 50`
4. Increase Docker memory limit in docker-compose.yml

## Development

### Project Structure

```
project-clustermon/
├── backend/
│   ├── app/
│   │   ├── api/          # API endpoints
│   │   ├── collectors/   # Metrics collectors
│   │   ├── core/         # SSH managers (cvs_parallel_ssh_reliable.py, jump_host_pssh.py)
│   │   └── main.py       # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components (Dashboard, GPU Metrics, NIC Metrics, Configuration)
│   │   └── services/     # API client
│   └── package.json
├── config/               # Configuration files (gitignored)
├── Dockerfile            # Multi-stage build
├── docker-compose.yml    # Container orchestration
├── full-rebuild.sh       # Main deployment script
└── refresh-ssh-keys.sh   # SSH key update script
```

## License

MIT License - see LICENSE file for details

## Contributors

Built for monitoring AMD MI300/MI325 GPU clusters.

