# CVS Cluster Monitor

Real-time GPU cluster monitoring dashboard for AMD MI300/MI325 GPUs. Monitor GPU utilization, temperature, RDMA network statistics, and LLDP topology across your entire cluster through a web-based interface.

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- **Real-time Metrics Collection**: Monitor GPU utilization, temperature, power consumption
- **Network Monitoring**: RDMA statistics, LLDP topology discovery
- **Web Dashboard**: React-based UI with live updates via WebSocket
- **SSH-based Collection**: Direct SSH or jump host/bastion support
- **Configuration UI**: Web-based configuration management
- **Health Tracking**: Node health status with configurable failure thresholds
- **Package Management**: Install required packages (lldpd) across cluster

## Quick Start

### Prerequisites

- Docker and Docker Compose
- SSH access to cluster nodes
- Jump host access (if nodes are behind bastion)
- AMD MI300/MI325 GPUs on cluster nodes

### 1. Clone Repository

```bash
git clone <repository-url>
cd project-clustermon
```

### 2. Build and Start

Run the automated build script:

```bash
bash full-rebuild.sh
```

This script will:
- Build the Docker image (frontend + backend)
- Start the container
- Load configuration from `config/cluster.yaml`
- Start monitoring automatically

### 3. Configure via Web UI

1. Open browser: **http://<your-server-ip>:8005**
2. Go to **Configuration** page
3. **Upload SSH Keys**:
   - Upload jump host access key (if using jump host)
   - Upload node access key (if needed)
4. **Fill in Configuration**:
   - Add cluster node IPs/hostnames
   - Configure SSH username
   - Set up jump host (if applicable)
5. Click **"Save Configuration and Start Monitoring"**

### 4. Monitor Your Cluster

Access the dashboard and view:
- Real-time GPU metrics (utilization, temperature, memory)
- PCIE metrics (speed, bandwidth, errors)
- ECC memory errors
- Network metrics (RDMA, congestion control)
- Topology visualization

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

## Configuration

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
docker-compose logs -f
```

### SSH connection failures

1. Verify SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
2. Test manual SSH: `ssh -i ~/.ssh/id_rsa user@node`
3. Check jump host configuration if using bastion

### No metrics collected

1. Check node reachability via Configuration page
2. Verify GPU tools installed on nodes
3. Check backend logs for errors

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

