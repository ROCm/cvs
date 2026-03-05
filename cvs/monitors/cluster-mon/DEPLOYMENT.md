# Deployment Guide

This guide covers different deployment options for CVS Cluster Monitor.

## Docker Deployment (Recommended)

The Docker container packages **all** frontend and backend dependencies. Users don't need to install Node.js, Python, or any dependencies manually.

### What's Included in the Docker Image

- ✅ Python 3.10 with all backend dependencies
- ✅ Pre-built React frontend (production build)
- ✅ CVS library for GPU metrics
- ✅ All Node.js dependencies (build-time only)
- ✅ SSH client for cluster access

### Prerequisites

- Docker and Docker Compose installed on the host machine
- SSH keys for cluster access (mounted as volume)
- Network access to cluster nodes

### Step 1: Prepare Configuration

```bash
# Clone repository
git clone <repository-url>
cd project-clustermon

# Copy configuration templates
cp config/cluster.yaml.example config/cluster.yaml
cp config/nodes.txt.example config/nodes.txt

# Edit configuration files
nano config/cluster.yaml  # Add your SSH credentials
nano config/nodes.txt     # Add your cluster nodes
```

### Step 2: Build and Run

```bash
# Build and start container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Step 3: Access Dashboard

Open browser to **http://localhost:8005**

### Docker Compose Configuration

The `docker-compose.yml` file handles:

- **Port mapping**: 8001 (host) → 8001 (container)
- **Volume mounts**:
  - `./config:/app/config` - Configuration files
  - `~/.ssh:/root/.ssh:ro` - SSH keys (read-only)
- **Environment variables**: Polling interval, debug mode, etc.
- **Health checks**: Automatic container health monitoring
- **Restart policy**: Automatically restart on failure

### Custom Docker Run

Without docker-compose:

```bash
docker build -t cvs-cluster-monitor .

docker run -d \
  --name cvs-cluster-monitor \
  -p 8001:8001 \
  -v $(pwd)/config:/app/config \
  -v ~/.ssh:/root/.ssh:ro \
  -e POLLING__INTERVAL=60 \
  -e POLLING__FAILURE_THRESHOLD=5 \
  cvs-cluster-monitor
```

## Bare Metal Deployment

For development or if Docker is not available.

### Prerequisites

- Python 3.10+
- Node.js 18+
- SSH access to cluster nodes

### Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp ../config/cluster.yaml.example ../config/cluster.yaml
cp ../config/nodes.txt.example ../config/nodes.txt
# Edit configuration files

# Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend Setup (Development)

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Run development server
npm run dev
```

Frontend will be available at **http://localhost:5173**

### Frontend Build (Production)

```bash
cd frontend
npm install
npm run build

# Built files will be in frontend/dist/
# Copy to backend/static/ to serve from FastAPI
cp -r dist/* ../backend/static/
```

## Production Deployment Considerations

### Security

1. **SSH Keys**: Use read-only mounts
   ```yaml
   volumes:
     - ~/.ssh:/root/.ssh:ro
   ```

2. **Passwords**: Configure only via web UI (stored in memory only)

3. **Network**: Restrict port 8001 with firewall if needed
   ```bash
   # Example: Allow only from specific IP
   iptables -A INPUT -p tcp --dport 8001 -s 192.168.1.0/24 -j ACCEPT
   iptables -A INPUT -p tcp --dport 8001 -j DROP
   ```

### Reverse Proxy (Nginx)

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name cluster-monitor.example.com;

    location / {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

### Systemd Service (Bare Metal)

Create `/etc/systemd/system/cvs-cluster-monitor.service`:

```ini
[Unit]
Description=CVS Cluster Monitor
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/project-clustermon/backend
Environment="PATH=/path/to/project-clustermon/backend/venv/bin"
ExecStart=/path/to/project-clustermon/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cvs-cluster-monitor
sudo systemctl start cvs-cluster-monitor
sudo systemctl status cvs-cluster-monitor
```

## Monitoring and Logs

### Docker Logs

```bash
# View logs
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100

# Export logs
docker-compose logs > cluster-monitor.log
```

### Health Check

```bash
# Check container health
docker ps

# Manual health check
curl http://localhost:8001/health
```

### Metrics

The health endpoint returns:

```json
{
  "status": "healthy",
  "ssh_manager": true,
  "collecting": true,
  "clients": 2
}
```

## Scaling and Performance

### Resource Limits (Docker)

Add to `docker-compose.yml`:

```yaml
services:
  cvs-cluster-monitor:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### Polling Configuration

Adjust based on cluster size:

```bash
# Small cluster (< 10 nodes)
POLLING__INTERVAL=30

# Medium cluster (10-50 nodes)
POLLING__INTERVAL=60

# Large cluster (50+ nodes)
POLLING__INTERVAL=120
```

## Upgrading

### Docker Deployment

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Bare Metal

```bash
# Pull latest code
git pull

# Update backend
cd backend
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart cvs-cluster-monitor

# Update frontend (if needed)
cd ../frontend
npm install
npm run build
```

## Backup

### Configuration Backup

```bash
# Backup configuration
tar -czf cluster-monitor-config-$(date +%Y%m%d).tar.gz config/

# Restore
tar -xzf cluster-monitor-config-20260224.tar.gz
```

## Troubleshooting

See README.md troubleshooting section for common issues.

### Container Restart Loop

```bash
# Check logs
docker-compose logs

# Common issues:
# - Missing config files
# - Invalid YAML syntax
# - SSH key permissions
```

### High Memory Usage

```bash
# Check memory usage
docker stats cvs-cluster-monitor

# Reduce polling frequency or node count
```
