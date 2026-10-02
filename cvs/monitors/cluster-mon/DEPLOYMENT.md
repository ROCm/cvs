# Deployment Guide

CVS Cluster Monitor ships as **one Go binary** plus the pre-built React UI
inside a Docker image. Redis is a sibling container for RCCL history.

## Docker (recommended)

The image does **not** include Python. Build stages: Go 1.25, Node 18 (frontend
only), Alpine runtime.

### Prerequisites

- Docker and Docker Compose v2
- SSH access to cluster nodes (direct or jump host)
- A private key the nodes already trust (upload in the UI, or copy with
  `setup-ssh-keys.sh` / `full-rebuild.sh`)

### 1. Configuration

From `cvs/monitors/cluster-mon`:

```bash
cp config/cluster.yaml.example config/cluster.yaml
cp config/nodes.txt.example config/nodes.txt
# edit cluster.yaml (username, key_file) and nodes.txt
```

Optional `.env`:

```
CLUSTER_MON_PORT=8005
POLLING__INTERVAL=60
REDIS_PASSWORD=cvs_cluster_mon
# CLUSTER_MON_API_TOKEN=   # leave unset unless a proxy injects the header
# CORS_ORIGINS=http://localhost:5173
```

### 2. Build and run

```bash
./full-rebuild.sh
# or
sudo docker compose up -d --build
```

- Host port **8005** → container **8001** (`CLUSTER_MON_PORT` overrides the host side)
- Volumes: `./config:/app/config`, named volume `cluster_mon_ssh:/root/.ssh`
- Redis data: named volume `redis_data`

Do **not** bind-mount host `~/.ssh`. Do **not** `docker compose down -v` unless
you intend to wipe uploaded keys and RCCL history.

### 3. Dashboard

**http://\<host\>:8005**

```bash
curl -s http://localhost:8005/health
# {"status":"healthy","ssh_manager":true,"collecting":true,"clients":0}
```

If `nodes.txt` and a key are already present, the process probes SSH on start
and begins collecting. Otherwise use **Configuration** → upload key → Save →
Reload.

### Compose vs `docker run`

Prefer compose (Redis + volume + env). A lone container without Redis still
serves metrics; RCCL Timeline history will not survive restart.

```bash
docker build -t cvs-cluster-monitor .
docker run -d --name cvs-cluster-monitor \
  -p 8005:8001 \
  -v "$(pwd)/config:/app/config" \
  -v cluster_mon_ssh:/root/.ssh \
  -e POLLING__INTERVAL=60 \
  -e CLUSTER_MONITOR_HOME=/app \
  cvs-cluster-monitor
```

## Bare metal (development)

```bash
# Go API (no UI unless you pass -static)
make run LISTEN=:8001 CONFIG_DIR=./config

# UI
cd frontend && npm ci && npm run build
make run LISTEN=:8001 CONFIG_DIR=./config STATIC_DIR=frontend/dist
```

Point Redis at `STORAGE__REDIS__URL` / `STORAGE__REDIS__PASSWORD` if you want
RCCL history. Default yaml is `redis://localhost:6379`.

## Production

### Network

Do not expose the UI port on an open campus LAN. Prefer SSH tunnel, VPN, or
host firewall. Example:

```bash
ssh -L 8005:127.0.0.1:8005 user@monitor-host
```

### TLS and auth

The binary speaks HTTP. Terminate TLS on a reverse proxy. Optional shared
secret: set `CLUSTER_MON_API_TOKEN` and have the proxy add
`Authorization: Bearer <token>` or `X-API-Token` on `/api/ssh-keys` and
`/api/config`. The stock React UI does not send that header.

### Nginx (WebSocket)

Proxy **8005** (published host port) or **8001** if you use `network_mode: host`.

```nginx
server {
    listen 443 ssl;
    server_name cluster-monitor.example.com;

    location / {
        proxy_pass http://127.0.0.1:8005;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /ws/ {
        proxy_pass http://127.0.0.1:8005;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

### Systemd (bare metal)

```ini
[Unit]
Description=CVS Cluster Monitor
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/cluster-mon
ExecStart=/usr/local/bin/cluster-mon -listen :8001 -config-dir /opt/cluster-mon/config -static /opt/cluster-mon/static
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Logs and health

```bash
sudo docker compose logs -f cvs-cluster-monitor
sudo docker compose logs --tail=100 cvs-cluster-monitor

curl -s http://localhost:8005/health
```

Useful log lines: `pssh_reprobe_failed`, `initial_probe_done`,
`metrics_collected`, `rccl_state_transition`.

## Scaling

```bash
# small
POLLING__INTERVAL=30
# medium
POLLING__INTERVAL=60
# large
POLLING__INTERVAL=120
```

SSH is a persistent pool (not one process per command). Jump-host changes
rebuild the pool; node-list-only reloads do not.

## Upgrade

```bash
git pull
sudo docker compose up -d --build
```

Named volumes keep Redis AOF and `/root/.ssh`. Confirm with
`sudo docker logs cvs-cluster-monitor --tail 50`.

## Backup

```bash
tar -czf cluster-mon-config-$(date +%Y%m%d).tar.gz config/
# keys live in Docker volume cluster_mon_ssh, not in ./config
```

## Troubleshooting

See [README.md](README.md). Common deploy mistakes:

- `docker compose up` without `--build` after a source change (binary is in the image)
- `docker compose port … 8005` — the **container** port is `8001`
- Binding `~/.ssh` from the host
- Setting `CLUSTER_MON_API_TOKEN` without a proxy that injects it
