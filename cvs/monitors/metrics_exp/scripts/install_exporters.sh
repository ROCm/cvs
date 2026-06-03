#!/bin/bash
# AMD GPU Fleet Monitoring - Complete Exporter Installation Script
# This script installs all monitoring components on a GPU node
#
# Usage: ./install_exporters.sh [OPTIONS]
#   --loki-url URL     Loki push URL (required)
#   --node-group NAME  Node group name (required)
#   --hostname NAME    Override hostname
#   --gpu-port PORT    AMD exporter port (default: 5000)
#   --node-port PORT   Node exporter port (default: 9100)
#   --skip-gpu         Skip AMD GPU exporter
#   --skip-node        Skip node exporter
#   --skip-promtail    Skip promtail
#   --uninstall        Uninstall all components
#   -h, --help         Show this help

set -e

# Default values
GPU_PORT=5000
NODE_PORT=9100
SKIP_GPU=false
SKIP_NODE=false
SKIP_PROMTAIL=false
UNINSTALL=false
LOKI_URL=""
NODE_GROUP=""
HOSTNAME_OVERRIDE=""

# AMD Device Metrics Exporter image (fully qualified for Podman compatibility)
AMD_EXPORTER_IMAGE="docker.io/rocm/device-metrics-exporter:v1.5.0"

# Node exporter version
NODE_EXPORTER_VERSION="1.7.0"

# Promtail version
PROMTAIL_VERSION="2.9.5"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --loki-url)
            LOKI_URL="$2"
            shift 2
            ;;
        --node-group)
            NODE_GROUP="$2"
            shift 2
            ;;
        --hostname)
            HOSTNAME_OVERRIDE="$2"
            shift 2
            ;;
        --gpu-port)
            GPU_PORT="$2"
            shift 2
            ;;
        --node-port)
            NODE_PORT="$2"
            shift 2
            ;;
        --skip-gpu)
            SKIP_GPU=true
            shift
            ;;
        --skip-node)
            SKIP_NODE=true
            shift
            ;;
        --skip-promtail)
            SKIP_PROMTAIL=true
            shift
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        -h|--help)
            head -20 "$0" | tail -15
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get hostname
if [ -n "$HOSTNAME_OVERRIDE" ]; then
    HOSTNAME="$HOSTNAME_OVERRIDE"
else
    HOSTNAME=$(hostname)
fi

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    ARCH="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
fi

# Uninstall function
do_uninstall() {
    log_info "Uninstalling all monitoring components..."

    # Stop and remove AMD exporter container (both possible names)
    for container_name in device-metrics-exporter amd-metrics-exporter; do
        if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
            log_info "Removing ${container_name} container..."
            docker stop "$container_name" 2>/dev/null || true
            docker rm "$container_name" 2>/dev/null || true
        fi
    done

    # Stop and remove old systemd-based GPU exporter
    if systemctl is-active amd-gpu-exporter &>/dev/null 2>&1; then
        log_info "Removing old amd-gpu-exporter service..."
        systemctl stop amd-gpu-exporter 2>/dev/null || true
        systemctl disable amd-gpu-exporter 2>/dev/null || true
    fi
    rm -f /etc/systemd/system/amd-gpu-exporter.service /usr/local/bin/amd-gpu-exporter.py 2>/dev/null || true

    # Stop and remove node exporter
    if systemctl is-active node_exporter &>/dev/null; then
        log_info "Removing node exporter..."
        systemctl stop node_exporter
        systemctl disable node_exporter
        rm -f /usr/local/bin/node_exporter
        rm -f /etc/systemd/system/node_exporter.service
    fi

    # Stop and remove promtail
    if systemctl is-active promtail &>/dev/null; then
        log_info "Removing promtail..."
        systemctl stop promtail
        systemctl disable promtail
        rm -f /usr/local/bin/promtail
        rm -f /etc/systemd/system/promtail.service
        rm -rf /etc/promtail /var/lib/promtail
    fi

    # Stop and remove RDMA exporter
    if systemctl is-active rdma-exporter &>/dev/null 2>&1; then
        log_info "Removing rdma-exporter..."
        systemctl stop rdma-exporter
        systemctl disable rdma-exporter
    fi
    rm -f /usr/local/bin/rdma-exporter.py /etc/systemd/system/rdma-exporter.service 2>/dev/null || true

    systemctl daemon-reload
    log_info "Uninstall complete"
    exit 0
}

# Run uninstall if requested
if [ "$UNINSTALL" = true ]; then
    do_uninstall
fi

# Validate required arguments
if [ -z "$LOKI_URL" ] || [ -z "$NODE_GROUP" ]; then
    log_error "Missing required arguments: --loki-url and --node-group"
    exit 1
fi

log_info "Starting installation on $HOSTNAME"
log_info "Node Group: $NODE_GROUP"
log_info "Loki URL: $LOKI_URL"

# Check prerequisites
log_info "Checking prerequisites..."

# Check for root
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root"
    exit 1
fi

# Check for Docker (required for AMD exporter)
if ! command -v docker &>/dev/null; then
    log_warn "Docker not found - AMD GPU exporter will be skipped"
    SKIP_GPU=true
fi

# Check for ROCm
if ! command -v rocm-smi &>/dev/null && ! ls /opt/rocm*/bin/rocm-smi &>/dev/null 2>&1; then
    log_warn "ROCm not found - AMD GPU exporter may not work properly"
fi

# Install AMD Device Metrics Exporter
if [ "$SKIP_GPU" = false ]; then
    log_info "Installing AMD Device Metrics Exporter..."

    # === CLEANUP: Stop and remove any existing GPU exporter ===
    log_info "Cleaning up existing GPU exporter installations..."

    # Stop existing container if any (both possible names)
    for container_name in device-metrics-exporter amd-metrics-exporter; do
        if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
            log_info "  Stopping and removing ${container_name}..."
            docker stop "$container_name" 2>/dev/null || true
            docker rm "$container_name" 2>/dev/null || true
        fi
    done

    # Stop old systemd-based exporter if present
    if systemctl is-active amd-gpu-exporter &>/dev/null 2>&1; then
        log_info "  Stopping old systemd-based amd-gpu-exporter..."
        systemctl stop amd-gpu-exporter 2>/dev/null || true
        systemctl disable amd-gpu-exporter 2>/dev/null || true
    fi
    rm -f /etc/systemd/system/amd-gpu-exporter.service 2>/dev/null || true
    rm -f /usr/local/bin/amd-gpu-exporter.py 2>/dev/null || true
    systemctl daemon-reload 2>/dev/null || true

    log_info "  Cleanup complete"

    # Pull latest image
    log_info "Pulling AMD exporter image: $AMD_EXPORTER_IMAGE"
    docker pull "$AMD_EXPORTER_IMAGE"

    # Run container with GPU access
    log_info "Starting AMD exporter container on port $GPU_PORT..."
    docker run -d \
        --name device-metrics-exporter \
        --restart unless-stopped \
        --device=/dev/kfd \
        --device=/dev/dri \
        -v /sys:/sys:ro \
        -p "$GPU_PORT":5000 \
        "$AMD_EXPORTER_IMAGE"

    # Verify it's running
    sleep 3
    if curl -s "http://localhost:$GPU_PORT/metrics" | head -5 | grep -qi "gpu"; then
        log_info "AMD Device Metrics Exporter installed successfully on port $GPU_PORT"
    else
        log_warn "AMD exporter started but metrics endpoint not responding as expected"
    fi
fi

# Install Node Exporter
if [ "$SKIP_NODE" = false ]; then
    log_info "Installing Prometheus Node Exporter v$NODE_EXPORTER_VERSION..."

    # Check if already running
    if systemctl is-active node_exporter &>/dev/null; then
        log_info "Node exporter already running, skipping installation"
    else
        cd /tmp

        # Download
        DOWNLOAD_URL="https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH}.tar.gz"
        log_info "Downloading from $DOWNLOAD_URL"
        curl -LO "$DOWNLOAD_URL"

        # Extract and install
        tar xzf "node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH}.tar.gz"
        cp "node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH}/node_exporter" /usr/local/bin/
        chmod +x /usr/local/bin/node_exporter
        rm -rf "node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH}"*

        # Create systemd service
        cat > /etc/systemd/system/node_exporter.service << EOF
[Unit]
Description=Prometheus Node Exporter
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/node_exporter --web.listen-address=:$NODE_PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

        # Enable and start
        systemctl daemon-reload
        systemctl enable node_exporter
        systemctl start node_exporter

        # Verify
        sleep 2
        if curl -s "http://localhost:$NODE_PORT/metrics" | head -5 | grep -q "node_"; then
            log_info "Node Exporter installed successfully on port $NODE_PORT"
        else
            log_warn "Node exporter started but metrics endpoint not responding"
        fi
    fi
fi

# Install Promtail
if [ "$SKIP_PROMTAIL" = false ]; then
    log_info "Installing Promtail v$PROMTAIL_VERSION..."

    cd /tmp

    # Download
    DOWNLOAD_URL="https://github.com/grafana/loki/releases/download/v${PROMTAIL_VERSION}/promtail-linux-${ARCH}.zip"
    log_info "Downloading from $DOWNLOAD_URL"
    curl -LO "$DOWNLOAD_URL"

    # Extract and install
    unzip -o "promtail-linux-${ARCH}.zip"
    mv "promtail-linux-${ARCH}" /usr/local/bin/promtail
    chmod +x /usr/local/bin/promtail
    rm -f "promtail-linux-${ARCH}.zip"

    # Create directories
    mkdir -p /etc/promtail /var/lib/promtail

    # Create configuration
    cat > /etc/promtail/config.yml << EOF
# Promtail configuration for AMD GPU Fleet Monitoring
# Node: $HOSTNAME
# Node Group: $NODE_GROUP

server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /var/lib/promtail/positions.yaml

clients:
  - url: ${LOKI_URL}/loki/api/v1/push

scrape_configs:
  # Scrape systemd journal (includes kernel/dmesg messages)
  - job_name: journal
    journal:
      max_age: 12h
      path: /var/log/journal
      labels:
        job: systemd-journal
        host: $HOSTNAME
        node_group: $NODE_GROUP
    relabel_configs:
      - source_labels: ['__journal__systemd_unit']
        target_label: 'unit'
      - source_labels: ['__journal__transport']
        target_label: 'transport'
      - source_labels: ['__journal_priority_keyword']
        target_label: 'priority'
    pipeline_stages:
      - match:
          selector: '{transport="kernel"}'
          stages:
            - static_labels:
                log_type: kernel

  # Scrape syslog for additional system messages
  - job_name: syslog
    static_configs:
      - targets:
          - localhost
        labels:
          job: syslog
          host: $HOSTNAME
          node_group: $NODE_GROUP
          __path__: /var/log/syslog

  # Scrape kernel ring buffer
  - job_name: kern
    static_configs:
      - targets:
          - localhost
        labels:
          job: kern
          host: $HOSTNAME
          node_group: $NODE_GROUP
          __path__: /var/log/kern.log
EOF

    # Create systemd service
    cat > /etc/systemd/system/promtail.service << EOF
[Unit]
Description=Promtail Log Collector
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/promtail -config.file=/etc/promtail/config.yml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Enable and start
    systemctl daemon-reload
    systemctl enable promtail
    systemctl restart promtail

    # Verify
    sleep 2
    if systemctl is-active promtail &>/dev/null; then
        log_info "Promtail installed successfully"
    else
        log_warn "Promtail installed but service not active"
    fi
fi

# Install RDMA Exporter
RDMA_PORT=9417
if command -v rdma &>/dev/null; then
    log_info "Installing RDMA Metrics Exporter..."

    # Check if RDMA devices exist
    if rdma link 2>/dev/null | head -1 | grep -q .; then
        # Cleanup existing
        systemctl stop rdma-exporter 2>/dev/null || true
        systemctl disable rdma-exporter 2>/dev/null || true

        # Create the exporter script
        cat > /usr/local/bin/rdma-exporter.py << 'RDMAEOF'
#!/usr/bin/env python3
"""RDMA Metrics Exporter for Prometheus"""
import http.server, json, os, subprocess, threading, time

METRIC_PREFIX = "rdma"
metrics_output = "# No metrics yet\n"
metrics_lock = threading.Lock()

def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    except: return None

def get_hostname():
    return os.environ.get("HOSTNAME_OVERRIDE", os.uname().nodename)

def collect():
    global metrics_output
    hostname = get_hostname()
    lines = []
    links = run_cmd(["rdma", "link", "--json"]) or []
    for link in links:
        ifname, port, netdev = link.get("ifname", "unknown"), str(link.get("port", 0)), link.get("netdev", "")
        phys_state, state = link.get("physical_state", "").lower(), link.get("state", "").lower()
        phys_val = 1 if phys_state in ["link_up", "linkup", "active"] else 0
        state_val = 1 if state in ["active", "up"] else 0
        lines.append(f'{METRIC_PREFIX}_link_physical_state{{hostname="{hostname}",device="{ifname}",port="{port}",netdev="{netdev}"}} {phys_val}')
        lines.append(f'{METRIC_PREFIX}_link_state{{hostname="{hostname}",device="{ifname}",port="{port}",netdev="{netdev}"}} {state_val}')
        lines.append(f'{METRIC_PREFIX}_link_info{{hostname="{hostname}",device="{ifname}",port="{port}",netdev="{netdev}",link_layer="{link.get("link_layer", "unknown")}"}} 1')
    stats = run_cmd(["rdma", "statistic", "--json"]) or []
    for stat in stats:
        ifname, port = stat.get("ifname", "unknown"), str(stat.get("port", 0))
        counters = {}
        for key in ["stats", "hw_stats", "port_stats"]:
            if key in stat:
                v = stat[key]
                if isinstance(v, dict): counters.update(v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, dict): counters.update(x)
        for k, v in stat.items():
            if isinstance(v, (int, float)) and k not in ["ifindex", "port"]: counters[k] = v
        for k, v in counters.items():
            if isinstance(v, (int, float)):
                lines.append(f'{METRIC_PREFIX}_stat_{k.lower().replace("-", "_").replace(".", "_")}{{hostname="{hostname}",device="{ifname}",port="{port}"}} {v}')
    resources = run_cmd(["rdma", "resource", "--json"]) or {}
    for rtype in ["qp", "cm_id", "mr", "pd", "cq", "ctx", "srq"]:
        if rtype in resources and isinstance(resources[rtype], list):
            lines.append(f'{METRIC_PREFIX}_resource_{rtype}_total{{hostname="{hostname}"}} {len(resources[rtype])}')
    lines.append(f'{METRIC_PREFIX}_scrape_success{{hostname="{hostname}"}} {1 if links else 0}')
    with metrics_lock: metrics_output = "\n".join(lines) + "\n"

def collector_loop(interval):
    while True:
        try: collect()
        except Exception as e: print(f"Error: {e}")
        time.sleep(interval)

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == "/metrics":
            with metrics_lock: data = metrics_output
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers(); self.wfile.write(data.encode())
        elif self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"OK")
        else:
            self.send_response(404); self.end_headers()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=9417); p.add_argument("--interval", type=int, default=15)
    args = p.parse_args()
    t = threading.Thread(target=collector_loop, args=(args.interval,), daemon=True); t.start(); time.sleep(1)
    print(f"RDMA Exporter listening on port {args.port}")
    http.server.HTTPServer(("", args.port), Handler).serve_forever()
RDMAEOF
        chmod +x /usr/local/bin/rdma-exporter.py

        # Create systemd service
        cat > /etc/systemd/system/rdma-exporter.service << EOF
[Unit]
Description=RDMA Metrics Exporter for Prometheus
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /usr/local/bin/rdma-exporter.py --port $RDMA_PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

        systemctl daemon-reload
        systemctl enable rdma-exporter
        systemctl start rdma-exporter

        sleep 2
        if systemctl is-active rdma-exporter &>/dev/null; then
            log_info "RDMA Exporter installed successfully on port $RDMA_PORT"
        else
            log_warn "RDMA Exporter installed but service not active"
        fi
    else
        log_warn "No RDMA devices found, skipping RDMA exporter"
    fi
else
    log_warn "rdma command not found, skipping RDMA exporter"
fi

# Summary
echo ""
log_info "=========================================="
log_info "Installation Summary for $HOSTNAME"
log_info "=========================================="

if [ "$SKIP_GPU" = false ]; then
    if docker ps --format '{{.Names}}' | grep -q "device-metrics-exporter"; then
        log_info "AMD Metrics Exporter: RUNNING (port $GPU_PORT)"
    else
        log_error "AMD Metrics Exporter: NOT RUNNING"
    fi
fi

if [ "$SKIP_NODE" = false ]; then
    if systemctl is-active node_exporter &>/dev/null; then
        log_info "Node Exporter: RUNNING (port $NODE_PORT)"
    else
        log_error "Node Exporter: NOT RUNNING"
    fi
fi

if [ "$SKIP_PROMTAIL" = false ]; then
    if systemctl is-active promtail &>/dev/null; then
        log_info "Promtail: RUNNING (pushing to $LOKI_URL)"
    else
        log_error "Promtail: NOT RUNNING"
    fi
fi

if systemctl is-active rdma-exporter &>/dev/null; then
    log_info "RDMA Exporter: RUNNING (port $RDMA_PORT)"
fi

echo ""
log_info "Installation complete!"
