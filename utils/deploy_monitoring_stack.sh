#!/bin/bash

# CVS Monitoring Stack Deployment Script
# Deploys Device Metrics Exporter + Prometheus + Grafana

set -e

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Configuration with localhost fallback
CLUSTER_FILE="${1:-./input/cluster_file/local_test_cluster.json}"
MONITORING_CONFIG="${2:-./input/config_file/monitoring/monitoring_config.json}"

# Helper function to resolve placeholders
resolve_with_fallback() {
    local value="$1"
    local fallback="$2"
    
    # If value is empty or contains unresolved placeholder pattern {...}
    if [[ -z "$value" ]] || [[ "$value" =~ ^\{.*\}$ ]]; then
        echo "$fallback"
    else
        # Remove 'v' prefix if exists for version numbers
        echo "${value#v}"
    fi
}

# Read versions from config with fallback
if [ -f "$MONITORING_CONFIG" ] && command -v jq &> /dev/null; then
    PROM_RAW=$(jq -r '.monitoring.prometheus_version // "v2.55.0"' "$MONITORING_CONFIG")
    GRAF_RAW=$(jq -r '.monitoring.grafana_version // "10.4.1"' "$MONITORING_CONFIG")
    EXPO_RAW=$(jq -r '.monitoring.device_metrics_exporter_version // "v1.4.0"' "$MONITORING_CONFIG")
    
    PROMETHEUS_VERSION=$(resolve_with_fallback "$PROM_RAW" "2.55.0")
    GRAFANA_VERSION=$(resolve_with_fallback "$GRAF_RAW" "10.4.1")
    DEVICE_METRICS_VERSION=$(resolve_with_fallback "$EXPO_RAW" "v1.4.0")
else
    # Fallback defaults
    PROMETHEUS_VERSION="2.55.0"
    GRAFANA_VERSION="10.4.1"
    DEVICE_METRICS_VERSION="v1.4.0"
fi

echo "============================================"
echo "CVS Monitoring Stack Deployment"
echo "============================================"
echo ""
echo "Working Directory: $REPO_ROOT"
echo "Cluster File: $CLUSTER_FILE"
echo "Monitoring Config: $MONITORING_CONFIG"
echo "Prometheus Version: $PROMETHEUS_VERSION"
echo "Grafana Version: $GRAFANA_VERSION"
echo "Exporter Version: $DEVICE_METRICS_VERSION"
echo ""

# Step 1: Deploy Device Metrics Exporter on all GPU nodes using pytest
echo "Step 1: Deploying Device Metrics Exporter on all GPU nodes..."
echo "------------------------------------------------------------"
pytest -vv -s ./tests/monitoring/install_device_metrics_exporter.py \
    --cluster_file "$CLUSTER_FILE" \
    --config_file "$MONITORING_CONFIG" \
    --html=/tmp/device_metrics_install_report.html \
    --capture=tee-sys \
    --self-contained-html

if [ $? -ne 0 ]; then
    echo "ERROR: Device Metrics Exporter installation failed!"
    exit 1
fi

echo ""
echo "- Device Metrics Exporter deployed successfully!"
echo ""

# Step 2: Setup Prometheus on management node
echo "Step 2: Setting up Prometheus..."
echo "------------------------------------------------------------"
# Stop existing Prometheus if running
if systemctl is-active --quiet prometheus 2>/dev/null; then
    echo "Stopping existing Prometheus service..."
    sudo systemctl stop prometheus
    sleep 2
fi

sudo pkill -9 prometheus 2>/dev/null || true
sleep 2

if ! command -v prometheus &> /dev/null; then
    echo "Prometheus not found. Installing..."
    
    cd /tmp
    echo "Downloading Prometheus ${PROMETHEUS_VERSION} (~92MB)..."
    wget --progress=bar:force https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz
    echo "Download complete. Extracting..."
    tar xzf prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz
    
    sudo mkdir -p /opt/prometheus
    sudo cp -r prometheus-${PROMETHEUS_VERSION}.linux-amd64/* /opt/prometheus/
    sudo mkdir -p /var/lib/prometheus/data
    
    cd "$REPO_ROOT"
    
    # Copy config from repo
    if [ -f "./monitoring/prometheus/prometheus.yml" ]; then
        sudo cp ./monitoring/prometheus/prometheus.yml /opt/prometheus/
        echo "- Copied prometheus.yml"
    else
        echo "ERROR: prometheus.yml not found at ./monitoring/prometheus/prometheus.yml"
        exit 1
    fi
    
    if [ -f "./monitoring/prometheus/alert_rules.yml" ]; then
        sudo cp ./monitoring/prometheus/alert_rules.yml /opt/prometheus/
        echo "- Copied alert_rules.yml"
    else
        echo "WARNING: alert_rules.yml not found"
    fi
    
    # Create systemd service
    sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOF
[Unit]
Description=Prometheus
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/prometheus/prometheus \
    --config.file=/opt/prometheus/prometheus.yml \
    --storage.tsdb.path=/var/lib/prometheus/data \
    --web.listen-address=0.0.0.0:9090
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable prometheus
    sudo systemctl start prometheus
    
    echo " Prometheus installed and started"
else
    echo " Prometheus already installed"
    # Optionally restart with updated config
    if sudo systemctl is-active --quiet prometheus; then
        echo "  Reloading Prometheus config..."
        sudo systemctl reload prometheus || sudo systemctl restart prometheus
    fi
fi

# Step 3: Setup Grafana
echo ""
echo "Step 3: Setting up Grafana..."
echo "------------------------------------------------------------"

if ! docker ps -a --format '{{.Names}}' | grep -q '^grafana$'; then
    echo "Grafana not found. Installing via Docker..."
    
    # Stop any existing container
    docker stop grafana 2>/dev/null || true
    docker rm grafana 2>/dev/null || true
    
    docker run -d \
        -p 3000:3000 \
        --name grafana \
        --restart unless-stopped \
        -v grafana-storage:/var/lib/grafana \
        grafana/grafana:${GRAFANA_VERSION}
    
    echo " Grafana installed and started"
    echo "  Default credentials: admin/admin"
else
    echo " Grafana container already exists"
    if ! docker ps --format '{{.Names}}' | grep -q '^grafana$'; then
        echo "  Starting Grafana..."
        docker start grafana
    fi
fi

# Step 4: Verify everything is running
echo ""
echo "Step 4: Verifying installation..."
echo "------------------------------------------------------------"

# Wait a bit for services to be ready
sleep 3

# Check Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo " Prometheus is healthy"
else
    echo " Prometheus health check failed"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    echo " Grafana is healthy"
else
    echo " Grafana health check failed (may still be starting...)"
fi

# Check Device Metrics Exporter
if curl -s http://localhost:5000/metrics | head -1 > /dev/null 2>&1; then
    echo " Device Metrics Exporter is responding"
else
    echo " Device Metrics Exporter check failed"
fi

# Check targets if jq available
if command -v jq &> /dev/null; then
    echo ""
    echo "Prometheus Targets:"
    curl -s http://localhost:9090/api/v1/targets 2>/dev/null | \
        jq -r '.data.activeTargets[]? | "\(.labels.instance): \(.health)"' 2>/dev/null || \
        echo "  (Could not retrieve targets)"
fi

echo ""
echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo ""
echo "Access URLs:"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000"
echo "  Exporter:   http://localhost:5000/metrics"
echo ""
echo "Next Steps:"
echo "  1. Log into Grafana (admin/admin)"
echo "  2. Add Prometheus as datasource: http://localhost:9090"
echo "  3. Import dashboards from monitoring/grafana/dashboards/ (if available)"
echo "  4. Run CVS tests with --prometheus-url=http://localhost:9090"
echo ""
