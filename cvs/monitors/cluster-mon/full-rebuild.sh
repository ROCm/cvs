#!/bin/bash
# Rebuild and start the Go cluster-mon image + Redis.
# Named volumes (SSH keys, Redis) are kept. Do not pass -v to compose down.

set -euo pipefail

echo "========================================="
echo "CVS Cluster Monitor - Full Rebuild (Go)"
echo "========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONTAINER_NAME="cvs-cluster-monitor"
SERVICE_NAME="cvs-cluster-monitor"
ACTUAL_USER="${SUDO_USER:-$USER}"

echo "Project directory: $SCRIPT_DIR"
echo "Running as user:   $ACTUAL_USER"
echo ""

echo "Step 1: Stopping compose project (volumes kept)..."
sudo docker compose down --remove-orphans 2>/dev/null || true
sudo docker rm -f cvs-cluster-monitor cvs-cluster-monitor-test 2>/dev/null || true
echo "✓ Stopped"
echo ""

echo "Step 2: Building image..."
BUILD_START=$(date +%s)
sudo docker compose build --no-cache
BUILD_END=$(date +%s)
echo "✓ Image built in $((BUILD_END - BUILD_START))s"
echo ""

echo "Step 3: Configuration..."
if [ ! -f "config/cluster.yaml" ]; then
    if [ -f "config/cluster.yaml.example" ]; then
        cp config/cluster.yaml.example config/cluster.yaml
        sed -i "s/your_username/$ACTUAL_USER/g" config/cluster.yaml
        echo "✓ cluster.yaml created (username: $ACTUAL_USER)"
    else
        echo "⚠️  config/cluster.yaml.example missing"
    fi
else
    echo "✓ cluster.yaml already exists"
fi

if [ ! -f "config/nodes.txt" ]; then
    if [ -f "config/nodes.txt.example" ]; then
        cp config/nodes.txt.example config/nodes.txt
    else
        printf '# Add cluster nodes (one per line)\n' > config/nodes.txt
    fi
    echo "✓ nodes.txt created"
else
    echo "✓ nodes.txt already exists"
fi
echo ""

echo "Step 4: Starting Redis + monitor..."
sudo docker compose up -d
echo "✓ Started"
echo ""

echo "Step 5: Waiting for /health..."
HOST_PORT="$(sudo docker compose port "$SERVICE_NAME" 8001 2>/dev/null | awk -F: '{print $NF}' || true)"
if [ -z "$HOST_PORT" ]; then
    HOST_PORT="${CLUSTER_MON_PORT:-8005}"
fi
HEALTH_URL="http://127.0.0.1:${HOST_PORT}/health"
ready=0
for _ in $(seq 1 30); do
    if curl -sf -m 2 "$HEALTH_URL" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" -eq 1 ]; then
    echo "✓ $HEALTH_URL"
else
    echo "⚠️  $HEALTH_URL not ready yet — check: sudo docker compose logs cvs-cluster-monitor"
fi
echo ""

echo "Step 6: SSH keys (host copy into named volume)..."
if bash setup-ssh-keys.sh; then
    echo "✓ Keys copied"
else
    echo "⚠️  No host key copied. Upload a key in the Configuration UI, then Reload."
fi
echo ""

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ -z "$HOST_IP" ]; then
    HOST_IP="localhost"
fi
API_URL="http://${HOST_IP}:${HOST_PORT}"

echo "Step 7: Reload configuration..."
sleep 2
RELOAD_RESPONSE="$(curl -m 30 -s -X POST "$API_URL/api/config/reload" 2>&1 || true)"
if echo "$RELOAD_RESPONSE" | grep -q "success"; then
    echo "✓ $RELOAD_RESPONSE"
else
    echo "⚠️  Reload: ${RELOAD_RESPONSE:-no response}"
    echo "   Configure in the UI and click Save / Reload if jump host or keys still need setup."
fi
echo ""

echo "========================================="
echo "Verification"
echo "========================================="
echo ""
echo "Containers:"
sudo docker compose ps
echo ""
echo "SSH keys in container:"
sudo docker exec "$CONTAINER_NAME" ls -la /root/.ssh/ 2>/dev/null || echo "  (container not running)"
echo ""
echo "Health ($API_URL/health):"
curl -m 10 -s "$API_URL/health" 2>/dev/null || echo "  no response"
echo ""
echo ""
echo "========================================="
echo "✓ Rebuild complete"
echo "========================================="
echo ""
echo "Dashboard: $API_URL"
echo ""
echo "  Logs:    sudo docker compose logs -f cvs-cluster-monitor"
echo "  Stop:    sudo docker compose down          # keeps key + Redis volumes"
echo "  Restart: sudo docker compose up -d --build"
echo ""
