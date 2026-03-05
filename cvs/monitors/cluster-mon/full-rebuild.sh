#!/bin/bash
# Complete rebuild and setup for CVS Cluster Monitor

set -e

echo "========================================="
echo "CVS Cluster Monitor - Full Rebuild"
echo "========================================="
echo ""

# Get script directory (works regardless of where it's called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONTAINER_NAME="cvs-cluster-monitor"

# Detect actual user (not root when using sudo)
ACTUAL_USER="${SUDO_USER:-$USER}"

echo "Project directory: $SCRIPT_DIR"
echo "Running as user: $ACTUAL_USER"
echo ""

# Step 1: Stop and clean up all containers
echo "Step 1: Stopping all containers..."
sudo docker stop cvs-cluster-monitor cvs-cluster-monitor-test 2>/dev/null || true
sudo docker rm cvs-cluster-monitor cvs-cluster-monitor-test 2>/dev/null || true
echo "✓ Containers stopped and removed"
echo ""

# Step 2: Remove old images
echo "Step 2: Removing old Docker images..."
sudo docker rmi cvs-cluster-monitor:test project-clustermon-cvs-cluster-monitor 2>/dev/null || true
echo "✓ Old images removed"
echo ""

# Step 3: Build fresh Docker image
echo "Step 3: Building Docker image (this may take 2-3 minutes)..."
BUILD_START=$(date +%s)
sudo docker compose build --no-cache
BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))
echo "✓ Docker image built successfully in ${BUILD_TIME}s"
echo ""

# Step 4: Initialize configuration with current user
echo "Step 4: Initializing configuration..."
if [ ! -f "config/cluster.yaml" ]; then
    echo "Creating cluster.yaml from template..."
    cp config/cluster.yaml.example config/cluster.yaml
    # Update username to actual user
    sed -i "s/your_username/$ACTUAL_USER/g" config/cluster.yaml
    echo "✓ cluster.yaml created with username: $ACTUAL_USER"
else
    echo "✓ cluster.yaml already exists"
fi

if [ ! -f "config/nodes.txt" ]; then
    echo "Creating empty nodes.txt..."
    echo "# Add your cluster nodes here (one per line)" > config/nodes.txt
    echo "✓ nodes.txt created"
else
    echo "✓ nodes.txt already exists"
fi
echo ""

# Step 5: Start container
echo "Step 5: Starting container..."
sudo docker compose up -d
echo "✓ Container started"
echo ""

# Step 6: Wait for container to be ready
echo "Step 6: Waiting for container to initialize..."
sleep 5
echo "✓ Container initialized"
echo ""

# Step 7: Copy SSH keys
echo "Step 7: Setting up SSH keys..."
bash setup-ssh-keys.sh
echo ""

# Step 8: Detect API URL (skip auto-reload to avoid blocking)
echo "Step 8: Detecting API URL..."

# Now detect port (after container is running)
HOST_PORT=$(sudo docker compose port "$CONTAINER_NAME" 8005 2>/dev/null | cut -d: -f2)
if [ -z "$HOST_PORT" ]; then
    HOST_PORT="8005"  # Fallback default
fi

HOST_IP=$(hostname -I | awk '{print $1}')
if [ -z "$HOST_IP" ]; then
    HOST_IP="localhost"
fi

API_URL="http://${HOST_IP}:${HOST_PORT}"
echo "  API URL: $API_URL"
echo ""

# Trigger config reload automatically
echo "Triggering configuration reload..."
sleep 3
RELOAD_RESPONSE=$(curl -m 30 -s -X POST "$API_URL/api/config/reload" 2>&1 || echo "TIMEOUT")

if echo "$RELOAD_RESPONSE" | grep -q "success"; then
    echo "✓ Configuration loaded and monitoring started automatically"
else
    echo "⚠️  Reload had issues (this is OK if jump host needs setup)"
    echo "   Configure via web UI and click 'Save Configuration and Start Monitoring'"
fi
echo ""

# Step 9: Verify everything is working
echo "========================================="
echo "Verification"
echo "========================================="
echo ""

echo "Container status:"
sudo docker ps | grep cvs-cluster-monitor
echo ""

echo "SSH keys in container:"
sudo docker exec cvs-cluster-monitor ls -la /root/.ssh/
echo ""

echo "Testing health endpoint ($API_URL/health):"
HEALTH_RESPONSE=$(curl -m 10 -s "$API_URL/health" 2>&1)
if [ -n "$HEALTH_RESPONSE" ]; then
    echo "$HEALTH_RESPONSE"
else
    echo "  No response or timeout"
fi
echo ""

echo "Testing API endpoint ($API_URL/api/config/current):"
API_RESPONSE=$(curl -m 10 -s "$API_URL/api/config/current" 2>&1)
if [ -n "$API_RESPONSE" ]; then
    echo "$API_RESPONSE" | head -10
else
    echo "  No response or timeout"
fi
echo ""

echo "========================================="
echo "✓ Rebuild Complete!"
echo "========================================="
echo ""
echo "Access the dashboard at: $API_URL"
echo ""
echo "Useful commands:"
echo "  View logs:    sudo docker compose logs -f"
echo "  Stop:         sudo docker compose down"
echo "  Restart:      sudo docker compose restart"
echo "  Shell access: sudo docker exec -it cvs-cluster-monitor /bin/bash"
echo ""
echo "To monitor jump host connections:"
echo "  sudo docker compose logs -f | grep -i 'jump\\|Connected'"
echo ""
