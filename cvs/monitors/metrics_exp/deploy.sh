#!/bin/bash
#
# Fleet Manager - Clean Deployment Script
#
# This script performs a clean deployment of the entire stack,
# resetting all data and configurations.
#
# Usage: ./deploy.sh [OPTIONS]
#
#   --keep-data    Keep existing database data (default: reset everything)
#   --with-gpu     Include GPU exporter (use if monitoring server has GPUs)
#   --help         Show this help message
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

KEEP_DATA=false
WITH_GPU=false

show_help() {
    echo "Usage: ./deploy.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --keep-data    Keep existing database data (default: reset everything)"
    echo "  --with-gpu     Include GPU exporter (use if monitoring server has GPUs)"
    echo "  --help         Show this help message"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-data)
            KEEP_DATA=true
            shift
            ;;
        --with-gpu)
            WITH_GPU=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  AMD GPU Fleet Manager - Clean Deployment${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check docker access
if ! docker info &> /dev/null; then
    echo -e "${YELLOW}Docker requires elevated privileges. Re-running with sudo...${NC}"
    exec sudo "$0" "$@"
fi

# Docker compose command
if docker compose version &> /dev/null 2>&1; then
    COMPOSE="docker compose"
else
    COMPOSE="docker-compose"
fi

# Step 1: Stop and remove all existing containers with our names
echo -e "${YELLOW}Step 1: Stopping and removing existing containers...${NC}"

# List of container names we deploy
CONTAINERS="fleet-manager fleet-prometheus fleet-grafana fleet-loki fleet-postgres device-metrics-exporter"

for container in $CONTAINERS; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
        echo -e "  Stopping and removing ${container}..."
        docker stop "$container" 2>/dev/null || true
        docker rm "$container" 2>/dev/null || true
    fi
done

# Also stop compose-managed containers
$COMPOSE down --remove-orphans 2>/dev/null || true

echo -e "${GREEN}  Containers cleaned up${NC}"

# Step 2: Clean up volumes if not keeping data
if [ "$KEEP_DATA" = false ]; then
    echo -e "${YELLOW}Step 2: Removing all volumes (clean slate)...${NC}"
    $COMPOSE down -v 2>/dev/null || true

    # Also remove any orphaned volumes
    docker volume rm $(docker volume ls -q -f name=metrics_proj) 2>/dev/null || true
fi

# Step 3: Clean up Prometheus targets
echo -e "${YELLOW}Step 3: Cleaning Prometheus targets...${NC}"
rm -f server/config/prometheus/targets/*.json
# Keep the .gitkeep file and ensure the fleet-manager container can write here
touch server/config/prometheus/targets/.gitkeep
chmod 777 server/config/prometheus/targets/

# Step 4: Build UI
echo -e "${YELLOW}Step 4: Building UI...${NC}"
cd server/ui
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing npm dependencies...${NC}"
    npm install
fi
rm -rf dist
npm run build

# Copy to static folder
echo -e "${YELLOW}Copying UI build to static folder...${NC}"
rm -rf ../fleet_manager/static
cp -r dist ../fleet_manager/static
cd "$SCRIPT_DIR"

# Step 5: Build and start all services
echo -e "${YELLOW}Step 5: Building and starting services...${NC}"
if [ "$KEEP_DATA" = false ]; then
    export RESET_DATABASE=true
fi

# Build with or without GPU profile
if [ "$WITH_GPU" = true ]; then
    echo -e "${BLUE}  Including GPU exporter (--with-gpu specified)${NC}"
    $COMPOSE --profile gpu up -d --build
else
    $COMPOSE up -d --build
fi

# Step 6: Wait for services to be healthy
echo -e "${YELLOW}Step 6: Waiting for services to start...${NC}"
sleep 15

# Step 7: Verify health
echo -e "${BLUE}Checking service health...${NC}"
echo ""

check_health() {
    local name=$1
    local url=$2
    if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q "200"; then
        echo -e "  $name: ${GREEN}healthy${NC}"
        return 0
    else
        echo -e "  $name: ${RED}unhealthy${NC}"
        return 1
    fi
}

check_health "Fleet Manager" "http://localhost:30080/health"
check_health "Prometheus" "http://localhost:30090/-/healthy"
check_health "Grafana" "http://localhost:30030/api/health"
check_health "Loki" "http://localhost:30100/ready"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${BLUE}Access URLs:${NC}"
echo "  Fleet Manager UI: http://localhost:30080"
echo "  Grafana:          http://localhost:30030 (admin/admin)"
echo "  Prometheus:       http://localhost:30090"
echo "  Loki:             http://localhost:30100"
echo ""
echo -e "${BLUE}SSH Tunnel (if accessing remotely):${NC}"
echo "  ssh -L 30080:localhost:30080 -L 30030:localhost:30030 -L 30090:localhost:30090 -L 30100:localhost:30100 user@server"
echo ""
echo -e "${BLUE}Quick Start:${NC}"
echo "  1. Open http://localhost:30080 in your browser"
echo "  2. Go to 'Monitoring Servers' and add your monitoring server"
echo "     - Set Grafana port: 30030, Prometheus port: 30090, Loki port: 30100"
echo "  3. Go to 'Node Groups' and create a group with your GPU node IPs"
echo "  4. Upload SSH key and click 'Verify Connectivity'"
echo "  5. Click 'Install' to deploy exporters to the nodes"
echo "  6. View metrics in Grafana dashboards"
echo ""
