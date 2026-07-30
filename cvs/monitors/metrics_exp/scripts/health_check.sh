#!/bin/bash
# AMD GPU Fleet Monitoring - Health Check Script
# Checks the status of all monitoring components on a node
#
# Usage: ./health_check.sh [OPTIONS]
#   --gpu-port PORT    AMD exporter port (default: 5000)
#   --node-port PORT   Node exporter port (default: 9100)
#   --json             Output in JSON format
#   -h, --help         Show this help

set -e

GPU_PORT=5000
NODE_PORT=9100
JSON_OUTPUT=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-port)
            GPU_PORT="$2"
            shift 2
            ;;
        --node-port)
            NODE_PORT="$2"
            shift 2
            ;;
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        -h|--help)
            head -12 "$0" | tail -9
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check functions
check_amd_exporter() {
    local status="down"
    local message=""
    local metrics_count=0

    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "amd-metrics-exporter"; then
        local response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$GPU_PORT/metrics" 2>/dev/null || echo "000")
        if [ "$response" = "200" ]; then
            status="up"
            metrics_count=$(curl -s "http://localhost:$GPU_PORT/metrics" 2>/dev/null | grep -c "^gpu_" || echo "0")
            message="Serving $metrics_count GPU metrics"
        else
            status="error"
            message="Container running but HTTP $response"
        fi
    else
        message="Container not running"
    fi

    echo "$status|$message|$metrics_count"
}

check_node_exporter() {
    local status="down"
    local message=""
    local metrics_count=0

    if systemctl is-active node_exporter &>/dev/null; then
        local response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$NODE_PORT/metrics" 2>/dev/null || echo "000")
        if [ "$response" = "200" ]; then
            status="up"
            metrics_count=$(curl -s "http://localhost:$NODE_PORT/metrics" 2>/dev/null | grep -c "^node_" || echo "0")
            message="Serving $metrics_count node metrics"
        else
            status="error"
            message="Service running but HTTP $response"
        fi
    else
        message="Service not running"
    fi

    echo "$status|$message|$metrics_count"
}

check_promtail() {
    local status="down"
    local message=""

    if systemctl is-active promtail &>/dev/null; then
        status="up"
        # Check if positions file exists and is recent
        if [ -f /var/lib/promtail/positions.yaml ]; then
            local age=$(( $(date +%s) - $(stat -c %Y /var/lib/promtail/positions.yaml) ))
            if [ $age -lt 300 ]; then
                message="Active, positions updated ${age}s ago"
            else
                message="Active but positions stale (${age}s old)"
                status="warning"
            fi
        else
            message="Active, no positions file yet"
        fi
    else
        message="Service not running"
    fi

    echo "$status|$message"
}

check_gpu_info() {
    local gpu_count=0
    local gpu_model="Unknown"

    if command -v rocm-smi &>/dev/null; then
        gpu_count=$(rocm-smi --showproductname 2>/dev/null | grep -c "GPU" || echo "0")
        gpu_model=$(rocm-smi --showproductname 2>/dev/null | grep "GPU" | head -1 | awk '{print $NF}' || echo "Unknown")
    elif ls /opt/rocm*/bin/rocm-smi &>/dev/null 2>&1; then
        local rocm_smi=$(ls /opt/rocm*/bin/rocm-smi 2>/dev/null | head -1)
        gpu_count=$($rocm_smi --showproductname 2>/dev/null | grep -c "GPU" || echo "0")
        gpu_model=$($rocm_smi --showproductname 2>/dev/null | grep "GPU" | head -1 | awk '{print $NF}' || echo "Unknown")
    fi

    echo "$gpu_count|$gpu_model"
}

# Run checks
AMD_RESULT=$(check_amd_exporter)
NODE_RESULT=$(check_node_exporter)
PROMTAIL_RESULT=$(check_promtail)
GPU_INFO=$(check_gpu_info)

# Parse results
IFS='|' read -r AMD_STATUS AMD_MSG AMD_METRICS <<< "$AMD_RESULT"
IFS='|' read -r NODE_STATUS NODE_MSG NODE_METRICS <<< "$NODE_RESULT"
IFS='|' read -r PROMTAIL_STATUS PROMTAIL_MSG <<< "$PROMTAIL_RESULT"
IFS='|' read -r GPU_COUNT GPU_MODEL <<< "$GPU_INFO"

# Output
if [ "$JSON_OUTPUT" = true ]; then
    cat << EOF
{
  "hostname": "$(hostname)",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "gpu_info": {
    "count": $GPU_COUNT,
    "model": "$GPU_MODEL"
  },
  "components": {
    "amd_metrics_exporter": {
      "status": "$AMD_STATUS",
      "port": $GPU_PORT,
      "message": "$AMD_MSG",
      "metrics_count": $AMD_METRICS
    },
    "node_exporter": {
      "status": "$NODE_STATUS",
      "port": $NODE_PORT,
      "message": "$NODE_MSG",
      "metrics_count": $NODE_METRICS
    },
    "promtail": {
      "status": "$PROMTAIL_STATUS",
      "message": "$PROMTAIL_MSG"
    }
  }
}
EOF
else
    echo "=========================================="
    echo "Health Check: $(hostname)"
    echo "Timestamp: $(date)"
    echo "=========================================="
    echo ""
    echo "GPU Info:"
    echo "  Count: $GPU_COUNT"
    echo "  Model: $GPU_MODEL"
    echo ""
    echo "Components:"

    # AMD Exporter
    if [ "$AMD_STATUS" = "up" ]; then
        echo -e "  AMD Metrics Exporter: ${GREEN}UP${NC} (port $GPU_PORT)"
    elif [ "$AMD_STATUS" = "error" ]; then
        echo -e "  AMD Metrics Exporter: ${YELLOW}ERROR${NC}"
    else
        echo -e "  AMD Metrics Exporter: ${RED}DOWN${NC}"
    fi
    echo "    $AMD_MSG"

    # Node Exporter
    if [ "$NODE_STATUS" = "up" ]; then
        echo -e "  Node Exporter: ${GREEN}UP${NC} (port $NODE_PORT)"
    elif [ "$NODE_STATUS" = "error" ]; then
        echo -e "  Node Exporter: ${YELLOW}ERROR${NC}"
    else
        echo -e "  Node Exporter: ${RED}DOWN${NC}"
    fi
    echo "    $NODE_MSG"

    # Promtail
    if [ "$PROMTAIL_STATUS" = "up" ]; then
        echo -e "  Promtail: ${GREEN}UP${NC}"
    elif [ "$PROMTAIL_STATUS" = "warning" ]; then
        echo -e "  Promtail: ${YELLOW}WARNING${NC}"
    else
        echo -e "  Promtail: ${RED}DOWN${NC}"
    fi
    echo "    $PROMTAIL_MSG"

    echo ""
fi

# Exit code based on status
if [ "$AMD_STATUS" = "up" ] && [ "$NODE_STATUS" = "up" ] && [ "$PROMTAIL_STATUS" = "up" ]; then
    exit 0
else
    exit 1
fi
