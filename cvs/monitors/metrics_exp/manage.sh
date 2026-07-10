#!/bin/bash
#
# Fleet Manager - Service Management Script
#
# Usage: ./manage.sh [command] [options]
#
# Commands:
#   start       Start all services
#   stop        Stop all services
#   restart     Restart all services (no rebuild)
#   rebuild     Rebuild and restart services
#   logs        View logs (use -f for follow)
#   status      Show service status
#   ui          Build the UI only
#   clean       Stop and remove all containers, volumes
#
# Options:
#   --service   Specify a single service (e.g., --service fleet-manager)
#   --no-ui     Skip UI build during rebuild
#   -f          Follow logs (for logs command)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Save original arguments for sudo re-execution
ORIGINAL_ARGS=("$@")

# Prevent infinite sudo loop
if [ "${MANAGE_SH_SUDOED:-}" = "1" ]; then
    ALREADY_SUDOED="true"
else
    ALREADY_SUDOED="false"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Docker compose command (handle both old and new versions)
if command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE="docker compose"
else
    echo -e "${RED}Error: docker-compose or docker compose not found${NC}"
    exit 1
fi

# Check if running with sudo for docker
check_docker_access() {
    if ! docker info &> /dev/null; then
        if [ "$ALREADY_SUDOED" = "true" ]; then
            echo -e "${RED}Error: Docker is not accessible even with sudo.${NC}"
            echo -e "${RED}Please check if Docker daemon is running: sudo systemctl status docker${NC}"
            exit 1
        fi
        echo -e "${YELLOW}Docker requires elevated privileges. Re-running with sudo...${NC}"
        exec sudo MANAGE_SH_SUDOED=1 "$0" "${ORIGINAL_ARGS[@]}"
    fi
}

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  AMD GPU Fleet Manager - Service Management${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

print_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services (no rebuild)"
    echo "  rebuild     Rebuild and restart services"
    echo "  logs        View logs (use -f for follow)"
    echo "  status      Show service status"
    echo "  ui          Build the UI only"
    echo "  clean       Stop and remove all containers, volumes"
    echo "  db-shell    Open PostgreSQL shell"
    echo ""
    echo "Options:"
    echo "  --service NAME   Target a specific service (fleet-manager, prometheus, grafana, loki, postgres)"
    echo "  --no-ui          Skip UI build during rebuild"
    echo "  --no-cache       Force rebuild without Docker cache"
    echo "  -f, --follow     Follow logs (for logs command)"
    echo "  -n, --tail NUM   Number of log lines to show (default: 100)"
    echo ""
    echo "Examples:"
    echo "  $0 start                        # Start all services"
    echo "  $0 rebuild                      # Rebuild and restart everything"
    echo "  $0 rebuild --service fleet-manager   # Rebuild only fleet-manager"
    echo "  $0 logs -f --service fleet-manager   # Follow fleet-manager logs"
    echo "  $0 restart --service prometheus      # Restart prometheus only"
}

build_ui() {
    echo -e "${YELLOW}Building UI...${NC}"

    if [ ! -d "server/ui" ]; then
        echo -e "${RED}Error: UI directory not found at server/ui${NC}"
        return 1
    fi

    cd server/ui

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing npm dependencies...${NC}"
        npm install
    fi

    npm run build

    # Copy build to static folder
    if [ -d "dist" ]; then
        echo -e "${YELLOW}Copying build to fleet_manager/static...${NC}"
        rm -rf ../fleet_manager/static
        cp -r dist ../fleet_manager/static
        echo -e "${GREEN}UI build complete${NC}"
    else
        echo -e "${RED}Error: Build output not found${NC}"
        return 1
    fi

    cd "$SCRIPT_DIR"
}

cmd_start() {
    local service="$1"

    echo -e "${GREEN}Starting services...${NC}"

    if [ -n "$service" ]; then
        $COMPOSE up -d "$service"
    else
        $COMPOSE up -d
    fi

    echo -e "${GREEN}Services started${NC}"
    cmd_status
}

cmd_stop() {
    local service="$1"

    echo -e "${YELLOW}Stopping services...${NC}"

    if [ -n "$service" ]; then
        $COMPOSE stop "$service"
    else
        $COMPOSE down
    fi

    echo -e "${GREEN}Services stopped${NC}"
}

cmd_restart() {
    local service="$1"

    echo -e "${YELLOW}Restarting services...${NC}"

    if [ -n "$service" ]; then
        $COMPOSE restart "$service"
    else
        $COMPOSE restart
    fi

    echo -e "${GREEN}Services restarted${NC}"
    cmd_status
}

cmd_rebuild() {
    local service="$1"
    local skip_ui="$2"
    local no_cache="$3"

    echo -e "${YELLOW}Rebuilding services...${NC}"

    # Build UI first (unless skipped or targeting non-fleet-manager service)
    if [ "$skip_ui" != "true" ]; then
        if [ -z "$service" ] || [ "$service" = "fleet-manager" ]; then
            build_ui || echo -e "${YELLOW}Warning: UI build failed, continuing with backend...${NC}"
        fi
    fi

    # Build with or without cache
    local build_args="--build"
    if [ "$no_cache" = "true" ]; then
        echo -e "${YELLOW}Building without cache...${NC}"
        build_args="--build --force-recreate"
        # First build with no-cache
        if [ -n "$service" ]; then
            $COMPOSE build --no-cache "$service"
        else
            $COMPOSE build --no-cache
        fi
    fi

    if [ -n "$service" ]; then
        echo -e "${YELLOW}Rebuilding $service...${NC}"
        $COMPOSE up -d $build_args "$service"
    else
        echo -e "${YELLOW}Rebuilding all services...${NC}"
        $COMPOSE up -d $build_args
    fi

    echo -e "${GREEN}Rebuild complete${NC}"

    # Wait for services to be ready before checking health
    echo -e "${YELLOW}Waiting for services to start...${NC}"
    sleep 10

    cmd_status
}

cmd_logs() {
    local service="$1"
    local follow="$2"
    local tail="${3:-100}"

    local args="--tail=$tail"

    if [ "$follow" = "true" ]; then
        args="$args -f"
    fi

    if [ -n "$service" ]; then
        $COMPOSE logs $args "$service"
    else
        $COMPOSE logs $args
    fi
}

cmd_status() {
    echo -e "${BLUE}Service Status:${NC}"
    echo ""
    $COMPOSE ps
    echo ""

    # Check health endpoints
    echo -e "${BLUE}Health Checks:${NC}"

    # Fleet Manager
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:30080/health 2>/dev/null | grep -q "200"; then
        echo -e "  Fleet Manager: ${GREEN}healthy${NC}"
    else
        echo -e "  Fleet Manager: ${RED}unhealthy${NC}"
    fi

    # Prometheus
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:30090/-/healthy 2>/dev/null | grep -q "200"; then
        echo -e "  Prometheus:    ${GREEN}healthy${NC}"
    else
        echo -e "  Prometheus:    ${RED}unhealthy${NC}"
    fi

    # Grafana
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:30030/api/health 2>/dev/null | grep -q "200"; then
        echo -e "  Grafana:       ${GREEN}healthy${NC}"
    else
        echo -e "  Grafana:       ${RED}unhealthy${NC}"
    fi

    # Loki
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:30100/ready 2>/dev/null | grep -q "200"; then
        echo -e "  Loki:          ${GREEN}healthy${NC}"
    else
        echo -e "  Loki:          ${RED}unhealthy${NC}"
    fi

    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo "  Fleet Manager UI: http://localhost:30080"
    echo "  Grafana:          http://localhost:30030"
    echo "  Prometheus:       http://localhost:30090"
    echo "  Loki:             http://localhost:30100"
}

cmd_clean() {
    echo -e "${RED}WARNING: This will remove all containers and volumes!${NC}"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Stopping and removing all containers and volumes...${NC}"
        $COMPOSE down -v
        echo -e "${GREEN}Cleanup complete${NC}"
    else
        echo "Aborted"
    fi
}

cmd_db_shell() {
    echo -e "${BLUE}Opening PostgreSQL shell...${NC}"
    $COMPOSE exec postgres psql -U fleet -d fleet_monitor
}

# Main script
print_header

# Parse arguments
COMMAND=""
SERVICE=""
FOLLOW="false"
TAIL="100"
SKIP_UI="false"
NO_CACHE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        start|stop|restart|rebuild|logs|status|ui|clean|db-shell)
            COMMAND="$1"
            shift
            ;;
        --service)
            SERVICE="$2"
            shift 2
            ;;
        --no-ui)
            SKIP_UI="true"
            shift
            ;;
        --no-cache)
            NO_CACHE="true"
            shift
            ;;
        -f|--follow)
            FOLLOW="true"
            shift
            ;;
        -n|--tail)
            TAIL="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Check docker access for commands that need it
case $COMMAND in
    start|stop|restart|rebuild|logs|status|clean|db-shell)
        check_docker_access "$@"
        ;;
esac

# Execute command
case $COMMAND in
    start)
        cmd_start "$SERVICE"
        ;;
    stop)
        cmd_stop "$SERVICE"
        ;;
    restart)
        cmd_restart "$SERVICE"
        ;;
    rebuild)
        cmd_rebuild "$SERVICE" "$SKIP_UI" "$NO_CACHE"
        ;;
    logs)
        cmd_logs "$SERVICE" "$FOLLOW" "$TAIL"
        ;;
    status)
        cmd_status
        ;;
    ui)
        build_ui
        ;;
    clean)
        cmd_clean
        ;;
    db-shell)
        cmd_db_shell
        ;;
    "")
        print_usage
        exit 0
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        print_usage
        exit 1
        ;;
esac
