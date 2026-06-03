"""Remote installation of monitoring exporters on GPU nodes."""

import asyncio
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime

from .ssh_manager import SSHManager

logger = logging.getLogger(__name__)

SCRIPTS_PATH = os.environ.get("SCRIPTS_PATH", "/app/scripts")
TEMPLATES_PATH = os.environ.get("TEMPLATES_PATH", "/app/templates")


class NodeInstaller:
    """Handles installation of monitoring components on GPU nodes."""

    def __init__(
        self,
        ssh_manager: SSHManager,
        loki_url: str,
        node_group_name: str,
        hostname: Optional[str] = None,
    ):
        self.ssh = ssh_manager
        self.loki_url = loki_url
        self.node_group_name = node_group_name
        self.hostname = hostname or ssh_manager.host
        self.install_log: list = []

    def _log(self, component: str, message: str, success: bool = True):
        """Add entry to installation log."""
        self.install_log.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "component": component,
                "message": message,
                "success": success,
            }
        )
        if success:
            logger.info(f"[{self.ssh.host}] {component}: {message}")
        else:
            logger.error(f"[{self.ssh.host}] {component}: {message}")

    async def check_prerequisites(self) -> Dict[str, bool]:
        """Check if prerequisites are met on the node."""
        checks = {}

        # Check for amd-smi (preferred) or rocm-smi (fallback)
        result = await self.ssh.execute("which amd-smi || ls /opt/rocm*/bin/amd-smi 2>/dev/null | head -1")
        checks["rocm"] = result.success and result.stdout.strip() != ""

        # Check for systemd
        result = await self.ssh.execute("which systemctl")
        checks["systemd"] = result.success

        # Check for Docker or Podman (for AMD exporter container)
        result = await self.ssh.execute("which docker || which podman")
        checks["docker"] = result.success

        # Check for curl
        result = await self.ssh.execute("which curl")
        checks["curl"] = result.success

        return checks

    async def _ensure_container_runtime(self) -> str:
        """Ensure Docker or Podman is available, install podman-docker if needed.

        Returns the container command to use ('docker' or 'podman').
        """
        # Check if docker is available
        result = await self.ssh.execute("which docker")
        if result.success and result.stdout.strip():
            self._log("container_runtime", "Docker is available")
            return "docker"

        # Check if podman is available
        result = await self.ssh.execute("which podman")
        if result.success and result.stdout.strip():
            self._log("container_runtime", "Podman is available")
            return "podman"

        # Neither available - install podman-docker
        self._log("container_runtime", "No container runtime found, installing podman-docker...")

        # Detect package manager and install
        result = await self.ssh.execute("which apt-get")
        if result.success:
            # Debian/Ubuntu
            install_cmd = "sudo apt-get update -qq && sudo apt-get install -y -qq podman-docker"
        else:
            result = await self.ssh.execute("which dnf")
            if result.success:
                # Fedora/RHEL 8+
                install_cmd = "sudo dnf install -y -q podman-docker"
            else:
                result = await self.ssh.execute("which yum")
                if result.success:
                    # RHEL 7/CentOS
                    install_cmd = "sudo yum install -y -q podman-docker"
                else:
                    self._log("container_runtime", "Unsupported package manager", False)
                    return ""

        result = await self.ssh.execute(install_cmd, timeout=300)
        if not result.success:
            self._log("container_runtime", f"Failed to install podman-docker: {result.stderr}", False)
            return ""

        # Verify installation
        result = await self.ssh.execute("which podman || which docker")
        if result.success and result.stdout.strip():
            self._log("container_runtime", "Successfully installed podman-docker")
            # podman-docker creates a docker alias, so use docker command
            return "docker"

        self._log("container_runtime", "Installation completed but container runtime not found", False)
        return ""

    # Default version for the official ROCm device-metrics-exporter
    DEFAULT_EXPORTER_VERSION = "v1.5.0"

    async def install_amd_metrics_exporter(
        self,
        port: int = 5000,
        version: str = None,
    ) -> bool:
        """Install AMD GPU Metrics Exporter using the official ROCm device-metrics-exporter.

        Uses the official Docker-based exporter from AMD/ROCm:
        https://github.com/ROCm/device-metrics-exporter

        Args:
            port: Port to expose metrics on (default: 5000)
            version: Exporter version tag (default: v1.5.0)
        """
        if version is None:
            version = self.DEFAULT_EXPORTER_VERSION

        self._log("amd_exporter", f"Starting AMD GPU Metrics Exporter installation (official ROCm exporter {version})")

        # Load the amdgpu driver if not already loaded
        result = await self.ssh.execute("lsmod | grep -q amdgpu || sudo modprobe amdgpu")
        if result.success:
            self._log("amd_exporter", "AMD GPU driver loaded (or already present)")
        else:
            self._log("amd_exporter", f"Warning: Could not load amdgpu driver: {result.stderr}", False)

        # Ensure container runtime (Docker or Podman) is available
        container_cmd = await self._ensure_container_runtime()
        if not container_cmd:
            self._log("amd_exporter", "No container runtime available and failed to install podman-docker", False)
            return False

        # Check if /dev/kfd and /dev/dri exist (required for GPU access)
        result = await self.ssh.execute("test -e /dev/kfd && test -d /dev/dri && echo 'ok'")
        if not result.success or "ok" not in result.stdout:
            self._log("amd_exporter", "GPU devices (/dev/kfd, /dev/dri) not found", False)
            return False

        # === CLEANUP: Stop and remove any existing GPU exporter installations ===
        self._log("amd_exporter", "Cleaning up existing GPU exporter installations...")

        # Stop and remove existing containers (both possible names)
        for container_name in ["device-metrics-exporter", "amd-metrics-exporter"]:
            await self.ssh.execute(f"sudo {container_cmd} stop {container_name} 2>/dev/null || true")
            await self.ssh.execute(f"sudo {container_cmd} rm {container_name} 2>/dev/null || true")

        # Stop and remove old systemd-based exporter if present (to free up the port)
        await self.ssh.execute(
            "sudo systemctl stop amd-gpu-exporter 2>/dev/null || true; "
            "sudo systemctl disable amd-gpu-exporter 2>/dev/null || true; "
            "sudo rm -f /etc/systemd/system/amd-gpu-exporter.service 2>/dev/null || true; "
            "sudo rm -f /usr/local/bin/amd-gpu-exporter.py 2>/dev/null || true; "
            "sudo systemctl daemon-reload 2>/dev/null || true"
        )

        self._log("amd_exporter", "Cleanup complete")

        # Pull the official exporter image
        # Use fully qualified image name with docker.io registry for Podman compatibility
        image_name = f"docker.io/rocm/device-metrics-exporter:{version}"
        self._log("amd_exporter", f"Pulling image {image_name}...")
        result = await self.ssh.execute(f"sudo {container_cmd} pull {image_name}", timeout=300)
        if not result.success:
            self._log("amd_exporter", f"Failed to pull image: {result.stderr}", False)
            return False

        # Run the container with required device access
        # The official exporter uses these devices and mounts:
        # --device=/dev/dri - GPU device access
        # --device=/dev/kfd - AMD KFD kernel module
        # -v /sys:/sys:ro - Read-only system information (required for RAS monitoring)
        docker_run_cmd = f"""sudo {container_cmd} run -d \\
            --name device-metrics-exporter \\
            --restart=always \\
            --device=/dev/dri \\
            --device=/dev/kfd \\
            -v /sys:/sys:ro \\
            -p {port}:5000 \\
            {image_name}"""

        self._log("amd_exporter", "Starting exporter container...")
        result = await self.ssh.execute(docker_run_cmd, timeout=60)
        if not result.success:
            self._log("amd_exporter", f"Failed to start container: {result.stderr}", False)
            return False

        # Wait for container to start and verify it's running
        await asyncio.sleep(5)
        for attempt in range(5):
            # Check if container is running
            result = await self.ssh.execute(
                f"sudo {container_cmd} ps --filter 'name=device-metrics-exporter' --format '{{{{.Status}}}}'"
            )
            if result.success and "Up" in result.stdout:
                # Check if metrics endpoint is responding
                metrics_result = await self.ssh.execute(f"curl -s http://localhost:{port}/metrics | head -20")
                if metrics_result.success and (
                    "GPU_" in metrics_result.stdout or "rocm" in metrics_result.stdout.lower()
                ):
                    self._log("amd_exporter", f"Successfully installed official exporter {version} on port {port}")
                    return True
                elif metrics_result.success and metrics_result.stdout.strip():
                    # Container is up and returning something
                    self._log("amd_exporter", f"Exporter running on port {port} (metrics initializing)")
                    return True
            await asyncio.sleep(3)

        # Check container logs for any errors
        result = await self.ssh.execute(f"sudo {container_cmd} logs device-metrics-exporter --tail 20 2>&1")
        if result.success:
            self._log("amd_exporter", f"Container logs: {result.stdout[:500]}", False)

        # Final check - if container is at least running, consider it success
        result = await self.ssh.execute(f"sudo {container_cmd} ps --filter 'name=device-metrics-exporter' -q")
        if result.success and result.stdout.strip():
            self._log("amd_exporter", f"Container is running on port {port} (may need time to initialize)")
            return True

        self._log("amd_exporter", "Container not running", False)
        return False

    async def install_node_exporter(
        self,
        port: int = 9100,
        version: str = "1.7.0",
    ) -> bool:
        """Install Prometheus Node Exporter."""
        self._log("node_exporter", "Starting Node Exporter installation")

        # Check if already running
        result = await self.ssh.execute("systemctl is-active node_exporter 2>/dev/null || echo 'inactive'")
        if "active" in result.stdout and "inactive" not in result.stdout:
            self._log("node_exporter", "Already installed and running")
            return True

        # Download and install
        arch_result = await self.ssh.execute("uname -m")
        arch = "amd64" if "x86_64" in arch_result.stdout else "arm64"

        download_url = f"https://github.com/prometheus/node_exporter/releases/download/v{version}/node_exporter-{version}.linux-{arch}.tar.gz"

        install_script = f"""
set -e
cd /tmp
curl -LO {download_url}
tar xzf node_exporter-{version}.linux-{arch}.tar.gz
sudo cp node_exporter-{version}.linux-{arch}/node_exporter /usr/local/bin/
sudo chmod +x /usr/local/bin/node_exporter
rm -rf node_exporter-{version}.linux-{arch}*

# Create systemd service
sudo tee /etc/systemd/system/node_exporter.service > /dev/null << 'EOF'
[Unit]
Description=Prometheus Node Exporter
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/node_exporter --web.listen-address=:{port}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl start node_exporter
"""

        result = await self.ssh.execute(install_script, timeout=120)
        if not result.success:
            self._log("node_exporter", f"Installation failed: {result.stderr}", False)
            return False

        # Verify with retries
        for attempt in range(3):
            await asyncio.sleep(3)  # Give service time to start

            # First check if service is active
            status_result = await self.ssh.execute("systemctl is-active node_exporter")
            if not status_result.success or "active" not in status_result.stdout:
                self._log("node_exporter", f"Service not active (attempt {attempt + 1}/3), waiting...")
                continue

            # Then check metrics endpoint
            result = await self.ssh.execute(f"curl -s http://localhost:{port}/metrics | grep -c 'node_' || echo 0")
            try:
                metric_count = int(result.stdout.strip())
                if metric_count > 0:
                    self._log("node_exporter", f"Successfully installed on port {port}")
                    return True
            except ValueError:
                pass

            self._log("node_exporter", f"Metrics not ready (attempt {attempt + 1}/3), waiting...")

        # Final check - if service is running, consider it a success
        status_result = await self.ssh.execute("systemctl is-active node_exporter")
        if status_result.success and "active" in status_result.stdout:
            self._log("node_exporter", f"Service is running on port {port} (metrics endpoint may need more time)")
            return True

        self._log("node_exporter", "Installed but service not active", False)
        return False

    async def install_promtail(
        self,
        version: str = "2.9.5",
    ) -> bool:
        """Install Promtail for log collection."""
        self._log("promtail", "Starting Promtail installation")

        # Check if binary already exists
        binary_check = await self.ssh.execute("test -x /usr/local/bin/promtail && echo 'exists'")
        binary_exists = "exists" in binary_check.stdout

        if binary_exists:
            self._log("promtail", "Binary already installed, updating config only")
        else:
            # Download promtail
            arch_result = await self.ssh.execute("uname -m")
            arch = "amd64" if "x86_64" in arch_result.stdout else "arm64"

            download_url = f"https://github.com/grafana/loki/releases/download/v{version}/promtail-linux-{arch}.zip"

            install_script = f"""
set -e
cd /tmp

# Try to install unzip if not available, or use alternative method
if ! command -v unzip &> /dev/null; then
    # Try installing unzip
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq unzip
    elif command -v yum &> /dev/null; then
        sudo yum install -y -q unzip
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y -q unzip
    fi
fi

curl -LO {download_url}

# Check if unzip is available now
if command -v unzip &> /dev/null; then
    unzip -o promtail-linux-{arch}.zip
else
    # Fallback: use python or busybox
    python3 -c "import zipfile; zipfile.ZipFile('promtail-linux-{arch}.zip').extractall('.')" 2>/dev/null || \
    busybox unzip promtail-linux-{arch}.zip 2>/dev/null || \
    (echo "Cannot extract zip file - no unzip tool available" && exit 1)
fi

sudo mv promtail-linux-{arch} /usr/local/bin/promtail
sudo chmod +x /usr/local/bin/promtail
rm -f promtail-linux-{arch}.zip
sudo mkdir -p /etc/promtail /var/lib/promtail
"""

            result = await self.ssh.execute(install_script, timeout=120)
            if not result.success:
                self._log("promtail", f"Download failed: {result.stderr}", False)
                return False

        # Ensure directories exist
        await self.ssh.execute("sudo mkdir -p /etc/promtail /var/lib/promtail")

        # Create promtail config
        config = f"""
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /var/lib/promtail/positions.yaml

clients:
  - url: {self.loki_url}/loki/api/v1/push

scrape_configs:
  # Scrape journal for kernel and system logs (includes dmesg)
  - job_name: journal
    journal:
      max_age: 12h
      path: /var/log/journal
      labels:
        job: systemd-journal
        host: {self.hostname}
        node_group: {self.node_group_name}
    relabel_configs:
      - source_labels: ['__journal__systemd_unit']
        target_label: 'unit'
      - source_labels: ['__journal__transport']
        target_label: 'transport'
      - source_labels: ['__journal_priority_keyword']
        target_label: 'priority'
    pipeline_stages:
      - match:
          selector: '{{transport="kernel"}}'
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
          host: {self.hostname}
          node_group: {self.node_group_name}
          __path__: /var/log/syslog
    pipeline_stages:
      - regex:
          expression: '(?P<timestamp>\\w+ \\d+ \\d+:\\d+:\\d+) (?P<hostname>\\S+) (?P<process>\\S+): (?P<message>.*)'

  # Scrape dmesg directly if available
  - job_name: dmesg
    static_configs:
      - targets:
          - localhost
        labels:
          job: dmesg
          host: {self.hostname}
          node_group: {self.node_group_name}
          __path__: /var/log/dmesg
"""

        # Upload config using heredoc with base64 to avoid shell escaping issues
        import base64

        config_b64 = base64.b64encode(config.encode()).decode()
        result = await self.ssh.execute(
            f"cat << 'EOFB64' | base64 -d | sudo tee /etc/promtail/config.yml > /dev/null\n{config_b64}\nEOFB64"
        )
        if not result.success:
            self._log("promtail", f"Failed to upload config: {result.stderr}", False)
            return False

        # Create systemd service
        service_content = """[Unit]
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
"""
        service_b64 = base64.b64encode(service_content.encode()).decode()
        await self.ssh.execute(
            f"cat << 'EOFB64' | base64 -d | sudo tee /etc/systemd/system/promtail.service > /dev/null\n{service_b64}\nEOFB64"
        )

        # Start service (use longer timeout for systemctl operations through jump host)
        result = await self.ssh.execute(
            "sudo systemctl daemon-reload && sudo systemctl enable promtail && sudo systemctl restart promtail",
            timeout=120,
        )
        if not result.success:
            self._log("promtail", f"Failed to start service: {result.stderr}", False)
            return False

        await asyncio.sleep(2)
        result = await self.ssh.execute("systemctl is-active promtail")
        if result.success and "active" in result.stdout:
            self._log("promtail", "Successfully installed and running")
            return True

        self._log("promtail", "Installation completed but service not active", False)
        return False

    async def install_rdma_exporter(
        self,
        port: int = 9417,
    ) -> bool:
        """Install RDMA Metrics Exporter for network monitoring.

        Collects RDMA/RoCE network metrics using rdma-core tools.
        Works with any RDMA-capable NIC (Mellanox, Intel, etc.)

        Args:
            port: Port to expose metrics on (default: 9417)
        """
        self._log("rdma_exporter", "Starting RDMA Metrics Exporter installation")

        # Check if rdma command is available
        result = await self.ssh.execute("which rdma")
        if not result.success or not result.stdout.strip():
            self._log("rdma_exporter", "rdma command not found, skipping RDMA exporter", False)
            return False

        # Check if there are any RDMA devices
        result = await self.ssh.execute("rdma link 2>/dev/null | head -1")
        if not result.success or not result.stdout.strip():
            self._log("rdma_exporter", "No RDMA devices found, skipping RDMA exporter", False)
            return False

        # === CLEANUP: Stop and remove existing RDMA exporter ===
        await self.ssh.execute(
            "sudo systemctl stop rdma-exporter 2>/dev/null || true; "
            "sudo systemctl disable rdma-exporter 2>/dev/null || true"
        )

        # Create the exporter script
        rdma_exporter_script = '''#!/usr/bin/env python3
"""RDMA Metrics Exporter for Prometheus - Lightweight version"""

import http.server
import json
import os
import subprocess
import threading
import time

METRIC_PREFIX = "rdma"
metrics_output = "# No metrics yet\\n"
metrics_lock = threading.Lock()

def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    except:
        return None

def get_hostname():
    return os.environ.get("HOSTNAME_OVERRIDE", os.uname().nodename)

def collect():
    global metrics_output
    hostname = get_hostname()
    lines = []

    # Link metrics
    links = run_cmd(["rdma", "link", "--json"]) or []
    for link in links:
        ifname = link.get("ifname", "unknown")
        port = str(link.get("port", 0))
        netdev = link.get("netdev", "")
        phys_state = link.get("physical_state", "").lower()
        state = link.get("state", "").lower()
        link_layer = link.get("link_layer", "unknown")

        phys_val = 1 if phys_state in ["link_up", "linkup", "active"] else 0
        state_val = 1 if state in ["active", "up"] else 0

        lines.append(f'{METRIC_PREFIX}_link_physical_state{{hostname="{hostname}",device="{ifname}",port="{port}",netdev="{netdev}"}} {phys_val}')
        lines.append(f'{METRIC_PREFIX}_link_state{{hostname="{hostname}",device="{ifname}",port="{port}",netdev="{netdev}"}} {state_val}')
        lines.append(f'{METRIC_PREFIX}_link_info{{hostname="{hostname}",device="{ifname}",port="{port}",netdev="{netdev}",link_layer="{link_layer}"}} 1')

    # Statistics
    stats = run_cmd(["rdma", "statistic", "--json"]) or []
    for stat in stats:
        ifname = stat.get("ifname", "unknown")
        port = str(stat.get("port", 0))
        counters = {}
        for key in ["stats", "hw_stats", "port_stats"]:
            if key in stat:
                v = stat[key]
                if isinstance(v, dict):
                    counters.update(v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, dict):
                            counters.update(x)
        # Also check top-level numeric fields
        for k, v in stat.items():
            if isinstance(v, (int, float)) and k not in ["ifindex", "port"]:
                counters[k] = v

        for k, v in counters.items():
            if isinstance(v, (int, float)):
                name = k.lower().replace("-", "_").replace(".", "_")
                lines.append(f'{METRIC_PREFIX}_stat_{name}{{hostname="{hostname}",device="{ifname}",port="{port}"}} {v}')

    # Resources
    resources = run_cmd(["rdma", "resource", "--json"]) or {}
    for rtype in ["qp", "cm_id", "mr", "pd", "cq", "ctx", "srq"]:
        if rtype in resources and isinstance(resources[rtype], list):
            lines.append(f'{METRIC_PREFIX}_resource_{rtype}_total{{hostname="{hostname}"}} {len(resources[rtype])}')

    lines.append(f'{METRIC_PREFIX}_scrape_success{{hostname="{hostname}"}} {1 if links else 0}')

    with metrics_lock:
        metrics_output = "\\n".join(lines) + "\\n"

def collector_loop(interval):
    while True:
        try:
            collect()
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(interval)

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == "/metrics":
            with metrics_lock:
                data = metrics_output
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(data.encode())
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9417)
    p.add_argument("--interval", type=int, default=15)
    args = p.parse_args()

    t = threading.Thread(target=collector_loop, args=(args.interval,), daemon=True)
    t.start()
    time.sleep(1)

    print(f"RDMA Exporter listening on port {args.port}")
    http.server.HTTPServer(("", args.port), Handler).serve_forever()
'''

        # Upload the script
        import base64

        script_b64 = base64.b64encode(rdma_exporter_script.encode()).decode()
        result = await self.ssh.execute(
            f"cat << 'EOFB64' | base64 -d | sudo tee /usr/local/bin/rdma-exporter.py > /dev/null\n{script_b64}\nEOFB64"
        )
        if not result.success:
            self._log("rdma_exporter", f"Failed to upload script: {result.stderr}", False)
            return False

        await self.ssh.execute("sudo chmod +x /usr/local/bin/rdma-exporter.py")

        # Create systemd service
        service_content = f"""[Unit]
Description=RDMA Metrics Exporter for Prometheus
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /usr/local/bin/rdma-exporter.py --port {port}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
        service_b64 = base64.b64encode(service_content.encode()).decode()
        await self.ssh.execute(
            f"cat << 'EOFB64' | base64 -d | sudo tee /etc/systemd/system/rdma-exporter.service > /dev/null\n{service_b64}\nEOFB64"
        )

        # Start service
        result = await self.ssh.execute(
            "sudo systemctl daemon-reload && sudo systemctl enable rdma-exporter && sudo systemctl restart rdma-exporter",
            timeout=60,
        )
        if not result.success:
            self._log("rdma_exporter", f"Failed to start service: {result.stderr}", False)
            return False

        # Verify
        await asyncio.sleep(3)
        result = await self.ssh.execute(f"curl -s http://localhost:{port}/metrics | head -5")
        if result.success and "rdma" in result.stdout.lower():
            self._log("rdma_exporter", f"Successfully installed on port {port}")
            return True

        # Check if service is at least running
        result = await self.ssh.execute("systemctl is-active rdma-exporter")
        if result.success and "active" in result.stdout:
            self._log("rdma_exporter", f"Service running on port {port}")
            return True

        self._log("rdma_exporter", "Installation completed but service not responding", False)
        return False

    async def install_all(
        self,
        gpu_port: int = 5000,
        node_port: int = 9100,
        rdma_port: int = 9417,
    ) -> Dict[str, bool]:
        """Install all monitoring components."""
        results = {}

        # Check prerequisites
        prereqs = await self.check_prerequisites()
        self._log("prerequisites", f"Check results: {prereqs}")

        # AMD exporter uses amd-smi
        if not prereqs.get("rocm"):
            self._log("prerequisites", "amd-smi not available, cannot install AMD exporter", False)
            results["amd_exporter"] = False
        else:
            results["amd_exporter"] = await self.install_amd_metrics_exporter(port=gpu_port)

        results["node_exporter"] = await self.install_node_exporter(port=node_port)
        results["promtail"] = await self.install_promtail()
        results["rdma_exporter"] = await self.install_rdma_exporter(port=rdma_port)

        return results

    async def uninstall_all(self) -> Dict[str, bool]:
        """Uninstall all monitoring components."""
        results = {}

        # Stop and remove AMD exporter (Docker/Podman container - official ROCm exporter)
        # Try both docker and podman to handle either runtime
        result = await self.ssh.execute(
            "sudo docker stop device-metrics-exporter 2>/dev/null; "
            "sudo docker rm device-metrics-exporter 2>/dev/null; "
            "sudo podman stop device-metrics-exporter 2>/dev/null; "
            "sudo podman rm device-metrics-exporter 2>/dev/null; "
            # Also clean up old systemd-based exporter if present
            "sudo systemctl stop amd-gpu-exporter 2>/dev/null; "
            "sudo systemctl disable amd-gpu-exporter 2>/dev/null; "
            "sudo rm -f /usr/local/bin/amd-gpu-exporter.py /etc/systemd/system/amd-gpu-exporter.service"
        )
        results["amd_exporter"] = result.success

        # Stop and remove node exporter
        result = await self.ssh.execute(
            "sudo systemctl stop node_exporter 2>/dev/null; "
            "sudo systemctl disable node_exporter 2>/dev/null; "
            "sudo rm -f /usr/local/bin/node_exporter /etc/systemd/system/node_exporter.service"
        )
        results["node_exporter"] = result.success

        # Stop and remove promtail
        result = await self.ssh.execute(
            "sudo systemctl stop promtail 2>/dev/null; "
            "sudo systemctl disable promtail 2>/dev/null; "
            "sudo rm -f /usr/local/bin/promtail /etc/systemd/system/promtail.service; "
            "sudo rm -rf /etc/promtail"
        )
        results["promtail"] = result.success

        # Stop and remove RDMA exporter
        result = await self.ssh.execute(
            "sudo systemctl stop rdma-exporter 2>/dev/null; "
            "sudo systemctl disable rdma-exporter 2>/dev/null; "
            "sudo rm -f /usr/local/bin/rdma-exporter.py /etc/systemd/system/rdma-exporter.service"
        )
        results["rdma_exporter"] = result.success

        await self.ssh.execute("systemctl daemon-reload")

        return results

    async def health_check(self, gpu_port: int = 5000, node_port: int = 9100, rdma_port: int = 9417) -> Dict[str, Any]:
        """Check health of all installed components."""
        health = {}

        # AMD exporter
        result = await self.ssh.execute(f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{gpu_port}/metrics")
        health["amd_exporter"] = {
            "running": result.success and result.stdout.strip() == "200",
            "port": gpu_port,
        }

        # Node exporter
        result = await self.ssh.execute(
            f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{node_port}/metrics"
        )
        health["node_exporter"] = {
            "running": result.success and result.stdout.strip() == "200",
            "port": node_port,
        }

        # RDMA exporter
        result = await self.ssh.execute(
            f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{rdma_port}/metrics"
        )
        health["rdma_exporter"] = {
            "running": result.success and result.stdout.strip() == "200",
            "port": rdma_port,
        }

        # Promtail
        result = await self.ssh.execute("systemctl is-active promtail")
        health["promtail"] = {
            "running": result.success and "active" in result.stdout,
        }

        return health
