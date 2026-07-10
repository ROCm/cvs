#!/usr/bin/env python3
"""
RDMA Metrics Exporter for Prometheus

Collects RDMA network metrics using rdma-core tools and exposes them
in Prometheus format. Works with any RDMA-capable NIC (Mellanox, Intel, etc.)

Metrics collected from:
- rdma link --json: Link state and physical information
- rdma statistic --json: Traffic and error counters
- rdma resource --json: Resource utilization (QPs, CQs, MRs, etc.)

Usage:
    ./rdma_exporter.py [--port PORT] [--interval SECONDS]

Default port: 9417
Default interval: 15 seconds
"""

import argparse
import http.server
import json
import logging
import os
import subprocess
import threading
import time
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Metric prefix
METRIC_PREFIX = "rdma"

# Global metrics storage
metrics_data: Dict[str, Any] = {}
metrics_lock = threading.Lock()


def run_command(cmd: List[str], timeout: int = 10) -> Optional[Dict]:
    """Run a command and return JSON output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.warning(f"Command {' '.join(cmd)} failed: {result.stderr}")
            return None

        if not result.stdout.strip():
            return None

        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        logger.error(f"Command {' '.join(cmd)} timed out")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {' '.join(cmd)}: {e}")
        return None
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        return None
    except Exception as e:
        logger.error(f"Error running {' '.join(cmd)}: {e}")
        return None


def collect_rdma_link() -> List[Dict]:
    """Collect RDMA link information."""
    data = run_command(["rdma", "link", "--json"])
    if data is None:
        return []

    # rdma link --json returns a list of links
    if isinstance(data, list):
        return data
    return []


def collect_rdma_stats() -> List[Dict]:
    """Collect RDMA statistics."""
    data = run_command(["rdma", "statistic", "--json"])
    if data is None:
        return []

    if isinstance(data, list):
        return data
    return []


def collect_rdma_resource() -> Dict:
    """Collect RDMA resource information."""
    data = run_command(["rdma", "resource", "--json"])
    if data is None:
        return {}

    if isinstance(data, dict):
        return data
    return {}


def get_hostname() -> str:
    """Get the hostname."""
    return os.environ.get("HOSTNAME_OVERRIDE", os.uname().nodename)


def parse_link_metrics(links: List[Dict], hostname: str) -> List[str]:
    """Parse RDMA link data into Prometheus metrics."""
    metrics = []

    for link in links:
        # Extract common labels
        ifname = link.get("ifname", "unknown")
        port = str(link.get("port", 0))
        netdev = link.get("netdev", "")

        # Physical state (1=up, 0=down)
        phys_state = link.get("physical_state", "").lower()
        phys_state_val = 1 if phys_state in ["link_up", "linkup", "active"] else 0
        metrics.append(
            f'{METRIC_PREFIX}_link_physical_state{{hostname="{hostname}",device="{ifname}",port="{port}",netdev="{netdev}"}} {phys_state_val}'
        )

        # Link state (1=active, 0=inactive)
        state = link.get("state", "").lower()
        state_val = 1 if state in ["active", "up"] else 0
        metrics.append(
            f'{METRIC_PREFIX}_link_state{{hostname="{hostname}",device="{ifname}",port="{port}",netdev="{netdev}"}} {state_val}'
        )

        # Link layer type as info metric
        link_layer = link.get("link_layer", "unknown")
        metrics.append(
            f'{METRIC_PREFIX}_link_info{{hostname="{hostname}",device="{ifname}",port="{port}",netdev="{netdev}",link_layer="{link_layer}",physical_state="{phys_state}",state="{state}"}} 1'
        )

    return metrics


def parse_stats_metrics(stats: List[Dict], hostname: str) -> List[str]:
    """Parse RDMA statistics into Prometheus metrics."""
    metrics = []

    for stat in stats:
        ifname = stat.get("ifname", "unknown")
        port = str(stat.get("port", 0))

        # Get the counters - structure varies by driver
        # Try different possible locations for counters
        counters = {}

        # Some drivers put stats directly in the object
        if "cnp_ignored" in stat or "rx_write_requests" in stat:
            counters = stat

        # Some put them under "stats" key
        if "stats" in stat:
            if isinstance(stat["stats"], dict):
                counters.update(stat["stats"])
            elif isinstance(stat["stats"], list):
                for s in stat["stats"]:
                    if isinstance(s, dict):
                        counters.update(s)

        # HW stats
        if "hw_stats" in stat:
            counters.update(stat.get("hw_stats", {}))

        # Port stats
        if "port_stats" in stat:
            counters.update(stat.get("port_stats", {}))

        # Parse each counter
        for key, value in counters.items():
            # Skip non-numeric values and metadata
            if not isinstance(value, (int, float)):
                continue
            if key in ["ifindex", "port", "ifname"]:
                continue

            # Sanitize metric name
            metric_name = key.lower().replace("-", "_").replace(".", "_")
            metrics.append(
                f'{METRIC_PREFIX}_stat_{metric_name}{{hostname="{hostname}",device="{ifname}",port="{port}"}} {value}'
            )

    return metrics


def parse_resource_metrics(resources: Dict, hostname: str) -> List[str]:
    """Parse RDMA resource data into Prometheus metrics."""
    metrics = []

    # Resource summary counts
    for resource_type in ["qp", "cm_id", "mr", "pd", "cq", "ctx", "srq"]:
        if resource_type in resources:
            resource_list = resources[resource_type]
            if isinstance(resource_list, list):
                # Count total resources
                total = len(resource_list)
                metrics.append(f'{METRIC_PREFIX}_resource_{resource_type}_total{{hostname="{hostname}"}} {total}')

                # Count by device
                by_device: Dict[str, int] = {}
                for item in resource_list:
                    dev = item.get("ifname", item.get("dev", "unknown"))
                    by_device[dev] = by_device.get(dev, 0) + 1

                for dev, count in by_device.items():
                    metrics.append(
                        f'{METRIC_PREFIX}_resource_{resource_type}_count{{hostname="{hostname}",device="{dev}"}} {count}'
                    )

    return metrics


def collect_all_metrics() -> str:
    """Collect all RDMA metrics and format for Prometheus."""
    hostname = get_hostname()
    all_metrics = []

    # Add metadata
    all_metrics.append(f"# HELP {METRIC_PREFIX}_link_physical_state Physical state of RDMA link (1=up, 0=down)")
    all_metrics.append(f"# TYPE {METRIC_PREFIX}_link_physical_state gauge")
    all_metrics.append(f"# HELP {METRIC_PREFIX}_link_state Logical state of RDMA link (1=active, 0=inactive)")
    all_metrics.append(f"# TYPE {METRIC_PREFIX}_link_state gauge")
    all_metrics.append(f"# HELP {METRIC_PREFIX}_link_info RDMA link information")
    all_metrics.append(f"# TYPE {METRIC_PREFIX}_link_info gauge")

    # Collect link metrics
    links = collect_rdma_link()
    all_metrics.extend(parse_link_metrics(links, hostname))

    # Collect statistics
    all_metrics.append(f"# HELP {METRIC_PREFIX}_stat RDMA statistics counters")
    all_metrics.append(f"# TYPE {METRIC_PREFIX}_stat counter")
    stats = collect_rdma_stats()
    all_metrics.extend(parse_stats_metrics(stats, hostname))

    # Collect resource metrics
    all_metrics.append(f"# HELP {METRIC_PREFIX}_resource RDMA resource counts")
    all_metrics.append(f"# TYPE {METRIC_PREFIX}_resource gauge")
    resources = collect_rdma_resource()
    all_metrics.extend(parse_resource_metrics(resources, hostname))

    # Add scrape metadata
    all_metrics.append(f"# HELP {METRIC_PREFIX}_scrape_success Whether the RDMA metrics scrape was successful")
    all_metrics.append(f"# TYPE {METRIC_PREFIX}_scrape_success gauge")
    success = 1 if links or stats else 0
    all_metrics.append(f'{METRIC_PREFIX}_scrape_success{{hostname="{hostname}"}} {success}')

    return "\n".join(all_metrics) + "\n"


def metrics_collector(interval: int):
    """Background thread to collect metrics periodically."""
    global metrics_data

    while True:
        try:
            metrics_output = collect_all_metrics()
            with metrics_lock:
                metrics_data["output"] = metrics_output
                metrics_data["timestamp"] = time.time()
            logger.debug("Metrics collection completed")
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

        time.sleep(interval)


class MetricsHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/metrics":
            with metrics_lock:
                output = metrics_data.get("output", "# No metrics collected yet\n")

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(output.encode())

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK\n")

        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="RDMA Metrics Exporter for Prometheus")
    parser.add_argument("--port", type=int, default=9417, help="Port to listen on (default: 9417)")
    parser.add_argument("--interval", type=int, default=15, help="Collection interval in seconds (default: 15)")
    args = parser.parse_args()

    logger.info(f"Starting RDMA Metrics Exporter on port {args.port}")
    logger.info(f"Collection interval: {args.interval} seconds")

    # Check if rdma command is available
    if run_command(["rdma", "link", "--json"]) is None:
        logger.warning("rdma command not available or no RDMA devices found")

    # Start metrics collector thread
    collector_thread = threading.Thread(target=metrics_collector, args=(args.interval,), daemon=True)
    collector_thread.start()

    # Initial collection
    time.sleep(1)

    # Start HTTP server
    server = http.server.HTTPServer(("", args.port), MetricsHandler)
    logger.info(f"Serving metrics at http://0.0.0.0:{args.port}/metrics")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
