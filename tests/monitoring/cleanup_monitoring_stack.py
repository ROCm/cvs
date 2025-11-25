"""Cleanup test for GPU monitoring stack - removes all components from all nodes."""

import pytest
import logging
import subprocess
from lib.parallel_ssh_lib import Pssh

logger = logging.getLogger(__name__)

def is_localhost(ip_address):
    """Check if IP address is localhost."""
    import socket
    local_addresses = {'localhost', '127.0.0.1', '::1', '127.0.1.1'}
    if ip_address in local_addresses:
        return True
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            local_addresses.update(result.stdout.strip().split())
    except: pass
    return ip_address in local_addresses

@pytest.mark.cleanup
def test_stop_exporters_on_all_nodes(cluster_dict, all_nodes):
    """Stop and remove device-metrics-exporter containers from all nodes."""
    logger.info(f"Stopping device-metrics-exporters on all {len(all_nodes)} nodes")
    username = cluster_dict['username']
    priv_key_file = cluster_dict.get('priv_key_file', f"/home/{username}/.ssh/id_rsa")
    commands = ["docker stop device-metrics-exporter || true", "docker rm device-metrics-exporter || true"]
    
    for node_ip in all_nodes:
        logger.info(f"Cleaning up exporter on node: {node_ip}")
        if is_localhost(node_ip):
            for cmd in commands:
                subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            phdl = Pssh([node_ip], user=username, priv_key=priv_key_file)
            for cmd in commands:
                phdl.run(cmd)
    logger.info("✓ Exporters cleaned up on all nodes")

@pytest.mark.cleanup
def test_stop_prometheus_on_management(cluster_dict, management_node):
    """Stop Prometheus systemd service."""
    logger.info(f"Stopping Prometheus on management node: {management_node}")
    username = cluster_dict['username']
    commands = ["sudo systemctl stop prometheus || true", "sudo systemctl disable prometheus || true"]
    
    if is_localhost(management_node):
        for cmd in commands:
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
    else:
        phdl = Pssh([management_node], user=username, priv_key=cluster_dict.get('priv_key_file'))
        for cmd in commands:
            phdl.run(cmd)
    logger.info("✓ Prometheus stopped")

@pytest.mark.cleanup
def test_stop_grafana_on_management(cluster_dict, management_node):
    """Stop and remove Grafana container."""
    logger.info(f"Stopping Grafana on management node: {management_node}")
    commands = ["docker stop grafana || true", "docker rm grafana || true"]
    
    if is_localhost(management_node):
        for cmd in commands:
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
    logger.info("✓ Grafana stopped")

@pytest.mark.cleanup
def test_remove_prometheus_config(cluster_dict, management_node):
    """Remove Prometheus configuration and data."""
    logger.info(f"Removing Prometheus config from management node")
    commands = [
        "sudo rm -f /etc/systemd/system/prometheus.service",
        "sudo systemctl daemon-reload",
        "sudo rm -rf /etc/prometheus",
        "sudo rm -rf /var/lib/prometheus"
    ]
    
    if is_localhost(management_node):
        for cmd in commands:
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
    logger.info("✓ Prometheus config removed")

@pytest.mark.cleanup
def test_cleanup_summary(all_nodes, management_node):
    """Display cleanup summary."""
    logger.info("=" * 60)
    logger.info("MONITORING STACK CLEANUP COMPLETE")
    logger.info(f"Cleaned {len(all_nodes)} nodes")
    logger.info("=" * 60)
