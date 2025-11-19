# Drop-in replacement for tests/monitoring/install_device_metrics_exporter.py
# Key changes:
# 1. Added apply_monitoring_defaults to config_dict fixture
# 2. Updated metrics_host fixture to use resolved device_metrics_exporter_host
# 3. Fixed hardcoded localhost in test_check_gpu_metrics_exposed (line ~217)

import pytest
import re
import sys
import os
import time
import json
import logging

sys.path.insert(0, './lib')
from parallel_ssh_lib import *
from utils_lib import *

import globals

log = globals.log


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """Get cluster file path from pytest CLI"""
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """Get config file path from pytest CLI"""
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """Load cluster configuration"""
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    """Load monitoring configuration with localhost/version fallbacks"""
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t.get('monitoring', {})
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    # Apply defaults for unresolved placeholders
    config_dict = apply_monitoring_defaults(config_dict)
    log.info("Resolved monitoring config:")
    log.info(config_dict)
    return config_dict


@pytest.fixture(scope="module")
def metrics_host(config_dict):
    """Get metrics host with fallback to localhost"""
    return config_dict.get("device_metrics_exporter_host", "localhost")


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """Create parallel SSH handle for all nodes"""
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return phdl


def test_check_docker_installed(phdl):
    """Verify Docker is installed on all nodes"""
    globals.error_list = []
    log.info("Checking if Docker is installed on all nodes")
    
    out_dict = phdl.exec('docker --version')
    
    for node in out_dict.keys():
        if not re.search(r'Docker version', out_dict[node], re.I):
            fail_test(f"Docker is not installed on node {node}. Please install Docker first.")
    
    update_test_result()


def test_check_rocm_installed(phdl):
    """Verify ROCm is installed on all nodes"""
    globals.error_list = []
    log.info("Checking if ROCm is installed on all nodes")
    
    out_dict = phdl.exec('rocm-smi --version || amd-smi version')
    
    for node in out_dict.keys():
        if not re.search(r'ROCm|AMD', out_dict[node], re.I):
            fail_test(f"ROCm is not installed on node {node}. Please install ROCm first.")
    
    update_test_result()


def test_pull_device_metrics_exporter_image(phdl, config_dict):
    """Pull Device Metrics Exporter Docker image on all nodes"""
    globals.error_list = []
    log.info("Pulling Device Metrics Exporter Docker image on all nodes")
    
    version = config_dict['device_metrics_exporter_version']
    image = f"rocm/device-metrics-exporter:{version}"
    log.info(f"Using image: {image}")
    
    out_dict = phdl.exec(f'docker pull {image}', timeout=300)
    
    for node in out_dict.keys():
        if 'Error' in out_dict[node] or 'failed' in out_dict[node].lower():
            fail_test(f"Failed to pull Docker image on node {node}: {out_dict[node]}")
    
    update_test_result()


def test_stop_existing_device_metrics_exporter(phdl):
    """Stop and remove any existing Device Metrics Exporter containers"""
    globals.error_list = []
    log.info("Stopping existing Device Metrics Exporter containers (if any)")
    
    phdl.exec('docker stop device-metrics-exporter 2>/dev/null || true')
    phdl.exec('docker rm device-metrics-exporter 2>/dev/null || true')
    
    log.info("Cleaned up existing containers")
    update_test_result()


def test_start_device_metrics_exporter(phdl, config_dict):
    """Start Device Metrics Exporter container on all nodes"""
    globals.error_list = []
    log.info("Starting Device Metrics Exporter on all nodes")
    
    version = config_dict['device_metrics_exporter_version']
    port = config_dict['device_metrics_exporter_port']
    
    log.info(f"Starting exporter version {version} on port {port}")
    
    # Docker run command
    docker_cmd = f'''docker run -d \
        --device=/dev/dri \
        --device=/dev/kfd \
        --network=host \
        -p {port}:{port} \
        --restart unless-stopped \
        --name device-metrics-exporter \
        rocm/device-metrics-exporter:{version}'''
    
    out_dict = phdl.exec(docker_cmd)
    
    for node in out_dict.keys():
        if 'Error' in out_dict[node]:
            fail_test(f"Failed to start Device Metrics Exporter on node {node}: {out_dict[node]}")
    
    log.info("Device Metrics Exporter started on all nodes")
    update_test_result()


def test_verify_exporter_running(phdl):
    """Verify Device Metrics Exporter is running"""
    globals.error_list = []
    log.info("Verifying Device Metrics Exporter is running on all nodes")
    
    # Wait for containers to start
    time.sleep(10)
    
    out_dict = phdl.exec('docker ps --filter name=device-metrics-exporter --format "{{.Status}}"')
    
    for node in out_dict.keys():
        if 'Up' not in out_dict[node]:
            fail_test(f"Device Metrics Exporter is not running on node {node}")
    
    update_test_result()


def test_verify_metrics_endpoint(phdl, config_dict, metrics_host):
    """Verify metrics endpoint is accessible"""
    globals.error_list = []
    log.info("Verifying metrics endpoint is accessible on all nodes")
    
    port = config_dict['device_metrics_exporter_port']
    log.info(f"Testing endpoint: http://{metrics_host}:{port}/metrics")
    
    # Retry logic for slow container startup
    max_retries = 3
    out_dict = None
    
    for attempt in range(max_retries):
        out_dict = phdl.exec(f'curl -s http://{metrics_host}:{port}/metrics | head -20')
        
        # Check if we got output
        has_output = False
        for node in out_dict.keys():
            if len(out_dict[node]) > 0:
                has_output = True
                break
        
        if has_output:
            break
        else:
            log.info(f"Attempt {attempt+1}/{max_retries}: No output yet, waiting 5 seconds...")
            time.sleep(5)
    
    # Final validation
    for node in out_dict.keys():
        output = out_dict[node]
        log.info(f"Checking output from {node}, length: {len(output)}")
        
        if output and 'gpu_' in output.lower():
            log.info(f"Metrics endpoint verified on node {node}")
        else:
            log.error(f"Output sample: {output[:200]}")
            fail_test(f"Metrics endpoint not accessible on node {node}")
    
    update_test_result()


def test_check_gpu_metrics_exposed(phdl, config_dict, metrics_host):
    """Verify GPU metrics are being exposed"""
    globals.error_list = []
    log.info("Checking if GPU metrics are being exposed")
    
    port = config_dict['device_metrics_exporter_port']
    
    # Use metrics_host instead of hardcoded localhost
    out_dict = phdl.exec(f'curl -s http://{metrics_host}:{port}/metrics | head -50')
    
    for node in out_dict.keys():
        output = out_dict[node]
        log.info(f"Checking GPU metrics from {node}, length: {len(output)}")
        
        if output.strip() and 'gpu_' in output.lower():
            log.info(f"GPU metrics verified on node {node}")
            # Show sample
            lines = [line for line in output.split('\n') if 'gpu_' in line.lower()][:2]
            for line in lines:
                log.info(f"  Sample: {line[:80]}")
        else:
            log.error(f"No GPU metrics found. Output: {output[:300]}")
            fail_test(f"GPU metrics not found on node {node}")
    
    update_test_result()


def test_display_summary(phdl):
    """Display installation summary"""
    log.info("=" * 80)
    log.info("Device Metrics Exporter Installation Complete!")
    log.info("=" * 80)
    log.info("")
    log.info("Exporter Status:")
    
    out_dict = phdl.exec('docker ps --filter name=device-metrics-exporter --format "{{.Names}}: {{.Status}}"')
    
    for node in out_dict.keys():
        log.info(f"  {node}: {out_dict[node]}")
    
    log.info("Completed metrics tests successfully.")
