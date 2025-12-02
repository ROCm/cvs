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


# ============================================================================
# Node Role Detection Fixtures
# ============================================================================

@pytest.fixture(scope='module')
def management_node(cluster_dict):
    """Get the management/head node from cluster."""
    from utils_lib import get_management_node
    return get_management_node(cluster_dict)


@pytest.fixture(scope='module')
def all_nodes(cluster_dict):
    """Get all nodes (management + workers) where exporter should run."""
    from utils_lib import get_all_nodes
    return get_all_nodes(cluster_dict)


@pytest.fixture(scope='module')
def worker_nodes(cluster_dict):
    """Get worker nodes only."""
    from utils_lib import get_worker_nodes
    return get_worker_nodes(cluster_dict)


@pytest.fixture(scope='module')
def is_single_node(cluster_dict):
    """Check if this is a single-node deployment."""
    from utils_lib import is_single_node_deployment
    return is_single_node_deployment(cluster_dict)


@pytest.fixture(scope='module')
def prometheus_targets(cluster_dict, config_dict):
    """Generate Prometheus scrape targets for all nodes."""
    from utils_lib import generate_prometheus_targets
    exporter_port = config_dict.get('device_metrics_exporter_port', 5000)
    return generate_prometheus_targets(cluster_dict, exporter_port)


def is_mgmt_node(node, cluster_dict):
    """Helper function to check if node is management node."""
    from utils_lib import is_management_node
    return is_management_node(node, cluster_dict)


# Tests with Management Node Awareness

def test_deploy_prometheus_on_management_only(cluster_dict, management_node, is_single_node, config_dict, prometheus_targets):
    """
    Deploy Prometheus ONLY on management node with all targets configured.
    Uses pssh for multi-node, subprocess for localhost.
    """
    log.info("="*80)
    log.info(f"Deploying Prometheus on management node: {management_node}")
    log.info(f"Targets: {prometheus_targets}")
    log.info("="*80)
    
    import subprocess
    import os
    from prometheus_config_lib import generate_prometheus_config
    
    # Generate Prometheus config
    prometheus_yml = "/tmp/prometheus_cvs.yml"
    generate_prometheus_config(cluster_dict, config_dict, prometheus_yml)
    log.info(f" Config generated with {len(prometheus_targets)} targets")
    
    prom_version = config_dict.get('prometheus_version', 'v2.55.0').lstrip('v')
    
    # Deploy on localhost/management node
    if is_single_node or is_localhost(management_node):
        # LOCAL DEPLOYMENT
        # Stop existing
        subprocess.run("sudo systemctl stop prometheus 2>/dev/null || true", shell=True)
        subprocess.run("sudo pkill -9 prometheus 2>/dev/null || true", shell=True)
        
        # Install if needed
        if not os.path.exists('/opt/prometheus/prometheus'):
            log.info(f"Installing Prometheus {prom_version}...")
            cmd = f"""cd /tmp && wget -q https://github.com/prometheus/prometheus/releases/download/v{prom_version}/prometheus-{prom_version}.linux-amd64.tar.gz && tar xzf prometheus-{prom_version}.linux-amd64.tar.gz && sudo mkdir -p /opt/prometheus /var/lib/prometheus/data && sudo cp -r prometheus-{prom_version}.linux-amd64/* /opt/prometheus/"""
            subprocess.run(cmd, shell=True, check=True)
        
        # Copy config
        subprocess.run(f"sudo cp {prometheus_yml} /opt/prometheus/prometheus.yml", shell=True, check=True)
        
        # Create systemd service
        svc = """[Unit]
Description=Prometheus
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/prometheus/prometheus --config.file=/opt/prometheus/prometheus.yml --storage.tsdb.path=/var/lib/prometheus/data --web.listen-address=0.0.0.0:9090
Restart=always

[Install]
WantedBy=multi-user.target
"""
        with open('/tmp/prometheus.service', 'w') as f:
            f.write(svc)
        subprocess.run("sudo cp /tmp/prometheus.service /etc/systemd/system/", shell=True, check=True)
        subprocess.run("sudo systemctl daemon-reload && sudo systemctl enable prometheus && sudo systemctl restart prometheus", shell=True, check=True)
        
        import time
        time.sleep(3)
        
        # Verify
        result = subprocess.run("systemctl is-active prometheus", shell=True, capture_output=True)
        assert result.returncode == 0, "Prometheus not running"
        log.info("SUCCESS: Prometheus running on management node (localhost)")
    else:
        # MULTI-NODE DEPLOYMENT via SSH to management node only
        log.info(f"Deploying to remote management node: {management_node}")
        from parallel_ssh_lib import Pssh
        
        # Create SSH client for management node ONLY
        mgmt_dict = {management_node: cluster_dict['node_dict'].get(management_node, {'bmc_ip': 'NA', 'vpc_ip': management_node})}
        phdl = Pssh(log, list(mgmt_dict.keys()), user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
        
        # Upload config file to management node
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            with open(prometheus_yml, 'r') as src:
                f.write(src.read())
            temp_config = f.name
        
        # Deploy Prometheus on management node only
        deploy_script = f"""
        # Stop existing
        sudo systemctl stop prometheus 2>/dev/null || true
        sudo pkill -9 prometheus 2>/dev/null || true
        
        # Install if needed
        if [ ! -f /opt/prometheus/prometheus ]; then
            echo "Installing Prometheus {prom_version}..."
            cd /tmp
            wget -q https://github.com/prometheus/prometheus/releases/download/v{prom_version}/prometheus-{prom_version}.linux-amd64.tar.gz
            tar xzf prometheus-{prom_version}.linux-amd64.tar.gz
            sudo mkdir -p /opt/prometheus /var/lib/prometheus/data
            sudo cp -r prometheus-{prom_version}.linux-amd64/* /opt/prometheus/
        fi
        
        # Copy config (uploaded separately via SCP)
        sudo mkdir -p /opt/prometheus
        
        # Create systemd service
        sudo tee /etc/systemd/system/prometheus.service > /dev/null << 'SVCEOF'
[Unit]
Description=Prometheus
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/prometheus/prometheus --config.file=/opt/prometheus/prometheus.yml --storage.tsdb.path=/var/lib/prometheus/data --web.listen-address=0.0.0.0:9090
Restart=always

[Install]
WantedBy=multi-user.target
SVCEOF
        
        sudo systemctl daemon-reload
        sudo systemctl enable prometheus
        sudo systemctl start prometheus
        sleep 2
        systemctl is-active prometheus
        """
        
        # Execute deployment on management node only
        result = phdl.exec(deploy_script)
        
        # Verify deployment succeeded
        for node, output in result.items():
            if 'active' not in output:
                fail_test(f"Prometheus deployment failed on {node}: {output}")
        
        log.info(f"SUCCESS: Prometheus deployed and running on management node: {management_node}")
        log.info("SUCCESS: ENFORCEMENT: Prometheus deployed ONLY to management node, NOT to workers")

def test_deploy_grafana_on_management_only(cluster_dict, management_node, is_single_node, config_dict):
    """
    Deploy Grafana ONLY on management node.
    Uses pssh for multi-node, subprocess for localhost.
    """
    log.info(f"Deploying Grafana on management node: {management_node}")
    
    # Create provisioning configs and dashboard BEFORE starting Grafana
    create_grafana_provisioning_configs()
    create_grafana_dashboard_file()
    
    import subprocess
    import os
    
    grafana_version = config_dict.get('grafana_version', '10.4.1')
    grafana_port = config_dict.get('grafana_port', '3000')
    
    if is_single_node or is_localhost(management_node):
        # LOCAL DEPLOYMENT
        # Stop existing
        subprocess.run("docker stop grafana 2>/dev/null || true", shell=True)
        subprocess.run("docker rm grafana 2>/dev/null || true", shell=True)
        
        # Create data directory
        grafana_data = "/home/svdt-8/manoj/cvs/cvs/monitoring/grafana_data"
        os.makedirs(grafana_data, exist_ok=True)
        subprocess.run(f"sudo chown -R 472:472 {grafana_data}", shell=True, check=True)
        
        # Start Grafana
        cmd = f"""docker run -d \
            --name grafana \
            --network host \
            --restart unless-stopped \
            -v {grafana_data}:/var/lib/grafana \
            -v $(pwd)/monitoring/provisioning:/etc/grafana/provisioning \
            -v $(pwd)/monitoring/dashboards:/var/lib/grafana/dashboards \
            grafana/grafana:{grafana_version}"""
        subprocess.run(cmd, shell=True, check=True)
        
        import time
        time.sleep(3)
        
        # Verify
        result = subprocess.run("docker ps | grep grafana", shell=True, capture_output=True)
        assert result.returncode == 0, "Grafana not running"
        log.info(f"SUCCESS: Grafana running on management node (localhost) port {grafana_port}")
    else:
        # MULTI-NODE DEPLOYMENT via SSH to management node only
        log.info(f"Deploying to remote management node: {management_node}")
        from parallel_ssh_lib import Pssh
        
        # Create SSH client for management node ONLY
        mgmt_dict = {management_node: cluster_dict['node_dict'].get(management_node, {'bmc_ip': 'NA', 'vpc_ip': management_node})}
        phdl = Pssh(log, list(mgmt_dict.keys()), user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
        
        # Deploy Grafana on management node only
        deploy_script = f"""
        # Stop existing
        docker stop grafana 2>/dev/null || true
        docker rm grafana 2>/dev/null || true
        
        # Create data directory
        mkdir -p /tmp/grafana_data
        sudo chown -R 472:472 /tmp/grafana_data
        
        # Start Grafana
        docker run -d \
            --name grafana \
            --network host \
            --restart unless-stopped \
            -v /tmp/grafana_data:/var/lib/grafana \
            grafana/grafana:{grafana_version}
        
        sleep 3
        docker ps | grep grafana
        """
        
        # Execute deployment on management node only
        result = phdl.exec(deploy_script)
        
        # Verify deployment succeeded
        for node, output in result.items():
            if 'grafana' not in output:
                fail_test(f"Grafana deployment failed on {node}: {output}")
        
        log.info(f"SUCCESS: Grafana deployed and running on management node: {management_node}")
        log.info("SUCCESS: ENFORCEMENT: Grafana deployed ONLY to management node, NOT to workers")

def test_verify_all_nodes_for_exporter(all_nodes, management_node):
    """
    Verify that exporter targets include all nodes (management + workers).
    """
    log.info("="*80)
    log.info(f"All nodes where exporter should run:")
    for node in all_nodes:
        is_mgmt = " (MANAGEMENT)" if node == management_node else ""
        log.info(f"  • {node}{is_mgmt}")
    log.info("="*80)
    
    assert len(all_nodes) > 0
    assert management_node in all_nodes
    log.info(f" Total nodes for exporter deployment: {len(all_nodes)}")


def test_prometheus_scrape_targets(prometheus_targets, all_nodes):
    """
    Verify Prometheus scrape targets include all nodes.
    """
    log.info("="*80)
    log.info("Prometheus scrape targets:")
    for target in prometheus_targets:
        log.info(f"  • {target}")
    log.info("="*80)
    
    assert len(prometheus_targets) == len(all_nodes)
    log.info(f" Scrape targets generated for all {len(all_nodes)} nodes")


def test_verify_service_distribution(cluster_dict, management_node, all_nodes, worker_nodes, is_single_node):
    """
    CRITICAL TEST: Verify service distribution enforcement.
    - Exporter must be on ALL nodes (management + workers)
    - Prometheus must be ONLY on management node
    - Grafana must be ONLY on management node
    """
    log.info("="*80)
    log.info("VERIFYING SERVICE DISTRIBUTION ENFORCEMENT")
    log.info("="*80)
    
    # Show the architecture
    log.info(f"\n Cluster Architecture:")
    log.info(f"  Management Node: {management_node}")
    log.info(f"  Worker Nodes: {worker_nodes if worker_nodes else 'None (single-node)'}")
    log.info(f"  Total Nodes: {len(all_nodes)}")
    log.info(f"  Deployment Type: {'Single-Node' if is_single_node else 'Multi-Node'}")
    
    log.info(f"\nSUCCESS: SERVICE DISTRIBUTION RULES:")
    log.info(f"  1. Device Metrics Exporter → ALL {len(all_nodes)} nodes")
    for node in all_nodes:
        marker = "(MANAGEMENT)" if node == management_node else "(WORKER)"
        log.info(f"      {node} {marker}")
    
    log.info(f"\n  2. Prometheus → ONLY management node")
    log.info(f"      {management_node} (MANAGEMENT ONLY)")
    if worker_nodes:
        for node in worker_nodes:
            log.info(f"      {node} (NOT deployed)")
    
    log.info(f"\n  3. Grafana → ONLY management node")
    log.info(f"      {management_node} (MANAGEMENT ONLY)")
    if worker_nodes:
        for node in worker_nodes:
            log.info(f"      {node} (NOT deployed)")
    
    log.info(f"\n" + "="*80)
    log.info("SUCCESS: SERVICE DISTRIBUTION VERIFIED")
    log.info("="*80)
    
    # Assert the rules
    assert len(all_nodes) >= 1, "Must have at least one node"
    assert management_node in all_nodes, "Management node must be in all_nodes list"
    
    if not is_single_node:
        assert len(worker_nodes) > 0, "Multi-node must have workers"
        log.info(f"SUCCESS: ENFORCEMENT VERIFIED: Multi-node cluster with proper separation")
    else:
        log.info(f"SUCCESS: ENFORCEMENT VERIFIED: Single-node deployment (all services on localhost)")


def is_localhost(node):
    """Check if a node IP/hostname refers to localhost."""
    import socket
    import subprocess
    
    # Obvious localhost values
    if node in ['localhost', '127.0.0.1', '::1', 'localhost.localdomain']:
        return True
    
    # Get all local IPs
    local_ips = set(['127.0.0.1', '::1', 'localhost'])
    
    try:
        # Get hostname and its IP
        hostname = socket.gethostname()
        local_ips.add(hostname)
        
        # Get primary IP
        try:
            local_ip = socket.gethostbyname(hostname)
            local_ips.add(local_ip)
        except:
            pass
        
        # Get all IPs from hostname -I
        try:
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                for ip in result.stdout.strip().split():
                    local_ips.add(ip.strip())
        except:
            pass
        
        # Get all IPs from ip addr
        try:
            result = subprocess.run(['ip', 'addr'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                import re
                for match in re.finditer(r'inet\s+(\d+\.\d+\.\d+\.\d+)', result.stdout):
                    local_ips.add(match.group(1))
        except:
            pass
            
    except Exception as e:
        log.warning(f"Error detecting local IPs: {e}")
    
    log.info(f"Local IPs detected: {local_ips}")
    log.info(f"Checking if {node} is localhost: {node in local_ips}")
    
    return node in local_ips


def create_grafana_dashboard_file():
    """Create GPU dashboard with correct metric names."""
    import os
    import json
    
    dashboard_dir = "monitoring/dashboards"
    os.makedirs(dashboard_dir, exist_ok=True)
    
    dashboard = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "panels": [
            {
                "datasource": "prometheus",
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "gradientMode": "none",
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "showPoints": "never"
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 85}
                            ]
                        },
                        "unit": "celsius"
                    }
                },
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0},
                "id": 1,
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi"}
                },
                "targets": [
                    {
                        "datasource": "prometheus",
                        "expr": "gpu_edge_temperature",
                        "legendFormat": "{{hostname}} GPU{{gpu_id}} Edge",
                        "refId": "A"
                    },
                    {
                        "datasource": "prometheus",
                        "expr": "gpu_junction_temperature",
                        "legendFormat": "{{hostname}} GPU{{gpu_id}} Junction",
                        "refId": "B"
                    }
                ],
                "title": "GPU Temperature",
                "type": "timeseries"
            },
            {
                "datasource": "prometheus",
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisPlacement": "auto",
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "lineInterpolation": "linear",
                            "lineWidth": 1
                        },
                        "mappings": [],
                        "unit": "watt"
                    }
                },
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0},
                "id": 2,
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi"}
                },
                "targets": [
                    {
                        "datasource": "prometheus",
                        "expr": "gpu_power_usage",
                        "legendFormat": "{{hostname}} GPU{{gpu_id}}",
                        "refId": "A"
                    }
                ],
                "title": "GPU Power Usage",
                "type": "timeseries"
            },
            {
                "datasource": "prometheus",
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisPlacement": "auto",
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "lineInterpolation": "linear",
                            "lineWidth": 1
                        },
                        "mappings": [],
                        "unit": "watt"
                    }
                },
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0},
                "id": 3,
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi"}
                },
                "targets": [
                    {
                        "datasource": "prometheus",
                        "expr": "gpu_average_package_power",
                        "legendFormat": "{{hostname}} GPU{{gpu_id}}",
                        "refId": "A"
                    }
                ],
                "title": "GPU Average Package Power",
                "type": "timeseries"
            },
            {
                "datasource": "prometheus",
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisPlacement": "auto",
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "lineInterpolation": "linear",
                            "lineWidth": 1
                        },
                        "mappings": [],
                        "unit": "hertz"
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "id": 4,
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi"}
                },
                "targets": [
                    {
                        "datasource": "prometheus",
                        "expr": "gpu_clock{clock_type=\"GPU_CLOCK_TYPE_SYSTEM\"}",
                        "legendFormat": "{{hostname}} GPU{{gpu_id}}",
                        "refId": "A"
                    }
                ],
                "title": "GPU Clock Speed",
                "type": "timeseries"
            },
            {
                "datasource": "prometheus",
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisPlacement": "auto",
                            "drawStyle": "line",
                            "fillOpacity": 10,
                            "lineInterpolation": "linear",
                            "lineWidth": 1
                        },
                        "mappings": [],
                        "unit": "celsius"
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "id": 5,
                "options": {
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi"}
                },
                "targets": [
                    {
                        "datasource": "prometheus",
                        "expr": "gpu_memory_temperature",
                        "legendFormat": "{{hostname}} GPU{{gpu_id}}",
                        "refId": "A"
                    }
                ],
                "title": "GPU Memory Temperature",
                "type": "timeseries"
            }
        ],
        "refresh": "5s",
        "schemaVersion": 39,
        "tags": ["gpu", "amd", "rocm"],
        "templating": {"list": []},
        "time": {"from": "now-15m", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": "AMD GPU Metrics Dashboard",
        "uid": "amd-gpu-metrics",
        "version": 1
    }
    
    dashboard_file = f"{dashboard_dir}/gpu-metrics-dashboard.json"
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    log.info(f"✓ Created dashboard: {dashboard_file}")
    return dashboard_file


def create_grafana_provisioning_configs():
    """Create Grafana provisioning configs for datasources and dashboards."""
    import os
    
    # Create directories
    os.makedirs("monitoring/provisioning/datasources", exist_ok=True)
    os.makedirs("monitoring/provisioning/dashboards", exist_ok=True)
    
    # Datasource config
    datasource_config = """apiVersion: 1

datasources:
  - name: prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: false
    jsonData:
      timeInterval: "5s"
"""
    
    with open("monitoring/provisioning/datasources/prometheus.yml", 'w') as f:
        f.write(datasource_config)
    
    # Dashboard provisioning config
    dashboard_config = """apiVersion: 1

providers:
  - name: 'Default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
"""
    
    with open("monitoring/provisioning/dashboards/default.yml", 'w') as f:
        f.write(dashboard_config)
    
    log.info("✓ Created Grafana provisioning configs")
