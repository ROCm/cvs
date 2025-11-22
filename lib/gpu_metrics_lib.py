'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

"""
Device Metrics Integration Library for CVS

This module provides integration between CVS and AMD ROCm Device Metrics Exporter
via Prometheus. It enables CVS to query GPU metrics from Prometheus instead of 
(or in addition to) SSH-based amd-smi/rocm-smi commands.

Device Metrics Exporter: https://github.com/ROCm/device-metrics-exporter
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

log = logging.getLogger(__name__)


class PrometheusClient:
    """
    Client for querying Prometheus server that scrapes Device Metrics Exporter.
    """
    
    def __init__(self, prometheus_url: str , timeout: int = 30):
        if not prometheus_url:
            # fall back only if truly absent
            prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
        self.prometheus_url = prometheus_url.rstrip('/')
        self.timeout = timeout
        self.api_url = f"{self.prometheus_url}/api/v1"
        log.info(f"Initialized Prometheus client for {self.prometheus_url}")
        
    def check_health(self) -> bool:
        """Check if Prometheus server is healthy and reachable."""
        try:
            response = requests.get(
                f"{self.prometheus_url}/-/healthy",
                timeout=self.timeout
            )
            if response.status_code == 200:
                log.info(f"✓ Prometheus server at {self.prometheus_url} is healthy")
                return True
            else:
                log.error(f"✗ Prometheus health check failed with status {response.status_code}")
                return False
        except Exception as e:
            log.error(f"✗ Failed to connect to Prometheus at {self.prometheus_url}: {e}")
            return False
    
    def query_instant(self, query: str) -> Dict[str, Any]:
        """Execute an instant PromQL query."""
        try:
            response = requests.get(
                f"{self.api_url}/query",
                params={'query': query},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get('status') == 'success':
                log.debug(f"Query successful: {query[:50]}...")
            else:
                log.warning(f"Query returned non-success status: {result.get('error', 'Unknown error')}")
            
            return result
        except Exception as e:
            log.error(f"Prometheus instant query failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def query_range(self, query: str, start_time: datetime, end_time: datetime, 
                   step: str = "15s") -> Dict[str, Any]:
        """Execute a range PromQL query for time-series data."""
        try:
            response = requests.get(
                f"{self.api_url}/query_range",
                params={
                    'query': query,
                    'start': start_time.timestamp(),
                    'end': end_time.timestamp(),
                    'step': step
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get('status') == 'success':
                log.debug(f"Range query successful: {query[:50]}... [{start_time} to {end_time}]")
            
            return result
        except Exception as e:
            log.error(f"Prometheus range query failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_targets(self) -> List[Dict[str, Any]]:
        """Get list of all scrape targets (Device Metrics Exporters) and their status."""
        try:
            response = requests.get(
                f"{self.api_url}/targets",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'success':
                targets = data.get('data', {}).get('activeTargets', [])
                log.info(f"Retrieved {len(targets)} active targets from Prometheus")
                return targets
            return []
        except Exception as e:
            log.error(f"Failed to get Prometheus targets: {e}")
            return []


# Device Metrics Exporter metric names (as of v1.4.0)
DEVICE_METRICS_MAP = {
    # Temperature metrics
    'temperature_edge': 'amdgpu_temperature_edge_celsius',
    'temperature_junction': 'amdgpu_temperature_junction_celsius',
    'temperature_memory': 'amdgpu_temperature_memory_celsius',
    'temperature_hbm': 'amdgpu_temperature_hbm_celsius',
    
    # Utilization metrics
    'gpu_utilization': 'amdgpu_gpu_utilization_percent',
    'memory_utilization': 'amdgpu_memory_utilization_percent',
    
    # Power metrics
    'power_current': 'amdgpu_power_watts',
    'power_average': 'amdgpu_power_average_watts',
    'energy_consumed': 'amdgpu_energy_joules',
    
    # Memory metrics
    'memory_used': 'amdgpu_memory_used_bytes',
    'memory_total': 'amdgpu_memory_total_bytes',
    'memory_free': 'amdgpu_memory_free_bytes',
    
    # Clock metrics
    'clock_gpu': 'amdgpu_gpu_clock_mhz',
    'clock_memory': 'amdgpu_memory_clock_mhz',
    
    # PCIe metrics
    'pcie_bandwidth': 'amdgpu_pcie_bandwidth_bytes',
    'pcie_link_speed': 'amdgpu_pcie_link_speed_mbps',
    'pcie_link_width': 'amdgpu_pcie_link_width',
    'pcie_replay_count': 'amdgpu_pcie_replay_count_total',
    'pcie_nak_sent': 'amdgpu_pcie_nak_sent_total',
    'pcie_nak_received': 'amdgpu_pcie_nak_received_total',
    
    # Error metrics
    'ecc_correctable': 'amdgpu_ecc_correctable_errors_total',
    'ecc_uncorrectable': 'amdgpu_ecc_uncorrectable_errors_total',
    'ras_correctable': 'amdgpu_ras_correctable_error_count',
    'ras_uncorrectable': 'amdgpu_ras_uncorrectable_error_count',
}


def get_gpu_metrics_from_prometheus(prom_client: PrometheusClient, 
                                    node_list: Optional[List[str]] = None,
                                    metrics: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Query current GPU metrics from Prometheus for all or specific nodes.
    
    Returns:
        Dict with structure: {node: {gpu_id: {metric_name: value}}}
    """
    metrics_dict = {}
    
    if metrics is None:
        metrics = [
            'temperature_edge', 'temperature_junction', 'temperature_memory',
            'power_current', 'power_average',
            'gpu_utilization', 'memory_utilization',
            'memory_used', 'memory_total',
            'pcie_bandwidth', 'pcie_link_speed',
            'ecc_correctable', 'ecc_uncorrectable',
            'clock_gpu', 'clock_memory'
        ]
    
    for metric_key in metrics:
        if metric_key not in DEVICE_METRICS_MAP:
            log.warning(f"Unknown metric key: {metric_key}, skipping")
            continue
        
        metric_name = DEVICE_METRICS_MAP[metric_key]
        
        # Build query with optional node filter
        if node_list:
            node_filter = '|'.join([node.replace('.', '\\.') for node in node_list])
            query = f'{metric_name}{{instance=~"({node_filter}):.*"}}'
        else:
            query = metric_name
        
        result = prom_client.query_instant(query)
        
        if result.get('status') == 'success':
            for item in result.get('data', {}).get('result', []):
                labels = item.get('metric', {})
                instance = labels.get('instance', '')
                node = instance.split(':')[0] if ':' in instance else instance
                gpu_id = labels.get('gpu', labels.get('gpu_id', 'unknown'))
                value = item.get('value', [None, None])[1]
                
                try:
                    if value is not None:
                        value = float(value)
                except (ValueError, TypeError):
                    pass
                
                if node not in metrics_dict:
                    metrics_dict[node] = {}
                if gpu_id not in metrics_dict[node]:
                    metrics_dict[node][gpu_id] = {}
                
                metrics_dict[node][gpu_id][metric_key] = value
        else:
            log.warning(f"Failed to query metric {metric_key}: {result.get('error', 'Unknown error')}")
    
    log.info(f"Retrieved metrics for {len(metrics_dict)} nodes, {len(metrics)} metric types")
    return metrics_dict


def get_device_exporter_health(prom_client: PrometheusClient,
                               node_list: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Check health status of Device Metrics Exporter on all nodes.
    """
    health_dict = {}
    targets = prom_client.get_targets()
    
    for target in targets:
        labels = target.get('labels', {})
        instance = labels.get('instance', '')
        job = labels.get('job', '')
        
        if 'device-metrics' not in job.lower() and 'amd' not in job.lower():
            continue
        
        node = instance.split(':')[0] if ':' in instance else instance
        
        if node_list and node not in node_list:
            continue
        
        health_dict[node] = {
            'health': target.get('health', 'unknown'),
            'last_scrape': target.get('lastScrape', ''),
            'scrape_duration': target.get('lastScrapeDuration', 0),
            'last_error': target.get('lastError', ''),
            'scrape_url': target.get('scrapeUrl', ''),
            'labels': labels
        }
    
    up_count = sum(1 for h in health_dict.values() if h['health'] == 'up')
    down_count = sum(1 for h in health_dict.values() if h['health'] == 'down')
    log.info(f"Exporter health: {up_count} up, {down_count} down out of {len(health_dict)} nodes")
    
    return health_dict


def create_grafana_annotation(grafana_url: str, api_key: str, 
                              text: str, tags: List[str],
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> bool:
    """Create an annotation in Grafana to mark CVS test events."""
    try:
        url = f"{grafana_url.rstrip('/')}/api/annotations"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        if start_time is None:
            start_time = datetime.now()
        
        data = {
            'text': text,
            'tags': tags,
            'time': int(start_time.timestamp() * 1000)
        }
        
        if end_time:
            data['timeEnd'] = int(end_time.timestamp() * 1000)
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        
        log.info(f"✓ Created Grafana annotation: {text}")
        return True
        
    except Exception as e:
        log.error(f"✗ Failed to create Grafana annotation: {e}")
        return False


def compare_ssh_vs_prometheus(ssh_metrics: Dict, prom_metrics: Dict,
                              tolerance: float = 5.0) -> Dict:
    """Compare metrics collected via SSH vs Prometheus to validate consistency."""
    comparison = {
        'summary': {
            'total_nodes': 0,
            'matching_nodes': 0,
            'discrepancy_nodes': 0,
            'ssh_only_nodes': 0,
            'prom_only_nodes': 0
        },
        'node_comparisons': [],
        'discrepancies': []
    }
    
    ssh_nodes = set(ssh_metrics.keys())
    prom_nodes = set(prom_metrics.keys())
    
    comparison['summary']['total_nodes'] = len(ssh_nodes | prom_nodes)
    comparison['summary']['ssh_only_nodes'] = len(ssh_nodes - prom_nodes)
    comparison['summary']['prom_only_nodes'] = len(prom_nodes - ssh_nodes)
    
    for node in (ssh_nodes - prom_nodes):
        log.warning(f"Node {node} only in SSH metrics (not in Prometheus)")
        comparison['node_comparisons'].append({
            'node': node,
            'status': 'ssh_only',
            'gpu_count_match': False
        })
    
    for node in (prom_nodes - ssh_nodes):
        log.warning(f"Node {node} only in Prometheus metrics (not in SSH)")
        comparison['node_comparisons'].append({
            'node': node,
            'status': 'prom_only',
            'gpu_count_match': False
        })
    
    common_nodes = ssh_nodes & prom_nodes
    
    for node in common_nodes:
        node_comparison = {
            'node': node,
            'status': 'match',
            'gpu_count_match': True,
            'metric_comparisons': []
        }
        
        ssh_gpus = set(ssh_metrics[node].keys())
        prom_gpus = set(prom_metrics[node].keys())
        
        if ssh_gpus != prom_gpus:
            node_comparison['gpu_count_match'] = False
            node_comparison['status'] = 'discrepancy'
            log.warning(f"Node {node}: GPU count mismatch")
        
        common_gpus = ssh_gpus & prom_gpus
        for gpu_id in common_gpus:
            ssh_gpu = ssh_metrics[node][gpu_id]
            prom_gpu = prom_metrics[node][gpu_id]
            
            ssh_metric_keys = set(ssh_gpu.keys())
            prom_metric_keys = set(prom_gpu.keys())
            common_metrics = ssh_metric_keys & prom_metric_keys
            
            for metric_key in common_metrics:
                ssh_val = ssh_gpu[metric_key]
                prom_val = prom_gpu[metric_key]
                
                if ssh_val is None or prom_val is None:
                    continue
                
                try:
                    ssh_num = float(ssh_val)
                    prom_num = float(prom_val)
                    
                    if ssh_num != 0:
                        diff_percent = abs((prom_num - ssh_num) / ssh_num) * 100
                    else:
                        diff_percent = 0 if prom_num == 0 else 100
                    
                    if diff_percent > tolerance:
                        node_comparison['status'] = 'discrepancy'
                        comparison['discrepancies'].append({
                            'node': node,
                            'gpu': str(gpu_id),
                            'metric': metric_key,
                            'ssh_value': ssh_num,
                            'prom_value': prom_num,
                            'diff_percent': round(diff_percent, 2)
                        })
                except (ValueError, TypeError):
                    if str(ssh_val) != str(prom_val):
                        node_comparison['status'] = 'discrepancy'
                        comparison['discrepancies'].append({
                            'node': node,
                            'gpu': str(gpu_id),
                            'metric': metric_key,
                            'ssh_value': str(ssh_val),
                            'prom_value': str(prom_val),
                            'diff_percent': None
                        })
        
        comparison['node_comparisons'].append(node_comparison)
        
        if node_comparison['status'] == 'match':
            comparison['summary']['matching_nodes'] += 1
        else:
            comparison['summary']['discrepancy_nodes'] += 1
    
    log.info(f"Comparison complete: {comparison['summary']['matching_nodes']}/{len(common_nodes)} nodes match")
    if comparison['discrepancies']:
        log.warning(f"Found {len(comparison['discrepancies'])} metric discrepancies")
    
    return comparison
