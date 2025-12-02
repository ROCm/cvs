'''
Copyright 2025 Advanced Micro Devices, Inc.
Device Metrics Integration Library for CVS
'''

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

log = logging.getLogger(__name__)


class PrometheusClient:
    """Client for querying Prometheus API to retrieve GPU metrics."""
    
    def __init__(self, prometheus_url: str, timeout: int = 30):
        self.base_url = prometheus_url.rstrip('/')
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/v1"
        
    def check_health(self) -> bool:
        """Check if Prometheus server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/-/healthy", timeout=5)
            return response.status_code == 200
        except Exception as e:
            log.error(f"Prometheus health check failed: {e}")
            return False
    
    def query_instant(self, query: str, time: Optional[str] = None) -> Optional[Dict]:
        """Execute instant Prometheus query."""
        params = {'query': query}
        if time:
            params['time'] = time
            
        try:
            response = requests.get(
                f"{self.api_url}/query",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'success':
                return data.get('data')
            else:
                log.error(f"Prometheus query failed: {data.get('error')}")
                return None
                
        except Exception as e:
            log.error(f"Error querying Prometheus: {e}")
            return None
    
    def query_range(self, query: str, start: str, end: str, step: str = '15s') -> Optional[Dict]:
        """Execute range Prometheus query for time-series data."""
        params = {
            'query': query,
            'start': start,
            'end': end,
            'step': step
        }
        
        try:
            response = requests.get(
                f"{self.api_url}/query_range",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'success':
                return data.get('data')
            else:
                log.error(f"Prometheus range query failed: {data.get('error')}")
                return None
                
        except Exception as e:
            log.error(f"Error querying Prometheus range: {e}")
            return None


def get_gpu_metrics_from_prometheus(
    prom_client: PrometheusClient,
    node: str,
    gpu_ids: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Retrieve GPU metrics from Prometheus for a specific node.
    
    Returns:
        {
            '0': {'temperature': 45.0, 'power': 300.5, 'utilization': 85.0},
            '1': {'temperature': 46.0, 'power': 295.3, 'utilization': 82.0}
        }
    """
    metrics_dict = {}
    
    # Query temperature
    temp_query = f'amdgpu_temperature_celsius{{node="{node}", sensor="edge"}}'
    temp_data = prom_client.query_instant(temp_query)
    
    if temp_data and temp_data.get('result'):
        for result in temp_data['result']:
            gpu_id = result['metric'].get('gpu', 'unknown')
            if gpu_ids is None or gpu_id in gpu_ids:
                if gpu_id not in metrics_dict:
                    metrics_dict[gpu_id] = {}
                metrics_dict[gpu_id]['temperature'] = float(result['value'][1])
    
    # Query power consumption
    power_query = f'amdgpu_power_watts{{node="{node}"}}'
    power_data = prom_client.query_instant(power_query)
    
    if power_data and power_data.get('result'):
        for result in power_data['result']:
            gpu_id = result['metric'].get('gpu', 'unknown')
            if gpu_ids is None or gpu_id in gpu_ids:
                if gpu_id not in metrics_dict:
                    metrics_dict[gpu_id] = {}
                metrics_dict[gpu_id]['power'] = float(result['value'][1])
    
    # Query GPU utilization
    util_query = f'amdgpu_gpu_busy_percent{{node="{node}"}}'
    util_data = prom_client.query_instant(util_query)
    
    if util_data and util_data.get('result'):
        for result in util_data['result']:
            gpu_id = result['metric'].get('gpu', 'unknown')
            if gpu_ids is None or gpu_id in gpu_ids:
                if gpu_id not in metrics_dict:
                    metrics_dict[gpu_id] = {}
                metrics_dict[gpu_id]['utilization'] = float(result['value'][1])
    
    return metrics_dict


def get_device_exporter_health(
    prom_client: PrometheusClient,
    nodes: List[str]
) -> Dict[str, bool]:
    """Check health status of Device Metrics Exporter on all nodes."""
    health_dict = {}
    
    for node in nodes:
        query = f'up{{job="device-metrics-exporter", node="{node}"}}'
        data = prom_client.query_instant(query)
        
        if data and data.get('result'):
            is_up = float(data['result'][0]['value'][1]) == 1.0
            health_dict[node] = is_up
        else:
            health_dict[node] = False
    
    return health_dict


def create_grafana_annotation(
    grafana_url: str,
    text: str,
    tags: List[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    time: Optional[int] = None
) -> bool:
    """Create annotation in Grafana to mark test events on dashboards."""
    if tags is None:
        tags = ['cvs-test']
    
    if time is None:
        time = int(datetime.now().timestamp() * 1000)
    
    url = f"{grafana_url.rstrip('/')}/api/annotations"
    
    payload = {
        'text': text,
        'tags': tags,
        'time': time
    }
    
    headers = {'Content-Type': 'application/json'}
    
    if not api_key and (not username or not password):
        log.warning("Grafana annotation requested without credentials or API key; skipping.")
        return False
    
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
        auth = None
    else:
        auth = (username, password)
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            auth=auth,
            timeout=10
        )
        response.raise_for_status()
        log.info(f"Created Grafana annotation: {text}")
        return True
        
    except Exception as e:
        log.error(f"Failed to create Grafana annotation: {e}")
        return False


# Test function
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python device_metrics_lib.py <prometheus_url> <node>")
        print("Example: python device_metrics_lib.py http://localhost:9090 localhost")
        sys.exit(1)
    
    prometheus_url = sys.argv[1]
    node = sys.argv[2]
    
    print(f"Testing Prometheus integration with {prometheus_url}")
    
    client = PrometheusClient(prometheus_url)
    
    if not client.check_health():
        print("ERROR: Prometheus server is not healthy")
        sys.exit(1)
    print("Prometheus server is healthy")
    
    metrics = get_gpu_metrics_from_prometheus(client, node)
    if metrics:
        print(f"Retrieved metrics for {len(metrics)} GPUs")
        for gpu_id, data in metrics.items():
            print(f"  GPU {gpu_id}: Temp={data.get('temperature', 'N/A')}°C, "
                  f"Power={data.get('power', 'N/A')}W")
    else:
        print("WARNING: No GPU metrics found")
