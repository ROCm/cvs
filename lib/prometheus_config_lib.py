'''
Copyright 2025 Advanced Micro Devices, Inc.
Prometheus Configuration Generator for CVS Monitoring
'''

import json
import yaml
import logging

log = logging.getLogger(__name__)


def generate_prometheus_config(cluster_dict, config_dict, output_file=None):
    """
    Generate Prometheus configuration with dynamic scrape targets.
    
    Args:
        cluster_dict: Cluster configuration
        config_dict: Monitoring configuration
        output_file: Optional output file path
    
    Returns:
        str: YAML configuration content
    """
    from utils_lib import generate_prometheus_targets
    
    # Get configuration values
    scrape_interval = config_dict.get('scrape_interval', '15s')
    scrape_timeout = config_dict.get('scrape_timeout', '10s')
    retention_days = config_dict.get('retention_days', 30)
    exporter_port = config_dict.get('device_metrics_exporter_port', 5000)
    
    # Generate targets for all nodes (management + workers)
    targets = generate_prometheus_targets(cluster_dict, exporter_port)
    
    log.info(f"Generating Prometheus config for {len(targets)} targets")
    for target in targets:
        log.info(f"  • {target}")
    
    # Build Prometheus configuration
    config = {
        'global': {
            'scrape_interval': scrape_interval,
            'scrape_timeout': scrape_timeout,
            'evaluation_interval': scrape_interval
        },
        'scrape_configs': [
            {
                'job_name': 'device-metrics-exporter',
                'static_configs': [
                    {
                        'targets': targets
                    }
                ],
                'metric_relabel_configs': [
                    {
                        'source_labels': ['__name__'],
                        'regex': 'gpu_.*',
                        'action': 'keep'
                    }
                ]
            }
        ]
    }
    
    # Convert to YAML
    yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(yaml_content)
        log.info(f"Prometheus config written to: {output_file}")
    
    return yaml_content


def update_prometheus_targets(prometheus_yml_path, cluster_dict, exporter_port=5000):
    """
    Update existing Prometheus config with new targets.
    
    Args:
        prometheus_yml_path: Path to prometheus.yml
        cluster_dict: Cluster configuration
        exporter_port: Exporter port (default: 5000)
    """
    from utils_lib import generate_prometheus_targets
    
    # Load existing config
    with open(prometheus_yml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate new targets
    targets = generate_prometheus_targets(cluster_dict, exporter_port)
    
    # Update targets in scrape config
    for scrape_config in config.get('scrape_configs', []):
        if scrape_config.get('job_name') == 'device-metrics-exporter':
            scrape_config['static_configs'] = [{'targets': targets}]
            log.info(f"Updated scrape targets: {targets}")
            break
    
    # Write back
    with open(prometheus_yml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    log.info(f"Prometheus config updated: {prometheus_yml_path}")
