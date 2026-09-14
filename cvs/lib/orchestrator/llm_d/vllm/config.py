'''Configuration loading and validation for the llm-d vLLM suite.'''

import json
from pathlib import Path

from cvs.lib.utils_lib import resolve_test_config_placeholders


class ConfigSection:
    '''Attribute access for a validated configuration mapping.'''

    def __init__(self, values):
        self._values = values

    def __getattr__(self, name):
        try:
            value = self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict):
            return ConfigSection(value)
        return value

    def __getitem__(self, name):
        return self._values[name]

    def get(self, name, default=None):
        return self._values.get(name, default)

    def as_dict(self):
        return dict(self._values)


def _resolve_path_references(raw):
    paths = raw.get("paths", {})
    for _ in range(len(paths) + 1):
        changed = False
        for key, value in paths.items():
            if not isinstance(value, str):
                continue
            resolved = value
            for path_key, path_value in paths.items():
                if isinstance(path_value, str):
                    resolved = resolved.replace("{" + path_key + "}", path_value)
            changed = changed or resolved != value
            paths[key] = resolved
        if not changed:
            break

    def walk(value):
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, str):
            for key, path_value in paths.items():
                if isinstance(path_value, str):
                    value = value.replace("{paths." + key + "}", path_value)
            return value
        return value

    return walk(raw)


def _require(mapping, keys, section):
    missing = [key for key in keys if mapping.get(key) in (None, "", [])]
    if missing:
        raise ValueError(f"{section} missing required fields: {missing}")


def _validate(raw, cluster_dict):
    _require(raw, ("paths", "container", "server_params", "gateway", "workers"), "config")
    _require(raw["paths"], ("shared_fs", "models_dir", "log_dir", "hf_token_file"), "paths")
    _require(raw["container"], ("image", "runtime"), "container")
    _require(
        raw["server_params"],
        ("model", "served_model_name", "tensor_parallel_size", "hip_visible_devices"),
        "server_params",
    )
    _require(
        raw["gateway"],
        (
            "node",
            "listen_port",
            "admin_port",
            "epp_grpc_port",
            "epp_grpc_health_port",
            "epp_metrics_port",
            "envoy_image",
            "epp_image",
            "epp_version",
        ),
        "gateway",
    )
    if len(raw["workers"]) < 2:
        raise ValueError("llm-d vLLM requires at least two workers")

    nodes = cluster_dict.get("node_dict") or {}
    names = set()
    endpoints = set()
    for index, worker in enumerate(raw["workers"]):
        _require(worker, ("name", "node", "port"), f"workers[{index}]")
        if worker["name"] in names:
            raise ValueError(f"duplicate worker name: {worker['name']}")
        endpoint = (worker["node"], int(worker["port"]))
        if endpoint in endpoints:
            raise ValueError(f"duplicate worker endpoint: {endpoint}")
        names.add(worker["name"])
        endpoints.add(endpoint)

    referenced = {raw["gateway"]["node"]} | {worker["node"] for worker in raw["workers"]}
    missing_nodes = sorted(referenced - set(nodes))
    if missing_nodes:
        raise ValueError(f"llm-d nodes are absent from cluster node_dict: {missing_nodes}")


def load_config(config_path, cluster_dict):
    '''Load, substitute, and validate one llm-d vLLM configuration.'''
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"llm-d config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = resolve_test_config_placeholders(raw, cluster_dict)
    raw = _resolve_path_references(raw)
    _validate(raw, cluster_dict)
    return ConfigSection(raw)
