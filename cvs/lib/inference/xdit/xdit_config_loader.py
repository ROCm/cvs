"""
xDiT configuration loader for unified and legacy inference configurations.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
from pathlib import Path

from pydantic import Field, create_model

from cvs.lib.utils.config_loader import (
    BaseVariantConfig,
    _Allow,
    _flatten_paths,
    _walk_substitute,
    substitute_config,
)
from cvs.lib.utils_lib import resolve_test_config_placeholders


XditParams = create_model(
    "XditParams",
    __base__=_Allow,
    flux1_dev_t2i=(dict, Field(default_factory=dict)),
    wan22_i2v_a14b=(dict, Field(default_factory=dict)),
)

XditVariantConfig = create_model(
    "XditVariantConfig",
    __base__=BaseVariantConfig,
    framework=(str, "xdit"),
    gpu_arch=(str, "mi3xx"),
    topology=(str, "single"),
    config_path=(str, ""),
    params=(XditParams, Field(default_factory=XditParams)),
    inference=(dict, Field(default_factory=dict)),
    benchmark_params=(dict, Field(default_factory=dict)),
)


def _mounts_to_volume_dict(mounts):
    volume_dict = {}
    for mount in mounts or []:
        host, separator, container = str(mount).partition(":")
        if separator and host and container:
            volume_dict[host] = container
    return volume_dict


def _volume_dict_to_mounts(volume_dict):
    return [f"{host}:{container}" for host, container in (volume_dict or {}).items()]


def _mounted_path(volume_dict, host_path, default):
    normalized = str(host_path or "").rstrip("/")
    best = None
    for host, container in (volume_dict or {}).items():
        host = str(host).rstrip("/")
        if normalized == host or normalized.startswith(host + "/"):
            if best is None or len(host) > len(best[0]):
                best = (host, str(container).rstrip("/"))
    if best is None:
        return default
    return best[1] + normalized[len(best[0]) :]


_INFERENCE_TO_ENV = {
    "nccl_ib_hca": "NCCL_IB_HCA",
    "nccl_ib_gid_index": "NCCL_IB_GID_INDEX",
    "nccl_socket_ifname": "NCCL_SOCKET_IFNAME",
    "gloo_socket_ifname": "GLOO_SOCKET_IFNAME",
    "gloo_tcp_ifname": "GLOO_TCP_IFNAME",
    "nccl_debug": "NCCL_DEBUG",
}
_ENV_TO_INFERENCE = {target: source for source, target in _INFERENCE_TO_ENV.items()}


def _container_env(inference):
    env = dict((inference.get("container_config") or {}).get("env_dict") or {})
    for source, target in _INFERENCE_TO_ENV.items():
        value = inference.get(source)
        if value is not None and str(value).strip():
            env[target] = str(value)
    return {str(key): str(value) for key, value in env.items() if value is not None}


def _merged_container_env(container):
    runtime_args = dict((container.get("runtime") or {}).get("args") or {})
    env = dict(container.get("env") or {})
    env.update(dict(runtime_args.get("env") or {}))
    return {str(key): str(value) for key, value in env.items() if value is not None}


def _flatten_orchestrator_container(container):
    block = dict(container)
    runtime = dict(block.get("runtime") or {})
    runtime_args = dict(runtime.get("args") or {})
    env = _merged_container_env(block)
    runtime_args.pop("env", None)
    runtime["args"] = runtime_args
    block["runtime"] = runtime
    block["env"] = env
    return block


_LOCAL_MODEL_MOUNT = "/model"


def _append_container_volume(raw, mount):
    container = dict(raw.get("container") or {})
    runtime = dict(container.get("runtime") or {})
    args = dict(runtime.get("args") or {})
    volumes = list(args.get("volumes") or [])
    if mount not in volumes:
        volumes.append(mount)
    args["volumes"] = volumes
    runtime["args"] = args
    container["runtime"] = runtime
    raw["container"] = container


def _legacy_container(inference, volume_dict):
    container_config = inference.get("container_config") or {}
    runtime_args = {
        "network": "host",
        "ipc": "host",
        "privileged": True,
        "volumes": _volume_dict_to_mounts(volume_dict),
        "devices": list(container_config.get("device_list") or []),
        "cap_add": ["SYS_PTRACE"],
        "security_opt": ["seccomp=unconfined"],
    }
    return {
        "lifetime": inference.get("container_lifetime", "per_run"),
        "name": inference["container_name"],
        "image": inference["container_image"],
        "env": _container_env(inference),
        "runtime": {"name": "docker", "args": runtime_args},
    }


def _legacy_paths(inference):
    output_base = str(inference["output_base_dir"]).rstrip("/")
    hf_home = str(inference["hf_home"]).rstrip("/")
    return {
        "shared_fs": str(Path(output_base).parent),
        "models_dir": hf_home,
        "log_dir": output_base,
        "hf_token_file": str(inference["hf_token_file"]),
    }


def _legacy_runtime_views(config, benchmark_params):
    inference = dict(config)
    container_config = dict(inference.get("container_config") or {})
    volume_dict = dict(container_config.get("volume_dict") or {})
    hf_home = str(inference["hf_home"]).rstrip("/")
    output_base = str(inference["output_base_dir"]).rstrip("/")
    volume_dict.setdefault(hf_home, "/hf_home")
    volume_dict.setdefault(output_base, "/outputs")

    model_repo = str(inference["model_repo"])
    if model_repo.startswith("/"):
        volume_dict.setdefault(model_repo.rstrip("/"), "/model")
        inference["_resolved_model_mount_host"] = model_repo.rstrip("/")
        inference["_resolved_model_path_container"] = "/model"
        inference["_resolved_ckpt_dir_container"] = "/model"

    container_config["volume_dict"] = volume_dict
    inference["container_config"] = container_config
    inference["output_base_dir_container"] = _mounted_path(volume_dict, output_base, "/outputs")
    return inference, benchmark_params, volume_dict


def _load_without_threshold(config_path, cluster_dict):
    with open(config_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    raw = resolve_test_config_placeholders(raw, cluster_dict)
    paths = dict(raw.get("paths") or {})
    for _ in range(len(paths) + 1):
        updated = {
            key: _walk_substitute(value, {name: item for name, item in paths.items() if isinstance(item, str)})
            for key, value in paths.items()
        }
        if updated == paths:
            break
        paths = updated
    raw["paths"] = paths
    raw = _walk_substitute(raw, _flatten_paths({"paths": paths}))
    return {key: value for key, value in raw.items() if not str(key).startswith("_")}, {}


def _read_unified(config_path, cluster_dict):
    with open(config_path, encoding="utf-8") as handle:
        peek = json.load(handle)
    has_sibling_threshold = bool(list(Path(config_path).parent.glob("*threshold.json")))
    if peek.get("threshold_json") or has_sibling_threshold:
        raw, thresholds = substitute_config(config_path, cluster_dict)
        raw = resolve_test_config_placeholders(raw, cluster_dict)
        return raw, thresholds
    return _load_without_threshold(config_path, cluster_dict)


def _workload_key(params):
    if params.get("model_format") == "diffusers" or params.get("wan_diffusers_launcher"):
        return "wan22_i2v_a14b"
    if "ulysses_degree" in params or "height" in params:
        return "flux1_dev_t2i"
    if "frame_num" in params or "size" in params:
        return "wan22_i2v_a14b"
    raise ValueError("cannot infer xDiT workload from benchmark_params")


def _normalize_root(raw):
    raw = dict(raw)
    server = dict(raw.get("server_params") or {})
    if server:
        model_id = server.get("model") or ""
        if model_id and not raw.get("model"):
            raw["model"] = {"id": str(model_id), "remote": 0}
        nnodes = int(server.get("nnodes") or raw.get("nnodes") or 1)
        raw.setdefault("nnodes", nnodes)
        raw.setdefault("topology", "distributed" if nnodes > 1 else "single")
        for key in ("benchmark_serv_node", "master_addr", "master_port", "server_node_list"):
            if key in server and key not in raw:
                raw[key] = server[key]
        inference = dict(raw.get("inference") or {})
        if server.get("model_rev"):
            inference.setdefault("model_rev", server["model_rev"])
        if server.get("model"):
            inference.setdefault("model_repo", server["model"])
        raw["inference"] = inference
        raw.setdefault("gpu_arch", str(raw.get("gpu_name") or "mi3xx"))
        raw.setdefault("framework", str(server.get("backend") or "xdit"))

    params = dict(raw.get("params") or {})
    nested = any(isinstance(params.get(key), dict) for key in ("flux1_dev_t2i", "wan22_i2v_a14b"))
    bp = raw.get("benchmark_params")
    if not nested:
        if isinstance(bp, dict) and any(isinstance(bp.get(key), dict) for key in ("flux1_dev_t2i", "wan22_i2v_a14b")):
            params = dict(bp)
        elif isinstance(bp, dict) and bp:
            params = {_workload_key(bp): dict(bp)}
        raw["params"] = params
        raw["benchmark_params"] = params
    return raw


def _unified_runtime_views(raw):
    paths = dict(raw.get("paths") or {})
    container = dict(raw.get("container") or {})
    runtime_args = dict((container.get("runtime") or {}).get("args") or {})
    volume_dict = _mounts_to_volume_dict(runtime_args.get("volumes") or [])
    model = dict(raw.get("model") or {})
    params = dict(raw.get("params") or raw.get("benchmark_params") or {})
    thresholds = dict(raw.get("thresholds") or {})
    if thresholds:
        workload_keys = [key for key in ("flux1_dev_t2i", "wan22_i2v_a14b") if isinstance(params.get(key), dict)]
        if len(workload_keys) != 1:
            raise ValueError(
                "unified xDiT config must define exactly one FLUX or WAN workload when threshold_json is used"
            )
        workload = dict(params[workload_keys[0]])
        workload["expected_results"] = thresholds
        params[workload_keys[0]] = workload

    inference = dict(raw.get("inference") or {})
    inference.setdefault("container_image", container.get("image"))
    inference.setdefault("container_name", container.get("name"))
    inference.setdefault("hf_token_file", paths.get("hf_token_file"))
    inference.setdefault("hf_home", paths.get("models_dir"))
    inference.setdefault(
        "hf_home_container",
        _mounted_path(volume_dict, paths.get("models_dir"), "/hf_home"),
    )
    env_dict = _merged_container_env(container)
    env_dict.setdefault("HF_HOME", inference["hf_home_container"])
    runtime = dict(container.get("runtime") or {})
    args = dict(runtime.get("args") or {})
    args["env"] = env_dict
    runtime["args"] = args
    container["runtime"] = runtime
    raw["container"] = container
    inference.setdefault("output_base_dir", paths.get("log_dir"))
    inference.setdefault(
        "output_base_dir_container",
        _mounted_path(volume_dict, paths.get("log_dir"), paths.get("log_dir")),
    )
    inference.setdefault("model_repo", model.get("id"))
    inference.setdefault("model_rev", "")
    env_dict = _merged_container_env(container)
    inference["container_config"] = {
        "device_list": list(runtime_args.get("devices") or []),
        "volume_dict": volume_dict,
        "env_dict": env_dict,
    }
    for env_key, inference_key in _ENV_TO_INFERENCE.items():
        if inference_key not in inference and env_dict.get(env_key):
            inference[inference_key] = env_dict[env_key]

    model_id = str(model.get("id") or "").rstrip("/")
    if model_id.startswith("/"):
        mounted_model = _mounted_path(volume_dict, model_id, "")
        if not mounted_model:
            # A host model directory is invisible to the workload unless it is
            # bind-mounted, so cover it instead of passing the host path through.
            mounted_model = _LOCAL_MODEL_MOUNT
            volume_dict[model_id] = mounted_model
            _append_container_volume(raw, f"{model_id}:{mounted_model}")
        inference.setdefault("_resolved_model_mount_host", model_id)
        inference.setdefault("_resolved_model_path_container", mounted_model)
        inference.setdefault("_resolved_ckpt_dir_container", mounted_model)

    topology = str(raw.get("topology") or "")
    if topology == "distributed":
        inference.setdefault("nnodes", int(raw.get("nnodes") or 2))
    for key in (
        "nnodes",
        "master_port",
        "nccl_ib_hca",
        "nccl_ib_gid_index",
        "nccl_socket_ifname",
        "gloo_socket_ifname",
        "gloo_tcp_ifname",
        "nccl_debug",
    ):
        if key in raw and key not in inference:
            inference[key] = raw[key]
    return inference, params


def _is_legacy_root(raw):
    return isinstance(raw.get("config"), dict) and isinstance(raw.get("benchmark_params"), dict)


def orchestrator_container_from_variant(variant):
    return _flatten_orchestrator_container(variant.container.model_dump())


def _apply_distributed_host_scope(inference, cluster_dict, topology):
    if topology != "distributed":
        return
    from cvs.lib.inference.xdit.xdit_flux_job import resolve_distributed_execution_hosts

    cluster_dict = cluster_dict or {}
    nnodes = int(inference.get("nnodes") or 0)
    if nnodes < 2:
        raise ValueError(f"distributed xDiT requires nnodes >= 2, got {nnodes}")
    if cluster_dict.get("node_dict"):
        inference["_execution_hosts"] = resolve_distributed_execution_hosts(cluster_dict, inference)


def load_variant(config_path, cluster_dict):
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"variant config not found: {path}")

    with open(path, encoding="utf-8") as handle:
        peek = json.load(handle)

    if _is_legacy_root(peek):
        config = resolve_test_config_placeholders(peek["config"], cluster_dict)
        benchmark_params = resolve_test_config_placeholders(peek["benchmark_params"], cluster_dict)
        inference, benchmark_params, volume_dict = _legacy_runtime_views(config, benchmark_params)
        topology = "distributed" if int(inference.get("nnodes") or 1) > 1 else "single"
        _apply_distributed_host_scope(inference, cluster_dict, topology)
        raw = {
            "schema_version": 1,
            "framework": "xdit",
            "gpu_arch": str(peek.get("gpu_arch") or "mi3xx"),
            "topology": topology,
            "paths": _legacy_paths(inference),
            "model": {"id": str(inference["model_repo"]), "remote": 0},
            "container": _legacy_container(inference, volume_dict),
            "thresholds": {},
            "config_path": str(path.resolve()),
            "params": benchmark_params,
            "inference": inference,
            "benchmark_params": benchmark_params,
        }
        return XditVariantConfig(**raw)

    raw, thresholds = _read_unified(str(path), cluster_dict)
    raw = _normalize_root(raw)
    framework = raw.get("framework")
    if framework not in (None, "xdit", "pytorch_xdit"):
        raise ValueError(f"unsupported framework {framework!r} in {config_path!r}; expected 'xdit'")

    raw["container"] = {
        key: value for key, value in dict(raw.get("container") or {}).items() if not str(key).startswith("_")
    }
    raw["thresholds"] = thresholds
    inference, benchmark_params = _unified_runtime_views(raw)
    topology = str(raw.get("topology") or ("distributed" if int(inference.get("nnodes") or 1) > 1 else "single"))
    _apply_distributed_host_scope(inference, cluster_dict, topology)
    known = {
        "schema_version": raw.get("schema_version", 1),
        "framework": "xdit",
        "gpu_arch": str(raw.get("gpu_arch") or raw.get("gpu_name") or "mi3xx"),
        "topology": topology,
        "paths": raw.get("paths"),
        "model": raw.get("model"),
        "container": raw.get("container"),
        "threshold_json": str(raw.get("threshold_json") or ""),
        "thresholds": thresholds,
        "enforce_thresholds": raw.get("enforce_thresholds", True),
        "config_path": str(path.resolve()),
        "params": benchmark_params,
        "inference": inference,
        "benchmark_params": benchmark_params,
    }
    return XditVariantConfig(**known)
