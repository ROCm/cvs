"""
Minimal RCCL runner used by the rccl_cvs pytest suite.
"""

import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cvs.lib import globals
from cvs.lib.utils_lib import fail_test, resolve_test_config_placeholders
from cvs.lib.verify_lib import verify_dmesg_for_errors
from cvs.schema.rccl import RcclTests, RcclTestsMultinodeRaw

log = globals.log

DEFAULT_COLLECTIVES = [
    "all_reduce_perf",
    "all_gather_perf",
    "scatter_perf",
    "gather_perf",
    "reduce_scatter_perf",
    "sendrecv_perf",
    "alltoall_perf",
    "alltoallv_perf",
    "broadcast_perf",
]

RCCL_ERROR_PATTERNS = {
    "orte": r"ORTE does not know how to route|ORTE was unable to reliably start",
    "nccl": r"NCCL ERROR|Test failure",
    "fs": r"No such file or directory",
}


@dataclass(frozen=True)
class RcclConfig:
    mode: str
    collectives: list[str]
    datatype: str
    num_ranks: int
    ranks_per_node: int
    rccl_tests_dir: str
    rocm_path: str
    mpi_root: str | None
    mpirun_path: str | None
    env_script: str | None
    output_json: str
    start_size: str
    end_size: str
    step_factor: str
    warmups: str
    iterations: str
    cycles: str
    verify_bus_bw: bool
    verify_bw_dip: bool
    verify_lat_dip: bool
    thresholds: dict[str, Any]
    rccl_library_path: str | None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coalesce(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] not in (None, ""):
            return config[key]
    return default


def _as_collective_list(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_COLLECTIVES)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("RCCL collectives must be a string or list")


def _normalize_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    aliases = {
        "single_node": "single_node",
        "single-node": "single_node",
        "singlenode": "single_node",
        "multi_node": "multi_node",
        "multi-node": "multi_node",
        "multinode": "multi_node",
    }
    if mode not in aliases:
        raise ValueError("RCCL mode must be 'single_node' or 'multi_node'")
    return aliases[mode]


def _load_thresholds(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("results") is not None:
        return config["results"]

    results_file = config.get("results_file")
    if not results_file:
        return {}

    with open(results_file) as handle:
        return json.load(handle)


def _normalize_datatype(config: dict[str, Any]) -> str:
    explicit = _coalesce(config, "datatype", "data_type")
    if explicit:
        return str(explicit)

    maybe_list = _coalesce(config, "data_types", "data_type_list")
    if maybe_list is None:
        return "float"
    if isinstance(maybe_list, list) and len(maybe_list) == 1:
        return str(maybe_list[0])
    raise ValueError("Use a single RCCL datatype in the simplified rccl_cvs config")


def load_rccl_config(config_file: str, cluster_dict: dict[str, Any]) -> RcclConfig:
    with open(config_file) as handle:
        raw = json.load(handle)

    if "rccl" not in raw or not isinstance(raw["rccl"], dict):
        raise ValueError("RCCL config file must contain a top-level 'rccl' object")

    config = resolve_test_config_placeholders(raw["rccl"], cluster_dict)
    mode = _normalize_mode(_coalesce(config, "mode", default="multi_node"))
    collectives = _as_collective_list(_coalesce(config, "collectives", "rccl_collective"))
    datatype = _normalize_datatype(config)

    num_ranks = int(
        _coalesce(
            config,
            "num_ranks",
            "no_of_global_ranks",
            "no_of_local_ranks",
            "ranks_per_node",
            default=1,
        )
    )
    ranks_per_node = int(_coalesce(config, "ranks_per_node", "no_of_local_ranks", default=num_ranks))
    if num_ranks <= 0 or ranks_per_node <= 0:
        raise ValueError("num_ranks and ranks_per_node must be positive integers")

    if mode == "multi_node" and num_ranks % ranks_per_node != 0:
        raise ValueError("num_ranks must be divisible by ranks_per_node for multi_node RCCL runs")

    rocm_path = str(_coalesce(config, "rocm_path", "rocm_path_var", default="/opt/rocm"))
    rccl_tests_dir = _coalesce(config, "rccl_tests_dir")
    if not rccl_tests_dir:
        raise ValueError("rccl_tests_dir is required")
    rccl_tests_dir = str(rccl_tests_dir)

    mpi_root = _coalesce(config, "mpi_root", "mpi_path_var")
    mpirun_path = _coalesce(config, "mpirun_path")
    mpi_dir = _coalesce(config, "mpi_dir")
    if mpirun_path is None:
        if mpi_root:
            mpirun_path = f"{mpi_root}/bin/mpirun"
        elif mpi_dir:
            mpirun_path = f"{mpi_dir.rstrip('/')}/mpirun"

    if mode == "multi_node" and not mpirun_path:
        raise ValueError("multi_node RCCL runs require either mpi_root or mpirun_path")

    env_script = _coalesce(config, "env_script", "env_source_script")
    if env_script and str(env_script).strip().lower() == "none":
        env_script = None

    output_json = str(_coalesce(config, "output_json", "output_file", "rccl_result_file", default="/tmp/rccl_result.json"))

    return RcclConfig(
        mode=mode,
        collectives=collectives,
        datatype=datatype,
        num_ranks=num_ranks,
        ranks_per_node=ranks_per_node,
        rccl_tests_dir=rccl_tests_dir,
        rocm_path=rocm_path,
        mpi_root=str(mpi_root) if mpi_root else None,
        mpirun_path=str(mpirun_path) if mpirun_path else None,
        env_script=str(env_script) if env_script else None,
        output_json=output_json,
        start_size=str(_coalesce(config, "start_size", "start_msg_size", default="1024")),
        end_size=str(_coalesce(config, "end_size", "end_msg_size", default="16g")),
        step_factor=str(_coalesce(config, "step_factor", "step_function", default="2")),
        warmups=str(_coalesce(config, "warmups", "warmup_iterations", default="10")),
        iterations=str(_coalesce(config, "iterations", "no_of_iterations", default="20")),
        cycles=str(_coalesce(config, "cycles", "no_of_cycles", default="1")),
        verify_bus_bw=_as_bool(config.get("verify_bus_bw"), default=False),
        verify_bw_dip=_as_bool(config.get("verify_bw_dip"), default=True),
        verify_lat_dip=_as_bool(config.get("verify_lat_dip"), default=True),
        thresholds=_load_thresholds(config),
        rccl_library_path=_coalesce(config, "rccl_library_path", "rccl_path_var", "rccl_dir"),
    )


def _ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _select_nodes(cluster_dict: dict[str, Any], config: RcclConfig) -> tuple[list[str], list[str]]:
    cluster_nodes = list(cluster_dict["node_dict"].keys())
    if not cluster_nodes:
        raise ValueError("cluster_file does not contain any nodes")

    if config.mode == "single_node":
        selected_nodes = [cluster_nodes[0]]
    else:
        required_nodes = config.num_ranks // config.ranks_per_node
        if len(cluster_nodes) < required_nodes:
            raise ValueError(
                f"cluster_file provides {len(cluster_nodes)} node(s), but rccl_cvs requires {required_nodes}"
            )
        selected_nodes = cluster_nodes[:required_nodes]

    launch_hosts = [
        cluster_dict["node_dict"][node].get("vpc_ip")
        or cluster_dict["node_dict"][node].get("mgmt_ip")
        or node
        for node in selected_nodes
    ]
    return selected_nodes, launch_hosts


def _build_shell_payload(config: RcclConfig, collective: str, remote_result_file: str) -> str:
    exports = []
    path_parts = []
    if config.mpi_root:
        path_parts.append(f"{config.mpi_root}/bin")
    if config.rocm_path:
        path_parts.append(f"{config.rocm_path}/bin")
    if path_parts:
        exports.append(f'export PATH="{":".join(path_parts)}:$PATH"')

    library_parts = []
    if config.rccl_library_path:
        library_parts.append(config.rccl_library_path.rstrip("/"))
    if config.mpi_root:
        library_parts.append(f"{config.mpi_root}/lib")
    if config.rocm_path:
        library_parts.append(f"{config.rocm_path.rstrip('/')}/lib")
    if library_parts:
        exports.append(f'export LD_LIBRARY_PATH="{":".join(library_parts)}:$LD_LIBRARY_PATH"')

    if config.env_script:
        exports.append(f"source {shlex.quote(config.env_script)}")

    binary = f"{config.rccl_tests_dir.rstrip('/')}/{collective}"
    gpus_per_rank = config.ranks_per_node if config.mode == "single_node" else 1
    benchmark_cmd = (
        f"{shlex.quote(binary)} "
        f"-b {shlex.quote(config.start_size)} "
        f"-e {shlex.quote(config.end_size)} "
        f"-f {shlex.quote(config.step_factor)} "
        f"-g {gpus_per_rank} "
        f"-c 1 "
        f"-w {shlex.quote(config.warmups)} "
        f"-d {shlex.quote(config.datatype)} "
        f"-n {shlex.quote(config.iterations)} "
        f"-N {shlex.quote(config.cycles)} "
        f"-Z json -x {shlex.quote(remote_result_file)}"
    )
    return "; ".join(exports + [benchmark_cmd])


def _prepare_hostfile(shdl: Any, launch_hosts: list[str], ranks_per_node: int) -> str:
    hostfile = f"/tmp/rccl_hosts_{int(time.time())}.txt"
    hostfile_lines = "".join(f"{host} slots={ranks_per_node}\n" for host in launch_hosts)
    shdl.exec(f"rm -f {shlex.quote(hostfile)}")
    shdl.exec(f"printf %s {shlex.quote(hostfile_lines)} > {shlex.quote(hostfile)}")
    return hostfile


def build_collective_command(
    config: RcclConfig, collective: str, remote_result_file: str, launch_hosts: list[str], shdl: Any
) -> str:
    shell_payload = _build_shell_payload(config, collective, remote_result_file)
    shell_cmd = f"bash -lc {shlex.quote(shell_payload)}"

    if config.mode == "single_node":
        return shell_cmd

    hostfile = _prepare_hostfile(shdl, launch_hosts, config.ranks_per_node)
    return (
        f"{shlex.quote(config.mpirun_path or 'mpirun')} "
        f"--np {config.num_ranks} "
        "--allow-run-as-root "
        "--bind-to numa "
        f"--hostfile {shlex.quote(hostfile)} "
        f"{shell_cmd}"
    )


def _scan_rccl_stdout(output: str) -> None:
    warnings = []
    for line in output.splitlines():
        for pattern in RCCL_ERROR_PATTERNS.values():
            if re.search(pattern, line):
                raise RuntimeError(f"RCCL execution failed: {line}")
        if "NCCL WARN" in line:
            warnings.append(line)

    if warnings:
        print("Following warnings were observed in the RCCL test")
        print("#============#")
        print(warnings)
        print("#============#")

    if not re.search(r"#\sAvg bus bandwidth", output):
        raise RuntimeError("RCCL output did not contain '# Avg bus bandwidth'")


def _is_severe_wrong_corruption_error(err: ValidationError) -> bool:
    try:
        for item in err.errors():
            message = item.get("msg", "") or ""
            if "SEVERE DATA CORRUPTION" in message or "'#wrong'" in message or "wrong=" in message:
                return True
    except Exception:
        pass
    return "SEVERE DATA CORRUPTION" in str(err) or "'#wrong'" in str(err)


def _normalize_threshold_map(collective_thresholds: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    if not collective_thresholds:
        return {}

    if "bus_bw" in collective_thresholds and isinstance(collective_thresholds["bus_bw"], dict):
        collective_thresholds = collective_thresholds["bus_bw"]

    normalized: dict[str, dict[str, float]] = {}
    for message_size, value in collective_thresholds.items():
        if isinstance(value, dict):
            bus_bw = value.get("bus_bw")
        else:
            bus_bw = value
        if bus_bw is None:
            continue
        normalized[str(message_size)] = {"bus_bw": float(bus_bw)}
    return normalized


def _matching_rows(collective: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if re.search(r"alltoall|all_to_all", collective, re.I):
        return [row for row in rows if row["inPlace"] == 0]
    return [row for row in rows if row["inPlace"] == 1]


def _check_bus_bw(collective: str, rows: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> None:
    for row in _matching_rows(collective, rows):
        message_size = str(row["size"])
        if message_size not in thresholds:
            continue
        expected = float(thresholds[message_size]["bus_bw"])
        actual = float(row["busBw"])
        if actual < expected * 0.95:
            raise RuntimeError(
                f"{collective} bus BW {actual} for message size {message_size} is below expected {expected}"
            )


def _check_bw_dip(collective: str, rows: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> None:
    if not thresholds:
        return

    previous_bw = 0.0
    previous_size = None
    valid_sizes = set(thresholds.keys())
    for row in _matching_rows(collective, rows):
        if str(row["size"]) not in valid_sizes:
            continue
        current_bw = float(row["busBw"])
        if previous_bw and current_bw < previous_bw * 0.95:
            raise RuntimeError(
                f"{collective} bus BW dip detected: {current_bw} at size {row['size']} after {previous_bw} at size {previous_size}"
            )
        previous_bw = current_bw
        previous_size = row["size"]


def _check_lat_dip(collective: str, rows: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> None:
    if not thresholds:
        return

    previous_time = 0.0
    previous_size = None
    valid_sizes = set(thresholds.keys())
    for row in _matching_rows(collective, rows):
        if str(row["size"]) not in valid_sizes:
            continue
        current_time = float(row["time"])
        if previous_time and current_time < previous_time * 0.95:
            raise RuntimeError(
                f"{collective} latency dip detected: {current_time} at size {row['size']} after {previous_time} at size {previous_size}"
            )
        previous_time = current_time
        previous_size = row["size"]


def parse_and_validate_results(
    config: RcclConfig,
    collective: str,
    raw_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validator = RcclTests if config.mode == "single_node" else RcclTestsMultinodeRaw
    try:
        validated = [validator.model_validate(result) for result in raw_results]
    except ValidationError as err:
        if _is_severe_wrong_corruption_error(err):
            raise RuntimeError(f"RCCL results for {collective} are corrupted (#wrong > 0)") from err
        raise RuntimeError(f"RCCL schema validation failed for {collective}: {err}") from err

    normalized_rows = [row.model_dump() for row in validated]
    thresholds = _normalize_threshold_map(config.thresholds.get(collective))
    validation_summary = {
        "schema": "passed",
        "validated_rows": len(normalized_rows),
        "thresholds_present": bool(thresholds),
        "bus_bw_check": "skipped",
        "bw_dip_check": "skipped",
        "lat_dip_check": "skipped",
    }

    if config.verify_bus_bw and thresholds:
        _check_bus_bw(collective, normalized_rows, thresholds)
        validation_summary["bus_bw_check"] = "passed"

    if config.verify_bw_dip and thresholds:
        _check_bw_dip(collective, normalized_rows, thresholds)
        validation_summary["bw_dip_check"] = "passed"

    if config.verify_lat_dip and thresholds:
        _check_lat_dip(collective, normalized_rows, thresholds)
        validation_summary["lat_dip_check"] = "passed"

    return normalized_rows, validation_summary


def _run_preflight(phdl: Any) -> dict[str, Any]:
    preflight = {
        "host_info_collected": True,
        "network_info_collected": True,
        "firewall_ok": True,
    }

    phdl.exec("cat /opt/rocm/.info/version 2>/dev/null || true")
    phdl.exec("hipconfig 2>/dev/null || true")
    phdl.exec("rocm_agent_enumerator 2>/dev/null || true")
    phdl.exec("rdma link 2>/dev/null || true")
    phdl.exec("ibv_devinfo 2>/dev/null || true")

    firewall_status = phdl.exec("sudo service ufw status 2>&1 || true")
    for node, output in firewall_status.items():
        if not re.search(r"inactive|dead|stopped|disabled|not loaded|could not be found", output, re.I):
            preflight["firewall_ok"] = False
            fail_test(f"Service ufw not disabled properly on node {node}")

    return preflight


def _load_remote_result(shdl: Any, head_node: str, remote_result_file: str) -> list[dict[str, Any]]:
    result_dict = shdl.exec(f"cat {shlex.quote(remote_result_file)}")
    return json.loads(result_dict[head_node].replace("\n", "").replace("\r", ""))


def _collective_result_path(collective: str) -> str:
    safe_name = collective.replace("/", "_")
    return f"/tmp/{safe_name}_{int(time.time() * 1000)}.json"


def run_rccl(cluster_dict: dict[str, Any], config: RcclConfig) -> dict[str, Any]:
    from cvs.lib.parallel_ssh_lib import Pssh

    selected_nodes, launch_hosts = _select_nodes(cluster_dict, config)
    head_node = selected_nodes[0]
    user = cluster_dict["username"]
    pkey = cluster_dict["priv_key_file"]

    phdl = Pssh(log, selected_nodes, user=user, pkey=pkey)
    shdl = Pssh(log, [head_node], user=user, pkey=pkey)

    preflight = _run_preflight(phdl)

    artifact: dict[str, Any] = {
        "metadata": {
            "suite": "rccl_cvs",
            "mode": config.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nodes": selected_nodes,
            "launch_hosts": launch_hosts,
            "num_ranks": config.num_ranks,
            "ranks_per_node": config.ranks_per_node,
            "datatype": config.datatype,
            "env_script": config.env_script,
            "output_json": config.output_json,
            "preflight": preflight,
        },
        "collectives": [],
    }

    for collective in config.collectives:
        remote_result_file = _collective_result_path(collective)
        command = build_collective_command(config, collective, remote_result_file, launch_hosts, shdl)

        phdl.exec(f'sudo echo "Starting RCCL {collective}" | sudo tee /dev/kmsg')
        start_time = phdl.exec('date +"%a %b %e %H:%M"')

        try:
            output_dict = shdl.exec(command, timeout=500)
            stdout = output_dict[head_node]
            _scan_rccl_stdout(stdout)
            raw_results = _load_remote_result(shdl, head_node, remote_result_file)
            validated_rows, validation_summary = parse_and_validate_results(config, collective, raw_results)
        finally:
            phdl.exec(f'sudo echo "Completed RCCL {collective}" | sudo tee /dev/kmsg')

        end_time = phdl.exec('date +"%a %b %e %H:%M"')
        verify_dmesg_for_errors(phdl, start_time, end_time, till_end_flag=True)

        artifact["collectives"].append(
            {
                "collective": collective,
                "command": command,
                "result_file": remote_result_file,
                "rows": validated_rows,
                "validation": validation_summary,
            }
        )

    _ensure_parent_dir(config.output_json)
    with open(config.output_json, "w") as handle:
        json.dump(artifact, handle, indent=2)

    return artifact
