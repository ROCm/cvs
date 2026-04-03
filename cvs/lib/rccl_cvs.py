"""
RCCL CVS runner used by the rccl_cvs pytest suite.

nested rccl.run, rccl.validation, rccl.artifacts. Topology is inferred from num_ranks and ranks_per_node (no mode field).
The JSON file must contain only the top-level key ``rccl``

Artifacts: one run directory per invocation under artifacts.output_dir, canonical ``run.json``,
optional ``raw/<case_id>.json`` when export_raw is true.
"""

import base64
import hashlib
import json
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cvs.lib import globals
from cvs.lib.utils_lib import fail_test, resolve_test_config_placeholders
from cvs.lib.verify_lib import verify_dmesg_for_errors
from cvs.schema.rccl import RCCL_WRONG_NONZERO_ERROR_TYPE, RcclTests, RcclTestsMultinodeRaw
from cvs.schema.rccl_config import (
    RcclConfigFileRoot,
    format_rccl_config_validation_error,
    parse_rccl_thresholds_payload,
)

log = globals.log

RCCL_ERROR_PATTERNS = {
    "orte": r"ORTE does not know how to route|ORTE was unable to reliably start",
    "nccl": r"NCCL ERROR|Test failure",
    "fs": r"No such file or directory",
}


@dataclass(frozen=True)
class RcclConfig:
    """Resolved RCCL CVS config (single logical run, no matrix expansion)."""

    required_nodes: int
    collectives: list[str]
    datatype: str
    num_ranks: int
    ranks_per_node: int
    rccl_tests_dir: str
    rocm_path: str
    mpi_root: str | None
    mpirun_path: str | None
    env_script: str | None
    artifacts_output_dir: str
    artifacts_export_raw: bool
    start_size: str
    end_size: str
    step_factor: str
    warmups: str
    iterations: str
    cycles: str
    validation_profile: str
    thresholds: dict[str, Any]
    rccl_library_path: str | None
    # Echo of validated rccl.* input (run, validation, artifacts, matrix) for run.json config section.
    config_echo: dict[str, Any] = field(default_factory=dict)

    @property
    def is_single_node(self) -> bool:
        return self.required_nodes == 1


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


def _resolve_threshold_payload_from_file(
    path: str,
    collectives: list[str],
) -> dict[str, Any]:
    with open(path) as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("rccl.validation: thresholds_file must contain a JSON object")
    if len(loaded) == 0:
        raise ValueError(
            "rccl.validation: thresholds_file must contain a non-empty thresholds object "
            "with the same shape as rccl.validation.thresholds"
        )
    try:
        parsed = parse_rccl_thresholds_payload(loaded)
    except ValidationError as exc:
        raise ValueError(format_rccl_config_validation_error(exc)) from exc
    collective_set = set(collectives)
    for key in parsed:
        if not key or not str(key).strip():
            raise ValueError("rccl.validation: thresholds_file collective keys must be non-empty strings")
        if key not in collective_set:
            raise ValueError(f"rccl.validation: threshold key {key!r} is not listed in rccl.run.collectives")
    return {k: v.model_dump(mode="json") for k, v in parsed.items()}


def load_rccl_config(config_file: str, cluster_dict: dict[str, Any]) -> RcclConfig:
    with open(config_file) as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("RCCL config file must be a JSON object")
    extra_top = [k for k in raw if k != "rccl"]
    if extra_top:
        raise ValueError(
            f"RCCL config file must have exactly one top-level key 'rccl' (found extra: {sorted(extra_top)!r})"
        )
    if "rccl" not in raw or not isinstance(raw["rccl"], dict):
        raise ValueError("RCCL config file must contain a top-level 'rccl' object")

    rccl = resolve_test_config_placeholders(raw["rccl"], cluster_dict)
    rccl_for_schema = {k: v for k, v in rccl.items() if not str(k).startswith("_")}

    try:
        parsed = RcclConfigFileRoot.model_validate({"rccl": rccl_for_schema})
    except ValidationError as exc:
        raise ValueError(format_rccl_config_validation_error(exc)) from exc

    nested = parsed.rccl
    run = nested.run
    validation = nested.validation
    artifacts = nested.artifacts

    num_ranks = run.num_ranks
    ranks_per_node = run.ranks_per_node
    required_nodes = num_ranks // ranks_per_node
    collectives = list(run.collectives)

    if run.rocm_path:
        rocm_path = run.rocm_path
    else:
        rocm_path = "/opt/rocm"

    mpi_root = run.mpi_root
    mpirun_path = run.mpirun_path or (f"{mpi_root.rstrip('/')}/bin/mpirun" if mpi_root else None)

    if required_nodes > 1 and not mpirun_path:
        raise ValueError(
            "Multi-node RCCL runs (num_ranks / ranks_per_node > 1) require rccl.run.mpi_root or rccl.run.mpirun_path"
        )

    profile = validation.profile
    thresholds: dict[str, Any] = {}
    if profile in {"thresholds", "strict"}:
        # RcclValidationInput already enforces exactly one non-empty source for these profiles.
        if validation.thresholds:
            thresholds = {k: v.model_dump(mode="json") for k, v in validation.thresholds.items()}
        else:
            thresholds = _resolve_threshold_payload_from_file(validation.thresholds_file, collectives)

    config_echo = nested.model_dump(mode="json")

    return RcclConfig(
        required_nodes=required_nodes,
        collectives=collectives,
        datatype=run.datatype,
        num_ranks=num_ranks,
        ranks_per_node=ranks_per_node,
        rccl_tests_dir=run.rccl_tests_dir,
        rocm_path=rocm_path,
        mpi_root=mpi_root,
        mpirun_path=mpirun_path,
        env_script=run.env_script,
        artifacts_output_dir=artifacts.output_dir,
        artifacts_export_raw=artifacts.export_raw,
        config_echo=config_echo,
        start_size=run.start_size,
        end_size=run.end_size,
        step_factor=run.step_factor,
        warmups=run.warmups,
        iterations=run.iterations,
        cycles=run.cycles,
        validation_profile=profile,
        thresholds=thresholds,
        rccl_library_path=run.rccl_library_path,
    )


def _ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _format_run_id_utc(now: datetime | None = None) -> str:
    dt = now if now is not None else datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H-%M-%SZ")


def _no_matrix_case_id(collective_index: int, collective: str) -> str:
    return f"c{collective_index}_{_slug(collective)}"


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _ensure_unique_case_id(base: str, used: set[str], resolved: dict[str, Any]) -> str:
    if base not in used:
        used.add(base)
        return base
    digest = hashlib.sha256(_canonical_json_bytes(resolved)).hexdigest()[:8]
    n = 0
    while True:
        suffix = f"__dup{digest}" + (f"_{n}" if n else "")
        candidate = f"{base}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        n += 1


def _resolved_case_payload(config: RcclConfig, collective: str) -> dict[str, Any]:
    return {
        "collective": collective,
        "datatype": config.datatype,
        "start_size": config.start_size,
        "end_size": config.end_size,
        "step_factor": config.step_factor,
        "warmups": config.warmups,
        "iterations": config.iterations,
        "cycles": config.cycles,
        "env": {},
    }


def _build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    passed = sum(1 for c in cases if c.get("status") == "passed")
    failed = sum(1 for c in cases if c.get("status") == "failed")
    skipped = sum(1 for c in cases if c.get("status") == "skipped")
    if total > 0 and failed == 0:
        overall = "passed"
    else:
        overall = "failed"
    return {
        "cases_total": total,
        "cases_passed": passed,
        "cases_failed": failed,
        "cases_skipped": skipped,
        "overall_status": overall,
    }


def _env_setup_lines(config: RcclConfig) -> list[str]:
    lines: list[str] = []
    path_parts = []
    if config.mpi_root:
        path_parts.append(f"{config.mpi_root}/bin")
    if config.rocm_path:
        path_parts.append(f"{config.rocm_path}/bin")
    if path_parts:
        lines.append(f'export PATH="{":".join(path_parts)}:$PATH"')

    library_parts = []
    if config.rccl_library_path:
        library_parts.append(config.rccl_library_path.rstrip("/"))
    if config.mpi_root:
        library_parts.append(f"{config.mpi_root}/lib")
    if config.rocm_path:
        library_parts.append(f"{config.rocm_path.rstrip('/')}/lib")
    if library_parts:
        lines.append(f'export LD_LIBRARY_PATH="{":".join(library_parts)}:$LD_LIBRARY_PATH"')

    if config.env_script:
        lines.append(f"source {shlex.quote(config.env_script)}")
    return lines


def _select_nodes(cluster_dict: dict[str, Any], config: RcclConfig) -> tuple[list[str], list[str]]:
    cluster_nodes = list(cluster_dict["node_dict"].keys())
    if not cluster_nodes:
        raise ValueError("cluster_file does not contain any nodes")

    if config.is_single_node:
        selected_nodes = [cluster_nodes[0]]
    else:
        required_nodes = config.required_nodes
        if len(cluster_nodes) < required_nodes:
            raise ValueError(
                f"cluster_file provides {len(cluster_nodes)} node(s), but rccl_cvs requires {required_nodes}"
            )
        selected_nodes = cluster_nodes[:required_nodes]

    launch_hosts = [
        cluster_dict["node_dict"][node].get("vpc_ip") or cluster_dict["node_dict"][node].get("mgmt_ip") or node
        for node in selected_nodes
    ]
    return selected_nodes, launch_hosts


def _build_shell_payload(config: RcclConfig, collective: str, remote_result_file: str) -> str:
    binary = f"{config.rccl_tests_dir.rstrip('/')}/{collective}"
    gpus_per_rank = config.ranks_per_node if config.is_single_node else 1
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
    return "; ".join(_env_setup_lines(config) + [benchmark_cmd])


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

    if config.is_single_node:
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
        log.warning(
            "NCCL warnings observed in RCCL test output:\n%s",
            "\n".join(warnings),
        )

    if not re.search(r"#\sAvg bus bandwidth", output):
        raise RuntimeError("RCCL output did not contain '# Avg bus bandwidth'")


def _validation_has_wrong_nonzero(err: ValidationError) -> bool:
    """True if failure is due to rccl-tests wrong>0 (schema RcclTests.validate_wrong_is_zero)."""
    return any(item.get("type") == RCCL_WRONG_NONZERO_ERROR_TYPE for item in err.errors())


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


def _check_series_dip(
    collective: str,
    rows: list[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
    *,
    field: str,
    label: str,
) -> None:
    """Flag a >5% drop vs the previous in-threshold size (monotonic expectation for bus_bw or time)."""
    if not thresholds:
        return

    previous = 0.0
    previous_size = None
    valid_sizes = set(thresholds.keys())
    for row in _matching_rows(collective, rows):
        if str(row["size"]) not in valid_sizes:
            continue
        current = float(row[field])
        if previous and current < previous * 0.95:
            raise RuntimeError(
                f"{collective} {label} dip detected: {current} at size {row['size']} "
                f"after {previous} at size {previous_size}"
            )
        previous = current
        previous_size = row["size"]


def parse_and_validate_results(
    config: RcclConfig,
    collective: str,
    raw_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = config.validation_profile
    validation_summary: dict[str, Any] = {
        "parse": "passed",
        "schema": "skipped",
        "thresholds_bus_bw": "skipped",
        "bw_dip": "skipped",
        "lat_dip": "skipped",
    }

    if profile == "none":
        normalized_rows = list(raw_results)
        return normalized_rows, validation_summary

    validator = RcclTests if config.is_single_node else RcclTestsMultinodeRaw
    try:
        validated = [validator.model_validate(result) for result in raw_results]
    except ValidationError as err:
        if _validation_has_wrong_nonzero(err):
            raise RuntimeError(f"RCCL results for {collective} are corrupted (#wrong > 0)") from err
        raise RuntimeError(f"RCCL schema validation failed for {collective}: {err}") from err

    normalized_rows = [row.model_dump() for row in validated]
    validation_summary["schema"] = "passed"

    if profile == "smoke":
        return normalized_rows, validation_summary

    norm_thresholds = _normalize_threshold_map(config.thresholds.get(collective))
    if norm_thresholds:
        _check_bus_bw(collective, normalized_rows, norm_thresholds)
        validation_summary["thresholds_bus_bw"] = "passed"
        if profile == "strict":
            _check_series_dip(collective, normalized_rows, norm_thresholds, field="busBw", label="bus BW")
            validation_summary["bw_dip"] = "passed"
            _check_series_dip(collective, normalized_rows, norm_thresholds, field="time", label="latency")
            validation_summary["lat_dip"] = "passed"

    if profile in {"thresholds", "strict"}:
        return normalized_rows, validation_summary

    raise RuntimeError(f"Unexpected validation profile {profile!r}")


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
    parsed = json.loads(result_dict[head_node].replace("\n", "").replace("\r", ""))
    if not isinstance(parsed, list):
        raise RuntimeError("RCCL JSON output must be a JSON array")
    return parsed


def _collective_result_path(collective: str) -> str:
    safe_name = collective.replace("/", "_")
    return f"/tmp/{safe_name}_{int(time.time() * 1000)}.json"


def _capture_filtered_environment(shdl: Any, head_node: str, config: RcclConfig) -> dict[str, str]:
    """Variables after the same export/source sequence as benchmarks, filtered per spec."""
    dump_py = """import json, os
_PREFIXES = ("NCCL_", "RCCL_", "UCX_", "HIP_", "ROCR_", "HSA_")
_EXACT = frozenset({"LD_LIBRARY_PATH", "PATH"})
out = {k: v for k, v in os.environ.items() if k in _EXACT or k.startswith(_PREFIXES)}
print(json.dumps(out))
"""
    b64_payload = base64.b64encode(dump_py.encode()).decode("ascii")
    setup_shell = "; ".join(_env_setup_lines(config))
    py_bootstrap = f'import base64; exec(base64.b64decode("{b64_payload}").decode())'
    remote_bash = f"{setup_shell}; python3 -c {shlex.quote(py_bootstrap)}"
    cmd = f"bash -lc {shlex.quote(remote_bash)}"
    try:
        result = shdl.exec(cmd, timeout=120)
        text = (result.get(head_node) or "").strip()
        if not text:
            log.warning("RCCL environment capture returned empty stdout on %s", head_node)
            return {}
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return {}
        return {str(k): str(v) for k, v in parsed.items()}
    except Exception as exc:  # noqa: BLE001
        log.warning("RCCL environment capture failed: %s", exc)
        return {}


def _write_optional_raw(run_dir: str, case_id: str, raw_results: list[dict[str, Any]], export_raw: bool) -> None:
    if not export_raw:
        return
    raw_dir = os.path.join(run_dir, "raw")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(raw_dir, f"{case_id}.json")
    with open(path, "w") as handle:
        json.dump(raw_results, handle, indent=2)


def run_rccl(cluster_dict: dict[str, Any], config: RcclConfig) -> dict[str, Any]:
    from cvs.lib.parallel_ssh_lib import Pssh

    selected_nodes, launch_hosts = _select_nodes(cluster_dict, config)
    head_node = selected_nodes[0]
    user = cluster_dict["username"]
    pkey = cluster_dict["priv_key_file"]

    phdl = Pssh(log, selected_nodes, user=user, pkey=pkey)
    shdl = Pssh(log, [head_node], user=user, pkey=pkey)

    _run_preflight(phdl)

    run_id = _format_run_id_utc()
    run_dir = os.path.join(config.artifacts_output_dir, run_id)
    Path(config.artifacts_output_dir).mkdir(parents=True, exist_ok=True)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    captured_vars = _capture_filtered_environment(shdl, head_node, config)

    cases: list[dict[str, Any]] = []
    used_case_ids: set[str] = set()

    for i, collective in enumerate(config.collectives):
        resolved = _resolved_case_payload(config, collective)
        case_id = _ensure_unique_case_id(_no_matrix_case_id(i, collective), used_case_ids, resolved)
        remote_result_file = _collective_result_path(collective)
        command = build_collective_command(config, collective, remote_result_file, launch_hosts, shdl)

        phdl.exec(f'sudo echo "Starting RCCL {collective}" | sudo tee /dev/kmsg')
        start_time = phdl.exec('date +"%a %b %e %H:%M"')

        try:
            output_dict = shdl.exec(command, timeout=500)
            stdout = output_dict[head_node]
            _scan_rccl_stdout(stdout)
            raw_results = _load_remote_result(shdl, head_node, remote_result_file)
            _write_optional_raw(run_dir, case_id, raw_results, config.artifacts_export_raw)
            validated_rows, validation_summary = parse_and_validate_results(config, collective, raw_results)
        finally:
            phdl.exec(f'sudo echo "Completed RCCL {collective}" | sudo tee /dev/kmsg')

        end_time = phdl.exec('date +"%a %b %e %H:%M"')
        verify_dmesg_for_errors(phdl, start_time, end_time, till_end_flag=True)

        cases.append(
            {
                "case_id": case_id,
                "name": f"{collective} ({config.datatype})",
                "resolved": resolved,
                "command": command,
                "status": "passed",
                "validation": validation_summary,
                "metrics": {"rows": validated_rows},
            }
        )

    artifact: dict[str, Any] = {
        "schema_version": "rccl_cvs.run.v1",
        "suite": "rccl_cvs",
        "run_id": run_id,
        "topology": {
            "num_ranks": config.num_ranks,
            "ranks_per_node": config.ranks_per_node,
            "required_nodes": config.required_nodes,
        },
        "cluster": {
            "selected_nodes": selected_nodes,
            "launch_hosts": launch_hosts,
        },
        "config": dict(config.config_echo),
        "environment": {
            "env_script": config.env_script,
            "captured_vars": captured_vars,
        },
        "cases": cases,
        "summary": _build_summary(cases),
    }

    run_json_path = os.path.join(run_dir, "run.json")
    _ensure_parent_dir(run_json_path)
    with open(run_json_path, "w") as handle:
        json.dump(artifact, handle, indent=2)

    return artifact
