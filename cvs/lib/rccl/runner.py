from __future__ import annotations

import base64
import logging
import json
import os
import shlex
import time
from posixpath import dirname as _posix_dirname
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from cvs.lib import globals
from cvs.lib.verify_lib import verify_dmesg_for_errors

from .artifacts import (
    _build_summary,
    _ensure_parent_dir,
    _format_run_id_utc,
    _write_host_outputs,
    _write_optional_raw,
    _write_text_artifact,
)
from .config import RcclConfig
from .launcher import _env_setup_lines, _scan_rccl_stdout, build_collective_command
from .matrix_expand import expand_rccl_matrix_cases, expansion_input_from_rccl_config
from .validator import parse_and_validate_results

log = globals.log

_UFW_INACTIVE_SUBSTRINGS = (
    "inactive",
    "dead",
    "stopped",
    "disabled",
    "not loaded",
    "could not be found",
)


@contextmanager
def _temporary_log_level(logger_names: list[str], level: int):
    previous_levels: list[tuple[logging.Logger, int]] = []
    for name in logger_names:
        logger = logging.getLogger(name)
        previous_levels.append((logger, logger.level))
        logger.setLevel(level)
    try:
        yield
    finally:
        for logger, previous_level in previous_levels:
            logger.setLevel(previous_level)


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


def _run_preflight(phdl: Any) -> dict[str, Any]:
    preflight = {
        "host_info_collected": True,
        "network_info_collected": True,
        "firewall_ok": True,
    }
    errors: list[str] = []
    per_node_logs: dict[str, list[str]] = {}

    def capture(label: str, output: dict[str, str]) -> None:
        for node, text in output.items():
            per_node_logs.setdefault(node, []).append(f"$ {label}\n{text.rstrip()}\n")

    command = "cat /opt/rocm/.info/version 2>/dev/null || true"
    capture(command, phdl.exec(command, print_console=False))
    command = "hipconfig 2>/dev/null || true"
    capture(command, phdl.exec(command, print_console=False))
    command = "rocm_agent_enumerator 2>/dev/null || true"
    capture(command, phdl.exec(command, print_console=False))
    command = "rdma link 2>/dev/null || true"
    capture(command, phdl.exec(command, print_console=False))
    command = "ibv_devinfo 2>/dev/null || true"
    capture(command, phdl.exec(command, print_console=False))

    firewall_cmd = "sudo service ufw status 2>&1 || true"
    firewall_status = phdl.exec(firewall_cmd, print_console=False)
    capture(firewall_cmd, firewall_status)
    for node, output in firewall_status.items():
        lower_output = output.lower()
        if not any(needle in lower_output for needle in _UFW_INACTIVE_SUBSTRINGS):
            preflight["firewall_ok"] = False
            errors.append(f"Service ufw not disabled properly on node {node}")

    return {
        "summary": preflight,
        "logs": {node: "\n".join(chunks).rstrip() + "\n" for node, chunks in per_node_logs.items()},
        "errors": errors,
    }


def _load_remote_result(shdl: Any, head_node: str, remote_result_file: str) -> list[dict[str, Any]]:
    result_dict = shdl.exec(f"cat {shlex.quote(remote_result_file)}", print_console=False)
    raw = result_dict.get(head_node)
    if raw is None:
        raise RuntimeError(f"no stdout from head node {head_node!r} when reading {remote_result_file}")
    text = raw.strip()
    if not text:
        raise RuntimeError(f"empty remote RCCL result file {remote_result_file}")
    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = text.find("[")
        if start < 0:
            raise RuntimeError(
                f"remote RCCL JSON in {remote_result_file} is not valid JSON and has no '[' to anchor an array"
            ) from None
        try:
            parsed, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"remote RCCL JSON in {remote_result_file} could not be decoded") from exc
    if not isinstance(parsed, list):
        raise RuntimeError("RCCL JSON output must be a JSON array")
    return parsed


def _remote_case_result_json(remote_work_dir: str, run_id: str, case_id: str) -> str:
    base = remote_work_dir.rstrip("/")
    safe_case = case_id.replace("/", "_")
    return f"{base}/{run_id}_{safe_case}.json"


def _persist_run_json(run_dir: str, artifact: dict[str, Any]) -> None:
    run_json_path = os.path.join(run_dir, "run.json")
    _ensure_parent_dir(run_json_path)
    with open(run_json_path, "w") as handle:
        json.dump(artifact, handle, indent=2)
    log.info("RCCL stage: wrote artifact %s", run_json_path)


def _build_run_artifact(
    run_id: str,
    config: RcclConfig,
    selected_nodes: list[str],
    launch_hosts: list[str],
    captured_vars: dict[str, str],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
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
        result = shdl.exec(cmd, timeout=120, print_console=False)
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


def _write_case_command_log(run_dir: str, case_id: str, command: str, output: str) -> None:
    path = os.path.join(run_dir, "logs", "cases", case_id, "command_output.txt")
    text = f"$ {command}\n{output}"
    _write_text_artifact(path, text if text.endswith("\n") else f"{text}\n")


def _log_run_context(
    config: RcclConfig,
    run_id: str,
    run_dir: str,
    selected_nodes: list[str],
    launch_hosts: list[str],
) -> None:
    log.info(
        "RCCL run: run_id=%s output_dir=%s env_script=%s topology=%s ranks (%s/node across %s node(s))",
        run_id,
        run_dir,
        config.env_script,
        config.num_ranks,
        config.ranks_per_node,
        config.required_nodes,
    )
    log.info(
        "RCCL run: selected_nodes=%s launch_hosts=%s collectives=%s",
        ",".join(selected_nodes),
        ",".join(launch_hosts),
        ",".join(config.collectives),
    )


def run_rccl(cluster_dict: dict[str, Any], config: RcclConfig) -> dict[str, Any]:
    from cvs.lib.parallel_ssh_lib import Pssh

    with _temporary_log_level(["pssh.host_logger", "pssh"], logging.WARNING):
        selected_nodes, launch_hosts = _select_nodes(cluster_dict, config)
        head_node = selected_nodes[0]
        user = cluster_dict["username"]
        pkey = cluster_dict["priv_key_file"]

        phdl = Pssh(log, selected_nodes, user=user, pkey=pkey)
        shdl = Pssh(log, [head_node], user=user, pkey=pkey)

        run_id = _format_run_id_utc()
        run_dir = os.path.join(config.artifacts_output_dir, run_id)
        Path(config.artifacts_output_dir).mkdir(parents=True, exist_ok=True)
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        _log_run_context(config, run_id, run_dir, selected_nodes, launch_hosts)
        log.info("RCCL stage: preflight")
        preflight = _run_preflight(phdl)
        _write_host_outputs(os.path.join(run_dir, "logs", "preflight"), preflight["logs"])
        preflight_errors = preflight.get("errors") or []
        if preflight_errors:
            err_text = "; ".join(preflight_errors)
            cases_pf: list[dict[str, Any]] = [
                {
                    "case_id": "preflight",
                    "name": "preflight",
                    "resolved": {},
                    "command": "",
                    "status": "failed",
                    "error": err_text,
                }
            ]
            artifact_pf = _build_run_artifact(
                run_id, config, selected_nodes, launch_hosts, {}, cases_pf
            )
            _persist_run_json(run_dir, artifact_pf)
            raise RuntimeError(err_text)

        log.info("RCCL stage: capture environment")
        captured_vars = _capture_filtered_environment(shdl, head_node, config)

        cases: list[dict[str, Any]] = []
        resolved_specs = expand_rccl_matrix_cases(expansion_input_from_rccl_config(config))
        total_cases = len(resolved_specs)

        for i, spec in enumerate(resolved_specs):
            resolved = spec.resolved
            collective = str(resolved["collective"])
            datatype = str(resolved.get("datatype", config.datatype))
            raw_env = resolved.get("env", {})
            env_overlay: dict[str, str]
            if isinstance(raw_env, Mapping):
                env_overlay = {str(k): str(v) for k, v in raw_env.items()}
            else:
                env_overlay = {}
            case_id = spec.case_id
            name = spec.name
            remote_result_file = _remote_case_result_json(config.artifacts_remote_work_dir, run_id, case_id)
            remote_parent = _posix_dirname(remote_result_file)
            shdl.exec(f"mkdir -p {shlex.quote(remote_parent)}", print_console=False)
            command = build_collective_command(
                config,
                collective,
                remote_result_file,
                launch_hosts,
                shdl,
                datatype=datatype,
                env_overlay=env_overlay,
            )

            phdl.exec(f'sudo echo "Starting RCCL {collective}" | sudo tee /dev/kmsg', print_console=False)
            start_time = phdl.exec('date +"%a %b %e %H:%M:%S"', print_console=False)
            log.info("RCCL stage: running %s (%d/%d)", name, i + 1, total_cases)
            log.info("RCCL command [%s]: %s", case_id, command)
            case_started_at = time.monotonic()

            failures: list[str] = []
            validated_rows: list[dict[str, Any]] = []
            validation_summary: dict[str, Any] = {}

            try:
                output_dict = shdl.exec(command, timeout=500, print_console=True, print_console_prefix=False)
                stdout = output_dict[head_node]
                _write_case_command_log(run_dir, case_id, command, stdout)
                _scan_rccl_stdout(stdout)
                raw_results = _load_remote_result(shdl, head_node, remote_result_file)
                _write_optional_raw(run_dir, case_id, raw_results, config.artifacts_export_raw)
                validated_rows, validation_summary = parse_and_validate_results(config, collective, raw_results)
            except Exception as exc:  # noqa: BLE001
                failures.append(str(exc))
            finally:
                phdl.exec(f'sudo echo "Completed RCCL {collective}" | sudo tee /dev/kmsg', print_console=False)

            end_time = phdl.exec('date +"%a %b %e %H:%M:%S"', print_console=False)
            try:
                verify_dmesg_for_errors(
                    phdl,
                    start_time,
                    end_time,
                    till_end_flag=False,
                    print_console=False,
                    raise_on_error=True,
                    output_consumer=lambda output_dict, case_id=case_id: _write_host_outputs(
                        os.path.join(run_dir, "logs", "cases", case_id, "dmesg"),
                        output_dict,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(str(exc))

            if failures:
                error_text = "; ".join(failures)
                cases.append(
                    {
                        "case_id": case_id,
                        "name": name,
                        "resolved": resolved,
                        "command": command,
                        "status": "failed",
                        "error": error_text,
                    }
                )
                artifact = _build_run_artifact(
                    run_id, config, selected_nodes, launch_hosts, captured_vars, cases
                )
                _persist_run_json(run_dir, artifact)
                raise RuntimeError(error_text)

            log.info(
                "RCCL stage: completed %s (%d/%d) elapsed=%.1fs",
                name,
                i + 1,
                total_cases,
                time.monotonic() - case_started_at,
            )

            cases.append(
                {
                    "case_id": case_id,
                    "name": name,
                    "resolved": resolved,
                    "command": command,
                    "status": "passed",
                    "validation": validation_summary,
                    "metrics": {"rows": validated_rows},
                }
            )

        artifact = _build_run_artifact(run_id, config, selected_nodes, launch_hosts, captured_vars, cases)
        _persist_run_json(run_dir, artifact)
        return artifact
