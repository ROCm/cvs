"""
PyTorch XDit FLUX benchmark launcher (FLUX.1-dev, FLUX.2-dev; single-node + unified distributed).

Single mode:
  - One independent torchrun job per node in ``s_phdl.host_list``.
  - Each node writes to ``flux_{hostname}_outputs``.

Distributed mode:
  - One coordinated torchrun job across ``nnodes`` with distinct ``--node_rank``.
  - All nodes share rank-0 output dir ``flux_{rank0_hostname}_outputs``.
  - Requires parallel-degree product == nnodes × torchrun_nproc.

FLUX.2-dev is launched via ``/app/external/xdit/examples/flux2_example.py`` (xFuserFlux2Pipeline).
FLUX.1-dev continues to use ``/app/Flux/run_usp.py``.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from cvs.lib import globals
from cvs.lib.parallel_ssh_lib import Pssh

log = globals.log

DEFAULT_BENCHMARK_TIMEOUT_S = 1800
FLUX2_DEFAULT_BENCHMARK_TIMEOUT_S = 3600
DEFAULT_MASTER_PORT = 29500
RUN_USP_PATH = "/app/Flux/run_usp.py"
FLUX2_EXAMPLE_PATH = "/app/external/xdit/examples/flux2_example.py"
CONTAINER_OUTPUT_MOUNT = "/outputs"
CONTAINER_MODEL_MOUNT = "/model"

FLUX2_DEFAULT_HF_REPO = "black-forest-labs/FLUX.2-dev"
FLUX2_KLEIN_DEFAULT_HF_REPO = "black-forest-labs/FLUX.2-klein-4B"

FLUX2_DEFAULT_GUIDANCE_SCALE = 4.0
FLUX_KONTEXT_DEFAULT_GUIDANCE_SCALE = 2.5


def as_node_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _redact_secrets(text: str) -> str:
    if not text:
        return text
    return re.sub(r"(HF_TOKEN=)[^\s]+", r"\1<redacted>", text)


def log_benchmark_failure_excerpt(
    node: str,
    output: str,
    *,
    max_lines: int = 120,
) -> None:
    """Log tail of captured remote benchmark output after a benchmark failure."""
    if not (output or "").strip():
        log.error("Benchmark failed on %s but captured output was empty", node)
        return

    lines = _redact_secrets(output).splitlines()
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    log.error(
        "=== Benchmark failure excerpt (%s, last %d line(s)) ===",
        node,
        len(tail),
    )
    for line in tail:
        log.error("%s", line)
    log.error("=== end benchmark failure excerpt (%s) ===", node)


def _secret_str(value: Any) -> str:
    return "" if value is None else str(value)


def _ssh_credential_source(s_phdl) -> Any:
    """Return the object holding SSH credentials (handles MultiProcessPssh wrapper)."""
    inner = getattr(s_phdl, "pssh", None)
    if inner is not None:
        return inner
    return s_phdl


def _phdl_connection_kwargs(s_phdl) -> Dict[str, Any]:
    """Best-effort SSH connection kwargs for a scoped one-node Pssh handle."""
    src = _ssh_credential_source(s_phdl)
    env_vars = getattr(s_phdl, "env_vars", None)
    if env_vars is None:
        env_vars = getattr(src, "env_vars", None)
    return {
        "user": getattr(src, "user", None),
        "password": getattr(src, "password", None),
        "pkey": getattr(src, "pkey", "id_rsa"),
        "env_vars": env_vars,
    }


def _exec_result_output(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("output") or "")
    return str(value or "")


def _exec_result_exit_code(value: Any) -> int:
    if isinstance(value, dict):
        return int(value.get("exit_code", -1))
    return 0


def _normalize_exec_results(raw_results: Mapping[str, Any], nodes: Sequence[str]) -> Dict[str, str]:
    return {node: _exec_result_output((raw_results or {}).get(node)) for node in nodes}


def _exec_on_single_node(
    s_phdl,
    node: str,
    cmd: str,
    *,
    timeout: Optional[int] = None,
    print_console: bool = False,
    detailed: bool = False,
) -> Any:
    """Run ``cmd`` on exactly one node, even when ``s_phdl`` covers more hosts."""
    phdl_hosts = list(getattr(s_phdl, "host_list", []) or [])
    if phdl_hosts == [node]:
        out = s_phdl.exec(
            cmd,
            timeout=timeout,
            print_console=print_console,
            detailed=detailed,
        )
    else:
        scoped = Pssh(
            getattr(s_phdl, "log", log),
            [node],
            **_phdl_connection_kwargs(s_phdl),
        )
        out = scoped.exec(
            cmd,
            timeout=timeout,
            print_console=print_console,
            detailed=detailed,
        )

    node_out = (out or {}).get(node)
    if detailed:
        if isinstance(node_out, dict):
            return node_out
        return {"output": str(node_out or ""), "exit_code": -1}
    return str(node_out or "")


def _exec_on_nodes(
    s_phdl,
    nodes: Sequence[str],
    cmd: str,
    *,
    timeout: Optional[int] = None,
    print_console: bool = False,
    detailed: bool = False,
) -> Dict[str, Any]:
    """Run the same command on an explicit node subset."""
    node_list = list(nodes)
    phdl_hosts = list(getattr(s_phdl, "host_list", []) or [])

    if phdl_hosts == node_list:
        return s_phdl.exec(
            cmd,
            timeout=timeout,
            print_console=print_console,
            detailed=detailed,
        ) or {}

    results: Dict[str, Any] = {}
    for node in node_list:
        results[node] = _exec_on_single_node(
            s_phdl,
            node,
            cmd,
            timeout=timeout,
            print_console=print_console,
            detailed=detailed,
        )
    return results


def _exec_cmd_list_on_nodes(
    s_phdl,
    nodes: Sequence[str],
    cmd_list: Sequence[str],
    *,
    timeout: Optional[int] = None,
    print_console: bool = False,
    detailed: bool = False,
) -> Dict[str, Any]:
    """
    Run per-node commands on an explicit node subset.

    ``Pssh.exec_cmd_list`` maps commands to ``s_phdl.host_list`` order. This helper
    avoids mis-launch when the participating node set is a subset or reordered.

    When ``detailed=True``, runs one ``exec(..., detailed=True)`` per node so callers
    receive structured ``{'output', 'exit_code'}`` values. ``exec_cmd_list`` does not
    expose exit codes.
    """
    node_list = list(nodes)
    commands = list(cmd_list)
    if len(node_list) != len(commands):
        raise ValueError(f"node/cmd length mismatch: {len(node_list)} nodes vs {len(commands)} commands")

    phdl_hosts = list(getattr(s_phdl, "host_list", []) or [])
    if not detailed and phdl_hosts == node_list:
        return s_phdl.exec_cmd_list(commands, timeout=timeout, print_console=print_console) or {}

    results: Dict[str, Any] = {}
    for node, cmd in zip(node_list, commands):
        results[node] = _exec_on_single_node(
            s_phdl,
            node,
            cmd,
            timeout=timeout,
            print_console=print_console,
            detailed=detailed,
        )
    return results


def resolve_server_nodes(cluster_dict: Mapping[str, Any], inference_dict: Mapping[str, Any]) -> List[str]:
    explicit = inference_dict.get("server_node_list")
    if explicit:
        return as_node_list(explicit)
    return list(cluster_dict["node_dict"].keys())


def resolve_nnodes(inference_dict: Mapping[str, Any], server_nodes: Sequence[str]) -> int:
    configured = inference_dict.get("nnodes")
    if configured is not None and str(configured).strip() != "":
        return int(configured)
    return len(server_nodes)


def resolve_master_addr(
    inference_dict: Mapping[str, Any],
    node_to_hostname: Mapping[str, str],
    rank0_node: str,
    *,
    s_phdl=None,
) -> str:
    """
    Resolve torchrun rendezvous address.

    Prefer explicit config. Otherwise use rank-0 IP when possible, then hostname.
    """
    addr = (inference_dict.get("master_addr") or "").strip()
    if addr:
        return addr

    if s_phdl is not None:
        ip_cmd = "hostname -I | awk '{print $1}'"
        ip_out = _exec_on_single_node(s_phdl, rank0_node, ip_cmd, print_console=False).strip()
        first_ip = (ip_out.split() or [""])[0].strip()
        if first_ip:
            log.info("Resolved master_addr from rank-0 node %s: %s", rank0_node, first_ip)
            return first_ip

    hostname = node_to_hostname.get(rank0_node, rank0_node)
    log.info("Using hostname for master_addr on rank-0 node %s: %s", rank0_node, hostname)
    return hostname


def parallel_product(flux_params: Mapping[str, Any]) -> int:
    return (
        int(flux_params["ulysses_degree"])
        * int(flux_params["ring_degree"])
        * int(flux_params.get("pipefusion_parallel_degree", 1))
        * int(flux_params.get("tensor_parallel_degree", 1))
        * int(flux_params.get("data_parallel_degree", 1))
    )


def compute_world_size(nnodes: int, nproc_per_node: int) -> int:
    return nnodes * nproc_per_node


def validate_parallelism(
    nnodes: int,
    flux_params: Mapping[str, Any],
) -> Tuple[int, int, Optional[str]]:
    nproc = int(flux_params["torchrun_nproc"])
    world_size = compute_world_size(nnodes, nproc)
    product = parallel_product(flux_params)
    if product != world_size:
        return (
            world_size,
            product,
            (
                f"Parallel degree product {product} != world_size {world_size} "
                f"(nnodes={nnodes} × nproc={nproc}). "
                f"Check ulysses/ring/pipefusion/tensor_parallel/data_parallel."
            ),
        )
    return world_size, product, None


def infer_flux_model_type(model_repo: str, explicit_model_type: Optional[str] = None) -> Optional[str]:
    """
    Resolve run_usp.py --model_type from a single repo/path string.

    Prefer :func:`resolve_flux_model_type` when multiple hints are available.
    """
    return resolve_flux_model_type(explicit_model_type, model_repo)


def detect_flux_model_type_from_model_index(model_index: Mapping[str, Any]) -> Optional[str]:
    """Detect FLUX model family from a diffusers model_index.json payload."""
    class_name = str(model_index.get("_class_name", ""))
    if "Klein" in class_name:
        return "flux2_klein"
    if "Flux2" in class_name:
        return "flux2"
    if "Kontext" in class_name:
        return "flux_kontext"
    return None


def is_flux2_model(model_type: Optional[str]) -> bool:
    return model_type in {"flux2", "flux2_klein"}


def resolve_flux_model_type_for_job(
    flux_params: Mapping[str, Any],
    *,
    model_repo: str,
    model_repo_hints: Optional[Sequence[str]] = None,
    resolved_model_type: Optional[str] = None,
) -> Optional[str]:
    hints = [model_repo, *(model_repo_hints or [])]
    return resolve_flux_model_type(
        flux_params.get("model_type") or resolved_model_type,
        *hints,
    )


def store_resolved_flux_model_type_from_index(
    inference_dict: Dict[str, Any],
    model_index: Mapping[str, Any],
) -> None:
    """Persist detected model type on inference_dict for runner script selection."""
    model_type = detect_flux_model_type_from_model_index(model_index)
    if model_type:
        inference_dict["_resolved_flux_model_type"] = model_type
        log.info("Detected FLUX model type from model_index.json: %s", model_type)
    name_or_path = model_index.get("_name_or_path")
    if name_or_path and not str(name_or_path).startswith("/"):
        inference_dict["_resolved_flux_hf_repo_id"] = str(name_or_path)


def resolve_flux2_hf_repo_id(
    model_type: Optional[str],
    model_repo: str,
    model_repo_hints: Optional[Sequence[str]] = None,
    resolved_hf_repo_id: Optional[str] = None,
) -> str:
    """Hugging Face repo id used to fetch FLUX.2 tokenizer assets (e.g. chat_template.jinja)."""
    if resolved_hf_repo_id and not str(resolved_hf_repo_id).startswith("/"):
        return str(resolved_hf_repo_id)
    for hint in [model_repo, *(model_repo_hints or [])]:
        if hint and not str(hint).startswith("/") and "/" in str(hint):
            return str(hint)
    if model_type == "flux2_klein":
        return FLUX2_KLEIN_DEFAULT_HF_REPO
    return FLUX2_DEFAULT_HF_REPO


def build_flux2_ensure_chat_template_cmd(
    *,
    hf_repo_id: str,
    model_mount: str = CONTAINER_MODEL_MOUNT,
) -> str:
    """
    Container-side guard: FLUX.2 encode_prompt requires tokenizer.chat_template.

    Locally staged model trees often omit ``tokenizer/chat_template.jinja``; fetch it
    from Hugging Face into the mounted model directory when missing.
    """
    py_script = (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"model = Path({json.dumps(model_mount)})\n"
        "tok = model / 'tokenizer'\n"
        "if (tok / 'chat_template.jinja').is_file() or (tok / 'chat_template.json').is_file():\n"
        "    sys.exit(0)\n"
        "cfg_path = tok / 'tokenizer_config.json'\n"
        "if cfg_path.is_file():\n"
        "    try:\n"
        "        if json.loads(cfg_path.read_text(encoding='utf-8')).get('chat_template'):\n"
        "            sys.exit(0)\n"
        "    except json.JSONDecodeError:\n"
        "        pass\n"
        "repo = os.environ.get('FLUX2_HF_REPO_ID') or "
        f"{json.dumps(hf_repo_id)}\n"
        "token = os.environ.get('HF_TOKEN') or None\n"
        "try:\n"
        "    from huggingface_hub import hf_hub_download\n"
        "except ImportError as exc:\n"
        "    print(f'huggingface_hub required to fetch FLUX.2 chat template: {exc}', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "tok.mkdir(parents=True, exist_ok=True)\n"
        "hf_hub_download(\n"
        "    repo_id=repo,\n"
        "    filename='tokenizer/chat_template.jinja',\n"
        "    local_dir=str(model),\n"
        "    token=token,\n"
        ")\n"
        "if not (tok / 'chat_template.jinja').is_file():\n"
        "    print('Failed to install tokenizer/chat_template.jinja for FLUX.2', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    return f"python3 -c {shlex.quote(py_script)}"


def build_flux2_chat_template_host_check_cmd(host_model_path: str) -> str:
    """Return a shell command that prints OK when FLUX.2 chat template files are present."""
    base = shlex.quote(host_model_path.rstrip("/"))
    return (
        f"test -f {base}/tokenizer/chat_template.jinja "
        f"-o -f {base}/tokenizer/chat_template.json "
        f"&& echo OK || echo MISSING"
    )


def build_flux2_chat_template_host_repair_from_cache_cmd(
    host_model_path: str,
    hf_home: str,
    hf_repo_id: str,
) -> str:
    """Copy chat_template.jinja from a local Hugging Face cache snapshot into a model dir."""
    model_q = shlex.quote(host_model_path.rstrip("/"))
    hf_q = shlex.quote(hf_home.rstrip("/"))
    repo_safe = hf_repo_id.replace("/", "--")
    return (
        f"MODEL={model_q}; HF={hf_q}; "
        f"if test -f \"$MODEL/tokenizer/chat_template.jinja\" "
        f"-o -f \"$MODEL/tokenizer/chat_template.json\"; then echo OK; exit 0; fi; "
        f"for SNAP in \"$HF/hub/models--{repo_safe}/snapshots\"/*; do "
        f"if test -f \"$SNAP/tokenizer/chat_template.jinja\"; then "
        f"mkdir -p \"$MODEL/tokenizer\" && "
        f"cp \"$SNAP/tokenizer/chat_template.jinja\" \"$MODEL/tokenizer/\" && "
        f"echo OK && exit 0; fi; done; echo MISSING"
    )


def ensure_flux2_chat_template_on_host(
    s_phdl,
    nodes: Sequence[str],
    host_model_path: str,
    hf_home: str,
    *,
    model_type: Optional[str],
    model_repo: str,
    resolved_hf_repo_id: Optional[str] = None,
) -> None:
    """
    Best-effort repair of missing FLUX.2 ``tokenizer/chat_template.jinja`` on host model dirs.

    Copies from the local Hugging Face cache when available. The container launch path fetches
    from the Hub when the file is still missing at runtime (requires ``HF_TOKEN`` for gated repos).
    """
    if not is_flux2_model(model_type):
        return

    hf_repo = resolve_flux2_hf_repo_id(
        model_type,
        model_repo,
        [host_model_path],
        resolved_hf_repo_id,
    )
    check_cmd = build_flux2_chat_template_host_check_cmd(host_model_path)
    check = _exec_on_nodes(s_phdl, list(nodes), check_cmd, print_console=False)
    missing = [node for node in nodes if "OK" not in (check.get(node) or "")]
    if not missing:
        log.info("FLUX.2 chat template present under %s on all node(s)", host_model_path)
        return

    log.info(
        "FLUX.2 chat template missing on %d node(s); attempting copy from HF cache (%s)",
        len(missing),
        hf_repo,
    )
    repair_cmd = build_flux2_chat_template_host_repair_from_cache_cmd(
        host_model_path,
        hf_home,
        hf_repo,
    )
    repair = _exec_on_nodes(s_phdl, missing, repair_cmd, print_console=False)
    still_missing = [node for node in missing if "OK" not in (repair.get(node) or "")]
    if still_missing:
        log.warning(
            "FLUX.2 chat template still missing on %d node(s) after HF cache copy; "
            "benchmark container will download tokenizer/chat_template.jinja from %s",
            len(still_missing),
            hf_repo,
        )
    else:
        log.info("FLUX.2 chat template repaired from HF cache on all previously missing node(s)")


def resolve_flux_model_type(
    explicit_model_type: Optional[str] = None,
    *repo_hints: Optional[str],
) -> Optional[str]:
    """
    Resolve FLUX model family from config and one or more repo/path hints.

    FLUX.2 selects flux2_example.py; FLUX.1 uses run_usp.py (default when unset).
    When the container model path is ``/model``, hints must include the original
    host ``model_repo`` or model_index metadata.
    """
    if explicit_model_type:
        return str(explicit_model_type).strip() or None

    for hint in repo_hints:
        if not hint:
            continue
        repo_lower = str(hint).lower()
        if "flux.2" in repo_lower or "flux2" in repo_lower:
            if "klein" in repo_lower:
                return "flux2_klein"
            return "flux2"
        if "kontext" in repo_lower:
            return "flux_kontext"
    return None


def resolve_flux_guidance_scale(
    model_type: Optional[str],
    explicit_guidance_scale: Any = None,
) -> Optional[float]:
    """Return guidance scale for run_usp.py, applying model-family defaults when omitted."""
    if explicit_guidance_scale is not None:
        return float(explicit_guidance_scale)
    if model_type == "flux2" or model_type == "flux2_klein":
        return FLUX2_DEFAULT_GUIDANCE_SCALE
    if model_type == "flux_kontext":
        return FLUX_KONTEXT_DEFAULT_GUIDANCE_SCALE
    return None


def build_nccl_env(inference_dict: Mapping[str, Any]) -> Dict[str, str]:
    env: Dict[str, str] = {
        "HSA_FORCE_FINE_GRAIN_PCIE": "1",
        "NCCL_PROTO": "Simple",
    }
    mapping = {
        "nccl_ib_hca": "NCCL_IB_HCA",
        "nccl_socket_ifname": "NCCL_SOCKET_IFNAME",
        "gloo_socket_ifname": "GLOO_SOCKET_IFNAME",
        "nccl_debug": "NCCL_DEBUG",
    }
    for src, dst in mapping.items():
        val = inference_dict.get(src)
        if val:
            env[dst] = str(val)
    gid = inference_dict.get("nccl_ib_gid_index")
    if gid is not None and str(gid).strip() != "":
        env["NCCL_IB_GID_INDEX"] = str(gid)
    return env


def build_run_usp_args(
    flux_params: Mapping[str, Any],
    *,
    model_repo: str,
    output_dir_container: str = CONTAINER_OUTPUT_MOUNT,
) -> str:
    flags: List[str] = []
    if flux_params.get("no_use_resolution_binning"):
        flags.append("--no_use_resolution_binning")
    if flux_params.get("use_torch_compile"):
        flags.append("--use-torch-compile")

    pf = int(flux_params.get("pipefusion_parallel_degree", 1))
    tp = int(flux_params.get("tensor_parallel_degree", 1))
    dp = int(flux_params.get("data_parallel_degree", 1))

    log.info("FLUX.1 run_usp: model=%s", model_repo)

    return (
        f"--model {shlex.quote(model_repo)} "
        f"--prompt {shlex.quote(str(flux_params['prompt']))} "
        f"--seed {int(flux_params['seed'])} "
        f"--num_inference_steps {int(flux_params['num_inference_steps'])} "
        f"--max_sequence_length {int(flux_params['max_sequence_length'])} "
        f"{' '.join(flags)} "
        f"--warmup_steps {int(flux_params['warmup_steps'])} "
        f"--warmup_calls {int(flux_params['warmup_calls'])} "
        f"--num_repetitions {int(flux_params['num_repetitions'])} "
        f"--height {int(flux_params['height'])} "
        f"--width {int(flux_params['width'])} "
        f"--ulysses_degree {int(flux_params['ulysses_degree'])} "
        f"--ring_degree {int(flux_params['ring_degree'])} "
        f"--pipefusion_parallel_degree {pf} "
        f"--tensor_parallel_degree {tp} "
        f"--data_parallel_degree {dp} "
        f"--benchmark_output_directory {shlex.quote(output_dir_container)}"
    )


def build_flux2_example_args(
    flux_params: Mapping[str, Any],
    *,
    model_repo: str,
    model_type: Optional[str],
) -> str:
    flags: List[str] = []
    if flux_params.get("no_use_resolution_binning"):
        flags.append("--no_use_resolution_binning")
    if flux_params.get("use_torch_compile"):
        flags.append("--use_torch_compile")

    pf = int(flux_params.get("pipefusion_parallel_degree", 1))
    tp = int(flux_params.get("tensor_parallel_degree", 1))
    dp = int(flux_params.get("data_parallel_degree", 1))

    guidance_scale = resolve_flux_guidance_scale(model_type, flux_params.get("guidance_scale"))
    guidance_scale_flag = (
        f"--guidance_scale {guidance_scale} " if guidance_scale is not None else ""
    )

    log.info(
        "FLUX.2 flux2_example: model=%s model_type=%s guidance_scale=%s",
        model_repo,
        model_type,
        guidance_scale if guidance_scale is not None else "n/a",
    )

    return (
        f"--model {shlex.quote(model_repo)} "
        f"--prompt {shlex.quote(str(flux_params['prompt']))} "
        f"--seed {int(flux_params['seed'])} "
        f"--num_inference_steps {int(flux_params['num_inference_steps'])} "
        f"--max_sequence_length {int(flux_params['max_sequence_length'])} "
        f"{guidance_scale_flag}"
        f"{' '.join(flags)} "
        f"--warmup_steps {int(flux_params['warmup_steps'])} "
        f"--height {int(flux_params['height'])} "
        f"--width {int(flux_params['width'])} "
        f"--ulysses_degree {int(flux_params['ulysses_degree'])} "
        f"--ring_degree {int(flux_params['ring_degree'])} "
        f"--pipefusion_parallel_degree {pf} "
        f"--tensor_parallel_degree {tp} "
        f"--data_parallel_degree {dp} "
        f"--output_type pil"
    )


def _build_torchrun_prefix(
    *,
    distributed: bool,
    nproc: int,
    node_rank: int,
    nnodes: int,
    master_addr: str,
    master_port: int,
) -> str:
    if distributed:
        return (
            f"torchrun "
            f"--nnodes={nnodes} "
            f"--node_rank={node_rank} "
            f"--nproc_per_node={nproc} "
            f"--master_addr={shlex.quote(master_addr)} "
            f"--master_port={master_port}"
        )
    return f"torchrun --nproc_per_node={nproc}"


def build_flux2_benchmark_cmd(
    flux_params: Mapping[str, Any],
    *,
    model_repo: str,
    model_type: Optional[str],
    distributed: bool,
    node_rank: int = 0,
    nnodes: int = 1,
    nproc_per_node: Optional[int] = None,
    master_addr: str = "127.0.0.1",
    master_port: int = DEFAULT_MASTER_PORT,
    output_dir_container: str = CONTAINER_OUTPUT_MOUNT,
    hf_repo_id: Optional[str] = None,
    model_repo_hints: Optional[Sequence[str]] = None,
    resolved_hf_repo_id: Optional[str] = None,
) -> str:
    """
    Run flux2_example.py once inside a single torchrun session and write results/timing.json.

    flux2_example.py loads the model once, runs ``warmup_steps`` internally, prints
    ``epoch time: X.XX sec`` for the timed pass, then exits. Re-invoking torchrun for
    each repetition would reload FLUX.2 on every GPU and take hours.
    """
    nproc = int(nproc_per_node or flux_params["torchrun_nproc"])
    flux2_args = build_flux2_example_args(flux_params, model_repo=model_repo, model_type=model_type)
    resolved_repo = resolve_flux2_hf_repo_id(
        model_type,
        model_repo,
        model_repo_hints=model_repo_hints,
        resolved_hf_repo_id=resolved_hf_repo_id or hf_repo_id,
    )
    ensure_chat_template = build_flux2_ensure_chat_template_cmd(hf_repo_id=resolved_repo)
    torchrun_prefix = _build_torchrun_prefix(
        distributed=distributed,
        nproc=nproc,
        node_rank=node_rank,
        nnodes=nnodes,
        master_addr=master_addr,
        master_port=master_port,
    )
    run_once = f"{torchrun_prefix} {FLUX2_EXAMPLE_PATH} {flux2_args}"

    timing_writer = (not distributed) or (node_rank == nnodes - 1)
    if distributed and not timing_writer:
        log.info(
            "FLUX.2 distributed node_rank=%d: torchrun only (timing.json on last node rank=%d)",
            node_rank,
            nnodes - 1,
        )
        inner = f"cd {shlex.quote(output_dir_container)} && {ensure_chat_template} && {run_once}"
        return f"bash -c {shlex.quote(inner)}"

    if distributed:
        log.info(
            "FLUX.2 distributed timing.json on node_rank=%d (flux2_example prints epoch time on world rank %d)",
            node_rank,
            nnodes * nproc - 1,
        )

    log.info(
        "FLUX.2 benchmark uses one torchrun session (warmup_steps=%s); "
        "num_repetitions/warmup_calls in config apply to FLUX.1 run_usp only",
        flux_params.get("warmup_steps", 0),
    )

    py_script = (
        "import json, re, subprocess, sys\n"
        f"run_cmd = {json.dumps(run_once)}\n"
        "proc = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)\n"
        "sys.stdout.write(proc.stdout)\n"
        "sys.stderr.write(proc.stderr)\n"
        "if proc.returncode != 0:\n"
        "    sys.exit(proc.returncode)\n"
        'match = re.search(r"epoch time:\\s*([\\d.]+)", proc.stdout + proc.stderr)\n'
        "if not match:\n"
        '    print("Could not parse epoch time from flux2_example output", file=sys.stderr)\n'
        "    sys.exit(1)\n"
        'times = [{"pipe_time": float(match.group(1))}]\n'
        'with open("results/timing.json", "w", encoding="utf-8") as handle:\n'
        "    json.dump(times, handle)\n"
    )

    inner = (
        f"cd {shlex.quote(output_dir_container)} && "
        f"mkdir -p results && "
        f"{ensure_chat_template} && "
        f"python3 -c {shlex.quote(py_script)}"
    )
    return f"bash -c {shlex.quote(inner)}"


def build_torchrun_cmd(
    flux_params: Mapping[str, Any],
    *,
    model_repo: str,
    distributed: bool,
    node_rank: int = 0,
    nnodes: int = 1,
    nproc_per_node: Optional[int] = None,
    master_addr: str = "127.0.0.1",
    master_port: int = DEFAULT_MASTER_PORT,
    model_repo_hints: Optional[Sequence[str]] = None,
    resolved_model_type: Optional[str] = None,
    resolved_hf_repo_id: Optional[str] = None,
) -> str:
    nproc = int(nproc_per_node or flux_params["torchrun_nproc"])
    model_type = resolve_flux_model_type_for_job(
        flux_params,
        model_repo=model_repo,
        model_repo_hints=model_repo_hints,
        resolved_model_type=resolved_model_type,
    )

    if is_flux2_model(model_type):
        return build_flux2_benchmark_cmd(
            flux_params,
            model_repo=model_repo,
            model_type=model_type,
            distributed=distributed,
            node_rank=node_rank,
            nnodes=nnodes,
            nproc_per_node=nproc,
            master_addr=master_addr,
            master_port=master_port,
            model_repo_hints=model_repo_hints,
            resolved_hf_repo_id=resolved_hf_repo_id,
        )

    run_usp_args = build_run_usp_args(flux_params, model_repo=model_repo)
    torchrun_prefix = _build_torchrun_prefix(
        distributed=distributed,
        nproc=nproc,
        node_rank=node_rank,
        nnodes=nnodes,
        master_addr=master_addr,
        master_port=master_port,
    )
    return f"{torchrun_prefix} {RUN_USP_PATH} {run_usp_args}"


def verify_distributed_logs(output: str, *, world_size: int) -> Tuple[bool, str]:
    if not output:
        return False, "Empty benchmark output"

    if re.search(rf"\bworld[_ ]?size[=:\s]+{world_size}\b", output, re.I):
        return True, f"Saw world_size={world_size} in logs"

    if re.search(r"Initialized process group|process group initialized|c10d", output, re.I):
        return True, "Saw distributed process-group initialization in logs"

    rank_refs = len(re.findall(r"\brank[=:\s]+\d+\b", output, re.I))
    log.info("Saw %d rank references in logs", rank_refs)
    if rank_refs >= 2:
        return True, f"Saw {rank_refs} rank references in logs"

    return False, (f"No distributed proof in logs for world_size={world_size}. ")


@dataclass
class FluxLaunchPlan:
    mkdir_cmds: List[str] = field(default_factory=list)
    docker_cmds: List[str] = field(default_factory=list)
    node_order: List[str] = field(default_factory=list)
    node_to_hostname: Dict[str, str] = field(default_factory=dict)
    output_dirs_by_node: Dict[str, str] = field(default_factory=dict)
    primary_output_dir: str = ""
    distributed: bool = False
    world_size: int = 0


class FluxBenchmarkJob:
    """Build and run FLUX docker+torchrun commands (FLUX.1 via run_usp, FLUX.2 via flux2_example)."""

    def __init__(
        self,
        s_phdl,
        inference_dict: Dict[str, Any],
        benchmark_params_dict: Mapping[str, Any],
        hf_token: Any = "",
        *,
        distributed: bool = False,
        cluster_dict: Optional[Mapping[str, Any]] = None,
    ):
        self.s_phdl = s_phdl
        self.inference_dict = inference_dict
        self.flux_params = benchmark_params_dict["flux1_dev_t2i"]
        self.hf_token = hf_token
        self.distributed = distributed
        self.cluster_dict = cluster_dict or {}

        self.nproc_per_node = int(self.flux_params["torchrun_nproc"])
        self.server_nodes = self._resolve_execution_nodes()
        self.nnodes = len(self.server_nodes) if self.distributed else 1

    def _resolve_execution_nodes(self) -> List[str]:
        if self.distributed:
            if not self.cluster_dict:
                raise ValueError("distributed=True requires cluster_dict")
            nodes = resolve_server_nodes(self.cluster_dict, self.inference_dict)
            nnodes = resolve_nnodes(self.inference_dict, nodes)
            if nnodes < 2:
                raise ValueError(f"Distributed mode requires nnodes >= 2, got {nnodes}")
            if len(nodes) < nnodes:
                raise ValueError(f"Cluster/server_node_list has {len(nodes)} node(s) but nnodes={nnodes}")
            return nodes[:nnodes]
        return list(self.s_phdl.host_list)

    def validate_parallelism(self) -> Optional[str]:
        if not self.distributed:
            _, _, err = validate_parallelism(1, self.flux_params)
        else:
            _, _, err = validate_parallelism(self.nnodes, self.flux_params)
        if err:
            return err

        world_size, product, _ = validate_parallelism(
            self.nnodes if self.distributed else 1,
            self.flux_params,
        )
        log.info(
            "Parallelism OK (%s): world_size=%s product=%s (ulysses=%s ring=%s pipefusion=%s tp=%s dp=%s)",
            "distributed" if self.distributed else "single-node",
            world_size,
            product,
            self.flux_params["ulysses_degree"],
            self.flux_params["ring_degree"],
            self.flux_params.get("pipefusion_parallel_degree", 1),
            self.flux_params.get("tensor_parallel_degree", 1),
            self.flux_params.get("data_parallel_degree", 1),
        )
        return None

    def check_kfd(self) -> List[str]:
        log.info("Checking /dev/kfd on %d node(s)", len(self.server_nodes))
        kfd_check = _exec_on_nodes(
            self.s_phdl,
            self.server_nodes,
            "test -e /dev/kfd && echo KFD_OK || echo KFD_MISSING",
            print_console=False,
        )
        missing = []
        for node in self.server_nodes:
            output = kfd_check.get(node, "")
            if "KFD_OK" not in (output or ""):
                missing.append(node)
                log.error("ROCm device node /dev/kfd not found on %s", node)
            else:
                log.info("/dev/kfd found on %s", node)
        return missing

    def _fetch_hostnames(self) -> Dict[str, str]:
        log.info("Getting hostnames from %d node(s)", len(self.server_nodes))
        hostname_result = _exec_on_nodes(self.s_phdl, self.server_nodes, "hostname")
        return {node: (hostname_result.get(node, "") or "").strip() or node for node in self.server_nodes}

    def _resolved_model_repo(self) -> str:
        return self.inference_dict.get("_resolved_model_path_container") or self.inference_dict["model_repo"]

    def _flux_model_type_hints(self) -> List[str]:
        hints: List[str] = []
        for key in ("model_repo", "_resolved_model_mount_host", "_resolved_model_path_container"):
            value = self.inference_dict.get(key)
            if value and str(value) not in hints:
                hints.append(str(value))
        return hints

    def _build_env_args(self) -> str:
        user_env = dict(self.inference_dict["container_config"].get("env_dict") or {})
        env_dict: Dict[str, str] = {}
        if self.distributed:
            env_dict.update(build_nccl_env(self.inference_dict))
        env_dict.update(user_env)
        env_dict["OMP_NUM_THREADS"] = "16"
        env_dict["HF_HOME"] = "/hf_home"
        env_dict["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.nproc_per_node))
        if self.hf_token:
            env_dict["HF_TOKEN"] = _secret_str(self.hf_token)
        model_type = resolve_flux_model_type_for_job(
            self.flux_params,
            model_repo=self._resolved_model_repo(),
            model_repo_hints=self._flux_model_type_hints(),
            resolved_model_type=self.inference_dict.get("_resolved_flux_model_type"),
        )
        if is_flux2_model(model_type):
            env_dict["FLUX2_HF_REPO_ID"] = resolve_flux2_hf_repo_id(
                model_type,
                self.inference_dict.get("model_repo", ""),
                self._flux_model_type_hints(),
                self.inference_dict.get("_resolved_flux_hf_repo_id"),
            )
        return " ".join(f"-e {key}={value}" for key, value in env_dict.items())

    def _build_volume_args(self, host_output_dir: str) -> str:
        volume_dict = dict(self.inference_dict["container_config"].get("volume_dict") or {})
        volume_dict[host_output_dir] = CONTAINER_OUTPUT_MOUNT
        volume_dict[self.inference_dict["hf_home"]] = "/hf_home"
        mount_host = self.inference_dict.get("_resolved_model_mount_host")
        if mount_host:
            volume_dict[mount_host] = "/model"
        return " ".join(f"--mount type=bind,source={src},target={dst}" for src, dst in volume_dict.items())

    def _build_docker_cmd(
        self,
        *,
        node_rank: int,
        host_output_dir: str,
        master_addr: str,
        master_port: int,
    ) -> str:
        device_list = self.inference_dict["container_config"]["device_list"]
        device_args = " ".join(f"--device={dev}" for dev in device_list)
        env_args = self._build_env_args()
        volume_args = self._build_volume_args(host_output_dir)

        torchrun_cmd = build_torchrun_cmd(
            self.flux_params,
            model_repo=self._resolved_model_repo(),
            distributed=self.distributed,
            node_rank=node_rank,
            nnodes=self.nnodes if self.distributed else 1,
            nproc_per_node=self.nproc_per_node,
            master_addr=master_addr,
            master_port=master_port,
            model_repo_hints=self._flux_model_type_hints(),
            resolved_model_type=self.inference_dict.get("_resolved_flux_model_type"),
            resolved_hf_repo_id=self.inference_dict.get("_resolved_flux_hf_repo_id"),
        )

        container_name = self.inference_dict["container_name"]
        if self.distributed:
            container_name = f"{container_name}-rank{node_rank}"

        return (
            f"docker run "
            f"--cap-add=SYS_PTRACE "
            f"--security-opt seccomp=unconfined "
            f"--user root "
            f"{device_args} "
            f"--ipc=host "
            f"--network host "
            f"--rm "
            f"--privileged "
            f"--name {container_name} "
            f"{volume_args} "
            f"{env_args} "
            f"{self.inference_dict['container_image']} "
            f"{torchrun_cmd}"
        )

    def build_launch_plan(self) -> FluxLaunchPlan:
        node_to_hostname = self._fetch_hostnames()
        output_base_dir = self.inference_dict["output_base_dir"]
        master_port = int(self.inference_dict.get("master_port") or DEFAULT_MASTER_PORT)

        plan = FluxLaunchPlan(
            distributed=self.distributed,
            node_order=list(self.server_nodes),
            node_to_hostname=dict(node_to_hostname),
        )

        if self.distributed:
            rank0_node = self.server_nodes[0]
            master_addr = resolve_master_addr(
                self.inference_dict,
                node_to_hostname,
                rank0_node,
                s_phdl=self.s_phdl,
            )
            primary_output_dir = f"{output_base_dir}/flux_{node_to_hostname[rank0_node]}_outputs"
            plan.primary_output_dir = primary_output_dir
            plan.world_size = compute_world_size(self.nnodes, self.nproc_per_node)

            for node_rank, node in enumerate(self.server_nodes):
                plan.mkdir_cmds.append(f"mkdir -p {shlex.quote(primary_output_dir)}")
                plan.output_dirs_by_node[node] = primary_output_dir
                plan.docker_cmds.append(
                    self._build_docker_cmd(
                        node_rank=node_rank,
                        host_output_dir=primary_output_dir,
                        master_addr=master_addr,
                        master_port=master_port,
                    )
                )
                log.info(
                    "Distributed node %s (%s) rank=%d master=%s:%d output=%s",
                    node,
                    node_to_hostname[node],
                    node_rank,
                    master_addr,
                    master_port,
                    primary_output_dir,
                )
            return plan

        for node in self.server_nodes:
            hostname = node_to_hostname[node]
            host_output_dir = f"{output_base_dir}/flux_{hostname}_outputs"
            plan.mkdir_cmds.append(f"mkdir -p {shlex.quote(host_output_dir)}")
            plan.output_dirs_by_node[node] = host_output_dir
            plan.docker_cmds.append(
                self._build_docker_cmd(
                    node_rank=0,
                    host_output_dir=host_output_dir,
                    master_addr="127.0.0.1",
                    master_port=master_port,
                )
            )
            log.info("Single-node job on %s (%s) output=%s", node, hostname, host_output_dir)

        if len(self.server_nodes) == 1:
            only_node = self.server_nodes[0]
            plan.primary_output_dir = plan.output_dirs_by_node[only_node]
        else:
            plan.primary_output_dir = ""

        plan.world_size = self.nproc_per_node
        return plan

    def run(
        self,
        *,
        timeout: int = DEFAULT_BENCHMARK_TIMEOUT_S,
    ) -> Tuple[Dict[str, str], FluxLaunchPlan, List[str]]:
        errors: List[str] = []

        par_err = self.validate_parallelism()
        if par_err:
            errors.append(par_err)
            return {}, FluxLaunchPlan(), errors

        missing_kfd = self.check_kfd()
        if missing_kfd:
            errors.append(
                f"ROCm device node /dev/kfd not found on {len(missing_kfd)} node(s): "
                f"{', '.join(missing_kfd)}. Run on GPU compute nodes."
            )
            return {}, FluxLaunchPlan(), errors

        plan = self.build_launch_plan()

        if not plan.docker_cmds:
            errors.append("No docker commands generated")
            return {}, plan, errors

        log.info(
            "Creating output directories on %d node(s)",
            len(plan.node_order),
        )
        try:
            _exec_cmd_list_on_nodes(
                self.s_phdl,
                plan.node_order,
                plan.mkdir_cmds,
            )
        except Exception as exc:
            errors.append(f"Failed to create output directories: {exc}")
            return {}, plan, errors

        mode_label = "distributed unified" if self.distributed else "single-node"
        log.info(
            "Running FLUX benchmark (%s) on %d node command(s)",
            mode_label,
            len(plan.docker_cmds),
        )
        log.debug("Docker command (sample): %s", _redact_secrets(plan.docker_cmds[0]))

        try:
            raw_results = _exec_cmd_list_on_nodes(
                self.s_phdl,
                plan.node_order,
                plan.docker_cmds,
                timeout=timeout,
                detailed=True,
            )
        except Exception as exc:
            errors.append(f"Benchmark execution failed with exception: {exc}")
            return {}, plan, errors

        results = _normalize_exec_results(raw_results, plan.node_order)
        combined_output = "\n".join(results.values())
        if self.distributed:
            ok, msg = verify_distributed_logs(combined_output, world_size=plan.world_size)
            log.info("Distributed log proof: %s", msg)
            if not ok:
                errors.append(msg)

        failed_nodes = []
        for node in plan.node_order:
            raw = (raw_results or {}).get(node)
            output = _exec_result_output(raw)
            exit_code = _exec_result_exit_code(raw)
            if exit_code != 0:
                log.error("Benchmark exited with code %s on %s", exit_code, node)
                log_benchmark_failure_excerpt(node, output)
                failed_nodes.append(node)
            else:
                log.info("Benchmark on %s completed successfully (exit 0)", node)

        if failed_nodes:
            errors.append(f"Benchmark failed on {len(failed_nodes)} node(s): {', '.join(failed_nodes)}")

        return results or {}, plan, errors

    def store_output_dir_hint(self, plan: FluxLaunchPlan) -> None:
        if plan.primary_output_dir:
            self.inference_dict["_test_output_dir"] = plan.primary_output_dir
            return

        if not self.distributed and len(plan.node_order) == 1:
            node = plan.node_order[0]
            self.inference_dict["_test_output_dir"] = plan.output_dirs_by_node[node]


def launch_flux_benchmark(
    s_phdl,
    inference_dict: Dict[str, Any],
    benchmark_params_dict: Mapping[str, Any],
    hf_token: Any = "",
    *,
    distributed: bool = False,
    cluster_dict: Optional[Mapping[str, Any]] = None,
    timeout: int = DEFAULT_BENCHMARK_TIMEOUT_S,
) -> List[str]:
    """
    Run the FLUX benchmark and store ``_test_output_dir`` on success.

    Returns a list of error messages (empty == success). Intended for tests to
    map into ``fail_test`` / ``update_test_result``.
    """
    job = FluxBenchmarkJob(
        s_phdl,
        inference_dict,
        benchmark_params_dict,
        hf_token,
        distributed=distributed,
        cluster_dict=cluster_dict,
    )
    if timeout == DEFAULT_BENCHMARK_TIMEOUT_S:
        model_type = resolve_flux_model_type_for_job(
            job.flux_params,
            model_repo=job._resolved_model_repo(),
            model_repo_hints=job._flux_model_type_hints(),
            resolved_model_type=inference_dict.get("_resolved_flux_model_type"),
        )
        if is_flux2_model(model_type):
            timeout = FLUX2_DEFAULT_BENCHMARK_TIMEOUT_S

    _, plan, errors = job.run(timeout=timeout)
    if not errors:
        job.store_output_dir_hint(plan)
    return errors


def validate_flux_parallelism_config(
    inference_dict: Mapping[str, Any],
    benchmark_params_dict: Mapping[str, Any],
    *,
    distributed: bool,
    cluster_dict: Optional[Mapping[str, Any]] = None,
    node_count: Optional[int] = None,
) -> Optional[str]:
    """Standalone parallelism validation for a dedicated pytest preflight."""
    flux_params = benchmark_params_dict["flux1_dev_t2i"]
    if distributed:
        if not cluster_dict:
            return "distributed parallelism validation requires cluster_dict"
        nodes = resolve_server_nodes(cluster_dict, inference_dict)
        nnodes = resolve_nnodes(inference_dict, nodes)
        _, _, err = validate_parallelism(nnodes, flux_params)
        return err
    if node_count is not None and node_count > 1:
        return None
    _, _, err = validate_parallelism(1, flux_params)
    return err


def build_output_cleanup_cmd(output_base_dir: str, *, use_sudo: bool = True) -> str:
    prefix = "sudo " if use_sudo else ""
    # Glob must expand in shell — do not quote the *
    return f"bash -c {shlex.quote(f'{prefix}rm -rf {output_base_dir}/flux_*_outputs')}"
