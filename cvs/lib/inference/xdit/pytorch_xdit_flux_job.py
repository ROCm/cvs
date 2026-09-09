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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from cvs.lib import globals
from cvs.lib.parallel.phandle import ParallelHandle

log = globals.log

DEFAULT_BENCHMARK_TIMEOUT_S = 1800
FLUX2_DEFAULT_BENCHMARK_TIMEOUT_S = 3600
DEFAULT_MASTER_PORT = 29500
RUN_USP_PATH = "/app/Flux/run_usp.py"
FLUX2_EXAMPLE_PATH = "/app/external/xdit/examples/flux2_example.py"
FLUX2_EXAMPLE_MOUNT_PATH = "/benchmark/flux2_example.py"
FLUX2_EXAMPLE_HOST_SCRIPT = Path(__file__).resolve().parent / "scripts" / "flux2_example.py"
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
    """Return the object holding SSH credentials (handles MultiProcessParallelHandle wrapper)."""
    inner = getattr(s_phdl, "phandle", None)
    if inner is not None:
        return inner
    return s_phdl


def _phdl_connection_kwargs(s_phdl) -> Dict[str, Any]:
    """Best-effort SSH connection kwargs for a scoped one-node ParallelHandle."""
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


def _exec_on_nodes_concurrently(
    nodes: Sequence[str],
    runner: Callable[[str], Any],
) -> Dict[str, Any]:
    """Run ``runner(node)`` on each node; use a thread per node when len > 1."""
    node_list = list(nodes)
    if not node_list:
        return {}
    if len(node_list) == 1:
        return {node_list[0]: runner(node_list[0])}

    results: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(node_list)) as executor:
        future_to_node = {executor.submit(runner, node): node for node in node_list}
        for future in as_completed(future_to_node):
            node = future_to_node[future]
            results[node] = future.result()
    return results


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
        scoped = ParallelHandle(
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
        return (
            s_phdl.exec(
                cmd,
                timeout=timeout,
                print_console=print_console,
                detailed=detailed,
            )
            or {}
        )

    def _run(node: str) -> Any:
        return _exec_on_single_node(
            s_phdl,
            node,
            cmd,
            timeout=timeout,
            print_console=print_console,
            detailed=detailed,
        )

    return _exec_on_nodes_concurrently(node_list, _run)


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

    ``ParallelHandle.exec_cmd_list`` maps commands to ``s_phdl.host_list`` order. This helper
    avoids mis-launch when the participating node set is a subset or reordered.

    When ``detailed=True``, runs one ``exec(..., detailed=True)`` per node so callers
    receive structured ``{'output', 'exit_code'}`` values. Multi-node ``detailed``
    launches run concurrently so distributed torchrun rendezvous is not serialized.
    ``exec_cmd_list`` does not expose exit codes.
    """
    node_list = list(nodes)
    commands = list(cmd_list)
    if len(node_list) != len(commands):
        raise ValueError(f"node/cmd length mismatch: {len(node_list)} nodes vs {len(commands)} commands")

    phdl_hosts = list(getattr(s_phdl, "host_list", []) or [])
    if not detailed and phdl_hosts == node_list:
        return s_phdl.exec_cmd_list(commands, timeout=timeout, print_console=print_console) or {}

    cmd_by_node = dict(zip(node_list, commands))

    def _run(node: str) -> Any:
        return _exec_on_single_node(
            s_phdl,
            node,
            cmd_by_node[node],
            timeout=timeout,
            print_console=print_console,
            detailed=detailed,
        )

    return _exec_on_nodes_concurrently(node_list, _run)


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
    guidance_scale_flag = f"--guidance_scale {guidance_scale} " if guidance_scale is not None else ""

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


def default_flux2_example_host_path() -> str:
    """Host path of the CVS-shipped flux2_example.py (same layout as wan_i2v_example.py)."""
    return str(FLUX2_EXAMPLE_HOST_SCRIPT)


def resolve_flux2_example_host_mount(
    inference_dict: Mapping[str, Any],
) -> Optional[Tuple[str, str]]:
    """Return ``(host_path, container_path)`` if flux2_example.py is already in volume_dict."""
    volume_dict = dict(inference_dict.get("container_config", {}).get("volume_dict") or {})
    for host_path, mount_target in volume_dict.items():
        if mount_target in {FLUX2_EXAMPLE_MOUNT_PATH, FLUX2_EXAMPLE_PATH}:
            return str(host_path), str(mount_target)
    return None


def build_flux2_example_image_probe_cmd(container_image: str) -> str:
    """Print PRESENT when the image already contains flux2_example.py."""
    img = shlex.quote(container_image)
    path = shlex.quote(FLUX2_EXAMPLE_PATH)
    return (
        f"docker run --rm --network none --entrypoint test {img} -f {path} "
        f">/dev/null 2>&1 && echo FLUX2_EXAMPLE_PRESENT || echo FLUX2_EXAMPLE_MISSING"
    )


def build_flux2_example_host_check_cmd(host_path: str) -> str:
    quoted = shlex.quote(host_path)
    return f"test -e {quoted} && echo FLUX2_EXAMPLE_HOST_OK || echo FLUX2_EXAMPLE_HOST_MISSING"


def ensure_flux2_example_available(
    s_phdl,
    nodes: Sequence[str],
    inference_dict: Dict[str, Any],
    flux_params: Mapping[str, Any],
) -> List[str]:
    """
    If the image lacks ``FLUX2_EXAMPLE_PATH``, bind-mount CVS ``scripts/flux2_example.py``
    to ``FLUX2_EXAMPLE_MOUNT_PATH`` (``/benchmark/flux2_example.py``), same as WAN xFuser.

    An explicit ``volume_dict`` mapping wins over image probing. Skipped for FLUX.1.
    """
    model_type = resolve_flux_model_type_for_job(
        flux_params,
        model_repo=str(inference_dict.get("_resolved_model_path_container") or inference_dict.get("model_repo") or ""),
        model_repo_hints=[
            str(inference_dict.get("model_repo") or ""),
            str(inference_dict.get("_resolved_model_mount_host") or ""),
        ],
        resolved_model_type=inference_dict.get("_resolved_flux_model_type"),
    )
    if not is_flux2_model(model_type):
        return []

    node_list = list(nodes)
    if not node_list:
        return ["FLUX.2 example setup requires at least one execution node"]

    existing_mount = resolve_flux2_example_host_mount(inference_dict)
    if existing_mount:
        host_path, container_path = existing_mount
        log.info(
            "FLUX.2 example already bind-mounted from %s to %s",
            host_path,
            container_path,
        )
    else:
        probe_cmd = build_flux2_example_image_probe_cmd(str(inference_dict["container_image"]))
        try:
            probe = _exec_on_nodes(s_phdl, node_list, probe_cmd, print_console=False)
        except Exception as exc:
            return [f"Failed to probe container image for {FLUX2_EXAMPLE_PATH}: {exc}"]

        missing = [node for node in node_list if "FLUX2_EXAMPLE_PRESENT" not in (probe.get(node) or "")]
        if not missing:
            inference_dict["_flux2_example_container_path"] = FLUX2_EXAMPLE_PATH
            log.info("FLUX.2 example present in image at %s", FLUX2_EXAMPLE_PATH)
            return []

        host_path = default_flux2_example_host_path()
        container_path = FLUX2_EXAMPLE_MOUNT_PATH
        log.info(
            "FLUX.2 example missing in image on %d node(s); bind-mounting %s -> %s",
            len(missing),
            host_path,
            container_path,
        )

    check_cmd = build_flux2_example_host_check_cmd(host_path)
    try:
        checks = _exec_on_nodes(s_phdl, node_list, check_cmd, print_console=False)
    except Exception as extra:
        return [f"Failed to verify FLUX.2 example host path {host_path}: {extra}"]

    missing_host = [node for node in node_list if "FLUX2_EXAMPLE_HOST_OK" not in (checks.get(node) or "")]
    if missing_host:
        return [
            f"FLUX.2 example {FLUX2_EXAMPLE_PATH} is not in the container image and "
            f"host file {host_path} is missing on {len(missing_host)} node(s): "
            f"{', '.join(missing_host)}. Copy cvs/lib/inference/xdit/scripts/flux2_example.py "
            f"onto those nodes or set volume_dict like WAN, for example: "
            f'"/home/{{user-id}}/cvs/cvs/lib/inference/xdit/scripts/flux2_example.py": '
            f'"{FLUX2_EXAMPLE_MOUNT_PATH}"'
        ]

    if not existing_mount:
        container_config = inference_dict.setdefault("container_config", {})
        volume_dict = dict(container_config.get("volume_dict") or {})
        volume_dict[host_path] = container_path
        container_config["volume_dict"] = volume_dict
        log.info("Added FLUX.2 example bind-mount %s -> %s", host_path, container_path)

    inference_dict["_flux2_example_container_path"] = container_path
    return []


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
    example_path: str = FLUX2_EXAMPLE_PATH,
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
    run_once = f"{torchrun_prefix} {example_path} {flux2_args}"

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
    flux2_example_path: Optional[str] = None,
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
            example_path=flux2_example_path or FLUX2_EXAMPLE_PATH,
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


from cvs.lib.inference.xdit.pytorch_xdit_benchmark_job import (  # noqa: E402
    BenchmarkLaunchPlan,
    PytorchXditBenchmarkJob,
)

FluxLaunchPlan = BenchmarkLaunchPlan


class FluxBenchmarkJob(PytorchXditBenchmarkJob):
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
        self.flux_params = benchmark_params_dict["flux1_dev_t2i"]
        super().__init__(
            s_phdl,
            inference_dict,
            hf_token,
            distributed=distributed,
            cluster_dict=cluster_dict,
            nproc_per_node=int(self.flux_params["torchrun_nproc"]),
        )

    def _benchmark_name(self) -> str:
        return "FLUX"

    def _host_output_dir(self, output_base_dir: str, hostname: str) -> str:
        return f"{output_base_dir}/flux_{hostname}_outputs"

    def validate_parallelism(self) -> Optional[str]:
        nnodes = self.nnodes if self.distributed else 1
        world_size, product, err = validate_parallelism(nnodes, self.flux_params)
        if err:
            return err
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

    def build_launch_plan(self) -> BenchmarkLaunchPlan:
        self._flux2_example_setup_errors = ensure_flux2_example_available(
            self.s_phdl,
            self.server_nodes,
            self.inference_dict,
            self.flux_params,
        )
        return super().build_launch_plan()

    def _pre_launch_validation(self, plan: BenchmarkLaunchPlan) -> List[str]:
        return list(getattr(self, "_flux2_example_setup_errors", None) or [])

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

    def _build_torchrun_cmd(
        self,
        *,
        node_rank: int,
        host_output_dir: str,
        master_addr: str,
        master_port: int,
    ) -> str:
        return build_torchrun_cmd(
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
            flux2_example_path=self.inference_dict.get("_flux2_example_container_path"),
        )


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
