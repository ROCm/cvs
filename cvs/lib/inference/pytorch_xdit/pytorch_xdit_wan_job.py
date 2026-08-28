"""
PyTorch XDit WAN 2.2 benchmark launcher (single-node scale-out + unified distributed).

Single mode:
  - One independent torchrun job per node in ``s_phdl.host_list``.
  - Each node writes to ``wan_22_{hostname}_outputs``.

Distributed mode:
  - One coordinated torchrun job across ``nnodes`` with distinct ``--node_rank``.
  - All nodes share rank-0 output dir ``wan_22_{rank0_hostname}_outputs``.
  - Requires ulysses_size × ring_size == nnodes × torchrun_nproc.

WAN checkpoints use either the native Wan2.2 layout (``Wan-AI/Wan2.2-I2V-A14B`` via
``/app/Wan2.2/run.py``) or the Diffusers layout (``Wan-AI/Wan2.2-I2V-A14B-Diffusers`` via
``/app/Wan/run.py``). Model format is inferred from ``model_repo`` or ``model_index.json``.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from cvs.lib import globals
from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux_job import (
    DEFAULT_BENCHMARK_TIMEOUT_S,
    DEFAULT_MASTER_PORT,
    build_nccl_env,
    compute_world_size,
    resolve_nnodes,
    resolve_server_nodes,
    log_benchmark_failure_excerpt,
    _exec_cmd_list_on_nodes,
)

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_benchmark_job import (
    BenchmarkLaunchPlan,
    PytorchXditBenchmarkJob,
)

log = globals.log

WAN_MODEL_FORMAT_NATIVE = "native"
WAN_MODEL_FORMAT_DIFFUSERS = "diffusers"

WAN_DIFFUSERS_LAUNCHER_PACKAGED = "packaged"
WAN_DIFFUSERS_LAUNCHER_XFUSER = "xfuser_example"

RUN_WAN_NATIVE_PATH = "/app/Wan2.2/run.py"
RUN_WAN_DIFFUSERS_PATH = "/app/Wan/run.py"
WAN_XFUSER_EXAMPLE_CONTAINER_PATH = "/benchmark/wan_i2v_example.py"
CONTAINER_OUTPUT_MOUNT = "/outputs"
WAN_DIFFUSERS_BENCHMARK_OUTPUT_DIR = "results/outputs"
WAN_XFUSER_RESULTS_DIR = f"{CONTAINER_OUTPUT_MOUNT}/results"
WAN_XFUSER_BENCHMARK_OUTPUT_DIR = CONTAINER_OUTPUT_MOUNT
WAN_XFUSER_VIDEO_CONTAINER_PATH = f"{WAN_XFUSER_RESULTS_DIR}/video_i2v.mp4"
WAN_XFUSER_TIMING_JSON_CONTAINER_PATH = f"{WAN_XFUSER_RESULTS_DIR}/timing.json"
WAN_XFUSER_AUTO_INPUT_IMAGE = "/tmp/i2v_input.jpg"
I2V_INPUT_IMAGE_NATIVE = "/app/Wan2.2/examples/i2v_input.JPG"
I2V_INPUT_IMAGE_DIFFUSERS = "/app/Wan/i2v_input.JPG"

RUN_WAN_PATH = RUN_WAN_NATIVE_PATH
I2V_INPUT_IMAGE = I2V_INPUT_IMAGE_NATIVE

WAN_DIFFUSERS_DEFAULT_NUM_INFERENCE_STEPS = 40
WAN_DIFFUSERS_DEFAULT_SEED = 42
WAN_DEFAULT_RING_SIZE = 1
WAN_DISTRIBUTED_BENCHMARK_TIMEOUT_S = 7200


def resolve_wan_benchmark_timeout(
    *,
    distributed: bool,
    explicit_timeout: Optional[int] = None,
) -> int:
    if explicit_timeout is not None:
        return int(explicit_timeout)
    if distributed:
        return WAN_DISTRIBUTED_BENCHMARK_TIMEOUT_S
    return DEFAULT_BENCHMARK_TIMEOUT_S


def build_wan_distributed_container_cleanup_cmds(container_name: str, nnodes: int) -> List[str]:
    """Best-effort remove ranked WAN docker containers after a hung torchrun."""
    cleanup_cmds: List[str] = []
    for rank in range(nnodes):
        name = f"{container_name}-rank{rank}"
        inner = f"docker rm -f {shlex.quote(name)} 2>/dev/null || true"
        cleanup_cmds.append(f"bash -c {shlex.quote(inner)}")
    return cleanup_cmds


def _ulysses_size(wan_params: Mapping[str, Any]) -> int:
    if "ulysses_size" in wan_params:
        return int(wan_params["ulysses_size"])
    return int(wan_params["torchrun_nproc"])


def _ring_size(wan_params: Mapping[str, Any]) -> int:
    return int(wan_params.get("ring_size", WAN_DEFAULT_RING_SIZE))


WAN_FATAL_OUTPUT_PATTERNS_EXTRA = (
    r"No AMD GPU detected",
    r"0 active drivers \(\[\]\)\. There should only be one\.",
    r"can't open file",
    r"Error response from daemon",
    r"bind source path does not exist",
    r"invalid mount config",
)

WAN_BENIGN_OUTPUT_LINE_PATTERNS = (r"skipped \(ModuleNotFoundError\)",)

WAN_CORE_FATAL_OUTPUT_PATTERNS = (
    r"\bTraceback\b",
    r"\bChildFailedError\b",
    r"\bOSError:\b",
)


def _secret_str(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def parallel_product(wan_params: Mapping[str, Any]) -> int:
    return _ulysses_size(wan_params) * _ring_size(wan_params)


def validate_parallelism(
    nnodes: int,
    wan_params: Mapping[str, Any],
) -> Tuple[int, int, Optional[str]]:
    nproc = int(wan_params["torchrun_nproc"])
    world_size = compute_world_size(nnodes, nproc)
    if nnodes == 1:
        return world_size, world_size, None

    product = parallel_product(wan_params)
    if product != world_size:
        return (
            world_size,
            product,
            (
                f"Parallel degree product {product} != world_size {world_size} "
                f"(nnodes={nnodes} × nproc={nproc}). "
                f"Check ulysses_size and ring_size."
            ),
        )
    return world_size, product, None


def detect_wan_model_format_from_model_index(model_index: Mapping[str, Any]) -> Optional[str]:
    """Detect WAN model layout from a diffusers ``model_index.json`` payload."""
    class_name = str(model_index.get("_class_name", ""))
    if "WanImageToVideo" in class_name:
        return WAN_MODEL_FORMAT_DIFFUSERS
    return None


def store_resolved_wan_model_format_from_index(
    inference_dict: Dict[str, Any],
    model_index: Mapping[str, Any],
) -> None:
    """Persist detected WAN model format on ``inference_dict`` for launcher routing."""
    model_format = detect_wan_model_format_from_model_index(model_index)
    if model_format:
        inference_dict["_resolved_wan_model_format"] = model_format
        log.info("Detected WAN model format from model_index.json: %s", model_format)


def resolve_wan_model_format(
    explicit_model_format: Optional[str] = None,
    *repo_hints: Optional[str],
) -> str:
    """
    Resolve WAN checkpoint layout.

    ``native`` selects ``/app/Wan2.2/run.py`` (``--ckpt_dir``). ``diffusers`` selects
    ``/app/Wan/run.py`` (``--model``).
    """
    if explicit_model_format in {WAN_MODEL_FORMAT_NATIVE, WAN_MODEL_FORMAT_DIFFUSERS}:
        return str(explicit_model_format)

    for hint in repo_hints:
        if hint and "diffusers" in str(hint).lower():
            return WAN_MODEL_FORMAT_DIFFUSERS
    return WAN_MODEL_FORMAT_NATIVE


def resolve_wan_diffusers_launcher(
    wan_params: Mapping[str, Any],
    inference_dict: Optional[Mapping[str, Any]] = None,
    *,
    run_script: Optional[str] = None,
) -> str:
    for source in (wan_params, inference_dict or {}):
        explicit = source.get("wan_diffusers_launcher")
        if explicit in {WAN_DIFFUSERS_LAUNCHER_PACKAGED, WAN_DIFFUSERS_LAUNCHER_XFUSER}:
            return str(explicit)

    script = run_script
    if script is None:
        for source in (wan_params, inference_dict or {}):
            configured = source.get("wan_diffusers_run_script")
            if configured:
                script = str(configured)
                break

    if script and "wan_i2v_example" in script:
        return WAN_DIFFUSERS_LAUNCHER_XFUSER
    if script and script not in {RUN_WAN_DIFFUSERS_PATH, ""}:
        return WAN_DIFFUSERS_LAUNCHER_XFUSER
    return WAN_DIFFUSERS_LAUNCHER_PACKAGED


def resolve_wan_diffusers_run_script(
    wan_params: Mapping[str, Any],
    inference_dict: Optional[Mapping[str, Any]] = None,
) -> str:
    """Resolve the in-container Diffusers Wan launcher script."""
    for source in (wan_params, inference_dict or {}):
        explicit = source.get("wan_diffusers_run_script")
        if explicit:
            return str(explicit)
    resolved = (inference_dict or {}).get("_resolved_wan_diffusers_run_script")
    if resolved:
        return str(resolved)

    if resolve_wan_diffusers_launcher(wan_params, inference_dict) == WAN_DIFFUSERS_LAUNCHER_XFUSER:
        return WAN_XFUSER_EXAMPLE_CONTAINER_PATH
    return RUN_WAN_DIFFUSERS_PATH


def resolve_wan_diffusers_i2v_image(
    wan_params: Mapping[str, Any],
    inference_dict: Optional[Mapping[str, Any]] = None,
) -> str:
    if should_wan_xfuser_auto_generate_input_image(wan_params, inference_dict):
        return WAN_XFUSER_AUTO_INPUT_IMAGE
    for source in (wan_params, inference_dict or {}):
        explicit = source.get("wan_diffusers_i2v_image")
        if explicit:
            return str(explicit)
    return I2V_INPUT_IMAGE_DIFFUSERS


def should_wan_xfuser_auto_generate_input_image(
    wan_params: Mapping[str, Any],
    inference_dict: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when xFuser should synthesize a placeholder I2V input inside the container."""
    if resolve_wan_diffusers_launcher(wan_params, inference_dict) != WAN_DIFFUSERS_LAUNCHER_XFUSER:
        return False

    explicit_flag = wan_params.get("wan_xfuser_auto_input_image")
    if explicit_flag is not None:
        return bool(explicit_flag)

    configured_image = None
    for source in (wan_params, inference_dict or {}):
        value = source.get("wan_diffusers_i2v_image")
        if value:
            configured_image = str(value)
            break

    if configured_image and configured_image.lower() in {"auto", "generate"}:
        return True
    if configured_image:
        host_mount = resolve_host_path_for_container_mount(inference_dict or {}, configured_image)
        return host_mount is None
    return True


def build_wan_xfuser_auto_input_image_cmd(wan_params: Mapping[str, Any]) -> str:
    """Create a solid-color JPEG at ``WAN_XFUSER_AUTO_INPUT_IMAGE`` matching config resolution."""
    height, width = parse_wan_size(str(wan_params["size"]))
    return (
        "python3 -c "
        f"\"from PIL import Image; Image.new('RGB', ({width}, {height}), (120, 160, 200))"
        f".save('{WAN_XFUSER_AUTO_INPUT_IMAGE}')\""
    )


def build_wan_xfuser_video_deps_cmd() -> str:
    return "pip install -q imageio imageio-ffmpeg"


def diffusers_run_script_missing_hint(container_image: str, run_script: str) -> str:
    return (
        f"Diffusers Wan launcher {run_script!r} was not found in container {container_image!r}. "
        f"Use amdsiloai/pytorch-xdit for the packaged /app/Wan/run.py harness, or set "
        f"wan_diffusers_launcher to {WAN_DIFFUSERS_LAUNCHER_XFUSER!r}, mount "
        f"cvs/lib/inference/pytorch_xdit/scripts/wan_i2v_example.py into the container, and set "
        f"wan_diffusers_run_script plus wan_diffusers_i2v_image."
    )


def parse_wan_size(size: str) -> Tuple[int, int]:
    """Parse ``720*1280`` into ``(height, width)``."""
    parts = str(size).split("*")
    if len(parts) != 2:
        raise ValueError(f"Invalid WAN size {size!r}, expected height*width")
    return int(parts[0]), int(parts[1])


def resolve_wan_model_format_for_job(
    wan_params: Mapping[str, Any],
    *,
    model_repo_hints: Optional[Sequence[str]] = None,
    resolved_model_format: Optional[str] = None,
) -> str:
    hints = list(model_repo_hints or [])
    return resolve_wan_model_format(
        wan_params.get("model_format") or resolved_model_format,
        *hints,
    )


def build_run_wan_native_args(
    wan_params: Mapping[str, Any],
    *,
    ckpt_dir: str,
    output_dir_container: str = CONTAINER_OUTPUT_MOUNT,
) -> str:
    compile_flag = "--compile" if wan_params.get("compile") else ""
    return (
        f"--task i2v-A14B "
        f"--size {shlex.quote(str(wan_params['size']))} "
        f"--ckpt_dir {shlex.quote(ckpt_dir)} "
        f"--image {I2V_INPUT_IMAGE_NATIVE} "
        f"--save_file {CONTAINER_OUTPUT_MOUNT}/outputs/video.mp4 "
        f"--ulysses_size {_ulysses_size(wan_params)} "
        f"--ring_size {_ring_size(wan_params)} "
        f"--vae_dtype bfloat16 "
        f"--frame_num {int(wan_params['frame_num'])} "
        f"--prompt {shlex.quote(str(wan_params['prompt']))} "
        f"--benchmark_output_directory {shlex.quote(output_dir_container)} "
        f"--num_benchmark_steps {int(wan_params['num_benchmark_steps'])} "
        f"--offload_model 0 "
        f"--allow_tf32 "
        f"{compile_flag}"
    ).strip()


def is_wan_diffusers_model(model_format: str) -> bool:
    return model_format == WAN_MODEL_FORMAT_DIFFUSERS


def build_run_wan_xfuser_example_args(
    wan_params: Mapping[str, Any],
    *,
    model_path: str,
    i2v_image_path: str,
) -> str:
    height, width = parse_wan_size(str(wan_params["size"]))
    num_inference_steps = _optional_int(
        wan_params.get("num_inference_steps"),
        WAN_DIFFUSERS_DEFAULT_NUM_INFERENCE_STEPS,
    )
    num_repetitions = _optional_int(
        wan_params.get("num_repetitions"),
        int(wan_params["num_benchmark_steps"]),
    )
    warmup_steps = _optional_int(wan_params.get("warmup_steps"), 1)
    output_type = str(wan_params.get("wan_xfuser_output_type") or "pil")
    output_directory = WAN_XFUSER_BENCHMARK_OUTPUT_DIR
    save_video_path = str(wan_params.get("wan_diffusers_save_video_path") or WAN_XFUSER_VIDEO_CONTAINER_PATH)
    timing_json_path = str(wan_params.get("wan_diffusers_timing_json_path") or "results/timing.json")
    video_fps = _optional_int(wan_params.get("wan_diffusers_video_fps"), 16)

    log.info(
        "WAN xFuser wan_i2v_example: model=%s size=%dx%d repetitions=%s warmup=%s",
        model_path,
        height,
        width,
        num_repetitions,
        warmup_steps,
    )

    return (
        f"--model {shlex.quote(model_path)} "
        f"--input_image {shlex.quote(i2v_image_path)} "
        f"--height {height} "
        f"--width {width} "
        f"--num_frames {int(wan_params['frame_num'])} "
        f"--num_inference_steps {num_inference_steps} "
        f"--ulysses_degree {_ulysses_size(wan_params)} "
        f"--ring_degree {_ring_size(wan_params)} "
        f"--warmup_steps {warmup_steps} "
        f"--num_repetitions {num_repetitions} "
        f"--output_type {shlex.quote(output_type)} "
        f"--output_directory {shlex.quote(output_directory)} "
        f"--save_video_path {shlex.quote(save_video_path)} "
        f"--timing_json_path {shlex.quote(timing_json_path)} "
        f"--video_fps {video_fps} "
        f"--prompt {shlex.quote(str(wan_params['prompt']))}"
    ).strip()


def build_run_wan_diffusers_args(
    wan_params: Mapping[str, Any],
    *,
    model_path: str,
    i2v_image_path: str = I2V_INPUT_IMAGE_DIFFUSERS,
) -> str:
    height, width = parse_wan_size(str(wan_params["size"]))
    seed = _optional_int(wan_params.get("seed"), WAN_DIFFUSERS_DEFAULT_SEED)
    num_inference_steps = _optional_int(
        wan_params.get("num_inference_steps"),
        WAN_DIFFUSERS_DEFAULT_NUM_INFERENCE_STEPS,
    )
    num_repetitions = _optional_int(
        wan_params.get("num_repetitions"),
        int(wan_params["num_benchmark_steps"]),
    )
    compile_flag = "--use_torch_compile" if wan_params.get("compile") else ""
    ring = _ring_size(wan_params)
    ring_flag = f"--ring_degree {ring} " if ring > 1 else ""

    log.info(
        "WAN diffusers run.py: model=%s size=%dx%d num_repetitions=%s num_inference_steps=%s",
        model_path,
        height,
        width,
        num_repetitions,
        num_inference_steps,
    )

    return (
        f"--task i2v "
        f"--height {height} "
        f"--width {width} "
        f"--model {shlex.quote(model_path)} "
        f"--img_file_path {i2v_image_path} "
        f"--ulysses_degree {_ulysses_size(wan_params)} "
        f"{ring_flag}"
        f"--seed {seed} "
        f"--num_frames {int(wan_params['frame_num'])} "
        f"--prompt {shlex.quote(str(wan_params['prompt']))} "
        f"--num_repetitions {num_repetitions} "
        f"--num_inference_steps {num_inference_steps} "
        f"{compile_flag}"
    ).strip()


def build_run_wan_args(
    wan_params: Mapping[str, Any],
    *,
    ckpt_dir: str,
    output_dir_container: str = CONTAINER_OUTPUT_MOUNT,
) -> str:
    """Backward-compatible alias for :func:`build_run_wan_native_args`."""
    return build_run_wan_native_args(
        wan_params,
        ckpt_dir=ckpt_dir,
        output_dir_container=output_dir_container,
    )


def _build_wan_torchrun_body(
    wan_params: Mapping[str, Any],
    *,
    model_path: str,
    model_format: str,
    distributed: bool,
    node_rank: int,
    nnodes: int,
    nproc: int,
    master_addr: str,
    master_port: int,
    inference_dict: Optional[Mapping[str, Any]] = None,
) -> str:
    if is_wan_diffusers_model(model_format):
        run_script = resolve_wan_diffusers_run_script(wan_params, inference_dict)
        i2v_image = resolve_wan_diffusers_i2v_image(wan_params, inference_dict)
        launcher = resolve_wan_diffusers_launcher(
            wan_params,
            inference_dict,
            run_script=run_script,
        )
        if launcher == WAN_DIFFUSERS_LAUNCHER_XFUSER:
            run_args = build_run_wan_xfuser_example_args(
                wan_params,
                model_path=model_path,
                i2v_image_path=i2v_image,
            )
        else:
            run_args = build_run_wan_diffusers_args(
                wan_params,
                model_path=model_path,
                i2v_image_path=i2v_image,
            )
        log.info(
            "WAN diffusers launcher=%s script=%s i2v_image=%s",
            launcher,
            run_script,
            i2v_image,
        )
    else:
        run_script = RUN_WAN_NATIVE_PATH
        run_args = build_run_wan_native_args(wan_params, ckpt_dir=model_path)
        log.info("WAN native run.py: ckpt_dir=%s", model_path)

    if distributed:
        torchrun = (
            f"torchrun "
            f"--nnodes={nnodes} "
            f"--node_rank={node_rank} "
            f"--nproc_per_node={nproc} "
            f"--master_addr={shlex.quote(master_addr)} "
            f"--master_port={master_port} "
            f"{run_script} "
            f"{run_args}"
        )
    else:
        torchrun = f"torchrun --nproc_per_node={nproc} {run_script} {run_args}"

    if is_wan_diffusers_model(model_format):
        launcher = resolve_wan_diffusers_launcher(
            wan_params,
            inference_dict,
            run_script=run_script if is_wan_diffusers_model(model_format) else None,
        )
        if launcher == WAN_DIFFUSERS_LAUNCHER_XFUSER:
            output_subdir = WAN_XFUSER_RESULTS_DIR
        else:
            output_subdir = f"{CONTAINER_OUTPUT_MOUNT}/{WAN_DIFFUSERS_BENCHMARK_OUTPUT_DIR}"
        prep_cmds: List[str] = []
        if launcher == WAN_DIFFUSERS_LAUNCHER_XFUSER:
            if should_wan_xfuser_auto_generate_input_image(wan_params, inference_dict):
                prep_cmds.append(build_wan_xfuser_auto_input_image_cmd(wan_params))
            if wan_params.get("wan_xfuser_install_video_deps", True):
                prep_cmds.append(build_wan_xfuser_video_deps_cmd())
        prep_prefix = " && ".join(prep_cmds)
        if prep_prefix:
            prep_prefix = f"{prep_prefix} && "
        inner = (
            f"cd {shlex.quote(CONTAINER_OUTPUT_MOUNT)} && "
            f"mkdir -p {shlex.quote(output_subdir)} && "
            f"{prep_prefix}{torchrun}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return torchrun


def build_torchrun_cmd(
    wan_params: Mapping[str, Any],
    *,
    ckpt_dir: str,
    distributed: bool,
    node_rank: int = 0,
    nnodes: int = 1,
    nproc_per_node: Optional[int] = None,
    master_addr: str = "127.0.0.1",
    master_port: int = DEFAULT_MASTER_PORT,
    model_format: Optional[str] = None,
    model_repo_hints: Optional[Sequence[str]] = None,
    resolved_model_format: Optional[str] = None,
    inference_dict: Optional[Mapping[str, Any]] = None,
) -> str:
    nproc = int(nproc_per_node or wan_params["torchrun_nproc"])
    resolved_format = resolve_wan_model_format_for_job(
        wan_params,
        model_repo_hints=model_repo_hints,
        resolved_model_format=model_format or resolved_model_format,
    )
    return _build_wan_torchrun_body(
        wan_params,
        model_path=ckpt_dir,
        model_format=resolved_format,
        distributed=distributed,
        node_rank=node_rank,
        nnodes=nnodes,
        nproc=nproc,
        master_addr=master_addr,
        master_port=master_port,
        inference_dict=inference_dict,
    )


def _sanitize_wan_benchmark_output(output: str) -> str:
    """Drop known-benign xFuser/xDiT log lines before fatal-pattern scanning."""
    kept: List[str] = []
    for line in (output or "").splitlines():
        if any(re.search(p, line, re.I) for p in WAN_BENIGN_OUTPUT_LINE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def scan_wan_fatal_output(output: str) -> bool:
    text = _sanitize_wan_benchmark_output(output)
    if any(re.search(p, text, re.I) for p in WAN_CORE_FATAL_OUTPUT_PATTERNS):
        return True
    if any(re.search(p, text, re.I) for p in WAN_FATAL_OUTPUT_PATTERNS_EXTRA):
        return True
    if re.search(r"\bModuleNotFoundError\b", text, re.I):
        return bool(
            re.search(r"\bTraceback\b", text, re.I) or re.search(r"\[rank\d+\]:.*ModuleNotFoundError", text, re.I)
        )
    return False


def scan_wan_xfuser_benchmark_output(output: str) -> bool:
    """True when xFuser wan_i2v_example emitted at least one timed epoch line."""
    return bool(re.search(r"epoch time:\s*[\d.]+", output or "", re.I))


def build_wan_output_verify_cmd(host_output_dir: str) -> str:
    """Return a shell command that prints WAN_OUTPUT_OK when rank0 timing JSON exists."""
    quoted = shlex.quote(host_output_dir)
    inner = (
        f"find {quoted} -name 'rank0_step*.json' -print -quit 2>/dev/null | grep -q . "
        f"&& echo WAN_OUTPUT_OK || echo WAN_OUTPUT_MISSING"
    )
    return f"bash -c {shlex.quote(inner)}"


def build_wan_xfuser_output_verify_cmd(host_output_dir: str) -> str:
    """Return a shell command that prints WAN_OUTPUT_OK when Flux-style timing.json exists."""
    quoted = shlex.quote(host_output_dir)
    timing_json = f"{quoted}/results/timing.json"
    inner = f"test -f {timing_json} && echo WAN_OUTPUT_OK || echo WAN_OUTPUT_MISSING"
    return f"bash -c {shlex.quote(inner)}"


def resolve_host_path_for_container_mount(
    inference_dict: Mapping[str, Any],
    container_path: str,
) -> Optional[str]:
    """Find the host bind-mount source for a container path from ``volume_dict``."""
    volume_dict = dict(inference_dict.get("container_config", {}).get("volume_dict") or {})
    for host_path, mount_target in volume_dict.items():
        if mount_target == container_path:
            return str(host_path)
    return None


def _is_placeholder_mount_path(host_path: str) -> bool:
    normalized = str(host_path).strip().lower()
    return normalized.startswith("/path/to/") or "<changeme>" in normalized


def validate_wan_xfuser_mounts(
    s_phdl,
    nodes: Sequence[str],
    inference_dict: Mapping[str, Any],
    wan_params: Mapping[str, Any],
) -> List[str]:
    """
    Preflight xFuser bind mounts and input assets on each execution node.

    Catches placeholder paths and missing host files before ``docker run``.
    """
    if resolve_wan_diffusers_launcher(wan_params, inference_dict) != WAN_DIFFUSERS_LAUNCHER_XFUSER:
        return []

    errors: List[str] = []
    run_script_container = resolve_wan_diffusers_run_script(wan_params, inference_dict)
    i2v_image_container = resolve_wan_diffusers_i2v_image(wan_params, inference_dict)

    mount_checks: List[Tuple[str, str]] = []
    script_host = resolve_host_path_for_container_mount(inference_dict, run_script_container)
    if not script_host:
        errors.append(
            f"volume_dict must bind-mount wan_i2v_example.py to {run_script_container}. "
            f"Example: "
            f'"/home/{{user-id}}/cvs/cvs/lib/inference/pytorch_xdit/scripts/wan_i2v_example.py": '
            f'"{run_script_container}"'
        )
    else:
        mount_checks.append((script_host, f"xFuser launcher script ({run_script_container})"))

    i2v_host = resolve_host_path_for_container_mount(inference_dict, i2v_image_container)
    auto_input = should_wan_xfuser_auto_generate_input_image(wan_params, inference_dict)
    if auto_input:
        log.info(
            "xFuser I2V input will be generated in-container at %s (no host image mount required)",
            WAN_XFUSER_AUTO_INPUT_IMAGE,
        )
    elif not i2v_host:
        errors.append(
            f"volume_dict must bind-mount an I2V input image to {i2v_image_container}, "
            f"or set wan_xfuser_auto_input_image to true to generate a placeholder JPEG "
            f"in-container at {WAN_XFUSER_AUTO_INPUT_IMAGE}."
        )
    else:
        mount_checks.append((i2v_host, f"I2V input image ({i2v_image_container})"))

    for host_path, label in mount_checks:
        if _is_placeholder_mount_path(host_path):
            errors.append(f"Replace placeholder {label} mount path in volume_dict: {host_path}")

    if errors:
        return errors

    check_cmds = [
        " && ".join(f"test -e {shlex.quote(host_path)}" for host_path, _ in mount_checks)
        + " && echo WAN_MOUNT_OK || echo WAN_MOUNT_MISSING"
    ] * len(nodes)

    try:
        check_results = _exec_cmd_list_on_nodes(s_phdl, nodes, check_cmds, print_console=False)
    except Exception as exc:
        return [f"Failed to verify xFuser mount paths on cluster nodes: {exc}"]

    for node in nodes:
        if "WAN_MOUNT_OK" in ((check_results or {}).get(node, "")):
            continue
        details = []
        for host_path, label in mount_checks:
            details.append(f"{label}: {host_path}")
        errors.append(
            f"xFuser mount preflight failed on {node}. Missing or unreadable host path(s): " + "; ".join(details)
        )
    return errors


def summarize_wan_benchmark_log(output: str, *, max_lines: int = 8) -> str:
    """Return a short tail snippet from benchmark output for error messages."""
    lines = [line for line in (output or "").splitlines() if line.strip()]
    if not lines:
        return "docker benchmark log was empty"
    tail = lines[-max_lines:]
    return " | ".join(line.strip() for line in tail)


def build_wan_output_cleanup_cmd(output_base_dir: str, *, use_sudo: bool = True) -> str:
    prefix = "sudo " if use_sudo else ""
    return f"bash -c {shlex.quote(f'{prefix}rm -rf {output_base_dir}/wan_22_*_outputs')}"


WanLaunchPlan = BenchmarkLaunchPlan


class WanBenchmarkJob(PytorchXditBenchmarkJob):
    """Build and run WAN 2.2 docker+torchrun commands via a Pssh-like handle."""

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
        self.wan_params = benchmark_params_dict["wan22_i2v_a14b"]
        super().__init__(
            s_phdl,
            inference_dict,
            hf_token,
            distributed=distributed,
            cluster_dict=cluster_dict,
            nproc_per_node=int(self.wan_params["torchrun_nproc"]),
        )

    def _benchmark_name(self) -> str:
        return "WAN 2.2"

    def _host_output_dir(self, output_base_dir: str, hostname: str) -> str:
        return f"{output_base_dir}/wan_22_{hostname}_outputs"

    def _mkdir_cmd(self, host_output_dir: str) -> str:
        return f"mkdir -p {shlex.quote(host_output_dir + '/outputs')}"

    def validate_parallelism(self) -> Optional[str]:
        if not self.distributed:
            log.info(
                "Single-node WAN run: using torchrun_nproc=%s (ulysses/ring inferred when omitted)",
                self.nproc_per_node,
            )
            return None

        world_size, product, err = validate_parallelism(self.nnodes, self.wan_params)
        if err:
            return err
        log.info(
            "Parallelism OK (%s): world_size=%s product=%s (ulysses=%s ring=%s)",
            "distributed" if self.distributed else "single-node",
            world_size,
            product,
            _ulysses_size(self.wan_params),
            _ring_size(self.wan_params),
        )
        return None

    def _resolved_ckpt_dir(self) -> str:
        ckpt_dir = self.inference_dict.get("_resolved_ckpt_dir_container")
        if ckpt_dir:
            return ckpt_dir
        model_repo = self.inference_dict["model_repo"]
        model_rev = self.inference_dict.get("model_rev") or ""
        model_path_safe = model_repo.replace("/", "--")
        return f"/hf_home/hub/models--{model_path_safe}/snapshots/{model_rev}"

    def _wan_model_repo_hints(self) -> List[str]:
        hints: List[str] = []
        for key in ("model_repo", "_resolved_model_mount_host", "_resolved_ckpt_dir_container"):
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
        env_dict["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.nproc_per_node))
        env_dict["OMP_NUM_THREADS"] = "16"
        env_dict["HF_HOME"] = "/hf_home"
        if self.hf_token:
            env_dict["HF_TOKEN"] = _secret_str(self.hf_token)
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
            self.wan_params,
            ckpt_dir=self._resolved_ckpt_dir(),
            distributed=self.distributed,
            node_rank=node_rank,
            nnodes=self.nnodes if self.distributed else 1,
            nproc_per_node=self.nproc_per_node,
            master_addr=master_addr,
            master_port=master_port,
            model_repo_hints=self._wan_model_repo_hints(),
            resolved_model_format=self.inference_dict.get("_resolved_wan_model_format"),
            inference_dict=self.inference_dict,
        )

    def _pre_launch_validation(self, plan: BenchmarkLaunchPlan) -> List[str]:
        return validate_wan_xfuser_mounts(
            self.s_phdl,
            plan.node_order,
            self.inference_dict,
            self.wan_params,
        )

    def _resolve_run_timeout(self, timeout: Optional[int]) -> int:
        return resolve_wan_benchmark_timeout(
            distributed=self.distributed,
            explicit_timeout=timeout,
        )

    def _rank0_benchmark_log(self, plan: WanLaunchPlan, results: Mapping[str, str]) -> str:
        if not plan.node_order:
            return ""
        return (results or {}).get(plan.node_order[0], "") or ""

    def _benchmark_logs_indicate_success(self, plan: WanLaunchPlan, results: Mapping[str, str]) -> bool:
        rank0_log = self._rank0_benchmark_log(plan, results)
        if not rank0_log or scan_wan_fatal_output(rank0_log):
            return False
        model_format = resolve_wan_model_format_for_job(
            self.wan_params,
            model_repo_hints=self._wan_model_repo_hints(),
            resolved_model_format=self.inference_dict.get("_resolved_wan_model_format"),
        )
        if is_wan_diffusers_model(model_format):
            launcher = resolve_wan_diffusers_launcher(self.wan_params, self.inference_dict)
            if launcher == WAN_DIFFUSERS_LAUNCHER_XFUSER:
                return scan_wan_xfuser_benchmark_output(rank0_log)
        return True

    def _append_wan_benchmark_failure_hints(
        self,
        errors: List[str],
        *,
        node: str,
        output: str,
        diffusers_script_hint_added: bool,
    ) -> bool:
        if diffusers_script_hint_added:
            return True

        model_format = resolve_wan_model_format_for_job(
            self.wan_params,
            model_repo_hints=self._wan_model_repo_hints(),
            resolved_model_format=self.inference_dict.get("_resolved_wan_model_format"),
        )
        if is_wan_diffusers_model(model_format) and re.search(r"can't open file .*run\.py", output or "", re.I):
            run_script = resolve_wan_diffusers_run_script(self.wan_params, self.inference_dict)
            errors.append(
                diffusers_run_script_missing_hint(
                    self.inference_dict["container_image"],
                    run_script,
                )
            )
            return True

        if re.search(r"can't open file .*wan_i2v_example\.py", output or "", re.I):
            run_script = resolve_wan_diffusers_run_script(self.wan_params, self.inference_dict)
            script_host = resolve_host_path_for_container_mount(self.inference_dict, run_script)
            errors.append(
                f"xFuser launcher {run_script!r} was not found in the container on {node}. "
                f"Bind-mount the host script into the container"
                + (f" (expected host path: {script_host})" if script_host else "")
                + f" and confirm wan_diffusers_launcher is {WAN_DIFFUSERS_LAUNCHER_XFUSER!r}."
            )
            return True

        if re.search(r"Error response from daemon|bind source path does not exist", output or "", re.I):
            errors.append(
                f"Docker mount failure on {node}. Fix volume_dict host paths on every "
                f"execution node (no /path/to/ placeholders). "
                f"Log tail: {summarize_wan_benchmark_log(output)}"
            )
            return True

        return False

    def _cleanup_stuck_containers(self, plan: WanLaunchPlan) -> None:
        if not self.distributed:
            return
        container_name = self.inference_dict["container_name"]
        cleanup_cmds = build_wan_distributed_container_cleanup_cmds(container_name, self.nnodes)
        try:
            _exec_cmd_list_on_nodes(
                self.s_phdl,
                plan.node_order,
                cleanup_cmds,
                print_console=False,
            )
        except Exception as exc:
            log.warning("Failed to clean up ranked WAN docker containers: %s", exc)

    def _handle_benchmark_exec_exception(
        self,
        exc: Exception,
        plan: WanLaunchPlan,
        results: Mapping[str, str],
    ) -> Tuple[List[str], bool]:
        self._cleanup_stuck_containers(plan)
        if self._benchmark_logs_indicate_success(plan, results):
            log.warning(
                "Benchmark docker exec failed (%s) but rank-0 logs indicate success; "
                "treating run as success (likely container exit hang after inference). "
                "Validate artifacts from shared output path in parse step.",
                exc,
            )
            return [], True
        errors = [f"Benchmark execution failed with exception: {exc}"]
        rank0_log = self._rank0_benchmark_log(plan, results)
        if rank0_log:
            log_benchmark_failure_excerpt(plan.node_order[0], rank0_log)
        return errors, False

    def _collect_benchmark_failures(
        self,
        raw_results: Mapping[str, Any],
        plan: WanLaunchPlan,
    ) -> Tuple[List[str], List[str]]:
        from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux_job import (
            _exec_result_exit_code,
            _exec_result_output,
            log_benchmark_failure_excerpt,
        )

        failed_nodes: List[str] = []
        extra_errors: List[str] = []
        diffusers_script_hint_added = False
        for node in plan.node_order:
            raw = (raw_results or {}).get(node)
            output = _exec_result_output(raw)
            exit_code = _exec_result_exit_code(raw)
            if exit_code != 0:
                log.error("Benchmark exited with code %s on %s", exit_code, node)
                log_benchmark_failure_excerpt(node, output)
                failed_nodes.append(node)
                diffusers_script_hint_added = self._append_wan_benchmark_failure_hints(
                    extra_errors,
                    node=node,
                    output=output,
                    diffusers_script_hint_added=diffusers_script_hint_added,
                )
            else:
                log.info("Benchmark on %s completed successfully (exit 0)", node)
        return failed_nodes, extra_errors


def launch_wan_benchmark(
    s_phdl,
    inference_dict: Dict[str, Any],
    benchmark_params_dict: Mapping[str, Any],
    hf_token: Any = "",
    *,
    distributed: bool = False,
    cluster_dict: Optional[Mapping[str, Any]] = None,
    timeout: Optional[int] = None,
) -> List[str]:
    """
    Run the WAN benchmark and store ``_test_output_dir`` on success.

    Returns a list of error messages (empty == success).
    """
    job = WanBenchmarkJob(
        s_phdl,
        inference_dict,
        benchmark_params_dict,
        hf_token,
        distributed=distributed,
        cluster_dict=cluster_dict,
    )
    _, plan, errors = job.run(timeout=timeout)
    if not errors:
        job.store_output_dir_hint(plan)
    return errors


def validate_wan_parallelism_config(
    inference_dict: Mapping[str, Any],
    benchmark_params_dict: Mapping[str, Any],
    *,
    distributed: bool,
    cluster_dict: Optional[Mapping[str, Any]] = None,
    node_count: Optional[int] = None,
) -> Optional[str]:
    """Standalone parallelism validation for a dedicated pytest preflight."""
    wan_params = benchmark_params_dict["wan22_i2v_a14b"]
    if distributed:
        if not cluster_dict:
            return "distributed parallelism validation requires cluster_dict"
        nodes = resolve_server_nodes(cluster_dict, inference_dict)
        nnodes = resolve_nnodes(inference_dict, nodes)
        _, _, err = validate_parallelism(nnodes, wan_params)
        return err
    return None
