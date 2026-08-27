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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from cvs.lib import globals
from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux_job import (
    DEFAULT_BENCHMARK_TIMEOUT_S,
    DEFAULT_MASTER_PORT,
    build_nccl_env,
    compute_world_size,
    resolve_master_addr,
    resolve_nnodes,
    resolve_server_nodes,
    scan_fatal_output,
    verify_distributed_logs,
    log_benchmark_failure_excerpt,
    _exec_cmd_list_on_nodes,
    _exec_on_nodes,
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
    save_video_path = str(
        wan_params.get("wan_diffusers_save_video_path") or WAN_XFUSER_VIDEO_CONTAINER_PATH
    )
    timing_json_path = str(
        wan_params.get("wan_diffusers_timing_json_path") or "results/timing.json"
    )
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


def scan_wan_fatal_output(output: str) -> bool:
    if scan_fatal_output(output):
        return True
    return any(re.search(p, output or "", re.I) for p in WAN_FATAL_OUTPUT_PATTERNS_EXTRA)


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
            errors.append(
                f"Replace placeholder {label} mount path in volume_dict: {host_path}"
            )

    if errors:
        return errors

    check_cmds = [
        " && ".join(
            f"test -e {shlex.quote(host_path)}"
            for host_path, _ in mount_checks
        )
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
            f"xFuser mount preflight failed on {node}. Missing or unreadable host path(s): "
            + "; ".join(details)
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


@dataclass
class WanLaunchPlan:
    mkdir_cmds: List[str] = field(default_factory=list)
    docker_cmds: List[str] = field(default_factory=list)
    node_order: List[str] = field(default_factory=list)
    node_to_hostname: Dict[str, str] = field(default_factory=dict)
    output_dirs_by_node: Dict[str, str] = field(default_factory=dict)
    primary_output_dir: str = ""
    distributed: bool = False
    world_size: int = 0


class WanBenchmarkJob:
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
        self.s_phdl = s_phdl
        self.inference_dict = inference_dict
        self.wan_params = benchmark_params_dict["wan22_i2v_a14b"]
        self.hf_token = hf_token
        self.distributed = distributed
        self.cluster_dict = cluster_dict or {}

        self.nproc_per_node = int(self.wan_params["torchrun_nproc"])
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
            log.info(
                "Single-node WAN run: using torchrun_nproc=%s (ulysses/ring inferred when omitted)",
                self.nproc_per_node,
            )
            return None

        _, _, err = validate_parallelism(self.nnodes, self.wan_params)
        if err:
            return err

        world_size, product, _ = validate_parallelism(self.nnodes, self.wan_params)
        log.info(
            "Parallelism OK (%s): world_size=%s product=%s (ulysses=%s ring=%s)",
            "distributed" if self.distributed else "single-node",
            world_size,
            product,
            _ulysses_size(self.wan_params),
            _ring_size(self.wan_params),
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

    def build_launch_plan(self) -> WanLaunchPlan:
        node_to_hostname = self._fetch_hostnames()
        output_base_dir = self.inference_dict["output_base_dir"]
        master_port = int(self.inference_dict.get("master_port") or DEFAULT_MASTER_PORT)

        plan = WanLaunchPlan(
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
            primary_output_dir = f"{output_base_dir}/wan_22_{node_to_hostname[rank0_node]}_outputs"
            plan.primary_output_dir = primary_output_dir
            plan.world_size = compute_world_size(self.nnodes, self.nproc_per_node)

            for node_rank, node in enumerate(self.server_nodes):
                plan.mkdir_cmds.append(f"mkdir -p {shlex.quote(primary_output_dir + '/outputs')}")
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
            host_output_dir = f"{output_base_dir}/wan_22_{hostname}_outputs"
            plan.mkdir_cmds.append(f"mkdir -p {shlex.quote(host_output_dir + '/outputs')}")
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
    ) -> Tuple[Dict[str, str], WanLaunchPlan, List[str]]:
        errors: List[str] = []

        par_err = self.validate_parallelism()
        if par_err:
            errors.append(par_err)
            return {}, WanLaunchPlan(), errors

        missing_kfd = self.check_kfd()
        if missing_kfd:
            errors.append(
                f"ROCm device node /dev/kfd not found on {len(missing_kfd)} node(s): "
                f"{', '.join(missing_kfd)}. Run on GPU compute nodes."
            )
            return {}, WanLaunchPlan(), errors

        plan = self.build_launch_plan()
        if not plan.docker_cmds:
            errors.append("No docker commands generated")
            return {}, plan, errors

        mount_errors = validate_wan_xfuser_mounts(
            self.s_phdl,
            plan.node_order,
            self.inference_dict,
            self.wan_params,
        )
        if mount_errors:
            errors.extend(mount_errors)
            return {}, plan, errors

        log.info("Creating output directories on %d node(s)", len(plan.node_order))
        try:
            _exec_cmd_list_on_nodes(self.s_phdl, plan.node_order, plan.mkdir_cmds)
        except Exception as exc:
            errors.append(f"Failed to create output directories: {exc}")
            return {}, plan, errors

        mode_label = "distributed unified" if self.distributed else "single-node"
        log.info(
            "Running WAN 2.2 benchmark (%s) on %d node command(s)",
            mode_label,
            len(plan.docker_cmds),
        )

        try:
            results = _exec_cmd_list_on_nodes(
                self.s_phdl,
                plan.node_order,
                plan.docker_cmds,
                timeout=timeout,
            )
        except Exception as exc:
            errors.append(f"Benchmark execution failed with exception: {exc}")
            return {}, plan, errors

        combined_output = "\n".join((results or {}).values())
        if self.distributed:
            ok, msg = verify_distributed_logs(combined_output, world_size=plan.world_size)
            log.info("Distributed log proof: %s", msg)
            if not ok:
                errors.append(msg)

        failed_nodes = []
        diffusers_script_hint_added = False
        for node in plan.node_order:
            output = (results or {}).get(node, "")
            if scan_wan_fatal_output(output):
                log.error("Benchmark output indicates failure on %s", node)
                log_benchmark_failure_excerpt(node, output)
                failed_nodes.append(node)
                if (
                    not diffusers_script_hint_added
                    and is_wan_diffusers_model(
                        resolve_wan_model_format_for_job(
                            self.wan_params,
                            model_repo_hints=self._wan_model_repo_hints(),
                            resolved_model_format=self.inference_dict.get("_resolved_wan_model_format"),
                        )
                    )
                    and re.search(r"can't open file .*run\.py", output or "", re.I)
                ):
                    run_script = resolve_wan_diffusers_run_script(self.wan_params, self.inference_dict)
                    errors.append(
                        diffusers_run_script_missing_hint(
                            self.inference_dict["container_image"],
                            run_script,
                        )
                    )
                    diffusers_script_hint_added = True
                elif (
                    not diffusers_script_hint_added
                    and re.search(r"can't open file .*wan_i2v_example\.py", output or "", re.I)
                ):
                    run_script = resolve_wan_diffusers_run_script(self.wan_params, self.inference_dict)
                    script_host = resolve_host_path_for_container_mount(self.inference_dict, run_script)
                    errors.append(
                        f"xFuser launcher {run_script!r} was not found in the container on {node}. "
                        f"Bind-mount the host script into the container"
                        + (f" (expected host path: {script_host})" if script_host else "")
                        + f" and confirm wan_diffusers_launcher is {WAN_DIFFUSERS_LAUNCHER_XFUSER!r}."
                    )
                    diffusers_script_hint_added = True
                elif re.search(r"Error response from daemon|bind source path does not exist", output or "", re.I):
                    errors.append(
                        f"Docker mount failure on {node}. Fix volume_dict host paths on every "
                        f"execution node (no /path/to/ placeholders). "
                        f"Log tail: {summarize_wan_benchmark_log(output)}"
                    )
            else:
                log.info("Benchmark docker finished on %s (output verification pending)", node)

        if failed_nodes:
            errors.append(f"Benchmark failed on {len(failed_nodes)} node(s): {', '.join(failed_nodes)}")
            return results or {}, plan, errors

        model_format = resolve_wan_model_format_for_job(
            self.wan_params,
            model_repo_hints=self._wan_model_repo_hints(),
            resolved_model_format=self.inference_dict.get("_resolved_wan_model_format"),
        )
        launcher = ""
        if is_wan_diffusers_model(model_format):
            launcher = resolve_wan_diffusers_launcher(self.wan_params, self.inference_dict)

        verify_cmds = []
        for node in plan.node_order:
            host_output_dir = plan.output_dirs_by_node[node]
            if launcher == WAN_DIFFUSERS_LAUNCHER_XFUSER:
                verify_cmds.append(build_wan_xfuser_output_verify_cmd(host_output_dir))
            else:
                verify_cmds.append(build_wan_output_verify_cmd(host_output_dir))
        try:
            verify_results = _exec_cmd_list_on_nodes(
                self.s_phdl,
                plan.node_order,
                verify_cmds,
                print_console=False,
            )
        except Exception as exc:
            errors.append(f"Failed to verify WAN benchmark outputs: {exc}")
            return results or {}, plan, errors

        missing_output_nodes = []
        for node in plan.node_order:
            host_output_dir = plan.output_dirs_by_node[node]
            verify_output = (verify_results or {}).get(node, "")
            if "WAN_OUTPUT_OK" in (verify_output or ""):
                log.info("WAN benchmark outputs verified under %s on %s", host_output_dir, node)
                continue

            missing_output_nodes.append(node)
            node_log = (results or {}).get(node, "")
            log_tail = summarize_wan_benchmark_log(node_log)
            if launcher == WAN_DIFFUSERS_LAUNCHER_XFUSER:
                log.error(
                    "No results/timing.json found under %s on %s",
                    host_output_dir,
                    node,
                )
                if not scan_wan_xfuser_benchmark_output(node_log):
                    errors.append(
                        f"No results/timing.json under {host_output_dir} on {node} and no xFuser "
                        f"'epoch time:' lines in benchmark log. Likely causes: placeholder "
                        f"volume_dict paths, missing bind mount for "
                        f"{resolve_wan_diffusers_run_script(self.wan_params, self.inference_dict)}, "
                        f"or stale mounted wan_i2v_example.py. Log tail: {log_tail}"
                    )
                else:
                    errors.append(
                        f"No results/timing.json found under {host_output_dir} on {node}. "
                        f"Log tail: {log_tail}"
                    )
            else:
                log.error(
                    "No rank0_step*.json files found under %s on %s",
                    host_output_dir,
                    node,
                )
                errors.append(
                    f"No rank0_step*.json files found under {host_output_dir} on {node}. "
                    f"Log tail: {log_tail}"
                )

        if missing_output_nodes:
            for node in missing_output_nodes:
                log_benchmark_failure_excerpt(node, (results or {}).get(node, ""))

        return results or {}, plan, errors

    def store_output_dir_hint(self, plan: WanLaunchPlan) -> None:
        if plan.primary_output_dir:
            self.inference_dict["_test_output_dir"] = plan.primary_output_dir
            return

        if not self.distributed and len(plan.node_order) == 1:
            node = plan.node_order[0]
            self.inference_dict["_test_output_dir"] = plan.output_dirs_by_node[node]


def launch_wan_benchmark(
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
