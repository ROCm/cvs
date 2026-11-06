import json
import logging
import os
import time
import torch

from xfuser import xFuserArgs, xFuserFlux2Pipeline, xFuserFlux2KleinPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
)


def detect_pipeline_class(model_path):
    """Auto-detect Flux2Pipeline vs Flux2KleinPipeline from model_index.json."""
    model_index_path = os.path.join(model_path, "model_index.json")
    with open(model_index_path, "r") as f:
        model_index = json.load(f)
    class_name = model_index.get("_class_name", "")
    if "Klein" in class_name:
        return xFuserFlux2KleinPipeline, "klein"
    else:
        return xFuserFlux2Pipeline, "dev"


def _fid_png_count(directory):
    if not os.path.isdir(directory):
        return 0
    return sum(1 for name in os.listdir(directory) if name.lower().endswith(".png"))


def _run_flux2_fid_sweep(args, pipe, input_config, elapsed_time, fid_started):
    raw_steps = str(getattr(args, "fid_steps", "") or "").strip()
    if not raw_steps:
        return
    from fid_sweep import run_fid_sweep

    steps = [int(part) for part in raw_steps.split(",") if part.strip()]
    samples = int(getattr(args, "fid_samples", 2))
    rank = get_world_group().rank
    output_root = os.path.abspath("results")
    pipe_s = float(elapsed_time) if elapsed_time and elapsed_time > 0 else 0.0

    def generate_one(nfe, sample_index):
        wall_start = time.time()
        generated = pipe(
            height=input_config.height,
            width=input_config.width,
            prompt=input_config.prompt,
            num_inference_steps=int(nfe),
            output_type=input_config.output_type,
            max_sequence_length=input_config.max_sequence_length,
            guidance_scale=input_config.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(int(input_config.seed) + sample_index + 1),
        )
        torch.cuda.synchronize()
        elapsed = time.time() - wall_start
        images = list(getattr(generated, "images", None) or [])
        if pipe.is_dp_last_group() and images:
            dest = os.path.join(output_root, "fid", f"nfe_{int(nfe)}")
            os.makedirs(dest, exist_ok=True)
            for index, image in enumerate(images):
                image.save(os.path.join(dest, f"seed_{sample_index}_{index}.png"))
            print(f"FID wrote {len(images)} png(s) to {dest}", flush=True)
        return elapsed

    budget_s = float(getattr(args, "fid_budget_s", 0) or 0)
    try:
        run_fid_sweep(
            rank,
            steps,
            samples,
            budget_s,
            fid_started,
            pipe_s if rank == 0 else 0.0,
            int(input_config.num_inference_steps),
            generate_one,
            os.path.join(output_root, "fid", "manifest.json"),
        )
        if rank == 0:
            for step in steps:
                dest = os.path.join(output_root, "fid", f"nfe_{int(step)}")
                print(f"FID samples at {dest}: {_fid_png_count(dest)} png(s)", flush=True)
    except Exception as exc:
        print(f"FID sweep failed after the timed benchmark: {exc}")


def main():
    fid_started = time.monotonic()
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument("--fid_steps", default="")
    parser.add_argument("--fid_samples", type=int, default=2)
    parser.add_argument("--fid_budget_s", type=float, default=0)
    parser.add_argument("--num_repetitions", type=int, default=1)
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank

    model_path = engine_config.model_config.model
    PipelineClass, variant = detect_pipeline_class(model_path)
    if get_world_group().rank == 0:
        print(f"Detected Flux2 variant: {variant}, using {PipelineClass.__name__}")

    pipe = PipelineClass.from_pretrained(
        pretrained_model_name_or_path=model_path,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    pipe.prepare_run(input_config)

    repetitions = max(int(args.num_repetitions), 1)
    pipe_times = []
    output = None
    peak_memory = 0
    for _step in range(repetitions):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            prompt=input_config.prompt,
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            max_sequence_length=input_config.max_sequence_length,
            guidance_scale=input_config.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        )
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        pipe_times.append(elapsed_time)
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
        if get_world_group().rank == get_world_group().world_size - 1:
            print(
                f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory / 1e9:.2f} GB, memory: {peak_memory / 1e9:.2f} GB"
            )
    elapsed_time = pipe_times[-1]

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = (
                    f"flux2_{variant}_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                )
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        os.makedirs("results", exist_ok=True)
        with open("results/timing.json", "w", encoding="utf-8") as handle:
            json.dump([{"pipe_time": float(value)} for value in pipe_times], handle)
    _run_flux2_fid_sweep(args, pipe, input_config, elapsed_time, fid_started)
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
