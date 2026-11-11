# Flux inference with USP
# from https://github.com/chengzeyi/ParaAttention/blob/main/examples/run_flux.py

import functools
import os
import sys
import json
import logging
import time
import torch
import warnings

if torch.version.hip:
    try:
        from opt_groupnorm import OPTGroupNorm

        torch.nn.GroupNorm = OPTGroupNorm
    except (ImportError, ModuleNotFoundError):
        warnings.warn("OPTGroupNorm not found, using default torch.nn.GroupNorm")
from accelerate.utils import set_seed
from diffusers import DiffusionPipeline, FluxPipeline

from xfuser import xFuserArgs
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_dp_last_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)

from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxAttnProcessor

from torch.profiler import profile, record_function, ProfilerActivity

# flux_utils.py is bind-mounted beside this file. /app/Flux is the image copy's location.
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _path in ("/app/Flux", _script_dir):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from flux_utils import time_pipe_func, get_pipe_kwargs, print_model_args, create_arg_parser  # noqa: E402


def parallelize_transformer(pipe: DiffusionPipeline):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        *args,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        **kwargs,
    ):
        assert hidden_states.shape[0] % get_classifier_free_guidance_world_size() == 0, (
            f"Cannot split dim 0 of hidden_states ({hidden_states.shape[0]}) into {get_classifier_free_guidance_world_size()} parts."
        )
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size() != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(), dim=0)[
                get_classifier_free_guidance_rank()
            ]
        hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(), dim=0)[
            get_classifier_free_guidance_rank()
        ]
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[
            get_sequence_parallel_rank()
        ]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(), dim=0)[
            get_classifier_free_guidance_rank()
        ]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2)[
                get_sequence_parallel_rank()
            ]
        img_ids = torch.chunk(img_ids, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            txt_ids = torch.chunk(txt_ids, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]

        output = original_forward(
            hidden_states,
            encoder_hidden_states,
            *args,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = get_sp_group().all_gather(sample, dim=-2)
        sample = get_cfg_group().all_gather(sample, dim=0)
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

    for block in transformer.transformer_blocks + transformer.single_transformer_blocks:
        block.attn.processor = xFuserFluxAttnProcessor()


def _run_flux_fid_sweep(args, pipe, input_config, is_last_process, last_wall_s):
    raw_steps = str(getattr(args, "fid_steps", "") or "").strip()
    if not raw_steps or args.profile_output:
        return
    from fid_sweep import run_fid_sweep, seconds_until

    steps = [int(part) for part in raw_steps.split(",") if part.strip()]
    rank = get_world_group().rank

    def generate_one(nfe, sample_index):
        kwargs = get_pipe_kwargs(input_config)
        kwargs["num_inference_steps"] = int(nfe)
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(int(input_config.seed) + sample_index + 1)
        wall_start = time.time()
        generated = pipe(**kwargs)
        torch.cuda.synchronize()
        elapsed = time.time() - wall_start
        if is_last_process and getattr(generated, "images", None):
            dest = os.path.join(args.benchmark_output_directory, "results", "fid", f"nfe_{int(nfe)}")
            os.makedirs(dest, exist_ok=True)
            for index, image in enumerate(generated.images):
                image.save(os.path.join(dest, f"seed_{sample_index}_{index}.png"))
        return elapsed

    try:
        run_fid_sweep(
            rank,
            steps,
            int(args.fid_samples),
            seconds_until(getattr(args, "fid_deadline", 0)),
            time.monotonic(),
            last_wall_s if rank == 0 else 0.0,
            int(input_config.num_inference_steps),
            generate_one,
            os.path.join(args.benchmark_output_directory, "results", "fid", "manifest.json"),
        )
    except Exception as exc:
        print(f"FID sweep failed after the timed benchmark: {exc}")


def main():
    args = create_arg_parser()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    set_seed(input_config.seed)
    if get_world_group().rank == get_world_group().world_size - 1:
        print_model_args(engine_config, input_config, filename="flux_usp_example.py")

    if args.rep_sleep_duration:
        warnings.warn("--rep_sleep_duration is deprecated and will be removed in future versions.")

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."

    pipe = FluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        torch_dtype=torch.bfloat16,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    if get_world_group().world_size == 1:
        if torch.version.hip:
            pipe.transformer.set_attention_backend("aiter")

    initialize_runtime_state(pipe, engine_config)
    get_runtime_state().set_input_parameters(
        height=input_config.height,
        width=input_config.width,
        batch_size=1,
        num_inference_steps=input_config.num_inference_steps,
        max_condition_sequence_length=input_config.max_sequence_length,
        split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
    )

    parallelize_transformer(pipe)

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="default")

        # one step to warmup the torch compiler
        output = pipe(**get_pipe_kwargs(input_config))

    torch.cuda.synchronize()
    is_last_process = is_dp_last_group()
    last_wall_s = 0.0

    if args.profile_output != "":
        all_experiments_data = None  # will not save timings from profiling runs
        # Ensure output directory exists
        destination = os.path.join(args.benchmark_output_directory, f"traces/{args.profile_output}/")
        if is_last_process:
            os.makedirs(destination, exist_ok=True)

        def trace_handler(prof):
            trace_output_file = os.path.join(destination, f"flux_traces_rank{get_world_group().rank}.json.gz")
            prof.export_chrome_trace(trace_output_file)

        schedule = torch.profiler.schedule(
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
        )
        n_reps = args.profile_wait + args.profile_warmup + args.profile_active
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=schedule,
            on_trace_ready=trace_handler,
        ) as prof:
            for i in range(n_reps):
                with record_function("model_inference"):
                    output = pipe(**get_pipe_kwargs(input_config))
                torch.cuda.synchronize()
                prof.step()
    else:
        if args.warmup_calls:
            if is_last_process:
                print(f"Warming up for {args.warmup_calls} calls.")
            for i in range(args.warmup_calls):
                pipe_kwargs = get_pipe_kwargs(input_config)
                output, events = time_pipe_func(pipe=pipe, pipe_kwargs=pipe_kwargs)
                if is_last_process:
                    pipe_elapsed_time = events["start"].elapsed_time(events["end"]) / 1000
                    last_inference_step_idx = pipe_kwargs['num_inference_steps'] - 1
                    vae_elapsed_time = (
                        events[f"inference_step_end_{last_inference_step_idx}"].elapsed_time(events["end"]) / 1000
                    )
                    print(
                        f"Warmup call {i + 1} out of {args.warmup_calls} completed [Pipe time {pipe_elapsed_time:.2f}, VAE elapsed time {vae_elapsed_time:.2f}]."
                    )

        all_experiments_data = []
        for i in range(1, args.num_repetitions + 1):
            wall_start = time.time()
            pipe_kwargs = get_pipe_kwargs(input_config)
            output, events = time_pipe_func(pipe=pipe, pipe_kwargs=pipe_kwargs)
            last_wall_s = time.time() - wall_start
            pipe_elapsed_time = events["start"].elapsed_time(events["end"]) / 1000

            if input_config.output_type != "latent":
                last_inference_step_idx = pipe_kwargs['num_inference_steps'] - 1
                vae_elapsed_time = (
                    events[f"inference_step_end_{last_inference_step_idx}"].elapsed_time(events["end"]) / 1000
                )

            if is_last_process:
                print(f"Iteration {i}")
                print(f"Pipe epoch time: {pipe_elapsed_time:.2f} sec")
                data = {
                    "pipe_time": pipe_elapsed_time,
                }
                if input_config.output_type != "latent":
                    print(f"VAE decode epoch time: {vae_elapsed_time:.2f} sec")
                    data["vae_decode_time"] = vae_elapsed_time

                all_experiments_data.append(data)

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
        if is_last_process:
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = f"flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}_{input_config.height}x{input_config.width}.png"
                destination = os.path.join(args.benchmark_output_directory, "results")
                os.makedirs(destination, exist_ok=True)
                image_filename = os.path.abspath(os.path.join(destination, image_name))
                image.save(image_filename)
                print(f"image {i} saved to {image_filename}")

            # Save data from runs if they exist
            if all_experiments_data is not None:
                timing_filename = os.path.abspath(os.path.join(destination, "timing.json"))
                with open(timing_filename, "w") as f:
                    json.dump(all_experiments_data, f, indent=4)

    _run_flux_fid_sweep(
        args,
        pipe,
        input_config,
        is_last_process,
        last_wall_s,
    )

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
