import torch

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser


def _add_if_missing(parser, *names, **kwargs):
    dest = kwargs.get("dest")
    if dest is None:
        dest = names[0].lstrip("-").replace("-", "_")
    existing = {action.dest for action in parser._actions}
    if dest not in existing:
        parser.add_argument(*names, **kwargs)


def create_arg_parser():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser = xFuserArgs.add_cli_args(parser)
    _add_if_missing(parser, "--benchmark_output_directory", type=str, default=".")
    _add_if_missing(parser, "--num_repetitions", type=int, default=1)
    _add_if_missing(parser, "--rep_sleep_duration", type=float, default=0)
    _add_if_missing(parser, "--warmup_calls", type=int, default=0)
    _add_if_missing(parser, "--profile_output", type=str, default="")
    _add_if_missing(parser, "--profile_wait", type=int, default=0)
    _add_if_missing(parser, "--profile_warmup", type=int, default=0)
    _add_if_missing(parser, "--profile_active", type=int, default=0)
    return parser.parse_args()


def get_pipe_kwargs(input_config):
    kwargs = {
        "height": input_config.height,
        "width": input_config.width,
        "prompt": input_config.prompt,
        "num_inference_steps": input_config.num_inference_steps,
        "output_type": getattr(input_config, "output_type", "pil"),
        "max_sequence_length": input_config.max_sequence_length,
        "generator": torch.Generator(device="cuda").manual_seed(input_config.seed),
    }
    guidance_scale = getattr(input_config, "guidance_scale", None)
    if guidance_scale is not None:
        kwargs["guidance_scale"] = guidance_scale
    return kwargs


def time_pipe_func(pipe, pipe_kwargs):
    """Time one pipe call. Step events fall back to the call end when the pipeline has no callback."""
    events = {}
    step_events = {}

    def _on_step(_pipe, step_index, _timestep, callback_kwargs):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        step_events[step_index] = event
        return callback_kwargs

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call_kwargs = dict(pipe_kwargs)
    call_kwargs["callback_on_step_end"] = _on_step
    output = pipe(**call_kwargs)
    end.record()
    torch.cuda.synchronize()
    events["start"] = start
    events["end"] = end
    events.update({f"inference_step_end_{index}": event for index, event in step_events.items()})
    last_step = int(pipe_kwargs["num_inference_steps"]) - 1
    events.setdefault(f"inference_step_end_{last_step}", end)
    return output, events


def print_model_args(engine_config, input_config, filename=""):
    model = getattr(getattr(engine_config, "model_config", None), "model", "")
    print(
        f"{filename}: model={model} steps={getattr(input_config, 'num_inference_steps', '')} "
        f"size={getattr(input_config, 'height', '')}x{getattr(input_config, 'width', '')}"
    )
