'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

The ``cvs config migrate vllm`` map, from the old vLLM config format to
``VllmConfig``.

An old ``server_params`` or ``benchmark_params`` block holds the harness fields
and takes any other key as a ``vllm serve`` or ``vllm bench serve`` flag. Those
keys move into ``server_params.engine_args`` and ``benchmark_params.client_args``
unchanged, keeping the snake_case spelling the commands are built from.

``distributed_executor_backend`` moves to the ``vllm`` block, ``max_model_len``
becomes ``server_params.max_model_length``, and the lm-eval tasks move under
``accuracy.lm_eval``. A ``pipeline_parallel_size`` above 1 also sets
``server_params.nnodes``, since every old multinode config runs one pipeline
stage per node. The client's dataset is always ``random``, it always ignores
EOS, and it trusts remote code exactly when the server does, so those three
fields go when they say the same, and to ``_unmapped`` when they do not.
``sweeps`` and ``runs`` keep their shape.

A field function gets a value but not its key, so ``FIELDS`` lists each
renamed key and each field of both blocks ahead of the ``*`` that takes every
other key. Since ``FIELDS`` runs before ``CONFIG``, the renames for those
blocks apply through ``FIELDS``; ``CONFIG`` keeps them as the old keys.
'''

from cvs.config_validation.migrate.engine import TO_UNMAPPED
from cvs.config_validation.schemas.inference.engines import VllmBenchmarkParams, VllmConfig, VllmServerParams

SCHEMA = VllmConfig

CONFIG = {
    "server_params.tensor_parallel_size": "server_params.tp",
    "server_params.pipeline_parallel_size": "server_params.pp",
    "server_params.dist_init_port": "server_params.master_port",
    "server_params.max_model_len": "server_params.max_model_length",
    "server_params.distributed_executor_backend": "vllm.distributed_executor_backend",
    "server_params.backend": None,
    "benchmark_params.backend": None,
    "benchmark_params.dataset_name": None,
    "benchmark_params.ignore_eos": None,
    "benchmark_params.trust_remote_code": None,
    "accuracy.tasks": "accuracy.lm_eval.tasks",
    "ib_hca_devices": None,
    "ib_netdev": None,
}

THRESHOLDS = {"max_ms": "max", "min_tok_s": "min"}


def _move_to(target):
    def move(value, config):
        return {} if target is None else {target: value}

    return move


def _left_out_when(defaults):
    def check(value, config):
        return {} if value in defaults else TO_UNMAPPED

    return check


def _pipeline_and_nodes(value, config):
    try:
        stages = int(value)
    except (TypeError, ValueError):
        return {"server_params.pp": value}
    return {"server_params.pp": value, "server_params.nnodes": stages} if stages > 1 else {"server_params.pp": value}


def _trusts_as_the_server_does(value, config):
    server = config.get("server_params")
    trusted = server.get("trust_remote_code", False) if isinstance(server, dict) else False
    return {} if value == trusted else TO_UNMAPPED


def _server_trust_remote_code(value, config):
    # False passes no flag, the same as leaving the key out.
    return {} if value is False else {"server_params.engine_args.trust_remote_code": value}


def _block_fields(block, model, flags, special):
    fields = {old: _move_to(new) for old, new in CONFIG.items() if old.startswith(f"{block}.")}
    # Today's schema refuses either backend unless it is vllm.
    fields[f"{block}.backend"] = _left_out_when(("vllm",))
    fields.update(special)
    for name in model.model_fields:
        fields.setdefault(f"{block}.{name}", _move_to(f"{block}.{name}"))
    fields[f"{block}.*"] = _move_to(f"{block}.{flags}.*")
    return fields


FIELDS = {
    **_block_fields(
        "server_params",
        VllmServerParams,
        "engine_args",
        {
            "server_params.pipeline_parallel_size": _pipeline_and_nodes,
            "server_params.trust_remote_code": _server_trust_remote_code,
        },
    ),
    **_block_fields(
        "benchmark_params",
        VllmBenchmarkParams,
        "client_args",
        {
            "benchmark_params.dataset_name": _left_out_when(("random",)),
            "benchmark_params.ignore_eos": _left_out_when((True,)),
            "benchmark_params.trust_remote_code": _trusts_as_the_server_does,
        },
    ),
    # Each listed value loads today and acts as the default the new format keeps: HCA discovery and
    # no socket fallback. vLLM exports any other ib_netdev, "auto" included, as the socket interface.
    "ib_hca_devices": _left_out_when(("auto", None, [])),
    "ib_netdev": _left_out_when((None, "")),
}
