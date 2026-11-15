'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

The ``cvs config migrate vllm`` map, from the old vLLM config format to
``VllmConfig``.

An old ``server_params`` or ``benchmark_params`` block holds the harness fields
and takes any other key as a ``vllm serve`` or ``vllm bench serve`` flag. Those
keys move into ``server_params.engine_args`` and ``benchmark_params.client_args``
unchanged, keeping the snake_case spelling the commands are built from.

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
    "server_params.backend": None,
    "benchmark_params.backend": None,
    "runs": "cells",
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


def _block_fields(block, model, flags):
    fields = {old: _move_to(new) for old, new in CONFIG.items() if old.startswith(f"{block}.")}
    # Today's schema refuses either backend unless it is vllm.
    fields[f"{block}.backend"] = _left_out_when(("vllm",))
    for name in model.model_fields:
        fields.setdefault(f"{block}.{name}", _move_to(f"{block}.{name}"))
    fields[f"{block}.*"] = _move_to(f"{block}.{flags}.*")
    return fields


def _no_cell_override(value, config):
    if isinstance(value, dict) and all(isinstance(key, str) and key.startswith("_") for key in value):
        return {}
    return TO_UNMAPPED


FIELDS = {
    **_block_fields("server_params", VllmServerParams, "engine_args"),
    **_block_fields("benchmark_params", VllmBenchmarkParams, "client_args"),
    # Each listed value loads today and acts as the default the new format keeps: HCA discovery and
    # no socket fallback. vLLM exports any other ib_netdev, "auto" included, as the socket interface.
    "ib_hca_devices": _left_out_when(("auto", None, [])),
    "ib_netdev": _left_out_when((None, "")),
    "sweeps.*": _no_cell_override,
}
