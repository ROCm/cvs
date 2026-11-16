'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.schemas.inference.llm_serving.
'''

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


from cvs.config_validation.schemas.inference.accuracy import Accuracy, AccuracyTask
from cvs.config_validation.schemas.inference.llm_serving import (
    LlmServingConfig,
    ServingServerParams,
)
from cvs.config_validation.schemas.sweeps import Axis, Cells
from cvs.config_validation.schemas.thresholds import Thresholds

KEY = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"
OTHER_KEY = "ISL=8192,OSL=1024,TP=8,PP=2,CONC=64"

# Engine and client flags of every JSON type, with a quoted number and a "_" key that must reach the engine as written.
ENGINE_ARGS = {
    "kv_cache_dtype": "fp8",
    "max_model_len": 2056,
    "gpu_memory_utilization": "0.5",
    "trust_remote_code": True,
    "enable_chunked_prefill": False,
    "limit_mm_per_prompt": {"image": 0},
    "_comment": "kept",
}
CLIENT_ARGS = {"temperature": 0.0, "goodput": ["ttft:500", "tpot:50"], "extra_body": {"top_k": 1}, "_note": "kept"}


class _Engine(LlmServingConfig):
    axes = (Axis("ISL", "isl"), Axis("OSL", "osl"), Axis("TP", "tp"), Axis("PP", "pp"), Axis("CONC", "concurrency"))


def _paths():
    return {
        "shared_fs": "/home/jdoe",
        "models_dir": "/models",
        "log_dir": "/home/jdoe/LOGS",
        "hf_token_file": "/home/jdoe/.hf_token",  # nosec B105 - a file path, not a secret
    }


def _full():
    return {
        "_comment": "distributed serving, TP8 PP2",
        "enforce_thresholds": False,
        "threshold_json": "mi325x_llama33-70b_fp8_distributed_threshold.json",
        "paths": {"_comment": "mounts", **_paths()},
        "container": {
            "_comment_image": "a ROCm vLLM image",
            "name": "vllm_llama33-70b_fp8_distributed",
            "image": "rocm/vllm:latest",
            "env": {"NCCL_IB_HCA": "rdma0,rdma1", "NCCL_DEBUG": "ERROR"},
            "runtime": {
                "name": "docker",
                "args": {"network": "host", "ipc": "host", "privileged": True, "devices": ["/dev/dri", "/dev/kfd"]},
            },
        },
        "server_params": {
            "_comment_context": "max_model_len covers ISL plus OSL",
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "tp": 8,
            "pp": 2,
            "engine_args": copy.deepcopy(ENGINE_ARGS),
        },
        "benchmark_params": {"_comment": "client", "num_prompts": 64, "client_args": copy.deepcopy(CLIENT_ARGS)},
        "_comment_cells": "run in order",
        "cells": [KEY, OTHER_KEY],
        "accuracy": {
            "tasks": [
                {"id": "gsm8k_flex", "task": "gsm8k", "num_fewshot": 5, "num_concurrent": 4},
                {"id": "gsm8k_strict", "task": "gsm8k", "metadata": {"filter": "strict-match"}},
            ]
        },
    }


def _as_json(value):
    # json.dumps tells True from 1 and 0.0 from 0, which == does not.
    return json.dumps(value)


class TestServingServerParams(unittest.TestCase):
    def test_comment_keys_are_dropped_but_engine_args_are_kept_as_written(self):
        params = ServingServerParams.model_validate(
            {"_comment": "x", "model": "m", "tp": 8, "engine_args": copy.deepcopy(ENGINE_ARGS)}
        )
        self.assertEqual(_as_json(params.engine_args), _as_json(ENGINE_ARGS))
        self.assertNotIn("_comment", params.model_dump())


class TestLlmServingConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config_file = Path(self._tmp.name) / "serving.json"

    def test_a_full_config_in_the_new_shape_validates(self):
        config = _Engine.validate_config(_full(), self.config_file)
        self.assertEqual(
            (config.enforce_thresholds, config.threshold_json),
            (False, "mi325x_llama33-70b_fp8_distributed_threshold.json"),
        )
        self.assertEqual((config.paths.log_dir, config.container.image), ("/home/jdoe/LOGS", "rocm/vllm:latest"))
        server = config.server_params
        self.assertEqual(
            (server.model, server.tp, server.pp, server.dp, server.ep),
            ("meta-llama/Llama-3.3-70B-Instruct", 8, 2, 1, 1),
        )
        self.assertEqual(_as_json(server.engine_args), _as_json(ENGINE_ARGS))
        self.assertEqual(_as_json(config.benchmark_params.client_args), _as_json(CLIENT_ARGS))
        cells = config.parsed_cells()
        self.assertEqual((config.cells, list(cells)), ([KEY, OTHER_KEY], [KEY, OTHER_KEY]))
        self.assertEqual((cells[OTHER_KEY].isl, cells[OTHER_KEY].tp, cells[OTHER_KEY].concurrency), (8192, 8, 64))
        self.assertEqual(
            [(type(task), task.id) for task in config.accuracy.tasks],
            [(AccuracyTask, "gsm8k_flex"), (AccuracyTask, "gsm8k_strict")],
        )

    def test_a_loader_hook_a_mixin_defines_overrides_the_suite_config_default(self):
        self.assertEqual(LlmServingConfig.old_keys({}), [])
        for mixin in (Accuracy, Cells, Thresholds):
            with self.subTest(mixin=mixin.__name__):
                with mock.patch.object(mixin, "old_keys", classmethod(lambda cls, raw: ["probe"]), create=True):
                    self.assertEqual(LlmServingConfig.old_keys({}), ["probe"])


if __name__ == "__main__":
    unittest.main()
