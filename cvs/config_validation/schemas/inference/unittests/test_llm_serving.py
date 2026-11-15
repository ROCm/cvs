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

from pydantic import ValidationError

from cvs.config_validation.schemas.base import problem_lines
from cvs.config_validation.schemas.inference.accuracy import Accuracy, AccuracyTask
from cvs.config_validation.schemas.inference.llm_serving import (
    LlmServingConfig,
    ServingBenchmarkParams,
    ServingServerParams,
)
from cvs.config_validation.schemas.sweeps import Axis, Cells
from cvs.config_validation.schemas.thresholds import ThresholdFile, Thresholds
from cvs.lib.conf_lib.loader import ConfigError
from cvs.unittests.support import FAKE_CRED

REJECT = {"placeholders": "reject"}
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


def _minimal():
    return {
        "paths": _paths(),
        "server_params": {"model": "meta-llama/Llama-3.3-70B-Instruct", "tp": 8},
        "benchmark_params": {"num_prompts": 64},
        "cells": [KEY],
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


def _errors(model, data, context=None):
    try:
        model.model_validate(data, context=context)
    except ValidationError as exc:
        return sorted((error["loc"], error["type"]) for error in exc.errors())
    raise AssertionError(f"{model.__name__} accepted {data!r}")


def _as_json(value):
    # json.dumps tells True from 1 and 0.0 from 0, which == does not.
    return json.dumps(value)


class TestServingServerParams(unittest.TestCase):
    def test_valid_block_loads_and_defaults_pp_dp_ep_to_one(self):
        params = ServingServerParams.model_validate({"model": "m", "tp": 8})
        self.assertEqual(
            (params.model, params.tp, params.pp, params.dp, params.ep, params.engine_args), ("m", 8, 1, 1, 1, {})
        )
        params = ServingServerParams.model_validate({"model": "m", "tp": "4", "pp": 2, "dp": 2, "ep": 8})
        self.assertEqual((params.tp, params.pp, params.dp, params.ep), (4, 2, 2, 8))

    def test_missing_fields_are_named(self):
        self.assertEqual(_errors(ServingServerParams, {}), [(("model",), "missing"), (("tp",), "missing")])

    def test_wrong_types_are_rejected(self):
        for data, expected in (
            ({"model": 5, "tp": 8}, [(("model",), "string_type")]),
            ({"model": "m", "tp": "eight"}, [(("tp",), "int_parsing")]),
            ({"model": "m", "tp": 8.5}, [(("tp",), "int_from_float")]),
            ({"model": "m", "tp": 8, "pp": None}, [(("pp",), "int_type")]),
            ({"model": "m", "tp": 8, "engine_args": ["--enforce-eager"]}, [(("engine_args",), "dict_type")]),
        ):
            with self.subTest(data=data):
                self.assertEqual(_errors(ServingServerParams, data), expected)

    def test_old_names_and_engine_flags_outside_engine_args_are_rejected(self):
        for key in ("tensor_parallel_size", "pipeline_parallel_size", "port", "nnodes", "kv_cache_dtype"):
            with self.subTest(key=key):
                self.assertEqual(
                    _errors(ServingServerParams, {"model": "m", "tp": 8, key: "1"}), [((key,), "extra_forbidden")]
                )

    def test_comment_keys_are_dropped_but_engine_args_are_kept_as_written(self):
        params = ServingServerParams.model_validate(
            {"_comment": "x", "model": "m", "tp": 8, "engine_args": copy.deepcopy(ENGINE_ARGS)}
        )
        self.assertEqual(_as_json(params.engine_args), _as_json(ENGINE_ARGS))
        self.assertNotIn("_comment", params.model_dump())

    def test_model_rejects_a_placeholder_only_under_the_reject_context(self):
        data = {"model": "<changeme>", "tp": 8}
        self.assertEqual(_errors(ServingServerParams, data, context=REJECT), [(("model",), "value_error")])
        self.assertEqual(ServingServerParams.model_validate(data).model, "<changeme>")


class TestServingBenchmarkParams(unittest.TestCase):
    def test_valid_block_loads_and_client_args_default_to_empty(self):
        self.assertEqual(
            ServingBenchmarkParams.model_validate({"num_prompts": 320}).model_dump(),
            {"num_prompts": 320, "client_args": {}},
        )

    def test_missing_field_and_wrong_types_are_named(self):
        self.assertEqual(_errors(ServingBenchmarkParams, {}), [(("num_prompts",), "missing")])
        self.assertEqual(_errors(ServingBenchmarkParams, {"num_prompts": "many"}), [(("num_prompts",), "int_parsing")])
        self.assertEqual(
            _errors(ServingBenchmarkParams, {"num_prompts": 1, "client_args": "--seed 0"}),
            [(("client_args",), "dict_type")],
        )

    def test_client_flags_outside_client_args_are_rejected(self):
        for key in ("temperature", "dataset_name", "client_poll_iterations"):
            with self.subTest(key=key):
                self.assertEqual(
                    _errors(ServingBenchmarkParams, {"num_prompts": 1, key: 0}), [((key,), "extra_forbidden")]
                )

    def test_comment_keys_are_dropped_but_client_args_are_kept_as_written(self):
        params = ServingBenchmarkParams.model_validate(
            {"_comment": "x", "num_prompts": 1, "client_args": copy.deepcopy(CLIENT_ARGS)}
        )
        self.assertEqual(_as_json(params.model_dump()), _as_json({"num_prompts": 1, "client_args": CLIENT_ARGS}))


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

    def test_comment_keys_are_dropped_from_every_block(self):
        dump = _Engine.validate_config(_full(), self.config_file).model_dump()
        for where, block in (
            ("config", dump),
            ("paths", dump["paths"]),
            ("container", dump["container"]),
            ("server_params", dump["server_params"]),
            ("benchmark_params", dump["benchmark_params"]),
        ):
            with self.subTest(block=where):
                self.assertEqual([key for key in block if key.startswith("_")], [])

    def test_a_minimal_config_takes_every_default(self):
        config = _Engine.validate_config(_minimal(), self.config_file)
        self.assertEqual((config.threshold_json, config.enforce_thresholds, config.container), ("", True, None))
        self.assertEqual(
            (config.accuracy.tasks, config.server_params.pp, config.benchmark_params.client_args), ([], 1, {})
        )
        self.assertIsInstance(config.thresholds, ThresholdFile)
        self.assertEqual((config.thresholds.model_extra, config.thresholds.accuracy), ({}, {}))

    def test_missing_required_blocks_are_named(self):
        self.assertEqual(
            _errors(LlmServingConfig, {}),
            [
                (("benchmark_params",), "missing"),
                (("cells",), "missing"),
                (("paths",), "missing"),
                (("server_params",), "missing"),
            ],
        )

    def test_sweeps_and_runs_are_refused_as_unknown_keys(self):
        raw = _full()
        raw.update(sweeps={KEY: {}}, runs=[KEY])
        self.assertEqual(_errors(_Engine, raw), [(("runs",), "extra_forbidden"), (("sweeps",), "extra_forbidden")])

    def test_cell_problems_are_named_with_the_key(self):
        for cells, expected in (
            ([], "cells: needs at least one cell key"),
            ([KEY, OTHER_KEY, KEY], f"cells: cell key {KEY!r} is listed more than once"),
            (
                [KEY, "ISL=1024,OSL=1024"],
                "cells: cell key 'ISL=1024,OSL=1024' is missing axis TP, PP, CONC; "
                "expected ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>",
            ),
        ):
            raw = _full()
            raw["cells"] = cells
            with self.subTest(cells=cells):
                with self.assertRaises(ValidationError) as ctx:
                    _Engine.model_validate(raw)
                self.assertEqual(problem_lines(ctx.exception), [expected])

    def test_fields_and_config_come_from_every_base_in_file_order(self):
        # pydantic orders inherited fields by reverse MRO, so listing the mixins first puts paths first.
        self.assertEqual(
            list(LlmServingConfig.model_fields),
            [
                "paths",
                "container",
                "threshold_json",
                "enforce_thresholds",
                "cells",
                "accuracy",
                "thresholds",
                "server_params",
                "benchmark_params",
            ],
        )
        self.assertEqual(LlmServingConfig.model_config["extra"], "forbid")

    def test_a_loader_hook_a_mixin_defines_overrides_the_suite_config_default(self):
        self.assertEqual(LlmServingConfig.old_keys({}), [])
        for mixin in (Accuracy, Cells, Thresholds):
            with self.subTest(mixin=mixin.__name__):
                with mock.patch.object(mixin, "old_keys", classmethod(lambda cls, raw: ["probe"]), create=True):
                    self.assertEqual(LlmServingConfig.old_keys({}), ["probe"])

    def test_each_bases_checks_run_on_the_combined_class(self):
        for base, change, expected in (
            ("WorkloadConfig", lambda raw: raw["paths"].pop("log_dir"), [(("paths", "log_dir"), "missing")]),
            (
                "WorkloadConfig",
                lambda raw: raw["container"].update(lifetime="forever"),
                [(("container", "lifetime"), "literal_error")],
            ),
            (
                "Thresholds",
                lambda raw: raw.update(enforce_thresholds="sometimes"),
                [(("enforce_thresholds",), "bool_parsing")],
            ),
            (
                "ThresholdFile",
                lambda raw: raw.update(thresholds={KEY: {"output_throughput": {"kind": "min_tok_s", "value": 1}}}),
                [(("thresholds", KEY, "output_throughput", "kind"), "literal_error")],
            ),
            ("Cells", lambda raw: raw["cells"].append("ISL=1024,OSL=1024"), [(("cells",), "value_error")]),
            ("Cells", lambda raw: raw.update(cells=KEY), [(("cells",), "list_type")]),
            ("Cells", lambda raw: raw["cells"].append(16), [(("cells", 2), "string_type")]),
            (
                "Accuracy",
                lambda raw: raw["accuracy"]["tasks"].append({"id": "gsm8k_flex", "task": "gsm8k"}),
                [(("accuracy",), "value_error")],
            ),
        ):
            raw = _full()
            change(raw)
            with self.subTest(base=base, expected=expected):
                self.assertEqual(_errors(_Engine, raw), expected)

    def test_unknown_keys_are_refused_at_each_level_naming_fields_not_values(self):
        raw = _full()
        raw["ib_netdev"] = FAKE_CRED
        raw["server_params"]["tensor_parallel_size"] = FAKE_CRED
        raw["benchmark_params"]["temperature"] = FAKE_CRED
        with self.assertRaises(ConfigError) as ctx:
            _Engine.validate_config(raw, self.config_file)
        self.assertEqual(
            sorted(line.split(":", 1)[0] for line in ctx.exception.problems),
            ["benchmark_params.temperature", "ib_netdev", "server_params.tensor_parallel_size"],
        )
        self.assertNotIn(FAKE_CRED, str(ctx.exception))

    def test_placeholders_are_refused_only_under_the_reject_context(self):
        raw = _full()
        raw["server_params"]["model"] = "<changeme>"
        raw["paths"]["models_dir"] = "<changeme-models-mount>"
        raw["container"]["image"] = "<changeme>"
        self.assertEqual(
            _errors(_Engine, raw, context=REJECT),
            [
                (("container", "image"), "value_error"),
                (("paths", "models_dir"), "value_error"),
                (("server_params", "model"), "value_error"),
            ],
        )
        config = _Engine.validate_config(raw, self.config_file)
        self.assertEqual((config.server_params.model, config.container.image), ("<changeme>", "<changeme>"))


if __name__ == "__main__":
    unittest.main()
