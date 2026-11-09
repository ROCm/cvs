'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for the shared inference config base in cvs.lib.conf_lib.schemas.inference.
'''

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from pydantic import Field

from cvs.lib.conf_lib.loader import ConfigError, load_config
from cvs.lib.conf_lib.schemas.inference import (
    ContainerEnvTokenError,
    InferenceConfig,
    Port,
    RunCell,
    ServingConfig,
    ServingServerParams,
    format_cell_key,
)
from cvs.unittests.support import FAKE_CRED


CELL = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"
CELL_32 = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=32"
DP_CELL = "ISL=1024,OSL=1024,TP=8,PP=1,DP=2,CONC=16"
CLUSTER = {"username": "test"}


class _EngineServerParams(ServingServerParams):
    port: Port = 9000
    executor: str = "mp"

    @classmethod
    def reserved_engine_args(cls):
        return super().reserved_engine_args() | {"tensor_parallel_size"}


class _EngineConfig(ServingConfig):
    server_params: _EngineServerParams

    @classmethod
    def non_cell_threshold_keys(cls):
        return super().non_cell_threshold_keys() | {"mtp_quality"}


class _NonServingConfig(InferenceConfig):
    benchmark_params: Dict[str, Any] = Field(default_factory=dict)


def _spec(value=100.0):
    return {"output_throughput": {"kind": "min", "value": value}}


def _config(**overrides):
    value = {
        "enforce_thresholds": True,
        "threshold_json": "threshold.json",
        "paths": {
            "shared_fs": "/home/test",
            "models_dir": "/models",
            "log_dir": "/logs",
            "hf_token_file": "/home/test/.hf_token",
        },
        "container": {"name": "engine", "image": "image", "runtime": {"name": "docker", "args": {}}},
        "server_params": {"model": "/models/Llama-3.3-70B-Instruct", "tp": 8, "pp": 2},
        "benchmark_params": {"num_prompts": 25},
        "sweeps": {CELL: {}},
        "runs": [CELL],
    }
    value.update(overrides)
    return value


class _LoadCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _load(self, config, thresholds=None, model_cls=ServingConfig, cluster=CLUSTER):
        (self.tmp / "config.json").write_text(json.dumps(config))
        (self.tmp / "threshold.json").write_text(json.dumps({CELL: _spec()} if thresholds is None else thresholds))
        return load_config(model_cls, self.tmp / "config.json", cluster)

    def _load_error(self, config, thresholds=None, model_cls=ServingConfig):
        with self.assertRaises(ConfigError) as ctx:
            self._load(config, thresholds, model_cls)
        return str(ctx.exception)


class TestExtension(_LoadCase):
    def test_engine_subclass_loads_its_own_fields_and_merges_the_container(self):
        config = _config()
        config["server_params"].update({"executor": "ray", "engine_args": {"kv_cache_dtype": "fp8"}})
        config["container"]["env"] = {"NCCL_DEBUG": "INFO"}
        cluster = {
            "username": "test",
            "container": {
                "env": {"RCCL_MSCCL_ENABLE": "0"},
                "runtime": {"name": "docker", "args": {"shm_size": "64g"}},
            },
        }
        thresholds = {CELL: _spec(), "mtp_quality": {"acceptance": {"kind": "min", "value": 0.5}}}

        loaded = self._load(config, thresholds, _EngineConfig, cluster)

        server = loaded.config.server_params
        self.assertEqual((server.executor, server.port), ("ray", 9000))
        self.assertEqual(server.engine_args, {"kv_cache_dtype": "fp8"})
        self.assertIn("mtp_quality", loaded.config.thresholds)
        self.assertEqual(loaded.container["env"], {"RCCL_MSCCL_ENABLE": "0", "NCCL_DEBUG": "INFO"})
        self.assertEqual(loaded.container["runtime"]["args"], {"shm_size": "64g"})

    def test_non_serving_subclass_loads_without_serving_fields(self):
        config = _config(benchmark_params={"prompt": "a cat", "num_inference_steps": 28})
        for key in ("sweeps", "runs"):
            del config[key]
        config["server_params"] = {"model": "/models/FLUX.1-dev"}
        thresholds = {"expected_results": {"fid": {"kind": "max", "value": 30.0}}}

        loaded = self._load(config, thresholds, _NonServingConfig)

        self.assertEqual(loaded.config.benchmark_params["num_inference_steps"], 28)
        self.assertEqual(loaded.config.thresholds, thresholds)
        self.assertTrue(loaded.config.enforce_thresholds)


class TestStrictness(_LoadCase):
    def test_unknown_keys_fail_at_load_naming_their_location(self):
        cases = (
            ("server_params", "tensor_parallel_size", 8),
            ("benchmark_params", "temperature", 0.0),
            (None, "nnodes", 2),
        )
        for section, key, value in cases:
            with self.subTest(key=key):
                config = _config()
                (config[section] if section else config)[key] = value
                location = f"{section}.{key}" if section else key
                self.assertIn(f"{location}: Extra inputs are not permitted", self._load_error(config))

    def test_unknown_runtime_keys_fail(self):
        config = _config()
        config["container"]["runtime"]["args"] = {"volumes": ["/models:/models"], "mount_everything": True}
        self.assertIn("container.runtime.args.mount_everything", self._load_error(config))
        config = _config()
        config["container"]["runtime"]["flavor"] = "rootless"
        self.assertIn("container.runtime.flavor: Extra inputs are not permitted", self._load_error(config))


class TestContainer(_LoadCase):
    def test_runtime_args_env_fails(self):
        config = _config()
        config["container"]["runtime"]["args"] = {"env": {"NCCL_DEBUG": "INFO"}}
        self.assertIn("set these variables in container.env", self._load_error(config))

    def test_hf_token_in_container_env_fails_without_echoing_its_value(self):
        for name in ("HF_TOKEN", "hugging_face_hub_token"):
            with self.subTest(name=name):
                config = _config()
                config["container"]["env"] = {"NCCL_DEBUG": "INFO", name: FAKE_CRED}
                with self.assertRaises(ContainerEnvTokenError) as ctx:
                    self._load(config)
                self.assertIn(name, str(ctx.exception))
                self.assertNotIn(FAKE_CRED, repr(ctx.exception.args))


class TestEngineArgs(_LoadCase):
    def _with_engine_args(self, args, section="server_params"):
        config = _config()
        config[section]["engine_args"] = args
        return config

    def test_rejects_values_the_engine_cannot_receive(self):
        cases = (
            ({"enforce_eager": False}, "enforce_eager=false is ambiguous"),
            ({"cuda_graph_sizes": [[1, 2]]}, "cuda_graph_sizes lists may contain scalar values only"),
            ({"lora_modules": [{"name": "a"}]}, "lora_modules lists may contain scalar values only"),
            ({"Kv_Cache_Dtype": "fp8"}, "'Kv_Cache_Dtype'"),
            ({"kv--cache": "fp8"}, "'kv--cache'"),
            ({"kv-": "fp8"}, "'kv-'"),
            ({"max_num_seqs": 64, "max-num-seqs": 128}, "'max_num_seqs' and 'max-num-seqs' set the same option"),
            ({"tokenizer": "<changeme>"}, "server_params.engine_args.tokenizer: still a <changeme> placeholder"),
            ({"hf_overrides": {"rope": "<changeme-scaling>"}}, "server_params.engine_args.hf_overrides: still a"),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertIn(expected, self._load_error(self._with_engine_args(args)))

    def test_null_loads_and_comment_keys_are_dropped(self):
        config = self._with_engine_args({"_comment": "flags", "enable_prefix_caching": None, "max-num-seqs": 64})
        server = self._load(config).config.server_params
        self.assertEqual(server.engine_args, {"enable_prefix_caching": None, "max-num-seqs": 64})

    def test_reserved_names_fail_in_either_spelling(self):
        cases = (
            ("server_params", {"model": "/models/other"}),
            ("server_params", {"master-addr": "10.0.0.1"}),
            ("server_params", {"node_rank": 1}),
            ("server_params", {"max-model-len": 4096}),
            ("server_params", {"tp": 4}),
            ("benchmark_params", {"base-url": "http://localhost:8000"}),
            ("benchmark_params", {"max_concurrency": 64}),
            ("benchmark_params", {"random-input-len": 2048}),
            ("benchmark_params", {"num_prompts": 10}),
        )
        for section, args in cases:
            with self.subTest(args=args):
                error = self._load_error(self._with_engine_args(args, section))
                self.assertIn(f"{section}: engine_args cannot set fields CVS renders: {sorted(args)}", error)

    def test_subclass_reserves_its_own_names(self):
        for name in ("tensor_parallel_size", "tensor-parallel-size"):
            with self.subTest(name=name):
                config = self._with_engine_args({name: 8})
                self.assertIn(
                    f"engine_args cannot set fields CVS renders: ['{name}']",
                    self._load_error(config, model_cls=_EngineConfig),
                )
                self.assertEqual(self._load(config).config.server_params.engine_args, {name: 8})


class TestParallelism(_LoadCase):
    def test_ep_is_reserved(self):
        config = _config()
        config["server_params"]["ep"] = 2
        self.assertIn("server_params.ep: ep is reserved", self._load_error(config))

    def test_dp_loads_with_a_dp_cell_key(self):
        config = _config(sweeps={DP_CELL: {}}, runs=[DP_CELL])
        config["server_params"].update({"pp": 1, "dp": 2})
        variant = self._load(config, {DP_CELL: _spec()}).config
        self.assertEqual(variant.cell(DP_CELL).dp, 2)
        self.assertEqual(variant.cell_key(1024, 1024, 16), DP_CELL)

    def test_tp_must_be_positive(self):
        config = _config()
        config["server_params"]["tp"] = 0
        self.assertIn("server_params.tp", self._load_error(config))


class TestRunCell(unittest.TestCase):
    def test_parses_canonical_keys_with_optional_dp(self):
        cell = RunCell.parse(CELL)
        self.assertEqual(
            (cell.isl, cell.osl, cell.tp, cell.pp, cell.dp, cell.ep, cell.concurrency), (1024, 1024, 8, 2, 1, 1, 16)
        )
        self.assertEqual(RunCell.parse(DP_CELL).dp, 2)

    def test_rejects_non_canonical_keys(self):
        cases = (
            ("ISL=1024,OSL=1024,TP=8,PP=1,DP=1,CONC=16", "omit DP=1"),
            ("ISL=1024,OSL=1024,TP=8,PP=1,EP=2,CONC=16", "EP is reserved"),
            ("ISL=1024,OSL=1024,TP=8,PP=1,DP=2,EP=2,CONC=16", "EP is reserved"),
            ("ISL=01024,OSL=1024,TP=8,PP=2,CONC=16", "cell key must be"),
            ("OSL=1024,ISL=1024,TP=8,PP=2,CONC=16", "cell key must be"),
            ("ISL=1024,OSL=1024,TP=8,PP=1,CONC=16,DP=2", "cell key must be"),
            ("ISL=1024,OSL=1024,TP=8,CONC=16", "cell key must be"),
        )
        for key, expected in cases:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, expected):
                    RunCell.parse(key)

    def test_format_writes_dp_only_when_two_or_more(self):
        self.assertEqual(format_cell_key(1024, 1024, 8, 2, 16), CELL)
        self.assertEqual(format_cell_key(1024, 1024, 8, 1, 16, dp=2), DP_CELL)
        for key in (CELL, DP_CELL):
            self.assertEqual(RunCell.parse(key).key, key)


class TestRuns(_LoadCase):
    def test_runs_must_be_nonempty_unique_and_known(self):
        cases = (
            ([], "runs must be a nonempty explicit list"),
            ([CELL, CELL], "runs contains duplicate cells"),
            ([CELL, CELL_32], f"runs reference unknown sweeps: ['{CELL_32}']"),
        )
        for runs, expected in cases:
            with self.subTest(runs=runs):
                self.assertIn(expected, self._load_error(_config(runs=runs)))

    def test_sweep_keys_must_match_server_parallelism(self):
        wrong_tp = "ISL=1024,OSL=1024,TP=4,PP=2,CONC=16"
        wrong_dp = "ISL=1024,OSL=1024,TP=8,PP=2,DP=2,CONC=16"
        for key in (wrong_tp, wrong_dp):
            with self.subTest(key=key):
                config = _config(sweeps={key: {}}, runs=[key])
                self.assertIn(
                    f"{key} does not match server_params tp, pp, dp and ep", self._load_error(config, {key: _spec()})
                )

    def test_malformed_sweep_key_names_the_key(self):
        key = "ISL=1024,OSL=1024,TP=8,PP=2,DP=1,CONC=16"
        error = self._load_error(_config(sweeps={key: {}}, runs=[key]), {key: _spec()})
        self.assertIn(f"sweeps.{key}: omit DP=1", error)


class TestSweeps(_LoadCase):
    def _two_cells(self, cell_override):
        config = _config(sweeps={CELL: cell_override, CELL_32: {}}, runs=[CELL, CELL_32])
        config["benchmark_params"]["engine_args"] = {"temperature": 0.0}
        return config

    def test_override_reaches_only_its_own_run(self):
        override = {"num_prompts": 50, "engine_args": {"top_p": 1.0}}
        variant = self._load(self._two_cells(override), {CELL: _spec(), CELL_32: _spec()}).config
        mine, other = variant.run_benchmark_params(CELL), variant.run_benchmark_params(CELL_32)
        self.assertEqual((mine.num_prompts, mine.engine_args), (50, {"temperature": 0.0, "top_p": 1.0}))
        self.assertEqual((other.num_prompts, other.engine_args), (25, {"temperature": 0.0}))
        self.assertEqual(variant.benchmark_params.num_prompts, 25)

    def test_override_replaces_the_base_value_in_either_spelling(self):
        variant = self._load(
            self._two_cells({"engine_args": {"temperature": 0.5}}), {CELL: _spec(), CELL_32: _spec()}
        ).config
        self.assertEqual(variant.run_benchmark_params(CELL).engine_args, {"temperature": 0.5})
        config = self._two_cells({"engine_args": {"request-rate": 4}})
        config["benchmark_params"]["engine_args"] = {"request_rate": 1}
        variant = self._load(config, {CELL: _spec(), CELL_32: _spec()}).config
        self.assertEqual(variant.run_benchmark_params(CELL).engine_args, {"request-rate": 4})

    def test_comment_keys_in_sweeps_are_dropped(self):
        config = self._two_cells({"_note": "longer prompts", "num_prompts": 50})
        config["sweeps"]["_comment"] = "cells for the 2-node run"
        variant = self._load(config, {CELL: _spec(), CELL_32: _spec()}).config
        self.assertEqual(set(variant.sweeps), {CELL, CELL_32})
        self.assertEqual(variant.sweeps[CELL], {"num_prompts": 50})

    def test_bad_override_fails_naming_its_sweep(self):
        for override, expected in (
            ({"num_prompts": "many"}, f"sweeps.{CELL}: num_prompts:"),
            ({"temperature": 0.0}, f"sweeps.{CELL}: temperature: Extra inputs are not permitted"),
            ({"engine_args": {"enforce_eager": False}}, f"sweeps.{CELL}: engine_args.enforce_eager=false is ambiguous"),
            ({"engine_args": {"max_concurrency": 64}}, f"sweeps.{CELL}: engine_args cannot set fields CVS renders"),
            ({"engine_args": {"tokenizer": "<changeme>"}}, f"sweeps.{CELL}: engine_args.tokenizer: still a <changeme>"),
            ({"dataset_name": "<changeme>"}, f"sweeps.{CELL}: dataset_name: still a <changeme>"),
        ):
            with self.subTest(override=override):
                self.assertIn(expected, self._load_error(self._two_cells(override), {CELL: _spec(), CELL_32: _spec()}))


class TestThresholds(_LoadCase):
    def test_threshold_cell_with_no_sweep_fails(self):
        error = self._load_error(_config(), {CELL: _spec(), CELL_32: _spec()})
        self.assertIn(f"threshold keys match no sweep cell: ['{CELL_32}']", error)

    def test_enforcement_requires_every_selected_run_to_have_a_cell(self):
        config = _config(sweeps={CELL: {}, CELL_32: {}}, runs=[CELL, CELL_32])
        self.assertIn(f"selected runs missing threshold coverage: ['{CELL_32}']", self._load_error(config))
        config["enforce_thresholds"] = False
        self.assertEqual(set(self._load(config).config.thresholds), {CELL})

    def test_non_cell_keys_are_accuracy_plus_what_the_engine_adds(self):
        thresholds = {CELL: _spec(), "accuracy": {"gsm8k": {}}, "mtp_quality": {}}
        self.assertIn("threshold keys match no sweep cell: ['mtp_quality']", self._load_error(_config(), thresholds))
        self.assertIn("mtp_quality", self._load(_config(), thresholds, _EngineConfig).config.thresholds)
        accuracy_only = {CELL: _spec(), "accuracy": {"gsm8k": {}}}
        self.assertIn("accuracy", self._load(_config(), accuracy_only).config.thresholds)

    def test_spec_with_unknown_kind_fails_even_in_an_unselected_cell(self):
        bad = {"output_throughput": {"kind": "at_least", "value": 1.0}}
        error = self._load_error(_config(), {CELL: bad})
        self.assertIn(f"thresholds: {CELL}.output_throughput: Input tag 'at_least'", error)
        config = _config(sweeps={CELL: {}, CELL_32: {}})
        error = self._load_error(config, {CELL: _spec(), CELL_32: bad})
        self.assertIn(f"thresholds: {CELL_32}.output_throughput: Input tag 'at_least'", error)

    def test_comment_metrics_are_dropped_and_other_underscore_keys_fail(self):
        cell = {**_spec(), "_comment": "from the 0.3.0 run"}
        self.assertEqual(self._load(_config(), {CELL: cell}).config.thresholds, {CELL: _spec()})
        for bad, expected in (
            ({"_output_throughput": {"kind": "min", "value": 1.0}}, "unknown metric '_output_throughput'"),
            ({"output_throughput": {"kind": "min", "value": 1.0, "_why": "x"}}, "unknown threshold fields ['_why']"),
        ):
            with self.subTest(bad=bad):
                self.assertIn(expected, self._load_error(_config(), {CELL: bad}))


class TestThresholdFile(_LoadCase):
    def test_named_file_replaces_thresholds_written_in_the_config(self):
        config = _config(thresholds={CELL: _spec(1.0)})
        self.assertEqual(self._load(config).config.thresholds, {CELL: _spec()})

    def test_no_threshold_json_loads_record_only_with_a_warning(self):
        config = _config()
        del config["threshold_json"]
        with self.assertLogs(level="WARNING") as logs:
            variant = self._load(config).config
        self.assertEqual((variant.thresholds, variant.enforce_thresholds), ({}, False))
        self.assertIn("names no threshold_json", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
