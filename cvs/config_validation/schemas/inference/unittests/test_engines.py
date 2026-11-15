'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.schemas.inference.engines.
'''

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from cvs.config_validation.loaders.suite_config import load_suite_config
from cvs.config_validation.schemas.base import SecretInputError
from cvs.config_validation.schemas.inference.engines import (
    CLIENT_RESERVED,
    SERVER_RESERVED,
    ContainerEnvTokenError,
    VllmBenchmarkParams,
    VllmConfig,
    VllmContainerSpec,
    VllmServerParams,
    VllmThresholdFile,
)
from cvs.lib.conf_lib.config_blocks import PLACEHOLDER_MESSAGE
from cvs.lib.conf_lib.loader import ConfigError
from cvs.unittests.support import FAKE_CRED

REJECT = {"placeholders": "reject"}
KEY = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"
OTHER_KEY = "ISL=8192,OSL=1024,TP=8,PP=2,CONC=64"
BAD_KEY = "ISL=1024,OSL=1024,TP=8,PP=2"
EXPECTED = "ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>"
SOCKET_ENV = {"NCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0", "TP_SOCKET_IFNAME": "eth0"}
CLUSTER = {"username": "jdoe", "priv_key_file": "/home/jdoe/.ssh/id_rsa", "node_dict": {"node0": {}}}

# Serve flags of every JSON type that today's shipped configs pass, written as they are today.
ENGINE_ARGS = {
    "kv_cache_dtype": "fp8",
    "max_model_len": 2056,
    "gpu_memory_utilization": "0.5",
    "trust_remote_code": True,
    "limit_mm_per_prompt": {"image": 0},
    "no_enable_prefix_caching": True,
    "max_num_seqs": None,
}


def _server(**overrides):
    return {"engine": "vllm", "model": "meta-llama/Llama-3.3-70B-Instruct", "tp": 8, "pp": 2, **overrides}


def _spec(kind, value=0):
    return {"kind": kind, "value": value}


def _cell():
    return {"failed": _spec("max"), "output_throughput": _spec("min"), "mean_ttft_ms": _spec("max")}


def _config(**overrides):
    config = {
        "_comment": "TP8 PP2 on two nodes",
        "enforce_thresholds": False,
        "threshold_json": "threshold.json",
        "paths": {
            "shared_fs": "/home/jdoe",
            "models_dir": "/models",
            "log_dir": "/home/jdoe/LOGS",
            "hf_token_file": "/home/jdoe/.hf_token",  # nosec B105 - a file path, not a secret
        },
        "container": {"name": "vllm", "image": "rocm/vllm:latest", "env": {**SOCKET_ENV, "NCCL_IB_HCA": "rdma0"}},
        "server_params": _server(engine_args=copy.deepcopy(ENGINE_ARGS)),
        "benchmark_params": {"num_prompts": 64, "client_args": {"temperature": 0.0}},
        "cells": [KEY],
        "accuracy": {"tasks": [{"id": "gsm8k_flex", "task": "gsm8k", "num_fewshot": 5}]},
        "thresholds": {KEY: _cell()},
    }
    config.update(overrides)
    return config


def _errors(model, data, context=None):
    try:
        model.model_validate(data, context=context)
    except ValidationError as exc:
        return [(".".join(str(part) for part in error["loc"]), error["msg"]) for error in exc.errors()]
    raise AssertionError(f"{model.__name__} accepted {data!r}")


def _types(model, data):
    try:
        model.model_validate(data)
    except ValidationError as exc:
        return sorted((".".join(str(part) for part in error["loc"]), error["type"]) for error in exc.errors())
    raise AssertionError(f"{model.__name__} accepted {data!r}")


class TestVllmContainerSpec(unittest.TestCase):
    def test_runtime_defaults_match_todays_dump(self):
        spec = VllmContainerSpec.model_validate({"name": "vllm", "image": "img"})
        self.assertEqual(
            spec.model_dump(),
            {
                "lifetime": "per_run",
                "name": "vllm",
                "image": "img",
                "runtime": {
                    "name": "docker",
                    "args": {"volumes": [], "devices": [], "network": "host", "ipc": "host", "privileged": True},
                },
                "env": {},
            },
        )

    def test_name_and_image_are_required(self):
        self.assertEqual(_types(VllmContainerSpec, {}), [("image", "missing"), ("name", "missing")])

    def test_runtime_args_comments_are_dropped_and_written_args_are_kept(self):
        args = {"_comment_bnxt_mount": "x", "volumes": ["/a:/a"], "devices": ["/dev/kfd"], "shm_size": "16g"}
        spec = VllmContainerSpec.model_validate(
            {"name": "v", "image": "i", "runtime": {"name": "docker", "args": args}}
        )
        self.assertEqual(
            spec.model_dump()["runtime"]["args"],
            {
                "volumes": ["/a:/a"],
                "devices": ["/dev/kfd"],
                "shm_size": "16g",
                "network": "host",
                "ipc": "host",
                "privileged": True,
            },
        )

    def test_runtime_refuses_env_an_unknown_arg_another_runtime_and_an_unknown_key(self):
        for runtime, expected in (
            ({"args": {"env": {}}}, ("runtime.args", "vLLM does not read runtime.args.env")),
            ({"args": {"env": None}}, ("runtime.args", "vLLM does not read runtime.args.env")),
            ({"args": {"bogus": 1}}, ("runtime.args.bogus", "Extra inputs are not permitted")),
            ({"name": "podman"}, ("runtime.name", "Input should be 'docker'")),
            ({"bogus": 1}, ("runtime.bogus", "Extra inputs are not permitted")),
        ):
            with self.subTest(runtime=runtime):
                (loc, msg), *rest = _errors(VllmContainerSpec, {"name": "v", "image": "i", "runtime": runtime})
                self.assertEqual((loc, rest), (expected[0], []))
                self.assertIn(expected[1], msg)

    def test_a_hugging_face_token_name_raises_without_its_value(self):
        for name in ("HF_TOKEN", "hugging_face_hub_token"):
            with self.subTest(name=name):
                with self.assertRaises(ContainerEnvTokenError) as ctx:
                    VllmContainerSpec.model_validate({"name": "v", "image": "i", "env": {name: FAKE_CRED}})
                self.assertIn(repr([name]), str(ctx.exception))
                self.assertNotIn(FAKE_CRED, str(ctx.exception))
        self.assertTrue(issubclass(ContainerEnvTokenError, SecretInputError))
        self.assertFalse(issubclass(ContainerEnvTokenError, ValueError))

    def test_socket_variables_are_set_all_together_and_never_empty(self):
        for env, message in (
            (
                {"NCCL_SOCKET_IFNAME": "eth0"},
                "container.env must set all socket interface variables together; "
                "missing: ['GLOO_SOCKET_IFNAME', 'TP_SOCKET_IFNAME']",
            ),
            (
                {**SOCKET_ENV, "TP_SOCKET_IFNAME": " "},
                "container.env socket interface variables cannot be empty: ['TP_SOCKET_IFNAME']",
            ),
            ({"NCCL_IB_HCA": " , "}, "container.env.NCCL_IB_HCA must name at least one device"),
        ):
            with self.subTest(env=env):
                self.assertEqual(
                    _errors(VllmContainerSpec, {"name": "v", "image": "i", "env": env}),
                    [("", f"Value error, {message}")],
                )

    def test_hca_list_and_socket_flag_read_the_env(self):
        spec = VllmContainerSpec.model_validate({"name": "v", "image": "i", "env": {"NCCL_IB_HCA": " rdma0, ,rdma1"}})
        self.assertEqual((spec.nccl_ib_hcas, spec.socket_env_configured), (["rdma0", "rdma1"], False))
        spec = VllmContainerSpec.model_validate({"name": "v", "image": "i", "env": SOCKET_ENV})
        self.assertEqual((spec.nccl_ib_hcas, spec.socket_env_configured), (None, True))

    def test_a_placeholder_in_a_volume_fails_only_when_placeholders_are_rejected(self):
        data = {"name": "v", "image": "i", "runtime": {"args": {"volumes": ["<changeme-models-mount>:/models"]}}}
        VllmContainerSpec.model_validate(data)
        self.assertEqual(
            _errors(VllmContainerSpec, data, REJECT),
            [("runtime.args.volumes.0", f"Value error, {PLACEHOLDER_MESSAGE}")],
        )


class TestVllmServerParams(unittest.TestCase):
    def test_defaults_are_todays(self):
        params = VllmServerParams.model_validate({"engine": "vllm", "model": "m", "tp": 8})
        self.assertEqual(
            params.model_dump(),
            {
                "model": "m",
                "tp": 8,
                "pp": 1,
                "dp": 1,
                "ep": 1,
                "engine_args": {},
                "engine": "vllm",
                "port": 8888,
                "master_port": 29501,
                "server_poll_iterations": 60,
                "server_poll_wait_s": 60,
                "server_warmup_wait_s": 330,
                "distributed_executor_backend": "mp",
            },
        )

    def test_quoted_numbers_load_as_today(self):
        params = VllmServerParams.model_validate(_server(tp="8", pp="2", port="8888", master_port="29501"))
        self.assertEqual((params.tp, params.pp, params.port, params.master_port), (8, 2, 8888, 29501))

    def test_engine_is_required_and_must_be_vllm(self):
        self.assertEqual(_types(VllmServerParams, {"model": "m", "tp": 8}), [("engine", "missing")])
        self.assertEqual(_types(VllmServerParams, _server(engine="sglang")), [("engine", "literal_error")])

    def test_wrong_types_are_refused(self):
        for overrides, expected in (
            ({"port": "eighty"}, ("port", "int_parsing")),
            ({"master_port": 29501.5}, ("master_port", "int_from_float")),
            ({"server_poll_wait_s": None}, ("server_poll_wait_s", "int_type")),
            ({"distributed_executor_backend": "torch"}, ("distributed_executor_backend", "literal_error")),
            ({"dp": 2}, ("dp", "literal_error")),
            ({"ep": 8}, ("ep", "literal_error")),
            ({"engine_args": ["--enforce-eager"]}, ("engine_args", "dict_type")),
        ):
            with self.subTest(overrides=overrides):
                self.assertEqual(_types(VllmServerParams, _server(**overrides)), [expected])

    def test_old_names_launcher_flags_and_serve_flags_outside_engine_args_are_refused(self):
        for key in ("nnodes", "master_addr", "node_rank", "tensor_parallel_size", "dist_init_port", "backend", "dtype"):
            with self.subTest(key=key):
                self.assertEqual(_types(VllmServerParams, _server(**{key: 1})), [(key, "extra_forbidden")])

    def test_engine_args_are_kept_as_written(self):
        params = VllmServerParams.model_validate(_server(engine_args=copy.deepcopy(ENGINE_ARGS)))
        self.assertEqual(json.dumps(params.engine_args), json.dumps(ENGINE_ARGS))

    def test_engine_args_refuse_every_field_and_harness_name(self):
        names = SERVER_RESERVED | set(VllmServerParams.model_fields) - {"engine_args"}
        self.assertTrue({"nnodes", "master_addr", "tensor_parallel_size", "server_poll_iterations"} <= names)
        for name in sorted(names):
            with self.subTest(name=name):
                self.assertEqual(
                    _errors(VllmServerParams, _server(engine_args={"dtype": "auto", name: 1})),
                    [("engine_args", f"Value error, cannot set harness fields: ['{name}']")],
                )

    def test_engine_args_refuse_flags_that_cannot_be_written_on_the_command_line(self):
        for flags, message in (
            ({"kv-cache-dtype": "fp8"}, "flag names must be snake_case: 'kv-cache-dtype'"),
            ({"Dtype": "auto"}, "flag names must be snake_case: 'Dtype'"),
            ({"enforce_eager": False}, "enforce_eager=false is ambiguous; omit it or use the flag's negative form"),
            ({"cuda_graph_sizes": [1, [2]]}, "cuda_graph_sizes: lists may contain scalar values only"),
            ({"hf_overrides": {"a": {1, 2}}}, "hf_overrides must be JSON serializable"),
        ):
            with self.subTest(flags=flags):
                self.assertEqual(
                    _errors(VllmServerParams, _server(engine_args=flags)), [("engine_args", f"Value error, {message}")]
                )

    def test_a_placeholder_in_an_engine_arg_fails_only_when_placeholders_are_rejected(self):
        for value in ("<changeme>", {"path": "<changeme-dir>"}):
            with self.subTest(value=value):
                data = _server(engine_args={"download_dir": value})
                VllmServerParams.model_validate(data)
                self.assertEqual(
                    _errors(VllmServerParams, data, REJECT),
                    [("engine_args.download_dir", f"Value error, {PLACEHOLDER_MESSAGE}")],
                )


class TestVllmBenchmarkParams(unittest.TestCase):
    def test_defaults_are_todays_without_backend(self):
        self.assertEqual(
            VllmBenchmarkParams().model_dump(),
            {
                "num_prompts": 3200,
                "client_args": {},
                "dataset_name": "random",
                "request_rate": "inf",
                "burstiness": 1.0,
                "tokenizer_mode": "auto",
                "seed": 0,
                "random_range_ratio": 0.0,
                "random_prefix_len": 0,
                "client_poll_iterations": 20,
                "client_poll_wait_s": 60,
                "client_initial_wait_s": 120,
                "trust_remote_code": False,
                "ignore_eos": True,
            },
        )

    def test_quoted_numbers_load_and_a_numeric_request_rate_is_a_float(self):
        params = VllmBenchmarkParams.model_validate({"burstiness": "1.0", "seed": "0", "request_rate": 8})
        self.assertEqual((params.burstiness, params.seed, params.request_rate), (1.0, 0, 8.0))

    def test_backend_and_client_flags_outside_client_args_are_refused(self):
        for key in ("backend", "temperature", "max_concurrency"):
            with self.subTest(key=key):
                self.assertEqual(_types(VllmBenchmarkParams, {key: 1}), [(key, "extra_forbidden")])
        self.assertEqual(_types(VllmBenchmarkParams, {"ignore_eos": "maybe"}), [("ignore_eos", "bool_parsing")])

    def test_client_args_are_kept_and_refuse_every_field_and_harness_name(self):
        params = VllmBenchmarkParams.model_validate({"client_args": {"temperature": 0.0, "goodput": ["ttft:500"]}})
        self.assertEqual(json.dumps(params.client_args), json.dumps({"temperature": 0.0, "goodput": ["ttft:500"]}))
        names = CLIENT_RESERVED | set(VllmBenchmarkParams.model_fields) - {"client_args"}
        self.assertTrue({"base_url", "max_concurrency", "result_dir", "backend"} <= names)
        for name in sorted(names):
            with self.subTest(name=name):
                self.assertEqual(
                    _errors(VllmBenchmarkParams, {"client_args": {"temperature": 0, name: 1}}),
                    [("client_args", f"Value error, cannot set harness fields: ['{name}']")],
                )

    def test_client_args_refuse_false_and_placeholders_when_rejected(self):
        self.assertEqual(
            _errors(VllmBenchmarkParams, {"client_args": {"disable_tqdm": False}}),
            [("client_args", "Value error, disable_tqdm=false is ambiguous; omit it or use the flag's negative form")],
        )
        data = {"client_args": {"label": "<changeme>"}}
        VllmBenchmarkParams.model_validate(data)
        self.assertEqual(
            _errors(VllmBenchmarkParams, data, REJECT), [("client_args.label", f"Value error, {PLACEHOLDER_MESSAGE}")]
        )


class TestVllmThresholdFile(unittest.TestCase):
    def test_each_kind_is_the_metrics_direction_or_info(self):
        cell = {"failed": _spec("max"), "output_throughput": _spec("min", 900), "mean_ttft_ms": _spec("info", 1.5)}
        thresholds = VllmThresholdFile.model_validate({"_comment": "x", KEY: cell})
        kinds = {metric: spec.kind for metric, spec in thresholds.model_extra[KEY].items()}
        self.assertEqual(kinds, {"failed": "max", "output_throughput": "min", "mean_ttft_ms": "info"})

    def test_a_kind_against_the_metrics_direction_or_an_old_kind_is_refused(self):
        for metric, spec, loc, message in (
            ("output_throughput", _spec("max"), KEY, "output_throughput threshold kind must be 'min' or 'info'"),
            ("mean_ttft_ms", _spec("min"), KEY, "mean_ttft_ms threshold kind must be 'max' or 'info'"),
        ):
            with self.subTest(metric=metric):
                self.assertEqual(_errors(VllmThresholdFile, {KEY: {metric: spec}}), [(loc, f"Value error, {message}")])
        self.assertEqual(
            _types(VllmThresholdFile, {KEY: {"mean_ttft_ms": _spec("max_ms")}}),
            [(f"{KEY}.mean_ttft_ms.kind", "literal_error")],
        )

    def test_an_unknown_metric_or_a_stray_underscore_metric_is_refused_and_comments_are_dropped(self):
        self.assertEqual(
            _errors(VllmThresholdFile, {KEY: {"output_throughputs": _spec("min")}}),
            [(KEY, "Value error, unknown vLLM threshold metric 'output_throughputs'")],
        )
        self.assertEqual(
            _errors(VllmThresholdFile, {KEY: {"_output_throughput": _spec("min")}}),
            [(KEY, "Value error, unknown vLLM threshold metric '_output_throughput'")],
        )
        thresholds = VllmThresholdFile.model_validate(
            {KEY: {"_comment_cell": "x", "_example_metric": {"kind": "min"}, "failed": _spec("max")}}
        )
        self.assertEqual(list(thresholds.model_extra[KEY]), ["failed"])

    def test_an_underscore_field_or_optional_in_a_spec_is_refused(self):
        self.assertEqual(
            _errors(VllmThresholdFile, {KEY: {"failed": {**_spec("max"), "_unit": "count", "_note": "x"}}}),
            [(f"{KEY}.failed", "Value error, unknown threshold fields ['_note', '_unit']")],
        )
        self.assertEqual(
            _errors(VllmThresholdFile, {KEY: {"failed": {**_spec("max"), "optional": False}}}),
            [(f"{KEY}.failed", "Value error, vLLM thresholds do not support optional")],
        )

    def test_a_value_must_be_a_finite_json_number(self):
        for value in (True, "0", float("nan"), None):
            with self.subTest(value=value):
                self.assertEqual(
                    [loc for loc, _ in _errors(VllmThresholdFile, {KEY: {"failed": _spec("max", value)}})],
                    [f"{KEY}.failed.value"],
                )

    def test_each_cell_key_must_parse_on_the_vllm_axes(self):
        for key, problem in (
            (BAD_KEY, "is missing axis CONC"),
            ("ISL=1024,OSL=1024,PP=2,TP=8,CONC=16", "has its axes out of order"),
            ("ISL=1024,OSL=1024,TP=08,PP=2,CONC=16", "needs a positive integer for TP"),
            ("ISL=1024,OSL=1024,TP=8,PP=2,CONC=16,DP=1", "has unknown axis 'DP'"),
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    _errors(VllmThresholdFile, {key: _cell()}),
                    [("", f"Value error, cell key '{key}' {problem}; expected {EXPECTED}")],
                )

    def test_accuracy_sections_keep_the_shared_rules(self):
        accuracy = {"gsm8k_flex": {"exact_match,flexible-extract": {**_spec("min", 0.5), "optional": True}}}
        thresholds = VllmThresholdFile.model_validate({"accuracy": accuracy})
        self.assertIs(thresholds.accuracy["gsm8k_flex"]["exact_match,flexible-extract"].optional, True)


class TestVllmConfig(unittest.TestCase):
    def test_a_full_config_loads_and_parses_its_cells_on_the_vllm_axes(self):
        config = VllmConfig.model_validate(_config(cells=[KEY, OTHER_KEY], thresholds={KEY: _cell(), OTHER_KEY: {}}))
        self.assertEqual(VllmConfig.suite, "vllm")
        self.assertEqual(
            config.parsed_cells()[OTHER_KEY]._asdict(),
            {"key": OTHER_KEY, "isl": 8192, "osl": 1024, "tp": 8, "pp": 2, "concurrency": 64},
        )
        self.assertEqual((config.server_params.engine, config.server_params.tp), ("vllm", 8))
        self.assertEqual(config.benchmark_params.client_args, {"temperature": 0.0})
        self.assertEqual(config.container.runtime.args.network, "host")

    def test_container_and_server_params_are_required_and_benchmark_params_defaults(self):
        data = _config()
        del data["container"], data["server_params"], data["benchmark_params"]
        self.assertEqual(_types(VllmConfig, data), [("container", "missing"), ("server_params", "missing")])
        self.assertEqual(
            VllmConfig.model_validate({**_config(), "benchmark_params": {}}).benchmark_params.num_prompts, 3200
        )

    def test_old_keys_are_refused(self):
        for key in ("sweeps", "runs", "ib_hca_devices", "ib_netdev"):
            with self.subTest(key=key):
                self.assertEqual(_types(VllmConfig, _config(**{key: "auto"})), [(key, "extra_forbidden")])

    def test_cells_must_be_listed_once_each(self):
        self.assertEqual(_types(VllmConfig, _config(cells=[])), [("cells", "value_error")])
        self.assertEqual(_types(VllmConfig, _config(cells=[KEY, KEY])), [("cells", "value_error")])

    def test_each_cells_tp_and_pp_must_equal_server_params(self):
        for server in (_server(tp=4), _server(pp=1), _server(tp=-1)):
            with self.subTest(server=server):
                self.assertEqual(
                    _errors(VllmConfig, _config(server_params=server, thresholds={})),
                    [("", f"Value error, cell {KEY} conflicts with server_params.tp or server_params.pp")],
                )

    def test_a_threshold_cell_outside_cells_loads_whether_or_not_thresholds_are_enforced(self):
        for enforce in (False, True):
            with self.subTest(enforce=enforce):
                config = VllmConfig.model_validate(
                    _config(enforce_thresholds=enforce, thresholds={KEY: _cell(), OTHER_KEY: _cell()})
                )
                self.assertEqual((config.cells, list(config.thresholds.model_extra)), ([KEY], [KEY, OTHER_KEY]))

    def test_a_threshold_cell_key_that_does_not_parse_is_refused_in_the_threshold_file(self):
        self.assertEqual(
            _errors(VllmConfig, _config(thresholds={KEY: _cell(), BAD_KEY: _cell()})),
            [("thresholds", f"Value error, cell key '{BAD_KEY}' is missing axis CONC; expected {EXPECTED}")],
        )

    def test_each_threshold_cells_tp_and_pp_must_equal_server_params(self):
        for other in ("ISL=1024,OSL=1024,TP=4,PP=2,CONC=16", "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"):
            with self.subTest(other=other):
                self.assertEqual(
                    _errors(VllmConfig, _config(thresholds={KEY: _cell(), other: _cell()})),
                    [("", f"Value error, threshold cell {other} conflicts with server_params.tp or server_params.pp")],
                )

    def test_a_cell_without_thresholds_is_refused_only_when_thresholds_are_enforced(self):
        data = _config(cells=[KEY, OTHER_KEY], thresholds={KEY: _cell()})
        self.assertIs(VllmConfig.model_validate(data).enforce_thresholds, False)
        self.assertEqual(
            _errors(VllmConfig, {**data, "enforce_thresholds": True}),
            [("", f"Value error, cells missing threshold coverage: ['{OTHER_KEY}']")],
        )
        covered = {**data, "enforce_thresholds": True, "thresholds": {KEY: _cell(), OTHER_KEY: {}}}
        self.assertIs(VllmConfig.model_validate(covered).enforce_thresholds, True)

    def test_an_enforced_config_given_no_thresholds_is_refused_for_every_cell(self):
        data = _config(enforce_thresholds=True, cells=[KEY, OTHER_KEY])
        del data["thresholds"]
        self.assertEqual(
            _errors(VllmConfig, data),
            [("", f"Value error, cells missing threshold coverage: ['{KEY}', '{OTHER_KEY}']")],
        )

    def test_info_thresholds_load_and_a_stray_underscore_metric_is_refused_in_the_config(self):
        cell = {"failed": _spec("info"), "output_throughput": _spec("info")}
        config = VllmConfig.model_validate(_config(enforce_thresholds=True, thresholds={KEY: cell}))
        self.assertEqual({spec.kind for spec in config.thresholds.model_extra[KEY].values()}, {"info"})
        self.assertEqual(
            _errors(VllmConfig, _config(thresholds={KEY: {"_failed": _spec("max")}})),
            [(f"thresholds.{KEY}", "Value error, unknown vLLM threshold metric '_failed'")],
        )


class TestVllmConfigThroughTheLoader(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.cluster = self.tmp / "cluster.json"
        self.cluster.write_text(json.dumps(CLUSTER))

    def _load(self, config, thresholds):
        (self.tmp / "threshold.json").write_text(json.dumps(thresholds))
        path = self.tmp / "vllm.json"
        path.write_text(json.dumps({key: value for key, value in config.items() if key != "thresholds"}))
        return load_suite_config(VllmConfig, str(path), str(self.cluster), mock.Mock())

    def test_the_threshold_file_is_checked_with_vllms_rules_and_gates_coverage(self):
        loaded = self._load(_config(enforce_thresholds=True), {"_comment": "x", KEY: _cell()})
        self.assertEqual(
            list(loaded.config.thresholds.model_extra[KEY]), ["failed", "output_throughput", "mean_ttft_ms"]
        )
        with self.assertRaises(ConfigError) as ctx:
            self._load(_config(), {KEY: {"_failed": _spec("max")}})
        self.assertEqual(ctx.exception.problems, [f"{KEY}: unknown vLLM threshold metric '_failed'"])
        with self.assertRaises(ConfigError) as ctx:
            self._load(_config(enforce_thresholds=True, cells=[KEY, OTHER_KEY]), {KEY: _cell()})
        self.assertEqual(ctx.exception.problems, [f"(config): cells missing threshold coverage: ['{OTHER_KEY}']"])

    def test_a_threshold_file_covering_more_cells_than_the_config_runs_loads_and_a_bad_cell_key_fails(self):
        thresholds = {KEY: _cell(), OTHER_KEY: _cell()}
        loaded = self._load(_config(enforce_thresholds=True), thresholds)
        self.assertEqual((loaded.config.cells, list(loaded.config.thresholds.model_extra)), ([KEY], [KEY, OTHER_KEY]))
        with self.assertRaises(ConfigError) as ctx:
            self._load(_config(enforce_thresholds=True), {**thresholds, BAD_KEY: _cell()})
        self.assertEqual(
            ctx.exception.problems, [f"(config): cell key '{BAD_KEY}' is missing axis CONC; expected {EXPECTED}"]
        )


if __name__ == "__main__":
    unittest.main()
