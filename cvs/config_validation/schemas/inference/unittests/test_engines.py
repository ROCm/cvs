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

from cvs.config_validation.loaders.suite_config import load_suite_config, mode_from_module
from cvs.config_validation.schemas.base import SecretInputError
from cvs.config_validation.schemas.inference.engines import (
    CLIENT_RESERVED,
    SERVER_RESERVED,
    AtomConfig,
    SglangConfig,
    VllmBenchmarkParams,
    VllmConfig,
    VllmContainerSpec,
    VllmServerParams,
    VllmThresholdFile,
    serialize_cli_options,
)
from cvs.config_validation.schemas.inference.llm_serving import ContainerEnvTokenError
from cvs.core.run_layout import RunLayout
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
OLD_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "migrate"
    / "unittests"
    / "old"
    / "mi3xx_vllm_llama33-70b_fp8_distributed.json"
)

# Serve flags of every JSON type that today's shipped configs pass, written as they are today.
ENGINE_ARGS = {
    "kv_cache_dtype": "fp8",
    "gpu_memory_utilization": "0.5",
    "trust_remote_code": True,
    "limit_mm_per_prompt": {"image": 0},
    "no_enable_prefix_caching": True,
    "max_num_seqs": None,
}


def _server(**overrides):
    return {"model": "meta-llama/Llama-3.3-70B-Instruct", "tp": 8, "pp": 2, **overrides}


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
        "server_params": _server(max_model_length=2056, engine_args=copy.deepcopy(ENGINE_ARGS)),
        "benchmark_params": {"num_prompts": 64, "client_args": {"temperature": 0.0}},
        "sweeps": {KEY: {}},
        "runs": [KEY],
        "accuracy": {"lm_eval": {"tasks": [{"id": "gsm8k_flex", "task": "gsm8k", "num_fewshot": 5}]}},
        "thresholds": {KEY: _cell()},
        "vllm": {},
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

    def test_runtime_args_refuse_env_as_every_serving_config_does(self):
        data = _config()
        data["container"]["runtime"] = {"args": {"env": {"NCCL_DEBUG": "INFO"}}}
        self.assertEqual(
            _errors(VllmConfig, data),
            [("", "Value error, container.runtime.args.env is not read; set these variables in container.env")],
        )

    def test_a_hugging_face_token_name_raises_without_its_value(self):
        for name in ("HF_TOKEN", "hugging_face_hub_token"):
            with self.subTest(name=name):
                data = _config()
                data["container"]["env"][name] = FAKE_CRED
                with self.assertRaises(ContainerEnvTokenError) as ctx:
                    VllmConfig.model_validate(data)
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


class TestVllmServerParams(unittest.TestCase):
    def test_the_shared_fields_and_the_old_wait_take_todays_defaults(self):
        params = VllmServerParams.model_validate({"model": "m", "tp": 8})
        self.assertEqual(
            params.model_dump(),
            {
                "model": "m",
                "tp": 8,
                "pp": 1,
                "dp": 1,
                "ep": 1,
                "port": 8888,
                "master_port": 29501,
                "nnodes": None,
                "prefill_nnodes": None,
                "max_model_length": None,
                "long_context": {"chunked_prefill_size": None, "max_prefill_tokens": None},
                "engine_args": {},
            },
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
    def test_the_shared_fields_and_the_old_wait_take_todays_defaults(self):
        self.assertEqual(
            VllmBenchmarkParams.model_validate({"num_prompts": 64}).model_dump(),
            {
                "num_prompts": 64,
                "random_range_ratio": 0.0,
                "seed": 0,
                "request_rate": "inf",
                "burstiness": 1.0,
                "random_prefix_len": 0,
                "tokenizer_mode": "auto",
                "scaling_baseline_output_throughput": None,
                "client_args": {},
            },
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

    def test_a_threshold_cell_key_must_parse_on_the_vllm_axes(self):
        self.assertEqual(
            _errors(VllmThresholdFile, {BAD_KEY: _cell()}),
            [("", f"Value error, cell key '{BAD_KEY}' is missing axis CONC; expected {EXPECTED}")],
        )

    def test_accuracy_sections_keep_the_shared_rules(self):
        lm_eval = {"gsm8k_flex": {"exact_match,flexible-extract": {**_spec("min", 0.5), "optional": True}}}
        thresholds = VllmThresholdFile.model_validate({"accuracy": {"lm_eval": lm_eval}})
        self.assertIs(thresholds.accuracy.lm_eval["gsm8k_flex"]["exact_match,flexible-extract"].optional, True)


class TestVllmConfig(unittest.TestCase):
    def test_a_full_config_loads_and_parses_its_cells_on_the_vllm_axes(self):
        config = VllmConfig.model_validate(
            _config(sweeps={KEY: {}, OTHER_KEY: {}}, runs=[KEY, OTHER_KEY], thresholds={KEY: _cell(), OTHER_KEY: {}})
        )
        self.assertEqual(VllmConfig.suite, "vllm")
        self.assertEqual(
            config.cells()[OTHER_KEY]._asdict(),
            {"key": OTHER_KEY, "isl": 8192, "osl": 1024, "tp": 8, "pp": 2, "concurrency": 64},
        )
        self.assertEqual((config.server_params.tp, config.server_params.max_model_length), (8, 2056))
        self.assertEqual(config.benchmark_params.client_args, {"temperature": 0.0})
        self.assertEqual(config.container.runtime.args.network, "host")

    def test_the_vllm_block_is_required_and_its_backend_defaults_to_ray(self):
        self.assertEqual(VllmConfig.model_validate(_config()).vllm.distributed_executor_backend, "ray")
        mp = VllmConfig.model_validate(_config(vllm={"distributed_executor_backend": "mp"}))
        self.assertEqual(mp.vllm.distributed_executor_backend, "mp")
        data = _config()
        del data["vllm"]
        self.assertEqual(_errors(VllmConfig, data), [("vllm", "Field required")])
        self.assertEqual(
            _types(VllmConfig, _config(vllm={"distributed_executor_backend": "torch"})),
            [("vllm.distributed_executor_backend", "literal_error")],
        )

    def test_dp_and_ep_stay_one(self):
        for field in ("dp", "ep"):
            with self.subTest(field=field):
                self.assertEqual(
                    _errors(VllmConfig, _config(server_params=_server(**{field: 2}))),
                    [
                        (
                            "",
                            "Value error, vLLM runs with server_params.dp and server_params.ep of 1; "
                            "set its flags in engine_args",
                        )
                    ],
                )

    def test_engine_args_client_args_and_cell_overrides_refuse_harness_names(self):
        self.assertTrue({"nnodes", "master_addr", "max_model_len", "distributed_executor_backend"} <= SERVER_RESERVED)
        self.assertTrue({"base_url", "ignore_eos", "dataset_name", "trust_remote_code"} <= CLIENT_RESERVED)
        for data, message in (
            (
                _config(server_params=_server(engine_args={"dtype": "auto", "max_model_len": 4096})),
                "server_params.engine_args cannot set harness fields: ['max_model_len']",
            ),
            (
                _config(benchmark_params={"num_prompts": 64, "client_args": {"temperature": 0, "base_url": "x"}}),
                "benchmark_params.client_args cannot set harness fields: ['base_url']",
            ),
            (_config(sweeps={KEY: {"ignore_eos": True}}), f"sweeps.{KEY} cannot set harness fields: ['ignore_eos']"),
        ):
            with self.subTest(message=message):
                self.assertEqual(_errors(VllmConfig, data), [("", f"Value error, {message}")])
        overridden = VllmConfig.model_validate(_config(sweeps={KEY: {"num_prompts": 640, "temperature": 0.5}}))
        self.assertEqual(overridden.sweeps[KEY], {"num_prompts": 640, "temperature": 0.5})

    def test_quant_parity_mtp_quality_and_a_scaling_baseline_are_not_implemented(self):
        data = _config(
            accuracy={"quant_parity": {"enabled": True}, "mtp_quality": {"enabled": True}},
            benchmark_params={"num_prompts": 64, "scaling_baseline_output_throughput": 1500.0},
        )
        self.assertEqual(
            _errors(VllmConfig, data),
            [
                (
                    "",
                    "Value error, not implemented for vLLM: accuracy.quant_parity, accuracy.mtp_quality, "
                    "benchmark_params.scaling_baseline_output_throughput",
                )
            ],
        )

    def test_a_threshold_cell_outside_runs_loads_whether_or_not_thresholds_are_enforced(self):
        for enforce in (False, True):
            with self.subTest(enforce=enforce):
                config = VllmConfig.model_validate(
                    _config(enforce_thresholds=enforce, thresholds={KEY: _cell(), OTHER_KEY: _cell()})
                )
                self.assertEqual((config.runs, list(config.thresholds)), ([KEY], [KEY, OTHER_KEY]))

    def test_a_run_without_thresholds_is_refused_only_when_thresholds_are_enforced(self):
        data = _config(sweeps={KEY: {}, OTHER_KEY: {}}, runs=[KEY, OTHER_KEY], thresholds={KEY: _cell()})
        self.assertIs(VllmConfig.model_validate(data).enforce_thresholds, False)
        self.assertEqual(
            _errors(VllmConfig, {**data, "enforce_thresholds": True}),
            [("", f"Value error, runs missing threshold coverage: ['{OTHER_KEY}']")],
        )


class TestSglangAndAtomConfigs(unittest.TestCase):
    def _data(self, block, **overrides):
        data = {key: value for key, value in _config(**overrides).items() if key not in ("vllm", "thresholds")}
        data["container"] = {"name": "x", "image": "img", "runtime": {"name": "docker"}}
        data["server_params"] = _server(engine_args={"attention_backend": "aiter"})
        return {**data, **block}

    def test_each_needs_its_own_block_which_names_its_engine(self):
        self.assertEqual(SglangConfig.model_validate(self._data({"sglang": {}})).sglang.model_dump(), {})
        self.assertEqual(AtomConfig.model_validate(self._data({"atom": {}})).atom.driver, "vllm")
        self.assertEqual(
            _errors(SglangConfig, self._data({"sglang": {"router_policy": "x"}})),
            [("sglang.router_policy", "Extra inputs are not permitted")],
        )
        self.assertEqual(_errors(SglangConfig, self._data({})), [("sglang", "Field required")])
        self.assertEqual(
            _types(AtomConfig, self._data({"atom": {"driver": "trtllm"}})), [("atom.driver", "literal_error")]
        )

    def test_quant_parity_is_refused_on_sglang_and_left_to_atom(self):
        accuracy = {"quant_parity": {"enabled": True}}
        self.assertEqual(
            _errors(SglangConfig, self._data({"sglang": {}}, accuracy=accuracy)),
            [("", "Value error, not implemented for SGLang: accuracy.quant_parity")],
        )
        config = AtomConfig.model_validate(self._data({"atom": {"driver": "sglang"}}, accuracy=accuracy))
        self.assertIs(config.accuracy.quant_parity.enabled, True)


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
        self.assertEqual(list(loaded.config.thresholds[KEY]), ["failed", "output_throughput", "mean_ttft_ms"])
        with self.assertRaises(ConfigError) as ctx:
            self._load(_config(), {KEY: {"_failed": _spec("max")}})
        self.assertEqual(ctx.exception.problems, [f"{KEY}: unknown vLLM threshold metric '_failed'"])
        with self.assertRaises(ConfigError) as ctx:
            data = _config(enforce_thresholds=True, sweeps={KEY: {}, OTHER_KEY: {}}, runs=[KEY, OTHER_KEY])
            self._load(data, {KEY: _cell()})
        self.assertEqual(ctx.exception.problems, [f"(config): runs missing threshold coverage: ['{OTHER_KEY}']"])

    def test_an_old_format_config_fails_naming_its_old_keys_and_the_migrate_command(self):
        with self.assertRaises(ConfigError) as ctx:
            load_suite_config(VllmConfig, str(OLD_CONFIG), str(self.cluster), mock.Mock())
        (problem,) = ctx.exception.problems
        self.assertTrue(
            problem.startswith("old config format: server_params.backend, server_params.tensor_parallel_size")
        )
        self.assertIn("server_params.distributed_executor_backend", problem)
        self.assertIn(", accuracy.tasks;", problem)
        self.assertTrue(problem.endswith(f"; run cvs config migrate vllm {OLD_CONFIG} <new config>"))

    def test_a_wait_field_left_in_a_new_config_fails_as_an_old_key_not_as_a_flag(self):
        config = _config()
        config["server_params"]["server_poll_wait_s"] = 10
        config["benchmark_params"]["client_initial_wait_s"] = 10
        with self.assertRaises(ConfigError) as ctx:
            self._load(config, {KEY: _cell()})
        self.assertEqual(
            ctx.exception.problems,
            [
                "old config format: server_params.server_poll_wait_s, benchmark_params.client_initial_wait_s; "
                f"run cvs config migrate vllm {self.tmp / 'vllm.json'} <new config>"
            ],
        )

    def test_a_changeme_fails_naming_each_field_and_never_its_value(self):
        config = _config()
        config["container"]["runtime"] = {"args": {"volumes": ["<changeme>:/models"]}}
        config["server_params"]["model"] = "<changeme>"
        config["benchmark_params"]["client_args"] = {"tokenizer": "/data/<changeme>/tokenizer"}
        with self.assertRaises(ConfigError) as ctx:
            self._load(config, {KEY: _cell()})
        self.assertEqual(
            ctx.exception.problems,
            [
                f"container.runtime.args.volumes.0: {PLACEHOLDER_MESSAGE}",
                f"server_params.model: {PLACEHOLDER_MESSAGE}",
                f"benchmark_params.client_args.tokenizer: {PLACEHOLDER_MESSAGE}",
            ],
        )
        self.assertNotIn("/data/", str(ctx.exception))

    def test_schema_problems_name_each_field_and_never_its_value(self):
        config = _config(server_params=_server(tp="eight"), token=FAKE_CRED)
        with self.assertRaises(ConfigError) as ctx:
            self._load(config, {KEY: _cell()})
        exc = ctx.exception
        self.assertEqual(sorted(line.split(":", 1)[0] for line in exc.problems), ["server_params.tp", "token"])
        self.assertNotIn(FAKE_CRED, str(exc))
        self.assertIsNone(exc.__cause__)
        self.assertTrue(exc.__suppress_context__)

    def test_every_placeholder_is_filled_before_the_checks(self):
        RunLayout._reset()
        self.addCleanup(RunLayout._reset)
        layout = RunLayout.get(str(self.tmp / "workspace"))
        config = _config()
        config["paths"] = {
            "shared_fs": "/{home-mount-dir}/{user}",
            "models_dir": "{shared_fs}/{node-dir-name}/models",
            "log_dir": "{run_dir}/logs",
            "hf_token_file": "{home}/.hf_token",
        }
        config["container"]["name"] = "vllm-{user-id}"
        config["container"]["env"] = {**SOCKET_ENV, "LOG_DIR": "{paths.log_dir}"}
        self.cluster.write_text(json.dumps({**CLUSTER, "home_mount_dir_name": "mnt", "node_dir_name": "node7"}))
        home = mock.patch(
            "cvs.lib.utils_lib.os.path.expanduser", side_effect=lambda p: "/nfs/jdoe" if p == "~jdoe" else p
        )
        with home:
            loaded = self._load(config, {KEY: _cell()})
        paths = loaded.config.paths
        self.assertEqual(
            (paths.shared_fs, paths.models_dir, paths.log_dir, paths.hf_token_file),
            ("/mnt/jdoe", "/mnt/jdoe/node7/models", f"{layout.run_dir}/logs", "/nfs/jdoe/.hf_token"),
        )
        self.assertEqual(loaded.orch_inputs["container"]["name"], "vllm-jdoe")
        self.assertEqual(loaded.orch_inputs["container"]["env"]["LOG_DIR"], f"{layout.run_dir}/logs")

    def test_a_config_without_run_dir_creates_no_run_layout(self):
        RunLayout._reset()
        self.addCleanup(RunLayout._reset)
        self.assertEqual(self._load(_config(), {KEY: _cell()}).config.paths.log_dir, "/home/jdoe/LOGS")
        self.assertIsNone(RunLayout._instance)

    def test_orch_inputs_merge_the_config_container_onto_the_cluster_files_as_a_copy(self):
        cluster_container = {
            "name": "cluster-container",
            "image": "cluster/image",
            "env": {"A": "cluster", "B": "cluster"},
            "runtime": {
                "name": "docker",
                "args": {"volumes": ["/cluster:/cluster"], "shm_size": "64g", "registry": {"username": "jdoe"}},
            },
        }
        self.cluster.write_text(json.dumps({**CLUSTER, "container": cluster_container}))
        config = _config()
        config["container"]["env"] = {**SOCKET_ENV, "B": "config"}
        config["container"]["runtime"] = {"args": {"volumes": ["/config:/config"]}}
        loaded = self._load(config, {KEY: _cell()})
        self.assertEqual(loaded.orch_inputs["orchestrator"], "container")
        container = loaded.orch_inputs["container"]
        self.assertEqual((container["name"], container["image"]), ("vllm", "rocm/vllm:latest"))
        self.assertEqual(container["env"], {"A": "cluster", "B": "config", **SOCKET_ENV})
        self.assertEqual(container["runtime"]["args"]["volumes"], ["/config:/config"])
        self.assertEqual(container["runtime"]["args"]["shm_size"], "64g")
        container["runtime"]["args"]["registry"]["username"] = "changed"
        self.assertEqual(loaded.cluster["container"]["runtime"]["args"]["registry"], {"username": "jdoe"})


class TestVllmConfigHooks(unittest.TestCase):
    def test_the_module_name_picks_single_or_distributed_and_the_tests_launch_the_container(self):
        self.assertEqual(mode_from_module(VllmConfig, "cvs.tests.inference.vllm.vllm_single"), "single")
        self.assertEqual(mode_from_module(VllmConfig, "cvs.tests.inference.vllm.vllm_distributed"), "distributed")
        with self.assertRaisesRegex(ValueError, "must end with one of: _single, _distributed"):
            mode_from_module(VllmConfig, "cvs.tests.inference.vllm.vllm_disagg")
        self.assertIs(VllmConfig.launch_in_fixture, False)

    def test_old_keys_are_the_paths_migrate_would_move(self):
        old = {"server_params": {"tensor_parallel_size": 8, "model": "m"}, "accuracy": {"tasks": []}}
        self.assertEqual(VllmConfig.old_keys(old), ["server_params.tensor_parallel_size", "accuracy.tasks"])
        self.assertEqual(VllmConfig.old_keys(_config()), [])

    def test_single_takes_the_first_node_and_distributed_the_first_nnodes(self):
        config = VllmConfig.model_validate(_config(server_params=_server(nnodes=2)))
        by_name = {**CLUSTER, "node_dict": {"10.0.0.2": {}, "10.0.0.1": {}, "10.0.0.3": {}}}
        as_list = {**CLUSTER, "node_dict": [{"mgmt_ip": "10.0.0.3"}, {"mgmt_ip": "10.0.0.1"}]}
        self.assertEqual(VllmConfig.select_hosts(config, by_name, "single"), ["10.0.0.2"])
        self.assertEqual(VllmConfig.select_hosts(config, as_list, "single"), ["10.0.0.3"])
        self.assertEqual(VllmConfig.select_hosts(config, by_name, "distributed"), ["10.0.0.2", "10.0.0.1"])
        with self.assertRaisesRegex(ValueError, "distributed needs 2"):
            VllmConfig.select_hosts(config, CLUSTER, "distributed")

    def test_resolved_runs_follow_runs_with_the_benchmark_options_each_cells_overrides_and_lengths(self):
        sweeps = {KEY: {}, OTHER_KEY: {"num_prompts": 640, "temperature": 0.5}}
        config = VllmConfig.model_validate(_config(sweeps=sweeps, runs=[OTHER_KEY, KEY], thresholds={}))
        runs = config.resolved_runs()
        self.assertEqual([run.cell.key for run in runs], [OTHER_KEY, KEY])
        for run, lengths, prompts, temperature in zip(
            runs, ((8192, 1024, 64), (1024, 1024, 16)), (640, 64), (0.5, 0.0)
        ):
            options = run.benchmark_params
            self.assertEqual(
                (options["random_input_len"], options["random_output_len"], options["max_concurrency"]), lengths
            )
            self.assertEqual((options["num_prompts"], options["temperature"]), (prompts, temperature))
            self.assertNotIn("client_args", options)

    def test_cell_key_reads_tp_and_pp_from_server_params_and_model_id_is_the_model(self):
        config = VllmConfig.model_validate(_config())
        self.assertEqual(config.cell_key(isl=1024, osl=1024, concurrency=16, num_prompts=640), KEY)
        self.assertEqual(config.model_id, "meta-llama/Llama-3.3-70B-Instruct")

    def test_thresholds_are_kept_as_written_without_comment_keys(self):
        accuracy = {
            "_comment": "sections",
            "lm_eval": {
                "gsm8k_flex": {"_comment": "z", "exact_match,flexible-extract": {**_spec("min", 0.5), "optional": True}}
            },
        }
        cell = {"_comment": "y", "output_throughput": _spec("min", 900), "failed": _spec("info")}
        config = VllmConfig.model_validate(_config(thresholds={"_comment": "x", KEY: cell, "accuracy": accuracy}))
        self.assertEqual(
            config.thresholds,
            {
                KEY: {"output_throughput": _spec("min", 900), "failed": _spec("info")},
                "accuracy": {
                    "lm_eval": {"gsm8k_flex": {"exact_match,flexible-extract": {**_spec("min", 0.5), "optional": True}}}
                },
            },
        )
        self.assertIs(type(config.thresholds[KEY]["output_throughput"]["value"]), int)


class TestSerializeCliOptions(unittest.TestCase):
    def test_each_value_type_is_written_as_the_command_line_reads_it(self):
        options = {
            "max_num_seqs": None,
            "enforce_eager": True,
            "cuda_graph_sizes": [1, 2],
            "limit_mm_per_prompt": {"image": 0},
            "max_model_len": 2056,
        }
        self.assertEqual(
            serialize_cli_options(options),
            [
                "--enforce-eager",
                "--cuda-graph-sizes",
                "1",
                "2",
                "--limit-mm-per-prompt",
                '{"image":0}',
                "--max-model-len",
                "2056",
            ],
        )

    def test_false_a_non_snake_case_name_and_a_nested_list_are_refused(self):
        for options in ({"enforce_eager": False}, {"max-model-len": 1}, {"sizes": [[1]]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                serialize_cli_options(options)


if __name__ == "__main__":
    unittest.main()
