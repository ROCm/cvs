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


from cvs.config_validation.schemas.inference.accuracy import Accuracy, AccuracyTask, NiahEval
from cvs.config_validation.schemas.inference.llm_serving import (
    ContainerEnvTokenError,
    LlmServingConfig,
    ServingBenchmarkParams,
    ServingServerParams,
)
from cvs.config_validation.schemas.sweeps import Axis, SweepsAndRuns
from cvs.config_validation.schemas.thresholds import Thresholds
from cvs.lib.conf_lib.config_blocks import PLACEHOLDER_MESSAGE
from cvs.lib.conf_lib.loader import ConfigError
from cvs.unittests.support import FAKE_CRED

KEY = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"
OTHER_KEY = "ISL=8192,OSL=1024,TP=8,PP=2,CONC=64"

# Engine and client flags of every JSON type they take, with a quoted number that must reach the engine as written.
ENGINE_ARGS = {
    "kv_cache_dtype": "fp8",
    "max_num_seqs": 16,
    "gpu_memory_utilization": "0.5",
    "trust_remote_code": True,
    "limit_mm_per_prompt": {"image": 0},
}
CLIENT_ARGS = {"temperature": 0.0, "goodput": ["ttft:500", "tpot:50"], "extra_body": {"top_k": 1}}


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
        "timeouts": {"_comment": "seconds", "server_ready_s": 3630, "client_s": None},
        "sweeps": {KEY: {}, OTHER_KEY: {"num_prompts": 128, "temperature": 0.0}},
        "_comment_runs": "run in order",
        "runs": [OTHER_KEY, KEY],
        "accuracy": {
            "lm_eval": {
                "tasks": [
                    {"id": "gsm8k_flex", "task": "gsm8k", "num_fewshot": 5, "num_concurrent": 4},
                    {"id": "gsm8k_strict", "task": "gsm8k", "metadata": {"filter": "strict-match"}},
                ]
            },
            "custom_eval": [{"type": "niah", "id": "niah_8k", "isl": 8192}],
        },
    }


def _cluster(*hosts):
    return {"node_dict": {host: {"bmc_ip": "NA"} for host in hosts}, "head_node_dict": {"mgmt_ip": hosts[0]}}


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

    def test_unset_fields_take_the_shared_defaults(self):
        server = ServingServerParams.model_validate({"model": "m", "tp": 8})
        self.assertEqual((server.pp, server.dp, server.ep, server.port, server.master_port), (1, 1, 1, 8888, 29501))
        self.assertEqual((server.nnodes, server.prefill_nnodes, server.max_model_length), (None, None, None))
        self.assertEqual(server.long_context.model_dump(), {"chunked_prefill_size": None, "max_prefill_tokens": None})
        client = ServingBenchmarkParams.model_validate({"num_prompts": 320})
        self.assertEqual(
            (client.random_range_ratio, client.seed, client.request_rate, client.burstiness),
            (0.0, 0, "inf", 1.0),
        )
        self.assertEqual(
            (client.random_prefix_len, client.tokenizer_mode, client.scaling_baseline_output_throughput),
            (0, "auto", None),
        )

    def test_a_placeholder_in_an_engine_arg_fails_only_when_placeholders_are_rejected(self):
        for value in ("<changeme>", {"path": "<changeme-dir>"}):
            with self.subTest(value=value):
                data = {"model": "m", "tp": 8, "engine_args": {"download_dir": value}}
                ServingServerParams.model_validate(data)
                with self.assertRaises(ValidationError) as caught:
                    ServingServerParams.model_validate(data, context={"placeholders": "reject"})
                self.assertEqual(
                    [(error["loc"], error["msg"]) for error in caught.exception.errors()],
                    [(("engine_args", "download_dir"), f"Value error, {PLACEHOLDER_MESSAGE}")],
                )

    def test_flag_maps_take_snake_case_names_with_renderable_values(self):
        for flags, problem in (
            ({"kv-cache-dtype": "fp8"}, "flag names must be snake_case: 'kv-cache-dtype'"),
            ({"_comment": "x"}, "flag names must be snake_case: '_comment'"),
            ({"enforce_eager": False}, "enforce_eager=false is ambiguous; omit it or use the flag's negative form"),
            ({"goodput": [["ttft", 500]]}, "goodput: lists may contain scalar values only"),
        ):
            with self.subTest(flags=flags):
                for block, data in (
                    (ServingServerParams, {"model": "m", "tp": 8, "engine_args": flags}),
                    (ServingBenchmarkParams, {"num_prompts": 1, "client_args": flags}),
                ):
                    with self.assertRaises(ValueError) as caught:
                        block.model_validate(data)
                    self.assertIn(problem, str(caught.exception))


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
        self.assertEqual((config.timeouts.server_ready_s, config.timeouts.client_s), (3630, None))
        cells = config.cells()
        self.assertEqual((config.runs, list(cells)), ([OTHER_KEY, KEY], [KEY, OTHER_KEY]))
        self.assertEqual((cells[OTHER_KEY].isl, cells[OTHER_KEY].tp, cells[OTHER_KEY].concurrency), (8192, 8, 64))
        self.assertEqual(config.sweeps[OTHER_KEY], {"num_prompts": 128, "temperature": 0.0})
        self.assertEqual(
            [(type(task), task.id) for task in config.accuracy.lm_eval.tasks],
            [(AccuracyTask, "gsm8k_flex"), (AccuracyTask, "gsm8k_strict")],
        )
        self.assertEqual([(type(item), item.id) for item in config.accuracy.custom_eval], [(NiahEval, "niah_8k")])

    def test_a_loader_hook_a_mixin_defines_overrides_the_suite_config_default(self):
        self.assertEqual(LlmServingConfig.old_keys({}), [])
        for mixin in (Accuracy, SweepsAndRuns, Thresholds):
            with self.subTest(mixin=mixin.__name__):
                with mock.patch.object(mixin, "old_keys", classmethod(lambda cls, raw: ["probe"]), create=True):
                    self.assertEqual(LlmServingConfig.old_keys({}), ["probe"])

    def _refused(self, data):
        with self.assertRaises(ConfigError) as caught:
            _Engine.validate_config(data, self.config_file)
        return str(caught.exception)

    def test_container_env_is_the_only_env_block(self):
        data = _full()
        data["container"]["runtime"]["args"]["env"] = {"NCCL_DEBUG": "INFO"}
        self.assertIn(
            "container.runtime.args.env is not read; set these variables in container.env", self._refused(data)
        )

    def test_a_hugging_face_token_in_container_env_is_refused_without_its_value(self):
        for name in ("HF_TOKEN", "hugging_face_hub_token"):
            with self.subTest(name=name):
                data = _full()
                data["container"]["env"][name] = FAKE_CRED
                with self.assertRaises(ContainerEnvTokenError) as caught:
                    _Engine.validate_config(data, self.config_file)
                self.assertIn(name, str(caught.exception))
                self.assertNotIn(FAKE_CRED, str(caught.exception))

    def test_a_cell_override_may_not_set_a_harness_field_or_a_bad_flag(self):
        for override, problem in (
            ({"max_concurrency": 4}, "cannot set harness fields: ['max_concurrency']"),
            ({"result_dir": "/tmp/x"}, "cannot set harness fields: ['result_dir']"),
            ({"Temperature": 0}, "flag names must be snake_case: 'Temperature'"),
        ):
            with self.subTest(override=override):
                data = _full()
                data["sweeps"][KEY] = override
                self.assertIn(f"sweeps.{KEY}: {problem}", self._refused(data))

    def test_cells_and_threshold_cells_take_the_tp_and_pp_of_server_params(self):
        data = _full()
        data["server_params"]["pp"] = 1
        self.assertIn(f"cell {KEY} conflicts with server_params.tp or server_params.pp", self._refused(data))
        data = _full()
        other = "ISL=1024,OSL=1024,TP=4,PP=2,CONC=16"
        data["thresholds"] = {other: {"output_throughput": {"kind": "min", "value": 1}}}
        self.assertIn(
            f"threshold cell {other} conflicts with server_params.tp or server_params.pp", self._refused(data)
        )

    def test_enforced_thresholds_need_a_cell_for_every_run_but_not_every_cell_a_run(self):
        data = _full()
        data["enforce_thresholds"] = True
        spec = {"output_throughput": {"kind": "min", "value": 1}}
        data["thresholds"] = {KEY: spec}
        self.assertIn(f"runs missing threshold coverage: [{OTHER_KEY!r}]", self._refused(data))
        data["runs"] = [KEY]
        data["thresholds"] = {KEY: spec, OTHER_KEY: spec}
        self.assertEqual(_Engine.validate_config(data, self.config_file).runs, [KEY])


class TestSelectHosts(unittest.TestCase):
    def setUp(self):
        self.config = _Engine.model_validate(_full())

    def _with_nnodes(self, nnodes):
        return self.config.model_copy(
            update={"server_params": self.config.server_params.model_copy(update={"nnodes": nnodes})}
        )

    def test_single_takes_the_first_node_and_distributed_the_first_nnodes(self):
        cluster = _cluster("10.0.0.1", "10.0.0.2", "10.0.0.3")
        self.assertEqual(_Engine.select_hosts(self.config, cluster, "single"), ["10.0.0.1"])
        self.assertEqual(_Engine.select_hosts(self._with_nnodes(2), cluster, "distributed"), ["10.0.0.1", "10.0.0.2"])
        listed = {"node_dict": [{"mgmt_ip": "10.0.0.9"}, {"mgmt_ip": "10.0.0.8"}]}
        self.assertEqual(_Engine.select_hosts(self._with_nnodes(2), listed, "distributed"), ["10.0.0.9", "10.0.0.8"])

    def test_distributed_needs_two_or_more_nodes_and_a_cluster_that_holds_them(self):
        cluster = _cluster("10.0.0.1", "10.0.0.2")
        for nnodes in (None, 1):
            with self.subTest(nnodes=nnodes):
                with self.assertRaisesRegex(ValueError, "distributed runs need server_params.nnodes of 2 or more"):
                    _Engine.select_hosts(self._with_nnodes(nnodes), cluster, "distributed")
        with self.assertRaisesRegex(ValueError, r"the cluster file has 2 node\(s\); distributed needs 3"):
            _Engine.select_hosts(self._with_nnodes(3), cluster, "distributed")
        with self.assertRaisesRegex(ValueError, r"the cluster file has 0 node\(s\); single needs 1"):
            _Engine.select_hosts(self.config, {"node_dict": {}}, "single")

    def test_any_other_mode_keeps_every_node(self):
        self.assertIsNone(_Engine.select_hosts(self.config, _cluster("10.0.0.1", "10.0.0.2"), "disagg_distributed"))


if __name__ == "__main__":
    unittest.main()
