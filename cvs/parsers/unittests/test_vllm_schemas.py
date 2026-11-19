'''Unit tests for the vLLM config schema and its old-key resolution in cvs.parsers.schemas.'''

import copy
import json
import tempfile
import unittest
import warnings
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.conf_lib.loader import ConfigError, load_config
from cvs.parsers.schemas import (
    ContainerEnvTokenError,
    RunCell,
    VllmConfig,
    resolve_old_vllm_keys,
    serialize_cli_options,
)
from cvs.unittests.support import FAKE_CRED, write_filled_config


CELL = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"
FAKE_CRED_FILE = "/home/test/.hf_token"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "input" / "config_file" / "inference" / "vllm"
DISTRIBUTED_TEMPLATE = CONFIG_DIR / "mi3xx_vllm_llama33-70b_fp8_distributed.json"
MERGE_CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"


def _config(**overrides):
    value = {
        "enforce_thresholds": False,
        "threshold_json": "threshold.json",
        "paths": {
            "shared_fs": "/home/test",
            "models_dir": "/models",
            "log_dir": "/logs",
            "hf_token_file": FAKE_CRED_FILE,
        },
        "container": {"name": "vllm", "image": "image", "runtime": {"name": "docker", "args": {}}},
        "server_params": {"model": "/models/DeepSeek-R1-0528-FP8", "tp": 8, "pp": 2},
        "benchmark_params": {"num_prompts": 25},
        "sweeps": {CELL: {"num_prompts": 50}},
        "runs": [CELL],
        "thresholds": {},
    }
    value.update(overrides)
    return value


def _old_config(server=None, benchmark=None, **overrides):
    config = _config(**overrides)
    config["server_params"] = {
        "model": "/models/x",
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 2,
        **(server or {}),
    }
    config["benchmark_params"] = {"num_prompts": 25, **(benchmark or {})}
    return config


def _with_args(args):
    config = _config()
    config["container"]["runtime"]["args"] = args
    return config


def _validate(config):
    """Validate and return the model with the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        variant = VllmConfig(**config)
    return variant, [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]


def _cells(variant):
    return {key: variant.thresholds.get(key) for key in variant.thresholds.cells()}


class TestRunCell(unittest.TestCase):
    def test_parses_only_canonical_cell_keys(self):
        cell = RunCell.parse(CELL)
        self.assertEqual((cell.isl, cell.osl, cell.tp, cell.pp, cell.concurrency), (1024, 1024, 8, 2, 16))
        for invalid in (
            "ISL=1024,OSL=1024,TP=8,CONC=16",
            "OSL=1024,ISL=1024,TP=8,PP=2,CONC=16",
            "ISL=1024, OSL=1024,TP=8,PP=2,CONC=16",
            "ISL=0,OSL=1024,TP=8,PP=2,CONC=16",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    RunCell.parse(invalid)


class TestRuns(unittest.TestCase):
    def test_resolves_selected_run_with_cell_overrides(self):
        config = _config()
        config["benchmark_params"]["client_args"] = {"temperature": 0.0}
        (run,) = VllmConfig(**config).resolved_runs()
        self.assertEqual(run.cell.key, CELL)
        self.assertEqual(run.benchmark_params["num_prompts"], 50)
        self.assertEqual(run.benchmark_params["temperature"], 0.0)
        self.assertEqual(run.benchmark_params["random_input_len"], 1024)
        self.assertEqual(run.benchmark_params["random_output_len"], 1024)
        self.assertEqual(run.benchmark_params["max_concurrency"], 16)
        self.assertNotIn("client_args", run.benchmark_params)

    def test_rejects_unknown_root_and_legacy_shape(self):
        for kwargs in ({"unexpected": 1}, {"schema_version": 1}, {"params": {}}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValidationError):
                    VllmConfig(**_config(**kwargs))

    def test_vllm_block_defaults_to_mp_and_takes_ray(self):
        self.assertEqual(VllmConfig(**_config()).vllm.distributed_executor_backend, "mp")
        ray = VllmConfig(**_config(vllm={"distributed_executor_backend": "ray"}))
        self.assertEqual(ray.vllm.distributed_executor_backend, "ray")
        with self.assertRaises(ValidationError):
            VllmConfig(**_config(vllm={"distributed_executor_backend": "nccl"}))


class TestNotImplemented(unittest.TestCase):
    def test_each_refused_field_fails_naming_vllm_and_the_field(self):
        cases = {
            "server_params.dp": ("server_params", {"dp": 2}),
            "server_params.ep": ("server_params", {"ep": 2}),
            "server_params.prefill_nnodes": ("server_params", {"prefill_nnodes": 1}),
            "server_params.long_context": ("server_params", {"long_context": {"chunked_prefill_size": 8192}}),
            "benchmark_params.scaling_baseline_output_throughput": (
                "benchmark_params",
                {"scaling_baseline_output_throughput": 100.0},
            ),
            "accuracy.custom_eval": ("accuracy", {"custom_eval": [{"type": "niah", "id": "n", "isl": 8192}]}),
            "accuracy.quant_parity": ("accuracy", {"quant_parity": {"enabled": True}}),
            "accuracy.mtp_quality": ("accuracy", {"mtp_quality": {"enabled": True}}),
        }
        for field, (block, values) in cases.items():
            with self.subTest(field=field):
                config = _config()
                config[block] = {**config.get(block, {}), **values}
                with self.assertRaisesRegex(ValidationError, f"{field} is not implemented for VllmConfig"):
                    VllmConfig(**config)

    def test_refused_fields_at_their_defaults_pass(self):
        config = _config()
        config["server_params"].update({"dp": 1, "ep": 1, "prefill_nnodes": None})
        VllmConfig(**config)


class TestContainer(unittest.TestCase):
    def test_accepts_complete_container_network_environment(self):
        config = _config()
        config["container"]["env"] = {
            "NCCL_IB_HCA": "rdma0",
            "NCCL_SOCKET_IFNAME": "eno0",
            "GLOO_SOCKET_IFNAME": "eno0",
            "TP_SOCKET_IFNAME": "eno0",
        }
        variant = VllmConfig(**config)
        self.assertEqual(variant.container.nccl_ib_hcas, ["rdma0"])
        self.assertTrue(variant.container.socket_env_configured)

    def test_rejects_incomplete_container_socket_environment(self):
        names = ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME")
        for missing in names:
            with self.subTest(missing=missing):
                config = _config()
                config["container"]["env"] = {name: "eno0" for name in names if name != missing}
                with self.assertRaisesRegex(ValidationError, "must set all socket interface"):
                    VllmConfig(**config)

    def test_rejects_empty_container_hca(self):
        config = _config()
        config["container"]["env"] = {"NCCL_IB_HCA": " , "}
        with self.assertRaisesRegex(ValidationError, "must name at least one device"):
            VllmConfig(**config)

    def test_rejects_hf_token_in_container_env_without_echoing_value(self):
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token"):
            with self.subTest(name=name):
                config = _config()
                config["container"]["env"] = {"NCCL_SOCKET_IFNAME": "eno0", name: FAKE_CRED}
                with self.assertRaises(ContainerEnvTokenError) as ctx:
                    VllmConfig(**config)
                self.assertIn("paths.hf_token_file", str(ctx.exception))
                self.assertNotIn(FAKE_CRED, repr(ctx.exception.args))

    def test_runtime_args_accept_docker_keys_but_not_env(self):
        variant = VllmConfig(**_with_args({"shm_size": "64g", "cap_add": ["SYS_PTRACE"], "ulimit": ["memlock=-1"]}))
        self.assertEqual(variant.container.model_dump()["runtime"]["args"]["shm_size"], "64g")
        for args in ({"env": {"NCCL_DEBUG": "INFO"}}, {"unknown_flag": True}):
            with self.subTest(args=args):
                with self.assertRaises(ValidationError):
                    VllmConfig(**_with_args(args))


class TestFlags(unittest.TestCase):
    def test_comment_keys_are_ignored_but_flag_payloads_keep_theirs(self):
        config = _config(
            _comment="root",
            sweeps={"_comment": "cells", CELL: {"_comment": "cell", "json_arg": {"_comment": "upstream payload"}}},
            accuracy={"lm_eval": {"tasks": [{"_comment": "task", "id": "gsm8k", "task": "gsm8k"}]}},
        )
        config["server_params"]["engine_args"] = {"json_arg": {"_comment": "upstream payload"}}
        config["container"]["runtime"]["args"] = {"_comment_mounts": "notes", "volumes": ["/data:/data"]}
        variant = VllmConfig(**config)
        self.assertEqual(variant.server_params.engine_args, {"json_arg": {"_comment": "upstream payload"}})
        self.assertEqual(variant.sweeps, {CELL: {"json_arg": {"_comment": "upstream payload"}}})
        self.assertEqual(variant.accuracy.lm_eval.tasks[0].id, "gsm8k")

    def test_serializes_documented_value_shapes(self):
        options = {
            "trust_remote_code": True,
            "max_num_seqs": 16,
            "served_model_name": ["first", "second"],
            "structured_outputs": {"json": {"type": "object"}},
            "optional": None,
        }
        self.assertEqual(
            serialize_cli_options(options),
            [
                "--trust-remote-code",
                "--max-num-seqs",
                "16",
                "--served-model-name",
                "first",
                "second",
                "--structured-outputs",
                '{"json":{"type":"object"}}',
            ],
        )
        with self.assertRaises(ValueError):
            serialize_cli_options({"enforce_eager": False})

    def test_rejects_vllm_harness_flags_and_nested_lists(self):
        cases = (
            ("server_params", "engine_args", {"tensor_parallel_size": 8}),
            ("server_params", "engine_args", {"max_model_len": 4096}),
            ("server_params", "engine_args", {"distributed_executor_backend": "ray"}),
            ("server_params", "engine_args", {"headless": True}),
            ("benchmark_params", "client_args", {"dataset_name": "sharegpt"}),
            ("benchmark_params", "client_args", {"lora_modules": [["a"]]}),
        )
        for block, key, flags in cases:
            with self.subTest(flags=flags):
                config = _config()
                config[block][key] = flags
                with self.assertRaisesRegex(ValidationError, f"{key}"):
                    VllmConfig(**config)
        with self.assertRaisesRegex(ValidationError, "cannot set harness fields"):
            VllmConfig(**_config(sweeps={CELL: {"dataset_name": "sharegpt"}}))


class TestThresholds(unittest.TestCase):
    def _variant(self, thresholds, enforce=False):
        return VllmConfig(**_config(enforce_thresholds=enforce, thresholds=thresholds))

    def test_partial_and_empty_cells_load_with_enforcement_off_and_on(self):
        for thresholds in ({CELL: {"output_throughput": {"kind": "min", "value": 0}}}, {CELL: {}}):
            for enforce in (False, True):
                with self.subTest(thresholds=thresholds, enforce=enforce):
                    self.assertEqual(_cells(self._variant(thresholds, enforce=enforce)), thresholds)

    def test_enforcement_still_requires_selected_run_cell(self):
        self._variant({}, enforce=False)
        with self.assertRaisesRegex(ValidationError, "runs missing from the threshold file"):
            self._variant({}, enforce=True)

    def test_comment_metrics_are_ignored_and_other_underscore_keys_rejected(self):
        spec = {"kind": "min", "value": 0}
        variant = self._variant({CELL: {"_comment": "calibrate", "_example_floor": spec, "output_throughput": spec}})
        self.assertEqual(_cells(variant), {CELL: {"output_throughput": spec}})
        for cell in (
            {"_output_throughput": spec},
            {"_": spec},
            {"output_throughput": {**spec, "_note": "calibrated"}},
        ):
            with self.subTest(cell=cell):
                with self.assertRaisesRegex(ValidationError, "unknown (vLLM threshold metric|threshold fields)"):
                    self._variant({CELL: cell})

    def test_info_records_any_registered_metric(self):
        for metric in ("output_throughput", "mean_ttft_ms"):
            with self.subTest(metric=metric):
                cell = {metric: {"kind": "info", "value": 1}}
                self.assertEqual(_cells(self._variant({CELL: cell})), {CELL: cell})

    def test_rejects_specs_outside_the_vllm_direction_rule(self):
        for metric, spec in (
            ("output_throughput", {"kind": "max", "value": 1}),
            ("mean_ttft_ms", {"kind": "min", "value": 1}),
            ("output_throughput", {"kind": "within", "value": 1, "tolerance_pct": 5}),
            ("output_throughput", {"kind": "min", "value": 1, "optional": False}),
            ("output_throughput", {"kind": "min", "value": 1, "unit": "tok/s"}),
            ("output_throughput", {"kind": "min", "value": "1"}),
            ("client.output_throughput", {"kind": "min", "value": 1}),
            ("scaling.efficiency_pct", {"kind": "min", "value": 1}),
        ):
            with self.subTest(metric=metric, spec=spec):
                with self.assertRaises(ValidationError):
                    self._variant({CELL: {metric: spec}})

    def test_accuracy_thresholds_are_keyed_by_lm_eval_task(self):
        accuracy = {"lm_eval": {"gsm8k": {"gsm8k.exact_match__strict-match": {"kind": "min", "value": 0}}}}
        cell = {"output_throughput": {"kind": "min", "value": 0}}
        variant = self._variant({CELL: cell, "accuracy": accuracy}, enforce=True)
        self.assertEqual(list(variant.thresholds.accuracy.lm_eval), ["gsm8k"])
        with self.assertRaisesRegex(ValidationError, "cell key"):
            self._variant({CELL: cell, "accuracy": accuracy, "acuracy": {}}, enforce=True)
        with self.assertRaisesRegex(ValidationError, "accuracy.custom_eval is not implemented"):
            self._variant({CELL: cell, "accuracy": {"custom_eval": {"n": cell}}})

    def test_threshold_cells_need_a_matching_sweep(self):
        with self.assertRaisesRegex(ValidationError, "no matching sweep"):
            self._variant({CELL.replace("CONC=16", "CONC=32"): {}})


class TestOldKeys(unittest.TestCase):
    def test_renamed_keys_and_native_flags_move_with_one_warning(self):
        variant, messages = _validate(_old_config(server={"dist_init_port": 28947, "kv_cache_dtype": "fp8"}))
        self.assertEqual((variant.server_params.tp, variant.server_params.pp), (8, 2))
        self.assertEqual(variant.server_params.master_port, 28947)
        self.assertEqual(variant.server_params.engine_args, {"kv_cache_dtype": "fp8"})
        self.assertEqual(len(messages), 1)
        for name in (
            "server_params.tensor_parallel_size -> server_params.tp",
            "server_params.dist_init_port -> server_params.master_port",
            "server_params.kv_cache_dtype -> server_params.engine_args.kv_cache_dtype",
        ):
            self.assertIn(name, messages[0])

    def test_poll_and_warmup_keys_compute_the_timeouts(self):
        old = _old_config(
            server={"server_warmup_wait_s": "30", "server_poll_iterations": 360, "server_poll_wait_s": 10},
            benchmark={"client_initial_wait_s": 10, "client_poll_iterations": 540},
        )
        variant, _ = _validate(old)
        self.assertEqual(variant.timeouts.server_ready_s, 30 + 360 * 10)
        self.assertEqual(variant.timeouts.client_s, 10 + 540 * 60)
        variant, _ = _validate(_old_config())
        self.assertEqual((variant.timeouts.server_ready_s, variant.timeouts.client_s), (330 + 60 * 60, 120 + 20 * 60))

    def test_a_timeout_key_that_is_not_a_number_fails_without_its_value(self):
        with self.assertRaises(ValidationError) as ctx:
            VllmConfig(**_old_config(server={"server_poll_wait_s": "soon"}))
        self.assertIn("server_params.server_poll_wait_s must be a whole number", str(ctx.exception))
        self.assertNotIn("soon", str(ctx.exception))

    def test_other_client_keys_move_to_client_args_and_old_blocks_move(self):
        old = _old_config(
            server={"distributed_executor_backend": "ray", "max_model_len": 4096},
            benchmark={"temperature": 0.0},
            accuracy={"tasks": [{"id": "gsm8k", "task": "gsm8k"}]},
        )
        variant, _ = _validate(old)
        self.assertEqual(variant.benchmark_params.client_args, {"temperature": 0.0})
        self.assertEqual(variant.vllm.distributed_executor_backend, "ray")
        self.assertEqual(variant.server_params.max_model_length, 4096)
        self.assertEqual(variant.accuracy.lm_eval.tasks[0].id, "gsm8k")

    def test_fixed_client_keys_drop_at_the_value_the_harness_always_sets(self):
        old = _old_config(
            server={"backend": "vllm", "trust_remote_code": True},
            benchmark={"backend": "vllm", "dataset_name": "random", "ignore_eos": True, "trust_remote_code": True},
        )
        variant, (message,) = _validate(old)
        self.assertEqual(variant.server_params.engine_args, {"trust_remote_code": True})
        self.assertEqual(variant.benchmark_params.client_args, {})
        self.assertIn("benchmark_params.ignore_eos (dropped)", message)

    def test_a_fixed_client_key_at_another_value_fails_naming_it(self):
        for benchmark in (
            {"dataset_name": "sharegpt"},
            {"ignore_eos": False},
            {"backend": "openai"},
        ):
            with self.subTest(benchmark=benchmark):
                with self.assertRaisesRegex(ValidationError, f"benchmark_params.{next(iter(benchmark))}"):
                    VllmConfig(**_old_config(benchmark=benchmark))

    def test_client_trust_follows_the_server_unless_only_the_client_trusts(self):
        variant, _ = _validate(_old_config(benchmark={"trust_remote_code": True}))
        self.assertEqual(variant.benchmark_params.client_args, {"trust_remote_code": True})
        with self.assertRaisesRegex(ValidationError, "benchmark_params.trust_remote_code"):
            VllmConfig(**_old_config(server={"trust_remote_code": True}, benchmark={"trust_remote_code": False}))

    def test_old_cell_overrides_drop_fixed_keys_and_refuse_per_cell_waits(self):
        old = _old_config(server={"trust_remote_code": True})
        old["sweeps"] = {CELL: {"ignore_eos": True, "dataset_name": "random", "trust_remote_code": True, "seed": 3}}
        variant, _ = _validate(old)
        self.assertEqual(variant.sweeps, {CELL: {"seed": 3}})
        client_only = _old_config()
        client_only["sweeps"] = {CELL: {"trust_remote_code": True}}
        variant, _ = _validate(client_only)
        self.assertEqual(variant.sweeps, {CELL: {"trust_remote_code": True}})
        for override in ({"ignore_eos": False}, {"trust_remote_code": False}, {"client_poll_iterations": 5}):
            with self.subTest(override=override):
                old = _old_config(server={"trust_remote_code": True})
                old["sweeps"] = {CELL: override}
                with self.assertRaisesRegex(ValidationError, f"sweeps.{CELL}.{next(iter(override))}"):
                    VllmConfig(**old)

    def test_old_max_model_len_moves_only_when_it_is_a_length_or_null(self):
        for value, expected in ((4096, 4096), ("2056", 2056), (None, None)):
            with self.subTest(value=value):
                variant, _ = _validate(_old_config(server={"max_model_len": value}))
                self.assertEqual(variant.server_params.max_model_length, expected)
                self.assertIn("max_model_length", variant.server_params.model_fields_set)
        for value in ("auto", -1, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationError, "server_params.max_model_len"):
                    VllmConfig(**_old_config(server={"max_model_len": value}))

    def test_new_keys_in_an_old_config_stay_where_they_are(self):
        variant, _ = _validate(_old_config(server={"engine_args": {"kv_cache_dtype": "fp8"}, "max_model_length": 4096}))
        self.assertEqual(variant.server_params.engine_args, {"kv_cache_dtype": "fp8"})
        self.assertEqual(variant.server_params.max_model_length, 4096)

    def test_missing_num_prompts_takes_the_old_default(self):
        old = _old_config()
        del old["benchmark_params"]
        variant, _ = _validate(old)
        self.assertEqual(variant.benchmark_params.num_prompts, 3200)

    def test_old_network_keys_move_into_container_env_unless_it_sets_them(self):
        variant, _ = _validate(_old_config(ib_hca_devices=["rdma0", "rdma1"], ib_netdev="eth0"))
        self.assertEqual(
            variant.container.env,
            {
                "NCCL_IB_HCA": "rdma0,rdma1",
                "NCCL_SOCKET_IFNAME": "eth0",
                "GLOO_SOCKET_IFNAME": "eth0",
                "TP_SOCKET_IFNAME": "eth0",
            },
        )
        sockets = {"NCCL_SOCKET_IFNAME": "eno0", "GLOO_SOCKET_IFNAME": "eno0", "TP_SOCKET_IFNAME": "eno0"}
        old = _old_config(ib_hca_devices="auto", ib_netdev="eth0")
        old["container"]["env"] = dict(sockets)
        variant, (message,) = _validate(old)
        self.assertEqual(variant.container.env, sockets)
        self.assertIn("ib_netdev (dropped)", message)
        old = _old_config(ib_hca_devices=["rdma9"])
        old["container"]["env"] = {"NCCL_IB_HCA": "rdma0"}
        variant, (message,) = _validate(old)
        self.assertEqual(variant.container.env, {"NCCL_IB_HCA": "rdma0"})
        self.assertIn("ib_hca_devices (dropped)", message)

    def test_old_threshold_shapes_are_rewritten(self):
        thresholds = {
            CELL: {
                "mean_ttft_ms": {"kind": "max_ms", "value": 1},
                "output_throughput": {"kind": "min_tok_s", "value": 2},
            },
            "accuracy": {"gsm8k": {"gsm8k.exact_match__strict-match": {"kind": "min", "value": 0.5}}},
        }
        variant, (message,) = _validate(_config(thresholds=thresholds))
        self.assertEqual(
            variant.thresholds.get(CELL),
            {"mean_ttft_ms": {"kind": "max", "value": 1}, "output_throughput": {"kind": "min", "value": 2}},
        )
        self.assertEqual(list(variant.thresholds.accuracy.lm_eval), ["gsm8k"])
        self.assertIn("thresholds.accuracy.gsm8k -> thresholds.accuracy.lm_eval.gsm8k", message)

    def test_old_accuracy_kinds_are_rewritten_or_refused(self):
        spec = {"kind": "min_tok_s", "value": 1}
        variant, _ = _validate(_config(thresholds={"accuracy": {"gsm8k": {"m": spec}}}))
        self.assertEqual(variant.thresholds.accuracy.lm_eval["gsm8k"]["m"].kind, "min")
        within = {"kind": "within", "value": 1, "tolerance_pct": 5}
        with self.assertRaisesRegex(ValidationError, "thresholds.accuracy.gsm8k.m.kind"):
            VllmConfig(**_config(thresholds={"accuracy": {"gsm8k": {"m": within}}}))

    def test_a_current_accuracy_section_is_not_mistaken_for_a_task_id(self):
        thresholds = {"accuracy": {"lm_eval": {}, "lm-eval": {}}}
        with self.assertRaisesRegex(ValidationError, "lm-eval"):
            VllmConfig(**_config(thresholds=thresholds))

    def test_a_current_config_loads_without_a_warning_and_its_unknown_keys_still_fail(self):
        config = _config()
        self.assertEqual(resolve_old_vllm_keys(config), (config, None))
        _, messages = _validate(config)
        self.assertEqual(messages, [])
        config["server_params"]["kv_cache_dtype"] = "fp8"
        with self.assertRaisesRegex(ValidationError, "kv_cache_dtype"):
            VllmConfig(**config)


def _templates():
    return sorted(path for path in CONFIG_DIR.glob("*.json") if not path.name.endswith("threshold.json"))


class TestShippedTemplates(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_every_template_loads_once_its_placeholders_are_filled_without_a_warning(self):
        templates = _templates()
        self.assertTrue(templates)
        for template in templates:
            with self.subTest(config=template.name):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", FutureWarning)
                    load_config(VllmConfig, write_filled_config(template, self.tmp), {"username": "test"})

    def test_unfilled_distributed_template_names_the_placeholders_it_checks(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config(VllmConfig, DISTRIBUTED_TEMPLATE, {"username": "test"})
        self.assertEqual(
            sorted(problem.split(":", 1)[0] for problem in ctx.exception.problems),
            [
                "container.env.GLOO_SOCKET_IFNAME",
                "container.env.NCCL_IB_GID_INDEX",
                "container.env.NCCL_SOCKET_IFNAME",
                "container.env.TP_SOCKET_IFNAME",
                "container.image",
                "container.runtime.args.volumes.1",
                "container.runtime.args.volumes.3",
                "server_params.model",
            ],
        )

    def test_placeholder_in_a_flag_value_fails_the_load(self):
        filled = write_filled_config(DISTRIBUTED_TEMPLATE, self.tmp)
        config = json.loads(filled.read_text())
        config["server_params"]["engine_args"]["served_model_name"] = "<changeme-name>"
        (cell,) = config["runs"]
        config["sweeps"][cell]["tokenizer"] = "<changeme>"
        filled.write_text(json.dumps(config))
        with self.assertRaises(ConfigError) as ctx:
            load_config(VllmConfig, filled, {"username": "test"})
        self.assertEqual(
            sorted(problem.split(":", 1)[0] for problem in ctx.exception.problems),
            ["server_params.engine_args.served_model_name", f"sweeps.{cell}.tokenizer"],
        )


class TestThresholdFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _write(self, config):
        config = copy.deepcopy(config)
        config.pop("thresholds", None)
        path = self.tmp / "config.json"
        path.write_text(json.dumps(config))
        return path

    def test_template_loads_its_named_threshold_file(self):
        template = CONFIG_DIR / "mi3xx_vllm_llama33-70b_fp8_single.json"
        variant = load_config(VllmConfig, write_filled_config(template, self.tmp), {"username": "test"}).config
        self.assertEqual(sorted(variant.thresholds.cells()), sorted(variant.runs))

    def test_named_threshold_file_that_is_missing_fails_the_load(self):
        path = self._write(_config(threshold_json="no_such_threshold_xyz.json"))
        with self.assertRaises(FileNotFoundError):
            load_config(VllmConfig, path, {"username": "test"})

    def test_unnamed_threshold_file_runs_record_only_even_when_enforced(self):
        (self.tmp / "sibling_threshold.json").write_text(json.dumps({CELL: {}}))
        config = _config(enforce_thresholds=True)
        del config["threshold_json"]
        with self.assertLogs(level="WARNING"):
            variant = load_config(VllmConfig, self._write(config), {"username": "test"}).config
        self.assertEqual((variant.thresholds.cells(), variant.enforce_thresholds), ({}, False))

    def test_enforced_threshold_file_must_cover_every_selected_run(self):
        second = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=32"
        (self.tmp / "threshold.json").write_text(json.dumps({CELL: {}}))
        path = self._write(_config(enforce_thresholds=True, sweeps={CELL: {}, second: {}}, runs=[CELL, second]))
        with self.assertRaises(ConfigError) as ctx:
            load_config(VllmConfig, path, {"username": "test"})
        self.assertIn(f"runs missing from the threshold file: ['{second}']", str(ctx.exception))

    def test_an_old_config_logs_its_message_once_and_a_current_one_logs_nothing(self):
        (self.tmp / "threshold.json").write_text(json.dumps({CELL: {}, "accuracy": {"gsm8k": {}}}))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertLogs("cvs.parsers.schemas", level="WARNING") as logs:
                load_config(VllmConfig, self._write(_old_config()), {"username": "test"})
        (future,) = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
        self.assertEqual([record.getMessage() for record in logs.records], [future])
        (self.tmp / "threshold.json").write_text(json.dumps({CELL: {}}))
        with self.assertNoLogs("cvs.parsers.schemas", level="WARNING"):
            load_config(VllmConfig, self._write(_config()), {"username": "test"})

    def test_an_old_threshold_file_loads_through_the_old_keys(self):
        (self.tmp / "threshold.json").write_text(
            json.dumps({"_comment": "x", CELL: {}, "accuracy": {"gsm8k": {"m": {"kind": "min", "value": 0.5}}}})
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            variant = load_config(VllmConfig, self._write(_config()), {"username": "test"}).config
        self.assertEqual(list(variant.thresholds.accuracy.lm_eval), ["gsm8k"])
        self.assertEqual(len([w for w in caught if issubclass(w.category, FutureWarning)]), 1)


class TestContainerForOrchestrator(unittest.TestCase):
    def test_cluster_env_volumes_and_shm_size_merge_without_aliasing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "threshold.json").write_text("{}")
            config = {
                "enforce_thresholds": False,
                "threshold_json": str(tmp / "threshold.json"),
                "paths": {"shared_fs": "/home/x", "models_dir": "/models", "log_dir": "/logs", "hf_token_file": "/t"},
                "container": {
                    "name": "vllm",
                    "image": "rocm/vllm:1",
                    "env": {"B": "config"},
                    "runtime": {"name": "docker", "args": {"volumes": ["/config:/config"]}},
                },
                "server_params": {"model": "/models/m", "tp": 8},
                "benchmark_params": {"num_prompts": 1},
                "sweeps": {MERGE_CELL: {}},
                "runs": [MERGE_CELL],
            }
            (tmp / "config.json").write_text(json.dumps(config))
            cluster = {
                "container": {
                    "name": "from-cluster",
                    "env": {"A": "cluster", "B": "cluster"},
                    "runtime": {
                        "name": "docker",
                        "args": {
                            "volumes": ["/cluster:/cluster"],
                            "shm_size": "64g",
                            "registry": {"username": "user", "password_file": "/home/x/registry"},
                        },
                    },
                }
            }
            loaded = load_config(VllmConfig, tmp / "config.json", cluster)
        self.assertEqual(loaded.container["name"], "vllm")
        self.assertEqual(loaded.container["env"], {"A": "cluster", "B": "config"})
        self.assertEqual(loaded.container["runtime"]["args"]["volumes"], ["/config:/config"])
        self.assertEqual(loaded.container["runtime"]["args"]["shm_size"], "64g")
        loaded.container["env"]["A"] = "mutated"
        loaded.container["runtime"]["args"]["volumes"].append("/extra")
        loaded.container["runtime"]["args"]["registry"]["username"] = "mutated"
        cluster_container = loaded.cluster["container"]
        self.assertEqual(cluster_container["env"]["A"], "cluster")
        self.assertEqual(cluster_container["runtime"]["args"]["volumes"], ["/cluster:/cluster"])
        self.assertEqual(cluster_container["runtime"]["args"]["registry"]["username"], "user")


if __name__ == "__main__":
    unittest.main()
