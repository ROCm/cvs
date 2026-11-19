import unittest
import warnings

from pydantic import ValidationError

from cvs.parsers.schemas import (
    OLD_KEYS_QUIET,
    SglangConfig,
    resolve_old_sglang_keys,
    resolve_old_sglang_threshold_names,
)
from cvs.unittests.support import FAKE_CRED

CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=64"


def _config(**overrides):
    config = {
        "paths": {"shared_fs": "/fs", "models_dir": "/models", "log_dir": "/logs", "hf_token_file": "/fs/token"},
        "container": {"name": "sglang", "image": "image", "runtime": {"name": "docker"}},
        "server_params": {"model": "m", "tp": 8},
        "benchmark_params": {"num_prompts": 4},
        "sweeps": {CELL: {}},
        "runs": [CELL],
    }
    config.update(overrides)
    return config


def _old_config(server=None, **overrides):
    config = {
        "paths": {"shared_fs": "/fs", "models_dir": "/models", "log_dir": "/logs", "hf_token_file": "/fs/token"},
        "container": {"name": "sglang", "image": "image", "runtime": {"name": "docker", "args": {}}},
        "server_params": {"model": "m", "tensor_parallelism": "8", "pipeline_parallelism": "1", **(server or {})},
        "benchmark_params": {"num_prompts": "4"},
        "sweeps": {CELL: {}},
        "sweep": {"runs": [{"combo": CELL}]},
        "thresholds": {},
    }
    config.update(overrides)
    return config


class TestSglangConfig(unittest.TestCase):
    def test_new_config_validates_with_shared_defaults(self):
        config = SglangConfig.model_validate(_config())
        self.assertEqual(config.server_params.port, 8888)
        self.assertIsNone(config.timeouts.client_s)

    def test_each_refused_field_fails_naming_sglang_and_the_field(self):
        refused = {
            "server_params.dp": 2,
            "server_params.ep": 2,
            "server_params.prefill_nnodes": 1,
            "benchmark_params.seed": 3,
            "benchmark_params.request_rate": 4.0,
            "benchmark_params.burstiness": 2.0,
            "benchmark_params.random_prefix_len": 16,
            "benchmark_params.tokenizer_mode": "slow",
            "benchmark_params.scaling_baseline_output_throughput": 100.0,
            "benchmark_params.client_args": {"warmup_requests": 2},
            "timeouts.server_ready_s": 60,
            "accuracy.quant_parity": {"enabled": True},
            "accuracy.mtp_quality": {"enabled": True},
        }
        for path, value in refused.items():
            with self.subTest(field=path):
                config = _config()
                block, name = path.split(".")
                config.setdefault(block, {})[name] = value
                with self.assertRaisesRegex(ValidationError, rf"{path} is not implemented for SglangConfig"):
                    SglangConfig.model_validate(config)

    def test_an_lm_eval_option_sglang_does_not_pass_fails(self):
        task = {"id": "lm_eval_gsm8k", "tasks": "gsm8k", "device": "cuda"}
        with self.assertRaisesRegex(ValidationError, r"tasks\[0\]\.device is not implemented for SglangConfig"):
            SglangConfig.model_validate(_config(accuracy={"lm_eval": {"tasks": [task]}}))

    def test_lm_eval_options_sglang_passes_are_accepted(self):
        task = {"id": "lm_eval_gsm8k", "tasks": "gsm8k", "num_fewshot": 5, "batch_size": "auto", "limit": 100}
        config = SglangConfig.model_validate(_config(accuracy={"lm_eval": {"tasks": [task]}}))
        self.assertEqual(config.accuracy.lm_eval.tasks[0].limit, 100)

    def test_a_niah_eval_needs_the_long_context_server_flags(self):
        niah = {"type": "niah", "id": "niah_8k", "isl": 8192}
        with self.assertRaisesRegex(ValidationError, "max_model_length"):
            SglangConfig.model_validate(_config(accuracy={"custom_eval": [niah]}))
        server = {"model": "m", "tp": 8, "max_model_length": 16384, "long_context": {"chunked_prefill_size": 8192}}
        config = SglangConfig.model_validate(_config(server_params=server, accuracy={"custom_eval": [niah]}))
        self.assertEqual(config.accuracy.custom_eval[0].isl, 8192)


class TestSglangConfigOldKeys(unittest.TestCase):
    def test_validating_an_old_config_warns_once_naming_the_file(self):
        with warnings.catch_warnings(record=True) as caught, self.assertLogs("cvs.parsers.schemas", "WARNING") as logs:
            warnings.simplefilter("always")
            config = SglangConfig.model_validate(_old_config(), context={"config_path": "old.json"})
        self.assertEqual(config.server_params.tp, 8)
        self.assertEqual(len([w for w in caught if issubclass(w.category, FutureWarning)]), 1)
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(logs.records[0].getMessage().startswith("old.json: Deprecated SGLang config keys"))

    def test_a_quiet_load_converts_without_warning(self):
        with warnings.catch_warnings(record=True) as caught, self.assertNoLogs("cvs.parsers.schemas"):
            warnings.simplefilter("always")
            config = SglangConfig.model_validate(_old_config(), context={"old_keys": OLD_KEYS_QUIET})
        self.assertEqual(config.server_params.tp, 8)
        self.assertEqual([w for w in caught if issubclass(w.category, FutureWarning)], [])

    def test_an_underscore_metric_in_a_threshold_cell_fails_instead_of_being_dropped(self):
        thresholds = {CELL: {"_output_throughput": {"kind": "min", "value": 1}, "_comment": "kept"}}
        with self.assertRaisesRegex(ValidationError, "unknown SGLang threshold metric '_output_throughput'"):
            SglangConfig.model_validate(_config(thresholds=thresholds))

    def test_a_new_config_neither_warns_nor_logs(self):
        with warnings.catch_warnings(record=True) as caught, self.assertNoLogs("cvs.parsers.schemas"):
            warnings.simplefilter("always")
            SglangConfig.model_validate(_config())
        self.assertEqual(caught, [])


class TestResolveOldSglangKeys(unittest.TestCase):
    def test_a_config_without_old_keys_comes_back_unchanged(self):
        config = _config()
        converted, message = resolve_old_sglang_keys(config)
        self.assertIs(converted, config)
        self.assertIsNone(message)

    def test_server_keys_move_to_shared_names_and_engine_args(self):
        old = _old_config(
            {"memory_fraction": "0.85", "log_level": "info", "add_flags": ["--attention-backend aiter", "--disable-x"]}
        )
        converted, message = resolve_old_sglang_keys(old)
        server = converted["server_params"]
        self.assertEqual((server["tp"], server["pp"]), ("8", "1"))
        self.assertEqual(
            server["engine_args"],
            {"attention_backend": "aiter", "disable_x": True, "log_level": "info", "mem_fraction_static": 0.85},
        )
        self.assertEqual((server["port"], server["master_port"]), (8000, 40001))
        self.assertIn("server_params.tensor_parallelism -> server_params.tp", message)
        self.assertIn("server_params.add_flags -> server_params.engine_args", message)
        SglangConfig.model_validate(converted)

    def test_an_old_serve_port_and_dist_init_port_are_kept(self):
        converted, _ = resolve_old_sglang_keys(_old_config({"serve_port": "8010", "dist_init_port": "40005"}))
        self.assertEqual((converted["server_params"]["port"], converted["server_params"]["master_port"]), (8010, 40005))

    def test_poll_keys_become_main_effective_client_wait(self):
        cases = [
            ({"inference_poll_iterations": "16"}, 1500),
            ({"inference_poll_iterations": "40", "inference_poll_total_timeout_sec": "7200"}, 3660),
            ({"inference_poll_iterations": "40"}, 3660),
            ({"inference_poll_total_timeout_sec": "1000"}, 1060),
        ]
        for server, client_s in cases:
            with self.subTest(server=server):
                converted, message = resolve_old_sglang_keys(_old_config(server))
                self.assertEqual(converted["timeouts"], {"client_s": client_s})
                self.assertIn("-> timeouts.client_s", message)

    def test_a_poll_pair_with_no_wait_is_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, r"server_params\.inference_poll_iterations"):
            resolve_old_sglang_keys(_old_config({"inference_poll_iterations": "0"}))

    def test_dead_keys_drop_and_a_non_default_backend_is_refused(self):
        converted, message = resolve_old_sglang_keys(
            _old_config(
                {"max_concurrency": "64"}, gpu_name="mi300", benchmark_params={"num_prompts": "4", "backend": "sglang"}
            )
        )
        self.assertNotIn("gpu_name", converted)
        self.assertNotIn("backend", converted["benchmark_params"])
        self.assertIn("server_params.max_concurrency (dropped)", message)
        with self.assertRaisesRegex(ValueError, "benchmark_params.backend") as caught:
            resolve_old_sglang_keys(_old_config(benchmark_params={"num_prompts": "4", "backend": "vllm"}))
        self.assertNotIn("vllm", str(caught.exception))

    def test_runtime_env_moves_to_container_env_without_the_token(self):
        old = _old_config()
        old["container"]["runtime"]["args"]["env"] = {"NCCL_DEBUG": "ERROR", "HF_TOKEN": FAKE_CRED}
        converted, message = resolve_old_sglang_keys(old)
        self.assertEqual(converted["container"]["env"], {"NCCL_DEBUG": "ERROR"})
        self.assertNotIn("env", converted["container"]["runtime"]["args"])
        self.assertNotIn(FAKE_CRED, message)
        SglangConfig.model_validate(converted)

    def test_long_context_switch_on_moves_the_keys_and_acc_specs(self):
        server = {"lng_ctx_activate": True, "context_length": "131072", "chunked_prefill_size": "8192"}
        thresholds = {"ACC_ISL=8192,OSL=32": {"pass_rate": {"kind": "min", "value": 0.8}}}
        old = _old_config(server, long_ctx_niah={"num_prompts": "6", "tolerance_frac": "0.05"}, thresholds=thresholds)
        converted, _ = resolve_old_sglang_keys(old)
        self.assertNotIn("lng_ctx_activate", converted["server_params"])
        self.assertEqual(converted["server_params"]["max_model_length"], 131072)
        self.assertEqual(converted["server_params"]["long_context"], {"chunked_prefill_size": 8192})
        self.assertEqual(
            converted["accuracy"]["custom_eval"],
            [{"type": "niah", "id": "niah_isl8192_osl32", "isl": 8192, "osl": 32, "num_prompts": 6, "seed": 42}],
        )
        self.assertEqual(
            converted["thresholds"]["accuracy"]["custom_eval"]["niah_isl8192_osl32"],
            {"pass_rate": {"kind": "min", "value": 0.8}},
        )
        self.assertNotIn("long_ctx_niah", converted)
        SglangConfig.model_validate(converted)

    def test_long_context_switch_off_or_missing_drops_all_four_and_the_acc_specs(self):
        thresholds = {"ACC_ISL=8192,OSL=32": {"pass_rate": {"kind": "min", "value": 0.8}}}
        for server in ({"lng_ctx_activate": False}, {}):
            with self.subTest(server=server):
                server = {
                    **server,
                    "context_length": "131072",
                    "chunked_prefill_size": "8192",
                    "max_prefill_tokens": "1",
                }
                converted, message = resolve_old_sglang_keys(
                    _old_config(server, long_ctx_niah={"num_prompts": "6"}, thresholds=thresholds)
                )
                self.assertNotIn("lng_ctx_activate", converted["server_params"])
                for name in ("context_length", "chunked_prefill_size", "max_prefill_tokens"):
                    self.assertNotIn(name, converted["server_params"])
                self.assertNotIn("long_ctx_niah", converted)
                self.assertEqual(converted["thresholds"], {})
                self.assertIn("long_ctx_niah (dropped)", message)

    def test_a_niah_option_with_no_shared_key_is_refused_unless_at_its_default(self):
        server = {"lng_ctx_activate": True, "context_length": "131072", "chunked_prefill_size": "8192"}
        with self.assertRaisesRegex(ValueError, "long_ctx_niah.tolerance_frac"):
            resolve_old_sglang_keys(_old_config(server, long_ctx_niah={"tolerance_frac": "0.1"}))

    def test_threshold_specs_take_shared_names_and_kinds_in_one_line(self):
        thresholds = {
            CELL: {
                "output_throughput_per_sec": {"kind": "min_tok_s", "value": 900},
                "mean_e2e_latency_ms": {"kind": "max_ms", "value": 100},
                "mean_ttft_ms": 50,
                "mfu": {"kind": "min", "value": 0.02},
            },
            "BENCH=lm_eval_gsm8k": {"exact_match,flexible-extract": {"kind": "min", "value": 0.9}},
        }
        converted, message = resolve_old_sglang_keys(_old_config(thresholds=thresholds))
        self.assertEqual(
            converted["thresholds"][CELL],
            {
                "output_throughput": {"kind": "min", "value": 900},
                "mean_e2el_ms": {"kind": "max", "value": 100},
                "mean_ttft_ms": {"kind": "max", "value": 50.0},
            },
        )
        self.assertEqual(
            converted["thresholds"]["accuracy"]["lm_eval"]["lm_eval_gsm8k"],
            {"exact_match,flexible-extract": {"kind": "min", "value": 0.9}},
        )
        self.assertIn("metric output_throughput_per_sec -> output_throughput", message)
        self.assertIn("kind max_ms -> max", message)
        self.assertIn("metric mfu (dropped)", message)
        self.assertNotIn(f"thresholds.{CELL}.mean_ttft_ms", message)
        self.assertEqual(message.count("Deprecated"), 1)
        SglangConfig.model_validate(converted)

    def test_threshold_cells_take_the_config_tp_and_pp(self):
        other = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=64"
        alone = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=8"
        thresholds = {
            CELL: {"output_throughput_per_sec": {"kind": "min_tok_s", "value": 1}},
            other: {"output_throughput_per_sec": {"kind": "min_tok_s", "value": 2}},
            alone: {"output_throughput_per_sec": {"kind": "min_tok_s", "value": 3}},
        }
        converted, message = resolve_old_sglang_keys(_old_config(thresholds=thresholds))
        self.assertEqual(
            converted["thresholds"],
            {
                CELL: {"output_throughput": {"kind": "min", "value": 1}},
                "ISL=1024,OSL=1024,TP=8,PP=1,CONC=8": {"output_throughput": {"kind": "min", "value": 3}},
            },
        )
        SglangConfig.model_validate(converted)
        self.assertIn(f"cell {other} (dropped)", message)
        self.assertIn("cells keyed by TP=8,PP=1", message)

    def test_sweep_runs_become_runs_and_an_empty_list_takes_the_threshold_cells(self):
        second = "ISL=8192,OSL=1024,TP=8,PP=1,CONC=64"
        thresholds = {second: {}, CELL: {}}
        converted, _ = resolve_old_sglang_keys(_old_config(sweep={"runs": []}, thresholds=thresholds))
        self.assertEqual(converted["runs"], [CELL, second])
        self.assertEqual(converted["sweeps"][second], {})

    def test_sweeps_and_runs_take_the_config_tp_and_pp(self):
        run = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=64"
        own = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=64"
        old = _old_config(
            {"pipeline_parallelism": "2"}, sweeps={run: {"num_prompts": 8}}, sweep={"runs": [{"combo": run}]}
        )
        converted, _ = resolve_old_sglang_keys(old)
        self.assertEqual(converted["sweeps"], {own: {"num_prompts": 8}})
        self.assertEqual(converted["runs"], [own])
        SglangConfig.model_validate(converted)

    def test_a_cell_a_run_names_wins_among_cells_of_its_shape(self):
        tp8 = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=64"
        tp16 = "ISL=1024,OSL=1024,TP=16,PP=1,CONC=64"
        thresholds = {tp8: {"goodput": {"kind": "min", "value": 1}}, tp16: {"goodput": {"kind": "min", "value": 2}}}
        old = _old_config(
            sweeps={tp8: {"num_prompts": 8}, tp16: {}}, sweep={"runs": [{"combo": tp8}]}, thresholds=thresholds
        )
        converted, _ = resolve_old_sglang_keys(old)
        self.assertEqual(converted["sweeps"], {CELL: {"num_prompts": 8}})
        self.assertEqual(converted["thresholds"], {CELL: {"goodput": {"kind": "min", "value": 1}}})
        SglangConfig.model_validate(converted)

    def test_a_whole_number_float_is_read_as_a_count(self):
        converted, _ = resolve_old_sglang_keys(_old_config({"inference_poll_iterations": 16.0}))
        self.assertEqual(converted["timeouts"], {"client_s": 1500})

    def test_empty_or_missing_sweep_runs_every_threshold_cell_at_the_config_tp_and_pp(self):
        thresholds = {CELL: {"output_throughput_per_sec": {"kind": "min_tok_s", "value": 1}}}
        own = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=64"
        for overrides in ({"sweep": {"runs": []}}, {}):
            with self.subTest(overrides=overrides):
                old = _old_config({"pipeline_parallelism": "2"}, thresholds=thresholds, **overrides)
                old.pop("sweeps")
                if not overrides:
                    old.pop("sweep")
                converted, _ = resolve_old_sglang_keys(old)
                self.assertEqual(converted["runs"], [own])
                self.assertEqual(converted["sweeps"], {own: {}})
                self.assertEqual(list(converted["thresholds"]), [own])
                SglangConfig.model_validate(converted)

    def test_a_new_config_with_one_old_key_keeps_its_ports_and_env(self):
        config = _config(gpu_name="mi300")
        config["container"]["env"] = {"KEEP": "1"}
        converted, message = resolve_old_sglang_keys(config)
        self.assertNotIn("port", converted["server_params"])
        self.assertNotIn("master_port", converted["server_params"])
        self.assertEqual(converted["container"]["env"], {"KEEP": "1"})
        self.assertEqual(message.count("->") + message.count("(dropped)"), 1)

    def test_runtime_env_merges_into_an_existing_container_env(self):
        old = _old_config(server={"env": {"FROM_SERVER": "2"}})
        old["container"]["env"] = {"KEEP": "1"}
        old["container"]["runtime"]["args"]["env"] = {"OLD": "3", "HUGGING_FACE_HUB_TOKEN": FAKE_CRED}
        converted, message = resolve_old_sglang_keys(old)
        self.assertEqual(converted["container"]["env"], {"KEEP": "1", "OLD": "3", "FROM_SERVER": "2"})
        self.assertNotIn(FAKE_CRED, message)
        SglangConfig.model_validate(converted)

    def test_add_flags_as_a_string_or_with_an_equals_sign(self):
        for flags in ("--context-length=4096", ["--context-length 4096"]):
            with self.subTest(flags=flags):
                converted, _ = resolve_old_sglang_keys(_old_config({"add_flags": flags}))
                self.assertEqual(converted["server_params"]["engine_args"], {"context_length": 4096})

    def test_a_flag_value_that_is_not_exactly_a_number_stays_a_string(self):
        converted, _ = resolve_old_sglang_keys(_old_config({"add_flags": ["--cuda-graph-bs 08", "--x 1e3"]}))
        self.assertEqual(converted["server_params"]["engine_args"], {"cuda_graph_bs": "08", "x": "1e3"})

    def test_a_poll_count_that_is_not_a_number_names_the_key(self):
        with self.assertRaisesRegex(ValueError, "server_params.inference_poll_iterations must be a whole number"):
            resolve_old_sglang_keys(_old_config({"inference_poll_iterations": "many"}))

    def test_an_unknown_niah_option_is_refused_by_name(self):
        server = {"lng_ctx_activate": True, "context_length": "131072", "chunked_prefill_size": "8192"}
        with self.assertRaisesRegex(ValueError, "long_ctx_niah.extra_option"):
            resolve_old_sglang_keys(_old_config(server, long_ctx_niah={"extra_option": 1}))

    def test_an_lm_eval_option_with_no_shared_key_is_refused_unless_at_its_default(self):
        task = {"id": "lm_eval_gsm8k", "tasks": "gsm8k", "tolerance_frac": 0.05, "label": "gsm8k"}
        converted, _ = resolve_old_sglang_keys(_old_config(accuracy={"tasks": [task]}))
        self.assertEqual(converted["accuracy"]["lm_eval"]["tasks"], [{"id": "lm_eval_gsm8k", "tasks": "gsm8k"}])
        with self.assertRaisesRegex(ValueError, r"accuracy.tasks\[0\].tolerance_frac"):
            resolve_old_sglang_keys(_old_config(accuracy={"tasks": [{**task, "tolerance_frac": 0.2}]}))

    def test_a_new_threshold_file_is_not_rekeyed_for_a_new_config(self):
        other = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=64"
        config = _config(thresholds={other: {"output_throughput": {"kind": "min", "value": 1}}})
        converted, message = resolve_old_sglang_keys(config)
        self.assertIs(converted, config)
        self.assertIsNone(message)

    def test_a_spec_without_a_kind_is_left_for_validation(self):
        thresholds = {CELL: {"output_throughput_per_sec": {"value": 1}}}
        converted, _ = resolve_old_sglang_keys(_old_config(thresholds=thresholds))
        self.assertEqual(converted["thresholds"][CELL], {"output_throughput": {"value": 1}})

    def test_a_threshold_file_read_alone_keeps_its_cell_keys(self):
        other = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=64"
        thresholds = {other: {"mean_e2e_latency_ms": {"kind": "max_ms", "value": 5}}, "BENCH=lm_eval_gsm8k": {}}
        converted, message = resolve_old_sglang_threshold_names(thresholds)
        self.assertEqual(converted, {other: {"mean_e2el_ms": {"kind": "max", "value": 5}}, "BENCH=lm_eval_gsm8k": {}})
        self.assertIn("mean_e2e_latency_ms", message)


if __name__ == "__main__":
    unittest.main()
