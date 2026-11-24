"""Configuration validation entry-point compatibility."""

import json
import getpass
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.parsers.schemas import (
    InferenceVariantConfig,
    PreflightConfigFile,
    check_container_network_env,
    check_not_implemented,
    container_ib_hcas,
    container_sets_sockets,
    fill_info_specs,
    not_implemented_names,
    poll_schedule,
    validate_config_file,
    validation_problem_lines,
)


class TestAortaConfigDispatch(unittest.TestCase):
    def test_generic_validator_delegates_to_shared_variant_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = variant_dict()
            thresholds = raw.pop("thresholds")
            (root / "test_threshold.json").write_text(json.dumps(thresholds))
            path = root / "aorta.json"
            path.write_text(json.dumps(raw))
            for mode in ("auto", "aorta"):
                with self.subTest(mode=mode):
                    config = validate_config_file(path, config_type=mode)
                    self.assertIsInstance(config, AortaVariantConfig)
                    self.assertEqual(config.thresholds, thresholds)

    def test_generic_validator_uses_local_user_without_cluster_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = variant_dict()
            thresholds = raw.pop("thresholds")
            raw["container"]["name"] = "{user-id}_aorta"
            (root / "test_threshold.json").write_text(json.dumps(thresholds))
            path = root / "aorta.json"
            path.write_text(json.dumps(raw))
            config = validate_config_file(path)
            self.assertEqual(config.container.name, f"{getpass.getuser()}_aorta")


class TestPreflightRdmaGidType(unittest.TestCase):
    def test_expected_nic_count_validates_positive_integer(self):
        cfg = PreflightConfigFile.model_validate({'connectivity_check': {'rdma': {'expected_nic_count': 8}}})
        self.assertEqual(cfg.connectivity_check.rdma.expected_nic_count, 8)
        with self.assertRaises(ValueError):
            PreflightConfigFile.model_validate({'connectivity_check': {'rdma': {'expected_nic_count': 0}}})

    def test_default_gid_type(self):
        config = PreflightConfigFile.model_validate({'connectivity_check': {'rdma': {'gid_index': '3'}}})
        self.assertEqual(config.connectivity_check.rdma.gid_type, 'RoCE v2')

    def test_any_gid_type_round_trips(self):
        config = PreflightConfigFile.model_validate({'connectivity_check': {'rdma': {'gid_type': 'any'}}})
        self.assertEqual(config.connectivity_check.rdma.gid_type, 'any')
        self.assertEqual(config.model_dump()['connectivity_check']['rdma']['gid_type'], 'any')


CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
OTHER_CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32"


def inference_config(**overrides):
    config = {
        "enforce_thresholds": True,
        "threshold_json": "llama_threshold.json",
        "paths": {"shared_fs": "/home/u", "models_dir": "/models", "log_dir": "/home/u/LOGS", "hf_token_file": "/t"},
        "container": {
            "name": "vllm_llama",
            "image": "rocm/vllm:latest",
            "env": {},
            "runtime": {"name": "docker", "args": {"volumes": ["/data:/models:ro"]}},
        },
        "server_params": {"model": "meta-llama/Llama-3.3-70B", "tp": 8},
        "benchmark_params": {"num_prompts": 320},
        "sweeps": {CELL: {}, OTHER_CELL: {}},
        "runs": [CELL],
        "thresholds": {CELL: {"output_throughput": {"kind": "min", "value": 100}}},
    }
    config.update(overrides)
    return config


SECRET = "sk-SECRET123"
HF_TOKEN = "hf_AbCdEf123"
CELL_KEY_FORM = "ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>"


def problems(raw):
    try:
        InferenceVariantConfig.model_validate(raw)
    except ValidationError as exc:
        return validation_problem_lines(exc)
    return []


class TestInferenceVariantConfigPlaceholders(unittest.TestCase):
    def test_changeme_rejected_names_field(self):
        self.assertEqual(problems(inference_config()), [])
        cases = {
            "container.image": ("container", "image"),
            "paths.models_dir": ("paths", "models_dir"),
            "server_params.model": ("server_params", "model"),
        }
        for path, (block, key) in cases.items():
            with self.subTest(path=path):
                raw = inference_config()
                raw[block][key] = "/x/<changeme>"
                lines = problems(raw)
                self.assertEqual(len(lines), 1)
                self.assertIn(f"{block}: ", lines[0])
                self.assertIn(key, lines[0])
                self.assertNotIn("/x/", lines[0])

        raw = inference_config()
        raw["container"]["runtime"]["args"]["volumes"] = ["/data:/models:ro", "secret<changeme>:/m"]
        raw["container"]["env"] = {"NCCL_IB_HCA": "<changeme>"}
        lines = problems(raw)
        self.assertEqual(len(lines), 1)
        self.assertIn("runtime.args.volumes[1]", lines[0])
        self.assertIn("env.NCCL_IB_HCA", lines[0])
        self.assertNotIn("secret", lines[0])

    def test_all_changeme_fields_reported_together(self):
        raw = inference_config()
        raw["container"]["image"] = "<changeme>"
        raw["paths"]["models_dir"] = "<changeme>"
        raw["server_params"]["model"] = "<changeme>"
        lines = problems(raw)
        self.assertEqual(
            lines,
            [
                "paths: replace <changeme> with the value for your setup in models_dir",
                "container: replace <changeme> with the value for your setup in image",
                "server_params: replace <changeme> with the value for your setup in model",
            ],
        )

    def test_comment_keys_are_not_checked_for_changeme(self):
        raw = inference_config()
        raw["_comment"] = "replace every <changeme> before running"
        raw["container"]["runtime"]["args"]["_volumes_example"] = ["<changeme>:/models:ro"]
        self.assertEqual(problems(raw), [])


class TestInferenceVariantConfigBlocks(unittest.TestCase):
    def test_comment_keys_dropped(self):
        raw = inference_config()
        raw["_comment"] = "top"
        raw["paths"]["_comment"] = "paths"
        raw["container"]["runtime"]["args"]["_volumes_example"] = ["/data/models:/models:ro"]
        raw["server_params"]["_comment"] = "server"
        raw["server_params"]["engine_args"] = {"_comment": "flags", "kv_cache_dtype": "fp8"}
        raw["sweeps"]["_comment"] = "cells"
        raw["sweeps"][CELL] = {"_comment": "cell"}
        raw["thresholds"]["_comment"] = "file"
        raw["thresholds"][CELL]["_comment"] = "cell"
        config = InferenceVariantConfig.model_validate(raw)
        self.assertEqual(config.server_params.engine_args, {"kv_cache_dtype": "fp8"})
        self.assertEqual(config.sweeps, {CELL: {}, OTHER_CELL: {}})
        self.assertNotIn("_volumes_example", config.container.model_dump()["runtime"]["args"])
        self.assertEqual(config.thresholds.get(CELL), {"output_throughput": {"kind": "min", "value": 100}})

        raw = inference_config()
        raw["server_params"]["tpp"] = 8
        self.assertEqual(len(problems(raw)), 1)
        self.assertTrue(problems(raw)[0].startswith("server_params.tpp:"))

    def test_container_env_only_env_block(self):
        raw = inference_config()
        raw["container"]["env"] = {"NCCL_SOCKET_IFNAME": "eth0"}
        self.assertEqual(problems(raw), [])
        raw["container"]["runtime"]["args"]["env"] = {"NCCL_SOCKET_IFNAME": "eth0"}
        self.assertEqual(
            problems(raw), ["container: runtime.args.env is not read; set these variables in container.env"]
        )

    def test_hf_token_rejected(self):
        for name in ("HF_TOKEN", "hugging_face_hub_token"):
            with self.subTest(name=name):
                raw = inference_config()
                raw["container"]["env"] = {name: "hf_SECRET"}
                lines = problems(raw)
                self.assertEqual(len(lines), 1)
                self.assertIn(name, lines[0])
                self.assertNotIn("hf_SECRET", lines[0])
        self.assertEqual(problems(inference_config()), [])

    def test_threshold_kinds(self):
        for kind in ("min", "max", "info"):
            with self.subTest(kind=kind):
                raw = inference_config(thresholds={CELL: {"output_throughput": {"kind": kind, "value": 1.5}}})
                self.assertEqual(problems(raw), [])
        for spec in (
            {"kind": "max_ms", "value": 1},
            {"kind": "min_tok_s", "value": 1},
            {"kind": "within", "value": 1, "tolerance_pct": 5},
            {"kind": "min", "value": "100"},
        ):
            with self.subTest(spec=spec):
                raw = inference_config(thresholds={CELL: {"output_throughput": spec}})
                self.assertTrue(problems(raw))

    def test_threshold_coverage(self):
        raw = inference_config(thresholds={OTHER_CELL: {"output_throughput": {"kind": "min", "value": 1}}})
        lines = problems(raw)
        self.assertEqual(len(lines), 1)
        self.assertIn(CELL, lines[0])

        raw["runs"] = [OTHER_CELL]
        self.assertEqual(problems(raw), [])

        raw = inference_config(enforce_thresholds=False, thresholds={})
        self.assertEqual(problems(raw), [])

        lines = problems(inference_config(thresholds={CELL: {}, "throughput": {}}))
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("thresholds.<key>: not a cell key"), lines[0])

    def test_threshold_file_round_trips_in_its_file_shape(self):
        config = InferenceVariantConfig.model_validate(inference_config())
        dumped = config.model_dump()
        self.assertEqual(
            dumped["thresholds"],
            {
                CELL: {"output_throughput": {"kind": "min", "value": 100, "optional": False}},
                "accuracy": {"lm_eval": {}, "custom_eval": {}, "quant_parity": {}, "mtp_quality": {}},
            },
        )
        self.assertEqual(InferenceVariantConfig.model_validate(dumped), config)

    def test_threshold_file_dumps_that_leave_out_cells(self):
        config = InferenceVariantConfig.model_validate(inference_config())
        thresholds = config.thresholds
        self.assertEqual(
            thresholds.model_dump(exclude={"cells"}),
            {"accuracy": {"lm_eval": {}, "custom_eval": {}, "quant_parity": {}, "mtp_quality": {}}},
        )
        self.assertEqual(thresholds.model_dump(include={"accuracy"}), thresholds.model_dump(exclude={"cells"}))
        empty = InferenceVariantConfig.model_validate(inference_config(enforce_thresholds=False, thresholds={}))
        self.assertEqual(empty.thresholds.model_dump(exclude_defaults=True), {})
        self.assertEqual(
            thresholds.model_dump(exclude_defaults=True), {CELL: {"output_throughput": {"kind": "min", "value": 100}}}
        )

    def test_sweeps_and_runs(self):
        pp2_cell = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"
        raw = inference_config()
        raw["server_params"]["pp"] = 2
        raw["sweeps"] = {pp2_cell: {"num_prompts": 640, "temperature": 0.0}}
        raw["runs"] = [pp2_cell]
        raw["thresholds"] = {pp2_cell: {}}
        self.assertEqual(problems(raw), [])

        bad_cell = "OSL=1024,ISL=1024,TP=8,PP=1,CONC=16"
        tp4_cell = "ISL=1024,OSL=1024,TP=4,PP=1,CONC=16"
        bad = {
            "run missing from sweeps": (
                {"runs": [CELL, "ISL=1,OSL=1,TP=8,PP=1,CONC=1"], "enforce_thresholds": False},
                "runs names keys missing from sweeps",
            ),
            "repeated run": ({"runs": [CELL, CELL]}, "runs repeats"),
            "empty runs": ({"runs": []}, "runs must list at least one sweeps key"),
            "axes out of order": ({"sweeps": {CELL: {}, bad_cell: {}}}, "sweeps.<key>: not a cell key"),
            "TP unequal": ({"sweeps": {CELL: {}, tp4_cell: {}}}, "TP and PP must equal server_params"),
            "reserved override": ({"sweeps": {CELL: {"max_concurrency": 4}}}, "cannot set harness fields"),
            "server override": ({"sweeps": {CELL: {"port": 9000}}}, "cannot set harness fields"),
        }
        for name, (overrides, expected) in bad.items():
            with self.subTest(name=name):
                lines = problems(inference_config(**overrides))
                self.assertEqual(len(lines), 1, lines)
                self.assertIn(expected, lines[0])

    def test_rejected_sweep_key_not_printed(self):
        raw = inference_config()
        raw["sweeps"][SECRET] = {}
        self.assertEqual(problems(raw), [f"sweeps.<key>: not a cell key of the form {CELL_KEY_FORM}"])

    def test_rejected_threshold_key_not_printed(self):
        raw = inference_config(thresholds={CELL: {}, SECRET: {"output_throughput": {"kind": "min", "value": 1}}})
        self.assertEqual(problems(raw), [f"thresholds.<key>: not a cell key of the form {CELL_KEY_FORM}"])

    def test_bad_flag_name_not_printed(self):
        raw = inference_config()
        raw["server_params"]["engine_args"] = {SECRET: 1}
        raw["sweeps"][CELL] = {SECRET: 1}
        self.assertEqual(problems(raw), ["server_params: engine_args: flag names must be snake_case"])
        raw["server_params"]["engine_args"] = {}
        self.assertEqual(problems(raw), [f"(config): sweeps.{CELL}: flag names must be snake_case"])

    def test_rejected_key_with_bad_value_not_printed(self):
        raw = inference_config(thresholds={CELL: {}, SECRET: {"output_throughput": {"kind": "bad", "value": 1}}})
        raw["sweeps"][SECRET] = "not a dict"
        lines = problems(raw)
        self.assertEqual(
            lines,
            [
                f"sweeps.<key>: not a cell key of the form {CELL_KEY_FORM}",
                "sweeps.<key>: Input should be a valid dictionary",
                f"thresholds.<key>: not a cell key of the form {CELL_KEY_FORM}",
                "thresholds.<key>.output_throughput.kind: Input should be 'min', 'max' or 'info'",
            ],
        )

    def test_unknown_key_printed_only_when_plain(self):
        raw = inference_config()
        raw[HF_TOKEN] = 1
        raw["server_params"][SECRET] = 1
        raw["server_params"]["tpp"] = 8
        self.assertEqual(
            problems(raw),
            [
                "server_params.<key>: Extra inputs are not permitted",
                "server_params.tpp: Extra inputs are not permitted",
                "<key>: Extra inputs are not permitted",
            ],
        )

    def test_changeme_under_rejected_key_not_printed(self):
        raw = inference_config()
        raw["sweeps"][HF_TOKEN] = {"temperature": "<changeme>"}
        raw["container"]["env"] = {"NCCL_IB_HCA": "<changeme>", SECRET: "<changeme>"}
        self.assertEqual(
            problems(raw),
            [
                "container: replace <changeme> with the value for your setup in env.NCCL_IB_HCA, env.<key>",
                "sweeps: replace <changeme> with the value for your setup in <key>.temperature",
            ],
        )

    def test_bad_run_entry_not_printed(self):
        cases = {
            "unknown": ([CELL, SECRET], "(config): runs names keys missing from sweeps at runs.1"),
            "repeated": ([SECRET, CELL, SECRET], "(config): runs repeats an earlier entry at runs.2"),
        }
        for name, (runs, expected) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(problems(inference_config(runs=runs, enforce_thresholds=False)), [expected])

    def test_engine_and_client_flags(self):
        good = inference_config()
        good["server_params"]["engine_args"] = {"kv_cache_dtype": "fp8", "limit_mm_per_prompt": {"image": 0}}
        good["benchmark_params"]["client_args"] = {"temperature": 0.0, "extra": [1, 2]}
        self.assertEqual(problems(good), [])
        cases = {
            "server_params": {
                "Bad-Name": "flag names must be snake_case",
                "nested": "lists may hold scalar values only",
                "tp": "cannot set harness fields",
                "node_rank": "cannot set harness fields",
            },
            "benchmark_params": {
                "Bad-Name": "flag names must be snake_case",
                "nested": "lists may hold scalar values only",
                "num_prompts": "cannot set harness fields",
                "result_dir": "cannot set harness fields",
            },
        }
        for block, flags in cases.items():
            section = "engine_args" if block == "server_params" else "client_args"
            for name, expected in flags.items():
                with self.subTest(block=block, name=name):
                    raw = inference_config()
                    raw[block][section] = {name: [[1]] if name == "nested" else 1}
                    lines = problems(raw)
                    self.assertEqual(len(lines), 1, lines)
                    self.assertTrue(lines[0].startswith(f"{block}: "), lines[0])
                    self.assertIn(expected, lines[0])
        raw = inference_config()
        raw["server_params"]["engine_args"] = {"bad": {"x": object()}}
        self.assertIn("must be JSON serializable", problems(raw)[0])

    def test_lm_eval_accessors(self):
        config = InferenceVariantConfig.model_validate(inference_config())
        self.assertEqual(config.lm_eval_tasks(), [])
        self.assertEqual(config.lm_eval_thresholds("gsm8k_flex"), {})

        tasks = [{"id": "gsm8k_flex", "task": "gsm8k"}, {"id": "mmlu", "task": "mmlu"}]
        thresholds = {
            CELL: {"output_throughput": {"kind": "min", "value": 100}},
            "accuracy": {"lm_eval": {"gsm8k_flex": {"exact_match": {"kind": "min", "value": 0.8, "optional": True}}}},
        }
        config = InferenceVariantConfig.model_validate(
            inference_config(accuracy={"lm_eval": {"tasks": tasks}}, thresholds=thresholds)
        )
        self.assertEqual([task.id for task in config.lm_eval_tasks()], ["gsm8k_flex", "mmlu"])
        config.lm_eval_tasks().clear()
        self.assertEqual(len(config.accuracy.lm_eval.tasks), 2)
        self.assertEqual(
            config.lm_eval_thresholds("gsm8k_flex"), {"exact_match": {"kind": "min", "value": 0.8, "optional": True}}
        )
        self.assertEqual(config.lm_eval_thresholds("mmlu"), {})

    def test_accuracy_ids_unique(self):
        task = {"id": "gsm8k_flex", "task": "gsm8k"}
        niah = {"type": "niah", "id": "niah_8k", "isl": 8192}
        self.assertEqual(problems(inference_config(accuracy={"lm_eval": {"tasks": [task]}})), [])
        lines = problems(inference_config(accuracy={"lm_eval": {"tasks": [task, task]}}))
        self.assertEqual(len(lines), 1)
        self.assertIn("gsm8k_flex", lines[0])
        lines = problems(inference_config(accuracy={"custom_eval": [niah, niah]}))
        self.assertEqual(len(lines), 1)
        self.assertIn("niah_8k", lines[0])

    def test_quant_parity_has_no_reference_config_stem(self):
        raw = inference_config(accuracy={"quant_parity": {"reference_config_stem": "llama_fp16"}})
        lines = problems(raw)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("accuracy.quant_parity.reference_config_stem:"))


class TestValidateConfigFileErrors(unittest.TestCase):
    def test_validate_config_file_errors_without_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cluster.json"
            path.write_text(
                json.dumps({"username": "u", "priv_key_file": "/k", "head_node_dict": {}, "node_dict": "s3cr3t-value"})
            )
            with self.assertRaises(ValueError) as caught:
                validate_config_file(path)
        message = str(caught.exception)
        self.assertIn("node_dict: ", message)
        self.assertNotIn("s3cr3t-value", message)


INFO = {"kind": "info", "value": 0}


class TestFillInfoSpecs(unittest.TestCase):
    def test_adds_info_for_each_gated_metric_a_cell_lacks(self):
        thresholds = {CELL: {}, OTHER_CELL: {"failed": {"kind": "max", "value": 0}}}
        filled = fill_info_specs(thresholds, {"failed", "p99_itl_ms"})
        self.assertEqual(filled[CELL], {"failed": INFO, "p99_itl_ms": INFO})
        self.assertEqual(filled[OTHER_CELL], {"failed": {"kind": "max", "value": 0}, "p99_itl_ms": INFO})

    def test_keeps_every_spec_the_file_sets(self):
        cell = {"output_throughput": {"kind": "min", "value": 5}, "made_up_metric": {"kind": "min", "value": 1}}
        filled = fill_info_specs({CELL: cell}, {"output_throughput"})
        self.assertEqual(filled, {CELL: cell})
        self.assertIsNot(filled[CELL], cell)

    def test_fill_skips_underscore_cells_even_when_they_are_dicts(self):
        self.assertEqual(fill_info_specs({"_example": {}}, {"failed"}), {"_example": {}})

    def test_never_adds_an_unreported_metric(self):
        filled = fill_info_specs({CELL: {}}, {"failed", "p50_queue_time_ms"}, unreported={"p50_queue_time_ms"})
        self.assertEqual(filled[CELL], {"failed": INFO})

    def test_never_adds_a_required_metric_so_coverage_still_fails(self):
        filled = fill_info_specs({CELL: {}}, {"failed", "output_throughput"}, required={"output_throughput"})
        self.assertEqual(filled[CELL], {"failed": INFO})

    def test_leaves_accuracy_comments_and_missing_cells_alone(self):
        accuracy = {"lm_eval": {"gsm8k": {"gsm8k.exact_match__strict-match": {"kind": "min", "value": 0.5}}}}
        thresholds = {"_comment": "file", CELL: {}, "accuracy": accuracy}
        filled = fill_info_specs(thresholds, {"failed"})
        self.assertEqual(filled, {"_comment": "file", CELL: {"failed": INFO}, "accuracy": accuracy})
        self.assertEqual(thresholds[CELL], {})


class TestNotImplemented(unittest.TestCase):
    def test_not_implemented_check_names_each_field_and_metric(self):
        raw = inference_config(
            thresholds={CELL: {"mfu": {"kind": "min", "value": 1}, "failed": {"kind": "max", "value": 0}}},
            accuracy={"quant_parity": {"enabled": True}},
        )
        raw["server_params"].update({"dp": 2, "engine_args": {"speculative_config": "{}"}})
        raw["benchmark_params"]["scaling_baseline_output_throughput"] = 900.0
        config = InferenceVariantConfig.model_validate(raw)
        fields = (
            "server_params.dp",
            "server_params.ep",
            "server_params.engine_args.speculative_config",
            "benchmark_params.scaling_baseline_output_throughput",
            "accuracy.quant_parity",
        )
        names = not_implemented_names(config, fields=fields, metrics={"mfu"})
        self.assertEqual(
            names,
            [
                "server_params.dp",
                "server_params.engine_args.speculative_config",
                "benchmark_params.scaling_baseline_output_throughput",
                "accuracy.quant_parity",
                f"thresholds {CELL}.mfu",
            ],
        )
        pattern = r"^not implemented for TestEngine: server_params.dp, .*; leave them at their defaults$"
        with self.assertRaisesRegex(ValueError, pattern):
            check_not_implemented("TestEngine", names)
        check_not_implemented("TestEngine", [])

    def test_defaults_pass_and_an_info_spec_still_counts(self):
        config = InferenceVariantConfig.model_validate(inference_config())
        every_field = (
            "server_params.dp",
            "server_params.ep",
            "server_params.prefill_nnodes",
            "server_params.long_context",
            "benchmark_params.scaling_baseline_output_throughput",
            "accuracy.custom_eval",
            "accuracy.quant_parity",
            "accuracy.mtp_quality",
        )
        self.assertEqual(not_implemented_names(config, fields=every_field), [])
        config = InferenceVariantConfig.model_validate(
            inference_config(thresholds={CELL: {"mfu": {"kind": "info", "value": 0}}})
        )
        self.assertEqual(not_implemented_names(config, metrics={"mfu"}), [f"thresholds {CELL}.mfu"])

    def test_each_off_default_field_is_named(self):
        raw = inference_config(
            accuracy={"custom_eval": [{"id": "niah", "type": "niah", "isl": 1024}], "mtp_quality": {"enabled": True}}
        )
        raw["server_params"].update({"ep": True, "prefill_nnodes": 2, "long_context": {"max_prefill_tokens": 8}})
        config = InferenceVariantConfig.model_validate(raw)
        fields = (
            "server_params.ep",
            "server_params.prefill_nnodes",
            "server_params.long_context",
            "accuracy.custom_eval",
            "accuracy.mtp_quality",
            "server_params.engine_args.speculative_config",
        )
        self.assertEqual(not_implemented_names(config, fields=fields), list(fields[:5]))

    def test_threshold_accuracy_sections(self):
        accuracy = {"custom_eval": {"niah": {"pass_rate": {"kind": "min", "value": 0.8}}}}
        config = InferenceVariantConfig.model_validate(inference_config(thresholds={CELL: {}, "accuracy": accuracy}))
        names = not_implemented_names(config, threshold_sections=("custom_eval", "quant_parity"))
        self.assertEqual(names, ["thresholds accuracy.custom_eval"])

    def test_unknown_field_name_fails(self):
        config = InferenceVariantConfig.model_validate(inference_config())
        with self.assertRaisesRegex(ValueError, "no not-implemented check for server_params.dpp"):
            not_implemented_names(config, fields=("server_params.dpp",))
        with self.assertRaisesRegex(ValueError, "no accuracy threshold section niah"):
            not_implemented_names(config, threshold_sections=("niah",))


class TestContainerNetworkEnv(unittest.TestCase):
    def _container(self, env):
        raw = inference_config()
        raw["container"]["env"] = env
        return InferenceVariantConfig.model_validate(raw).container

    def test_socket_variables_all_or_none(self):
        every = {"NCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0", "TP_SOCKET_IFNAME": "eth0"}
        check_container_network_env(self._container(every))
        check_container_network_env(self._container({}))
        self.assertTrue(container_sets_sockets(self._container(every)))
        self.assertFalse(container_sets_sockets(self._container({})))
        with self.assertRaisesRegex(ValueError, r"missing: \['GLOO_SOCKET_IFNAME', 'TP_SOCKET_IFNAME'\]"):
            check_container_network_env(self._container({"NCCL_SOCKET_IFNAME": "eth0"}))
        with self.assertRaisesRegex(ValueError, r"cannot be empty: \['TP_SOCKET_IFNAME'\]"):
            check_container_network_env(self._container({**every, "TP_SOCKET_IFNAME": " "}))

    def test_valid_ib_hca_passes_and_one_socket_variable_counts(self):
        check_container_network_env(self._container({"NCCL_IB_HCA": "mlx5_0"}))
        self.assertTrue(container_sets_sockets(self._container({"NCCL_SOCKET_IFNAME": "eth0"})))

    def test_nccl_ib_hca_needs_a_device(self):
        self.assertIsNone(container_ib_hcas(self._container({})))
        self.assertEqual(container_ib_hcas(self._container({"NCCL_IB_HCA": " mlx5_0, ,mlx5_1"})), ["mlx5_0", "mlx5_1"])
        with self.assertRaisesRegex(ValueError, "NCCL_IB_HCA must name at least one device"):
            check_container_network_env(self._container({"NCCL_IB_HCA": " , "}))


class TestPollSchedule(unittest.TestCase):
    def test_first_wait_then_enough_polls_for_the_total(self):
        self.assertEqual(poll_schedule(3930, 330, 60), (330, 60))
        self.assertEqual(poll_schedule(5520, 120, 60), (120, 90))
        self.assertEqual(poll_schedule(361, 0, 60), (0, 7))

    def test_short_total_caps_the_wait_and_keeps_one_poll(self):
        self.assertEqual(poll_schedule(100, 330, 60), (100, 1))
        self.assertEqual(poll_schedule(330, 330, 60), (330, 1))
