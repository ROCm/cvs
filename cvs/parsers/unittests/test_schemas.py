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

    def test_bad_run_entry_not_printed(self):
        cases = {
            "unknown": ([CELL, SECRET], "(config): runs names keys missing from sweeps at positions [1]"),
            "repeated": ([SECRET, CELL, SECRET], "(config): runs repeats an earlier entry at positions [2]"),
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
