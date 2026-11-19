"""Configuration validation entry-point compatibility."""

import json
import getpass
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.parsers.schemas import (
    INFERENCE_METRIC_NAMES,
    ContainerEnvTokenError,
    InferenceConfig,
    InferenceThresholdFile,
    InferenceThresholdSpec,
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
    def test_default_gid_type(self):
        config = PreflightConfigFile.model_validate({'connectivity_check': {'rdma': {'gid_index': '3'}}})
        self.assertEqual(config.connectivity_check.rdma.gid_type, 'RoCE v2')

    def test_any_gid_type_round_trips(self):
        config = PreflightConfigFile.model_validate({'connectivity_check': {'rdma': {'gid_type': 'any'}}})
        self.assertEqual(config.connectivity_check.rdma.gid_type, 'any')
        self.assertEqual(config.model_dump()['connectivity_check']['rdma']['gid_type'], 'any')


FAKE_TOKEN = "hf_FAKE123"
CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
REJECT = {"placeholders": "reject"}


def _inference_dict(**overrides):
    value = {
        "paths": {"shared_fs": "/fs", "models_dir": "/models", "log_dir": "/logs", "hf_token_file": "/fs/.hf"},
        "container": {"name": "bench", "image": "rocm/vllm:latest", "runtime": {"name": "docker", "args": {}}},
        "server_params": {"model": "m", "tp": 8},
        "benchmark_params": {"num_prompts": 32},
        "sweeps": {CELL: {}},
        "runs": [CELL],
    }
    value.update(overrides)
    return value


def _problems(data, model_cls=InferenceConfig, context=None):
    try:
        model_cls.model_validate(data, context=context)
    except ValidationError as exc:
        return "\n".join(validation_problem_lines(exc))
    raise AssertionError("expected a validation error")


class TestSchemasImportsNoLibrary(unittest.TestCase):
    def test_importing_schemas_loads_no_cvs_lib_module(self):
        code = "import sys, cvs.parsers.schemas; print([m for m in sys.modules if m.startswith('cvs.lib')])"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "[]")


class TestInferenceServerAndClientBlocks(unittest.TestCase):
    def test_unknown_server_key_fails_naming_its_path(self):
        problems = _problems(_inference_dict(server_params={"model": "m", "tp": 8, "foo": 1}))
        self.assertIn("server_params.foo", problems)

    def test_the_same_key_under_engine_args_passes(self):
        config = InferenceConfig.model_validate(
            _inference_dict(server_params={"model": "m", "tp": 8, "engine_args": {"foo": 1}})
        )
        self.assertEqual(config.server_params.engine_args, {"foo": 1})

    def test_engine_args_naming_a_harness_field_fails(self):
        problems = _problems(_inference_dict(server_params={"model": "m", "tp": 8, "engine_args": {"tp": 4}}))
        self.assertIn("server_params.engine_args cannot set harness fields: ['tp']", problems)

    def test_client_args_naming_a_harness_flag_fails(self):
        problems = _problems(_inference_dict(benchmark_params={"num_prompts": 32, "client_args": {"base_url": "x"}}))
        self.assertIn("['base_url']", problems)

    def test_flag_names_must_be_snake_case(self):
        problems = _problems(
            _inference_dict(server_params={"model": "m", "tp": 8, "engine_args": {"kv-cache-dtype": "fp8"}})
        )
        self.assertIn("kv-cache-dtype: flag names must be snake_case", problems)


class TestInferenceErrorsCarryNoValues(unittest.TestCase):
    def test_placeholder_model_names_the_field_without_its_value(self):
        problems = _problems(
            _inference_dict(server_params={"model": f"{FAKE_TOKEN}<changeme>", "tp": 8}), context=REJECT
        )
        self.assertIn("server_params.model", problems)
        self.assertNotIn(FAKE_TOKEN, problems)

    def test_placeholder_in_engine_args_names_the_flag_without_its_value(self):
        data = _inference_dict(
            server_params={"model": "m", "tp": 8, "engine_args": {"token": [f"{FAKE_TOKEN}<changeme>"]}}
        )
        problems = _problems(data, context=REJECT)
        self.assertIn("server_params.engine_args.token", problems)
        self.assertNotIn(FAKE_TOKEN, problems)

    def test_fake_token_in_a_refused_field_is_not_echoed(self):
        problems = _problems(_inference_dict(server_params={"model": "m", "tp": 8, "token": FAKE_TOKEN}))
        self.assertIn("server_params.token", problems)
        self.assertNotIn(FAKE_TOKEN, problems)

    def test_hf_token_in_container_env_fails_without_its_value(self):
        container = {
            "name": "bench",
            "image": "img",
            "runtime": {"name": "docker", "args": {}},
            "env": {"hf_token": FAKE_TOKEN},
        }
        with self.assertRaises(ContainerEnvTokenError) as ctx:
            InferenceConfig.model_validate(_inference_dict(container=container))
        self.assertIn("hf_token", str(ctx.exception))
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))

    def test_runtime_args_env_fails(self):
        container = {"name": "bench", "image": "img", "runtime": {"name": "docker", "args": {"env": {"A": "1"}}}}
        problems = _problems(_inference_dict(container=container))
        self.assertIn(
            "container.runtime.args: runtime.args.env is not read; set these variables in container.env", problems
        )

    def test_validate_config_file_reports_paths_without_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preflight.json"
            path.write_text(json.dumps({"preflight": {"parallelism": {"parallel_group_size": FAKE_TOKEN}}}))
            with self.assertRaises(ValueError) as ctx:
                validate_config_file(path)
        self.assertIn("parallelism.parallel_group_size", str(ctx.exception))
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)


class TestInferenceThresholds(unittest.TestCase):
    def test_only_min_max_and_info_kinds_validate(self):
        for kind in ("min", "max", "info"):
            with self.subTest(kind=kind):
                spec = InferenceThresholdSpec.model_validate({"kind": kind, "value": 1})
                self.assertEqual(spec.kind, kind)
        for kind in ("max_ms", "min_tok_s"):
            with self.subTest(kind=kind):
                self.assertIn("kind", _problems({"kind": kind, "value": 1}, InferenceThresholdSpec))

    def test_threshold_file_sections_must_be_cell_keys(self):
        thresholds = InferenceThresholdFile.model_validate(
            {"_comment": "x", CELL: {"output_throughput": {"kind": "min", "value": 1}}}
        )
        self.assertEqual(list(thresholds.cells()), [CELL])
        self.assertIn("is not a cell key", _problems({"ISL=1024": {}}, InferenceThresholdFile))

    def test_accuracy_sections_take_specs_by_id(self):
        thresholds = InferenceThresholdFile.model_validate(
            {"accuracy": {"lm_eval": {"gsm8k": {"exact_match": {"kind": "min", "value": 0.9}}}}}
        )
        self.assertEqual(thresholds.accuracy.lm_eval["gsm8k"]["exact_match"].value, 0.9)

    def test_metric_names_include_every_vllm_registry_metric(self):
        from cvs.lib.inference.utils.vllm_metrics import METRIC_REGISTRY

        self.assertLessEqual({definition.name for definition in METRIC_REGISTRY}, INFERENCE_METRIC_NAMES)


class _NoDp(InferenceConfig):
    not_implemented = ("server_params.dp",)


class TestNotImplementedFields(unittest.TestCase):
    def test_a_listed_field_with_a_non_default_value_fails_naming_class_and_field(self):
        problems = _problems(_inference_dict(server_params={"model": "m", "tp": 8, "dp": 2}), _NoDp)
        self.assertIn("server_params.dp is not implemented for _NoDp", problems)

    def test_a_listed_field_at_its_default_passes(self):
        config = _NoDp.model_validate(_inference_dict(server_params={"model": "m", "tp": 8, "dp": 1}))
        self.assertEqual(config.server_params.dp, 1)

    def test_a_listed_block_is_compared_with_its_default(self):
        class NoQuantParity(InferenceConfig):
            not_implemented = ("accuracy.quant_parity",)

        NoQuantParity.model_validate(_inference_dict(accuracy={"quant_parity": {"enabled": False}}))
        problems = _problems(_inference_dict(accuracy={"quant_parity": {"enabled": True}}), NoQuantParity)
        self.assertIn("accuracy.quant_parity is not implemented for NoQuantParity", problems)

    def test_a_path_that_names_no_field_fails_when_the_class_is_defined(self):
        with self.assertRaises(TypeError):

            class Typo(InferenceConfig):
                not_implemented = ("server_params.dpp",)


class TestInferenceSweepsAndRuns(unittest.TestCase):
    def test_cell_tp_and_pp_must_match_server_params(self):
        other = "ISL=1024,OSL=1024,TP=4,PP=1,CONC=16"
        problems = _problems(_inference_dict(sweeps={other: {}}, runs=[other]))
        self.assertIn("TP and PP must equal server_params.tp and server_params.pp", problems)

    def test_runs_must_be_listed_once_and_exist_in_sweeps(self):
        self.assertIn("runs repeats", _problems(_inference_dict(runs=[CELL, CELL])))
        self.assertIn("missing from sweeps", _problems(_inference_dict(runs=["ISL=1,OSL=1,TP=8,PP=1,CONC=1"])))
        self.assertIn("at least one", _problems(_inference_dict(runs=[])))

    def test_cell_overrides_drop_comments_and_refuse_harness_flags(self):
        config = InferenceConfig.model_validate(_inference_dict(sweeps={CELL: {"_note": "x", "num_prompts": 64}}))
        self.assertEqual(config.sweeps[CELL], {"num_prompts": 64})
        self.assertIn("['max_concurrency']", _problems(_inference_dict(sweeps={CELL: {"max_concurrency": 4}})))

    def test_enforced_threshold_file_must_cover_every_run(self):
        data = _inference_dict(threshold_json="t.json", thresholds={})
        self.assertIn("runs missing from the threshold file", _problems(data))
        InferenceConfig.model_validate({**data, "enforce_thresholds": False})
        InferenceConfig.model_validate(_inference_dict(thresholds={}))

    def test_custom_eval_ids_are_unique(self):
        niah = {"type": "niah", "id": "n", "isl": 8192}
        self.assertIn("duplicate custom_eval", _problems(_inference_dict(accuracy={"custom_eval": [niah, niah]})))


if __name__ == "__main__":
    unittest.main()
