import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.inference.utils.vllm_config_loader import (
    RunCell,
    VariantConfig,
    _strip_metadata,
    load_variant,
    serialize_cli_options,
)


CELL = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"


def _config(**overrides):
    value = {
        "enforce_thresholds": False,
        "threshold_json": "threshold.json",
        "ib_hca_devices": "auto",
        "ib_netdev": "eth0",
        "paths": {
            "shared_fs": "/home/test",
            "models_dir": "/models",
            "log_dir": "/logs",
            "hf_token_file": "/home/test/.hf_token",
        },
        "container": {"name": "vllm", "image": "image", "runtime": {"name": "docker", "args": {}}},
        "server_params": {
            "model": "/models/DeepSeek-R1-0528-FP8",
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 2,
        },
        "benchmark_params": {"num_prompts": 25},
        "sweeps": {CELL: {"num_prompts": 50}},
        "runs": [CELL],
        "thresholds": {},
    }
    value.update(overrides)
    return value


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


class TestVllmConfigContract(unittest.TestCase):
    def test_resolves_selected_run_with_cell_overrides(self):
        variant = VariantConfig(**_config())
        (run,) = variant.resolved_runs()
        self.assertEqual(run.cell.key, CELL)
        self.assertEqual(run.benchmark_params["num_prompts"], 50)
        self.assertEqual(run.benchmark_params["random_input_len"], 1024)
        self.assertEqual(run.benchmark_params["random_output_len"], 1024)
        self.assertEqual(run.benchmark_params["max_concurrency"], 16)

    def test_rejects_unknown_root_and_legacy_shape(self):
        for kwargs in ({"unexpected": 1}, {"schema_version": 1}, {"params": {}}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValidationError):
                    VariantConfig(**_config(**kwargs))

    def test_requires_nonempty_explicit_runs(self):
        with self.assertRaisesRegex(ValidationError, "nonempty explicit"):
            VariantConfig(**_config(runs=[]))

    def test_rejects_mismatched_parallelism_and_missing_enforced_threshold(self):
        wrong_cell = "ISL=1024,OSL=1024,TP=4,PP=2,CONC=16"
        with self.assertRaisesRegex(ValidationError, "conflicts"):
            VariantConfig(**_config(sweeps={wrong_cell: {}}, runs=[wrong_cell]))
        with self.assertRaisesRegex(ValidationError, "missing threshold"):
            VariantConfig(**_config(enforce_thresholds=True))

    def test_record_only_mode_allows_unmeasured_selected_cells(self):
        self.assertEqual(VariantConfig(**_config()).expected_cells(), [CELL])

    def test_rejects_network_environment_collisions(self):
        config = _config()
        config["container"]["env"] = {"NCCL_IB_HCA": "rdma0"}
        with self.assertRaisesRegex(ValidationError, "generated network"):
            VariantConfig(**config)


class TestMetadataAndOptionSerialization(unittest.TestCase):
    def test_strips_local_comments_but_preserves_option_payload(self):
        value = _strip_metadata(
            {
                "_comment": "root",
                "server_params": {
                    "_comment": "server",
                    "json_arg": {"_comment": "upstream payload"},
                },
                "sweeps": {CELL: {"_comment": "cell", "json_arg": {"_comment": "upstream payload"}}},
                "accuracy": {"tasks": [{"_comment": "task", "id": "gsm8k", "task": "gsm8k"}]},
            }
        )
        self.assertNotIn("_comment", value)
        self.assertEqual(value["server_params"]["json_arg"], {"_comment": "upstream payload"})
        self.assertEqual(value["sweeps"][CELL]["json_arg"], {"_comment": "upstream payload"})
        self.assertNotIn("_comment", value["accuracy"]["tasks"][0])

    def test_serializes_documented_value_shapes(self):
        options = {
            "trust_remote_code": True,
            "max_model_len": 8192,
            "served_model_name": ["first", "second"],
            "structured_outputs": {"json": {"type": "object"}},
            "optional": None,
        }
        self.assertEqual(
            serialize_cli_options(options),
            [
                "--trust-remote-code",
                "--max-model-len",
                "8192",
                "--served-model-name",
                "first",
                "second",
                "--structured-outputs",
                '{"json":{"type":"object"}}',
            ],
        )

    def test_rejects_ambiguous_false_and_reserved_names(self):
        with self.assertRaises(ValueError):
            serialize_cli_options({"enforce_eager": False})
        with self.assertRaisesRegex(ValidationError, "cannot override harness"):
            VariantConfig(
                **_config(
                    server_params={
                        "model": "/models/x",
                        "tensor_parallel_size": 8,
                        "pipeline_parallel_size": 2,
                        "nnodes": 2,
                    }
                )
            )


class TestPackagedVllmCatalog(unittest.TestCase):
    def test_every_config_resolves_complete_run_contract(self):
        root = Path(__file__).resolve().parents[3] / "input" / "config_file" / "inference" / "vllm"
        configs = sorted(path for path in root.glob("*.json") if not path.name.endswith("threshold.json"))
        self.assertEqual(len(configs), 28)
        for path in configs:
            with self.subTest(config=path.name):
                variant = load_variant(path, {"username": "test"})
                runs = variant.resolved_runs()
                self.assertTrue(runs)
                self.assertEqual([run.cell.key for run in runs], variant.runs)
                self.assertTrue(set(variant.runs) <= set(variant.sweeps))
                self.assertTrue(set(variant.thresholds) - {"accuracy"} <= set(variant.sweeps))
                for run in runs:
                    self.assertEqual(run.cell.tp, variant.server_params.tensor_parallel_size)
                    self.assertEqual(run.cell.pp, variant.server_params.pipeline_parallel_size)


if __name__ == "__main__":
    unittest.main()
