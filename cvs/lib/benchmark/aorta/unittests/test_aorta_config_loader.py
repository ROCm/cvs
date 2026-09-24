"""Aorta schema and shared-loader regression tests."""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig, load_training_variant
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.lib.utils.config_loader import BaseVariantConfig


class TestAortaConfig(unittest.TestCase):
    def setUp(self):
        self.raw = variant_dict()

    def test_shared_schema_and_defaults(self):
        config = AortaVariantConfig.model_validate(self.raw)
        self.assertIsInstance(config, BaseVariantConfig)
        self.assertEqual(config.multi_node.master_launch_mode, "auto")
        self.assertTrue(config.multi_node.collect_traces)
        self.assertEqual(config.container.model_dump()["runtime"]["args"]["ipc"], "host")

    def test_unknown_fields_and_invalid_mode_port_rejected(self):
        for block in ({"bogus": True}, {"master_launch_mode": "magic"}, {"master_port": 80}, {"master_port": 65536}):
            with self.subTest(block=block):
                self.raw["multi_node"] = block
                with self.assertRaises(ValidationError):
                    AortaVariantConfig.model_validate(self.raw)

    def test_optional_launch_settings_and_extra_env(self):
        self.raw["multi_node"] = {
            "master_launch_mode": "torchrun",
            "master_port": 29500,
            "nproc_per_node": 4,
            "master_addr": "fabric-head",
            "extra_env": {"CUSTOM": 1},
        }
        self.assertEqual(AortaVariantConfig.model_validate(self.raw).multi_node.nproc_per_node, 4)

    def test_missing_gpu_count_rejected(self):
        del self.raw["gpus_per_node"]
        with self.assertRaises(ValidationError):
            AortaVariantConfig.model_validate(self.raw)

    def test_gpus_per_node_placeholder_reports_clear_error(self):
        self.raw["gpus_per_node"] = "<changeme>"
        with self.assertRaisesRegex(ValidationError, "Unresolved configuration placeholder"):
            AortaVariantConfig.model_validate(self.raw)

    def test_placeholders_rejected_in_nested_values(self):
        for value in ("<changeme>", "<CHANGEME>", "{paths.unknown}"):
            self.raw["multi_node"] = {"extra_env": {"NCCL_IB_HCA": value}}
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AortaVariantConfig.model_validate(self.raw)

    def test_opaque_brace_values_are_not_treated_as_placeholders(self):
        self.raw["multi_node"] = {
            "extra_train_args": ["--allocator={128}"],
            "extra_env": {"ALLOCATOR_CONFIG": "buckets:{128}"},
        }
        config = AortaVariantConfig.model_validate(self.raw)
        self.assertEqual(config.multi_node.extra_train_args, ["--allocator={128}"])

    def test_unknown_placeholder_rejected_in_structured_path(self):
        self.raw["paths"]["shared_fs"] = "/scratch/{bogus}"
        with self.assertRaisesRegex(ValidationError, "Unresolved configuration placeholder"):
            AortaVariantConfig.model_validate(self.raw)

    def test_paths_and_writable_mount_validated(self):
        for key, value in (
            ("aorta_path", "relative"),
            ("container_mount_path", "/"),
            ("base_config", "../escape.yaml"),
            ("build_script", "/absolute.sh"),
        ):
            raw = variant_dict()
            raw[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                AortaVariantConfig.model_validate(raw)
        self.raw["container"]["runtime"]["args"]["volumes"] = ["/scratch/tester/repo:/mnt:ro"]
        with self.assertRaisesRegex(ValidationError, "writable mount"):
            AortaVariantConfig.model_validate(self.raw)
        for suffix in ("rw,Z", "rw,cached", "Z"):
            raw = variant_dict()
            raw["container"]["runtime"]["args"]["volumes"] = [f"/scratch/tester/repo:/mnt:{suffix}"]
            with self.subTest(suffix=suffix):
                AortaVariantConfig.model_validate(raw)

    def test_writable_mount_tolerates_redundant_path_separators(self):
        raw = variant_dict()
        raw["container"]["runtime"]["args"]["volumes"] = [f"{raw['aorta_path']}/:/mnt"]
        AortaVariantConfig.model_validate(raw)

    def test_clone_requires_url(self):
        self.raw["aorta_auto_clone"] = True
        with self.assertRaisesRegex(ValidationError, "aorta_clone_url"):
            AortaVariantConfig.model_validate(self.raw)

    def test_invalid_environment_and_arguments(self):
        for block in (
            {"extra_env": {"BAD-NAME": "1"}},
            {"extra_env": {"NCCL_MAX_NCHANNELS": "257"}},
            {"extra_train_args": [1]},
        ):
            self.raw["multi_node"] = block
            with self.subTest(block=block), self.assertRaises(ValidationError):
                AortaVariantConfig.model_validate(self.raw)

    def test_thresholds_reject_typos_ranges_and_non_finite_values(self):
        for expected in (
            {"unknown": 1},
            {"min_compute_ratio": 1.1},
            {"min_overlap_ratio": -0.1},
            {"max_avg_iteration_ms": float("nan")},
            {"max_time_variance_ratio": "bad"},
        ):
            self.raw["thresholds"] = {"expected_results": expected}
            with self.subTest(expected=expected), self.assertRaises(ValidationError):
                AortaVariantConfig.model_validate(self.raw)

    def test_empty_thresholds_require_record_only(self):
        self.raw["thresholds"] = {}
        with self.assertRaisesRegex(ValidationError, "at least one"):
            AortaVariantConfig.model_validate(self.raw)
        self.raw["enforce_thresholds"] = False
        self.assertFalse(AortaVariantConfig.model_validate(self.raw).enforce_thresholds)

    def test_loader_substitution_and_threshold_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "variant.json"
            thresholds = self.raw.pop("thresholds")
            self.raw["threshold_json"] = "test_threshold.json"
            self.raw["paths"]["shared_fs"] = "/scratch/{user-id}"
            self.raw["paths"]["log_dir"] = "{shared_fs}/logs"
            self.raw["aorta_path"] = "{paths.shared_fs}/repo"
            self.raw["container"]["runtime"]["args"]["volumes"] = ["{paths.shared_fs}/repo:/mnt"]
            path.write_text(json.dumps(self.raw))
            (Path(tmp) / "test_threshold.json").write_text(json.dumps(thresholds))
            config = load_training_variant(path, {"username": "alice"})
            self.assertEqual(config.aorta_path, "/scratch/alice/repo")
            self.assertEqual(config.paths.log_dir, "/scratch/alice/logs")
            self.assertEqual(config.thresholds, thresholds)

    def test_shipped_variants_load_after_required_customization(self):
        root = Path(__file__).resolve().parents[4] / "input/config_file/benchmark/aorta"
        for mode in ("single", "distributed"):
            source = root / f"mi3xx_aorta_profile_overlap_2gpu_{mode}.json"
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                raw = json.loads(source.read_text())
                raw["paths"]["shared_fs"] = "/scratch/{user-id}"
                raw["gpus_per_node"] = 2
                raw["multi_node"]["extra_env"] = {}
                raw["threshold_json"] = str(root / raw["threshold_json"])
                target = Path(tmp) / "variant.json"
                target.write_text(json.dumps(raw))
                config = load_training_variant(target, {"username": "tester"})
                self.assertEqual(config.gpus_per_node, 2)
                self.assertNotIn("shm_size", config.container.runtime.args)
