import json
import tempfile
import unittest
from pathlib import Path

from cvs.cli_plugins.list_plugin import ListPlugin
from cvs.lib.config_catalog import ConfigCatalog, ConfigCatalogError, load_config_catalog


class TestConfigCatalog(unittest.TestCase):
    @staticmethod
    def _suite_names():
        return {suite for tests in ListPlugin.discover_tests().values() for suite in tests}

    def test_bundled_catalog_covers_runnable_suites(self):
        catalog = load_config_catalog()
        catalog.validate_suites(self._suite_names())

    def test_bundled_catalog_returns_specific_and_shared_configs(self):
        catalog = load_config_catalog()

        vllm_configs = catalog.configurations_for("vllm_gpt_oss_120b_single", "mi355")
        self.assertEqual([config.path for config in vllm_configs], ["inference/vllm/mi355x_vllm_single.json"])

        rccl_configs = catalog.configurations_for("rccl_perf", "mi355")
        self.assertEqual([config.path for config in rccl_configs], ["rccl/rccl_config.json"])

    def test_unknown_platform_has_a_clear_error(self):
        catalog = load_config_catalog()
        with self.assertRaisesRegex(ConfigCatalogError, "Unknown platform 'mi35x'"):
            catalog.configurations_for("rccl_perf", "mi35x")

    def test_catalog_rejects_missing_config_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_root = Path(temporary_directory)
            catalog_path = input_root / "config_catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "configurations": [
                            {"path": "rccl/missing.json", "platforms": ["all"], "suites": ["rccl_perf"]}
                        ],
                        "unavailable_suites": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigCatalogError, "does not exist"):
                ConfigCatalog.from_input_roots([input_root])

    def test_catalog_rejects_missing_suite_coverage(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_root = Path(temporary_directory)
            config_path = input_root / "config_file" / "rccl" / "rccl_config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("{}", encoding="utf-8")
            (input_root / "config_catalog.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "configurations": [
                            {"path": "rccl/rccl_config.json", "platforms": ["all"], "suites": ["rccl_perf"]}
                        ],
                        "unavailable_suites": [],
                    }
                ),
                encoding="utf-8",
            )
            catalog = ConfigCatalog.from_input_roots([input_root])

            with self.assertRaisesRegex(ConfigCatalogError, "missing suites"):
                catalog.validate_suites({"rccl_perf", "rccl_regression"})

    def test_catalog_rejects_unlisted_bundled_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_root = Path(temporary_directory)
            for relative_path in ("rccl/rccl_config.json", "rccl/extra_config.json"):
                config_path = input_root / "config_file" / relative_path
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text("{}", encoding="utf-8")
            (input_root / "config_catalog.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "configurations": [
                            {"path": "rccl/rccl_config.json", "platforms": ["all"], "suites": ["rccl_perf"]}
                        ],
                        "unavailable_suites": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigCatalogError, "missing config files"):
                ConfigCatalog.from_input_roots([input_root])

    def test_catalog_merges_optional_extension_catalog(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            core_input = Path(temporary_directory) / "core" / "input"
            extension_input = Path(temporary_directory) / "extension" / "input"
            for input_root, relative_path, suite in (
                (core_input, "rccl/rccl_config.json", "rccl_perf"),
                (extension_input, "health/health_config.json", "extension_health"),
            ):
                config_path = input_root / "config_file" / relative_path
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text("{}", encoding="utf-8")
                (input_root / "config_catalog.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "configurations": [{"path": relative_path, "platforms": ["all"], "suites": [suite]}],
                            "unavailable_suites": [],
                        }
                    ),
                    encoding="utf-8",
                )

            catalog = ConfigCatalog.from_input_roots([core_input, extension_input])

        self.assertEqual(
            [configuration.path for configuration in catalog.configurations],
            ["rccl/rccl_config.json", "health/health_config.json"],
        )
