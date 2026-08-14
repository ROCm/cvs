"""Drift guard for the cvs man config parameter registry.

These tests are what stop the documentation rotting: a parameter cannot be
added to a documented config without a description, and a key cannot appear in
a shipped sample without a matching schema field.
"""

import json
import os
import unittest

import cvs
from cvs.cli_plugins.list_plugin import ListPlugin
from cvs.lib.man_lib import iter_parameters
from cvs.parsers.config_registry import TEST_CONFIG_DOCS, resolve_sample_path

DOC_KEY_PREFIXES = ("_comment", "_example")

MIGRATED_SAMPLES = (
    "input/config_file/rccl/rccl_config.json",
    "input/config_file/health/mi300_health_config.json",
    "input/config_file/preflight/preflight_config.json",
    "input/config_file/training/megatron/mi3xx_megatron_llama_distributed.json",
    "input/config_file/training/megatron/mi3xx_megatron_llama_single.json",
    "input/config_file/training/megatron/mi35x_megatron_llama_single.json",
)

# rccl_config.json ships a results block at the rccl root, but rccl_lib.py only
# ever reads cvs_params.results, so the schema documents it there. Tracked
# separately as a functional bug; documenting the dead location would mislead.
KNOWN_UNDECLARED_SAMPLE_KEYS = {("rccl", "results")}


def _cvs_path(relative):
    return os.path.join(os.path.dirname(cvs.__file__), relative)


def _strip_doc_keys(node):
    if isinstance(node, dict):
        return {
            key: _strip_doc_keys(value)
            for key, value in node.items()
            if not (isinstance(key, str) and key.startswith(DOC_KEY_PREFIXES))
        }
    if isinstance(node, list):
        return [_strip_doc_keys(item) for item in node]
    return node


def _load_section(sample_relpath, section_key):
    """Return a sample's top-level section with doc keys stripped, or None if absent."""
    with open(_cvs_path(sample_relpath)) as handle:
        raw = json.load(handle)
    if section_key not in raw:
        return None
    return _strip_doc_keys(raw[section_key])


class TestConfigRegistry(unittest.TestCase):
    def test_resolve_sample_path_points_at_a_real_file(self):
        """The path cvs man prints to the user must exist on disk, not just parse."""
        for doc in TEST_CONFIG_DOCS.values():
            for sample in doc.samples:
                resolved = resolve_sample_path(sample)
                self.assertTrue(os.path.isfile(resolved), f"{resolved} does not exist")

    def test_registered_tests_exist(self):
        """Every documented test name is a real, discoverable test suite."""
        discovered = set()
        for tests in ListPlugin.discover_tests().values():
            discovered.update(tests)

        for test_name in TEST_CONFIG_DOCS:
            self.assertIn(test_name, discovered, f"'{test_name}' has a man page but is not a discoverable test")

    def test_samples_exist(self):
        for test_name, doc in TEST_CONFIG_DOCS.items():
            self.assertTrue(doc.samples, f"'{test_name}' declares no sample config")
            for sample in doc.samples:
                self.assertTrue(os.path.isfile(_cvs_path(sample)), f"missing sample {sample} for '{test_name}'")

    def test_every_parameter_is_documented(self):
        """No parameter may be added to a documented config without a description."""
        for test_name, doc in TEST_CONFIG_DOCS.items():
            for section in doc.sections:
                for param in iter_parameters(section.model, prefix=section.key):
                    self.assertTrue(
                        param.description.strip(),
                        f"{test_name}: parameter '{param.path}' has no description",
                    )

    def test_summaries_are_present(self):
        for test_name, doc in TEST_CONFIG_DOCS.items():
            self.assertTrue(doc.summary.strip(), f"'{test_name}' has no summary")

    def test_samples_validate_against_their_schema(self):
        """A shipped sample must satisfy the schema that documents it."""
        for test_name, doc in TEST_CONFIG_DOCS.items():
            for sample in doc.samples:
                for section in doc.sections:
                    payload = _load_section(sample, section.key)
                    if payload is None:
                        continue
                    with self.subTest(test=test_name, sample=sample, section=section.key):
                        section.model.model_validate(payload)

    def test_sample_keys_have_schema_fields(self):
        """Every key a user sees in a sample is explained by cvs man."""
        for test_name, doc in TEST_CONFIG_DOCS.items():
            for sample in doc.samples:
                for section in doc.sections:
                    payload = _load_section(sample, section.key)
                    if not isinstance(payload, dict):
                        continue
                    declared = {field.alias or name for name, field in section.model.model_fields.items()}
                    for key in payload:
                        if (section.key, key) in KNOWN_UNDECLARED_SAMPLE_KEYS:
                            continue
                        with self.subTest(test=test_name, sample=sample, section=section.key):
                            self.assertIn(
                                key,
                                declared,
                                f"{sample}: '{section.key}.{key}' is not declared in "
                                f"{section.model.__name__}, so cvs man cannot explain it",
                            )

    def test_migrated_samples_carry_no_doc_keys(self):
        """Documentation lives in the schemas now, not inside the config files."""

        def find_doc_keys(node, path=""):
            found = []
            if isinstance(node, dict):
                for key, value in node.items():
                    where = f"{path}.{key}" if path else key
                    if isinstance(key, str) and key.startswith(DOC_KEY_PREFIXES):
                        found.append(where)
                    found.extend(find_doc_keys(value, where))
            elif isinstance(node, list):
                for index, item in enumerate(node):
                    found.extend(find_doc_keys(item, f"{path}[{index}]"))
            return found

        for sample in MIGRATED_SAMPLES:
            with open(_cvs_path(sample)) as handle:
                raw = json.load(handle)
            leftover = find_doc_keys(raw)
            self.assertEqual([], leftover, f"{sample} still carries documentation keys: {leftover}")


if __name__ == "__main__":
    unittest.main()
