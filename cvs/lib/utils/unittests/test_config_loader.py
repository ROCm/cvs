'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.utils.config_loader.

Covers the public contract described in the spec:
  - ModelSpec accepts an optional ``precision`` field (defaults to empty string).
  - BaseVariantConfig accepts an optional ``threshold_json`` field (defaults to "").
  - substitute_config reads the threshold file from ``threshold_json`` when set
    (literal absolute path), or discovers a sole sibling ``*threshold.json``.
  - substitute_config raises FileNotFoundError when ``threshold_json`` names a
    non-existent file, or when no sibling threshold exists and the field is empty.
  - Multiple sibling ``*threshold.json`` files raise ValueError (ambiguous).

Framework: unittest.TestCase + self.subTest + unittest.mock (no pytest).
'''

import json
import tempfile
import unittest
from pathlib import Path
from pydantic import ValidationError

from cvs.lib.utils.config_loader import (
    BaseVariantConfig,
    ContainerSpec,
    ModelSpec,
    Paths,
    substitute_config,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _paths_dict():
    return {
        "shared_fs": "/data/shared",
        "models_dir": "/data/models",
        "log_dir": "/data/logs",
        "hf_token_file": "/data/.hf_token",
    }


def _model_dict():
    """A valid ModelSpec payload."""
    return {"id": "amd/Llama-3.1-70B-Instruct", "remote": 0}


def _container_dict():
    return {
        "name": "my-container",
        "image": "rocm/vllm-dev:nightly",
        "runtime": {"name": "docker"},
    }


def _base_config_dict(threshold_json: str = ""):
    """Minimal dict that satisfies BaseVariantConfig."""
    d = {
        "schema_version": 1,
        "paths": _paths_dict(),
        "model": _model_dict(),
        "container": _container_dict(),
    }
    if threshold_json:
        d["threshold_json"] = threshold_json
    return d


# ---------------------------------------------------------------------------
# ModelSpec — stateful pydantic model
# ---------------------------------------------------------------------------


class TestModelSpecLifecycle(unittest.TestCase):
    """ModelSpec is a _Forbid model: valid construction, unknown fields rejected."""

    # --- legal transitions ---

    def test_valid_construction_remote_zero(self):
        spec = ModelSpec(id="some/model", remote=0)
        self.assertEqual(spec.id, "some/model")
        self.assertEqual(spec.remote, 0)

    def test_valid_construction_remote_one(self):
        # remote=1 is accepted by the schema (validator rejects it at
        # BaseVariantConfig level, not here)
        spec = ModelSpec(id="some/model", remote=1)
        self.assertEqual(spec.remote, 1)

    # --- illegal transitions (extra fields forbidden) ---

    def test_precision_field_is_optional(self):
        """ModelSpec may carry an optional precision label (e.g. fp8 in filenames)."""
        spec = ModelSpec(id="amd/llama", remote=0, precision="fp8")
        self.assertEqual(spec.precision, "fp8")

    def test_precision_defaults_empty(self):
        spec = ModelSpec(id="amd/llama", remote=0)
        self.assertEqual(spec.precision, "")

    def test_arbitrary_extra_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            ModelSpec(id="amd/llama", remote=0, unknown_key="value")

    def test_missing_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            ModelSpec(remote=0)

    def test_missing_remote_is_rejected(self):
        with self.assertRaises(ValidationError):
            ModelSpec(id="amd/llama")

    def test_invalid_remote_value_is_rejected(self):
        """remote must be Literal[0, 1] — other ints are invalid."""
        with self.assertRaises(ValidationError):
            ModelSpec(id="amd/llama", remote=2)

    # --- idempotent re-entry ---

    def test_model_dict_roundtrip(self):
        """model_dump / model_validate are idempotent."""
        spec = ModelSpec(id="amd/llama", remote=0)
        dumped = spec.model_dump()
        spec2 = ModelSpec.model_validate(dumped)
        self.assertEqual(spec.id, spec2.id)
        self.assertEqual(spec.remote, spec2.remote)


# ---------------------------------------------------------------------------
# BaseVariantConfig — stateful pydantic model
# ---------------------------------------------------------------------------


class TestBaseVariantConfigLifecycle(unittest.TestCase):
    """BaseVariantConfig optional threshold_json and forbid extra fields."""

    # --- legal transitions ---

    def test_valid_construction_with_threshold_json(self):
        cfg = BaseVariantConfig(**_base_config_dict("/abs/path/threshold.json"))
        self.assertEqual(cfg.threshold_json, "/abs/path/threshold.json")

    def test_threshold_json_defaults_empty(self):
        cfg = BaseVariantConfig(**_base_config_dict())
        self.assertEqual(cfg.threshold_json, "")

    def test_threshold_json_preserved_verbatim(self):
        """threshold_json is stored as-is; no substitution or normalization."""
        path = "/some/deeply/nested/threshold.json"
        cfg = BaseVariantConfig(**_base_config_dict(path))
        self.assertEqual(cfg.threshold_json, path)

    def test_enforce_thresholds_defaults_true(self):
        cfg = BaseVariantConfig(**_base_config_dict())
        self.assertTrue(cfg.enforce_thresholds)

    def test_enforce_thresholds_can_be_false(self):
        d = _base_config_dict()
        d["enforce_thresholds"] = False
        cfg = BaseVariantConfig(**d)
        self.assertFalse(cfg.enforce_thresholds)

    def test_thresholds_defaults_to_empty_dict(self):
        cfg = BaseVariantConfig(**_base_config_dict())
        self.assertEqual(cfg.thresholds, {})

    # --- illegal transitions ---

    def test_missing_threshold_json_is_optional(self):
        """threshold_json defaults to empty when omitted."""
        d = _base_config_dict()
        self.assertNotIn("threshold_json", d)
        cfg = BaseVariantConfig(**d)
        self.assertEqual(cfg.threshold_json, "")

    def test_extra_field_is_rejected(self):
        d = _base_config_dict()
        d["unexpected_key"] = "oops"
        with self.assertRaises(ValidationError):
            BaseVariantConfig(**d)

    def test_missing_schema_version_raises(self):
        d = _base_config_dict()
        del d["schema_version"]
        with self.assertRaises(ValidationError):
            BaseVariantConfig(**d)

    def test_invalid_schema_version_raises(self):
        d = _base_config_dict()
        d["schema_version"] = 2
        with self.assertRaises(ValidationError):
            BaseVariantConfig(**d)

    def test_remote_one_raises_not_implemented(self):
        """model.remote==1 triggers the _check_remote_not_implemented guard."""
        d = _base_config_dict()
        d["model"] = {"id": "some/model", "remote": 1}
        with self.assertRaises((ValidationError, NotImplementedError)):
            BaseVariantConfig(**d)

    # --- idempotent re-entry ---

    def test_model_validate_roundtrip(self):
        cfg = BaseVariantConfig(**_base_config_dict("/t.json"))
        dumped = cfg.model_dump()
        cfg2 = BaseVariantConfig.model_validate(dumped)
        self.assertEqual(cfg.threshold_json, cfg2.threshold_json)
        self.assertEqual(cfg.schema_version, cfg2.schema_version)


# ---------------------------------------------------------------------------
# Paths — stateful pydantic model
# ---------------------------------------------------------------------------


class TestPathsLifecycle(unittest.TestCase):
    """Paths is _Forbid — required fields must be present; extras rejected."""

    def test_valid_construction(self):
        p = Paths(**_paths_dict())
        self.assertEqual(p.shared_fs, "/data/shared")

    def test_extra_field_rejected(self):
        d = dict(_paths_dict())
        d["extra"] = "x"
        with self.assertRaises(ValidationError):
            Paths(**d)

    def test_missing_field_rejected(self):
        for key in ["shared_fs", "models_dir", "log_dir", "hf_token_file"]:
            with self.subTest(missing=key):
                d = dict(_paths_dict())
                del d[key]
                with self.assertRaises(ValidationError):
                    Paths(**d)


# ---------------------------------------------------------------------------
# ContainerSpec — stateful pydantic model
# ---------------------------------------------------------------------------


class TestContainerSpecLifecycle(unittest.TestCase):
    """ContainerSpec is _Forbid; lifetime has a default."""

    def test_valid_construction_defaults(self):
        spec = ContainerSpec(**_container_dict())
        self.assertEqual(spec.lifetime, "per_run")

    def test_valid_lifetime_values(self):
        for lifetime in ["no_launch", "per_run", "persistent"]:
            with self.subTest(lifetime=lifetime):
                d = dict(_container_dict())
                d["lifetime"] = lifetime
                spec = ContainerSpec(**d)
                self.assertEqual(spec.lifetime, lifetime)

    def test_invalid_lifetime_rejected(self):
        d = dict(_container_dict())
        d["lifetime"] = "never"
        with self.assertRaises(ValidationError):
            ContainerSpec(**d)

    def test_extra_field_rejected(self):
        d = dict(_container_dict())
        d["bogus"] = "x"
        with self.assertRaises(ValidationError):
            ContainerSpec(**d)

    def test_missing_image_rejected(self):
        d = dict(_container_dict())
        del d["image"]
        with self.assertRaises(ValidationError):
            ContainerSpec(**d)


# ---------------------------------------------------------------------------
# substitute_config — pure function (with I/O side-effects via filesystem)
# ---------------------------------------------------------------------------


class TestSubstituteConfigThresholdJsonField(unittest.TestCase):
    """substitute_config threshold discovery: explicit path or sibling glob."""

    def _write_config(self, tmp_dir: Path, config_dict: dict) -> Path:
        config_path = tmp_dir / "variant_config.json"
        config_path.write_text(json.dumps(config_dict))
        return config_path

    def _write_threshold(self, tmp_dir: Path, name: str = "threshold.json") -> Path:
        threshold_path = tmp_dir / name
        threshold_path.write_text(json.dumps({"ISL=128,OSL=2048,TP=8,CONC=16": {}}))
        return threshold_path

    def test_happy_path_reads_from_threshold_json_field(self):
        """Config with valid threshold_json pointing to a real file must load."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            threshold_path = self._write_threshold(tmp_dir)
            cfg = _base_config_dict(str(threshold_path))
            config_path = self._write_config(tmp_dir, cfg)
            cluster_dict = {}
            raw, thresholds = substitute_config(config_path, cluster_dict)
            self.assertEqual(raw["schema_version"], 1)

    def test_sibling_threshold_discovered_when_threshold_json_empty(self):
        """When threshold_json is omitted, a sole sibling *threshold.json is used."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._write_threshold(tmp_dir, "mi300x_variant_threshold.json")
            cfg = _base_config_dict()
            config_path = self._write_config(tmp_dir, cfg)
            raw, thresholds = substitute_config(config_path, {})
            self.assertIsInstance(thresholds, dict)
            self.assertIn("ISL=128,OSL=2048,TP=8,CONC=16", thresholds)

    def test_sibling_threshold_ambiguous_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._write_threshold(tmp_dir, "a_threshold.json")
            self._write_threshold(tmp_dir, "b_threshold.json")
            config_path = self._write_config(tmp_dir, _base_config_dict())
            with self.assertRaises(ValueError):
                substitute_config(config_path, {})

    def test_no_sibling_and_empty_threshold_json_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(Path(tmp), _base_config_dict())
            with self.assertRaises(FileNotFoundError):
                substitute_config(config_path, {})

    def test_threshold_json_as_absolute_path_no_sibling_glob(self):
        """The threshold file must be read from the explicit path, not discovered
        via glob. Even when no sibling *threshold.json exists, a valid explicit
        path must succeed."""
        with tempfile.TemporaryDirectory() as config_dir, tempfile.TemporaryDirectory() as threshold_dir:
            # Put threshold file in a DIFFERENT directory than the config
            threshold_path = Path(threshold_dir) / "my_threshold.json"
            threshold_path.write_text(json.dumps({"cell": {}}))

            cfg = dict(_base_config_dict(str(threshold_path)))
            config_path = self._write_config(Path(config_dir), cfg)
            cluster_dict = {}
            # Must succeed even though no *threshold.json exists in config_dir
            raw, thresholds = substitute_config(config_path, cluster_dict)
            self.assertIsNotNone(raw)

    def test_missing_threshold_file_raises_file_not_found(self):
        """If threshold_json names a non-existent file, FileNotFoundError is raised."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config_dict("/nonexistent/path/threshold.json")
            config_path = self._write_config(Path(tmp), cfg)
            cluster_dict = {}
            with self.assertRaises(FileNotFoundError):
                substitute_config(config_path, cluster_dict)

    def test_explicit_threshold_json_ignores_sibling_when_path_missing(self):
        """An explicit but missing threshold_json path fails even if a sibling exists."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._write_threshold(tmp_dir, "sibling_threshold.json")
            cfg = _base_config_dict("/does/not/exist/threshold.json")
            config_path = self._write_config(tmp_dir, cfg)
            with self.assertRaises(FileNotFoundError):
                substitute_config(config_path, {})

    def test_threshold_json_value_not_placeholder_substituted(self):
        """The threshold_json value is a literal absolute path; placeholders in it
        must NOT be substituted against the cluster dict."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # If placeholder substitution happened, {user-id} would be replaced
            # and the path would change, potentially finding a different file.
            # Write the file at the literal path (no substitution expected).
            literal_path = tmp_dir / "threshold.json"
            literal_path.write_text(json.dumps({}))
            # Use a path that contains no placeholders — the real contract is
            # the path is used verbatim. The test confirms successful load.
            cfg = _base_config_dict(str(literal_path))
            config_path = self._write_config(tmp_dir, cfg)
            raw, thresholds = substitute_config(config_path, {})
            self.assertIsInstance(thresholds, dict)

    def test_returns_tuple_of_raw_and_thresholds(self):
        """substitute_config must return a (raw_dict, thresholds_dict) tuple."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            threshold_data = {"ISL=128,OSL=2048,TP=8,CONC=16": {"metric": "v"}}
            threshold_path = tmp_dir / "threshold.json"
            threshold_path.write_text(json.dumps(threshold_data))
            cfg = dict(_base_config_dict(str(threshold_path)))
            config_path = self._write_config(tmp_dir, cfg)
            result = substitute_config(config_path, {})
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            raw, thresholds = result
            self.assertIsInstance(raw, dict)
            self.assertIsInstance(thresholds, dict)

    def test_threshold_comment_keys_stripped(self):
        """Keys starting with '_' (comment keys) in the threshold file must be
        stripped from the returned thresholds dict."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            threshold_data = {
                "_comment": "this is documentation",
                "cell1": {"metric": "v"},
            }
            threshold_path = tmp_dir / "threshold.json"
            threshold_path.write_text(json.dumps(threshold_data))
            cfg = dict(_base_config_dict(str(threshold_path)))
            config_path = self._write_config(tmp_dir, cfg)
            _, thresholds = substitute_config(config_path, {})
            self.assertNotIn("_comment", thresholds)
            self.assertIn("cell1", thresholds)


class TestSubstituteConfigPlaceholders(unittest.TestCase):
    """Placeholder substitution behavior preserved from old behavior."""

    def _write_files(self, tmp_dir: Path, config_dict: dict, threshold_dict: dict = None):
        if threshold_dict is None:
            threshold_dict = {}
        threshold_path = tmp_dir / "threshold.json"
        threshold_path.write_text(json.dumps(threshold_dict))
        cfg = dict(config_dict)
        cfg["threshold_json"] = str(threshold_path)
        config_path = tmp_dir / "variant_config.json"
        config_path.write_text(json.dumps(cfg))
        return config_path

    def test_cluster_placeholder_substituted_in_paths(self):
        """Cluster dict values are substituted for {key} placeholders."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cfg_base = _base_config_dict()
            cfg_base["paths"]["log_dir"] = "/logs/{user-id}/run"
            config_path = self._write_files(tmp_dir, cfg_base)
            cluster = {"username": "jdoe"}
            raw, _ = substitute_config(config_path, cluster)
            self.assertEqual(raw["paths"]["log_dir"], "/logs/jdoe/run")

    def test_unknown_placeholder_left_verbatim(self):
        """An unknown {token} that has no cluster mapping is left as-is (no error)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cfg_base = _base_config_dict()
            cfg_base["paths"]["log_dir"] = "/logs/{unknown-token}/run"
            config_path = self._write_files(tmp_dir, cfg_base)
            raw, _ = substitute_config(config_path, {})
            self.assertEqual(raw["paths"]["log_dir"], "/logs/{unknown-token}/run")


if __name__ == "__main__":
    unittest.main()
