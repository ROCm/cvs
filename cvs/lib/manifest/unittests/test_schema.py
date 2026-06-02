"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/manifest/schema.py: the Manifest aggregate and its
# submodels.
#
# Pinned invariants:
#   - A fully-populated manifest (incl. a G2 ThresholdVerdict) round-trips
#     through write/read unchanged; never-set fields survive too.
#   - extra="forbid" rejects unknown manifest keys.
#   - B7 (security removed): ConfigInputs has no redacted_* fields and records
#     commands/env verbatim on disk (an inline HF token is NOT scrubbed).
#   - NaN/Inf scalars persist as null so the manifest stays strict-JSON, while a
#     finite scalar is never coerced to None.

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.manifest import ConfigInputs, Manifest, RunLayout

from ._fixtures import _full_manifest


class TestManifestRoundtrip(unittest.TestCase):
    def test_full_roundtrip_with_threshold_verdict(self):
        """Flow 1: durability + G2 ThresholdVerdict cross-group contract."""
        tmp = Path(tempfile.mkdtemp())
        layout = RunLayout(tmp, "suite", "cell-a", "hash1234", "run-1").ensure()
        manifest = _full_manifest()
        manifest.write(layout.manifest_path)
        again = Manifest.read(layout.manifest_path)
        self.assertEqual(again, manifest)
        self.assertEqual(again.verdicts.threshold_verdicts[0].metric, "throughput")
        self.assertTrue(again.verdicts.threshold_verdicts[0].passed)

    def test_rejects_unknown_field(self):
        """Flow 2: extra='forbid' rejects an unknown manifest key."""
        with self.assertRaises(ValidationError):
            Manifest.model_validate(
                {
                    "identity": {"run_id": "r", "test_id": "t", "cell_id": "c"},
                    "config": {},
                    "not_a_real_field": True,
                }
            )

    def test_never_touched_fields_roundtrip(self):
        """Populate rarely-set fields so write/read equality actually guards them."""
        tmp = Path(tempfile.mkdtemp())
        manifest = _full_manifest()
        manifest.schema_version = "9"
        manifest.identity.framework_versions = {"vllm": "0.6.0"}
        manifest.identity.cvs_version = "1.2.3"
        manifest.config.datasets = [{"name": "sharegpt", "sha": "abc"}]
        manifest.verdicts.flags = {"degraded": "true"}
        manifest.verdicts.failure_category = "liveness_timeout"
        manifest.verdicts.skip_reason = "no gpus"
        manifest.resources.oom = True
        manifest.sidecars.trajectory = "trajectory.parquet"
        layout = RunLayout(tmp, "s", "c", "h", "r").ensure()
        manifest.write(layout.manifest_path)
        self.assertEqual(Manifest.read(layout.manifest_path), manifest)


class TestB7NoRedaction(unittest.TestCase):
    def test_redacted_fields_rejected(self):
        """Flow 3a: the removed redacted_* keys are rejected (extra='forbid')."""
        with self.assertRaises(ValidationError):
            ConfigInputs.model_validate({"redacted_env": {"A": "b"}})
        with self.assertRaises(ValidationError):
            ConfigInputs.model_validate({"redacted_commands": ["x"]})

    def test_commands_env_recorded_as_is_on_disk(self):
        """Flow 3b: commands/env persist verbatim through Manifest.write (no redaction at write time)."""
        tmp = Path(tempfile.mkdtemp())
        token = "hf_DEADBEEFsecret"
        manifest = _full_manifest()
        manifest.config.env = {"HF_TOKEN": token}
        manifest.config.commands = [f"docker run -e HF_TOKEN={token} img"]
        layout = RunLayout(tmp, "suite", "cell-a", "h", "run-x").ensure()
        manifest.write(layout.manifest_path)
        # Prove the token survives the write() path verbatim: once in env, once in commands.
        raw = layout.manifest_path.read_text()
        self.assertEqual(raw.count(token), 2)
        reloaded = Manifest.read(layout.manifest_path)
        self.assertEqual(reloaded.config.env["HF_TOKEN"], token)
        self.assertEqual(reloaded.config.commands, [f"docker run -e HF_TOKEN={token} img"])


class TestScalarJsonSafety(unittest.TestCase):
    def test_nonfinite_scalars_are_valid_json(self):
        """A NaN/Inf scalar persists as null so the manifest stays strict-JSON and reads back."""
        tmp = Path(tempfile.mkdtemp())
        manifest = _full_manifest()
        manifest.verdicts.scalars["loss"] = float("nan")
        manifest.verdicts.scalars["tput"] = float("inf")
        manifest.resources.per_host["n0"]["gpu_util"] = float("-inf")
        layout = RunLayout(tmp, "s", "c", "h", "r").ensure()
        manifest.write(layout.manifest_path)
        raw = layout.manifest_path.read_text()

        def _reject(token):
            raise AssertionError(f"non-finite token {token!r} in manifest JSON")

        parsed = json.loads(raw, parse_constant=_reject)  # strict JSON: must not raise
        self.assertIsNone(parsed["verdicts"]["scalars"]["loss"])
        self.assertIsNone(parsed["verdicts"]["scalars"]["tput"])
        self.assertIsNone(parsed["resources"]["per_host"]["n0"]["gpu_util"])
        again = Manifest.read(layout.manifest_path)
        self.assertIsNone(again.verdicts.scalars["loss"])

    def test_finite_scalar_survives_as_float(self):
        """Guard: a finite scalar must NOT be coerced to None."""
        tmp = Path(tempfile.mkdtemp())
        manifest = _full_manifest(throughput=1234.5)
        layout = RunLayout(tmp, "s", "c", "h", "r").ensure()
        manifest.write(layout.manifest_path)
        self.assertEqual(Manifest.read(layout.manifest_path).verdicts.scalars["throughput"], 1234.5)


if __name__ == "__main__":
    unittest.main()
