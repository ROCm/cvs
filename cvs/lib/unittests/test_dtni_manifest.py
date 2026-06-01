"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from cvs.lib.config.thresholds import ThresholdVerdict
from cvs.lib.manifest import (
    ConfigInputs,
    EventWriter,
    HostFingerprint,
    Identity,
    Manifest,
    PatternMatch,
    PhaseTiming,
    ResourceSummary,
    RunLayout,
    SidecarPointers,
    SystemFingerprint,
    Verdicts,
    read_samples,
    read_trajectory,
    write_resolved_config,
    write_samples,
    write_trajectory,
)
from cvs.lib.manifest.events import EVENT_VOCAB, UnknownEventError
from cvs.lib.manifest.export import FACT_COLUMNS, _flatten_manifest, collect_manifests, export_runs


def _full_manifest(run_id: str = "run-1", **scalars) -> Manifest:
    """A manifest with every submodel populated, incl. a real G2 verdict."""
    return Manifest(
        identity=Identity(
            run_id=run_id,
            test_id="suite",
            cell_id="cell-a",
            config_hash="ch",
            workload_hash="wh",
            verification_hash="vh",
            cvs_git_sha="abc123",
            started_at="2025-01-01T00:00:00+00:00",
            finished_at="2025-01-01T00:01:00+00:00",
            invoker="tester",
        ),
        system=SystemFingerprint(
            hosts=[HostFingerprint(hostname="n0", gpus=["mi300x"] * 8, nics=["mlx5_0"])],
            topology_hash="th",
        ),
        config=ConfigInputs(
            resolved_config_path="/run/config.resolved.yaml",
            model="llama-3.1-70b",
            env={"HF_TOKEN": "hf_plaintext_token"},
            commands=["docker run -e HF_TOKEN=hf_plaintext_token ..."],
            seed=7,
        ),
        phases=[PhaseTiming(phase="prepare", duration_s=1.5, status="complete")],
        verdicts=Verdicts(
            overall_status="complete",
            threshold_verdicts=[
                ThresholdVerdict(
                    threshold_type="rate", metric="throughput", op=">=", expected=1200.0, actual=1500.0, passed=True
                ),
            ],
            pattern_matches=[PatternMatch(id="xgmi_err", severity="fatal", line="bad", node="n0", source="dmesg")],
            scalars={"total_throughput": 1500.0, **{k: float(v) for k, v in scalars.items()}},
        ),
        resources=ResourceSummary(per_host={"n0": {"gpu_util": 0.97}}, oom=False),
        sidecars=SidecarPointers(samples="samples.parquet", logs_dir="logs"),
    )


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


class TestLayout(unittest.TestCase):
    def test_local_paths_and_ensure(self):
        """Flow 4: content-addressable root + ensure() creates dirs."""
        tmp = Path(tempfile.mkdtemp())
        layout = RunLayout(tmp, "suite", "cell-a", "h123", "run-1")
        self.assertEqual(layout.root, tmp / "suite" / "cell-a" / "h123" / "run-1")
        self.assertEqual(layout.manifest_path.name, RunLayout.MANIFEST)
        self.assertEqual(layout.samples_path.name, RunLayout.SAMPLES)
        self.assertEqual(layout.events_path.name, RunLayout.EVENTS)
        self.assertFalse(layout.root.exists())
        layout.ensure()
        self.assertTrue(layout.root.is_dir())
        self.assertTrue(layout.logs_dir.is_dir())

    def test_remote_root_mirror(self):
        """Flow 5: A1 remote run-root mirrors the suffix; to_remote re-bases."""
        tmp = Path(tempfile.mkdtemp())
        layout = RunLayout(tmp, "suite", "cell-a", "h123", "run-1", remote_artifact_dir="/remote/artifacts")
        self.assertEqual(layout.remote_root, Path("/remote/artifacts") / "suite" / "cell-a" / "h123" / "run-1")
        self.assertEqual(layout.to_remote(layout.samples_path), layout.remote_root / RunLayout.SAMPLES)
        self.assertEqual(
            layout.to_remote(layout.logs_dir / "container.log"),
            layout.remote_root / "logs" / "container.log",
        )

    def test_remote_unset_is_none_and_fails_closed(self):
        """Flow 5: no remote dir -> remote_root None, to_remote raises."""
        tmp = Path(tempfile.mkdtemp())
        layout = RunLayout(tmp, "suite", "cell-a", "h123", "run-1")
        self.assertIsNone(layout.remote_root)
        with self.assertRaisesRegex(ValueError, "no remote_artifact_dir"):
            layout.to_remote(layout.samples_path)

    def test_to_remote_rejects_path_not_under_root(self):
        """Flow 5: a path outside root raises a *distinct* error from the no-remote case."""
        tmp = Path(tempfile.mkdtemp())
        layout = RunLayout(tmp, "suite", "cell-a", "h123", "run-1", remote_artifact_dir="/remote")
        with self.assertRaisesRegex(ValueError, "not under RunLayout.root"):
            layout.to_remote(Path("/etc/passwd"))

    def test_rejects_escaping_segment(self):
        """A path-bearing or absolute component must not silently escape the root."""
        tmp = Path(tempfile.mkdtemp())
        for bad in ("/etc/cron.d", "..", "a/b", ""):
            with self.assertRaisesRegex(ValueError, "single path segment"):
                RunLayout(tmp, "suite", bad, "h123", "run-1")


class TestEvents(unittest.TestCase):
    def test_closed_vocabulary(self):
        """Flow 6: valid name emits, unknown raises, vocab is frozen."""
        ew = EventWriter(None)
        ew.emit("prepare.start")
        with self.assertRaises(UnknownEventError):
            ew.emit("not.a.real.event")
        self.assertIn("verify.passed", EVENT_VOCAB)
        self.assertIsInstance(EVENT_VOCAB, frozenset)

    def test_jsonl_file_is_append_only(self):
        """Flow 7: events.jsonl written one parseable JSON object per line, appended."""
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "events.jsonl"
        with EventWriter(path) as ew:
            ew.emit("prepare.start", run_id="r1")
            ew.emit("step", step=1)
        with EventWriter(path) as ew:
            ew.emit("teardown.done", run_id="r1")
        lines = path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)
        records = [json.loads(line) for line in lines]
        self.assertEqual([r["event"] for r in records], ["prepare.start", "step", "teardown.done"])
        self.assertTrue(all("ts" in r for r in records))
        self.assertEqual(records[1]["step"], 1)

    def test_streaming_mode_does_not_buffer_in_memory(self):
        """Flow 7: when streaming to a file, records are not also retained in RAM."""
        tmp = Path(tempfile.mkdtemp())
        with EventWriter(tmp / "events.jsonl") as ew:
            for i in range(100):
                ew.emit("step", step=i)
            self.assertEqual(ew.records, [])
        # In-memory mode (no path) still buffers for callers that read .records.
        mem = EventWriter(None)
        mem.emit("step", step=0)
        self.assertEqual(len(mem.records), 1)


class TestSidecars(unittest.TestCase):
    def test_samples_roundtrip_wide(self):
        """Flow 8: per-sample wide rows round-trip with named columns."""
        tmp = Path(tempfile.mkdtemp())
        rows = [{"request_id": i, "ttft_ms": 10 + i, "role": "server", "host": "n0"} for i in range(5)]
        write_samples(tmp / "samples.parquet", rows)
        frame = read_samples(tmp / "samples.parquet")
        self.assertEqual(len(frame), 5)
        self.assertIn("ttft_ms", frame.columns)

    def test_trajectory_roundtrip_long_format(self):
        """Flow 9: long-format trajectory rows round-trip with stable columns."""
        tmp = Path(tempfile.mkdtemp())
        rows = [
            {"step": i, "metric": "loss", "value": 2.0 - i * 0.1, "role": "trainer", "host": "n0"} for i in range(4)
        ]
        write_trajectory(tmp / "trajectory.parquet", rows)
        frame = read_trajectory(tmp / "trajectory.parquet")
        self.assertEqual(len(frame), 4)
        self.assertEqual(sorted(frame.columns), sorted(["step", "metric", "value", "role", "host"]))

    def test_write_resolved_config_yaml(self):
        """Flow 10: resolved config dumped as YAML, reloads to the same dict."""
        tmp = Path(tempfile.mkdtemp())
        dump = {"framework": "vllm", "params": {"model": "llama", "tensor_parallelism": 8}}
        out = write_resolved_config(tmp / "config.resolved.yaml", dump)
        self.assertEqual(yaml.safe_load(out.read_text()), dump)

    def test_trajectory_rejects_wide_row(self):
        """Long-format invariant: a row with an off-schema column is rejected."""
        tmp = Path(tempfile.mkdtemp())
        wide = [{"step": 0, "loss": 2.0, "role": "trainer", "host": "n0"}]
        with self.assertRaisesRegex(ValueError, "long-format"):
            write_trajectory(tmp / "trajectory.parquet", wide)

    def test_empty_rows_preserve_declared_columns(self):
        """Empty input keeps the declared schema instead of a column-less frame."""
        tmp = Path(tempfile.mkdtemp())
        write_trajectory(tmp / "trajectory.parquet", [])
        traj = read_trajectory(tmp / "trajectory.parquet")
        self.assertEqual(len(traj), 0)
        self.assertEqual(sorted(traj.columns), sorted(["step", "metric", "value", "role", "host"]))
        write_samples(tmp / "samples.parquet", [])
        samples = read_samples(tmp / "samples.parquet")
        self.assertEqual(len(samples), 0)
        self.assertIn("request_id", samples.columns)


class TestExport(unittest.TestCase):
    def test_fan_in_skips_unreadable(self):
        """Flow 11: N manifests -> one fact table; junk manifest.json skipped."""
        tmp = Path(tempfile.mkdtemp())
        for rid, thr in [("run-1", 1000), ("run-2", 2000)]:
            layout = RunLayout(tmp, "suite", "cell-a", "h123", rid).ensure()
            _full_manifest(run_id=rid, extra_metric=thr).write(layout.manifest_path)
        # A partial/corrupt manifest.json must be skipped, not crash the export.
        junk = RunLayout(tmp, "suite", "cell-b", "h999", "run-junk").ensure()
        junk.manifest_path.write_text("{ not valid json")

        manifests = collect_manifests(tmp)
        self.assertEqual(len(manifests), 2)

        out = export_runs(tmp, tmp / "fact.parquet")
        import pandas as pd

        fact = pd.read_parquet(out)
        self.assertEqual(len(fact), 2)
        for col in (
            "run_id",
            "test_id",
            "workload_hash",
            "overall_status",
            "scalar_total_throughput",
            "scalar_extra_metric",
        ):
            self.assertIn(col, fact.columns)
        self.assertEqual(set(fact["run_id"]), {"run-1", "run-2"})


class TestManifestHardening(unittest.TestCase):
    """Latent-bug hardening (NaN/Inf JSON, emit-after-close, empty export, visible skip)."""

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

    def test_emit_after_close_raises(self):
        """A file-backed writer raises on emit-after-close; in-memory is unaffected."""
        tmp = Path(tempfile.mkdtemp())
        ew = EventWriter(tmp / "events.jsonl")
        ew.emit("prepare.start")
        ew.close()
        with self.assertRaisesRegex(ValueError, "after close"):
            ew.emit("prepare.done")
        mem = EventWriter(None)
        mem.close()
        self.assertEqual(mem.emit("step", step=1)["event"], "step")

    def test_empty_export_has_fact_columns(self):
        """Exporting an empty tree yields the fixed-schema fact table, not column-less."""
        import pandas as pd

        tmp = Path(tempfile.mkdtemp())
        fact = pd.read_parquet(export_runs(tmp, tmp / "fact.parquet"))
        self.assertEqual(len(fact), 0)
        self.assertEqual(list(fact.columns), FACT_COLUMNS)
        self.assertEqual(list(fact["run_id"]), [])  # column access must not KeyError

    def test_fact_columns_match_flatten(self):
        """Parity guard: FACT_COLUMNS must equal the static (non-scalar) keys of _flatten_manifest."""
        static = [k for k in _flatten_manifest(_full_manifest()) if not k.startswith("scalar_")]
        self.assertEqual(static, FACT_COLUMNS)

    def test_collect_logs_unreadable_skip(self):
        """A skipped corrupt/forward-incompatible manifest is logged, not silently dropped."""
        tmp = Path(tempfile.mkdtemp())
        good = RunLayout(tmp, "s", "c", "h", "good").ensure()
        _full_manifest(run_id="good").write(good.manifest_path)
        bad = RunLayout(tmp, "s", "c", "h", "bad").ensure()
        payload = json.loads(good.manifest_path.read_text())
        payload["unknown_future_field"] = 1  # rejected by extra="forbid"
        bad.manifest_path.write_text(json.dumps(payload))
        with self.assertLogs("cvs.lib.manifest.export", level="WARNING") as cm:
            manifests = collect_manifests(tmp)
        self.assertEqual([m.identity.run_id for m in manifests], ["good"])
        self.assertTrue(any("bad" in line for line in cm.output))

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


if __name__ == "__main__":
    unittest.main()
