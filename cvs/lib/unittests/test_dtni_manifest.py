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
from cvs.lib.manifest.export import collect_manifests, export_runs


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


if __name__ == "__main__":
    unittest.main()
