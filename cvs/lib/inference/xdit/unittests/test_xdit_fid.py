import importlib.util
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cvs.lib.inference.xdit.xdit_fid import (
    fid_absent_skip_reason,
    fid_budget_seconds,
    fid_cli_flags,
    fid_deadline_shell,
    fid_viewer_curves,
    parse_fid_curve_text,
    parse_fid_nfe_steps,
)
from cvs.lib.inference.xdit.xdit_flux_job import build_run_usp_args
from cvs.tests.inference.xdit._shared import _manifest_blocks_fid

_SWEEP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fid_sweep.py"
_spec = importlib.util.spec_from_file_location("fid_sweep", _SWEEP_PATH)
_fid_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fid_sweep)


def _stub_module(name):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


def _load_fid_score():
    """Import the container scorer. The unit-test env has no torch or Pillow."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        numpy = _stub_module("numpy")
        numpy.float32 = float
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = _stub_module("torch")
        nn = _stub_module("torch.nn")
        torch.nn = nn

        class Identity:
            pass

        nn.Identity = Identity
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        pil = _stub_module("PIL")
        image = _stub_module("PIL.Image")
        pil.Image = image
    path = Path(__file__).resolve().parents[1] / "scripts" / "fid_score.py"
    spec = importlib.util.spec_from_file_location("fid_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fid_score = _load_fid_score()


class TestFidCurve(unittest.TestCase):
    def test_steps_stay_empty_until_configured(self):
        self.assertEqual(parse_fid_nfe_steps({}), [])
        self.assertEqual(parse_fid_nfe_steps({"fid_nfe_steps": ""}), [])
        self.assertEqual(parse_fid_nfe_steps({"fid_nfe_steps": "10, 30, 10"}), [10, 30])

    def test_one_step_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_fid_nfe_steps({"fid_nfe_steps": [20]})

    def test_viewer_curves_keep_each_sampler(self):
        raw = {
            "points": [
                {"nfe": 50, "fid": 12.0, "sampler": "euler"},
                {"nfe": 10, "fid": 40.0, "sampler": "euler"},
                {"nfe": 10, "fid": 1.0, "sampler": "dpm"},
            ]
        }
        self.assertEqual(
            fid_viewer_curves(raw),
            {"FID · euler": [[10.0, 40.0], [50.0, 12.0]]},
        )

    def test_marker_text_parses(self):
        payload = parse_fid_curve_text(
            'noise\nCVS_FID_CURVE {"sampler": "euler", "points": [{"nfe": 10, "fid": 8}, {"nfe": 20, "fid": 4}]}\n'
        )
        self.assertEqual(payload["sampler"], "euler")

    def test_budget_leaves_three_minutes(self):
        self.assertEqual(fid_budget_seconds(1800), 1620)
        self.assertEqual(fid_budget_seconds(100), 0)
        self.assertEqual(fid_deadline_shell(1620), "$(($(date +%s) + 1620))")

    def test_seconds_left_uses_the_epoch_deadline(self):
        with patch.object(_fid_sweep.time, "time", return_value=1_000.0):
            self.assertEqual(_fid_sweep.seconds_until(1_100), 100.0)
            self.assertEqual(_fid_sweep.seconds_until(900), 0.0)
            self.assertEqual(_fid_sweep.seconds_until(0), 0.0)

    def test_next_sample_prefers_coverage_and_stops_before_the_deadline(self):
        steps = [10, 30]
        counts = {10: 0, 30: 0}
        first = _fid_sweep.next_fid_sample(steps, counts, 2, 0, 100, 10, 20)
        self.assertEqual(first, 10)
        counts[10] = 1
        second = _fid_sweep.next_fid_sample(steps, counts, 2, 0, 100, 10, 20)
        self.assertEqual(second, 30)
        self.assertIsNone(_fid_sweep.next_fid_sample(steps, counts, 2, 95, 100, 10, 20))

    def test_unknown_pipe_time_does_not_start_a_sample(self):
        self.assertIsNone(_fid_sweep.next_fid_sample([10, 20], {10: 0, 20: 0}, 2, 0, 100, 0, 20))

    def test_cli_flags_carry_the_budget(self):
        flags = fid_cli_flags({"fid_nfe_steps": [10, 20], "fid_samples": 2}, 1620)
        self.assertIn("--fid_steps 10,20", flags)
        self.assertIn("--fid_deadline $fid_deadline", flags)
        self.assertNotIn("--fid_budget_s", flags)
        params = {
            "prompt": "a cat",
            "seed": 0,
            "num_inference_steps": 20,
            "max_sequence_length": 256,
            "no_use_resolution_binning": True,
            "use_torch_compile": False,
            "warmup_steps": 1,
            "warmup_calls": 0,
            "num_repetitions": 1,
            "height": 1024,
            "width": 1024,
            "ulysses_degree": 1,
            "ring_degree": 1,
            "fid_nfe_steps": [10, 20],
            "_fid_budget_s": 1620,
        }
        self.assertIn("--fid_deadline $fid_deadline", build_run_usp_args(params, model_repo="/model"))
        bare = dict(params)
        bare.pop("fid_nfe_steps")
        bare.pop("_fid_budget_s")
        self.assertNotIn("--fid_steps", build_run_usp_args(bare, model_repo="/model"))

    def test_manifest_counts_videos_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            step_dir = Path(tmp) / "nfe_10"
            step_dir.mkdir()
            (step_dir / "seed_0.mp4").write_bytes(b"video")
            (step_dir / "seed_0_frame_0.png").write_bytes(b"png")
            manifest = Path(tmp) / "manifest.json"
            _fid_sweep.rewrite_manifest_from_disk(str(manifest), [10, 40], 2, 100)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["counts"], {"10": 1, "40": 0})
        self.assertTrue(payload["stopped_early"])

    def test_manifest_counts_frames_when_video_export_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            step_dir = Path(tmp) / "nfe_10"
            step_dir.mkdir()
            for slot in range(8):
                (step_dir / f"seed_0_frame_{slot}.png").write_bytes(b"png")
                (step_dir / f"seed_1_frame_{slot}.png").write_bytes(b"png")
            (step_dir / "seed_2.mp4").write_bytes(b"video")
            (step_dir / "notes.txt").write_text("x", encoding="utf-8")
            manifest = Path(tmp) / "manifest.json"
            _fid_sweep.rewrite_manifest_from_disk(str(manifest), [10, 40], 2, 100)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["counts"], {"10": 3, "40": 0})

    def test_manifest_rewrite_keeps_sweep_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            _fid_sweep.write_manifest(str(manifest), 100, 2, {10: 0, 40: 0}, [], error="boom")
            _fid_sweep.rewrite_manifest_from_disk(str(manifest), [10, 40], 2, 100)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["error"], "boom")
        self.assertEqual(payload["counts"], {"10": 0, "40": 0})

    def test_short_budget_manifest_skips_instead_of_failing(self):
        self.assertTrue(_manifest_blocks_fid({"stopped_early": True, "counts": {"10": 1, "20": 0}}))
        self.assertFalse(_manifest_blocks_fid({"stopped_early": True, "counts": {"10": 2, "20": 2}}))

    def test_a_single_point_is_not_a_line(self):
        raw = {"points": [{"nfe": 10, "fid": 1.0}]}
        self.assertEqual(fid_viewer_curves(raw), {})

    def _load_with(self, inception_v3, state):
        tv_models = types.ModuleType("torchvision.models")
        tv_models.inception_v3 = inception_v3
        tv = types.ModuleType("torchvision")
        tv.models = tv_models
        with patch.dict(sys.modules, {"torchvision": tv, "torchvision.models": tv_models}):
            with patch.object(_fid_score.torch, "load", return_value=state, create=True):
                return _fid_score._load_inception("inception_v3.pth", "cpu")

    def test_inception_matches_imagenet_checkpoint(self):
        model = MagicMock()
        seen = {}

        def inception_v3(**kwargs):
            seen["kwargs"] = kwargs
            return model

        model.to.return_value = model
        state = {"state_dict": {"module.conv.weight": object(), "fc.weight": object()}}
        loaded = self._load_with(inception_v3, state)
        self.assertIs(loaded, model)
        self.assertTrue(seen["kwargs"]["transform_input"])
        self.assertTrue(seen["kwargs"]["aux_logits"])
        self.assertFalse(seen["kwargs"]["init_weights"])
        args, kwargs = model.load_state_dict.call_args
        self.assertEqual(set(args[0]), {"conv.weight", "fc.weight"})
        self.assertTrue(kwargs["strict"])
        self.assertIsInstance(model.fc, _fid_score.torch.nn.Identity)
        model.eval.assert_called_once()
        model.to.assert_called_once_with("cpu")

    def test_inception_keeps_transform_when_init_weights_is_unsupported(self):
        model = MagicMock()
        seen = []

        def inception_v3(**kwargs):
            seen.append(kwargs)
            if "init_weights" in kwargs:
                raise TypeError("init_weights")
            return model

        self._load_with(inception_v3, {"conv.weight": object()})
        self.assertTrue(seen[0]["transform_input"])
        self.assertTrue(seen[1]["transform_input"])
        self.assertNotIn("init_weights", seen[1])
        _args, kwargs = model.load_state_dict.call_args
        self.assertTrue(kwargs["strict"])

    def test_inception_load_reports_a_bad_checkpoint(self):
        model = MagicMock()
        model.load_state_dict.side_effect = RuntimeError("missing keys")

        def inception_v3(**kwargs):
            return model

        with self.assertRaises(RuntimeError):
            self._load_with(inception_v3, {"conv.weight": object()})

    def test_reference_set_must_be_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing")
            with self.assertRaises(SystemExit) as raised:
                _fid_score._require_reference(missing, 8)
            self.assertIn(missing, str(raised.exception))

            empty = Path(tmp) / "empty"
            empty.mkdir()
            (empty / "notes.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                _fid_score._require_reference(str(empty), 8)
            self.assertIn("needs at least 2", str(raised.exception))

    def test_collective_returns_the_result_when_every_rank_succeeds(self):
        with patch.object(_fid_sweep, "_gather_errors", return_value=["", ""]):
            self.assertEqual(_fid_sweep.collective(0, lambda nfe, index: nfe + index)(10, 1), 11)

    def test_collective_raises_on_the_rank_that_failed(self):
        def generate_one(nfe, index):
            raise OSError("disk full")

        gathered = []

        def gather(error):
            gathered.append(error)
            return [error, ""]

        with patch.object(_fid_sweep, "_gather_errors", side_effect=gather):
            with self.assertRaises(_fid_sweep.CollectiveSweepError) as raised:
                _fid_sweep.collective(1, generate_one)(10, 0)
        self.assertEqual(gathered, ["rank 1: disk full"])
        self.assertIn("rank 1: disk full", str(raised.exception))

    def test_collective_raises_on_a_rank_whose_own_sample_succeeded(self):
        with patch.object(_fid_sweep, "_gather_errors", return_value=["", "rank 3: image.save failed"]):
            with self.assertRaises(_fid_sweep.CollectiveSweepError) as raised:
                _fid_sweep.collective(0, lambda nfe, index: 1.0)(10, 0)
        self.assertIn("rank 3: image.save failed", str(raised.exception))

    def test_sweep_stops_on_every_rank_when_one_sample_fails(self):
        calls = []

        def generate_one(nfe, index):
            calls.append((nfe, index))
            return 1.0

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            with patch.object(_fid_sweep, "_gather_errors", return_value=["", "rank 1: no space left"]):
                with self.assertRaises(_fid_sweep.CollectiveSweepError):
                    _fid_sweep.run_fid_sweep(
                        0, [10, 20], 2, 1000, time.monotonic(), 1.0, 10, generate_one, str(manifest)
                    )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(calls, [(10, 0)])
        self.assertIn("rank 1: no space left", payload["error"])
        self.assertTrue(payload["stopped_early"])
        self.assertEqual(payload["counts"], {"10": 0, "20": 0})

    def test_sweep_without_a_process_group_runs_to_the_budget(self):
        calls = []

        def generate_one(nfe, index):
            calls.append((nfe, index))
            return 0.01

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            produced = _fid_sweep.run_fid_sweep(
                0, [10, 20], 1, 1000, time.monotonic(), 1.0, 10, generate_one, str(manifest)
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(sorted(calls), [(10, 0), (20, 0)])
        self.assertEqual(len(produced), 2)
        self.assertEqual(payload["error"], "")
        self.assertFalse(payload["stopped_early"])

    def test_absent_checkpoint_or_reference_skips(self):
        empty = fid_absent_skip_reason("", "/hf_home/inception_v3.pth")
        self.assertIn("fid_reference_dir or fid_inception_path is empty", empty)
        self.assertIn("does not download", empty)
        missing_ckpt = fid_absent_skip_reason(
            "/hf_home/fid_reference",
            "/hf_home/inception_v3.pth",
            "n1: FID command failed (inception checkpoint not found: /hf_home/inception_v3.pth)",
        )
        self.assertIn("inception checkpoint not found", missing_ckpt)
        missing_ref = fid_absent_skip_reason(
            "/hf_home/fid_reference",
            "/hf_home/inception_v3.pth",
            "n1: FID command failed (FID reference directory not found: /hf_home/fid_reference)",
        )
        self.assertIn("FID reference directory not found", missing_ref)
        short = fid_absent_skip_reason(
            "/hf_home/fid_reference",
            "/hf_home/inception_v3.pth",
            "n1: FID command failed (/hf_home/fid_reference has 1 sample(s); FID reference needs at least 2)",
        )
        self.assertEqual(short, "")


if __name__ == "__main__":
    unittest.main()
