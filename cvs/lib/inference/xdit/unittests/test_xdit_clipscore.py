import base64
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.inference.xdit.xdit_clipscore import (
    CLIP_SCORE_MARKER,
    DEFAULT_CLIP_MODEL,
    DEFAULT_SAMPLE_FRAMES,
    WAN_DIFFUSERS_CLIP_VIDEO,
    WAN_NATIVE_CLIP_VIDEO,
    build_clip_model_probe_cmd,
    build_clipscore_cmd,
    clip_model_missing_message,
    clipscore_output_path,
    parse_clipscore_text,
    resolve_clip_model,
    resolve_clip_sample_frames,
    validate_clip_score,
    validate_mean_frame_clip_score,
    wan_clip_video_name,
)


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "clipscore.py"
    spec = importlib.util.spec_from_file_location("cvs_clipscore_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestClipScoreScript(unittest.TestCase):
    def test_finds_flux1_and_flux2_pngs_and_skips_empty_files(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            results.mkdir()
            (results / "flux_result.png").write_bytes(b"png")
            (results / "flux2_dev_result.png").write_bytes(b"png")
            (results / "flux_empty.png").write_bytes(b"")
            (results / "notes.txt").write_text("ignore", encoding="utf-8")

            found = [path.name for path in script.find_flux_images(tmp)]

        self.assertEqual(found, ["flux2_dev_result.png", "flux_result.png"])

    def test_payload_uses_mean_clip_score(self):
        script = _load_script()

        payload = script.build_clipscore_payload(
            "A small cat",
            DEFAULT_CLIP_MODEL,
            [("a.png", 30.0), ("b.png", 20.0)],
        )

        self.assertEqual(payload["clip_score"], 25.0)
        self.assertEqual(payload["image_count"], 2)
        self.assertEqual(payload["prompt"], "A small cat")
        self.assertFalse(payload["prompt_truncated"])

    def test_finds_wan_videos_and_skips_empty_files(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "outputs" / "outputs"
            nested.mkdir(parents=True)
            (root / "video.mp4").write_bytes(b"")
            (root / "outputs" / "video.mp4").write_bytes(b"mp4")
            (nested / "video.mp4").write_bytes(b"mp4")
            (root / "results").mkdir()
            (root / "results" / "video_i2v.mp4").write_bytes(b"mp4")

            native = script.find_wan_video(tmp, "video.mp4")
            diffusers = script.find_wan_video(tmp, "video_i2v.mp4")
            missing = script.find_wan_video(tmp, "missing.mp4")

        self.assertEqual(native, root / "outputs" / "video.mp4")
        self.assertEqual(diffusers, root / "results" / "video_i2v.mp4")
        self.assertIsNone(missing)

    def test_samples_endpoints_and_keeps_indexes_unique(self):
        script = _load_script()

        indices = script.sample_frame_indices(81, 8)

        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 80)
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertEqual(script.sample_frame_indices(3, 8), [0, 1, 2])
        for count in range(1, 31):
            for samples in range(1, count + 3):
                chosen = script.sample_frame_indices(count, samples)
                self.assertEqual(chosen, sorted(set(chosen)))
                self.assertEqual(len(chosen), min(samples, count))
                self.assertTrue(all(0 <= index < count for index in chosen))

    def test_empty_video_is_an_error(self):
        script = _load_script()

        with self.assertRaisesRegex(ValueError, "no frames"):
            script.sample_frame_indices(0, 8)

    def test_frame_payload_reports_mean_and_min(self):
        script = _load_script()

        payload = script.build_frame_clipscore_payload(
            "A cat walking",
            DEFAULT_CLIP_MODEL,
            "video.mp4",
            81,
            [(0, 30.0), (80, 20.0)],
        )

        self.assertEqual(payload["mean_frame_clip_score"], 25.0)
        self.assertEqual(payload["min_frame_clip_score"], 20.0)
        self.assertEqual(payload["frame_indices"], [0, 80])
        self.assertEqual(payload["frame_count"], 81)
        self.assertEqual(payload["sample_count"], 2)
        self.assertFalse(payload["prompt_truncated"])
        self.assertNotIn("clip_score", payload)

    def test_long_prompt_is_marked_truncated(self):
        script = _load_script()
        model = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=77), max_position_embeddings=99)
        )

        def tokenizer(prompt, add_special_tokens=True):
            del prompt, add_special_tokens
            return {"input_ids": list(range(99))}

        token_count = script.clip_token_count(tokenizer, "beach prompt")
        context = script.clip_context_length(model)
        payload = script.build_frame_clipscore_payload(
            "beach prompt",
            DEFAULT_CLIP_MODEL,
            "video.mp4",
            81,
            [(0, 25.0)],
            prompt_truncated=token_count > context,
        )

        self.assertEqual(context, 77)
        self.assertEqual(token_count, 99)
        self.assertTrue(payload["prompt_truncated"])
        fallback = script.clip_context_length(SimpleNamespace(config=SimpleNamespace()))
        self.assertEqual(fallback, script.CLIP_CONTEXT_LENGTH)

    def test_video_mode_reports_missing_mp4(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            code = script.main(
                [
                    "--prompt",
                    "A cat walking",
                    "--image-dir",
                    tmp,
                    "--output",
                    str(Path(tmp) / "results" / "clipscore.json"),
                    "--model",
                    DEFAULT_CLIP_MODEL,
                    "--sample-frames",
                    "8",
                    "--video-name",
                    "video.mp4",
                ]
            )
            self.assertFalse((Path(tmp) / "results" / "clipscore.json").exists())

        self.assertEqual(code, 1)

    def test_cosine_floor_is_zero(self):
        script = _load_script()

        self.assertEqual(script.clip_score_from_cosine(-0.2), 0.0)
        self.assertEqual(script.clip_score_from_cosine(0.25), 25.0)


class TestClipScoreCommand(unittest.TestCase):
    def test_command_embeds_script_and_quotes_prompt(self):
        cmd = build_clipscore_cmd(
            "A small cat",
            "/outputs/flux_host_outputs",
            "/outputs/flux_host_outputs/results/clipscore.json",
            DEFAULT_CLIP_MODEL,
        )

        self.assertIn("--prompt 'A small cat'", cmd)
        self.assertIn("--image-dir /outputs/flux_host_outputs", cmd)
        self.assertIn(DEFAULT_CLIP_MODEL, cmd)
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "clipscore.py"
        script_text = script_path.read_text(encoding="utf-8")
        encoded = base64.b64encode(script_text.encode("utf-8")).decode("ascii")
        self.assertIn(encoded, cmd)
        self.assertIn(CLIP_SCORE_MARKER.strip(), script_text)

    def test_resolve_model_prefers_config(self):
        self.assertEqual(resolve_clip_model({}), DEFAULT_CLIP_MODEL)
        self.assertEqual(resolve_clip_model({"clip_model": "/models/clip"}), "/models/clip")

    def test_wan_command_selects_video_and_sample_count(self):
        cmd = build_clipscore_cmd(
            "A cat walking",
            "/outputs/wan_host_outputs",
            "/outputs/wan_host_outputs/results/clipscore.json",
            DEFAULT_CLIP_MODEL,
            sample_frames=DEFAULT_SAMPLE_FRAMES,
            video_name=WAN_DIFFUSERS_CLIP_VIDEO,
        )

        self.assertIn("--sample-frames 8", cmd)
        self.assertIn(f"--video-name {WAN_DIFFUSERS_CLIP_VIDEO}", cmd)
        self.assertIn("HF_HUB_OFFLINE=1", cmd)
        self.assertEqual(wan_clip_video_name(False), WAN_NATIVE_CLIP_VIDEO)
        self.assertEqual(wan_clip_video_name(True), WAN_DIFFUSERS_CLIP_VIDEO)
        self.assertEqual(resolve_clip_sample_frames({}), DEFAULT_SAMPLE_FRAMES)
        self.assertEqual(resolve_clip_sample_frames({"clip_sample_frames": 4}), 4)
        with self.assertRaisesRegex(ValueError, "clip_sample_frames"):
            resolve_clip_sample_frames({"clip_sample_frames": 0})

    def test_probe_requires_a_local_clip_checkpoint(self):
        cmd = build_clip_model_probe_cmd(DEFAULT_CLIP_MODEL)

        self.assertIn("HF_HOME=/hf_home", cmd)
        self.assertIn("HF_HUB_OFFLINE=1", cmd)
        self.assertIn("local_files_only=True", cmd)
        self.assertIn(DEFAULT_CLIP_MODEL, cmd)
        custom = build_clip_model_probe_cmd(DEFAULT_CLIP_MODEL, "/cache/hf")
        self.assertIn("HF_HOME=/cache/hf", custom)
        self.assertIn("does not download", clip_model_missing_message(DEFAULT_CLIP_MODEL, ["n1"]))
        self.assertIn("enforce_thresholds", clip_model_missing_message(DEFAULT_CLIP_MODEL, ["n1"]))

    def test_output_path(self):
        self.assertEqual(
            clipscore_output_path("/outputs/flux_host_outputs"),
            "/outputs/flux_host_outputs/results/clipscore.json",
        )


class TestParseAndValidate(unittest.TestCase):
    def test_parse_uses_last_marker(self):
        first = {"clip_score": 1.0, "image_count": 1}
        last = {"clip_score": 27.5, "image_count": 1, "model": DEFAULT_CLIP_MODEL}
        text = f"noise\n{CLIP_SCORE_MARKER}{json.dumps(first)}\n{CLIP_SCORE_MARKER}{json.dumps(last)}\n"

        self.assertEqual(parse_clipscore_text(text)["clip_score"], 27.5)

    def test_parse_missing_marker(self):
        self.assertIsNone(parse_clipscore_text("No FLUX PNG images found"))

    def test_validate_pass_and_fail(self):
        thresholds = {"auto": {"min_clip_score": 20.0}, "mi355": {"min_clip_score": 30.0}}
        result = {"clip_score": 27.5, "image_count": 1, "model": DEFAULT_CLIP_MODEL}

        passed, message = validate_clip_score(result, thresholds, "auto")
        self.assertTrue(passed)
        self.assertIn("27.50", message)

        passed, message = validate_clip_score(result, thresholds, "mi355")
        self.assertFalse(passed)
        self.assertIn("min 30.00", message)

    def test_validate_falls_back_to_auto_and_requires_floor(self):
        passed, message = validate_clip_score({"clip_score": 27.5}, {"auto": {"min_clip_score": 20}}, "mi300x")
        self.assertTrue(passed)

        passed, message = validate_clip_score({"clip_score": 27.5}, {"auto": {"max_avg_pipe_time_s": 10}}, "auto")
        self.assertFalse(passed)
        self.assertIn("min_clip_score", message)

        passed, message = validate_clip_score({"clip_score": 27.5}, {}, "mi300x")
        self.assertFalse(passed)
        self.assertIn("No threshold found", message)

    def test_parse_wan_frame_payload(self):
        payload = {"mean_frame_clip_score": 24.5, "sample_count": 8, "model": DEFAULT_CLIP_MODEL}
        text = f"{CLIP_SCORE_MARKER}{json.dumps(payload)}\n"

        self.assertEqual(parse_clipscore_text(text)["mean_frame_clip_score"], 24.5)

    def test_validate_mean_frame_score(self):
        thresholds = {"auto": {"min_mean_frame_clip_score": 20.0}, "mi355": {"min_mean_frame_clip_score": 30.0}}
        result = {"mean_frame_clip_score": 27.5, "sample_count": 8, "model": DEFAULT_CLIP_MODEL}

        passed, message = validate_mean_frame_clip_score(result, thresholds, "auto")
        self.assertTrue(passed)
        self.assertIn("27.50", message)
        self.assertIn("frame(s)", message)

        passed, message = validate_mean_frame_clip_score(result, thresholds, "mi355")
        self.assertFalse(passed)
        self.assertIn("min 30.00", message)

        passed, message = validate_mean_frame_clip_score(
            result,
            {"auto": {"max_avg_total_time_s": 15.0}},
            "auto",
        )
        self.assertFalse(passed)
        self.assertIn("min_mean_frame_clip_score", message)


if __name__ == "__main__":
    unittest.main()
