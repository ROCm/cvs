"""Unit tests for pytorch_xdit_wan output parsing."""

import json
import tempfile
import unittest
from pathlib import Path

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_wan import (
    WanBenchmarkResult,
    WanOutputParser,
)


def _write_rank0_json(directory: Path, name: str, total_time: float) -> Path:
    path = directory / name
    path.write_text(json.dumps({"total_time": total_time}), encoding="utf-8")
    return path


class TestWanOutputParserHelpers(unittest.TestCase):
    def test_select_bench_dir_prefers_deepest_outputs_with_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "wan_22_node0_outputs"
            deep = run_dir / "outputs" / "outputs" / "outputs"
            deep.mkdir(parents=True)
            _write_rank0_json(deep, "rank0_step0.json", 10.0)
            _write_rank0_json(run_dir / "outputs", "rank0_step1.json", 20.0)

            bench_dir = WanOutputParser._select_bench_dir(run_dir)
            self.assertEqual(bench_dir, deep)

    def test_select_bench_dir_prefers_results_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "wan_22_node0_outputs"
            results_outputs = run_dir / "results" / "outputs"
            results_outputs.mkdir(parents=True)
            _write_rank0_json(results_outputs, "rank0_step0.json", 15.0)

            bench_dir = WanOutputParser._select_bench_dir(run_dir)
            self.assertEqual(bench_dir, results_outputs)

    def test_label_from_run_dir_strips_prefix_and_suffix(self):
        label = WanOutputParser._label_from_run_dir(Path("/tmp/wan_22_tus1-p3-g40_outputs"))
        self.assertEqual(label, "tus1-p3-g40")


class TestWanOutputParserParseBenchmarkJsons(unittest.TestCase):
    def test_parse_valid_json_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            files = [
                _write_rank0_json(root, "rank0_step0.json", 100.0),
                _write_rank0_json(root, "rank0_step1.json", 120.0),
            ]
            parser = WanOutputParser(tmpdir)
            step_times, errors = parser.parse_benchmark_jsons(files)
            self.assertEqual(errors, [])
            self.assertEqual(step_times, [100.0, 120.0])

    def test_parse_missing_total_time_reports_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "rank0_step0.json"
            bad.write_text(json.dumps({"pipe_time": 1.0}), encoding="utf-8")
            parser = WanOutputParser(tmpdir)
            step_times, errors = parser.parse_benchmark_jsons([bad])
            self.assertEqual(step_times, [])
            self.assertTrue(any("missing 'total_time'" in err for err in errors))

    def test_parse_non_numeric_total_time_reports_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "rank0_step0.json"
            bad.write_text(json.dumps({"total_time": "slow"}), encoding="utf-8")
            parser = WanOutputParser(tmpdir)
            step_times, errors = parser.parse_benchmark_jsons([bad])
            self.assertEqual(step_times, [])
            self.assertTrue(any("not numeric" in err for err in errors))

    def test_parse_invalid_json_reports_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "rank0_step0.json"
            bad.write_text("{not json", encoding="utf-8")
            parser = WanOutputParser(tmpdir)
            step_times, errors = parser.parse_benchmark_jsons([bad])
            self.assertEqual(step_times, [])
            self.assertTrue(any("JSON parse error" in err for err in errors))


class TestWanOutputParserParse(unittest.TestCase):
    def test_parse_computes_average_and_finds_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_rank0_json(root, "rank0_step0.json", 100.0)
            _write_rank0_json(root, "rank0_step1.json", 140.0)
            (root / "video.mp4").write_bytes(b"fake-video")

            parser = WanOutputParser(tmpdir)
            result, errors = parser.parse()

            self.assertEqual(errors, [])
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result.avg_total_time_s, 120.0)
            self.assertEqual(result.step_count, 2)
            self.assertEqual(result.step_times, [100.0, 140.0])
            self.assertTrue(result.artifact_path.endswith("video.mp4"))

    def test_parse_finds_nested_rank0_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "outputs" / "outputs" / "outputs"
            nested.mkdir(parents=True)
            _write_rank0_json(nested, "rank0_step0.json", 50.0)

            parser = WanOutputParser(tmpdir)
            result, errors = parser.parse()

            self.assertEqual(errors, [])
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result.avg_total_time_s, 50.0)

    def test_parse_missing_json_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = WanOutputParser(tmpdir)
            result, errors = parser.parse()
            self.assertIsNone(result)
            self.assertTrue(any("No rank0_step*.json" in err for err in errors))

    def test_parse_ignores_non_rank0_json_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "rank1_step0.json").write_text(json.dumps({"total_time": 1.0}), encoding="utf-8")

            parser = WanOutputParser(tmpdir)
            result, errors = parser.parse()
            self.assertIsNone(result)
            self.assertTrue(any("No rank0_step*.json" in err for err in errors))


class TestWanOutputParserParseRunsUnderBaseDir(unittest.TestCase):
    def test_parse_multiple_run_dirs_and_overall_average(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run_a = base / "wan_22_node-a_outputs"
            run_b = base / "wan_22_node-b_outputs"
            for run_dir, avg_time in ((run_a, 100.0), (run_b, 200.0)):
                bench = run_dir / "outputs"
                bench.mkdir(parents=True)
                _write_rank0_json(bench, "rank0_step0.json", avg_time)
                (run_dir / "video.mp4").write_bytes(b"vid")

            aggregate, errors = WanOutputParser.parse_runs_under_base_dir(str(base))

            self.assertEqual(errors, [])
            self.assertIsNotNone(aggregate)
            self.assertEqual(aggregate.result_count, 2)
            self.assertAlmostEqual(aggregate.overall_avg_total_time_s, 150.0)
            labels = {run.label for run in aggregate.per_run}
            self.assertEqual(labels, {"node-a", "node-b"})

    def test_parse_runs_filters_allowed_run_dir_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            allowed = base / "wan_22_allowed_outputs"
            skipped = base / "wan_22_skipped_outputs"
            for run_dir in (allowed, skipped):
                bench = run_dir / "outputs"
                bench.mkdir(parents=True)
                _write_rank0_json(bench, "rank0_step0.json", 10.0)
                (run_dir / "video.mp4").write_bytes(b"vid")

            aggregate, errors = WanOutputParser.parse_runs_under_base_dir(
                str(base),
                allowed_run_dir_names=["wan_22_allowed_outputs"],
            )

            self.assertEqual(errors, [])
            self.assertIsNotNone(aggregate)
            self.assertEqual(aggregate.result_count, 1)
            self.assertEqual(aggregate.per_run[0].label, "allowed")

    def test_parse_runs_requires_artifact_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "wan_22_node0_outputs"
            bench = run_dir / "outputs"
            bench.mkdir(parents=True)
            _write_rank0_json(bench, "rank0_step0.json", 10.0)

            aggregate, errors = WanOutputParser.parse_runs_under_base_dir(tmpdir)
            self.assertIsNone(aggregate)
            self.assertTrue(any("Artifact 'video.mp4' not found" in err for err in errors))

    def test_parse_runs_missing_base_dir(self):
        aggregate, errors = WanOutputParser.parse_runs_under_base_dir("/tmp/does-not-exist-wan-parser")
        self.assertIsNone(aggregate)
        self.assertTrue(any("Base directory does not exist" in err for err in errors))


class TestWanOutputParserValidateThreshold(unittest.TestCase):
    def _sample_result(self, avg_total_time_s: float) -> WanBenchmarkResult:
        return WanBenchmarkResult(
            avg_total_time_s=avg_total_time_s,
            step_count=1,
            step_times=[avg_total_time_s],
            json_files=["rank0_step0.json"],
            artifact_path="/tmp/video.mp4",
        )

    def test_validate_threshold_pass(self):
        parser = WanOutputParser("/tmp/unused")
        passed, message = parser.validate_threshold(
            self._sample_result(100.0),
            {"auto": {"max_avg_total_time_s": 200.0}},
            gpu_type="auto",
        )
        self.assertTrue(passed)
        self.assertIn("PASS", message)

    def test_validate_threshold_fail(self):
        parser = WanOutputParser("/tmp/unused")
        passed, message = parser.validate_threshold(
            self._sample_result(250.0),
            {"auto": {"max_avg_total_time_s": 200.0}},
            gpu_type="auto",
        )
        self.assertFalse(passed)
        self.assertIn("FAIL", message)

    def test_validate_threshold_uses_gpu_specific_entry(self):
        parser = WanOutputParser("/tmp/unused")
        passed, message = parser.validate_threshold(
            self._sample_result(150.0),
            {
                "mi300x": {"max_avg_total_time_s": 200.0},
                "auto": {"max_avg_total_time_s": 100.0},
            },
            gpu_type="mi300x",
        )
        self.assertTrue(passed)
        self.assertIn("mi300x", message)

    def test_validate_threshold_falls_back_to_auto(self):
        parser = WanOutputParser("/tmp/unused")
        passed, message = parser.validate_threshold(
            self._sample_result(90.0),
            {"auto": {"max_avg_total_time_s": 100.0}},
            gpu_type="mi350",
        )
        self.assertTrue(passed)
        self.assertIn("PASS", message)
        self.assertIn("100.00s", message)

    def test_validate_threshold_missing_config(self):
        parser = WanOutputParser("/tmp/unused")
        passed, message = parser.validate_threshold(
            self._sample_result(10.0),
            {"mi300x": {"max_avg_total_time_s": 100.0}},
            gpu_type="mi350",
        )
        self.assertFalse(passed)
        self.assertIn("No threshold found", message)


if __name__ == "__main__":
    unittest.main()
