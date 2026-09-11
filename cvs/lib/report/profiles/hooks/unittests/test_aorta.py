'''Unit tests for Aorta Run Deck helpers and profile integration.'''

import unittest

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.profiles.hooks.aorta import (
    aorta_run_card_display,
    build_aorta_series,
    update_aorta_run_summary,
)
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.parsers.schemas import AortaBenchmarkResult, AortaTraceMetrics
from cvs.runners._base_runner import RunResult, RunStatus


def _benchmark_result():
    metrics = [
        AortaTraceMetrics(
            rank=0,
            node="node-a",
            local_rank=0,
            total_time_us=10000,
            compute_time_us=7000,
            communication_time_us=4000,
            memory_time_us=500,
            peak_memory_gb=42.5,
            compute_kernel_count=20,
            comm_kernel_count=4,
        ),
        AortaTraceMetrics(
            rank=1,
            node="node-a",
            local_rank=1,
            total_time_us=12000,
            compute_time_us=7200,
            communication_time_us=3600,
        ),
    ]
    return AortaBenchmarkResult.from_rank_metrics(
        metrics,
        num_nodes=1,
        gpus_per_node=2,
        nccl_channels=112,
        compute_channels=144,
        rccl_branch="develop",
    )


def _variant():
    return {
        "num_nodes": 1,
        "gpus_per_node": 2,
        "total_gpus": 2,
        "nccl_channels": 112,
        "compute_channels": 144,
        "rccl_branch": "develop",
        "base_config": "config/profile.yaml",
        "image": "example/aorta:latest",
        "thresholds": {"max_avg_iteration_ms": 20},
    }


class TestAortaRunDeck(unittest.TestCase):
    def test_builds_series_from_real_aorta_metric_contract(self):
        series = build_aorta_series(_benchmark_result(), "TraceLensParser")

        ranks = series["TraceLensParser per-rank trace"]
        self.assertEqual(list(ranks), ["0", "1"])
        self.assertEqual(ranks["0"]["total_time_ms"], 10.0)
        self.assertEqual(ranks["0"]["compute_time_ms"], 7.0)
        self.assertEqual(ranks["0"]["communication_time_ms"], 4.0)
        self.assertEqual(ranks["0"]["overlap_ratio_pct"], 25.0)
        self.assertEqual(ranks["0"]["peak_memory_gb"], 42.5)
        self.assertIsNone(ranks["1"]["peak_memory_gb"])

    def test_updates_and_formats_run_summary(self):
        variant = _variant()
        run_result = RunResult(
            status=RunStatus.COMPLETED,
            start_time=10,
            end_time=12.5,
            artifacts={"torch_traces": "/tmp/traces"},
            metadata={"launch_mode": "torchrun"},
        )

        update_aorta_run_summary(variant, _benchmark_result(), "TraceLensParser", run_result)
        rows = aorta_run_card_display(variant, {})
        values = {label: value for label, value, _is_link in rows}

        self.assertEqual(values["Status"], "completed")
        self.assertEqual(values["Parser"], "TraceLensParser")
        self.assertEqual(values["Avg iteration"], "11.00 ms")
        self.assertEqual(values["Compute"], "65.00%")
        self.assertEqual(values["Compute/communication overlap"], "12.50%")
        self.assertEqual(values["Artifacts"], "torch_traces")

    def test_profile_builds_static_deck_with_graphs_and_full_table(self):
        result = _benchmark_result()
        variant = _variant()
        update_aorta_run_summary(variant, result, "AortaReportParser")
        profile = load_json_profile("test_aorta")

        payload = build_rundeck_payload(
            profile=profile,
            store={
                "cvs_results_dict": build_aorta_series(result, "AortaReportParser"),
                "variant_config": variant,
            },
            cvs_version="1.0.0",
        )
        doc = render_rundeck_html(payload)

        self.assertEqual(payload["overall_status"], "record")
        self.assertNotIn("viewer_config", payload)
        self.assertEqual(payload["metric_contract"], {"id": "aorta-tracelens", "version": 1})
        self.assertEqual(len(payload["results_table"]["rows"]), 2)
        self.assertEqual(
            set(payload["datasets"]["series"]["charts"]),
            {
                "total_time_ms",
                "compute_time_ms",
                "communication_time_ms",
                "compute_ratio_pct",
                "communication_ratio_pct",
                "overlap_ratio_pct",
            },
        )
        self.assertIn("Aorta Run Deck", doc)
        self.assertIn("Iteration time by rank", doc)
        self.assertIn("Exposed communication time by rank", doc)
        self.assertIn("Full per-rank trace results", doc)
        self.assertNotIn(">Viewer</a>", doc)


if __name__ == "__main__":
    unittest.main()
