"""Aorta Run Deck dataset, HTML, and publishing tests."""

import json
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.aorta_rundeck import AortaDeckVariant, deck_results
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.lib.report.auto_register import try_auto_register_suite_report
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.generate_rundeck import generate_rundeck
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report_plugins import HtmlReportManager
from cvs.parsers.schemas import AortaBenchmarkResult, AortaTraceMetrics
from cvs.parsers.tracelens import TraceLensParser


def _sources(checked=True, thresholds=None, parser=None, result=True):
    raw = variant_dict()
    if thresholds is not None:
        raw["thresholds"] = {"expected_results": thresholds}
    config = AortaVariantConfig.model_validate(raw)
    benchmark = (
        AortaBenchmarkResult.from_rank_metrics(
            [
                AortaTraceMetrics(rank=1, total_time_us=12000, compute_time_us=9000, communication_time_us=4000),
                AortaTraceMetrics(rank=0, total_time_us=10000, compute_time_us=8000, communication_time_us=3000),
            ],
            num_nodes=1,
            gpus_per_node=2,
            nccl_channels=112,
            rccl_branch="develop",
        )
        if result
        else None
    )
    lifecycle = SimpleNamespace(
        failed=False,
        thresholds_checked=checked,
        benchmark_result=benchmark,
        parser=parser or TraceLensParser(use_tracelens=False),
        deck_results=deck_results(benchmark, "test") if benchmark else {},
        report={"suite::test_run_benchmark": [("benchmark", 1.25, "s")]},
    )
    return lifecycle, AortaDeckVariant(config, lifecycle)


def _datasets(lifecycle, variant):
    return build_datasets(
        "training_sweep",
        {"results": lifecycle.deck_results, "variant": variant, "lifecycle_report": lifecycle.report},
        load_json_profile("aorta"),
    )


class TestAortaDeck(unittest.TestCase):
    def test_cells_statuses_table_and_charts(self):
        lifecycle, variant = _sources()
        datasets = _datasets(lifecycle, variant)
        self.assertEqual(len(datasets["cells"]), 1)
        self.assertEqual(datasets["cells"][0]["cell_id"], "test")
        self.assertEqual(list(datasets["cells"][0]["tiers"]), load_json_profile("aorta")["sweep"]["tier_order"])
        self.assertEqual(datasets["overall_status"], "pass")
        self.assertEqual(len(datasets["metric_charts"]["series"]), 6)
        headers = datasets["results_table"]["headers"]
        self.assertEqual(headers, [row[0] for row in load_json_profile("aorta")["sweep"]["results_table_columns"]])
        self.assertFalse(any(word in " ".join(headers).lower() for word in ("throughput", "tok", "exposed")))
        lifecycle, variant = _sources(thresholds={"max_avg_iteration_ms": 1, "min_compute_ratio": 0.01})
        datasets = _datasets(lifecycle, variant)
        self.assertEqual(datasets["overall_status"], "fail")
        self.assertEqual(datasets["cells"][0]["tiers"]["iteration_time"], "fail")
        self.assertEqual(datasets["cells"][0]["tiers"]["compute_ratio"], "pass")
        self.assertEqual(datasets["cells"][0]["tiers"]["overlap_ratio"], "na")
        self.assertEqual(datasets["cells"][0]["tiers"]["rank_balance"], "na")
        lifecycle.thresholds_checked = False
        self.assertEqual(_datasets(lifecycle, variant)["overall_status"], "record")
        lifecycle, variant = _sources(result=False)
        self.assertEqual(_datasets(lifecycle, variant)["overall_status"], "na")

    def test_gate_matrix_follows_parser_verdict(self):
        parser = MagicMock()
        parser.validate_thresholds.side_effect = lambda _result, expected: (
            ["gate says fail"] if "min_compute_ratio" in expected else []
        )
        lifecycle, variant = _sources(parser=parser, thresholds={"max_avg_iteration_ms": 20, "min_compute_ratio": 0.01})
        datasets = _datasets(lifecycle, variant)
        self.assertEqual(datasets["overall_status"], "fail")
        self.assertEqual(datasets["cells"][0]["tiers"]["compute_ratio"], "fail")
        self.assertEqual(datasets["gate_matrix"][0]["tiers"]["compute_ratio"], "fail")
        metric = next(m for m in datasets["cells"][0]["metrics"] if m["metric"] == "training.avg_compute_ratio")
        self.assertEqual(metric["status"], "fail")
        self.assertEqual(metric["spec"]["reason"], "gate says fail")

    def test_payload_and_html(self):
        lifecycle, variant = _sources()
        store = {
            "cvs_results_dict": lifecycle.deck_results,
            "variant_config": variant,
            "lifecycle_report": lifecycle.report,
        }
        payload = build_rundeck_payload(profile=load_json_profile("aorta"), store=store, cvs_version="test")
        self.assertIn("benchmark", payload["lifecycle"])
        html = render_rundeck_html(payload)
        for text in (
            "Aorta Run Deck",
            "Threshold gates",
            "Time by rank",
            "Share of iteration by rank",
            "max_avg_iteration_ms",
            "Metrics source",
        ):
            self.assertIn(text, html)

    def test_both_suites_publish_html_json_and_zip(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            root = Path(directory)
            patches.enter_context(
                patch("cvs.lib.report.rundeck.generate_rundeck.build_inference_report_provenance", return_value={})
            )
            patches.enter_context(patch("cvs.lib.report.rundeck.generate_rundeck.cvs_version", return_value="test"))
            patches.enter_context(patch("cvs.lib.report_plugins.sys.argv", ["cvs"]))
            old = getattr(HtmlReportManager, "_current_instance", None)
            self.addCleanup(setattr, HtmlReportManager, "_current_instance", old)
            for suite in ("aorta_single", "aorta_distributed"):
                with self.subTest(suite=suite):
                    html_path = root / f"{suite}.html"
                    html_path.write_text('<html><body><table id="environment"></table></body></html>')
                    config = SimpleNamespace(
                        _suite_name=suite,
                        _test_html_dir=f"{suite}_html",
                        _suite_report_config=None,
                        option=SimpleNamespace(htmlpath=str(html_path), log_file=None, self_contained_html=True),
                    )
                    self.assertTrue(try_auto_register_suite_report(config))
                    session, manager = SimpleNamespace(config=config), HtmlReportManager(config)
                    lifecycle, variant = _sources()
                    store = {
                        "cvs_results_dict": lifecycle.deck_results,
                        "variant_config": variant,
                        "lifecycle_report": lifecycle.report,
                    }
                    with patch("cvs.lib.report.rundeck.generate_rundeck.get_session_results", return_value=store):
                        artifacts = generate_rundeck(session, manager)
                    self.assertEqual(artifacts["html"], manager.log_dir / "aorta_run_deck.html")
                    self.assertTrue(artifacts["json"].is_file())
                    self.assertNotIn("viewer", artifacts)
                    self.assertNotIn("summary", artifacts)
                    self.assertEqual(json.loads(artifacts["json"].read_text())["overall_status"], "pass")
                    manager.create_zip_bundle(session)
                    self.assertIn("Aorta Run Deck", manager.htmlpath.read_text())
                    with zipfile.ZipFile(next(root.glob(f"{suite}_*.zip"))) as bundle:
                        self.assertIn(f"{suite}_html/aorta_run_deck.html", bundle.namelist())
                        self.assertIn(f"{suite}_html/aorta_run_deck.json", bundle.namelist())
