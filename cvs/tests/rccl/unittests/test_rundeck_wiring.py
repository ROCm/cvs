'''Unit tests for RCCL report wiring; cluster jobs are mocked.'''

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cvs.lib import globals, rccl_lib
from cvs.lib.report import benchmark_metric_registry as registry
from cvs.lib.report.profiles.hooks.rccl_run_card import rccl_run_card_display
from cvs.tests.rccl import conftest, rccl_pairwise, rccl_perf, rccl_regression


class _CapturingSubtests:
    def __init__(self):
        self.calls = []
        self.failures = []

    @contextmanager
    def test(self, **kwargs):
        self.calls.append(kwargs)
        try:
            yield
        except AssertionError as exc:
            self.failures.append(str(exc))


class TestRundeckWiring(unittest.TestCase):
    def setUp(self):
        self.raw = [
            {"name": "all_reduce_perf", "size": 1024, "inPlace": 1, "busBw": 350.0, "algBw": 200.0, "time": 10.0}
        ]
        patcher = patch.object(globals, "error_list", [])
        patcher.start()
        self.addCleanup(patcher.stop)
        registry._ROWS_BY_NODEID.clear()
        registry._COLUMNS_BY_NODEID.clear()
        registry._SUBTEST_SUMMARY_COUNTED.clear()
        registry._SUBTEST_SUMMARY.update({'failed': 0, 'passed': 0, 'skipped': 0, 'recorded': 0})
        self.addCleanup(registry._ROWS_BY_NODEID.clear)

    def test_perf_and_regression_publish_the_same_graph_as_amcharts(self):
        for module in (rccl_perf, rccl_regression):
            with self.subTest(module=module.__name__):
                store = {}
                request = SimpleNamespace(config=SimpleNamespace(_html_report_manager=Mock()))
                with (
                    patch.object(module, "rccl_res_dict", {"all_reduce_perf": self.raw}),
                    patch.object(module, "html_lib") as html,
                ):
                    module.test_gen_graph(request, store)
                self.assertEqual(store["all_reduce_perf"][1024]["bus_bw"], 350.0)
                self.assertEqual(html.build_rccl_amcharts_graph.call_args.args[2], store)

    def test_regression_collectives_use_rccl_test_params_or_legacy_fallback(self):
        cases = (
            ({}, rccl_lib.DEFAULT_COLLECTIVES),
            ({"rccl_collective": ["broadcast_perf"]}, ["broadcast_perf"]),
            ({"rccl_test_params": {"rccl_collective": ["all_gather_perf"]}}, ["all_gather_perf"]),
            (
                {
                    "rccl_test_params": {"rccl_collective": ["all_gather_perf"]},
                    "rccl_collective": ["broadcast_perf"],
                },
                ["all_gather_perf"],
            ),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as temp_dir:
                rccl = {"regression": {"NCCL_ALGO": ["Ring"]}, **overrides}
                config_path = Path(temp_dir) / "rccl.json"
                config_path.write_text(json.dumps({"rccl": rccl}), encoding="utf-8")
                metafunc = SimpleNamespace(
                    config=SimpleNamespace(getoption=lambda _name: str(config_path)),
                    fixturenames=["rccl_collective"],
                    parametrize=Mock(),
                )
                rccl_regression.pytest_generate_tests(metafunc)
                self.assertEqual(metafunc.parametrize.call_args_list[0].args, ("rccl_collective", expected))

    def test_pairwise_phase_labels_preserve_every_run(self):
        config = {
            "mpi_params": {"no_of_nodes": "99", "no_of_local_ranks": "8"},
            "cvs_params": {"pairwise_min_bw": "300"},
        }
        cluster = {"node_dict": {"n0": {}, "n1": {}, "n2": {}}}
        fixtures = {"config_dict": config, "cluster_dict": cluster}
        request = SimpleNamespace(module=rccl_pairwise, getfixturevalue=fixtures.__getitem__)
        with (
            patch.object(rccl_pairwise, "rccl_res_dict", {}),
            patch.object(rccl_pairwise, "rccl_run_nodes", {}),
            patch.object(rccl_pairwise, "_phase1_survivors", None),
            patch.object(rccl_pairwise, "is_managed_compute", return_value=False),
            patch.object(rccl_pairwise, "_persist_pairwise_artifact"),
            patch.object(rccl_pairwise.rccl_lib.RcclJob, "from_config") as job,
        ):
            job.return_value.run_perf.return_value = self.raw
            variant = conftest.variant_config.__wrapped__(request)
            node_list = list(cluster["node_dict"])
            case_request = SimpleNamespace(
                node=SimpleNamespace(nodeid='cvs/tests/rccl/rccl_pairwise.py::test_rccl_pairwise', stash={})
            )
            rccl_pairwise.test_rccl_pairwise(
                None, node_list, config, ["v0", "v1", "v2"], case_request, _CapturingSubtests()
            )
            rccl_pairwise.test_rccl_incremental(
                None, node_list, config, ["v0", "v1", "v2"], case_request, _CapturingSubtests()
            )
            store = {}
            rccl_pairwise.test_gen_graph(store)
            self.assertEqual(job.call_count, 5)
            self.assertEqual(
                set(store),
                {
                    "Phase0 sanity n0",
                    "Phase1 n0 <-> n1",
                    "Phase1 n0 <-> n2",
                    "Phase2 2-node cluster (adding n1)",
                    "Phase2 3-node cluster (adding n2)",
                },
            )
            rows = {label: value for label, value, _ in rccl_run_card_display(variant, {})}
            self.assertEqual(rows["MPI nodes"], "1, 2, 3")
            self.assertEqual(rows["MPI ranks"], "8, 16, 24")
            self.assertEqual(rows["Collectives"], "all_reduce_perf")
        self.assertEqual(config["mpi_params"]["no_of_nodes"], "99")

    def test_pairwise_run_strips_legacy_flat_thresholds_but_keeps_nic_keyed(self):
        config = {
            "mpi_params": {"no_of_nodes": "99", "no_of_local_ranks": "8"},
            "cvs_params": {},
            "results": {
                "all_reduce_perf": {"bus_bw": {"1024": "100"}},
                "ainic": {"all_reduce_perf-float-16": {"1024": {"bus_bw": "100"}}},
            },
        }
        with (
            patch.object(rccl_pairwise, "get_passwordless_sudo_status", return_value={}),
            patch.object(rccl_pairwise.rccl_lib.RcclJob, "from_config") as from_config,
        ):
            from_config.return_value.run_perf.return_value = self.raw
            rccl_pairwise.run_pairwise_rccl(None, ["v0", "v1"], ["n0", "n1"], config, "Phase1 n0 <-> n1")
        passed_config = from_config.call_args.args[2]
        self.assertEqual(passed_config["results"], {"ainic": config["results"]["ainic"]})
        self.assertEqual(config["results"]["all_reduce_perf"], {"bus_bw": {"1024": "100"}})

    def test_pairwise_report_failure_does_not_repeat_previous_test_failures(self):
        with patch.object(rccl_pairwise, "publish_graph", side_effect=ValueError("invalid report data")):
            globals.error_list.append("earlier hardware failure")
            rccl_pairwise.test_gen_graph({})
        self.assertEqual(globals.error_list, [])

    def test_variant_fixture_failure_is_optional(self):
        request = SimpleNamespace(getfixturevalue=Mock(side_effect=ValueError("invalid config")))
        self.assertIsNone(conftest.variant_config.__wrapped__(request))


if __name__ == "__main__":
    unittest.main()
