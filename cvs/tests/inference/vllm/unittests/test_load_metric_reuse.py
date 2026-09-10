'''Unit tests for vLLM server load metric reuse.'''

import unittest
from types import SimpleNamespace
from unittest import mock

from cvs.lib.inference.utils.vllm_metrics import UnknownMetricContractError, project_vllm_metrics
from cvs.tests.inference.vllm import _common


def _run(concurrency):
    cell = SimpleNamespace(
        key=f"ISL=128,OSL=128,TP=1,PP=1,CONC={concurrency}",
        isl=128,
        osl=128,
        concurrency=concurrency,
    )
    return SimpleNamespace(cell=cell, benchmark_params={"num_prompts": 2})


def _job(signature):
    job = mock.MagicMock()
    job.server_signature.return_value = signature
    job.nnodes = "1"
    job.hosts = ("head",)
    job.base_url = "http://0.0.0.0"
    job.port_no = "8888"
    job.parse_results.return_value = {"head": {"output_throughput": 1.0}}
    return job


def _request(nodeid):
    return SimpleNamespace(
        node=SimpleNamespace(nodeid=nodeid),
        config=SimpleNamespace(
            option=SimpleNamespace(htmlpath=None),
            _test_html_dir="test_html",
        ),
    )


def _lifecycle():
    return SimpleNamespace(
        failed=False,
        record=mock.Mock(),
        ib_hcas=[],
        live_server_sig=None,
        live_server_job=None,
        model_load_s=None,
        model_load_memory_mb=None,
    )


class TestLoadMetricReuse(unittest.TestCase):
    def setUp(self):
        self.variant = SimpleNamespace(model_id="model")
        self.results = {}
        self.lifecycle = _lifecycle()
        self.patches = [
            mock.patch.object(_common, "start_gpu_poller", return_value="poller"),
            mock.patch.object(_common, "stop_and_collect_gpu_poller", return_value=[]),
            mock.patch.object(_common, "scrape_vllm_metrics", return_value=None),
            mock.patch.object(
                _common,
                "agg_readings",
                return_value={
                    "peak_gpu_memory_mb": 0,
                    "gpu_bandwidth_util_pct": 0,
                    "gpu_compute_util_pct": 0,
                },
            ),
            mock.patch.object(_common, "to_prom_metrics", return_value={}),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def invoke(self, run, job, snapshots):
        with (
            mock.patch.object(_common, "VllmJob", return_value=job),
            mock.patch.object(_common, "_gpu_snap", side_effect=snapshots) as gpu_snap,
            mock.patch.object(_common.time, "monotonic", side_effect=[10.0, 12.0]),
        ):
            _common.test_vllm_inference(
                mock.Mock(),
                self.variant,
                "",
                (("head",),),
                run,
                self.results,
                self.lifecycle,
                _request(f"test_vllm_inference[{run.cell.key}]"),
            )
        return gpu_snap

    def metrics_for(self, run):
        key = _common._cell_result_key(self.variant, run)
        return self.results[key]["head"]

    def test_reuses_valid_load_measurements_for_same_server(self):
        first = _run(1)
        first_job = _job(("same",))
        self.invoke(
            first,
            first_job,
            [{"gpu.used_vram": 100.0}, {"gpu.used_vram": 125.0}],
        )

        second = _run(2)
        second_job = _job(("same",))
        gpu_snap = self.invoke(second, second_job, [])

        self.assertEqual(gpu_snap.call_count, 0)
        second_job.start_server.assert_not_called()
        self.assertEqual(self.metrics_for(first)["model_load_s"], 2.0)
        self.assertEqual(self.metrics_for(first)["model_load_memory_mb"], 25.0)
        self.assertEqual(self.metrics_for(second)["model_load_s"], 2.0)
        self.assertEqual(self.metrics_for(second)["model_load_memory_mb"], 25.0)

    def test_changed_signature_restarts_and_replaces_measurements(self):
        first = _run(1)
        self.invoke(
            first,
            _job(("first",)),
            [{"gpu.used_vram": 100}, {"gpu.used_vram": 125}],
        )
        second = _run(2)
        second_job = _job(("second",))
        self.invoke(
            second,
            second_job,
            [{"gpu.used_vram": 200}, {"gpu.used_vram": 230}],
        )

        second_job.stop_server.assert_called_once()
        second_job.start_server.assert_called_once()
        self.assertEqual(self.lifecycle.live_server_sig, ("second",))
        self.assertIs(self.lifecycle.live_server_job, second_job)
        self.assertEqual(self.lifecycle.model_load_s, 2.0)
        self.assertEqual(self.lifecycle.model_load_memory_mb, 30)

    def test_unavailable_snapshot_reuses_server_and_elapsed_load_time(self):
        first = _run(1)
        first_job = _job(("same",))
        self.invoke(first, first_job, [{}, {"gpu.used_vram": 125}])

        self.assertEqual(self.lifecycle.live_server_sig, ("same",))
        self.assertIs(self.lifecycle.live_server_job, first_job)
        self.assertEqual(self.lifecycle.model_load_s, 2.0)
        self.assertIsNone(self.lifecycle.model_load_memory_mb)
        self.assertEqual(self.metrics_for(first)["model_load_s"], 2.0)
        self.assertIsNone(self.metrics_for(first)["model_load_memory_mb"])

        second = _run(2)
        second_job = _job(("same",))
        gpu_snap = self.invoke(second, second_job, [])

        self.assertEqual(gpu_snap.call_count, 0)
        second_job.start_server.assert_not_called()
        self.assertEqual(self.metrics_for(second)["model_load_s"], 2.0)
        self.assertIsNone(self.metrics_for(second)["model_load_memory_mb"])

    def test_zero_memory_delta_is_preserved_and_reused(self):
        first = _run(1)
        self.invoke(
            first,
            _job(("same",)),
            [{"gpu.used_vram": 100}, {"gpu.used_vram": 100}],
        )
        second = _run(2)
        self.invoke(second, _job(("same",)), [])

        self.assertEqual(self.metrics_for(first)["model_load_memory_mb"], 0)
        self.assertEqual(self.metrics_for(second)["model_load_memory_mb"], 0)
        self.assertEqual(self.lifecycle.model_load_memory_mb, 0)

    def test_contract_drift_marks_lifecycle_failed_and_clears_server(self):
        run = _run(1)
        job = _job(("same",))
        job.parse_results.side_effect = lambda: project_vllm_metrics(
            {"output_throughput": 1.0, "new_metric": 2.0},
            tp=1,
            pp=1,
            isl=128,
            artifact_path="head:/tmp/results",
        )

        with self.assertRaises(UnknownMetricContractError):
            self.invoke(
                run,
                job,
                [{"gpu.used_vram": 100}, {"gpu.used_vram": 125}],
            )

        self.assertTrue(self.lifecycle.failed)
        self.assertIsNone(self.lifecycle.live_server_sig)
        self.assertIsNone(self.lifecycle.live_server_job)
        self.assertIsNone(self.lifecycle.model_load_s)
        self.assertIsNone(self.lifecycle.model_load_memory_mb)
        self.assertNotIn(_common._cell_result_key(self.variant, run), self.results)
        job.dump_server_log.assert_called_once()

    def test_generic_projection_value_error_stops_later_cells(self):
        run = _run(1)
        job = _job(("same",))
        job.parse_results.side_effect = ValueError("invalid result artifact")

        with self.assertRaisesRegex(ValueError, "invalid result artifact"):
            self.invoke(
                run,
                job,
                [{"gpu.used_vram": 100}, {"gpu.used_vram": 125}],
            )

        self.assertTrue(self.lifecycle.failed)
        self.assertIsNone(self.lifecycle.live_server_sig)
        self.assertIsNone(self.lifecycle.live_server_job)
        self.assertIsNone(self.lifecycle.model_load_s)
        self.assertIsNone(self.lifecycle.model_load_memory_mb)
        job.dump_server_log.assert_called_once()

    def test_constructor_failure_clears_state_without_unbound_job(self):
        run = _run(1)

        with (
            mock.patch.object(_common, "VllmJob", side_effect=RuntimeError("construction failed")),
            self.assertRaisesRegex(RuntimeError, "construction failed"),
        ):
            _common.test_vllm_inference(
                mock.Mock(),
                self.variant,
                "",
                (("head",),),
                run,
                self.results,
                self.lifecycle,
                _request("test_vllm_inference[construction-failure]"),
            )

        self.assertTrue(self.lifecycle.failed)
        self.assertIsNone(self.lifecycle.live_server_sig)
        self.assertIsNone(self.lifecycle.live_server_job)
        self.assertIsNone(self.lifecycle.model_load_s)
        self.assertIsNone(self.lifecycle.model_load_memory_mb)

    def test_merge_failure_does_not_publish_partial_results(self):
        run = _run(1)
        job = _job(("same",))
        job.parse_results.return_value = {
            "head": {"output_throughput": 1.0},
            "worker": {"output_throughput": 2.0},
        }
        merge_calls = 0

        def fail_second_merge(client_metrics, gpu_metrics, prom_metrics):
            nonlocal merge_calls
            merge_calls += 1
            if merge_calls == 2:
                raise ValueError("merge failed")
            return {**client_metrics, **gpu_metrics, **prom_metrics}

        with (
            mock.patch.object(_common, "merge_metric_sources", side_effect=fail_second_merge),
            self.assertRaisesRegex(ValueError, "merge failed"),
        ):
            self.invoke(
                run,
                job,
                [{"gpu.used_vram": 100}, {"gpu.used_vram": 125}],
            )

        self.assertTrue(self.lifecycle.failed)
        self.assertNotIn(_common._cell_result_key(self.variant, run), self.results)
        self.assertIsNone(self.lifecycle.live_server_sig)
        job.dump_server_log.assert_called_once()

    def test_success_publishes_all_hosts_atomically(self):
        run = _run(1)
        job = _job(("same",))
        parsed = {
            "head": {"output_throughput": 1.0},
            "worker": {"output_throughput": 2.0},
        }
        job.parse_results.return_value = parsed

        self.invoke(
            run,
            job,
            [{"gpu.used_vram": 100}, {"gpu.used_vram": 125}],
        )

        published = self.results[_common._cell_result_key(self.variant, run)]
        self.assertEqual(set(published), {"head", "worker"})
        self.assertEqual(published["head"]["output_throughput"], 1.0)
        self.assertEqual(published["worker"]["output_throughput"], 2.0)
        self.assertIsNot(published, parsed)
        self.assertEqual(
            parsed,
            {
                "head": {"output_throughput": 1.0},
                "worker": {"output_throughput": 2.0},
            },
        )

    def test_restart_failure_clears_server_identity_and_measurements(self):
        run = _run(1)
        job = _job(("new",))
        job.stop_server.side_effect = RuntimeError("stop failed")
        self.lifecycle.live_server_sig = ("old",)
        self.lifecycle.live_server_job = mock.Mock()
        self.lifecycle.model_load_s = 1.0
        self.lifecycle.model_load_memory_mb = 2.0

        with (
            mock.patch.object(_common, "VllmJob", return_value=job),
            self.assertRaisesRegex(RuntimeError, "stop failed"),
        ):
            _common.test_vllm_inference(
                mock.Mock(),
                self.variant,
                "",
                (("head",),),
                run,
                self.results,
                self.lifecycle,
                _request("test_vllm_inference[restart-failure]"),
            )

        self.assertIsNone(self.lifecycle.live_server_sig)
        self.assertIsNone(self.lifecycle.live_server_job)
        self.assertIsNone(self.lifecycle.model_load_s)
        self.assertIsNone(self.lifecycle.model_load_memory_mb)

    def test_failure_clears_server_identity_and_measurements(self):
        run = _run(1)
        job = _job(("same",))
        job.run_client.side_effect = RuntimeError("client failed")
        self.lifecycle.live_server_sig = ("old",)
        self.lifecycle.live_server_job = mock.Mock()
        self.lifecycle.model_load_s = 1.0
        self.lifecycle.model_load_memory_mb = 2.0

        with (
            mock.patch.object(_common, "VllmJob", return_value=job),
            mock.patch.object(
                _common,
                "_gpu_snap",
                side_effect=[{"gpu.used_vram": 100}, {"gpu.used_vram": 125}],
            ),
            mock.patch.object(_common.time, "monotonic", side_effect=[10.0, 12.0]),
            self.assertRaisesRegex(RuntimeError, "client failed"),
        ):
            _common.test_vllm_inference(
                mock.Mock(),
                self.variant,
                "",
                (("head",),),
                run,
                self.results,
                self.lifecycle,
                _request("test_vllm_inference[failure]"),
            )

        self.assertIsNone(self.lifecycle.live_server_sig)
        self.assertIsNone(self.lifecycle.live_server_job)
        self.assertIsNone(self.lifecycle.model_load_s)
        self.assertIsNone(self.lifecycle.model_load_memory_mb)


if __name__ == "__main__":
    unittest.main()
