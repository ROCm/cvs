'''Unit tests for vLLM server load metric reuse.'''

import unittest
from types import SimpleNamespace
from unittest import mock

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
    return SimpleNamespace(failed=False, record=mock.Mock(), ib_hcas=[])


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
        self.assertEqual(self.lifecycle.live_server_state, (("second",), 2.0, 30))

    def test_unavailable_snapshot_does_not_create_reusable_state(self):
        first = _run(1)
        self.invoke(first, _job(("same",)), [{}, {"gpu.used_vram": 125}])

        self.assertIsNone(self.lifecycle.live_server_state)
        self.assertIsNone(self.metrics_for(first)["model_load_s"])
        self.assertIsNone(self.metrics_for(first)["model_load_memory_mb"])

        second = _run(2)
        second_job = _job(("same",))
        self.invoke(
            second,
            second_job,
            [{"gpu.used_vram": 125}, {"gpu.used_vram": 150}],
        )
        second_job.start_server.assert_called_once()

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

    def test_failure_clears_live_server_state(self):
        run = _run(1)
        job = _job(("same",))
        job.run_client.side_effect = RuntimeError("client failed")
        self.lifecycle.live_server_state = (("old",), 1.0, 2.0)
        self.lifecycle.live_server_job = mock.Mock()

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

        self.assertIsNone(self.lifecycle.live_server_state)
        self.assertIsNone(self.lifecycle.live_server_job)


if __name__ == "__main__":
    unittest.main()
