"""Unit tests for xDiT pytest wiring helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cvs.tests.inference.xdit import conftest
from cvs.tests.inference.xdit._shared import (
    Lifecycle,
    _output_dirs_by_host,
    _report_dimensions,
    _report_threshold,
    benchmark_params_from_variant,
    inference_from_variant,
    log_topology,
    resolve_execution_hosts,
    scoped_cluster_dict,
    suite_spec,
)


class _Variant:
    def __init__(self):
        self.inference = {"model_repo": "/models/flux"}
        self.benchmark_params = {"flux1_dev_t2i": {}}


class TestVariantHelpers(unittest.TestCase):
    def test_reads_object_variant(self):
        variant = _Variant()

        self.assertEqual(inference_from_variant(variant), variant.inference)
        self.assertEqual(benchmark_params_from_variant(variant), variant.benchmark_params)

    def test_reads_mapping_variant(self):
        variant = {"config": {"model_repo": "/models/wan"}, "benchmark_params": {"wan22_i2v_a14b": {}}}

        self.assertEqual(inference_from_variant(variant), variant["config"])
        self.assertEqual(benchmark_params_from_variant(variant), variant["benchmark_params"])


class TestSuiteSpec(unittest.TestCase):
    def test_flux_single(self):
        self.assertEqual(
            suite_spec("cvs.tests.inference.xdit.pytorch_xdit_flux_dev_single"),
            {"family": "flux", "distributed": False, "diffusers": False},
        )

    def test_wan_diffusers_distributed(self):
        self.assertEqual(
            suite_spec("pytorch_xdit_wan22_14b_diffusers_distributed"),
            {"family": "wan", "distributed": True, "diffusers": True},
        )


class TestReportResults(unittest.TestCase):
    def test_flux_dimensions_use_shape_steps_and_total_workers(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="black-forest-labs/FLUX.1-dev"),
            inference={"nnodes": 2},
        )

        values = _report_dimensions(
            variant,
            {"height": 1024, "width": 768, "num_inference_steps": 25, "torchrun_nproc": 8},
            {"family": "flux", "distributed": True, "diffusers": False},
        )

        self.assertEqual(values[:5], ("black-forest-labs/FLUX.1-dev", "1024x768", 25, "diffusers", 16))
        self.assertEqual(values[5], "ISL=1024x768,OSL=25,C=16")

    def test_distributed_workers_use_execution_hosts_when_nnodes_missing(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="FLUX.1-dev"),
            inference={"_execution_hosts": ["10.0.0.1", "10.0.0.2"]},
        )

        values = _report_dimensions(
            variant,
            {"height": 1024, "width": 1024, "num_inference_steps": 25, "torchrun_nproc": 8},
            {"family": "flux", "distributed": True, "diffusers": False},
        )

        self.assertEqual(values[4], 16)
        self.assertEqual(values[5], "ISL=1024x1024,OSL=25,C=16")

    def test_report_threshold_uses_gpu_then_auto_fallback(self):
        thresholds = {
            "mi300x": {"max_avg_pipe_time_s": 3.0},
            "auto": {"max_avg_pipe_time_s": 10.0},
        }

        self.assertEqual(_report_threshold(thresholds, "mi300x", "avg_pipe_time_s"), {"kind": "max", "value": 3.0})
        self.assertEqual(_report_threshold(thresholds, "other", "avg_pipe_time_s"), {"kind": "max", "value": 10.0})


class TestOutputDirsByHost(unittest.TestCase):
    def test_prefers_per_node_map(self):
        lifecycle = Lifecycle()
        inference = {
            "_test_output_dirs_by_node": {"node-a": "/out/a", "node-b": "/out/b"},
            "_test_output_dir": "/out/a",
        }

        self.assertEqual(_output_dirs_by_host(inference, lifecycle), {"node-a": "/out/a", "node-b": "/out/b"})

    def test_falls_back_to_benchmark_host(self):
        lifecycle = Lifecycle()
        lifecycle.benchmark_host = "node-a"

        self.assertEqual(_output_dirs_by_host({"_test_output_dir": "/out/a"}, lifecycle), {"node-a": "/out/a"})

    def test_collapses_hosts_that_share_one_output_dir(self):
        lifecycle = Lifecycle()
        lifecycle.benchmark_host = "10.32.80.110"
        inference = {
            "_test_output_dirs_by_node": {
                "10.32.80.110": "/out/flux_rank0_outputs",
                "10.32.80.111": "/out/flux_rank0_outputs",
            }
        }

        self.assertEqual(
            _output_dirs_by_host(inference, lifecycle),
            {"10.32.80.110": "/out/flux_rank0_outputs"},
        )


class TestLogTopology(unittest.TestCase):
    def _messages(self, variant, cluster_dict, spec):
        with patch("cvs.tests.inference.xdit._shared.log") as mock_log:
            log_topology(variant, cluster_dict, spec)
        return [str(call.args[0]) % tuple(call.args[1:]) for call in mock_log.info.call_args_list]

    def test_distributed_flux_lists_ranks_and_world_size(self):
        variant = SimpleNamespace(
            inference={"nnodes": 2, "master_addr": "10.0.0.1", "master_port": 29502},
            benchmark_params={
                "flux1_dev_t2i": {"torchrun_nproc": 8, "ulysses_degree": 8, "ring_degree": 2},
            },
        )
        cluster = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}

        messages = self._messages(variant, cluster, {"family": "flux", "distributed": True, "diffusers": False})

        self.assertIn("Distributed FLUX topology", messages)
        self.assertIn("  rank 0 -> 10.0.0.1 (8 GPUs)", messages)
        self.assertIn("  rank 1 -> 10.0.0.2 (8 GPUs)", messages)
        self.assertIn("Total GPU ranks (world_size): 16 = 2 nodes × 8 nproc", messages)
        self.assertIn("Rendezvous: 10.0.0.1:29502", messages)
        self.assertIn("Parallelism check: PASS (product 16 == world_size 16)", messages)

    def test_single_node_lists_one_job_per_host(self):
        variant = SimpleNamespace(
            inference={"_execution_hosts": ["10.0.0.1", "10.0.0.2"]},
            benchmark_params={
                "flux1_dev_t2i": {"torchrun_nproc": 8, "ulysses_degree": 8, "ring_degree": 1},
            },
        )

        messages = self._messages(variant, {}, {"family": "flux", "distributed": False, "diffusers": False})

        self.assertIn("Single-node FLUX topology", messages)
        self.assertIn("Independent jobs: 2 (one per node)", messages)
        self.assertIn("Total GPU ranks (world_size): 8 = 1 nodes × 8 nproc", messages)
        self.assertNotIn("Rendezvous: :29500", messages)

    def test_wan_uses_ulysses_and_ring_sizes(self):
        variant = SimpleNamespace(
            inference={"_execution_hosts": ["10.0.0.1"]},
            benchmark_params={"wan22_i2v_a14b": {"torchrun_nproc": 8, "ulysses_size": 8}},
        )

        messages = self._messages(variant, {}, {"family": "wan", "distributed": False, "diffusers": False})

        self.assertIn("Single-node WAN topology", messages)
        self.assertIn("xDiT parallel layout: ulysses=8 × ring=1 = 8", messages)


class TestUlyssesRing(unittest.TestCase):
    def test_flux_reads_degree_keys(self):
        from cvs.tests.inference.xdit._shared import _ulysses_ring

        self.assertEqual(
            _ulysses_ring(
                {"ulysses_degree": 8, "ring_degree": 1},
                {"family": "flux"},
            ),
            (8, 1),
        )

    def test_wan_reads_size_keys(self):
        from cvs.tests.inference.xdit._shared import _ulysses_ring

        self.assertEqual(
            _ulysses_ring(
                {"torchrun_nproc": 8, "ulysses_size": 4, "ring_size": 2},
                {"family": "wan"},
            ),
            (4, 2),
        )


class TestHostScoping(unittest.TestCase):
    def setUp(self):
        self.cluster = {
            "node_dict": {
                "node-a": {"mgmt_ip": "node-a"},
                "node-b": {"mgmt_ip": "node-b"},
                "unused": {"mgmt_ip": "unused"},
            },
            "username": "tester",
        }

    def test_single_uses_every_cluster_node(self):
        hosts = resolve_execution_hosts(self.cluster, {}, distributed=False)

        self.assertEqual(hosts, ["node-a", "node-b", "unused"])

    def test_single_ignores_benchmark_serv_node(self):
        hosts = resolve_execution_hosts(
            self.cluster,
            {"benchmark_serv_node": "node-b"},
            distributed=False,
        )

        self.assertEqual(hosts, ["node-a", "node-b", "unused"])

    def test_scoped_cluster_excludes_unrelated_hosts(self):
        scoped = scoped_cluster_dict(self.cluster, ["node-a", "node-b"])

        self.assertEqual(list(scoped["node_dict"]), ["node-a", "node-b"])
        self.assertEqual(scoped["head_node_dict"], {"mgmt_ip": "node-a"})
        self.assertEqual(scoped["username"], "tester")

    def test_empty_cluster_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "could not resolve an execution host"):
            resolve_execution_hosts({"node_dict": {}}, {}, distributed=False)


class TestConftestHelpers(unittest.TestCase):
    def test_deep_merge_preserves_cluster_runtime_fields(self):
        merged = conftest._deep_merge(
            {"runtime": {"name": "docker", "args": {"network": "host"}}, "lifetime": "per_run"},
            {"runtime": {"args": {"ipc": "host"}}, "image": "xdit:latest"},
        )

        self.assertEqual(merged["runtime"]["name"], "docker")
        self.assertEqual(merged["runtime"]["args"], {"network": "host", "ipc": "host"})
        self.assertEqual(merged["image"], "xdit:latest")


if __name__ == "__main__":
    unittest.main()
