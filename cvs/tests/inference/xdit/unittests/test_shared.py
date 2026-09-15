"""Unit tests for xDiT pytest wiring helpers."""

import unittest
from types import SimpleNamespace

from cvs.tests.inference.xdit import conftest
from cvs.tests.inference.xdit._shared import (
    _report_dimensions,
    _report_threshold,
    benchmark_params_from_variant,
    inference_from_variant,
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

    def test_report_threshold_uses_gpu_then_auto_fallback(self):
        thresholds = {
            "mi300x": {"max_avg_pipe_time_s": 3.0},
            "auto": {"max_avg_pipe_time_s": 10.0},
        }

        self.assertEqual(_report_threshold(thresholds, "mi300x", "avg_pipe_time_s"), {"kind": "max", "value": 3.0})
        self.assertEqual(_report_threshold(thresholds, "other", "avg_pipe_time_s"), {"kind": "max", "value": 10.0})


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

    def test_single_uses_only_configured_benchmark_host(self):
        hosts = resolve_execution_hosts(
            self.cluster,
            {"benchmark_serv_node": "node-b"},
            distributed=False,
        )

        self.assertEqual(hosts, ["node-b"])

    def test_scoped_cluster_excludes_unrelated_hosts(self):
        scoped = scoped_cluster_dict(self.cluster, ["node-a", "node-b"])

        self.assertEqual(list(scoped["node_dict"]), ["node-a", "node-b"])
        self.assertEqual(scoped["head_node_dict"], {"mgmt_ip": "node-a"})
        self.assertEqual(scoped["username"], "tester")

    def test_missing_single_host_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            resolve_execution_hosts(
                self.cluster,
                {"benchmark_serv_node": "missing"},
                distributed=False,
            )

    def test_single_requires_benchmark_host(self):
        with self.assertRaisesRegex(ValueError, "requires benchmark_serv_node"):
            resolve_execution_hosts(self.cluster, {}, distributed=False)


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
