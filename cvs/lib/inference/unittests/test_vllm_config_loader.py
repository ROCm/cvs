'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.vllm_config_loader's gpu.*/prom.*
threshold coverage in _check_thresholds_cover_sweep. No hardware.

_check_thresholds_cover_sweep only requires that every sweep cell have a
threshold entry -- it never requires that entry name every gated metric.
Operators may gate only the metrics they care about; test_metric/
test_gpu_metric/test_prom_metric already treat an absent spec as
"don't gate this metric" at evaluation time.
'''

import unittest

from cvs.lib.inference.utils.vllm_config_loader import (
    GATED_GPU_METRICS,
    GATED_PROM_METRICS,
    Run,
    SeqCombo,
    Sweep,
    VariantConfig,
)
from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS


def _combo(name, isl="128", osl="2048"):
    return SeqCombo(name=name, isl=isl, osl=osl)


def _full_gated_specs():
    """A spec for every gated client.*, gpu.*, and prom.* metric -- the
    minimum that satisfies coverage. Values are inert so the set passes
    without asserting anything; these tests pin the coverage gate, not the
    numbers."""
    out = {}
    for m in GATED_METRICS:
        kind = "max_ms" if m.endswith("_ms") else "max" if m == "failed" else "min"
        out[f"client.{m}"] = {"kind": kind, "value": 0 if kind == "min" else 1e12}
    for m in GATED_GPU_METRICS:
        kind = "max" if m in ("peak_gpu_memory_mb", "model_load_memory_mb", "model_load_s") else "min"
        out[f"gpu.{m}"] = {"kind": kind, "value": 0 if kind == "min" else 1e12}
    for m in GATED_PROM_METRICS:
        out[f"prom.{m}"] = {"kind": "max_ms", "value": 1e12}
    return out


class TestGpuGatedMetricCoverage(unittest.TestCase):
    """The gpu.* axis of vllm_config_loader's _check_thresholds_cover_sweep."""

    _CELL = "ISL=128,OSL=2048,TP=8,CONC=16"

    def _variant_with(self, thresholds, enforce):
        sw = Sweep(
            sequence_combinations=[_combo("a")],
            runs=[Run(combo="a", concurrency=16)],
        )
        return VariantConfig(
            schema_version=1,
            framework="vllm",
            gpu_arch="mi300x",
            enforce_thresholds=enforce,
            paths={
                "shared_fs": "/home/x",
                "models_dir": "/home/x/models",
                "log_dir": "/home/x/LOGS",
                "hf_token_file": "/home/x/.hf",
            },
            model={"id": "amd/Llama-3.1-70B-Instruct-FP8-KV", "remote": 0},
            params={"tensor_parallelism": "8"},
            sweep=sw,
            thresholds=thresholds,
        )

    def test_full_gated_set_constructs(self):
        vc = self._variant_with({self._CELL: _full_gated_specs()}, enforce=True)
        self.assertEqual(vc.enforce_thresholds, True)

    def test_missing_gpu_metric_does_not_raise_when_enforced(self):
        # Operators may gate only a subset of gpu.* metrics; an absent one is
        # simply not gated, not an authoring error.
        specs = _full_gated_specs()
        del specs["gpu.peak_gpu_memory_mb"]
        vc = self._variant_with({self._CELL: specs}, enforce=True)
        self.assertNotIn("gpu.peak_gpu_memory_mb", vc.thresholds[self._CELL])

    def test_no_gpu_specs_at_all_does_not_raise_when_enforced(self):
        vc = self._variant_with({self._CELL: {}}, enforce=True)
        self.assertEqual(vc.thresholds[self._CELL], {})

    def test_all_five_gpu_metrics_are_gated(self):
        self.assertEqual(
            GATED_GPU_METRICS,
            {
                "peak_gpu_memory_mb",
                "model_load_memory_mb",
                "model_load_s",
                "gpu_bandwidth_util_pct",
                "gpu_compute_util_pct",
            },
        )


class TestPromGatedMetricCoverage(unittest.TestCase):
    """The prom.* axis of vllm_config_loader's _check_thresholds_cover_sweep.

    Mirrors TestGpuGatedMetricCoverage: prom.* is a fully separate, parallel
    gated family, not part of client.*'s tiering machinery, so its coverage
    is proven independently here rather than in test_vllm_report_preset.py.
    """

    _CELL = "ISL=128,OSL=2048,TP=8,CONC=16"

    def _variant_with(self, thresholds, enforce):
        sw = Sweep(
            sequence_combinations=[_combo("a")],
            runs=[Run(combo="a", concurrency=16)],
        )
        return VariantConfig(
            schema_version=1,
            framework="vllm",
            gpu_arch="mi300x",
            enforce_thresholds=enforce,
            paths={
                "shared_fs": "/home/x",
                "models_dir": "/home/x/models",
                "log_dir": "/home/x/LOGS",
                "hf_token_file": "/home/x/.hf",
            },
            model={"id": "amd/Llama-3.1-70B-Instruct-FP8-KV", "remote": 0},
            params={"tensor_parallelism": "8"},
            sweep=sw,
            thresholds=thresholds,
        )

    def test_full_gated_set_constructs(self):
        vc = self._variant_with({self._CELL: _full_gated_specs()}, enforce=True)
        self.assertEqual(vc.enforce_thresholds, True)

    def test_missing_prom_metric_does_not_raise_when_enforced(self):
        # Operators may gate only a subset of prom.* metrics; an absent one is
        # simply not gated, not an authoring error.
        specs = _full_gated_specs()
        del specs["prom.queue_time_p50_ms"]
        vc = self._variant_with({self._CELL: specs}, enforce=True)
        self.assertNotIn("prom.queue_time_p50_ms", vc.thresholds[self._CELL])

    def test_only_one_prom_metric_gated_does_not_raise_when_enforced(self):
        specs = {"prom.queue_time_p50_ms": {"kind": "max_ms", "value": 200}}
        vc = self._variant_with({self._CELL: specs}, enforce=True)
        self.assertEqual(vc.thresholds[self._CELL], specs)

    def test_all_four_prom_metrics_are_gated(self):
        self.assertEqual(
            GATED_PROM_METRICS,
            {
                "queue_time_p50_ms",
                "queue_time_p95_ms",
                "prefill_time_p50_ms",
                "prefill_time_p95_ms",
            },
        )


if __name__ == "__main__":
    unittest.main()
