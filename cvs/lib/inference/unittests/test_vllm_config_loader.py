'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.vllm_config_loader's gpu.* gated-metric
coverage extension to _check_thresholds_cover_sweep. No hardware.
'''

import unittest
import warnings

from pydantic import ValidationError

from cvs.lib.inference.utils.vllm_config_loader import (
    GATED_GPU_METRICS,
    Run,
    SeqCombo,
    Sweep,
    VariantConfig,
)
from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS


def _combo(name, isl="128", osl="2048"):
    return SeqCombo(name=name, isl=isl, osl=osl)


def _full_gated_specs():
    """A spec for every gated client.* and gpu.* metric -- the minimum that
    satisfies coverage. Values are inert so the set passes without asserting
    anything; these tests pin the coverage gate, not the numbers."""
    out = {}
    for m in GATED_METRICS:
        kind = "max_ms" if m.endswith("_ms") else "max" if m == "failed" else "min"
        out[f"client.{m}"] = {"kind": kind, "value": 0 if kind == "min" else 1e12}
    for m in GATED_GPU_METRICS:
        kind = "max" if m in ("peak_gpu_memory_mb", "model_load_memory_mb", "model_load_s") else "min"
        out[f"gpu.{m}"] = {"kind": kind, "value": 0 if kind == "min" else 1e12}
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

    def test_missing_gpu_metric_raises_when_enforced(self):
        specs = _full_gated_specs()
        del specs["gpu.peak_gpu_memory_mb"]
        with self.assertRaises(ValidationError) as ctx:
            self._variant_with({self._CELL: specs}, enforce=True)
        self.assertIn("missing gated-metric specs", str(ctx.exception))
        self.assertIn("gpu.peak_gpu_memory_mb", str(ctx.exception))

    def test_missing_gpu_metric_warns_when_record_only(self):
        specs = _full_gated_specs()
        del specs["gpu.model_load_s"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._variant_with({self._CELL: specs}, enforce=False)
        self.assertTrue(any("missing gated-metric specs" in str(x.message) for x in caught))

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


if __name__ == "__main__":
    unittest.main()
