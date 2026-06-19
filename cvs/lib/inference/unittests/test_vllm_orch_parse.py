'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for VllmJob.parse_results (stock `results` artifact -> client.* +
derived metrics) and run_client's --goodput / --metric-percentiles flag
construction. No hardware: a fake orch returns committed fixture text.
'''

import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.utils.verdict import ThresholdViolation, evaluate_all
from cvs.lib.inference.vllm_orch import VllmJob

_HERE = Path(__file__).parent
_FIXTURES = _HERE / "fixtures"
_REPO = _HERE.parents[3]  # cvs/lib/inference/unittests -> repo root
_SHARED = _REPO / "cvs/tests/inference/vllm/_shared.py"
_THRESHOLD = (
    _REPO / "cvs/input/config_file/inference/vllm_single/w1_llama31_70b_fp8kv/llama31_70b_fp8_threshold.json"
)

# isl/tp used to build the job; must match the fixture's run for the derived
# math assertions to be meaningful (real artifact: isl=128, tp=8).
_ISL = 128
_TP = 8


class FakeOrch:
    """Minimal stand-in for ContainerOrchestrator: records commands, returns a canned dict."""

    def __init__(self, exec_return=None):
        self.exec_return = exec_return if exec_return is not None else {}
        self.commands = []

    def exec(self, cmd, **kwargs):
        self.commands.append(cmd)
        return self.exec_return


def _fake_variant():
    """A SimpleNamespace tree carrying exactly the attributes VllmJob.__init__ reads."""
    params = SimpleNamespace(
        tensor_parallelism=str(_TP),
        port_no="8888",
        random_range_ratio="0.8",
        random_prefix_len="0",
        burstiness="1.0",
        seed="0",
        request_rate="inf",
        tokenizer_mode="auto",
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="50,90,95,99",
        base_url="http://0.0.0.0",
        dataset_name="random",
        backend="vllm",
    )
    return SimpleNamespace(
        params=params,
        model=SimpleNamespace(id="amd/Llama-3.1-70B-Instruct-FP8-KV"),
        roles=SimpleNamespace(server=SimpleNamespace(serve_args={}, env={})),
        paths=SimpleNamespace(log_dir="/tmp/logs", models_dir="/tmp/models"),
    )


def _make_job(orch, goodput_slo=None):
    return VllmJob(
        orch=orch,
        variant=_fake_variant(),
        hf_token="tok",
        isl=_ISL,
        osl=2048,
        concurrency=256,
        num_prompts=12800,
        goodput_slo=goodput_slo,
    )


def _load_fixture(name):
    return (_FIXTURES / name).read_text()


class TestParseResults(unittest.TestCase):
    def setUp(self):
        self.real = json.loads(_load_fixture("vllm_results_sample.json"))
        self.widened = json.loads(_load_fixture("vllm_results_widened.json"))

    def _parse(self, fixture_name):
        orch = FakeOrch({"fakehost": _load_fixture(fixture_name)})
        job = _make_job(orch)
        return job.parse_results()["fakehost"]

    def test_all_stock_scalars_namespaced_and_numeric(self):
        m = self._parse("vllm_results_widened.json")
        # Every stock scalar appears 1:1 under client.* with its value preserved.
        for k, v in self.widened.items():
            ck = f"client.{k}"
            self.assertIn(ck, m, f"missing {ck}")
            self.assertEqual(m[ck], v)
        # Spot-check a few are numeric (not the old scrape's strings).
        for ck in ("client.total_token_throughput", "client.mean_ttft_ms", "client.p99_itl_ms"):
            self.assertIsInstance(m[ck], (int, float))

    def test_derived_metrics_exact(self):
        m = self._parse("vllm_results_widened.json")
        w = self.widened
        self.assertAlmostEqual(m["client.per_gpu_throughput"], w["total_token_throughput"] / _TP)
        self.assertAlmostEqual(m["client.normalized_ttft_ms_per_tok"], w["mean_ttft_ms"] / _ISL)
        self.assertAlmostEqual(m["client.decode_latency_ratio"], w["p99_itl_ms"] / w["p50_itl_ms"])
        self.assertAlmostEqual(m["client.decode_throughput_p50"], 1000.0 / w["median_tpot_ms"])
        self.assertAlmostEqual(m["client.success_rate"], w["completed"] / (w["completed"] + w["failed"]))

    def test_goodput_passthrough_null_and_value(self):
        # Real artifact: request_goodput is null (ran without --goodput).
        m_null = self._parse("vllm_results_sample.json")
        self.assertIsNone(m_null["client.goodput"])
        # Widened fixture: non-null goodput passed straight through.
        m_val = self._parse("vllm_results_widened.json")
        self.assertEqual(m_val["client.goodput"], self.widened["request_goodput"])

    def test_decode_latency_ratio_none_when_p50_absent(self):
        # The real artifact has no p50_itl_ms (it ran at metric_percentiles=99),
        # so the ratio must degrade to None, not raise.
        m = self._parse("vllm_results_sample.json")
        self.assertIsNone(m["client.decode_latency_ratio"])

    def test_missing_artifact_raises(self):
        orch = FakeOrch({"fakehost": ""})
        job = _make_job(orch)
        with self.assertRaises(RuntimeError):
            job.parse_results()

    def test_unparseable_artifact_raises(self):
        orch = FakeOrch({"fakehost": "not json {{{"})
        job = _make_job(orch)
        with self.assertRaises(RuntimeError):
            job.parse_results()


class TestRunClientFlags(unittest.TestCase):
    def _client_cmd(self, goodput_slo):
        orch = FakeOrch()
        job = _make_job(orch, goodput_slo=goodput_slo)
        job.run_client()
        # run_client issues exactly one exec: bash -c '<client_cmd>'.
        self.assertEqual(len(orch.commands), 1)
        return orch.commands[0]

    def test_metric_percentiles_flag_present(self):
        cmd = self._client_cmd(None)
        self.assertIn("--metric-percentiles", cmd)
        self.assertIn("50,90,95,99", cmd)

    def test_goodput_flag_omitted_when_none(self):
        cmd = self._client_cmd(None)
        self.assertNotIn("--goodput", cmd)

    def test_goodput_flag_built_from_slo_dict(self):
        slo = {"ttft_ms": 500.0, "tpot_ms": 50.0, "e2el_ms": 60000.0}
        cmd = self._client_cmd(slo)
        self.assertIn("--goodput", cmd)
        for tok in ("ttft:500.0", "tpot:50.0", "e2el:60000.0"):
            self.assertIn(tok, cmd)


class TestKeyConsistency(unittest.TestCase):
    """Mechanical guard (verification #5): every key the table reads and every
    threshold cell key must be a key parse_results actually emits. Catches a
    silent `-` column or a silent threshold skip WITHOUT a hardware run."""

    def _producer_keys(self):
        orch = FakeOrch({"fakehost": _load_fixture("vllm_results_widened.json")})
        return set(_make_job(orch).parse_results()["fakehost"].keys())

    def test_table_keys_are_produced(self):
        produced = self._producer_keys()
        shared_src = _SHARED.read_text()
        table_keys = set(re.findall(r'_cell\(m,\s*"(client\.[^"]+)"', shared_src))
        self.assertTrue(table_keys, "no client.* table keys found in _shared.py")
        missing = table_keys - produced
        self.assertEqual(missing, set(), f"table reads keys parse_results never emits: {missing}")

    def test_threshold_keys_are_produced(self):
        produced = self._producer_keys()
        thr = json.loads(_THRESHOLD.read_text())
        threshold_metric_keys = set()
        for cell, metrics in thr.items():
            if cell.startswith("_"):
                continue
            threshold_metric_keys.update(metrics.keys())
        self.assertTrue(threshold_metric_keys, "no threshold metric keys found")
        missing = threshold_metric_keys - produced
        self.assertEqual(missing, set(), f"threshold asserts keys parse_results never emits: {missing}")


class TestVerdictNoneGuard(unittest.TestCase):
    """Regression (review fix #1): parse_results now emits metrics that are
    legitimately None (derived ratio with no p50, goodput with no SLO run).
    If a threshold targets one, evaluate_all must raise a clean
    ThresholdViolation -- NOT a float(None) TypeError."""

    def test_none_actual_raises_threshold_violation_not_typeerror(self):
        actuals = {"client.goodput": None}
        thresholds = {"client.goodput": {"kind": "min", "value": 1.0}}
        with self.assertRaises(ThresholdViolation) as ctx:
            evaluate_all(actuals, thresholds)
        self.assertIn("value is None", str(ctx.exception))

    def test_none_actual_does_not_mask_other_real_violations(self):
        actuals = {"client.goodput": None, "client.total_token_throughput": 5.0}
        thresholds = {
            "client.goodput": {"kind": "min", "value": 1.0},
            "client.total_token_throughput": {"kind": "min", "value": 10.0},
        }
        with self.assertRaises(ThresholdViolation) as ctx:
            evaluate_all(actuals, thresholds)
        msg = str(ctx.exception)
        self.assertIn("value is None", msg)
        self.assertIn("client.total_token_throughput", msg)

    def test_non_none_actual_still_evaluated_normally(self):
        # A satisfied threshold must still pass (guard is None-specific).
        actuals = {"client.goodput": 5.0}
        thresholds = {"client.goodput": {"kind": "min", "value": 1.0}}
        evaluate_all(actuals, thresholds)  # no raise


class TestTableCellRendering(unittest.TestCase):
    """Regression (code-review F1): a metric that is present-but-None (goodput
    with no SLO run, a derived ratio with no p50) must render as "-" in the
    results table, not the literal "None" (m.get returns None when the key
    exists)."""

    def _cell(self):
        import importlib.util
        import sys as _sys
        from types import ModuleType

        # _shared.py imports tabulate at module scope; the _cell helper does not
        # need it. Stub it so the module loads in the bare unittest interpreter.
        _sys.modules.setdefault("tabulate", ModuleType("tabulate"))
        if not hasattr(_sys.modules["tabulate"], "tabulate"):
            _sys.modules["tabulate"].tabulate = lambda *a, **k: ""
        spec = importlib.util.spec_from_file_location("_shared_under_test", str(_SHARED))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._cell

    def test_present_but_none_renders_dash(self):
        cell = self._cell()
        self.assertEqual(cell({"client.goodput": None}, "client.goodput"), "-")

    def test_absent_key_renders_dash(self):
        cell = self._cell()
        self.assertEqual(cell({}, "client.goodput"), "-")

    def test_real_value_passes_through(self):
        cell = self._cell()
        self.assertEqual(cell({"client.goodput": 4.76}, "client.goodput"), 4.76)

    def test_zero_is_not_dashed(self):
        # 0.0 is a real measurement, must NOT become '-'.
        cell = self._cell()
        self.assertEqual(cell({"client.request_throughput": 0.0}, "client.request_throughput"), 0.0)


class TestVerdictMinRatioReferenceNone(unittest.TestCase):
    """Regression (code-review F2): min_ratio dereferences a SECOND (reference)
    metric; if that reference is a None-valued derived metric, evaluate_all must
    raise a clean ThresholdViolation, not float(None) TypeError."""

    def test_min_ratio_none_reference_raises_violation_not_typeerror(self):
        actuals = {"client.per_gpu_throughput": 1000.0, "client.decode_latency_ratio": None}
        thresholds = {
            "client.per_gpu_throughput": {
                "kind": "min_ratio",
                "reference": "client.decode_latency_ratio",
                "value": 0.5,
            }
        }
        with self.assertRaises(ThresholdViolation) as ctx:
            evaluate_all(actuals, thresholds)
        self.assertIn("is None", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TestDerivedMaxModelLen(unittest.TestCase):
    """MAX_MODEL_LEN is derived per cell from isl/osl/random_range_ratio/random_prefix_len,
    not read from config. Worst-case sequence = (isl+osl)*(1+r) + prefix; +pad for rounding."""

    def _env_cmd(self, job):
        orch = FakeOrch()
        job.orch = orch
        job.build_server_cmd()
        # build_server_cmd issues: printf the env script, then mkdir the out-dir.
        # The env script (with MAX_MODEL_LEN) is in the first exec.
        return orch.commands[0]

    def test_derived_value_for_default_cell(self):
        # isl=128, osl=2048, r=0.8, prefix=0 -> ceil(2176*1.8)=3917, +0 +8 = 3925.
        job = _make_job(FakeOrch())
        self.assertEqual(job._derive_max_model_len(), "3925")

    def test_low_ratio_shrinks_window(self):
        # Dropping the ratio must shrink the derived window automatically.
        job = _make_job(FakeOrch())
        job.random_range_ratio = "0.1"
        # ceil(2176*1.1)=2394, +0 +8 = 2402.
        self.assertEqual(job._derive_max_model_len(), "2402")

    def test_zero_ratio_is_fixed_length_plus_pad(self):
        job = _make_job(FakeOrch())
        job.random_range_ratio = "0.0"
        # (128+2048) + 0 + 8 = 2184.
        self.assertEqual(job._derive_max_model_len(), "2184")

    def test_prefix_len_added(self):
        job = _make_job(FakeOrch())
        job.random_range_ratio = "0.0"
        job.random_prefix_len = "64"
        # 2176 + 64 + 8 = 2248.
        self.assertEqual(job._derive_max_model_len(), "2248")

    def test_server_argv_carries_derived_value(self):
        # The derived max-model-len is passed as the --max-model-len flag on the
        # `vllm serve` argv (it is no longer exported into the env script).
        job = _make_job(FakeOrch())
        argv = job._server_argv()
        self.assertIn("--max-model-len", argv)
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "3925")


import importlib.util as _ilu_t

_VS_PATH = _REPO / "cvs/tests/inference/vllm/vllm_single.py"


def _load_vllm_single():
    """Import the suite module standalone to reach _METRICS and test_metric.

    The module's only collection-time work is a top-level importlib exec of the
    sibling _shared.py; everything else is function/constant defs, so importing it
    outside pytest is safe and hardware-free.
    """
    spec = _ilu_t.spec_from_file_location("_vllm_single_under_test", str(_VS_PATH))
    mod = _ilu_t.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeNode:
    def __init__(self):
        self.user_properties = []


class _FakeRequest:
    def __init__(self):
        self.node = _FakeNode()


class _FakeLifecycle:
    def __init__(self, failed=False):
        self.failed = failed


def _fake_variant_config(enforce=False, thresholds=None):
    """Stand-in carrying exactly what test_metric reads: model.id, gpu_arch,
    enforce_thresholds, thresholds, and the real cell_key builder."""
    params = SimpleNamespace(tensor_parallelism=str(_TP))
    vc = SimpleNamespace(
        model=SimpleNamespace(id="amd/Llama-3.1-70B-Instruct-FP8-KV"),
        gpu_arch="mi300x",
        params=params,
        enforce_thresholds=enforce,
        thresholds=thresholds or {},
    )
    vc.cell_key = lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP={params.tensor_parallelism},CONC={conc}"
    return vc


class TestMetricTests(unittest.TestCase):
    """test_metric is the per-metric pytest row. These exercise its three paths
    (cell-missing skip, record-only PASS, enforced violation) with fakes -- no
    hardware, no real pytest collection."""

    def setUp(self):
        self.vs = _load_vllm_single()
        # A realistic per-cell actuals dict, as test_vllm_inference would stash.
        orch = FakeOrch({"fakehost": _load_fixture("vllm_results_widened.json")})
        self.actuals = _make_job(orch).parse_results()["fakehost"]
        self.seq_combo = {"isl": "128", "osl": "2048", "name": "throughput"}
        self.conc = 64
        self.key = ("amd/Llama-3.1-70B-Instruct-FP8-KV", "mi300x", "128", "2048", "throughput", 64)

    def test_every_metric_key_is_produced(self):
        """Every _METRICS short name must resolve to a client.* key parse_results
        emits -- otherwise its row would silently show '-'."""
        produced = set(self.actuals.keys())
        missing = [short for short, _u in self.vs._METRICS if ("client." + short) not in produced]
        self.assertEqual(missing, [], f"_METRICS names with no producer: {missing}")

    def test_gated_metrics_are_all_produced_and_displayed(self):
        """Every GATED_METRICS name must be both produced (parse_results emits
        client.<name>) and displayed (in _METRICS) -- a gated metric with no
        producer would gate a '-', and one absent from _METRICS would assert a
        value that never appears in the report."""
        from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS
        produced = set(self.actuals.keys())
        displayed = {short for short, _u in self.vs._METRICS}
        no_producer = sorted(m for m in GATED_METRICS if ("client." + m) not in produced)
        no_row = sorted(m for m in GATED_METRICS if m not in displayed)
        self.assertEqual(no_producer, [], f"gated metrics with no producer: {no_producer}")
        self.assertEqual(no_row, [], f"gated metrics not in _METRICS: {no_row}")

    def test_skips_when_cell_absent(self):
        import pytest as _pt
        with self.assertRaises(_pt.skip.Exception):
            self.vs.test_metric(
                self.seq_combo, self.conc, "p99_e2el_ms",
                {}, _fake_variant_config(), _FakeLifecycle(), _FakeRequest(),
            )

    def test_skips_when_lifecycle_failed(self):
        import pytest as _pt
        with self.assertRaises(_pt.skip.Exception):
            self.vs.test_metric(
                self.seq_combo, self.conc, "p99_e2el_ms",
                {self.key: {"fakehost": self.actuals}},
                _fake_variant_config(), _FakeLifecycle(failed=True), _FakeRequest(),
            )

    def test_record_only_records_value_and_unit(self):
        req = _FakeRequest()
        self.vs.test_metric(
            self.seq_combo, self.conc, "p99_e2el_ms",
            {self.key: {"fakehost": self.actuals}},
            _fake_variant_config(enforce=False), _FakeLifecycle(), req,
        )
        props = dict(req.node.user_properties)
        self.assertEqual(props["metric_value"], self.actuals["client.p99_e2el_ms"])
        self.assertEqual(props["metric_unit"], "ms")

    def test_enforce_raises_on_violation(self):
        # mean_ttft_ms is ~hundreds; a max_ms of 1.0 must trip evaluate_all.
        cell = f"ISL=128,OSL=2048,TP={_TP},CONC=64"
        thr = {cell: {"client.mean_ttft_ms": {"kind": "max_ms", "value": 1.0}}}
        with self.assertRaises(ThresholdViolation):
            self.vs.test_metric(
                self.seq_combo, self.conc, "mean_ttft_ms",
                {self.key: {"fakehost": self.actuals}},
                _fake_variant_config(enforce=True, thresholds=thr), _FakeLifecycle(), _FakeRequest(),
            )

    def test_enforce_no_spec_is_record_only(self):
        # enforce=true but no threshold for this metric -> record-only, no raise.
        req = _FakeRequest()
        self.vs.test_metric(
            self.seq_combo, self.conc, "p95_tpot_ms",
            {self.key: {"fakehost": self.actuals}},
            _fake_variant_config(enforce=True, thresholds={}), _FakeLifecycle(), req,
        )
        self.assertEqual(dict(req.node.user_properties)["metric_unit"], "ms")


from cvs.lib.inference.utils.vllm_parsing import _safe_div, to_client_metrics


class TestToClientMetricsPure(unittest.TestCase):
    """Direct tests of the pure transform -- no FakeOrch, no VllmJob.

    parse_results already covers the wiring; these pin the vocabulary + math in
    isolation so distributed/disagg/InferenceMax reuse rests on a tested seam.
    """

    def setUp(self):
        self.raw = json.loads(_load_fixture("vllm_results_widened.json"))

    def test_namespaces_every_stock_scalar(self):
        m = to_client_metrics(self.raw, tp=_TP, isl=_ISL)
        for k, v in self.raw.items():
            self.assertEqual(m[f"client.{k}"], v)

    def test_goodput_alias(self):
        m = to_client_metrics(self.raw, tp=_TP, isl=_ISL)
        self.assertEqual(m["client.goodput"], self.raw["request_goodput"])

    def test_derived_metrics_exact(self):
        m = to_client_metrics(self.raw, tp=_TP, isl=_ISL)
        w = self.raw
        self.assertAlmostEqual(m["client.per_gpu_throughput"], w["total_token_throughput"] / _TP)
        self.assertAlmostEqual(m["client.normalized_ttft_ms_per_tok"], w["mean_ttft_ms"] / _ISL)
        self.assertAlmostEqual(m["client.decode_latency_ratio"], w["p99_itl_ms"] / w["p50_itl_ms"])
        self.assertAlmostEqual(m["client.decode_throughput_p50"], 1000.0 / w["median_tpot_ms"])
        self.assertAlmostEqual(m["client.success_rate"], w["completed"] / (w["completed"] + w["failed"]))

    def test_pure_no_mutation_of_input(self):
        snapshot = dict(self.raw)
        to_client_metrics(self.raw, tp=_TP, isl=_ISL)
        self.assertEqual(self.raw, snapshot)

    def test_missing_inputs_degrade_to_none_not_raise(self):
        m = to_client_metrics({}, tp=_TP, isl=_ISL)
        for d in ("per_gpu_throughput", "normalized_ttft_ms_per_tok",
                  "decode_latency_ratio", "decode_throughput_p50", "success_rate"):
            self.assertIsNone(m[f"client.{d}"])


class TestSafeDivPure(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(_safe_div(10, 2), 5.0)

    def test_zero_divisor_is_none(self):
        self.assertIsNone(_safe_div(1, 0))

    def test_none_operands_are_none(self):
        self.assertIsNone(_safe_div(None, 2))
        self.assertIsNone(_safe_div(2, None))

    def test_non_numeric_is_none(self):
        self.assertIsNone(_safe_div("x", 2))
