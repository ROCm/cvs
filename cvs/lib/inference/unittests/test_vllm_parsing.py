'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/inference/utils/vllm_parsing.py.

Impl-blind, spec-derived (Spec A1: per_gpu_throughput accounts for pipeline-
parallel size). Every case drives `to_client_metrics` / `_gpu_count` /
`_safe_div` directly with plain dict fixtures -- no orchestrator, no VllmJob,
no hardware. Written greenfield (RED) before the implementation adds the
required `pp` kwarg and the `_gpu_count` helper; the implementer makes them
green and cannot edit this file.
'''

import unittest

from cvs.lib.inference.utils import vllm_parsing


# ---------------------------------------------------------------------------
# Fixtures: a fresh, comprehensive raw benchmark artifact per call so no test
# mutates state another test reads (avoids the shared-mutable-fixture pitfall).
# Values mirror the shape of a real `vllm bench serve` `results` artifact.
# total_token_throughput defaults to a round 4800.0 so /8 and /16 are exact.
# ---------------------------------------------------------------------------
def _raw(**overrides):
    base = {
        "num_prompts": 3200,
        "completed": 1791,
        "failed": 1409,
        "duration": 564.1147056743503,
        "request_throughput": 3.174886210170704,
        "request_goodput": 3.174886210170704,
        "output_throughput": 4099.180497760143,
        "total_token_throughput": 4800.0,
        "max_output_tokens_per_s": 4590.0,
        "max_concurrency": 64,
        "max_concurrent_requests": 76,
        "total_input_tokens": 229974,
        "total_output_tokens": 2312408,
        "rtfx": 0.0,
        "mean_ttft_ms": 291.88843878398956,
        "median_ttft_ms": 73.2587007805705,
        "p90_ttft_ms": 85.64653992652893,
        "p95_ttft_ms": 91.81479038670659,
        "p99_ttft_ms": 6259.152442589402,
        "mean_tpot_ms": 15.032167084785439,
        "median_tpot_ms": 15.03020008988302,
        "p90_tpot_ms": 15.209271169796184,
        "p95_tpot_ms": 15.24944565870449,
        "p99_tpot_ms": 15.943741001209947,
        "mean_itl_ms": 15.019441688915942,
        "median_itl_ms": 14.628257602453232,
        "p50_itl_ms": 14.628257602453232,
        "p95_itl_ms": 17.392832040786708,
        "p99_itl_ms": 27.511265948414778,
        "mean_e2el_ms": 19668.871285726116,
        "median_e2el_ms": 19835.17629932612,
        "p90_e2el_ms": 30145.559770055115,
        "p95_e2el_ms": 31765.095902141184,
        "p99_e2el_ms": 34034.53387795014,
    }
    base.update(overrides)
    return base


_ISL = "128"  # str, matching production (VllmJob stores self.isl = str(isl))


def _metrics(raw=None, tp="8", isl=_ISL, pp="1"):
    if raw is None:
        raw = _raw()
    return vllm_parsing.to_client_metrics(raw, tp=tp, isl=isl, pp=pp)


# ===========================================================================
# _gpu_count -- pure helper (Spec AC2). int(tp)*int(pp) for valid numeric
# input (str or int), None for missing/None/non-numeric, never raises.
# Range/equivalence table + zero boundary + no-raise + commutativity invariant.
# ===========================================================================
class TestGpuCount(unittest.TestCase):
    def test_gpu_count_grid(self):
        cases = [
            # (tp, pp, expected)
            ("8", "2", 16),          # both numeric strings -> product
            (8, 2, 16),              # both ints
            (8, "2", 16),            # mixed int/str (no str-repetition trap)
            ("8", 2, 16),            # mixed str/int
            ("1", "8", 8),           # single-node style
            ("16", "1", 16),
            ("0", "8", 0),           # zero is a real product, not None
            ("8", "0", 0),
            (None, "8", None),       # missing/None -> None
            ("8", None, None),
            (None, None, None),
            ("auto", "8", None),     # non-numeric -> None (int('auto') raises)
            ("8", "auto", None),
            ("", "8", None),         # empty string -> None
            ("2.5", "8", None),      # int('2.5') raises ValueError -> None
        ]
        for tp, pp, expected in cases:
            with self.subTest(tp=tp, pp=pp):
                self.assertEqual(vllm_parsing._gpu_count(tp, pp), expected)

    def test_gpu_count_zero_is_int_not_none(self):
        # 0 (a degenerate but real count) must be distinct from None so that
        # _safe_div can then apply its zero-divisor guard downstream.
        result = vllm_parsing._gpu_count("0", "8")
        self.assertEqual(result, 0)
        self.assertIsNotNone(result)

    def test_gpu_count_never_raises_on_bad_input(self):
        # Degrade-to-None contract: must not raise out on any bad input.
        for tp, pp in [(None, None), ("auto", "auto"), ("", ""), (object(), 8), (8, [1])]:
            with self.subTest(tp=tp, pp=pp):
                try:
                    self.assertIsNone(vllm_parsing._gpu_count(tp, pp))
                except Exception as exc:  # noqa: BLE001 - the whole point is no raise
                    self.fail(f"_gpu_count({tp!r}, {pp!r}) raised {exc!r}")

    def test_gpu_count_commutative_invariant(self):
        # int(tp)*int(pp) == int(pp)*int(tp): the helper must be symmetric.
        for a, b in [("8", "2"), (4, 3), ("1", "16"), ("0", "8"), ("auto", "2")]:
            with self.subTest(a=a, b=b):
                self.assertEqual(
                    vllm_parsing._gpu_count(a, b),
                    vllm_parsing._gpu_count(b, a),
                )


# ===========================================================================
# _safe_div -- pure helper underpinning every derived metric's None-degrade.
# Spec: None/0 divisors -> None; None numerator -> None; zero numerator with
# a real divisor is a real 0.0 result (not None).
# ===========================================================================
class TestSafeDiv(unittest.TestCase):
    def test_safe_div_grid(self):
        cases = [
            (10, 2, 5.0),
            (9, 4, 2.25),
            (0, 5, 0.0),        # zero numerator -> real 0.0, NOT None
            (10, 0, None),      # zero divisor -> None
            (10, None, None),   # None divisor -> None
            (None, 5, None),    # None numerator -> None
            (None, None, None),
        ]
        for num, den, expected in cases:
            with self.subTest(num=num, den=den):
                result = vllm_parsing._safe_div(num, den)
                if expected is None:
                    self.assertIsNone(result)
                else:
                    self.assertIsNotNone(result)
                    self.assertAlmostEqual(result, expected)

    def test_safe_div_zero_numerator_is_real_zero(self):
        result = vllm_parsing._safe_div(0, 5)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0)


# ===========================================================================
# per_gpu_throughput -- the spec's focus (AC3/AC4/AC5). Pure; value grid over
# the (tp, pp) space + None-degradation + numeric invariants.
# ===========================================================================
class TestPerGpuThroughput(unittest.TestCase):
    KEY = "client.per_gpu_throughput"

    def test_per_gpu_throughput_over_tp_pp_grid(self):
        T = 4800.0
        cases = [
            # (tp, pp, expected)  -- expected None means degrade-to-None
            ("8", "1", T / 8),      # AC4: single-node, == pre-fix ttot/tp
            ("8", "2", T / 16),     # AC3/AC5: pp accounted -> ttot/(tp*pp)
            (8, "2", T / 16),       # mixed int tp
            ("8", 2, T / 16),       # mixed int pp
            (8, 2, T / 16),         # both int
            ("16", "1", T / 16),
            ("4", "4", T / 16),
            ("8", None, None),      # pp None -> None
            ("8", "auto", None),    # pp non-numeric -> None
            ("auto", "1", None),    # tp non-numeric -> None
            ("0", "8", None),       # zero gpu count -> _safe_div guards -> None
            ("8", "0", None),       # zero gpu count -> None
        ]
        for tp, pp, expected in cases:
            with self.subTest(tp=tp, pp=pp):
                m = vllm_parsing.to_client_metrics(
                    _raw(total_token_throughput=T), tp=tp, isl=_ISL, pp=pp
                )
                if expected is None:
                    self.assertIsNone(m[self.KEY])
                else:
                    self.assertIsNotNone(m[self.KEY])
                    self.assertAlmostEqual(m[self.KEY], expected)

    def test_pp1_equals_ttot_over_tp(self):
        # AC4: single-node (pp="1") is byte-identical to the pre-fix formula.
        raw = _raw()
        m = _metrics(raw, tp="8", pp="1")
        self.assertAlmostEqual(m[self.KEY], raw["total_token_throughput"] / 8)

    def test_pp2_is_exactly_half_of_pp1(self):
        # AC5: a pp="2" cell yields exactly half of the pre-fix (ttot/tp) value.
        v1 = _metrics(_raw(), tp="8", pp="1")[self.KEY]
        v2 = _metrics(_raw(), tp="8", pp="2")[self.KEY]
        self.assertAlmostEqual(v2, v1 / 2)

    def test_per_gpu_throughput_monotonic_decreasing_in_pp(self):
        # Invariant: with tp and ttot fixed, more pipeline stages -> strictly
        # lower per-GPU throughput.
        vals = [
            _metrics(_raw(), tp="8", pp=str(pp))[self.KEY] for pp in (1, 2, 4, 8)
        ]
        for higher, lower in zip(vals, vals[1:]):
            self.assertGreater(higher, lower)

    def test_none_when_total_token_throughput_missing_or_none(self):
        # AC3: unavailable ttot -> None regardless of tp/pp (unchanged _safe_div).
        raw_missing = _raw()
        del raw_missing["total_token_throughput"]
        self.assertIsNone(_metrics(raw_missing, tp="8", pp="2")[self.KEY])
        raw_none = _raw(total_token_throughput=None)
        self.assertIsNone(_metrics(raw_none, tp="8", pp="2")[self.KEY])

    def test_pp_defaults_to_one(self):
        # Callers with no pipeline-parallel concept (e.g. InferenceX ATOM) omit
        # `pp` entirely; it must silently behave as pp="1", not raise.
        omitted = vllm_parsing.to_client_metrics(_raw(), tp="8", isl=_ISL)
        explicit = _metrics(_raw(), tp="8", pp="1")
        self.assertEqual(omitted[self.KEY], explicit[self.KEY])


# ===========================================================================
# The other four _safe_div-guarded derived metrics + goodput alias.
# Restores TestToClientMetricsPure coverage: value + None-degradation per
# metric, table-driven.
# ===========================================================================
class TestDerivedMetrics(unittest.TestCase):
    def test_derived_metric_values(self):
        raw = _raw()
        m = _metrics(raw, tp="8", isl=_ISL, pp="2")
        cases = [
            ("client.normalized_ttft_ms_per_tok", raw["mean_ttft_ms"] / 128),
            ("client.decode_latency_ratio", raw["p99_itl_ms"] / raw["p50_itl_ms"]),
            ("client.decode_throughput_p50", 1000.0 / raw["median_tpot_ms"]),
            ("client.success_rate", raw["completed"] / (raw["completed"] + raw["failed"])),
        ]
        for key, expected in cases:
            with self.subTest(metric=key):
                self.assertIsNotNone(m[key])
                self.assertAlmostEqual(m[key], expected)

    def test_derived_metric_none_degradation(self):
        # Drop the raw scalar each derived metric depends on -> it degrades to
        # None (does not raise, does not compute a wrong number).
        cases = [
            ("client.normalized_ttft_ms_per_tok", "mean_ttft_ms"),
            ("client.decode_latency_ratio", "p50_itl_ms"),
            ("client.decode_throughput_p50", "median_tpot_ms"),
        ]
        for key, drop in cases:
            with self.subTest(metric=key, dropped=drop):
                raw = _raw()
                del raw[drop]
                m = _metrics(raw, tp="8", isl=_ISL, pp="2")
                self.assertIsNone(m[key])

    def test_success_rate_none_when_denominator_zero(self):
        # completed=0, failed=0 -> _safe_div(0, 0) -> None (not a crash, not 0).
        m = _metrics(_raw(completed=0, failed=0), tp="8", pp="1")
        self.assertIsNone(m["client.success_rate"])

    def test_goodput_alias_value_passthrough(self):
        m = _metrics(_raw(request_goodput=42.5), tp="8", pp="1")
        self.assertEqual(m["client.goodput"], 42.5)

    def test_goodput_alias_none_passthrough(self):
        # Ran without --goodput -> request_goodput is null -> client.goodput None.
        m = _metrics(_raw(request_goodput=None), tp="8", pp="1")
        self.assertIsNone(m["client.goodput"])


# ===========================================================================
# 1:1 stock-scalar namespacing (client.<key> == raw[key]) and AC7 isolation.
# ===========================================================================
class TestStockScalarNamespacing(unittest.TestCase):
    def test_returns_a_dict(self):
        # Contract: to_client_metrics always returns a dict, never None/other.
        # (A type-level assertion so a no-op stub is caught as a genuine
        # assertion FAILURE rather than a downstream TypeError/ERROR.)
        m = _metrics(_raw(), tp="8", pp="1")
        self.assertIsInstance(m, dict)

    def test_every_stock_scalar_namespaced_one_to_one(self):
        raw = _raw()
        m = _metrics(raw, tp="8", pp="1")
        for key, value in raw.items():
            with self.subTest(key=key):
                nk = f"client.{key}"
                self.assertIn(nk, m)
                self.assertEqual(m[nk], value)

    def test_zero_valued_scalar_preserved_not_dropped(self):
        # 0.0 is a real measurement; it must survive namespacing as 0.0, not be
        # coerced to None or dropped.
        m = _metrics(_raw(rtfx=0.0, request_throughput=0.0), tp="8", pp="1")
        self.assertEqual(m["client.request_throughput"], 0.0)
        self.assertIsNotNone(m["client.request_throughput"])
        self.assertEqual(m["client.rtfx"], 0.0)

    def test_numeric_scalars_stay_numeric(self):
        m = _metrics(_raw(), tp="8", pp="1")
        for nk in ("client.total_token_throughput", "client.mean_ttft_ms", "client.p99_itl_ms"):
            with self.subTest(key=nk):
                self.assertIsInstance(m[nk], (int, float))

    def test_only_per_gpu_throughput_changes_with_pp(self):
        # AC7: varying pp changes per_gpu_throughput and NOTHING else.
        m1 = _metrics(_raw(), tp="8", isl=_ISL, pp="1")
        m2 = _metrics(_raw(), tp="8", isl=_ISL, pp="2")
        self.assertEqual(set(m1), set(m2))
        for key in m1:
            if key == "client.per_gpu_throughput":
                continue
            with self.subTest(key=key):
                self.assertEqual(m1[key], m2[key])
        self.assertNotEqual(
            m1["client.per_gpu_throughput"], m2["client.per_gpu_throughput"]
        )


# ===========================================================================
# client.failed fallback derivation (vllm_parsing.py:70-78).
# When failed is missing/None but completed and num_prompts are present:
#   failed = max(0, int(num_prompts) - int(completed)), guarded -> None on bad input.
# ===========================================================================
class TestFailedFallbackDerivation(unittest.TestCase):
    def test_failed_absent_and_completed_le_num_prompts_derives(self):
        raw = _raw()
        del raw["failed"]
        raw["num_prompts"] = 3200
        raw["completed"] = 1791
        m = _metrics(raw, tp="8", pp="1")
        self.assertEqual(m["client.failed"], 3200 - 1791)

    def test_failed_absent_and_completed_gt_num_prompts_clamped_to_zero(self):
        # max(0, ...) must clamp -- never emit a negative failed count.
        raw = _raw()
        del raw["failed"]
        raw["num_prompts"] = 100
        raw["completed"] = 150
        m = _metrics(raw, tp="8", pp="1")
        self.assertEqual(m["client.failed"], 0)

    def test_failed_absent_and_none_valued_still_derives(self):
        # failed present-but-None is treated as missing -> fallback fires.
        raw = _raw(failed=None)
        raw["num_prompts"] = 3200
        raw["completed"] = 1791
        m = _metrics(raw, tp="8", pp="1")
        self.assertEqual(m["client.failed"], 3200 - 1791)

    def test_failed_absent_nonnumeric_inputs_not_injected(self):
        # Guarded try/except -> failed stays None and the key is NOT injected.
        cases = [
            {"num_prompts": "auto", "completed": 1791},
            {"num_prompts": 3200, "completed": "auto"},
            {"num_prompts": None, "completed": 1791},
        ]
        for overrides in cases:
            with self.subTest(**overrides):
                raw = _raw()
                del raw["failed"]
                raw.update(overrides)
                m = _metrics(raw, tp="8", pp="1")
                self.assertNotIn("client.failed", m)

    def test_failed_present_fallback_not_invoked(self):
        # An explicit failed value wins; the fallback must not overwrite it even
        # when num_prompts-completed would compute a different number.
        raw = _raw(failed=5)
        raw["num_prompts"] = 3200
        raw["completed"] = 1791  # would derive 1409, must be ignored
        m = _metrics(raw, tp="8", pp="1")
        self.assertEqual(m["client.failed"], 5)


# ===========================================================================
# Module constants -- pin the spec's non-functional "no change" requirements
# and the record-only (ungated) status of per_gpu_throughput.
# ===========================================================================
class TestModuleConstants(unittest.TestCase):
    def test_per_gpu_throughput_is_record_only_not_gated(self):
        # Spec: per_gpu_throughput is NOT in GATED_METRICS -> the fix cannot
        # flip any pass/fail gate.
        self.assertNotIn("per_gpu_throughput", vllm_parsing.GATED_METRICS)

    def test_per_gpu_throughput_registered_in_client_metrics(self):
        units = dict(vllm_parsing.CLIENT_METRICS)
        self.assertIn("per_gpu_throughput", units)
        self.assertEqual(units["per_gpu_throughput"], "tok/s")

    def test_client_metrics_short_names_are_unique(self):
        # CLIENT_METRIC_UNITS is `dict(CLIENT_METRICS)`, which silently collapses
        # a duplicate short name to its last entry -- assert there are none.
        short_names = [short for short, _unit in vllm_parsing.CLIENT_METRICS]
        self.assertEqual(len(short_names), len(set(short_names)))

    def test_client_metric_units_matches_client_metrics(self):
        self.assertEqual(
            vllm_parsing.CLIENT_METRIC_UNITS["total_token_throughput"], "tok/s"
        )
        self.assertEqual(vllm_parsing.CLIENT_METRIC_UNITS["mean_ttft_ms"], "ms")

    def test_gated_metrics_subset_of_client_metrics(self):
        client_short = {short for short, _unit in vllm_parsing.CLIENT_METRICS}
        self.assertEqual(vllm_parsing.GATED_METRICS - client_short, set())


if __name__ == "__main__":
    unittest.main()
