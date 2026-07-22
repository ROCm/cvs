'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.utils.gpu.

Black-box tests authored from the behavioral spec only (impl-blind). The module
contains pure parsers for `amd-smi metric --json` output: no I/O, no hardware,
pure dict transformations.

Contract under test (from spec):
  parse_usage(gpu_entry)     -> {"gpu.gfx_activity", "gpu.umc_activity",
                                 "gpu.mm_activity"}; int|None each. Degrades to
                                 None for any missing key or "N/A" value; never raises.
  parse_mem_usage(gpu_entry) -> {"gpu.total_vram", "gpu.used_vram",
                                 "gpu.free_vram"}; int|None each. Degrades; never raises.
  parse_energy(gpu_entry)    -> {"gpu.energy_j"}; float|None. Degrades; never raises.
  parse_gpu_metrics(raw)     -> single dict with all 7 gpu.* keys.
                                 activity fields averaged across GPUs;
                                 vram + energy_j summed across GPUs.
                                 [] -> all 7 keys present, all None. Never raises.
  GPU_METRICS / GPU_METRIC_UNITS: every metric short_name has a matching unit;
                                 parse_gpu_metrics([full]) emits "gpu.<k>" for every k.

Framework: unittest.TestCase + self.subTest + unittest.mock (no pytest).
'''

import pathlib
import unittest
from unittest.mock import MagicMock, patch

from cvs.lib.utils.gpu import (
    GPU_METRICS,
    GPU_METRIC_UNITS,
    _RAW_GPU_FIELDS,
    _RAW_GPU_FIELD_UNITS,
    _RECORD_SEP,
    GpuPollerHandle,
    _mean,
    agg_readings,
    capture_gpu_metrics,
    start_gpu_poller,
    stop_and_collect_gpu_poller,
    parse_usage,
    parse_mem_usage,
    parse_energy,
    parse_gpu_metrics,
)

# ---------------------------------------------------------------------------
# Shared fixtures — amd-smi JSON schema (one GPU entry)
# ---------------------------------------------------------------------------

# The seven spec'd metrics, each as the bare "gpu.<short_name>" key produced by
# the parsers / aggregator.
ACTIVITY_KEYS = ["gpu.gfx_activity", "gpu.umc_activity", "gpu.mm_activity"]
VRAM_KEYS = ["gpu.total_vram", "gpu.used_vram", "gpu.free_vram"]
ENERGY_KEY = "gpu.energy_j"
ALL_KEYS = ACTIVITY_KEYS + VRAM_KEYS + [ENERGY_KEY]


def _full_gpu_entry(gfx=30, umc=20, mm=10, total=196608, used=4096, free=192512, energy=12345.5):
    """A complete amd-smi entry for one GPU with all seven fields present."""
    return {
        "usage": {
            "gfx_activity": {"value": gfx},
            "umc_activity": {"value": umc},
            "mm_activity": {"value": mm},
        },
        "mem_usage": {
            "total_vram": {"value": total},
            "used_vram": {"value": used},
            "free_vram": {"value": free},
        },
        "energy": {
            "total_energy_consumption": {"value": energy},
        },
    }


# ---------------------------------------------------------------------------
# parse_usage — pure function (dict -> dict)
# ---------------------------------------------------------------------------


class TestParseUsage(unittest.TestCase):
    """parse_usage extracts ["usage"]; degrades to None; never raises."""

    def test_full_entry_extracts_all_three(self):
        out = parse_usage(_full_gpu_entry(gfx=55, umc=44, mm=33))
        self.assertEqual(
            out,
            {
                "gpu.gfx_activity": 55,
                "gpu.umc_activity": 44,
                "gpu.mm_activity": 33,
            },
        )

    def test_returns_exactly_the_three_activity_keys(self):
        out = parse_usage(_full_gpu_entry())
        self.assertEqual(set(out.keys()), set(ACTIVITY_KEYS))

    def test_value_types_are_int(self):
        out = parse_usage(_full_gpu_entry(gfx=1, umc=2, mm=3))
        for k in ACTIVITY_KEYS:
            with self.subTest(key=k):
                self.assertIsInstance(out[k], int)

    def test_degradation_table(self):
        """Each degraded-input shape maps every activity field to None.

        Boundary classes: empty dict, missing "usage", "N/A" string value.
        """
        na = "N/A"
        cases = [
            # (description, gpu_entry)
            ("empty entry", {}),
            ("missing usage key", {"mem_usage": {}}),
            (
                "all fields N/A",
                {
                    "usage": {
                        "gfx_activity": {"value": na},
                        "umc_activity": {"value": na},
                        "mm_activity": {"value": na},
                    }
                },
            ),
        ]
        expected = {k: None for k in ACTIVITY_KEYS}
        for desc, entry in cases:
            with self.subTest(case=desc):
                self.assertEqual(parse_usage(entry), expected)

    def test_partial_entry_degrades_only_missing_field(self):
        """One field missing/N/A -> None for that field; others extracted."""
        entry = {
            "usage": {
                "gfx_activity": {"value": 77},
                "umc_activity": {"value": "N/A"},
                # mm_activity entirely absent
            }
        }
        out = parse_usage(entry)
        self.assertEqual(out["gpu.gfx_activity"], 77)
        self.assertIsNone(out["gpu.umc_activity"])
        self.assertIsNone(out["gpu.mm_activity"])

    def test_zero_values_not_coerced_to_none(self):
        """0 is a valid reading (fully idle GPU); must not degrade to None."""
        out = parse_usage(_full_gpu_entry(gfx=0, umc=0, mm=0))
        self.assertEqual(out["gpu.gfx_activity"], 0)
        self.assertEqual(out["gpu.umc_activity"], 0)
        self.assertEqual(out["gpu.mm_activity"], 0)

    def test_never_raises_on_malformed_shapes(self):
        """Contract: degrades, never raises. Always returns all three keys as None."""
        malformed = [
            {},
            {"usage": {}},
            {"usage": {"gfx_activity": {}}},
        ]
        for entry in malformed:
            with self.subTest(entry=entry):
                out = parse_usage(entry)
                self.assertEqual(set(out.keys()), set(ACTIVITY_KEYS))
                for k in ACTIVITY_KEYS:
                    self.assertIsNone(out[k])


# ---------------------------------------------------------------------------
# parse_mem_usage — pure function (dict -> dict)
# ---------------------------------------------------------------------------


class TestParseMemUsage(unittest.TestCase):
    """parse_mem_usage extracts ["mem_usage"]; degrades; never raises."""

    def test_full_entry_extracts_all_three(self):
        out = parse_mem_usage(_full_gpu_entry(total=196608, used=4096, free=192512))
        self.assertEqual(
            out,
            {
                "gpu.total_vram": 196608,
                "gpu.used_vram": 4096,
                "gpu.free_vram": 192512,
            },
        )

    def test_returns_exactly_the_three_vram_keys(self):
        out = parse_mem_usage(_full_gpu_entry())
        self.assertEqual(set(out.keys()), set(VRAM_KEYS))

    def test_value_types_are_int(self):
        out = parse_mem_usage(_full_gpu_entry(total=10, used=3, free=7))
        for k in VRAM_KEYS:
            with self.subTest(key=k):
                self.assertIsInstance(out[k], int)

    def test_degradation_table(self):
        na = "N/A"
        cases = [
            ("empty entry", {}),
            ("missing mem_usage", {"usage": {}}),
            (
                "all N/A",
                {
                    "mem_usage": {
                        "total_vram": {"value": na},
                        "used_vram": {"value": na},
                        "free_vram": {"value": na},
                    }
                },
            ),
        ]
        expected = {k: None for k in VRAM_KEYS}
        for desc, entry in cases:
            with self.subTest(case=desc):
                self.assertEqual(parse_mem_usage(entry), expected)

    def test_partial_entry_degrades_only_missing_field(self):
        entry = {
            "mem_usage": {
                "total_vram": {"value": 1000},
                "used_vram": {"value": "N/A"},
                # free_vram absent
            }
        }
        out = parse_mem_usage(entry)
        self.assertEqual(out["gpu.total_vram"], 1000)
        self.assertIsNone(out["gpu.used_vram"])
        self.assertIsNone(out["gpu.free_vram"])

    def test_zero_values_not_coerced_to_none(self):
        """0 is a valid reading (idle GPU); must not degrade to None."""
        out = parse_mem_usage(_full_gpu_entry(total=0, used=0, free=0))
        self.assertEqual(out["gpu.total_vram"], 0)
        self.assertEqual(out["gpu.used_vram"], 0)
        self.assertEqual(out["gpu.free_vram"], 0)

    def test_never_raises_on_malformed_shapes(self):
        malformed = [
            {},
            {"mem_usage": {}},
            {"mem_usage": {"used_vram": {}}},
        ]
        for entry in malformed:
            with self.subTest(entry=entry):
                out = parse_mem_usage(entry)
                self.assertEqual(set(out.keys()), set(VRAM_KEYS))
                for k in VRAM_KEYS:
                    self.assertIsNone(out[k])


# ---------------------------------------------------------------------------
# parse_energy — pure function (dict -> dict)
# ---------------------------------------------------------------------------


class TestParseEnergy(unittest.TestCase):
    """parse_energy extracts total_energy_consumption; degrades; never raises."""

    def test_full_entry_extracts_energy(self):
        out = parse_energy(_full_gpu_entry(energy=99999.25))
        self.assertEqual(out, {"gpu.energy_j": 99999.25})

    def test_returns_exactly_the_energy_key(self):
        out = parse_energy(_full_gpu_entry())
        self.assertEqual(set(out.keys()), {ENERGY_KEY})

    def test_value_type_is_float(self):
        out = parse_energy(_full_gpu_entry(energy=1.5))
        self.assertIsInstance(out[ENERGY_KEY], float)

    def test_degradation_table(self):
        na = "N/A"
        cases = [
            ("empty entry", {}),
            ("missing energy", {"usage": {}}),
            ("missing total_energy_consumption", {"energy": {}}),
            (
                "N/A value",
                {"energy": {"total_energy_consumption": {"value": na}}},
            ),
        ]
        for desc, entry in cases:
            with self.subTest(case=desc):
                self.assertEqual(parse_energy(entry), {ENERGY_KEY: None})

    def test_never_raises_on_malformed_shapes(self):
        malformed = [
            {},
            {"energy": {}},
            {"energy": {"total_energy_consumption": {}}},
        ]
        for entry in malformed:
            with self.subTest(entry=entry):
                out = parse_energy(entry)
                self.assertEqual(set(out.keys()), {ENERGY_KEY})
                self.assertIsNone(out[ENERGY_KEY])

    def test_zero_energy_not_coerced_to_none(self):
        """0.0 is a valid reading (GPU powered but idle); must not degrade to None."""
        out = parse_energy(_full_gpu_entry(energy=0.0))
        self.assertEqual(out[ENERGY_KEY], 0.0)
        self.assertIsInstance(out[ENERGY_KEY], float)

    def test_int_energy_coerced_to_float(self):
        """parse_energy must return float even when the raw value is a Python int."""
        out = parse_energy(_full_gpu_entry(energy=100))
        self.assertIsInstance(out[ENERGY_KEY], float)


# ---------------------------------------------------------------------------
# parse_gpu_metrics — pure aggregator (list -> dict)
# ---------------------------------------------------------------------------


class TestParseGpuMetrics(unittest.TestCase):
    """Aggregates per-GPU entries: activity averaged, vram + energy summed."""

    # --- key-presence contract ---

    def test_all_seven_keys_present_for_full_entry(self):
        out = parse_gpu_metrics([_full_gpu_entry()])
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out.keys()), set(ALL_KEYS))

    def test_empty_list_yields_all_keys_none(self):
        """[] -> all 7 keys present, every value None. Never raises."""
        out = parse_gpu_metrics([])
        self.assertEqual(set(out.keys()), set(ALL_KEYS))
        for k in ALL_KEYS:
            with self.subTest(key=k):
                self.assertIsNone(out[k])

    # --- single-GPU identity invariant ---

    def test_single_gpu_equals_that_gpus_values(self):
        """Single GPU: averaged/summed result equals that GPU's values exactly."""
        entry = _full_gpu_entry(gfx=30, umc=20, mm=10, total=196608, used=4096, free=192512, energy=500.0)
        out = parse_gpu_metrics([entry])
        self.assertEqual(out["gpu.gfx_activity"], 30)
        self.assertEqual(out["gpu.umc_activity"], 20)
        self.assertEqual(out["gpu.mm_activity"], 10)
        self.assertEqual(out["gpu.total_vram"], 196608)
        self.assertEqual(out["gpu.used_vram"], 4096)
        self.assertEqual(out["gpu.free_vram"], 192512)
        self.assertEqual(out["gpu.energy_j"], 500.0)

    # --- aggregation semantics: average vs sum ---

    def test_activity_fields_averaged_across_gpus(self):
        """gfx/umc/mm averaged. Odd-sum pair verifies true division, not floor."""
        g0 = _full_gpu_entry(gfx=10, umc=40, mm=60)
        g1 = _full_gpu_entry(gfx=21, umc=80, mm=20)
        out = parse_gpu_metrics([g0, g1])
        self.assertEqual(out["gpu.gfx_activity"], 15.5)  # (10+21)/2 — not 15
        self.assertEqual(out["gpu.umc_activity"], 60)  # (40+80)/2
        self.assertEqual(out["gpu.mm_activity"], 40)  # (60+20)/2

    def test_vram_and_energy_summed_across_gpus(self):
        """total/used/free_vram and energy_j summed across GPUs."""
        g0 = _full_gpu_entry(total=100, used=30, free=70, energy=1.5)
        g1 = _full_gpu_entry(total=200, used=50, free=150, energy=2.5)
        out = parse_gpu_metrics([g0, g1])
        self.assertEqual(out["gpu.total_vram"], 300)
        self.assertEqual(out["gpu.used_vram"], 80)
        self.assertEqual(out["gpu.free_vram"], 220)
        self.assertEqual(out["gpu.energy_j"], 4.0)

    def test_activity_aggregation_is_average_not_sum(self):
        """Guards against an impl that sums activity instead of averaging:
        two equal nonzero GPUs must yield the per-GPU value, not double it."""
        g = _full_gpu_entry(gfx=50, umc=50, mm=50)
        out = parse_gpu_metrics([g, _full_gpu_entry(gfx=50, umc=50, mm=50)])
        self.assertEqual(out["gpu.gfx_activity"], 50)
        self.assertNotEqual(out["gpu.gfx_activity"], 100)

    def test_vram_aggregation_is_sum_not_average(self):
        """Guards against an impl that averages vram/energy instead of summing:
        two equal GPUs must total double, not stay equal."""
        g0 = _full_gpu_entry(total=100, used=40, free=60, energy=10.0)
        g1 = _full_gpu_entry(total=100, used=40, free=60, energy=10.0)
        out = parse_gpu_metrics([g0, g1])
        self.assertEqual(out["gpu.total_vram"], 200)
        self.assertEqual(out["gpu.energy_j"], 20.0)
        self.assertNotEqual(out["gpu.total_vram"], 100)

    # --- partial-entry aggregation ---

    def test_partial_entry_field_excluded_others_aggregated(self):
        """A field missing on one GPU -> aggregate the remaining GPUs for it;
        other fields still aggregate across all GPUs that have them."""
        g0 = _full_gpu_entry(gfx=20, total=100, used=40, free=60, energy=5.0)
        # g1 has no usage block at all -> gfx None for g1
        g1 = {
            "mem_usage": {
                "total_vram": {"value": 200},
                "used_vram": {"value": 60},
                "free_vram": {"value": 140},
            },
            "energy": {"total_energy_consumption": {"value": 7.0}},
        }
        out = parse_gpu_metrics([g0, g1])
        # activity only present on g0 -> aggregate is just g0's values
        self.assertEqual(out["gpu.gfx_activity"], 20)
        self.assertEqual(out["gpu.umc_activity"], 20)  # g0 fixture default
        self.assertEqual(out["gpu.mm_activity"], 10)  # g0 fixture default
        # vram present on both -> summed
        self.assertEqual(out["gpu.total_vram"], 300)
        self.assertEqual(out["gpu.used_vram"], 100)
        self.assertEqual(out["gpu.free_vram"], 200)
        # energy present on both -> summed
        self.assertEqual(out["gpu.energy_j"], 12.0)

    def test_field_absent_on_all_gpus_yields_none(self):
        """If no GPU supplies a field, the aggregate for that field is None,
        while present fields still aggregate."""
        no_energy = {
            "usage": {
                "gfx_activity": {"value": 10},
                "umc_activity": {"value": 10},
                "mm_activity": {"value": 10},
            },
            "mem_usage": {
                "total_vram": {"value": 100},
                "used_vram": {"value": 50},
                "free_vram": {"value": 50},
            },
        }
        out = parse_gpu_metrics([no_energy, dict(no_energy)])
        self.assertIsNone(out["gpu.energy_j"])
        self.assertEqual(out["gpu.gfx_activity"], 10)
        self.assertEqual(out["gpu.total_vram"], 200)

    def test_single_gpu_aggregated_field_types(self):
        """Activity and vram fields from a single full entry must be int (or float for energy)."""
        out = parse_gpu_metrics([_full_gpu_entry(gfx=10, umc=20, mm=30, total=1000, used=200, free=800, energy=5.0)])
        for k in ACTIVITY_KEYS:
            with self.subTest(key=k):
                self.assertIsInstance(out[k], (int, float))
        for k in VRAM_KEYS:
            with self.subTest(key=k):
                self.assertIsInstance(out[k], int)
        self.assertIsInstance(out[ENERGY_KEY], float)

    def test_partial_vram_none_excluded_from_sum(self):
        """GPU with no mem_usage block: its vram fields are None and excluded;
        only the GPU that has vram contributes to the sum."""
        g0 = _full_gpu_entry(total=100, used=40, free=60, energy=2.0)
        g1 = {
            "usage": {"gfx_activity": {"value": 10}, "umc_activity": {"value": 10}, "mm_activity": {"value": 10}},
            "energy": {"total_energy_consumption": {"value": 3.0}},
        }
        out = parse_gpu_metrics([g0, g1])
        self.assertEqual(out["gpu.total_vram"], 100)
        self.assertEqual(out["gpu.used_vram"], 40)
        self.assertEqual(out["gpu.free_vram"], 60)
        self.assertEqual(out["gpu.energy_j"], 5.0)

    def test_partial_energy_none_excluded_from_sum(self):
        """GPU with no energy block: its energy is None and excluded;
        only the GPU that has energy contributes to the sum."""
        g0 = _full_gpu_entry(energy=500.0)
        g1 = {
            "usage": {"gfx_activity": {"value": 5}, "umc_activity": {"value": 5}, "mm_activity": {"value": 5}},
            "mem_usage": {"total_vram": {"value": 50}, "used_vram": {"value": 10}, "free_vram": {"value": 40}},
        }
        out = parse_gpu_metrics([g0, g1])
        self.assertEqual(out["gpu.energy_j"], 500.0)

    def test_zero_vram_not_excluded_from_aggregation(self):
        """total_vram=0 is valid; a falsy-zero aggregation bug (if val: acc += val)
        would skip 0 and return None instead of 0. Single-GPU with all-zero VRAM."""
        out = parse_gpu_metrics([_full_gpu_entry(total=0, used=0, free=0)])
        self.assertEqual(out["gpu.total_vram"], 0)
        self.assertEqual(out["gpu.used_vram"], 0)
        self.assertEqual(out["gpu.free_vram"], 0)

    def test_zero_energy_not_excluded_from_aggregation(self):
        """energy=0.0 is valid; a falsy-zero aggregation bug (if energy: skip) would
        incorrectly exclude it. Two GPUs each with energy=0.0 must sum to 0.0."""
        g0 = _full_gpu_entry(energy=0.0)
        g1 = _full_gpu_entry(energy=0.0)
        out = parse_gpu_metrics([g0, g1])
        self.assertEqual(out["gpu.energy_j"], 0.0)
        self.assertIsInstance(out["gpu.energy_j"], float)

    def test_zero_activity_not_excluded_from_average(self):
        """gfx_activity=0 is valid (GPU idle). A falsy-zero bug in the aggregator
        would exclude it from the average, giving wrong denominator+numerator."""
        g0 = _full_gpu_entry(gfx=0)
        g1 = _full_gpu_entry(gfx=20)
        out = parse_gpu_metrics([g0, g1])
        self.assertEqual(out["gpu.gfx_activity"], 10.0)  # (0+20)/2, not 20/1=20

    def test_partial_none_activity_averaging_three_gpus(self):
        """With N=3 where one has no activity, mean of non-None values is:
        (30 + 60) / 2 = 45.0 — not sum=90, not divide-by-3=30."""
        g_no_usage = {
            "mem_usage": {"total_vram": {"value": 50}, "used_vram": {"value": 20}, "free_vram": {"value": 30}},
            "energy": {"total_energy_consumption": {"value": 1.0}},
        }
        g0 = _full_gpu_entry(gfx=30)
        g1 = _full_gpu_entry(gfx=60)
        out = parse_gpu_metrics([g0, g_no_usage, g1])
        self.assertEqual(out["gpu.gfx_activity"], 45.0)

    def test_activity_averaging_three_full_gpus(self):
        """N=3 averaging: (10+20+30)/3=20.0. Guards against hardcoded denominator=2."""
        g0 = _full_gpu_entry(gfx=10, umc=0, mm=5)
        g1 = _full_gpu_entry(gfx=20, umc=60, mm=5)
        g2 = _full_gpu_entry(gfx=30, umc=120, mm=5)
        out = parse_gpu_metrics([g0, g1, g2])
        self.assertEqual(out["gpu.gfx_activity"], 20.0)  # (10+20+30)/3
        self.assertEqual(out["gpu.umc_activity"], 60.0)  # (0+60+120)/3
        self.assertEqual(out["gpu.mm_activity"], 5.0)  # (5+5+5)/3

    def test_vram_and_energy_summed_three_gpus(self):
        """N=3 sum: guards against loop body that caps at 2 entries or re-inits acc."""
        g0 = _full_gpu_entry(total=100, used=10, free=90, energy=1.0)
        g1 = _full_gpu_entry(total=200, used=20, free=180, energy=2.0)
        g2 = _full_gpu_entry(total=300, used=30, free=270, energy=3.0)
        out = parse_gpu_metrics([g0, g1, g2])
        self.assertEqual(out["gpu.total_vram"], 600)
        self.assertEqual(out["gpu.used_vram"], 60)
        self.assertEqual(out["gpu.free_vram"], 540)
        self.assertEqual(out["gpu.energy_j"], 6.0)

    def test_never_raises_on_list_of_empty_entries(self):
        """Contract: never raises. All-empty entries -> all keys present, None."""
        out = parse_gpu_metrics([{}, {}, {}])
        self.assertEqual(set(out.keys()), set(ALL_KEYS))
        for k in ALL_KEYS:
            with self.subTest(key=k):
                self.assertIsNone(out[k])


# ---------------------------------------------------------------------------
# capture_gpu_metrics — I/O subsystem (orch-delegating, not a pure parser)
# Classification: integration boundary; tested only at the mock seam.
# Contract: calls orch to run amd-smi, passes the JSON list to parse_gpu_metrics,
# returns whatever parse_gpu_metrics returns. Never raises on malformed output.
# ---------------------------------------------------------------------------


class TestCaptureGpuMetrics(unittest.TestCase):
    """capture_gpu_metrics delegates to parse_gpu_metrics and wraps the orch call.

    The function requires a live ContainerOrchestrator to invoke amd-smi, so
    unit tests mock the orch dependency and verify delegation semantics only.
    They never assert on hardware-specific values.
    """

    def _make_orch(self, raw_gpu_list):
        """Return a mock orchestrator whose exec_on_head result decodes to raw_gpu_list.

        The real ContainerOrchestrator.exec_on_head(cmd) returns {host: str};
        we mock the same shape so tests are grounded in the actual interface contract.
        """
        import json

        orch = MagicMock()
        orch.exec_on_head.return_value = {"node0": json.dumps(raw_gpu_list)}
        return orch

    def test_happy_path_key_set_matches_all_keys(self):
        """Given a valid amd-smi JSON list, capture_gpu_metrics returns all 7 keys,
        delegates to parse_gpu_metrics, and passes the parsed values through."""
        orch = self._make_orch([_full_gpu_entry()])
        with patch("cvs.lib.utils.gpu.parse_gpu_metrics", wraps=parse_gpu_metrics) as mock_parse:
            out = capture_gpu_metrics(orch)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out.keys()), set(ALL_KEYS))
        mock_parse.assert_called_once_with([_full_gpu_entry()])
        # Pin the exact command string sent to amd-smi (host-side, no sudo needed).
        orch.exec_on_head.assert_called_once_with("amd-smi metric --json")
        # Verify parse result is actually returned, not silently discarded.
        self.assertEqual(out["gpu.gfx_activity"], 30)
        self.assertIsNotNone(out["gpu.total_vram"])

    def test_multi_host_entries_aggregated_together(self):
        """All hosts' GPU entries must be pooled before aggregation.

        A mutant that reads only the first host's data would yield gfx=10
        (average of one entry), not 15.0 (average across both hosts' entries).
        """
        import json

        orch = MagicMock()
        orch.exec_on_head.return_value = {
            "node0": json.dumps([_full_gpu_entry(gfx=10)]),
            "node1": json.dumps([_full_gpu_entry(gfx=20)]),
        }
        out = capture_gpu_metrics(orch)
        self.assertEqual(set(out.keys()), set(ALL_KEYS))
        self.assertAlmostEqual(out["gpu.gfx_activity"], 15.0)

    def test_no_raise_on_empty_gpu_list(self):
        """Empty GPU list -> all 7 keys, all None. Must not raise."""
        orch = self._make_orch([])
        out = capture_gpu_metrics(orch)
        self.assertEqual(set(out.keys()), set(ALL_KEYS))
        for k in ALL_KEYS:
            with self.subTest(key=k):
                self.assertIsNone(out[k])

    def test_no_raise_on_malformed_orch_output(self):
        """If orch returns non-JSON text, capture_gpu_metrics degrades; never raises."""
        orch = MagicMock()
        orch.exec_on_head.return_value = {"node0": "not valid json at all"}
        try:
            out = capture_gpu_metrics(orch)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"capture_gpu_metrics raised unexpectedly: {exc!r}")
        else:
            self.assertEqual(set(out.keys()), set(ALL_KEYS))
            for k in ALL_KEYS:
                with self.subTest(key=k):
                    self.assertIsNone(out[k])

    def test_gpu_data_envelope_unwrapped(self):
        """ROCm 6.x amd-smi wraps the GPU list as {"gpu_data": [...]}; must be unwrapped."""
        import json

        orch = MagicMock()
        orch.exec_on_head.return_value = {"node0": json.dumps({"gpu_data": [_full_gpu_entry(gfx=42)]})}
        out = capture_gpu_metrics(orch)
        self.assertEqual(set(out.keys()), set(ALL_KEYS))
        self.assertEqual(out["gpu.gfx_activity"], 42)

    def test_no_raise_on_valid_json_wrong_type(self):
        """Valid JSON that decodes to a non-list (dict, null, scalar, string)
        must degrade gracefully — never raises, returns all-None."""
        import json

        non_list_values = [{}, None, 42, "string"]
        for val in non_list_values:
            with self.subTest(decoded_type=type(val).__name__):
                orch = MagicMock()
                orch.exec_on_head.return_value = {"node0": json.dumps(val)}
                try:
                    out = capture_gpu_metrics(orch)
                except Exception as exc:  # noqa: BLE001
                    self.fail(f"capture_gpu_metrics raised on decoded {val!r}: {exc!r}")
                else:
                    self.assertEqual(set(out.keys()), set(ALL_KEYS))
                    for k in ALL_KEYS:
                        self.assertIsNone(out[k])


# ---------------------------------------------------------------------------
# GPU_METRICS / GPU_METRIC_UNITS — module constants (invariants)
# ---------------------------------------------------------------------------


class TestGpuMetricsConstants(unittest.TestCase):
    """Invariants tying GPU_METRICS, GPU_METRIC_UNITS, _RAW_GPU_FIELDS, and parser output keys."""

    # Raw amd-smi parser output fields (internal; not surfaced as HTML rows).
    EXPECTED_RAW_NAMES = {
        "gfx_activity",
        "umc_activity",
        "mm_activity",
        "total_vram",
        "used_vram",
        "free_vram",
        "energy_j",
    }

    # --- GPU_METRICS (derived, human-readable) ---

    def test_derived_unit_strings_match_spec(self):
        """Unit strings pinned to spec values."""
        EXPECTED_UNITS = {
            "peak_gpu_memory_mb": "MB",
            "model_load_memory_mb": "MB",
            "model_load_s": "s",
            "gpu_bandwidth_util_pct": "%",
            "gpu_compute_util_pct": "%",
        }
        self.assertEqual(GPU_METRIC_UNITS, EXPECTED_UNITS)

    # --- _RAW_GPU_FIELDS (amd-smi parser output) ---

    def test_raw_fields_covers_all_seven_amd_smi_fields(self):
        raw_names = {short for short, _unit in _RAW_GPU_FIELDS}
        self.assertEqual(raw_names, self.EXPECTED_RAW_NAMES)

    def test_raw_unit_strings_match_spec(self):
        EXPECTED_RAW_UNITS = {
            "gfx_activity": "%",
            "umc_activity": "%",
            "mm_activity": "%",
            "total_vram": "MB",
            "used_vram": "MB",
            "free_vram": "MB",
            "energy_j": "J",
        }
        self.assertEqual(_RAW_GPU_FIELD_UNITS, EXPECTED_RAW_UNITS)

    def test_parse_gpu_metrics_emits_key_for_every_raw_field(self):
        """parse_gpu_metrics([full]) produces "gpu.<k>" for every k in _RAW_GPU_FIELDS."""
        self.assertGreater(len(_RAW_GPU_FIELDS), 0, "_RAW_GPU_FIELDS must not be empty")
        out = parse_gpu_metrics([_full_gpu_entry()])
        for short, _unit in _RAW_GPU_FIELDS:
            with self.subTest(metric=short):
                self.assertIn(f"gpu.{short}", out)

    def test_derived_metrics_not_emitted_by_parser(self):
        """GPU_METRICS (derived) are computed by the calling suite, not by the
        parser. parse_gpu_metrics must NOT emit keys for derived short names."""
        out = parse_gpu_metrics([_full_gpu_entry()])
        for short, _unit in GPU_METRICS:
            with self.subTest(metric=short):
                self.assertNotIn(f"gpu.{short}", out)


class TestMean(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(_mean([]))

    def test_all_none(self):
        self.assertIsNone(_mean([None, None]))

    def test_normal(self):
        self.assertAlmostEqual(_mean([1.0, 3.0]), 2.0)

    def test_skips_none(self):
        self.assertAlmostEqual(_mean([None, 4.0, None, 2.0]), 3.0)


class TestAggReadings(unittest.TestCase):
    def test_empty(self):
        result = agg_readings([])
        self.assertIsNone(result["peak_gpu_memory_mb"])
        self.assertIsNone(result["gpu_compute_util_pct"])
        self.assertIsNone(result["gpu_bandwidth_util_pct"])

    def test_all_none_values(self):
        readings = [{"gpu.used_vram": None, "gpu.gfx_activity": None, "gpu.umc_activity": None}]
        result = agg_readings(readings)
        self.assertIsNone(result["peak_gpu_memory_mb"])

    def test_normal(self):
        readings = [
            {"gpu.used_vram": 1000, "gpu.gfx_activity": 80.0, "gpu.umc_activity": 60.0},
            {"gpu.used_vram": 2000, "gpu.gfx_activity": 90.0, "gpu.umc_activity": 70.0},
        ]
        result = agg_readings(readings)
        self.assertEqual(result["peak_gpu_memory_mb"], 2000)
        self.assertAlmostEqual(result["gpu_compute_util_pct"], 85.0)
        self.assertAlmostEqual(result["gpu_bandwidth_util_pct"], 65.0)


class TestCaptureGpuMetricsMultiNode(unittest.TestCase):
    """Tests for capture_gpu_metrics with the nodes= parameter (orch.exec(hosts=...))."""

    def _make_gpu_json(self, used_vram: int, gfx: float = 80.0) -> str:
        import json

        return json.dumps(
            [
                {
                    "usage": {
                        "gfx_activity": {"value": gfx},
                        "umc_activity": {"value": 10.0},
                        "mm_activity": {"value": "N/A"},
                    },
                    "mem_usage": {
                        "used_vram": {"value": used_vram},
                        "total_vram": {"value": used_vram + 1000},
                        "free_vram": {"value": 1000},
                    },
                    "energy": {"total_energy_consumption": {"value": 50.0}},
                }
            ]
        )

    def _make_exec_by_hosts(self, host_to_vram: dict, gfx: float = 80.0):
        """Build an orch.exec side_effect keyed by the hosts= kwarg."""

        def _exec(cmd, hosts=None):
            return {h: self._make_gpu_json(host_to_vram[h], gfx) for h in hosts}

        return _exec

    def test_nodes_none_calls_exec_on_head(self):
        """nodes=None must call orch.exec_on_head (regression guard)."""
        orch = MagicMock()
        orch.exec_on_head.return_value = {"host0": self._make_gpu_json(1000)}
        from cvs.lib.utils.gpu import capture_gpu_metrics

        result = capture_gpu_metrics(orch, nodes=None)
        orch.exec_on_head.assert_called_once_with("amd-smi metric --json")
        self.assertEqual(result["gpu.used_vram"], 1000)

    def test_nodes_provided_calls_orch_exec_with_hosts_not_exec_on_head(self):
        """nodes provided: orch.exec(cmd, hosts=...) is called, orch.exec_on_head is NOT."""
        orch = MagicMock()
        orch.exec.side_effect = self._make_exec_by_hosts({"prefill-host": 2000, "decode-host": 3000})
        from cvs.lib.utils.gpu import capture_gpu_metrics

        capture_gpu_metrics(
            orch,
            nodes=[("prefill-0", ["prefill-host"]), ("decode-0", ["decode-host"])],
        )
        orch.exec_on_head.assert_not_called()
        orch.exec.assert_any_call("amd-smi metric --json", hosts=["prefill-host"])
        orch.exec.assert_any_call("amd-smi metric --json", hosts=["decode-host"])

    def test_nodes_vram_summed_across_nodes(self):
        """VRAM from all nodes is summed in the merged result."""
        orch = MagicMock()
        orch.exec.side_effect = self._make_exec_by_hosts({"p": 2000, "d": 3000})
        from cvs.lib.utils.gpu import capture_gpu_metrics

        result = capture_gpu_metrics(orch, nodes=[("prefill-0", ["p"]), ("decode-0", ["d"])])
        self.assertEqual(result["gpu.used_vram"], 5000)

    def test_nodes_activity_averaged_across_nodes(self):
        """GFX activity from all nodes is averaged."""
        orch = MagicMock()

        def _exec(cmd, hosts=None):
            gfx = 60.0 if hosts == ["p"] else 100.0
            return {hosts[0]: self._make_gpu_json(1000, gfx)}

        orch.exec.side_effect = _exec
        from cvs.lib.utils.gpu import capture_gpu_metrics

        result = capture_gpu_metrics(orch, nodes=[("prefill-0", ["p"]), ("decode-0", ["d"])])
        self.assertAlmostEqual(result["gpu.gfx_activity"], 80.0)

    def test_nodes_exception_propagates(self):
        """Exception from orch.exec propagates (not swallowed)."""
        orch = MagicMock()
        orch.exec.side_effect = RuntimeError("ssh failed")
        from cvs.lib.utils.gpu import capture_gpu_metrics

        with self.assertRaises(RuntimeError):
            capture_gpu_metrics(orch, nodes=[("prefill-0", ["p"])])

    def test_nodes_gpu_data_envelope_unwrapped_and_merged(self):
        """Each node may independently use the {"gpu_data": [...]} envelope."""
        import json

        orch = MagicMock()

        def _exec(cmd, hosts=None):
            vram = {"p": 1000, "d": 2000}[hosts[0]]
            return {
                hosts[0]: json.dumps(
                    {
                        "gpu_data": [
                            {
                                "usage": {
                                    "gfx_activity": {"value": 50.0},
                                    "umc_activity": {"value": 10.0},
                                    "mm_activity": {"value": "N/A"},
                                },
                                "mem_usage": {
                                    "used_vram": {"value": vram},
                                    "total_vram": {"value": vram + 100},
                                    "free_vram": {"value": 100},
                                },
                                "energy": {"total_energy_consumption": {"value": 1.0}},
                            }
                        ]
                    }
                )
            }

        orch.exec.side_effect = _exec
        from cvs.lib.utils.gpu import capture_gpu_metrics

        result = capture_gpu_metrics(orch, nodes=[("prefill-0", ["p"]), ("decode-0", ["d"])])
        self.assertEqual(result["gpu.used_vram"], 3000)


# ---------------------------------------------------------------------------
# start_gpu_poller / stop_and_collect_gpu_poller / GpuPollerHandle
#
# poll_gpu_metrics (thread + shared-SSH polling) was replaced by a detached
# remote background script per node, read back via ordinary sequential
# exec/exec_on_head calls -- avoids a second OS thread sharing the
# orchestrator's SSH transport with the main log-tail polling thread (real
# HW-observed SessionError(OutOfBoundaryError()) race).
# ---------------------------------------------------------------------------


def _gpu_chunk_text(used_vram: int = 1000, gfx: float = 80.0, umc: float = 60.0) -> str:
    """One well-formed amd-smi --json chunk (single GPU entry)."""
    import json

    return json.dumps([_full_gpu_entry(gfx=gfx, umc=umc, used=used_vram)])


def _poller_file_text(*chunks: str) -> str:
    """Join chunks with _RECORD_SEP, well-terminated (trailing separator)."""
    return "".join(c + _RECORD_SEP + "\n" for c in chunks)


class TestGpuPollerHandle(unittest.TestCase):
    def test_fields(self):
        handle = GpuPollerHandle(run_id="r1", marker="cvs_gpu_poll_r1", nodes=None, paths="/tmp/cvs_gpu_poll_r1.log")
        self.assertEqual(handle.run_id, "r1")
        self.assertEqual(handle.marker, "cvs_gpu_poll_r1")
        self.assertIsNone(handle.nodes)
        self.assertEqual(handle.paths, "/tmp/cvs_gpu_poll_r1.log")


class TestStartGpuPollerSingleNode(unittest.TestCase):
    """nodes=None: one write+launch pair via orch.exec_on_head."""

    def test_run_id_sanitized_in_marker_and_paths(self):
        """A raw pytest node id (::, [, ], /) must not leak into marker/paths.

        handle.paths is a legitimate filesystem path (e.g. /tmp/<marker>.log),
        so it's the basename -- not the whole path string -- that must be
        free of the sanitized-away characters.
        """
        orch = MagicMock()
        orch.exec_on_head.return_value = {"head": ""}
        handle = start_gpu_poller(orch, run_id="test_foo.py::test_bar[isl-osl-4]")
        basename = pathlib.Path(handle.paths).name
        for bad_char in ("::", "[", "]", "/"):
            with self.subTest(bad_char=bad_char):
                self.assertNotIn(bad_char, handle.marker)
                self.assertNotIn(bad_char, basename)
        self.assertTrue(handle.marker.startswith("cvs_gpu_poll_"))

    def test_launch_uses_exec_on_head_not_exec(self):
        orch = MagicMock()
        orch.exec_on_head.return_value = {"head": ""}
        start_gpu_poller(orch, run_id="r1")
        orch.exec.assert_not_called()
        self.assertTrue(orch.exec_on_head.called)

    def test_launch_command_contains_nohup_and_record_sep(self):
        orch = MagicMock()
        orch.exec_on_head.return_value = {"head": ""}
        start_gpu_poller(orch, run_id="r1")
        all_cmds = " ".join(c.args[0] for c in orch.exec_on_head.call_args_list)
        self.assertIn("nohup", all_cmds)
        self.assertIn(_RECORD_SEP, all_cmds)

    def test_default_hard_cap_yields_960_iterations(self):
        """hard_cap_s=14400, poll_interval_s=15 -> 14400 // 15 == 960."""
        orch = MagicMock()
        orch.exec_on_head.return_value = {"head": ""}
        start_gpu_poller(orch, run_id="r1", poll_interval_s=15, hard_cap_s=14400)
        all_cmds = " ".join(c.args[0] for c in orch.exec_on_head.call_args_list)
        self.assertIn("960", all_cmds)

    def test_returns_handle_with_nodes_none_and_str_path(self):
        orch = MagicMock()
        orch.exec_on_head.return_value = {"head": ""}
        handle = start_gpu_poller(orch, run_id="r1")
        self.assertIsNone(handle.nodes)
        self.assertIsInstance(handle.paths, str)

    def test_launch_failure_propagates(self):
        orch = MagicMock()
        orch.exec_on_head.side_effect = RuntimeError("ssh failed")
        with self.assertRaises(RuntimeError):
            start_gpu_poller(orch, run_id="r1")


class TestStartGpuPollerMultiNode(unittest.TestCase):
    """nodes=[hosts]: one write+launch pair per host via orch.exec(hosts=[host])."""

    def test_one_launch_call_per_host(self):
        orch = MagicMock()
        orch.exec.return_value = {"h1": ""}
        start_gpu_poller(orch, run_id="r1", nodes=["h1", "h2"])
        orch.exec_on_head.assert_not_called()
        hosts_seen = [c.kwargs.get("hosts") for c in orch.exec.call_args_list]
        self.assertIn(["h1"], hosts_seen)
        self.assertIn(["h2"], hosts_seen)

    def test_returns_handle_with_per_host_paths(self):
        orch = MagicMock()
        orch.exec.return_value = {"h1": ""}
        handle = start_gpu_poller(orch, run_id="r1", nodes=["h1", "h2"])
        self.assertEqual(handle.nodes, ["h1", "h2"])
        self.assertIsInstance(handle.paths, dict)
        self.assertEqual(set(handle.paths), {"h1", "h2"})


class TestStopAndCollectGpuPollerSingleNode(unittest.TestCase):
    """nodes=None: pkill via exec_on_head, read-back via exec_on_head cat."""

    def _handle(self):
        return GpuPollerHandle(run_id="r1", marker="cvs_gpu_poll_r1", nodes=None, paths="/tmp/cvs_gpu_poll_r1.log")

    def test_pkill_contains_marker(self):
        orch = MagicMock()
        orch.exec_on_head.return_value = {"head": ""}
        stop_and_collect_gpu_poller(orch, self._handle())
        all_cmds = " ".join(c.args[0] for c in orch.exec_on_head.call_args_list)
        self.assertIn("cvs_gpu_poll_r1", all_cmds)

    def test_well_terminated_chunks_all_parse(self):
        """3 well-formed chunks + correct trailing separator -> 3 readings, 0 failed."""
        text = _poller_file_text(_gpu_chunk_text(1000), _gpu_chunk_text(2000), _gpu_chunk_text(3000))
        orch = MagicMock()
        orch.exec_on_head.side_effect = [{"head": ""}, {"head": text}]
        readings = stop_and_collect_gpu_poller(orch, self._handle())
        self.assertEqual(len(readings), 3)
        self.assertEqual(readings[-1]["gpu.used_vram"], 3000)

    def test_trailing_phantom_not_counted_as_failure_but_mid_malformed_is(self):
        """Trailing separator's phantom empty chunk must not count as failed;
        a genuinely empty/malformed chunk mixed in the middle must."""
        text = (
            _gpu_chunk_text(1000)
            + _RECORD_SEP
            + "\n"
            + ""
            + _RECORD_SEP
            + "\n"
            + _gpu_chunk_text(3000)
            + _RECORD_SEP
            + "\n"
        )
        orch = MagicMock()
        orch.exec_on_head.side_effect = [{"head": ""}, {"head": text}]
        log_path = str(pathlib.Path(__file__).parent / "_tmp_test_gpu_poller_log.txt")
        try:
            readings = stop_and_collect_gpu_poller(orch, self._handle(), log_path=log_path)
            self.assertEqual(len(readings), 2)
            content = pathlib.Path(log_path).read_text()
            self.assertIn("samples:              3 (1 failed, excluded)", content)
        finally:
            pathlib.Path(log_path).unlink(missing_ok=True)

    def test_read_failure_degrades_not_raises(self):
        orch = MagicMock()
        orch.exec_on_head.side_effect = [{"head": ""}, RuntimeError("ssh failed")]
        try:
            readings = stop_and_collect_gpu_poller(orch, self._handle())
        except Exception as exc:  # noqa: BLE001
            self.fail(f"stop_and_collect_gpu_poller raised unexpectedly: {exc!r}")
        else:
            self.assertEqual(readings, [])

    def test_pkill_failure_degrades_not_raises_and_readback_still_happens(self):
        text = _poller_file_text(_gpu_chunk_text(1000))
        orch = MagicMock()
        orch.exec_on_head.side_effect = [RuntimeError("pkill failed"), {"head": text}]
        try:
            readings = stop_and_collect_gpu_poller(orch, self._handle())
        except Exception as exc:  # noqa: BLE001
            self.fail(f"stop_and_collect_gpu_poller raised unexpectedly: {exc!r}")
        else:
            self.assertEqual(len(readings), 1)

    def test_summary_log_matches_agg_readings(self):
        text = _poller_file_text(_gpu_chunk_text(1000, gfx=80.0, umc=60.0), _gpu_chunk_text(2000, gfx=90.0, umc=70.0))
        orch = MagicMock()
        orch.exec_on_head.side_effect = [{"head": ""}, {"head": text}]
        log_path = str(pathlib.Path(__file__).parent / "_tmp_test_gpu_poller_summary.txt")
        try:
            readings = stop_and_collect_gpu_poller(
                orch, self._handle(), log_path=log_path, model_load_s=12.3, model_load_memory_mb=456
            )
            agg = agg_readings(readings)
            content = pathlib.Path(log_path).read_text()
            self.assertIn("--- summary ---", content)
            self.assertIn(f"peak_gpu_memory_mb:   {agg['peak_gpu_memory_mb']:.0f} MB", content)
            self.assertIn("model_load_memory_mb: 456 MB", content)
            self.assertIn("model_load_s:         12.3 s", content)
            self.assertIn(f"gpu_compute_util_pct:  {agg['gpu_compute_util_pct']:.1f} %", content)
            self.assertIn(f"gpu_bandwidth_util_pct: {agg['gpu_bandwidth_util_pct']:.1f} %", content)
        finally:
            pathlib.Path(log_path).unlink(missing_ok=True)


class TestStopAndCollectGpuPollerMultiNode(unittest.TestCase):
    """nodes=[hosts]: pkill via exec(hosts=...), round-aligned merge across hosts."""

    def _handle(self, nodes):
        return GpuPollerHandle(
            run_id="r1",
            marker="cvs_gpu_poll_r1",
            nodes=nodes,
            paths={h: f"/tmp/cvs_gpu_poll_r1.log" for h in nodes},
        )

    def test_round_alignment_longer_host_extends_not_truncates(self):
        """Host A: 3 chunks, host B: 2 chunks -> 3 rounds, round 3 uses only A."""
        text_a = _poller_file_text(_gpu_chunk_text(1000), _gpu_chunk_text(2000), _gpu_chunk_text(3000))
        text_b = _poller_file_text(_gpu_chunk_text(500), _gpu_chunk_text(600))
        orch = MagicMock()

        def _exec(cmd, hosts=None):
            if "pkill" in cmd:
                return {h: "" for h in hosts}
            host = hosts[0]
            return {host: text_a if host == "A" else text_b}

        orch.exec.side_effect = _exec
        readings = stop_and_collect_gpu_poller(orch, self._handle(["A", "B"]))
        self.assertEqual(len(readings), 3)
        # round 3 (index 2) merges only host A's 3000 vram entry.
        self.assertEqual(readings[2]["gpu.used_vram"], 3000)

    def test_round_alignment_not_counted_as_failed_when_one_host_has_data(self):
        text_a = _poller_file_text(_gpu_chunk_text(1000), _gpu_chunk_text(2000), _gpu_chunk_text(3000))
        text_b = _poller_file_text(_gpu_chunk_text(500), _gpu_chunk_text(600))
        orch = MagicMock()

        def _exec(cmd, hosts=None):
            if "pkill" in cmd:
                return {h: "" for h in hosts}
            host = hosts[0]
            return {host: text_a if host == "A" else text_b}

        orch.exec.side_effect = _exec
        log_path = str(pathlib.Path(__file__).parent / "_tmp_test_gpu_poller_multinode.txt")
        try:
            stop_and_collect_gpu_poller(orch, self._handle(["A", "B"]), log_path=log_path)
            content = pathlib.Path(log_path).read_text()
            self.assertIn("samples:              3", content)
            self.assertNotIn("failed", content)
        finally:
            pathlib.Path(log_path).unlink(missing_ok=True)

    def test_round_failed_when_every_contributing_host_malformed(self):
        """Round where every host's chunk at that index is malformed/empty -> counted as failed."""
        text_a = _poller_file_text(_gpu_chunk_text(1000), "")
        text_b = _poller_file_text(_gpu_chunk_text(500), "")
        orch = MagicMock()

        def _exec(cmd, hosts=None):
            if "pkill" in cmd:
                return {h: "" for h in hosts}
            host = hosts[0]
            return {host: text_a if host == "A" else text_b}

        orch.exec.side_effect = _exec
        readings = stop_and_collect_gpu_poller(orch, self._handle(["A", "B"]))
        self.assertEqual(len(readings), 1)

    def test_per_node_vram_summary_reflects_last_successful_round(self):
        text_a = _poller_file_text(_gpu_chunk_text(1000), _gpu_chunk_text(2000), _gpu_chunk_text(3000))
        text_b = _poller_file_text(_gpu_chunk_text(500), _gpu_chunk_text(600))
        orch = MagicMock()

        def _exec(cmd, hosts=None):
            if "pkill" in cmd:
                return {h: "" for h in hosts}
            host = hosts[0]
            return {host: text_a if host == "A" else text_b}

        orch.exec.side_effect = _exec
        log_path = str(pathlib.Path(__file__).parent / "_tmp_test_gpu_poller_pernode.txt")
        try:
            stop_and_collect_gpu_poller(orch, self._handle(["A", "B"]), log_path=log_path)
            content = pathlib.Path(log_path).read_text()
            self.assertIn("node_vram_mb [A]: 3000 MB", content)
            self.assertIn("node_vram_mb [B]: 600 MB", content)
        finally:
            pathlib.Path(log_path).unlink(missing_ok=True)

    def test_pkill_broadcasts_to_all_nodes(self):
        orch = MagicMock()
        orch.exec.return_value = {"A": "", "B": ""}
        stop_and_collect_gpu_poller(orch, self._handle(["A", "B"]))
        pkill_calls = [c for c in orch.exec.call_args_list if "pkill" in c.args[0]]
        self.assertTrue(any(c.kwargs.get("hosts") == ["A", "B"] for c in pkill_calls))

    def test_read_failure_for_one_host_degrades_not_raises(self):
        text_a = _poller_file_text(_gpu_chunk_text(1000))
        orch = MagicMock()

        def _exec(cmd, hosts=None):
            if "pkill" in cmd:
                return {h: "" for h in hosts}
            host = hosts[0]
            if host == "B":
                raise RuntimeError("ssh failed")
            return {host: text_a}

        orch.exec.side_effect = _exec
        try:
            readings = stop_and_collect_gpu_poller(orch, self._handle(["A", "B"]))
        except Exception as exc:  # noqa: BLE001
            self.fail(f"stop_and_collect_gpu_poller raised unexpectedly: {exc!r}")
        else:
            self.assertEqual(len(readings), 1)
            self.assertEqual(readings[0]["gpu.used_vram"], 1000)


if __name__ == "__main__":
    unittest.main()
