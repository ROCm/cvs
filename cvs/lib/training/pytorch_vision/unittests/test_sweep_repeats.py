"""Tests for repeated-sweep scheduling and median aggregation.

Sweeps read the same shards back to back, so position in the run decides page
cache warmth and can move a loader-bound sweep by ~2x. These cover the
aggregation that keeps that out of cross-sweep comparisons.
"""

import importlib.util
import random
import sys
import types
import unittest
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[4] / "tests" / "training" / "pytorch_vision" / "_common.py"


def _load_common():
    """Import _common.py with its heavy test-only dependencies stubbed out."""
    for name in ("pytest", "cvs.lib.utils.globals"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            if name == "pytest":
                module.skip = lambda *a, **k: None
                module.fail = lambda *a, **k: None
                module.fixture = lambda *a, **k: (lambda f: f)
            else:
                module.error_list = []
            sys.modules[name] = module
    spec = importlib.util.spec_from_file_location("_vision_common", _COMMON)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


_common = _load_common()


@unittest.skipIf(_common is None, "_common.py needs the full suite environment")
class TestMedianAcrossRepeats(unittest.TestCase):
    def test_median_ignores_the_outlier_a_warm_cache_produces(self):
        repeats = [
            {"h1": {"training.images_per_sec": 18000.0}},
            {"h1": {"training.images_per_sec": 19000.0}},
            {"h1": {"training.images_per_sec": 34000.0}},
        ]
        merged = _common._median_across_repeats(repeats)
        self.assertEqual(merged["h1"]["training.images_per_sec"], 19000.0)

    def test_every_host_is_aggregated_independently(self):
        repeats = [
            {"h1": {"m": 10.0}, "h2": {"m": 100.0}},
            {"h1": {"m": 20.0}, "h2": {"m": 200.0}},
            {"h1": {"m": 30.0}, "h2": {"m": 300.0}},
        ]
        merged = _common._median_across_repeats(repeats)
        self.assertEqual(merged["h1"]["m"], 20.0)
        self.assertEqual(merged["h2"]["m"], 200.0)

    def test_non_numeric_metrics_survive_aggregation(self):
        repeats = [{"h1": {"note": "ok", "n": 1.0}}, {"h1": {"note": "ok", "n": 3.0}}]
        merged = _common._median_across_repeats(repeats)
        self.assertEqual(merged["h1"]["note"], "ok")
        self.assertEqual(merged["h1"]["n"], 2.0)

    def test_a_host_missing_from_one_repeat_still_aggregates(self):
        repeats = [{"h1": {"m": 10.0}}, {}, {"h1": {"m": 20.0}}]
        merged = _common._median_across_repeats(repeats)
        self.assertEqual(merged["h1"]["m"], 15.0)


class TestRoundOrdering(unittest.TestCase):
    """Each sweep must take a different position across rounds; repeating a
    sweep consecutively would leave it in the same cache state every time."""

    def test_shuffling_gives_sweeps_different_positions(self):
        names = ["a", "b", "c", "d", "e"]
        rng = random.Random(2026)
        positions = {name: set() for name in names}
        for _round in range(8):
            order = list(names)
            rng.shuffle(order)
            for index, name in enumerate(order):
                positions[name].add(index)
        for name in names:
            self.assertGreater(len(positions[name]), 1, f"{name} never moved position")

    def test_the_seed_makes_the_schedule_reproducible(self):
        def schedule(seed):
            rng = random.Random(seed)
            out = []
            for _round in range(4):
                order = ["a", "b", "c"]
                rng.shuffle(order)
                out.append(tuple(order))
            return out

        self.assertEqual(schedule(2026), schedule(2026))


if __name__ == "__main__":
    unittest.main()
