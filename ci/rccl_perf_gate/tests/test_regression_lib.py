#!/usr/bin/env python3
"""Self-checks for the A/B regression detector (``cvs.lib.regression_lib``).

These pin the behaviours that make the perf gate *honest* rather than merely
green: that a truncated or unbalanced run is reported as untrustworthy instead
of passing, that the threshold estimator is robust to a couple of flaky keys,
and that an uncalibrated collective falls back to the tightest threshold rather
than the loosest.

Deliberately NOT under ``cvs/tests/`` -- that tree is collected by the CVS
harness during a real perf run, and these are pure-CPU unit tests that have no
business consuming a GPU allocation. Run them on the login node:

    python3 -m pytest cvs/ci/rccl_perf_gate/tests/test_regression_lib.py -q
    python3 cvs/ci/rccl_perf_gate/tests/test_regression_lib.py      # no pytest

"""

import os
import random
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cvs.lib import regression_lib as R  # noqa: E402

SIZES = [1024, 2048, 4096, 8192, 1 << 20, 1 << 24, 1 << 28, 1 << 29]
LARGE_CUTOFF = 64 * 1024 * 1024


def rows(vals, name="all_reduce_perf"):
    return [{"name": name, "size": s, "type": "float", "inPlace": 0, "busBw": v}
            for s, v in vals.items()]


def noisy(base, cv=0.0024):
    return base * (1 + random.gauss(0, cv))


def test_clean_ab_is_a_trustworthy_pass():
    a = [rows({s: 100.0 + i * 0.1 for s in SIZES}) for i in range(7)]
    r = R.detect_regressions(a, list(a))
    assert r["summary"]["regressions"] == 0
    assert r["summary"]["trustworthy"] is True
    assert r["summary"]["keys_compared"] == len(SIZES)


def test_small_bandwidth_band_is_still_measured():
    """A scalar floor of 0.5 GB/s excluded the entire small band -- 100% of the
    keys most likely to expose a latency regression."""
    small = {1024: 0.02, 2048: 0.05, 4096: 0.09, 8192: 0.2}
    a = [rows(small) for _ in range(7)]
    r = R.detect_regressions(a, list(a))
    assert r["summary"]["inconclusive"] == 0


def test_truncated_candidate_is_untrustworthy_not_a_pass():
    """Sizes present in A but absent from B used to vanish from the comparison,
    so a run that died two-thirds of the way through reported a clean PASS."""
    a = [rows({s: 100.0 for s in SIZES}) for _ in range(7)]
    b = [rows({s: 100.0 for s in SIZES[:5]}) for _ in range(7)]
    r = R.detect_regressions(a, b)
    assert r["summary"]["missing_keys"] == 3
    assert r["summary"]["trustworthy"] is False


def test_unbalanced_repeats_are_untrustworthy():
    a = [rows({s: 100.0 for s in SIZES}) for _ in range(7)]
    b = [rows({s: 100.0 for s in SIZES}) for _ in range(3)]
    r = R.detect_regressions(a, b)
    assert r["summary"]["inconclusive"] == len(SIZES)
    assert r["summary"]["trustworthy"] is False


def test_tightened_thresholds_catch_a_real_ten_percent_regression():
    """The live 0.129 large-tier threshold was ~53x the measured median noise,
    so a 10% drop on the biggest messages sailed through as PASS."""
    random.seed(7)
    a = [rows({s: noisy(100.0) for s in SIZES}) for _ in range(7)]
    b = [rows({s: noisy(90.0 if s > LARGE_CUTOFF else 100.0) for s in SIZES})
         for _ in range(7)]
    old = R.detect_regressions(
        a, b, config={"thresholds": {"small": 0.172, "mid": 0.123, "large": 0.129}})
    new = R.detect_regressions(
        a, b, config={"thresholds": {"small": 0.10, "mid": 0.05, "large": 0.03}})
    assert old["summary"]["regressions"] == 0, "regression baseline changed"
    assert new["summary"]["regressions"] == 2


def test_derive_thresholds_is_robust_to_a_few_flaky_keys():
    """Reproduces the pathology in the live calibration file: most keys at
    ~0.24% CV, two wild ones dragging p95 (and therefore 2*p95) to ~6.5%."""
    random.seed(11)
    control = []
    for _rep in range(8):
        vals = {}
        for i, s in enumerate(SIZES):
            cv = 0.065 if i in (0, 1) else 0.0024
            vals[s] = noisy(100.0, cv)
        control.append(rows(vals))
    d = R.derive_thresholds(control)
    assert d["estimator"] == "median+k*MAD"
    assert d["thresholds"]["large"] <= 0.06, "max_thresholds clamp not applied"
    assert "all_reduce_perf" in d.get("thresholds_by_collective", {})


def test_uncalibrated_collective_falls_back_to_the_tightest_threshold():
    """Falling back to the loosest would let a brand-new collective in with an
    alltoall-sized blind spot."""
    cfg = R.merge_config({"thresholds": {
        "all_reduce_perf": {"small": .10, "mid": .05, "large": .03},
        "alltoall_perf": {"small": .20, "mid": .15, "large": .10},
    }})
    assert R.threshold_for(1 << 29, cfg, "all_reduce_perf") == 0.03
    assert R.threshold_for(1 << 29, cfg, "nope_perf") == 0.03


def test_legacy_scalar_floor_and_flat_thresholds_still_work():
    cfg = R.merge_config({"min_bandwidth_floor": 0.5,
                          "thresholds": {"small": .2, "mid": .1, "large": .05}})
    assert R.floor_for(1024, cfg) == 0.5
    assert R.threshold_for(1024, cfg, "anything") == 0.2


def _main():
    """Run without pytest, so the checks stay available on a bare login node."""
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a harness, report and continue
            failures += 1
            print(f"FAIL  {name}: {exc}")
        else:
            print(f"PASS  {name}")
    print()
    print("ALL PASS" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
