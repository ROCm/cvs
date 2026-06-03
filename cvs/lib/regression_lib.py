'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

"""
Paired A/B regression detector for RCCL performance results.

Motivation
----------
Detecting RCCL performance regressions against a *static* baseline is unreliable,
especially for small messages (1 KiB .. a few MiB) where bus bandwidth is
latency-bound and has large run-to-run variation. A fixed threshold either
produces false positives (CI noise) or has to be set so loose that it hides real
regressions.

Instead we run a *candidate* (B) and a *reference* (A) back-to-back on the same
nodes inside the same allocation, ideally interleaved over several repeats. Most
environmental noise (thermals, stragglers, NIC state, neighbour jobs) is then
common-mode and largely cancels in the paired comparison, so the difference
``A - B`` is far more stable than either absolute number.

This module is intentionally pure-Python with no cluster / SSH / pandas
dependencies so it can be unit-tested exhaustively (including Monte-Carlo
false-positive sweeps) on a login node without GPUs.

Design summary
--------------
For every fully-qualified key ``(collective, size, type, inPlace)`` we collect a
sample of bandwidth measurements for side A and side B (one per repeat) and apply
THREE independent gates. A regression is only *confirmed* when all of them agree,
which is what makes the detector trustworthy in CI:

1. Size-tiered relative threshold
   - small  (<= 1 MiB)   : 20 %   (very noisy, latency-bound)
   - mid    (<= 64 MiB)  : 10 %
   - large  (>  64 MiB)  :  5 %    (bandwidth-bound, stable, regressions matter)
   B must be slower than A by MORE than the tier threshold (median vs median).

2. Non-parametric separation gate
   B's upper quartile must sit below A's lower quartile (``p75(B) < p25(A)``),
   i.e. the two distributions barely overlap. This is robust to single-run
   outliers / stragglers and needs no distributional assumptions.

3. Adjacency confirmation
   A real regression usually spans a contiguous band of message sizes, whereas
   noise tends to be isolated. A candidate size is only confirmed if it belongs
   to a run of >= ``adjacency_min_run`` consecutive candidate sizes within the
   same ``(collective, type, inPlace)`` group.

Additional safety: keys whose reference bandwidth is below ``min_bandwidth_floor``
or that have fewer than ``min_repeats`` samples per side are reported as
INCONCLUSIVE (never as a regression).
"""

import statistics
from copy import deepcopy

# Verdict constants
PASS = "pass"
REGRESSION = "regression"
INCONCLUSIVE = "inconclusive"

KiB = 1024
MiB = 1024 * 1024

DEFAULT_CONFIG = {
    # Metric to compare and its direction. For bandwidth higher is better; a
    # regression means B < A. (Set "higher_is_better": False for latency-like
    # metrics, where a regression means B > A.)
    "metric": "busBw",
    "higher_is_better": True,

    # Relative regression thresholds per size tier (fraction of A).
    "thresholds": {
        "small": 0.20,
        "mid": 0.10,
        "large": 0.05,
    },
    # Inclusive upper byte boundaries for the small / mid tiers.
    "tier_boundaries": {
        "small_max_bytes": 1 * MiB,
        "mid_max_bytes": 64 * MiB,
    },

    # Non-parametric separation gate.
    "separation_gate": True,
    "separation_b_percentile": 75,
    "separation_a_percentile": 25,

    # Adjacency confirmation. Set to 1 to disable (flag isolated sizes too).
    "adjacency_min_run": 2,

    # Keys whose reference (A) median metric is below this floor are skipped
    # (relative deltas explode near zero). Units match the metric (GB/s).
    "min_bandwidth_floor": 0.5,

    # Minimum repeats per side; below this a key is INCONCLUSIVE.
    "min_repeats": 2,
}


def merge_config(overrides=None):
    """Return DEFAULT_CONFIG deep-merged with ``overrides`` (one level deep on dicts)."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


def percentile(samples, pct):
    """
    Linear-interpolation percentile (numpy 'linear' method) without numpy.

    Args:
      samples: non-empty iterable of numbers.
      pct: percentile in [0, 100].

    Returns:
      float percentile value.
    """
    data = sorted(float(s) for s in samples)
    if not data:
        raise ValueError("percentile() requires at least one sample")
    if len(data) == 1:
        return data[0]
    rank = (pct / 100.0) * (len(data) - 1)
    low = int(rank)
    high = min(low + 1, len(data) - 1)
    frac = rank - low
    return data[low] + (data[high] - data[low]) * frac


def median(samples):
    """Median via the 50th percentile helper."""
    return percentile(samples, 50)


def summarize_samples(samples):
    """Return robust summary statistics for a list of samples."""
    data = [float(s) for s in samples]
    return {
        "n": len(data),
        "min": min(data),
        "max": max(data),
        "median": median(data),
        "mean": sum(data) / len(data),
        "p25": percentile(data, 25),
        "p75": percentile(data, 75),
    }


def size_tier(size_bytes, config=None):
    """Classify a message size into 'small' | 'mid' | 'large'."""
    cfg = config or DEFAULT_CONFIG
    bounds = cfg["tier_boundaries"]
    size_bytes = int(size_bytes)
    if size_bytes <= bounds["small_max_bytes"]:
        return "small"
    if size_bytes <= bounds["mid_max_bytes"]:
        return "mid"
    return "large"


def threshold_for(size_bytes, config=None):
    """Return the relative regression threshold (fraction) for a message size."""
    cfg = config or DEFAULT_CONFIG
    return cfg["thresholds"][size_tier(size_bytes, cfg)]


def _group_runs_by_key(runs, metric):
    """
    Flatten a list of runs (each a list of rccl rows) into a mapping of
    ``(name, size, type, inPlace) -> [metric samples]``.

    Rows missing the metric or any key field are ignored.
    """
    samples = {}
    for run in runs:
        for row in run:
            try:
                key = (
                    row["name"],
                    int(row["size"]),
                    row.get("type", "NA"),
                    row.get("inPlace", "NA"),
                )
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            samples.setdefault(key, []).append(value)
    return samples


def _relative_drop(a_value, b_value, higher_is_better):
    """
    Relative regression magnitude (fraction). Positive means B is worse than A.

    For higher-is-better metrics this is ``(A - B) / A``; for lower-is-better
    metrics it is ``(B - A) / A``.
    """
    if a_value == 0:
        return 0.0
    if higher_is_better:
        return (a_value - b_value) / a_value
    return (b_value - a_value) / a_value


def compare_key(a_samples, b_samples, size_bytes, config=None):
    """
    Evaluate a single fully-qualified key and return a candidate verdict dict.

    The returned verdict is a *candidate* only (threshold + separation gates).
    Adjacency confirmation is applied later by ``detect_regressions`` because it
    needs the neighbouring sizes.
    """
    cfg = merge_config(config)
    metric = cfg["metric"]
    hib = cfg["higher_is_better"]

    a_stats = summarize_samples(a_samples)
    b_stats = summarize_samples(b_samples)

    result = {
        "metric": metric,
        "size": int(size_bytes),
        "tier": size_tier(size_bytes, cfg),
        "threshold": threshold_for(size_bytes, cfg),
        "a": a_stats,
        "b": b_stats,
        "rel_drop": _relative_drop(a_stats["median"], b_stats["median"], hib),
        "candidate": False,
        "verdict": PASS,
        "reasons": [],
    }

    # Guard: insufficient repeats.
    if a_stats["n"] < cfg["min_repeats"] or b_stats["n"] < cfg["min_repeats"]:
        result["verdict"] = INCONCLUSIVE
        result["reasons"].append(
            f"insufficient repeats (A={a_stats['n']}, B={b_stats['n']}, need {cfg['min_repeats']})"
        )
        return result

    # Guard: reference too small to compare reliably.
    if a_stats["median"] < cfg["min_bandwidth_floor"]:
        result["verdict"] = INCONCLUSIVE
        result["reasons"].append(
            f"reference median {a_stats['median']:.3f} below floor {cfg['min_bandwidth_floor']}"
        )
        return result

    # Gate 1: size-tiered relative threshold.
    passed_threshold = result["rel_drop"] > result["threshold"]

    # Gate 2: non-parametric separation.
    if cfg["separation_gate"]:
        b_hi = percentile(b_samples, cfg["separation_b_percentile"])
        a_lo = percentile(a_samples, cfg["separation_a_percentile"])
        if hib:
            # regression => B clearly below A
            passed_separation = b_hi < a_lo
        else:
            # regression => B clearly above A
            b_lo = percentile(b_samples, 100 - cfg["separation_b_percentile"])
            a_hi = percentile(a_samples, 100 - cfg["separation_a_percentile"])
            passed_separation = b_lo > a_hi
        result["separation"] = {"b_edge": b_hi if hib else b_lo, "a_edge": a_lo if hib else a_hi}
    else:
        passed_separation = True

    if passed_threshold and passed_separation:
        result["candidate"] = True
    else:
        if not passed_threshold:
            result["reasons"].append(
                f"rel_drop {result['rel_drop']:.3f} <= threshold {result['threshold']:.3f}"
            )
        if not passed_separation:
            result["reasons"].append("distributions overlap (separation gate not met)")
    return result


def detect_regressions(a_runs, b_runs, config=None):
    """
    Run the full paired A/B regression analysis.

    Args:
      a_runs: list of reference runs. Each run is a list of rccl-test rows
        (dicts with 'name', 'size', 'type', 'inPlace' and the configured metric).
      b_runs: list of candidate runs in the same row format.
      config: optional dict of overrides for DEFAULT_CONFIG.

    Returns:
      dict report:
        {
          "config": <effective config>,
          "summary": {"keys_compared", "regressions", "inconclusive",
                      "candidates", "has_regression"},
          "keys": [ per-key verdict dicts, with 'confirmed' set ],
          "regressions": [ confirmed regression verdicts ],
        }
    """
    cfg = merge_config(config)
    metric = cfg["metric"]

    a_samples = _group_runs_by_key(a_runs, metric)
    b_samples = _group_runs_by_key(b_runs, metric)

    common_keys = set(a_samples) & set(b_samples)

    # Evaluate each common key for candidacy.
    per_key = {}
    for key in common_keys:
        name, size, dtype, in_place = key
        verdict = compare_key(a_samples[key], b_samples[key], size, cfg)
        verdict["key"] = {"name": name, "size": size, "type": dtype, "inPlace": in_place}
        verdict["confirmed"] = False
        per_key[key] = verdict

    # Adjacency confirmation: within each (name, type, inPlace) group, sort by
    # size and confirm candidates that belong to a run of >= adjacency_min_run
    # consecutive candidate sizes.
    min_run = max(1, int(cfg["adjacency_min_run"]))
    groups = {}
    for key in per_key:
        name, size, dtype, in_place = key
        groups.setdefault((name, dtype, in_place), []).append(key)

    for group_keys in groups.values():
        group_keys.sort(key=lambda k: k[1])  # by size
        run_start = 0
        n = len(group_keys)
        i = 0
        while i < n:
            if per_key[group_keys[i]]["candidate"]:
                j = i
                while j < n and per_key[group_keys[j]]["candidate"]:
                    j += 1
                run_len = j - i
                if run_len >= min_run:
                    for k in range(i, j):
                        per_key[group_keys[k]]["confirmed"] = True
                        per_key[group_keys[k]]["verdict"] = REGRESSION
                else:
                    for k in range(i, j):
                        per_key[group_keys[k]]["reasons"].append(
                            f"isolated candidate (run length {run_len} < adjacency_min_run {min_run})"
                        )
                i = j
            else:
                i += 1
        _ = run_start  # silence unused

    keys_list = [per_key[k] for k in sorted(per_key, key=lambda k: (k[0], k[2], k[3], k[1]))]
    regressions = [v for v in keys_list if v["verdict"] == REGRESSION]
    inconclusive = [v for v in keys_list if v["verdict"] == INCONCLUSIVE]
    candidates = [v for v in keys_list if v["candidate"]]

    report = {
        "config": cfg,
        "summary": {
            "keys_compared": len(keys_list),
            "regressions": len(regressions),
            "inconclusive": len(inconclusive),
            "candidates": len(candidates),
            "has_regression": len(regressions) > 0,
        },
        "keys": keys_list,
        "regressions": regressions,
    }
    return report


def measure_noise(control_runs, config=None):
    """
    Measure per-tier run-to-run noise from a *control* dataset.

    A control dataset is produced by running the SAME build as both sides (A=B),
    so any spread across repeats is pure run-to-run / environmental noise. This
    is the empirical noise floor used to choose trustworthy thresholds.

    Args:
      control_runs: list of runs (each a list of rccl rows) from one build.
      config: optional overrides (uses 'metric' and 'min_bandwidth_floor').

    Returns:
      dict {tier: {'n_keys', 'cv_median', 'cv_p95', 'rel_range_p95'} or None}
      where cv is the coefficient of variation (stdev/median) per key.
    """
    cfg = merge_config(config)
    samples = _group_runs_by_key(control_runs, cfg["metric"])
    per_tier = {"small": [], "mid": [], "large": []}
    for (name, size, dtype, in_place), vals in samples.items():
        if len(vals) < 2:
            continue
        med = median(vals)
        if med < cfg["min_bandwidth_floor"]:
            continue
        cv = statistics.pstdev(vals) / med if med else 0.0
        rel_range = (max(vals) - min(vals)) / med if med else 0.0
        per_tier[size_tier(size, cfg)].append((cv, rel_range))

    out = {}
    for tier, lst in per_tier.items():
        if not lst:
            out[tier] = None
            continue
        cvs = [x[0] for x in lst]
        ranges = [x[1] for x in lst]
        out[tier] = {
            "n_keys": len(lst),
            "cv_median": percentile(cvs, 50),
            "cv_p95": percentile(cvs, 95),
            "rel_range_p95": percentile(ranges, 95),
        }
    return out


def derive_thresholds(control_runs, config=None, safety_factor=2.0, min_thresholds=None):
    """
    Recommend per-tier regression thresholds from a control (A=B) dataset.

    The recommended threshold for a tier is ``safety_factor * p95(CV)`` of that
    tier's measured run-to-run noise, clamped to a sensible minimum. Sitting the
    threshold a couple of noise-widths above the observed spread is what keeps
    the detector from firing on noise while still catching real shifts.

    Args:
      control_runs: control dataset (same build run repeatedly).
      config: optional config overrides.
      safety_factor: multiple of the p95 noise CV to use as the threshold.
      min_thresholds: per-tier floors; defaults to small=0.10, mid=0.05, large=0.03.

    Returns:
      dict {'thresholds': {tier: value}, 'noise': <measure_noise output>,
            'safety_factor': ...}
    """
    cfg = merge_config(config)
    noise = measure_noise(control_runs, cfg)
    mins = min_thresholds or {"small": 0.10, "mid": 0.05, "large": 0.03}
    thresholds = {}
    for tier in ("small", "mid", "large"):
        tier_noise = noise.get(tier)
        base = tier_noise["cv_p95"] * safety_factor if tier_noise else mins[tier]
        thresholds[tier] = round(max(base, mins[tier]), 3)
    return {"thresholds": thresholds, "noise": noise, "safety_factor": safety_factor}


def format_report(report, max_rows=50):
    """Render a compact human-readable summary of a detect_regressions() report."""
    s = report["summary"]
    lines = []
    lines.append("==================== RCCL A/B Regression Report ====================")
    lines.append(
        f"keys compared : {s['keys_compared']}   "
        f"confirmed regressions : {s['regressions']}   "
        f"inconclusive : {s['inconclusive']}"
    )
    lines.append(f"verdict       : {'REGRESSION DETECTED' if s['has_regression'] else 'PASS'}")
    if report["regressions"]:
        lines.append("")
        lines.append("Confirmed regressions:")
        lines.append(
            f"  {'collective':<20} {'type':<10} {'inPl':>4} {'size':>12} "
            f"{'A_med':>10} {'B_med':>10} {'drop%':>7} {'thr%':>6}"
        )
        for v in report["regressions"][:max_rows]:
            k = v["key"]
            lines.append(
                f"  {k['name']:<20} {str(k['type']):<10} {str(k['inPlace']):>4} {k['size']:>12} "
                f"{v['a']['median']:>10.2f} {v['b']['median']:>10.2f} "
                f"{v['rel_drop'] * 100:>6.1f}% {v['threshold'] * 100:>5.1f}%"
            )
    lines.append("====================================================================")
    return "\n".join(lines)
