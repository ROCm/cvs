'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

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
(per tier), that have fewer than ``min_repeats`` samples per side, or whose two
sides have unequal sample counts are reported as INCONCLUSIVE (never as a
regression).

Trustworthiness vs. verdict
---------------------------
``summary.has_regression`` answers "is the candidate slower?". It is only
meaningful when ``summary.trustworthy`` is also true, which answers the prior
question "did we actually measure the thing?". A report is untrustworthy when no
keys were compared, when keys are present on only one side (a truncated sweep),
or when more than ``max_inconclusive_frac`` of the matrix was excluded. Callers
MUST gate on both: a green ``has_regression`` over an untrustworthy report is
the single worst output a regression gate can produce, because it looks
identical to a real pass.
'''

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
    #
    # May also be given per collective, which is how a calibration run writes
    # them now:  {"all_reduce_perf": {"small": .., "mid": .., "large": ..}, ...}
    # A bare {tier: value} dict is still accepted and applies to every
    # collective. See threshold_for().
    "thresholds": {
        "small": 0.20,
        "mid": 0.10,
        "large": 0.05,
    },
    # Hard ceiling on a *derived* threshold, per tier. A calibration run that
    # sees one pathologically noisy key must not be able to raise the bar to a
    # level that hides real regressions: the large tier is bandwidth-bound and
    # stable (measured median CV ~0.24%), so a 12.9% threshold there — which is
    # what safety_factor * p95(CV) produced — is 50x the actual noise and blind
    # to any plausible regression. Derived values are clamped into
    # [min_thresholds, max_thresholds]; config-supplied values are not touched.
    "max_thresholds": {
        "small": 0.15,
        "mid": 0.08,
        "large": 0.06,
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
    #
    # PER TIER, and deliberately much lower for small messages than the single
    # 0.5 GB/s value this used to be. That flat floor silently excluded the
    # ENTIRE 1 KiB - 8 KiB band on every single run (measured: 83 of 368 keys,
    # 22.6%, all with reason "reference median below floor 0.5") because a
    # 32-rank collective simply cannot reach 0.5 GB/s busbw at those sizes. The
    # small-message band is latency-bound and is exactly where a protocol or
    # launch-path regression shows up first, so excluding it by accident made
    # the gate blind to a whole class of regression while still reporting PASS.
    #
    # The floor's only real job is to stop _relative_drop() dividing by a number
    # near zero. These values are ~2 orders of magnitude above the measurement
    # quantisation at each tier, which is enough for that and nothing more.
    # A bare scalar is still accepted and applies to every tier.
    "min_bandwidth_floor": {
        "small": 0.005,
        "mid": 0.05,
        "large": 0.5,
    },
    # Minimum repeats per side; below this a key is INCONCLUSIVE.
    "min_repeats": 2,
    # Require both sides to have the SAME number of samples for a key. A sweep
    # that dies at repeat 4 of 7 still reaches the analysis, and both remaining
    # gates degrade toward PASS with small n (the separation gate in particular
    # almost never fires at n=2), so unequal or truncated samples bias the
    # verdict green. Off => legacy behaviour.
    "require_balanced_samples": True,
    # Fraction of compared keys allowed to be INCONCLUSIVE before the run is
    # considered untrustworthy. Inconclusive keys are not regressions, but a run
    # where most keys were excluded is not a pass either -- it is a run that
    # measured nothing. Reported as summary.inconclusive_exceeded; the caller
    # decides whether to fail. Set to 1.0 to disable.
    "max_inconclusive_frac": 0.10,
}


def merge_config(overrides=None):
    """Return DEFAULT_CONFIG deep-merged with ``overrides`` (one level deep on dicts)."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            # A one-level merge is wrong when the override changes the dict's
            # SHAPE rather than some of its entries. Per-collective thresholds
            # ({collective: {tier: v}}) merged into the default {tier: v} would
            # yield a mixed dict that is neither shape, and threshold_for()
            # would silently fall back to the default tier values -- i.e. a
            # calibrated per-collective table would be accepted and then
            # ignored. Replace outright in that case.
            if key == "thresholds" and _is_per_collective(value) != _is_per_collective(cfg[key]):
                cfg[key] = deepcopy(value)
            else:
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


def _is_per_collective(thresholds):
    """True if ``thresholds`` is {collective: {tier: value}} rather than {tier: value}."""
    return bool(thresholds) and all(isinstance(v, dict) for v in thresholds.values())


def threshold_for(size_bytes, config=None, collective=None):
    """
    Return the relative regression threshold (fraction) for a message size.

    Accepts both threshold shapes:
      {tier: value}                     -- applies to every collective
      {collective: {tier: value}}       -- per-collective calibration

    For the per-collective shape an unknown collective falls back to a
    "__default__" entry if present, else to the tightest (smallest) threshold
    across the known collectives for that tier. Falling back to the *tightest*
    rather than the loosest is deliberate: an uncalibrated collective should
    err toward reporting a regression for a human to look at, never toward
    silently passing one.
    """
    cfg = config or DEFAULT_CONFIG
    tier = size_tier(size_bytes, cfg)
    thresholds = cfg["thresholds"]

    if not _is_per_collective(thresholds):
        return thresholds[tier]

    if collective is not None and collective in thresholds:
        return thresholds[collective][tier]
    if "__default__" in thresholds:
        return thresholds["__default__"][tier]
    return min(v[tier] for v in thresholds.values() if tier in v)


def floor_for(size_bytes, config=None):
    """
    Return the minimum reference bandwidth for a size, honouring per-tier floors.

    Accepts a scalar (legacy, one floor for every tier) or a {tier: value} dict.
    """
    cfg = config or DEFAULT_CONFIG
    floor = cfg["min_bandwidth_floor"]
    if isinstance(floor, dict):
        return float(floor[size_tier(size_bytes, cfg)])
    return float(floor)


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


def compare_key(a_samples, b_samples, size_bytes, config=None, collective=None):
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
    floor = floor_for(size_bytes, cfg)

    result = {
        "metric": metric,
        "size": int(size_bytes),
        "tier": size_tier(size_bytes, cfg),
        "threshold": threshold_for(size_bytes, cfg, collective),
        "floor": floor,
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

    # Guard: unequal sample counts between the sides.
    #
    # This means one side's sweep died partway (the harness appends per repeat,
    # and the analysis test runs even when the pair test failed). Comparing 7
    # A-samples against 3 B-samples is not a paired comparison: the sides no
    # longer share the same thermal/neighbour conditions, and the separation
    # gate's percentiles are computed over different-width distributions. Call
    # it inconclusive rather than quietly returning a green verdict from it.
    if cfg.get("require_balanced_samples", True) and a_stats["n"] != b_stats["n"]:
        result["verdict"] = INCONCLUSIVE
        result["reasons"].append(f"unbalanced samples (A={a_stats['n']}, B={b_stats['n']}) — a sweep did not complete")
        return result

    # Guard: reference too small to compare reliably.
    if a_stats["median"] < floor:
        result["verdict"] = INCONCLUSIVE
        result["reasons"].append(f"reference median {a_stats['median']:.4f} below {result['tier']} floor {floor}")
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
            result["reasons"].append(f"rel_drop {result['rel_drop']:.3f} <= threshold {result['threshold']:.3f}")
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

    # Keys present on exactly one side.
    #
    # These used to be dropped by the set intersection with no trace anywhere.
    # That is a silent-wrong-verdict path: if the candidate aborts after
    # printing the smaller sizes, the largest messages — where bandwidth
    # regressions matter most and the threshold is tightest — simply vanish
    # from the comparison, and the report says "0 regressions" over a smaller
    # keys_compared that nothing checks. Surface them as a first-class count so
    # the caller can refuse to issue a verdict.
    missing_keys = []
    for key in sorted(set(a_samples) ^ set(b_samples)):
        name, size, dtype, in_place = key
        missing_keys.append(
            {
                "key": {"name": name, "size": size, "type": dtype, "inPlace": in_place},
                "present_in": "reference" if key in a_samples else "candidate",
            }
        )

    # Evaluate each common key for candidacy.
    per_key = {}
    for key in common_keys:
        name, size, dtype, in_place = key
        verdict = compare_key(a_samples[key], b_samples[key], size, cfg, collective=name)
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

    # A run where most keys were excluded measured nothing; that is not a pass.
    max_frac = float(cfg.get("max_inconclusive_frac", 1.0))
    total = len(keys_list)
    inconclusive_frac = (len(inconclusive) / total) if total else 0.0
    inconclusive_exceeded = total > 0 and inconclusive_frac > max_frac

    # Distinct from has_regression on purpose. has_regression means "we measured
    # this and the candidate is slower". trustworthy=False means "do not read a
    # verdict off this report at all" — no data, half the keys missing, or so
    # much of the matrix excluded that a green result is meaningless. Callers
    # must gate on both; only one of them is about the code under test.
    trustworthy = total > 0 and not missing_keys and not inconclusive_exceeded

    report = {
        "config": cfg,
        "summary": {
            "keys_compared": total,
            "regressions": len(regressions),
            "inconclusive": len(inconclusive),
            "inconclusive_frac": round(inconclusive_frac, 4),
            "inconclusive_exceeded": inconclusive_exceeded,
            "max_inconclusive_frac": max_frac,
            "missing_keys": len(missing_keys),
            "candidates": len(candidates),
            "has_regression": len(regressions) > 0,
            "trustworthy": trustworthy,
        },
        "keys": keys_list,
        "regressions": regressions,
        "missing_keys_detail": missing_keys[:50],
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
    per_collective = {}
    for (name, size, dtype, in_place), vals in samples.items():
        if len(vals) < 2:
            continue
        med = median(vals)
        if med < floor_for(size, cfg):
            continue
        cv = statistics.pstdev(vals) / med if med else 0.0
        rel_range = (max(vals) - min(vals)) / med if med else 0.0
        tier = size_tier(size, cfg)
        per_tier[tier].append((cv, rel_range))
        per_collective.setdefault(name, {"small": [], "mid": [], "large": []})[tier].append((cv, rel_range))

    def _stats(lst):
        if not lst:
            return None
        cvs = [x[0] for x in lst]
        ranges = [x[1] for x in lst]
        return {
            "n_keys": len(lst),
            "cv_median": percentile(cvs, 50),
            "cv_mad": _mad(cvs),
            "cv_p95": percentile(cvs, 95),
            "rel_range_p95": percentile(ranges, 95),
        }

    out = {tier: _stats(lst) for tier, lst in per_tier.items()}
    out["by_collective"] = {
        name: {tier: _stats(lst) for tier, lst in tiers.items()} for name, tiers in per_collective.items()
    }
    return out


def _mad(values):
    """
    Median absolute deviation, scaled to be comparable to a standard deviation
    for normally-distributed data (x1.4826).

    Used instead of p95 as the spread estimator when deriving thresholds. p95 of
    a per-key CV distribution is dominated by its worst few keys: on this
    cluster the large tier has a median CV of 0.24% but a p95 of 6.45%, so a
    p95-based threshold ends up ~50x the typical noise and blind to any real
    regression. MAD ignores that tail by construction, which is the whole point
    — the tail is a handful of flaky keys, not the noise level of the tier.
    """
    data = sorted(float(v) for v in values)
    if not data:
        return 0.0
    med = median(data)
    return 1.4826 * median([abs(v - med) for v in data])


def derive_thresholds(
    control_runs,
    config=None,
    safety_factor=2.0,
    min_thresholds=None,
    max_thresholds=None,
    mad_k=3.0,
    per_collective=True,
):
    """
    Recommend regression thresholds from a control (A=B) dataset.

    A tier's threshold is ``safety_factor * (median(CV) + mad_k * MAD(CV))``,
    clamped into ``[min_thresholds, max_thresholds]``.

    This replaces the previous ``safety_factor * p95(CV)``. p95 is an outlier
    statistic: it tracks the noisiest key in a tier, not the tier's noise. On
    this cluster that produced a 12.9% threshold for the large tier whose median
    CV is 0.24% — a ~54x gap that would let a genuine 10% bandwidth loss on
    every large size pass silently. ``median + k*MAD`` is the robust equivalent:
    it describes where the bulk of the keys actually sit and ignores the tail,
    and the explicit max clamp guarantees no calibration run can ever raise the
    bar past the point of usefulness.

    Args:
      control_runs: control dataset (same build run repeatedly).
      config: optional config overrides.
      safety_factor: multiple of the robust spread to use as the threshold.
      min_thresholds: per-tier floors; defaults to small=0.10, mid=0.05, large=0.03.
      max_thresholds: per-tier ceilings; defaults to config["max_thresholds"].
      mad_k: how many MADs above the median CV to sit.
      per_collective: also emit a per-collective threshold table. Collectives
        differ in noise by more than tiers do (alltoall in particular), so
        pooling them lets the noisiest collective set the bar for all of them.

    Returns:
      dict {'thresholds': ..., 'noise': ..., 'safety_factor', 'mad_k',
            'estimator': 'median+k*MAD'}
    """
    cfg = merge_config(config)
    noise = measure_noise(control_runs, cfg)
    mins = min_thresholds or {"small": 0.10, "mid": 0.05, "large": 0.03}
    maxes = max_thresholds or cfg.get("max_thresholds") or {"small": 0.15, "mid": 0.08, "large": 0.06}

    def _from_noise(tier_noise, tier):
        if not tier_noise:
            return mins[tier]
        spread = tier_noise["cv_median"] + mad_k * tier_noise["cv_mad"]
        return spread * safety_factor

    def _table(noise_by_tier):
        out = {}
        for tier in ("small", "mid", "large"):
            base = _from_noise(noise_by_tier.get(tier), tier)
            out[tier] = round(min(max(base, mins[tier]), maxes[tier]), 4)
        return out

    result = {
        "thresholds": _table(noise),
        "noise": noise,
        "safety_factor": safety_factor,
        "mad_k": mad_k,
        "estimator": "median+k*MAD",
        "min_thresholds": mins,
        "max_thresholds": maxes,
    }

    if per_collective and noise.get("by_collective"):
        table = {name: _table(tiers) for name, tiers in noise["by_collective"].items()}
        # Keep the pooled table as the fallback for any collective that was not
        # present in the control dataset.
        table["__default__"] = result["thresholds"]
        result["thresholds_by_collective"] = table

    return result


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

    # Print why the report cannot be trusted BEFORE the verdict line, so a human
    # skimming the log cannot read "PASS" without also seeing that it is unsafe.
    if not s.get("trustworthy", True):
        why = []
        if not s["keys_compared"]:
            why.append("no keys were compared at all")
        if s.get("missing_keys"):
            why.append(f"{s['missing_keys']} key(s) present on only one side")
        if s.get("inconclusive_exceeded"):
            why.append(
                f"{s['inconclusive_frac'] * 100:.1f}% inconclusive (budget {s['max_inconclusive_frac'] * 100:.0f}%)"
            )
        lines.append(f"UNTRUSTWORTHY : {'; '.join(why)}")
        lines.append("                a PASS from this report is not evidence of anything.")

    if not s.get("trustworthy", True):
        verdict = "NO VERDICT (untrustworthy)"
    elif s["has_regression"]:
        verdict = "REGRESSION DETECTED"
    else:
        verdict = "PASS"
    lines.append(f"verdict       : {verdict}")
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
