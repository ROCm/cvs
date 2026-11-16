"""Metric contract for PyTorch Training (LLM) artifacts."""

from __future__ import annotations

import math


ARTIFACT_METRICS = (
    ("tokens_per_sec", "tokens/s"),
    ("tokens_per_sec_per_gpu", "tokens/s/GPU"),
    ("tflops_per_sec_per_gpu", "TFLOPS/s/GPU"),
    ("mfu_pct", "%"),
    ("step_time_ms_mean", "ms"),
    ("step_time_ms_p50", "ms"),
    ("step_time_ms_p95", "ms"),
    ("peak_memory_allocated_mb", "MB"),
    ("peak_memory_reserved_mb", "MB"),
    ("checkpoint_save_seconds", "s"),
    ("checkpoint_load_seconds", "s"),
    ("checkpoint_state_match", "bool"),
    ("checkpoint_loss_delta", "-"),
    ("loss_initial", "-"),
    ("loss_final", "-"),
)
OPTIONAL_ARTIFACT_METRICS = (
    ("checkpoint_io_seconds", "s"),
    ("checkpoint_size_gb", "GB"),
    ("scaling_efficiency_pct", "%"),
    ("peak_memory_used_mb", "MB"),
    ("learning_rate_initial", "-"),
    ("learning_rate_final", "-"),
    ("loss_step_100", "-"),
    ("loss_step_500", "-"),
    ("loss_step_1000", "-"),
    ("loss_step_5000", "-"),
    ("eval_loss", "-"),
    ("perplexity", "-"),
    ("eval_tokens", "tokens"),
    ("eval_completed", "bool"),
    ("loss_curve_points", "count"),
    ("loss_curve_decreased", "bool"),
    ("convergence_step", "step"),
    ("convergence_time_seconds", "s"),
    ("energy_kwh", "kWh"),
    ("tokens_per_kwh", "tokens/kWh"),
    ("codecarbon_tracking_active", "bool"),
    ("mmlu_pro_pct", "%"),
    ("bbh_pct", "%"),
    ("math_lvl5_pct", "%"),
    ("gpqa_pct", "%"),
    ("musr_pct", "%"),
    ("gsm8k_pct", "%"),
    ("hellaswag_pct", "%"),
)
DERIVED_METRICS = (
    ("gradient_accumulation_overhead_pct", "%"),
    ("scale_parity_eval_loss_delta", "-"),
)
# Checkpoint benchmarks scored by lm-evaluation-harness, keyed by artifact
# metric. Each maps to (task, result key) with the score reported as percent.
BENCHMARK_TASKS = {
    "mmlu_pro_pct": ("leaderboard_mmlu_pro", "acc,none"),
    "bbh_pct": ("leaderboard_bbh", "acc_norm,none"),
    "math_lvl5_pct": ("leaderboard_math_hard", "exact_match,none"),
    "gpqa_pct": ("leaderboard_gpqa", "acc_norm,none"),
    "musr_pct": ("leaderboard_musr", "acc_norm,none"),
    "gsm8k_pct": ("gsm8k", "exact_match,strict-match"),
    "hellaswag_pct": ("hellaswag", "acc_norm,none"),
}
METRICS = ARTIFACT_METRICS + OPTIONAL_ARTIFACT_METRICS + DERIVED_METRICS

METRIC_UNITS = dict(METRICS)
METRIC_PREFIX = "training."

# Run Deck tiers. Anything in METRICS that is not listed here lands in "record".
METRIC_TIERS = {
    "throughput": (
        "tokens_per_sec",
        "tokens_per_sec_per_gpu",
        "tflops_per_sec_per_gpu",
        "mfu_pct",
        "scaling_efficiency_pct",
        "tokens_per_kwh",
    ),
    "convergence": (
        "loss_initial",
        "loss_final",
        "loss_curve_decreased",
        "eval_loss",
        "perplexity",
        "convergence_step",
        "convergence_time_seconds",
        *BENCHMARK_TASKS,
    ),
    "stability": (
        "step_time_ms_mean",
        "step_time_ms_p50",
        "step_time_ms_p95",
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
        "peak_memory_used_mb",
    ),
}
_TIERED = {short for names in METRIC_TIERS.values() for short in names}
RECORD_METRICS = tuple(short for short, _unit in METRICS if short not in _TIERED)

# A threshold cell must declare every one of these, so a gate cannot be
# dropped by omission. Specs may still be kind=info until calibrated.
GATED_METRICS = {
    "tokens_per_sec",
    "tokens_per_sec_per_gpu",
    "step_time_ms_p95",
    "peak_memory_allocated_mb",
    "peak_memory_reserved_mb",
    "checkpoint_state_match",
    "checkpoint_loss_delta",
}

RESULTS_COLUMNS = (
    ("Tokens/s", "training.tokens_per_sec"),
    ("Tokens/s/GPU", "training.tokens_per_sec_per_gpu"),
    ("TFLOPS/s/GPU", "training.tflops_per_sec_per_gpu"),
    ("MFU (%)", "training.mfu_pct"),
    ("Mean step (ms)", "training.step_time_ms_mean"),
    ("P50 step (ms)", "training.step_time_ms_p50"),
    ("P95 step (ms)", "training.step_time_ms_p95"),
    ("Peak allocated (MB)", "training.peak_memory_allocated_mb"),
    ("Peak reserved (MB)", "training.peak_memory_reserved_mb"),
    ("Checkpoint save (s)", "training.checkpoint_save_seconds"),
    ("Checkpoint load (s)", "training.checkpoint_load_seconds"),
    ("Checkpoint state match", "training.checkpoint_state_match"),
    ("Checkpoint loss delta", "training.checkpoint_loss_delta"),
    ("Final loss", "training.loss_final"),
    ("Eval loss", "training.eval_loss"),
    ("Perplexity", "training.perplexity"),
    ("Tokens/kWh", "training.tokens_per_kwh"),
    ("Scaling efficiency (%)", "training.scaling_efficiency_pct"),
)


def tier_metric_specs(thresholds_cell, tier):
    """Gating specs for one Run Deck tier, keyed by ``training.<short>``.

    ``kind: info`` specs are record-only and are left out so they cannot force
    a tier to ``na``. Metrics with no spec in this cell are omitted too.
    """
    names = RECORD_METRICS if tier == "record" else METRIC_TIERS.get(tier, ())
    specs = {}
    for short in names:
        full = METRIC_PREFIX + short
        spec = (thresholds_cell or {}).get(full)
        if spec is None or (isinstance(spec, dict) and spec.get("kind") == "info"):
            continue
        specs[full] = spec
    return specs


def percentile(values, pct):
    """Linear-interpolated percentile, matching numpy's default method."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def gradient_accumulation_overhead_pct(baseline_tokens_per_sec, accumulated_tokens_per_sec):
    """Per-token time overhead of a GA>1 sweep against GA=1 at the same micro batch.

    Holding the micro batch fixed keeps activation memory equal, so the only
    difference is how many micro batches share one optimizer step and one
    gradient reduce-scatter. Negative means accumulation amortized that work.
    """
    baseline = float(baseline_tokens_per_sec)
    accumulated = float(accumulated_tokens_per_sec)
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError(f"baseline throughput must be finite and positive, got {baseline_tokens_per_sec!r}")
    if not math.isfinite(accumulated) or accumulated <= 0:
        raise ValueError(f"accumulated throughput must be finite and positive, got {accumulated_tokens_per_sec!r}")
    return (baseline / accumulated - 1.0) * 100.0


def compute_scaling_efficiency(tokens_per_sec_total, num_nodes, baseline_tokens_per_sec_total, baseline_num_nodes=1):
    """Scaling efficiency % = throughput_N / ((N / ref_N) * throughput_ref) * 100.

    Same definition as the JAX MaxText and PyTorch Vision suites. Returns None
    when any input is missing or non-positive, so an uncalibrated baseline is
    record-only rather than a misleading number.
    """
    if not tokens_per_sec_total or not baseline_tokens_per_sec_total:
        return None
    if not num_nodes or not baseline_num_nodes:
        return None
    ideal = (num_nodes / baseline_num_nodes) * baseline_tokens_per_sec_total
    if ideal <= 0:
        return None
    return tokens_per_sec_total / ideal * 100.0


def perplexity(eval_loss):
    """exp(mean token cross-entropy), capped so a diverged run stays finite."""
    loss = float(eval_loss)
    if not math.isfinite(loss):
        raise ValueError(f"eval loss must be finite, got {eval_loss!r}")
    return math.exp(min(loss, 700.0))


def loss_curve_decreased(losses, minimum_points=2):
    """Return whether the least-squares loss slope is finite and negative."""
    if len(losses) < minimum_points:
        return False
    values = [float(value) for value in losses]
    if not all(math.isfinite(value) for value in values):
        return False
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return denominator > 0 and numerator / denominator < 0


def convergence_point(loss_series, target_train_loss=None, evaluations=None, target_eval_loss=None):
    """First (step, elapsed seconds) at which every configured loss target holds.

    An unset target is not a criterion, and a target that is never reached
    yields None, so an uncalibrated target can neither gate nor mislead. With
    both targets set, convergence is the later of the two points.
    """
    points = []
    if target_train_loss is not None:
        hit = next((p for p in loss_series or [] if p.get("loss", math.inf) <= target_train_loss), None)
        if hit is None:
            return None
        points.append((int(hit["step"]), float(hit["time_seconds"])))
    if target_eval_loss is not None:
        hit = next((e for e in evaluations or [] if e.get("eval_loss", math.inf) <= target_eval_loss), None)
        if hit is None:
            return None
        points.append((int(hit["step"]), float(hit["time_seconds"])))
    if not points:
        return None
    return max(points, key=lambda point: point[0])


def milestone_losses(loss_series, milestones):
    """Training loss at each milestone step that the run reached."""
    by_step = {int(p["step"]): float(p["loss"]) for p in loss_series or [] if p.get("loss") is not None}
    return {f"loss_step_{step}": by_step[step] for step in milestones if step in by_step}


def codecarbon_tracking_active(payload, expected_gpus):
    """Validate explicit CodeCarbon/AMDSMI activation evidence."""
    return (
        payload.get("version") == "3.2.4"
        and payload.get("tracker") == "amdsmi"
        and int(payload.get("gpu_count") or 0) >= expected_gpus
        and payload.get("active") is True
    )


def parse_codecarbon_metrics(payload, expected_gpus):
    """Parse CodeCarbon output without treating unavailable tracking as zero usage."""
    if not codecarbon_tracking_active(payload, expected_gpus):
        reason = payload.get("reason") or "AMDSMI tracking is not active"
        raise ValueError(f"CodeCarbon metrics unavailable: {reason}")
    metrics = {"training.codecarbon_tracking_active": 1.0}
    energy_kwh = payload.get("energy_kwh")
    if energy_kwh is not None:
        energy_kwh = float(energy_kwh)
        if not math.isfinite(energy_kwh) or energy_kwh <= 0:
            raise ValueError(f"invalid CodeCarbon energy value: {energy_kwh!r}")
        metrics["training.energy_kwh"] = energy_kwh
    return metrics


def benchmark_scores(lm_eval_results):
    """Map an lm-evaluation-harness ``results`` block onto artifact metrics in percent."""
    scores = {}
    for metric, (task, key) in BENCHMARK_TASKS.items():
        entry = (lm_eval_results or {}).get(task)
        if not isinstance(entry, dict) or entry.get(key) is None:
            continue
        value = float(entry[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"lm-eval {task} {key} is not a fraction in [0, 1]: {entry[key]!r}")
        scores[metric] = value * 100.0
    return scores


def required_metrics_for_run(run_mode, codecarbon_enabled=False, benchmarks=()):
    """Return metrics whose absence is a structural failure for this run."""
    required = set(GATED_METRICS)
    if run_mode != "perf":
        required.update({"eval_loss", "perplexity", "eval_tokens", "eval_completed"})
    if run_mode not in {"smoke", "perf"}:
        required.update({"loss_curve_points", "loss_curve_decreased"})
    if run_mode == "train_5k":
        required.update({"loss_step_100", "loss_step_500", "loss_step_1000", "loss_step_5000"})
    if codecarbon_enabled:
        required.update({"codecarbon_tracking_active", "energy_kwh", "tokens_per_kwh"})
    required.update(benchmarks)
    return required


def to_training_metrics(raw):
    """Validate a rank-zero artifact and return namespaced numeric metrics."""
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("training result artifact is missing a metrics object")

    out = {}
    missing = []
    for name, _unit in ARTIFACT_METRICS:
        value = metrics.get(name)
        if value is None:
            missing.append(name)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"training metric {name!r} is not numeric: {value!r}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"training metric {name!r} is not finite: {value!r}")
        out[f"training.{name}"] = numeric
    if missing:
        raise ValueError(f"training result artifact is missing metrics: {missing}")
    for name, _unit in OPTIONAL_ARTIFACT_METRICS:
        value = metrics.get(name)
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"training metric {name!r} is not finite: {value!r}")
        out[f"training.{name}"] = numeric
    if "eval_completed" in metrics and out["training.eval_completed"] != 1.0:
        raise ValueError("training evaluation did not complete")
    if "eval_tokens" in metrics and out["training.eval_tokens"] <= 0:
        raise ValueError("training evaluation token count must be positive")
    if "codecarbon_tracking_active" in metrics and out["training.codecarbon_tracking_active"] != 1.0:
        raise ValueError("training artifact claims inactive CodeCarbon tracking")
    return out
