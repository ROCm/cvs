'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Pure parsers for MaxText training log output.

MaxText logs per-step metrics in a comma-separated format:
  completed step: 50, seconds: 1.234, TFLOP/s/device: 185.4, Tokens/s/device: 3456.7, total_weights: 7e9, loss: 6.543

During rampup phases, TFLOP/s/device and Tokens/s/device may be omitted.
Profiler steps log a different line ("completed profiler activation/deactivation step").

The parser extracts per-step metrics from the full training log, then computes
aggregate metrics (averages over last N steps, final loss, loss decrease check).
'''

from __future__ import annotations

import re

# (short_name, unit) -- the display surface for training metrics.
# Values are looked up as "training.<short_name>" in the results dict.
TRAINING_METRICS = [
    ("tflops_per_sec_per_gpu", "TFLOP/s/GPU"),
    ("tokens_per_sec_per_gpu", "tok/s/GPU"),
    ("tokens_per_sec_total", "tok/s total"),
    ("scaling_efficiency_pct", "%"),
    ("step_time_seconds", "s/step"),
    ("step_time_mean_ms", "ms/step"),
    ("step_time_p50_ms", "ms/step"),
    ("step_time_p95_ms", "ms/step"),
    ("final_loss", "loss"),
    ("loss_decreased", "bool"),
    ("eval_loss", "loss"),
    ("steps_to_target", "steps"),
    ("time_to_target_seconds", "s"),
]
TRAINING_METRIC_UNITS = dict(TRAINING_METRICS)

# The perf SLO contract: the subset of TRAINING_METRICS a calibrated run must
# assert. Membership = "out of range means FAILURE". Record-only by default:
# a NEW metric is record-only until its name is added here.
GATED_METRICS = {
    "tflops_per_sec_per_gpu",
    "tokens_per_sec_per_gpu",
}

# Regex for a completed training step line.
# Example: "completed step: 50, seconds: 1.234, TFLOP/s/device: 185.4, Tokens/s/device: 3456.7, ..., loss: 6.543"
_STEP_RE = re.compile(r"completed step:\s*(\d+)")
_METRIC_RE = re.compile(r"(\S+?):\s*([\d.eE+\-]+)")

# Eval (validation) lines. MaxText emits an eval summary when eval_interval > 0,
# but the exact wording is image-dependent and none of the current logs ran with
# eval enabled -- so we match defensively: a line that mentions "eval" and carries
# an eval-loss-like token. Both a step index and the loss are optional per line.
# NOTE: confirm this against a real eval-enabled run and tighten if needed.
_EVAL_LINE_RE = re.compile(r"\beval", re.I)
_EVAL_STEP_RE = re.compile(r"step:?\s*(\d+)", re.I)
_EVAL_LOSS_RE = re.compile(r"eval[_ ]?loss[:=]?\s*([\d.eE+\-]+)", re.I)
# Fallback: a bare "loss: X" on an eval line when the token is not prefixed with "eval".
_LOSS_RE = re.compile(r"\bloss[:=]?\s*([\d.eE+\-]+)", re.I)
# Config-dump lines ("Config param target_eval_loss: 0.0") mention eval + loss but
# are not eval results -- exclude them so they never register as eval points.
_CONFIG_LINE_RE = re.compile(r"config param|pyconfig", re.I)


def compute_scaling_efficiency(
    tokens_per_sec_total,
    num_nodes,
    baseline_tokens_per_sec_total,
    baseline_num_nodes=1,
):
    """Scaling efficiency % for a training run.

    efficiency % = throughput_N / ((N / ref_N) * throughput_ref) * 100

    where throughput_N is this run's total tokens/sec on `num_nodes` nodes and
    throughput_ref is the reference (typically 1-node) total tokens/sec measured
    on `baseline_num_nodes` nodes. 100% means perfectly linear scaling; lower
    means communication/straggler overhead is eating into the added nodes.

    Returns None (record-only) when any input is missing or non-positive so an
    uncalibrated baseline never produces a misleading number or a crash.
    """
    if not tokens_per_sec_total or not baseline_tokens_per_sec_total:
        return None
    if not num_nodes or not baseline_num_nodes:
        return None
    ideal = (num_nodes / baseline_num_nodes) * baseline_tokens_per_sec_total
    if ideal <= 0:
        return None
    return tokens_per_sec_total / ideal * 100.0


def compute_convergence(step_metrics, eval_metrics, target_metric="auto", target_value=0.0):
    """Steps and wall-clock to reach a target loss (row 33).

    target_metric:
      - "eval_loss": converge on validation loss (eval_metrics points)
      - "train_loss": converge on per-step training loss (step_metrics)
      - "auto": use eval_metrics when present, else training loss

    A `target_value <= 0` disables the metric and returns (None, None) so an
    uncalibrated target never gates or misleads.

    Returns (steps_to_target, time_to_target_seconds), where the time is the
    cumulative sum of per-step `seconds` up to and including the target step.
    This is training compute time (it includes the step-0 compile spike and
    excludes eval/checkpoint overhead), not true wall-clock. Returns (None, None)
    when disabled or the target is never reached. Never raises.
    """
    if not target_value or target_value <= 0:
        return (None, None)

    use_eval = target_metric == "eval_loss" or (target_metric == "auto" and bool(eval_metrics))

    # Cumulative training seconds indexed by step, from the per-step lines.
    cum = {}
    running = 0.0
    for s in step_metrics or []:
        sec = s.get("seconds")
        if isinstance(sec, (int, float)):
            running += sec
        step = s.get("step")
        if step is not None:
            cum[step] = running

    target_step = None
    if use_eval:
        for e in eval_metrics or []:
            loss = e.get("eval_loss")
            if loss is not None and e.get("step") is not None and loss <= target_value:
                target_step = e.get("step")
                break
    else:
        for s in step_metrics or []:
            loss = s.get("loss")
            if loss is not None and s.get("step") is not None and loss <= target_value:
                target_step = s.get("step")
                break

    if target_step is None:
        return (None, None)

    time_to_target = cum.get(target_step)
    if time_to_target is None and cum:
        # An eval step may not line up with a training-step key; take the
        # cumulative time at the latest training step at or before the target.
        prior = [t for st, t in cum.items() if st <= target_step]
        time_to_target = max(prior) if prior else None

    return (target_step, time_to_target)


def sample_loss_curve(step_metrics, sample_every=10, milestone_steps=None):
    """Downsample per-step training loss for the loss curve (row 32).

    Keeps a point when its step is a multiple of `sample_every`, is one of the
    `milestone_steps` (e.g. 100/500/1k/5k), or is the first/last recorded step.
    The first/last inclusion keeps short runs (fewer than `sample_every` steps)
    from producing an empty curve.

    Returns an ordered, de-duplicated list of ``(step, loss)`` tuples. Only steps
    that carry a numeric `loss` are considered. Never raises.
    """
    milestones = set(milestone_steps or [])
    every = sample_every if sample_every and sample_every > 0 else 1

    loss_steps = [
        s for s in (step_metrics or []) if s.get("step") is not None and isinstance(s.get("loss"), (int, float))
    ]
    if not loss_steps:
        return []

    first_step = loss_steps[0]["step"]
    last_step = loss_steps[-1]["step"]

    picked = {}
    for s in loss_steps:
        step = s["step"]
        if step % every == 0 or step in milestones or step in (first_step, last_step):
            picked[step] = s["loss"]

    return [(step, picked[step]) for step in sorted(picked)]


def evaluate_loss_decreasing(points, max_slope=0.0):
    """Decide whether a sampled loss curve trends downward (row 32).

    Fits a least-squares line to ``points`` (a list of ``(step, loss)``) and
    treats the run as decreasing when the slope is below `max_slope` (default
    0.0, i.e. strictly negative). Uses a dependency-free closed form:

        slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2)

    Returns ``(decreasing: bool, slope: float, detail: str)`` or ``None`` when
    there are fewer than 2 points (verdict not computable). Never raises; a
    degenerate x-spread (all steps equal) also returns None.
    """
    if not points or len(points) < 2:
        return None

    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)

    denom = n * sxx - sx * sx
    if denom == 0:
        return None

    slope = (n * sxy - sx * sy) / denom
    decreasing = slope < max_slope
    detail = (
        f"loss slope {slope:.6g}/step over {n} points "
        f"(first={points[0][1]:.4f}@{points[0][0]}, last={points[-1][1]:.4f}@{points[-1][0]}); "
        f"{'decreasing' if decreasing else 'NOT decreasing'} (max_slope={max_slope})"
    )
    return (decreasing, slope, detail)


def _percentile(values, q):
    """Linear-interpolated percentile (q in [0, 100]) over a list of numbers.

    Returns None for an empty list. Matches numpy's default ('linear')
    interpolation so p50 equals the median for even-length samples.
    """
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (q / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _parse_step_line(line):
    """Parse a single 'completed step: N, ...' line into a dict.

    Returns None for non-step lines (profiler steps, rampup, etc.).
    """
    step_m = _STEP_RE.search(line)
    if not step_m:
        return None
    if "profiler" in line.lower():
        return None
    step = int(step_m.group(1))
    fields = {"step": step}
    # Parse all key: value pairs from the comma-separated line.
    # The regex grabs "key: numeric_value" pairs.
    for m in _METRIC_RE.finditer(line):
        key, val_str = m.group(1), m.group(2)
        try:
            val = float(val_str)
        except ValueError:
            continue
        if key == "step":
            continue
        fields[key] = val
    return fields


def extract_step_metrics(log_text):
    """Extract per-step metric dicts from a MaxText training log.

    Returns a list of dicts, each with at least 'step' and optionally:
    'seconds', 'TFLOP/s/device', 'Tokens/s/device', 'loss', 'total_weights'.
    """
    steps = []
    for line in log_text.splitlines():
        parsed = _parse_step_line(line)
        if parsed is not None:
            steps.append(parsed)
    return steps


def extract_eval_metrics(log_text):
    """Extract validation-loss points from a MaxText training log (row 34).

    Returns a list of ``{"step": int|None, "eval_loss": float}`` dicts, one per
    eval summary line. Defensive by design: MaxText only emits eval output when
    ``eval_interval > 0`` and the exact wording is image-dependent, so we accept
    any non-config line that mentions "eval" and carries a loss token. Config
    dumps (e.g. "Config param target_eval_loss: 0.0") are excluded. Returns
    ``[]`` when eval was not enabled or the format is unrecognized.

    NOTE: validate the matched format against a real eval-enabled run and
    tighten the regex if MaxText's eval line differs from what is assumed here.
    """
    evals = []
    for line in log_text.splitlines():
        if not _EVAL_LINE_RE.search(line):
            continue
        if _CONFIG_LINE_RE.search(line):
            continue
        m = _EVAL_LOSS_RE.search(line) or _LOSS_RE.search(line)
        if not m:
            continue
        try:
            loss = float(m.group(1))
        except ValueError:
            continue
        step_m = _EVAL_STEP_RE.search(line)
        step = int(step_m.group(1)) if step_m else None
        evals.append({"step": step, "eval_loss": loss})
    return evals


def parse_training_log(log_text, num_gpus, avg_last_n=10):
    """Parse MaxText training log into namespaced training.* metrics dict.

    Averages TFLOP/s/device and Tokens/s/device over the last `avg_last_n`
    steps (matching the MAD benchmark parser behavior). Computes total
    tokens/sec, final loss, and whether loss decreased from first to last step.

    Returns: {"training.<metric>": value, ...}
    """
    steps = extract_step_metrics(log_text)
    if not steps:
        return {
            "training.tflops_per_sec_per_gpu": None,
            "training.tokens_per_sec_per_gpu": None,
            "training.tokens_per_sec_total": None,
            "training.step_time_seconds": None,
            "training.step_time_mean_ms": None,
            "training.step_time_p50_ms": None,
            "training.step_time_p95_ms": None,
            "training.final_loss": None,
            "training.loss_decreased": None,
            "training.eval_loss": None,
        }

    # Filter to steps that have perf metrics (skip rampup steps without them).
    perf_steps = [s for s in steps if "TFLOP/s/device" in s or "Tokens/s/device" in s]
    tail = perf_steps[-avg_last_n:] if perf_steps else []

    def _avg(key):
        vals = [s[key] for s in tail if key in s]
        return sum(vals) / len(vals) if vals else None

    tflops = _avg("TFLOP/s/device")
    tokens_per_gpu = _avg("Tokens/s/device")
    step_time = _avg("seconds")

    tokens_total = tokens_per_gpu * num_gpus if tokens_per_gpu is not None else None

    # Step-time distribution (ms) over steady-state steps. perf_steps already
    # excludes rampup/profiler steps, whose compile-heavy outliers would inflate
    # the tail and mask real jitter. Percentiles use the full steady-state
    # window (not just `tail`) so p95 has enough samples to be meaningful.
    step_seconds = [s["seconds"] for s in perf_steps if "seconds" in s]
    step_time_mean_ms = (sum(step_seconds) / len(step_seconds) * 1000.0) if step_seconds else None
    p50 = _percentile(step_seconds, 50)
    p95 = _percentile(step_seconds, 95)
    step_time_p50_ms = p50 * 1000.0 if p50 is not None else None
    step_time_p95_ms = p95 * 1000.0 if p95 is not None else None

    # Loss metrics from all steps that have a loss value.
    loss_steps = [s for s in steps if "loss" in s]
    final_loss = loss_steps[-1]["loss"] if loss_steps else None
    first_loss = loss_steps[0]["loss"] if loss_steps else None
    loss_decreased = None
    if first_loss is not None and final_loss is not None:
        loss_decreased = 1 if final_loss < first_loss else 0

    # Validation loss (row 34): last eval point, or None when eval was disabled.
    eval_metrics = extract_eval_metrics(log_text)
    eval_loss = eval_metrics[-1]["eval_loss"] if eval_metrics else None

    return {
        "training.tflops_per_sec_per_gpu": tflops,
        "training.tokens_per_sec_per_gpu": tokens_per_gpu,
        "training.tokens_per_sec_total": tokens_total,
        "training.step_time_seconds": step_time,
        "training.step_time_mean_ms": step_time_mean_ms,
        "training.step_time_p50_ms": step_time_p50_ms,
        "training.step_time_p95_ms": step_time_p95_ms,
        "training.final_loss": final_loss,
        "training.loss_decreased": loss_decreased,
        "training.eval_loss": eval_loss,
    }
