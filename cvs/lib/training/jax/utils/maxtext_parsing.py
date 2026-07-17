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
    ("step_time_seconds", "s/step"),
    ("final_loss", "loss"),
    ("loss_decreased", "bool"),
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
            "training.final_loss": None,
            "training.loss_decreased": None,
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

    # Loss metrics from all steps that have a loss value.
    loss_steps = [s for s in steps if "loss" in s]
    final_loss = loss_steps[-1]["loss"] if loss_steps else None
    first_loss = loss_steps[0]["loss"] if loss_steps else None
    loss_decreased = None
    if first_loss is not None and final_loss is not None:
        loss_decreased = 1 if final_loss < first_loss else 0

    return {
        "training.tflops_per_sec_per_gpu": tflops,
        "training.tokens_per_sec_per_gpu": tokens_per_gpu,
        "training.tokens_per_sec_total": tokens_total,
        "training.step_time_seconds": step_time,
        "training.final_loss": final_loss,
        "training.loss_decreased": loss_decreased,
    }
