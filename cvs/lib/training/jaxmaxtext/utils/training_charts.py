'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

TensorBoard-derived training chart PNGs for the JAX MaxText Run Deck.

Consumes the ``{tag: [(step, value), ...]}`` mapping from
``tb_events.read_scalars`` and renders per-run charts: multi-loss overlay,
grad/param norms, LR schedule, step-time distribution, and MFU. Like
``loss_curve.py`` these do file I/O and lazily import matplotlib (headless Agg);
a missing dependency or absent tag degrades to ``None`` and never raises.
'''

from __future__ import annotations

import re

from cvs.lib import globals
from cvs.lib.training.jaxmaxtext.utils.gpu_peak_tflops import compute_mfu

log = globals.log

_STEP_TIME_TAG = "perf/step_time_seconds"
_TFLOPS_RATE_TAG = "perf/per_device_tflops_per_sec"
# Leading steps are compile/rampup outliers, excluded from steady-state stats.
_RAMPUP_STEPS = 2


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display on the CVS host
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:  # noqa: BLE001 - plotting must never break the run
        log.warning("training charts: matplotlib unavailable, skipping (%s)", e)
        return None


def _save(fig, plt, out_path, what):
    try:
        out_path = str(out_path)
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        log.info("training charts: wrote %s (%s)", out_path, what)
        return out_path
    except Exception as e:  # noqa: BLE001
        log.warning("training charts: failed to write %s (%s)", what, e)
        plt.close(fig)
        return None


def _series(scalars, tag):
    return [(s, v) for s, v in (scalars.get(tag) or []) if v is not None]


def _safe_tag_name(tag):
    """Filesystem-safe token from a scalar tag (``learning/grad_norm`` -> ``learning_grad_norm``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_") or "tag"


def render_scalar_charts(scalars, out_dir, filename_stem="tb"):
    """Render ONE line chart per scalar tag (TensorBoard-style: step vs value).

    Auto-discovers every tag present in ``scalars`` (``learning/*``, ``perf/*``,
    ...), so new metrics are charted with no code change. Returns
    ``[(tag, path), ...]`` sorted by tag (groups ``learning/*`` before ``perf/*``);
    empty series are skipped. Never raises.
    """
    tags = [t for t in sorted(scalars or {}) if _series(scalars, t)]
    if not tags:
        return []
    plt = _matplotlib()
    if plt is None:
        return []
    rendered = []
    for tag in tags:
        pts = _series(scalars, tag)
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.plot([s for s, _ in pts], [v for _, v in pts], color="#ff7f0e", linewidth=1.3)
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=7)
        path = _save(fig, plt, f"{out_dir}/{filename_stem}_{_safe_tag_name(tag)}.png", tag)
        if path:
            rendered.append((tag, path))
    return rendered


def _step_time_series(scalars, step_metrics):
    """Prefer the TB step-time tag; fall back to parsed step_metrics seconds."""
    tb = _series(scalars, _STEP_TIME_TAG)
    if tb:
        return tb
    return [
        (s.get("step"), s.get("step_time_seconds"))
        for s in (step_metrics or [])
        if s.get("step") is not None and s.get("step_time_seconds") is not None
    ]


def render_step_time_png(scalars, out_path, step_metrics=None, title=None):
    """Steady-state step-time histogram with a compile/rampup callout."""
    series = _step_time_series(scalars, step_metrics)
    if len(series) < 2:
        return None
    plt = _matplotlib()
    if plt is None:
        return None
    ordered = sorted(series, key=lambda sv: sv[0])
    rampup = ordered[:_RAMPUP_STEPS]
    steady = [v for _s, v in ordered[_RAMPUP_STEPS:]] or [v for _s, v in ordered]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(steady, bins=min(30, max(5, len(steady))), color="#1f77b4", alpha=0.85)
    ax.set_xlabel("step time (s)")
    ax.set_ylabel("count")
    ax.set_title(title or "Step-Time Distribution (steady state)")
    ax.grid(True, linestyle="--", alpha=0.4)

    steady_sorted = sorted(steady)
    p50 = steady_sorted[len(steady_sorted) // 2]
    p95 = steady_sorted[min(len(steady_sorted) - 1, int(round(0.95 * (len(steady_sorted) - 1))))]
    note = f"steady p50={p50:.2f}s  p95={p95:.2f}s"
    if rampup:
        note += f"\ncompile/rampup max={max(v for _s, v in rampup):.1f}s (steps {rampup[0][0]}\u2013{rampup[-1][0]})"
    ax.text(
        0.98,
        0.95,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )
    return _save(fig, plt, out_path, "step-time distribution")


def render_cross_sweep_loss_png(loss_by_label, out_path, title=None):
    """Overlay training loss vs step for multiple sweeps (one line per sweep).

    ``loss_by_label``: ``{sweep_label: [(step, loss), ...]}``. Returns ``None``
    unless at least two sweeps have points (the per-sweep curve already covers one).
    """
    series = {lbl: pts for lbl, pts in (loss_by_label or {}).items() if pts}
    if len(series) < 2:
        return None
    plt = _matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, pts in sorted(series.items()):
        ax.plot([s for s, _ in pts], [v for _, v in pts], linewidth=1.4, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title(title or "Loss vs Step (all sweeps)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, plt, out_path, f"{len(series)} sweep loss curves")


def render_cross_sweep_bar_png(value_by_label, out_path, ylabel, title=None):
    """Bar chart of one metric across sweeps (``{sweep_label: value}``)."""
    values = {lbl: v for lbl, v in (value_by_label or {}).items() if v is not None}
    if len(values) < 2:
        return None
    plt = _matplotlib()
    if plt is None:
        return None
    labels = sorted(values)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(labels)), [values[k] for k in labels], color="#1f77b4", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    return _save(fig, plt, out_path, f"{len(labels)}-sweep bar ({ylabel})")


def render_mfu_png(scalars, out_path, peak_tflops_per_gpu, title=None):
    """MFU% vs step from per_device_tflops_per_sec and the configured peak."""
    rate = _series(scalars, _TFLOPS_RATE_TAG)
    if not rate or not peak_tflops_per_gpu:
        return None
    plt = _matplotlib()
    if plt is None:
        return None
    steps = [s for s, _ in rate]
    mfu = [compute_mfu(v, peak_tflops_per_gpu) for _, v in rate]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, mfu, color="#d62728", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("MFU (%)")
    ax.set_title(title or "Model FLOPs Utilization")
    ax.grid(True, linestyle="--", alpha=0.4)
    return _save(fig, plt, out_path, "mfu")
