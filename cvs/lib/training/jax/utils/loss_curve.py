'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Loss-curve PNG rendering for the JAX MaxText suite (row 32).

Kept separate from the pure log parser (`maxtext_parsing.py`) because it does
file I/O and lazily imports matplotlib. matplotlib is imported inside the
function with the headless ``Agg`` backend so that importing this module never
hard-requires the dependency, and a missing/broken matplotlib degrades to
``None`` rather than failing the run -- the loss-curve verdict is computed
independently of the plot.
'''

from __future__ import annotations

from cvs.lib import globals

log = globals.log


def render_loss_curve_png(points, out_path, title=None):
    """Render a training loss curve to a PNG file.

    Args:
        points: ordered list of ``(step, loss)`` tuples (from
            ``maxtext_parsing.sample_loss_curve``).
        out_path: destination PNG path (str or Path).
        title: optional plot title.

    Returns:
        The ``out_path`` (as str) on success, or ``None`` if there is nothing to
        plot or matplotlib is unavailable / rendering failed. Never raises.
    """
    if not points:
        log.info("loss curve: no points to plot, skipping PNG")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display needed on the CVS host
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001 - plotting must never break the run
        log.warning("loss curve: matplotlib unavailable, skipping PNG (%s)", e)
        return None

    try:
        steps = [p[0] for p in points]
        losses = [p[1] for p in points]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, color="#1f77b4")
        ax.set_xlabel("step")
        ax.set_ylabel("training loss")
        ax.set_title(title or "Training Loss Curve")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        out_path = str(out_path)
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        log.info("loss curve: wrote PNG %s (%d points)", out_path, len(points))
        return out_path
    except Exception as e:  # noqa: BLE001
        log.warning("loss curve: failed to render PNG (%s)", e)
        return None
