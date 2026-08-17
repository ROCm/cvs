'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Loss curve PNG rendering for TorchTitan training suites.
'''

from __future__ import annotations

from cvs.lib import globals

log = globals.log


def render_loss_curve_png(points, out_path, title=None):
    """Render a training loss curve to a PNG file.

    Args:
        points:   Ordered list of ``(step, loss)`` tuples from
                  ``loss_curve.sample_loss_curve``.
        out_path: Destination PNG path (str or Path).
        title:    Optional plot title.

    Returns:
        ``out_path`` as str on success, or ``None`` if there is nothing to plot
        or matplotlib is unavailable / rendering failed. Never raises.
    """
    if not points:
        log.info("loss curve: no points to plot, skipping PNG")
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log.warning("loss curve: matplotlib unavailable, skipping PNG (%s)", e)
        return None

    try:
        steps = [p[0] for p in points]
        losses = [p[1] for p in points]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, color="#1f77b4")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title or "Training Loss Curve")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        out_path = str(out_path)
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        log.info("loss curve: wrote PNG %s (%d points)", out_path, len(points))
        return out_path
    except Exception as e:
        log.warning("loss curve: failed to render PNG (%s)", e)
        return None
