'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Loss-curve PNG rendering for the JAX MaxText suite (row 32).

The renderer now lives in ``cvs.lib.utils.loss_curve`` so the PyTorch Vision
training suite produces identical loss-curve artifacts. This module re-exports
it so existing JAX MaxText imports keep working.
'''

from cvs.lib.utils.loss_curve import render_loss_curve_png

__all__ = ["render_loss_curve_png"]
