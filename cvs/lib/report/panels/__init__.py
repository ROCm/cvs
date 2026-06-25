'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Optional report panels — composable sections independent of the base suite report.
'''

from cvs.lib.report.panels.scaling import build_scaling_panel
from cvs.lib.report.panels.training_parity import build_training_parity_panel

__all__ = ["build_scaling_panel", "build_training_parity_panel"]
