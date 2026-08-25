'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Primus-TorchTitan training job orchestration library (future extensibility).

PLACEHOLDER: This module provides a factory pattern for future Primus-TorchTitan
integration. Currently, PrimusTorchTitanTrainingJob is a pass-through to the base
TorchTitanTrainingJob class since no Primus-TorchTitan integration exists yet.

When Primus-TorchTitan integration is developed, this class can be extended to:
- Use Primus CLI instead of direct torchrun
- Handle PRIMUS_WORKSPACE environment variables
- Parse Primus-specific log formats
- Support Primus checkpoint management
'''

from __future__ import annotations

import re

from cvs.lib.training.torchtitan.torchtitan_lib import TorchTitanTrainingJob


class PrimusTorchTitanTrainingJob(TorchTitanTrainingJob):
    """
    Primus-wrapped TorchTitan training job (placeholder for future integration).

    PLACEHOLDER: Currently this is a direct pass-through to TorchTitanTrainingJob.
    When Primus-TorchTitan integration is available, this class will override
    methods to use Primus-specific execution patterns.

    Future overrides may include:
    - build_training_job_cmd() — use Primus CLI instead of torchrun
    - _read_last_node_log() — handle Primus log directory structure
    - _parse_step_losses() — adapt for Primus log format if different
    - checkpoint paths — use PRIMUS_WORKSPACE for checkpoint storage
    """

    def __init__(self, *args, **kwargs):
        """Initialize Primus-TorchTitan job (currently delegates to base class)."""
        super().__init__(*args, **kwargs)
        # Future: Set Primus-specific attributes here
        # self.primus_workspace = ...
        # self.primus_team = ...
        # self.primus_user = ...


def _parse_step_losses(log_text):
    """Parse step-to-loss mapping from Primus-TorchTitan training log.

    PLACEHOLDER: Currently uses TorchTitan pattern. When Primus-TorchTitan
    integration exists, this may need to parse different log formats.

    Args:
        log_text (str): Full training log text.

    Returns:
        dict: {step: loss} mapping for all logged steps.
    """
    losses = {}
    # TorchTitan pattern: step: N | loss: X.XX
    # Future: May need Primus-specific pattern
    pattern = re.compile(r'step:\s+(\d+)[^\n]*?loss:\s+([0-9.eE+\-]+)', re.I)
    for m in pattern.finditer(log_text):
        step = int(m.group(1))
        loss = float(m.group(2))
        losses[step] = loss
    return losses
