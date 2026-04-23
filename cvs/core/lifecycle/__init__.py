"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from cvs.core.lifecycle.base import Phase, PhaseError, Severity
from cvs.core.lifecycle.phases.multinode_ssh import MultinodeSshPhase
from cvs.core.lifecycle.pipeline import Pipeline

# Today's prepare pipeline is one element. The follow-up host-prep PR adds
# the rest of atnair/docker-pipeline's helpers (driver_recovery, exclusivity,
# host_sanitize, noise_floor, arch_detect) here as additional Phase classes.
PREPARE_PIPELINE = Pipeline("prepare", [MultinodeSshPhase()])

__all__ = [
    "Phase",
    "PhaseError",
    "Pipeline",
    "Severity",
    "MultinodeSshPhase",
    "PREPARE_PIPELINE",
]
