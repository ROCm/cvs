"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Legacy entry points -- removed once cvs/core/orchestrators/ is deleted.
# Imported under aliases so the new OrchestratorConfig dataclass wins
# `from cvs.core import OrchestratorConfig` for new code.
from cvs.core.orchestrators.factory import (
    OrchestratorConfig as _LegacyOrchestratorConfig,
    OrchestratorFactory,
)

# New canonical entry points.
from cvs.core.config import OrchestratorConfig, load_config
from cvs.core.errors import OrchestratorConfigError
from cvs.core.factory import create_orchestrator
from cvs.core.orchestrator import Orchestrator
from cvs.core.scope import ExecResult, ExecScope, ExecTarget

__all__ = [
    # Legacy (kept temporarily so the four un-migrated RCCL test files still
    # import OrchestratorFactory until Phase E rewrites them).
    "OrchestratorFactory",
    # New canonical
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorConfigError",
    "create_orchestrator",
    "load_config",
    "ExecScope",
    "ExecTarget",
    "ExecResult",
]
