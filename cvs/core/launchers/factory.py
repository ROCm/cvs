"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .mpi import MpiLauncher

if TYPE_CHECKING:
    from .base import WorkloadLauncher


def build_launchers(launchers_cfg: dict) -> dict[str, "WorkloadLauncher"]:
    """Build {name: WorkloadLauncher} from a parsed cluster.json launchers block.

    Today the only launcher is mpi. Future: torchrun, srun, ray. Adding one =
    one new class + one branch here.

    Unknown launcher names raise OrchestratorConfigError.
    """
    from cvs.core.errors import OrchestratorConfigError

    launchers: dict[str, "WorkloadLauncher"] = {}
    for name, config in (launchers_cfg or {}).items():
        if name == "mpi":
            launchers[name] = MpiLauncher.parse_config(config)
        else:
            raise OrchestratorConfigError(f"launcher '{name}' is not implemented")
    return launchers
