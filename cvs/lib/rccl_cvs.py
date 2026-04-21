"""
RCCL CVS runner used by the rccl_cvs pytest suite.

`cvs.lib.rccl_cvs` remains the stable import path for the suite and tests, while
the implementation is split across smaller internal modules under `cvs.lib.rccl`.
"""

from cvs.lib.rccl.artifacts import (
    _build_summary,
    _resolved_case_payload,
)
from cvs.lib.rccl.case_ids import _ensure_unique_case_id, _no_matrix_case_id, _slug
from cvs.lib.rccl.config import RcclConfig, load_rccl_config
from cvs.lib.rccl.matrix_expand import (
    RcclMatrixExpansionInput,
    RcclResolvedCaseSpec,
    expand_rccl_matrix_cases,
    expand_rccl_no_matrix_cases,
    expansion_input_from_rccl_config,
)
from cvs.lib.rccl.launcher import _scan_rccl_stdout, build_collective_command
from cvs.lib.rccl.runner import run_rccl
from cvs.lib.rccl.validator import _matching_rows, parse_and_validate_results

__all__ = [
    "RcclConfig",
    "RcclMatrixExpansionInput",
    "RcclResolvedCaseSpec",
    "_build_summary",
    "_ensure_unique_case_id",
    "_matching_rows",
    "_no_matrix_case_id",
    "_resolved_case_payload",
    "_scan_rccl_stdout",
    "_slug",
    "build_collective_command",
    "expand_rccl_matrix_cases",
    "expand_rccl_no_matrix_cases",
    "expansion_input_from_rccl_config",
    "load_rccl_config",
    "parse_and_validate_results",
    "run_rccl",
]
