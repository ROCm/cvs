"""
Parsers module - Layer 2: Data Abstraction & Validation.

Parsers are responsible for:
- Transforming raw benchmark outputs into structured data
- Validating results against Pydantic schemas
- Aggregating metrics across runs/ranks
- Validating configuration files (fail fast)

Parsers should NOT:
- Execute benchmarks
- Deploy infrastructure
- Make pass/fail decisions (validation only)
"""

from cvs.parsers.schemas import (
    # Result schemas
    AortaTraceMetrics,
    AortaBenchmarkResult,
    ParseResult,
    ParseStatus,
    # Config file schemas
    ClusterConfigFile,
    ClusterNodeConfig,
    AortaBenchmarkConfigFile,
    AortaDockerConfigFile,
    AortaRcclConfigFile,
    AortaEnvironmentConfigFile,
    AortaExpectedResultsConfigFile,
    AortaAnalysisConfigFile,
    # Validation helper
    validate_config_file,
)

# Parser implementations
from cvs.parsers.aorta_report import AortaReportParser
from cvs.parsers.tracelens import TraceLensParser

__all__ = [
    # Result schemas
    "AortaTraceMetrics",
    "AortaBenchmarkResult",
    "ParseResult",
    "ParseStatus",
    # Config file schemas
    "ClusterConfigFile",
    "ClusterNodeConfig",
    "AortaBenchmarkConfigFile",
    "AortaDockerConfigFile",
    "AortaRcclConfigFile",
    "AortaEnvironmentConfigFile",
    "AortaExpectedResultsConfigFile",
    "AortaAnalysisConfigFile",
    # Validation helper
    "validate_config_file",
    # Parser implementations
    "AortaReportParser",
    "TraceLensParser",
]
