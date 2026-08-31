"""
Parsers module - Layer 2: Data Abstraction & Validation.

Parsers are responsible for:
- Transforming raw benchmark outputs into structured data
- Validating results against Pydantic schemas
- Aggregating metrics across runs/ranks

Configuration file schemas live under ``cvs/schema/`` (mirroring ``cvs/input/``); use
``cvs.schema.validate.validate_config_file`` to validate configs before running benchmarks.

Parsers should NOT:
- Execute benchmarks
- Deploy infrastructure
- Make pass/fail decisions (validation only)
"""

from cvs.parsers.schemas import (
    AortaBenchmarkResult,
    AortaTraceMetrics,
    ParseResult,
    ParseStatus,
)

from cvs.parsers.aorta_report import AortaReportParser
from cvs.parsers.tracelens import TraceLensParser

__all__ = [
    "AortaTraceMetrics",
    "AortaBenchmarkResult",
    "ParseResult",
    "ParseStatus",
    "AortaReportParser",
    "TraceLensParser",
]
