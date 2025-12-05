"""
Runners module - Layer 1: Test Execution Wrappers.

Runners are responsible for:
- Deploying test environments (containers, etc.)
- Executing benchmarks
- Collecting raw artifacts (logs, trace files)

Runners should NOT:
- Parse results into structured data
- Validate performance thresholds
- Make pass/fail decisions
"""

from runners._base_runner import BaseRunner, RunResult, RunConfig, RunStatus

__all__ = [
    "BaseRunner",
    "RunResult", 
    "RunConfig",
    "RunStatus",
]

