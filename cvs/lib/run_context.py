"""RunContext: state passed through Job phases.

Lightweight, mutable dataclass. Adapters write to `result.scalars` in `parse`,
to `scratch` for ad-hoc bookkeeping, append to `containers` via _register.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    scalars: dict[str, float] = field(default_factory=dict)


@dataclass
class _NullEvents:
    """No-op events sink. v1 doesn't ship structured events; adapters may emit()."""
    def emit(self, *args: Any, **kwargs: Any) -> None:
        return None
    def close(self) -> None:
        return None


@dataclass
class RunContext:
    run_id: str
    arch: str
    cluster: dict                       # raw cluster file (dict)
    workload: dict                      # raw resolved workload (dict, post-substitution)
    thresholds: dict                    # raw thresholds (dict)
    workload_name: str                  # "vllm/qwen3_next_80b"
    workload_hash: str
    bindings: dict[str, list[str]]      # role -> [host]
    executor: Any                       # MultiHostExecutor or LocalExecutor
    artifacts_dir: Path                 # {artifacts_dir}/{run_id}/

    containers: list = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, str] = field(default_factory=dict)
    result: RunResult = field(default_factory=RunResult)
    events: Any = field(default_factory=_NullEvents)

    @property
    def framework(self) -> str:
        return self.workload["framework"]
