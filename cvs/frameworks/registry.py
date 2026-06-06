"""Framework registry. Maps framework name -> adapter class."""

from __future__ import annotations

from cvs.frameworks.vllm_single.adapter import VllmAdapter
from cvs.lib.base_adapter import BaseWorkloadAdapter

FRAMEWORK_REGISTRY: dict[str, type[BaseWorkloadAdapter]] = {
    "vllm_single": VllmAdapter,
}
