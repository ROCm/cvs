"""Framework registry. Maps framework name -> adapter class."""

from __future__ import annotations

from cvs.lib.dtni.frameworks.vllm_single_adapter import VllmAdapter
from cvs.lib.dtni.base_adapter import BaseWorkloadAdapter

FRAMEWORK_REGISTRY: dict[str, type[BaseWorkloadAdapter]] = {
    "vllm_single": VllmAdapter,
}
