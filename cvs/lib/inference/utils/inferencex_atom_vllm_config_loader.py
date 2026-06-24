'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceX vLLM parity suite config schema (``framework: inferencex_atom_vllm_single``).

Shares the ATOM variant shape but requires ``params.driver=vllm`` (ROCm vLLM serve +
``vllm bench serve``). IX ``atom_args`` from recipes are not applied.
'''

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from pydantic import model_validator
from typing_extensions import Literal

from cvs.lib.inference.utils.inferencex_atom_config_loader import (
    InferenceXAtomParams,
    InferenceXAtomRoleServer,
    InferenceXAtomRunCard,
    placeholder_gated_threshold_cell,
)
from cvs.lib.inference.utils.inferencing_config_loader import Sweep
from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS
from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config


class InferenceXAtomVllmRoles(_Forbid):
    server: InferenceXAtomRoleServer = InferenceXAtomRoleServer()


class InferenceXAtomVllmVariantConfig(BaseVariantConfig):
    framework: Literal["inferencex_atom_vllm_single"]
    gpu_arch: str
    ix_recipe_id: Optional[str] = None
    run_card: InferenceXAtomRunCard = InferenceXAtomRunCard()
    roles: InferenceXAtomVllmRoles = InferenceXAtomVllmRoles()
    params: InferenceXAtomParams
    sweep: Sweep

    def cell_key(self, isl, osl, concurrency):
        return f"ISL={isl},OSL={osl},TP={self.params.tensor_parallelism},CONC={concurrency}"

    def expected_cells(self) -> List[str]:
        by_name = {c.name: c for c in self.sweep.sequence_combinations}
        return [
            self.cell_key(by_name[r.combo].isl, by_name[r.combo].osl, r.concurrency)
            for r in self.sweep.runs
        ]

    @model_validator(mode="after")
    def _check_vllm_driver_and_thresholds(self):
        if self.params.driver != "vllm":
            raise ValueError(
                f"inferencex_atom_vllm_single requires params.driver='vllm', got {self.params.driver!r}"
            )
        expected = set(self.expected_cells())
        present = set(self.thresholds.keys())
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        problems = []
        if missing:
            problems.append(f"sweep cells with no threshold entry: {missing}")
        if extra:
            problems.append(f"threshold keys matching no sweep cell (typo?): {extra}")
        gated_keys = [f"client.{m}" for m in sorted(GATED_METRICS)]
        gated_gaps = {}
        for cell in sorted(expected & present):
            specs = self.thresholds.get(cell) or {}
            absent = [k for k in gated_keys if k not in specs]
            if absent:
                gated_gaps[cell] = absent
        if gated_gaps:
            problems.append(f"cells missing gated-metric specs: {gated_gaps}")
        if problems:
            msg = "threshold.json does not match the sweep matrix; " + "; ".join(problems)
            if self.enforce_thresholds:
                raise ValueError(msg)
            warnings.warn(f"{msg} (enforce_thresholds=false -> record-only)", stacklevel=2)
        return self


def load_variant(config_path, cluster_dict) -> InferenceXAtomVllmVariantConfig:
    """Load vLLM parity variant config + sibling ``*threshold.json``."""
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return InferenceXAtomVllmVariantConfig(**raw)


def orchestrator_container_from_variant(variant: InferenceXAtomVllmVariantConfig) -> Dict[str, Any]:
    block = variant.container.model_dump()
    server_env = variant.roles.server.env
    if server_env:
        block = {**block, "env": dict(server_env)}
    return block


__all__ = [
    "InferenceXAtomVllmVariantConfig",
    "load_variant",
    "orchestrator_container_from_variant",
    "placeholder_gated_threshold_cell",
]
