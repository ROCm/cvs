"""
SGLang single-node inference variant config schema.

Mirrors ``cvs/input/config_file/inference/sglang/``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from pydantic import Field, model_validator
from typing_extensions import Literal

from cvs.schema.common.base import BaseVariantConfig
from cvs.schema.base import _Forbid


class SglangRoleServer(_Forbid):
    env: Dict[str, str] = Field(default_factory=dict)
    serve_port: str = "8000"


class SglangRoles(_Forbid):
    server: SglangRoleServer = Field(default_factory=SglangRoleServer)


def perf_cell_key(bp_dict: Mapping[str, Any]) -> str:
    bench = (bp_dict.get("inference_tests") or {}).get("bench_serv_random") or {}
    return (
        f"ISL={bench.get('input_length', '-')},"
        f"OSL={bench.get('output_length', '-')},"
        f"TP={bp_dict.get('tensor_parallelism', '8')},"
        f"PP={bp_dict.get('pipeline_parallelism', '1')},"
        f"CONC={bp_dict.get('max_concurrency', '-')}"
    )


class SglangSingleVariantConfig(BaseVariantConfig):
    """Typed config for ``sglang_single`` + ContainerOrchestrator."""

    framework: Literal["sglang_single"]
    gpu_arch: str
    variant_key: str = ""
    config_path: str = ""
    inference: Dict[str, Any] = Field(default_factory=dict)
    benchmark_params: Dict[str, Any] = Field(default_factory=dict)
    roles: SglangRoles = Field(default_factory=SglangRoles)

    def cell_key(self, isl, osl, concurrency) -> str:
        tp = self.benchmark_params.get("tensor_parallelism", "-")
        pp = self.benchmark_params.get("pipeline_parallelism", "-")
        return f"ISL={isl},OSL={osl},TP={tp},PP={pp},CONC={concurrency}"

    def perf_cell_key(self) -> str:
        return perf_cell_key(self.benchmark_params)

    @property
    def hf_token_file(self) -> str:
        return self.paths.hf_token_file

    @model_validator(mode="after")
    def _sync_legacy_inference_container_name(self):
        if self.inference and self.container.name:
            self.inference["container_name"] = self.container.name
            self.inference["container_image"] = self.container.image
        return self
