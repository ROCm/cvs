"""
Pydantic models for the strict RCCL CVS input file (nested rccl.run / validation / artifacts).

Runtime paths (rccl-tests build dir, ROCm, MPI, RCCL libs) are not configured in JSON: ``rccl.run.env_script``
must source a site script that exports ``RCCL_TESTS_BUILD_DIR``, ``ROCM_HOME``, ``MPI_HOME``, ``RCCL_HOME``, etc.

Placeholders must be resolved on the raw dict before model_validate (see load_rccl_config in rccl_cvs).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator


def format_rccl_config_validation_error(err: ValidationError) -> str:
    """Single readable line for callers that raise ValueError (e.g. unit tests)."""
    parts: list[str] = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e.get("loc", ()) if x != "rccl")
        msg = e.get("msg", "validation error")
        if loc:
            parts.append(f"{loc}: {msg}")
        else:
            parts.append(msg)
    return "; ".join(parts) if parts else str(err)


class RcclRunInput(BaseModel):
    """rccl.run — benchmark intent plus required ``env_script`` (site paths via env vars only)."""

    model_config = ConfigDict(extra="forbid")

    env_script: str
    num_ranks: int = Field(gt=0)
    ranks_per_node: int = Field(gt=0)
    collectives: list[str] = Field(min_length=1)
    datatype: str
    start_size: str
    end_size: str
    step_factor: str
    warmups: str
    iterations: str
    cycles: str

    @field_validator("env_script", mode="before")
    @classmethod
    def _env_script_required_non_empty(cls, v: Any) -> str:
        if v is None:
            raise ValueError("must not be null")
        s = str(v).strip()
        if not s or s.lower() == "none":
            raise ValueError("must be a non-empty string")
        return s

    @field_validator(
        "datatype",
        "start_size",
        "end_size",
        "step_factor",
        "warmups",
        "iterations",
        "cycles",
        mode="before",
    )
    @classmethod
    def _required_non_empty_str(cls, v: Any) -> str:
        if v is None:
            raise ValueError("must not be null")
        s = str(v).strip()
        if not s:
            raise ValueError("must be a non-empty string")
        return s

    @field_validator("num_ranks", "ranks_per_node", mode="before")
    @classmethod
    def _coerce_positive_int(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("collectives", mode="before")
    @classmethod
    def _normalize_collectives(cls, v: Any) -> list[str]:
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("must be a non-empty array of strings")
        out: list[str] = []
        for item in v:
            s = str(item).strip()
            if not s:
                raise ValueError("entries must be non-empty strings")
            if "/" in s or "\\" in s:
                raise ValueError(f"entries must be basenames only (no path separators): {s!r}")
            out.append(s)
        return out

    @model_validator(mode="after")
    def _ranks_divisible(self) -> RcclRunInput:
        if self.num_ranks % self.ranks_per_node != 0:
            raise ValueError("rccl.run.num_ranks must be divisible by rccl.run.ranks_per_node")
        return self


class RcclInlineCollectiveThresholds(BaseModel):
    """Per-collective inline thresholds: ``bus_bw`` size → number map only."""

    model_config = ConfigDict(extra="forbid")

    bus_bw: dict[str, float]

    @field_validator("bus_bw", mode="before")
    @classmethod
    def _bus_bw_non_empty_numeric(cls, v: Any) -> dict[str, float]:
        if not isinstance(v, dict) or len(v) == 0:
            raise ValueError(
                "bus_bw must be a non-empty object mapping message-size strings to numbers "
                "(rccl-tests JSON size as a decimal string, e.g. '8589934592')"
            )
        out: dict[str, float] = {}
        for raw_k, raw_val in v.items():
            sk = str(raw_k).strip()
            if not sk:
                raise ValueError("bus_bw keys must be non-empty strings")
            if raw_val is None:
                raise ValueError(f"bus_bw[{sk!r}] must be a number")
            try:
                out[sk] = float(raw_val)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bus_bw[{sk!r}] must be a number") from exc
        return out


# Normative thresholds map: collective name -> { bus_bw: { size -> number } }
RcclThresholdsInlineMap = dict[str, RcclInlineCollectiveThresholds]
_rccl_thresholds_inline_adapter = TypeAdapter(RcclThresholdsInlineMap)


def parse_rccl_thresholds_payload(data: dict[str, Any]) -> dict[str, RcclInlineCollectiveThresholds]:
    """Validate inline or file-backed thresholds JSON to the expected ``bus_bw`` shape (no I/O)."""
    return _rccl_thresholds_inline_adapter.validate_python(data)


class RcclValidationInput(BaseModel):
    """rccl.validation — profile and optional threshold sources."""

    model_config = ConfigDict(extra="forbid")

    profile: str
    thresholds: Optional[RcclThresholdsInlineMap] = None
    thresholds_file: Optional[str] = None

    @field_validator("profile", mode="before")
    @classmethod
    def _normalize_profile(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not str(v).strip()):
            raise ValueError("rccl.validation.profile is required")
        return str(v).strip().lower()

    @field_validator("profile")
    @classmethod
    def _profile_literal(cls, v: str) -> str:
        allowed = {"none", "smoke", "thresholds", "strict"}
        if v not in allowed:
            raise ValueError(f"must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("thresholds_file", mode="before")
    @classmethod
    def _optional_thresholds_file(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    def _has_inline_thresholds(self) -> bool:
        return self.thresholds is not None and len(self.thresholds) > 0

    @model_validator(mode="after")
    def _at_most_one_threshold_source(self) -> RcclValidationInput:
        has_t = self._has_inline_thresholds()
        has_f = self.thresholds_file is not None and bool(str(self.thresholds_file).strip())
        if has_t and has_f:
            raise ValueError("rccl.validation: at most one of thresholds and thresholds_file may be set")
        return self

    @model_validator(mode="after")
    def _profile_threshold_source_rules(self) -> RcclValidationInput:
        p = self.profile
        has_t = self._has_inline_thresholds()
        has_f = self.thresholds_file is not None and bool(str(self.thresholds_file).strip())
        if p in ("thresholds", "strict"):
            if not has_t and not has_f:
                raise ValueError(f"rccl.validation: profile {p!r} requires non-empty thresholds or thresholds_file")
        elif has_t or has_f:
            raise ValueError(
                "rccl.validation.thresholds and thresholds_file are only allowed when profile is "
                f"'thresholds' or 'strict' (got {p!r})"
            )
        return self


class RcclArtifactsInput(BaseModel):
    """rccl.artifacts — local output dir, remote work dir, and raw export flag."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str
    remote_work_dir: str
    export_raw: bool

    @field_validator("output_dir", "remote_work_dir", mode="before")
    @classmethod
    def _non_empty_str(cls, v: Any) -> str:
        if v is None:
            raise ValueError("must not be null")
        s = str(v).strip()
        if not s:
            raise ValueError("must be a non-empty string")
        return s


class RcclNestedConfig(BaseModel):
    """Validated ``rccl`` object (after stripping ``_*`` keys): ``run``, ``validation``, ``artifacts``, optional ``matrix``."""

    model_config = ConfigDict(extra="forbid")

    run: RcclRunInput
    validation: RcclValidationInput
    artifacts: RcclArtifactsInput
    matrix: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _inline_threshold_keys_match_collectives(self) -> RcclNestedConfig:
        if not self.validation.thresholds:
            return self
        collective_set = set(self.run.collectives)
        for key in self.validation.thresholds:
            if not key or not str(key).strip():
                raise ValueError("rccl.validation.thresholds: collective keys must be non-empty strings")
            if key not in collective_set:
                raise ValueError(f"rccl.validation: threshold key {key!r} is not listed in rccl.run.collectives")
        return self

    @model_validator(mode="after")
    def _matrix_not_implemented(self) -> RcclNestedConfig:
        if self.matrix is not None and self.matrix != {}:
            raise ValueError("rccl.matrix is not implemented yet; remove it for this CVS build")
        return self


class RcclConfigFileRoot(BaseModel):
    """Full JSON file: exactly ``{ \"rccl\": { ... } }``."""

    model_config = ConfigDict(extra="forbid")

    rccl: RcclNestedConfig
