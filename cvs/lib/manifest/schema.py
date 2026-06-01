"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from cvs.lib.config.thresholds import ThresholdVerdict

OverallStatus = Literal["complete", "failed", "skipped", "error"]


class Identity(BaseModel):
    """Identity and provenance: everything needed to attribute and reproduce a run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    test_id: str
    cell_id: str
    config_hash: Optional[str] = None
    # Recorded from day one so v2.A reuse-manifests is a pure tack-on (no migration).
    workload_hash: Optional[str] = None
    verification_hash: Optional[str] = None
    cvs_version: Optional[str] = None
    cvs_git_sha: Optional[str] = None
    framework_image_digest: Optional[str] = None
    framework_versions: Dict[str, str] = Field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    invoker: Optional[str] = None


class HostFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hostname: str
    cpu: Optional[str] = None
    memory: Optional[str] = None
    kernel: Optional[str] = None
    os: Optional[str] = None
    gpus: List[str] = Field(default_factory=list)
    nics: List[str] = Field(default_factory=list)
    container_runtime: Optional[str] = None


class SystemFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hosts: List[HostFingerprint] = Field(default_factory=list)
    topology_hash: Optional[str] = None


class ConfigInputs(BaseModel):
    """Resolved config provenance.

    ``commands`` and ``env`` are recorded **as-is** (verbatim). Security/redaction
    was removed entirely (W7 / addendum B7, C4): there are no redaction fields
    here. Target clusters are closed/internal, so recorded commands and env --
    including any inline token -- are stored in plaintext.
    """

    model_config = ConfigDict(extra="forbid")

    resolved_config_path: Optional[str] = None
    datasets: List[Dict[str, str]] = Field(default_factory=list)
    model: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)
    commands: List[str] = Field(default_factory=list)
    seed: int = 0


class PhaseTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str
    start: Optional[str] = None
    end: Optional[str] = None
    duration_s: Optional[float] = None
    status: str = "complete"


class PatternMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    severity: str
    line: str
    node: Optional[str] = None
    source: Optional[str] = None


class Verdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_status: OverallStatus = "complete"
    failure_category: Optional[str] = None
    skip_reason: Optional[str] = None
    threshold_verdicts: List[ThresholdVerdict] = Field(default_factory=list)
    pattern_matches: List[PatternMatch] = Field(default_factory=list)
    scalars: Dict[str, float] = Field(default_factory=dict)
    flags: Dict[str, str] = Field(default_factory=dict)


class ResourceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_host: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    oom: bool = False


class SidecarPointers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: Optional[str] = None
    trajectory: Optional[str] = None
    events: Optional[str] = None
    logs_dir: Optional[str] = None
    dmesg: Optional[str] = None
    gpu_state: Optional[str] = None


class Manifest(BaseModel):
    """Small (5-50 KB), cat-able, jq-friendly per-run index.

    Bulky numeric arrays live in the Parquet sidecars; this file holds metadata,
    verdicts, scalars, and pointers. Its ``schema_version`` is versioned
    independently of the config schema.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    identity: Identity
    system: SystemFingerprint = Field(default_factory=SystemFingerprint)
    config: ConfigInputs
    phases: List[PhaseTiming] = Field(default_factory=list)
    verdicts: Verdicts = Field(default_factory=Verdicts)
    resources: ResourceSummary = Field(default_factory=ResourceSummary)
    sidecars: SidecarPointers = Field(default_factory=SidecarPointers)

    def write(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, default=str))
        return p

    @classmethod
    def read(cls, path) -> "Manifest":
        return cls.model_validate(json.loads(Path(path).read_text()))
