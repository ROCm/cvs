"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContainerRuntime(BaseModel):
    """Runtime block for the cluster's CVS-side container orchestrator.

    Separate from a workload's ``container`` block: that is per-test plumbing
    for the workload image; this is the *cluster's* CVS-side container
    (typically ``rocm/cvs:latest``) launched once per node and reused across
    tests. List values (volumes/devices/cap_add) append to backend defaults;
    scalars (network/ipc/privileged) override.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "docker"  # 'docker' or 'enroot'
    args: Dict[str, object] = Field(default_factory=dict)


class ClusterContainerOrchestrator(BaseModel):
    """Optional cluster-level CVS-side container orchestrator.

    Pre-DTNI ``cluster_container.json`` describes a CVS sidecar container --
    distinct from the workload container an adapter launches. Schema lives
    here so the field is round-trip-checked; consumers (the
    ContainerOrchestrator in cvs/core) are out of scope for this PR.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    launch: bool = True
    image: str = "rocm/cvs:latest"
    name: str = "cvs_container"
    runtime: ContainerRuntime = Field(default_factory=ContainerRuntime)


class Node(BaseModel):
    """A single physical node in the cluster pool.

    Pre-DTNI shape: just the addressing tuple. Hardware properties (GPU
    count) are *discovered at run time* via rocm-smi in the adapter's
    ``prepare`` phase, not declared statically -- the cluster file is
    authoritative for "what hosts I can reach + how to log in", nothing more.
    Container devices, NCCL fabric, weka paths, and the role topology all
    live on the workload YAML where the test author already maintains them.

    extra=forbid is load-bearing so typos in a cluster file fail fast at
    load_cluster_file rather than silently dropping fields.
    """

    model_config = ConfigDict(extra="forbid")

    vpc_ip: str
    bmc_ip: Optional[str] = None


class ClusterPool(BaseModel):
    """Cluster pool: "what hosts I can reach + how to log in", and nothing more.

    Pool-level credentials (``username``, ``priv_key_file``) replace the
    per-Node fields the G4 design carried; a single SSH key per cluster is
    the pre-DTNI convention and matches every site we run on. ``env_vars``
    is the pre-launch shell-export bag every command on every node receives
    (PATH overrides, ROCm/UCX flags, etc.). ``head_node`` MUST appear as a
    key in ``nodes`` and names the host that orchestrates distributed
    bootstraps. ``container`` is the optional CVS-side container
    orchestrator (pre-DTNI cluster_container.json shape).
    """

    model_config = ConfigDict(extra="forbid")

    username: str
    priv_key_file: str
    env_vars: Dict[str, str] = Field(default_factory=dict)
    head_node: Optional[str] = None
    # No ordering claim: the binder iterates lexicographically over keys.
    nodes: Dict[str, Node]
    container: Optional[ClusterContainerOrchestrator] = None

    @model_validator(mode="after")
    def _head_node_in_nodes(self) -> "ClusterPool":
        if self.head_node is not None and self.head_node not in self.nodes:
            raise ValueError(f"head_node {self.head_node!r} is not a key in nodes (known keys: {sorted(self.nodes)})")
        return self


def load_cluster_file(path: Union[str, Path]) -> ClusterPool:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    elif p.suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"unsupported cluster file extension {p.suffix!r} (use .yaml or .json)")
    return ClusterPool.model_validate(raw)
