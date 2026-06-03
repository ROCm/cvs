"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class NodePaths(BaseModel):
    """Site host-paths exposed by a node, addressable by A3 ContainerSpec.

    The cluster file is where *site* knowledge lives (e.g. the local HF model
    cache, dataset roots), not the per-framework config. ContainerSpec mounts
    these by name. extra=forbid is load-bearing so a typo (``model-cache``)
    fails at load instead of silently resolving to None at launch time. All
    fields are typed Optional[str] -- G2's str-contract discipline applied at
    the cluster layer.
    """

    model_config = ConfigDict(extra="forbid")

    model_cache: Optional[str] = None
    dataset_root: Optional[str] = None


class Node(BaseModel):
    """A single physical node in the cluster pool.

    Hardware-only: no role assignment lives here. Roles are matched to nodes by
    the binder at run time using ``labels``. extra=forbid is load-bearing so
    typos in a cluster file fail fast at load_cluster_file rather than silently
    dropping fields.
    """

    model_config = ConfigDict(extra="forbid")

    ip: str
    user: str
    ssh_key: str = "id_rsa"
    gpus: int = Field(default=0, ge=0)
    labels: List[str] = Field(default_factory=list)
    paths: NodePaths = Field(default_factory=NodePaths)


class ClusterPool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # No ordering claim here: the binder is deterministic content-based
    # (lexicographic hostname iteration), not insertion-order. Two cluster
    # files with the same nodes in different key order yield identical
    # BindResults -- this is what makes v2.A reuse-manifests sound.
    nodes: Dict[str, Node]


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
