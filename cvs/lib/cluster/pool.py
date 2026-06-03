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


class NodeDevices(BaseModel):
    """Site/hardware device tokens (e.g. /dev/infiniband/uverbs*) the
    container needs ``--device`` passthrough for.

    Device enumeration is site-dependent (a Thor2 cluster exposes
    ``/dev/infiniband/uverbs{0..7}`` + ``/dev/infiniband/rdma_cm``; an AINIC
    cluster looks different). Putting this on the cluster file rather than
    per-workload means the same vLLM/megatron/sglang config moves between
    clusters without an edit.

    ``ib`` and ``gpu`` are typed lists of docker --device tokens. ``extra``
    is the escape hatch for site-specific devices we haven't categorized.
    G5b merges all three into ``ContainerSpec.devices`` at launch time.
    """

    model_config = ConfigDict(extra="forbid")

    ib: List[str] = Field(default_factory=list)
    gpu: List[str] = Field(default_factory=list)
    extra: List[str] = Field(default_factory=list)


class NodeNetwork(BaseModel):
    """Site/fabric-determined NCCL/UCX/Gloo collective env.

    These knobs depend on the fabric (mlx5_* vs bnxt_re* vs rdma*), the GID
    table (RoCEv2 per site), the QoS class, and the host NIC names. They are
    NOT workload-dependent and should not be carried in per-workload
    ``container.env`` (which leaks site bleed across configs and breaks
    cross-site reuse).

    Six typed Optional[str] knobs covering the universal NCCL/UCX/Gloo
    site-env surface every multi-node framework hits (megatron, sglang-
    disagg, jax, pytorch-xdit). Site-only vars not in this list (NCCL_IB_TC,
    NCCL_NET_GDR_LEVEL, OFI provider, etc.) ride in ``extra_env`` until a
    second site asks for typing.

    G5b merges ``{typed fields as NCCL_*/UCX_*/GLOO_* env, then extra_env}``
    into ``ContainerSpec.env`` before applying the per-workload override --
    workload env wins on conflict (so per-run debug knobs still work).

    extra=forbid catches typos in the typed fields (``nccl_ib_gid_indx`` ->
    load-time error). Typos inside ``extra_env`` cannot be caught; that is
    the escape hatch's cost.
    """

    model_config = ConfigDict(extra="forbid")

    nccl_socket_ifname: Optional[str] = None  # -> NCCL_SOCKET_IFNAME
    nccl_ib_hca: Optional[str] = None  # -> NCCL_IB_HCA
    nccl_ib_gid_index: Optional[str] = None  # -> NCCL_IB_GID_INDEX
    nccl_ib_sl: Optional[str] = None  # -> NCCL_IB_SL
    ucx_net_devices: Optional[str] = None  # -> UCX_NET_DEVICES
    gloo_socket_ifname: Optional[str] = None  # -> GLOO_SOCKET_IFNAME
    extra_env: Dict[str, str] = Field(default_factory=dict)


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
    devices: NodeDevices = Field(default_factory=NodeDevices)
    network: NodeNetwork = Field(default_factory=NodeNetwork)


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
