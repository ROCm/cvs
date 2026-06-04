"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class Fabric(BaseModel):
    """Workload-side NCCL/UCX/Gloo fabric knobs (pre-DTNI shape).

    These describe how the workload's collectives reach the wire on a given
    site. They live on the *workload* config (matching the pre-DTNI
    ``mi3xx_megatron_llama_distributed.json`` shape) because in practice
    operators tune them per workload + site, not per cluster: the same NIC
    on the same host can need a different ``NCCL_IB_SL`` for a latency-
    sensitive vs throughput-sensitive workload, and a different
    ``NCCL_IB_HCA`` subset for a multi-tenant vs sole-occupant run.

    The adapter is responsible for folding these into ``ContainerSpec.env``
    at launch (the typed fields are lowered to their corresponding
    ``NCCL_*`` / ``UCX_*`` / ``GLOO_*`` environment variables via
    ``to_env``); workload ``container.env`` wins on conflict, so a per-run
    debug knob still overrides the fabric defaults.

    extra=forbid catches typos in the typed fields (``nccl_ib_gid_indx`` ->
    load-time error). Typos inside ``extra_env`` cannot be caught; that is
    the escape hatch's cost. Use ``extra_env`` for site-specific vars not
    yet typed (``NCCL_IB_TC``, ``NCCL_NET_GDR_LEVEL``, OFI provider, ...).
    """

    model_config = ConfigDict(extra="forbid")

    nic_type: Optional[str] = None  # free-form site tag: ainic|thor2|cx7|...
    nccl_socket_ifname: Optional[str] = None  # -> NCCL_SOCKET_IFNAME
    nccl_ib_hca: Optional[str] = None  # -> NCCL_IB_HCA
    nccl_ib_gid_index: Optional[str] = None  # -> NCCL_IB_GID_INDEX
    nccl_ib_sl: Optional[str] = None  # -> NCCL_IB_SL
    ucx_net_devices: Optional[str] = None  # -> UCX_NET_DEVICES
    gloo_socket_ifname: Optional[str] = None  # -> GLOO_SOCKET_IFNAME
    hca_id_pattern: Optional[str] = None  # selector pattern used by some training scripts
    extra_env: Dict[str, str] = Field(default_factory=dict)

    def to_env(self) -> Dict[str, str]:
        """Lower typed fabric fields + extra_env to a container env dict.

        Typed-field absence -> key omitted (the framework default applies);
        ``extra_env`` is unioned in last so site-specific knobs override
        typed defaults for the same key (rare, but supported -- the typed
        fields are the common case, extra_env is the surgical override).
        ``nic_type`` and ``hca_id_pattern`` are NOT lowered to env: they
        carry workload-classification info the adapter consumes directly.
        """
        env: Dict[str, str] = {}
        if self.nccl_socket_ifname is not None:
            env["NCCL_SOCKET_IFNAME"] = self.nccl_socket_ifname
        if self.nccl_ib_hca is not None:
            env["NCCL_IB_HCA"] = self.nccl_ib_hca
        if self.nccl_ib_gid_index is not None:
            env["NCCL_IB_GID_INDEX"] = self.nccl_ib_gid_index
        if self.nccl_ib_sl is not None:
            env["NCCL_IB_SL"] = self.nccl_ib_sl
        if self.ucx_net_devices is not None:
            env["UCX_NET_DEVICES"] = self.ucx_net_devices
        if self.gloo_socket_ifname is not None:
            env["GLOO_SOCKET_IFNAME"] = self.gloo_socket_ifname
        env.update(self.extra_env)
        return env
