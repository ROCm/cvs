'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Wide expert-parallel mode of the distributed SGLang controller.

Selected when the config sets ``server_params.enable_wideEP`` to true. The launch is the
unified multi-node ``sglang.launch_server`` of ``SglangDistributed`` with attention
data-parallel (``--dp-size`` with ``--enable-dp-attention``) and experts sharded across every
rank (``--ep-size``). Pipeline parallel stays 1 and ``expert_parallelism`` must equal
``tensor_parallelism``. The all-to-all backend comes from ``add_flags`` (shipped templates use
``--moe-a2a-backend mori``). The ROCm SGLang images reject ``--attn-dp-size``; newer SGLang
rewrites this older pair to that flag.
'''

from cvs.lib import globals
from cvs.lib.inference.sglang.sglang_common import (
    validate_wide_ep_layout,
    wide_ep_enabled,
    wide_ep_preflight_cmd,
    wide_ep_preflight_failures,
    wide_ep_uses_mori,
)
from cvs.lib.inference.sglang.sglang_distributed_lib import SglangDistributed

log = globals.log

WIDE_EP_PREFLIGHT_TIMEOUT_SEC = 300


def distributed_controller_class(params):
    """``SglangWideEP`` when ``enable_wideEP`` is true in ``params``, else ``SglangDistributed``."""
    return SglangWideEP if wide_ep_enabled(params) else SglangDistributed


class SglangWideEP(SglangDistributed):
    """Distributed SGLang serve + benchmark with expert parallel and data-parallel attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_wide_ep_layout(self.bp_dict, nnodes=self.nnodes)

    def _apply_bp_defaults(self):
        super()._apply_bp_defaults()
        self.bp_dict.setdefault('deepep_mode', 'normal')

    def _server_env_body(self):
        return (
            f"{super()._server_env_body()}"
            f"export EP={self.bp_dict['expert_parallelism']}\n"
            f"export ATTN_DP={self.bp_dict['attn_data_parallelism']}\n"
        )

    def _layout_flags_block(self):
        return (
            f"    --ep-size {self.bp_dict['expert_parallelism']} \\\n"
            f"    --dp-size {self.bp_dict['attn_data_parallelism']} \\\n"
            f"    --enable-dp-attention \\\n"
            f"    --deepep-mode {self.bp_dict['deepep_mode']} \\\n"
        )

    def wide_ep_preflight_failures(self):
        """Probe every server node's container before launch and return what would stop the server.

        Checks GPU count per node, MoRI importability when ``add_flags`` selects it as the
        all-to-all backend, and the model directory. An empty list means the launch can start.
        """
        check_mori = wide_ep_uses_mori(self.bp_dict)
        if not check_mori:
            log.warning('add_flags does not select --moe-a2a-backend mori; skipping the MoRI availability check')
        out_dict = self._container_exec(
            wide_ep_preflight_cmd(self.bp_dict['model'], check_mori),
            timeout=WIDE_EP_PREFLIGHT_TIMEOUT_SEC,
        )
        failures = [
            f'{node}: no preflight output from the orchestrator'
            for node in self.server_node_list
            if node not in out_dict
        ]
        failures += wide_ep_preflight_failures(
            out_dict,
            tp=self.bp_dict['tensor_parallelism'],
            nnodes=self.nnodes,
            model=self.bp_dict['model'],
            check_mori=check_mori,
        )
        return failures

    def sglang_distributed_gpu_counts(self, mem_threshold_mb=5000):
        result = super().sglang_distributed_gpu_counts(mem_threshold_mb=mem_threshold_mb)
        result['configured_ep'] = int(self.bp_dict['expert_parallelism'])
        result['configured_attn_dp'] = int(self.bp_dict['attn_data_parallelism'])
        log.info('Configured EP: %s, attention DP: %s', result['configured_ep'], result['configured_attn_dp'])
        return result
