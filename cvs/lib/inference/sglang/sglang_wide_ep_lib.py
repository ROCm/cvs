'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Wide expert-parallel SGLang inference controller.

One unified ``sglang.launch_server`` group on the first ``nnodes`` hosts.
Attention is data-parallel (``--attn-dp-size``) and experts are sharded across
every rank (``--ep-size``). Pipeline parallel stays 1. MORI all-to-all requires
``expert_parallelism == tensor_parallelism``.
'''

import time

from cvs.lib import globals
from cvs.lib.inference.sglang.sglang_common import (
    add_cli_flags_block,
    collect_sglang_gpu_topology,
    format_sglang_gpu_topology_lines,
    stage_and_start_launch_script,
)
from cvs.lib.inference.sglang.sglang_distributed_lib import SglangDistributed

log = globals.log


class SglangWideEP(SglangDistributed):
    """Wide-EP SGLang serve + benchmark. Host placement matches ``SglangDistributed``."""

    suite_name = "sglang_wide_ep"

    def __init__(
        self,
        model_name,
        inference_config_dict,
        benchmark_params_dict,
        orch=None,
        gpu_type='mi325',
        user_name=None,
        priv_key_file=None,
    ):
        super().__init__(
            model_name,
            inference_config_dict,
            benchmark_params_dict,
            orch=orch,
            gpu_type=gpu_type,
            user_name=user_name,
            priv_key_file=priv_key_file,
        )
        self._validate_wide_ep_parallelism()

    def _apply_bp_defaults(self):
        super()._apply_bp_defaults()
        self.bp_dict.setdefault('pipeline_parallelism', '1')
        self.bp_dict.setdefault('moe_a2a_backend', 'mori')
        self.bp_dict.setdefault('deepep_mode', 'normal')

    def _require_parallel_int(self, key):
        raw = self.bp_dict.get(key)
        if raw is None or str(raw).strip() == '':
            raise ValueError(f"sglang_wide_ep requires {key}")
        return int(raw)

    def _validate_wide_ep_parallelism(self):
        """MORI all-to-all only runs when the expert group equals the TP world.

        Attention DP equals that same group so each rank keeps its own KV cache
        and pipeline parallel stays unused.
        """
        tp = self._require_parallel_int('tensor_parallelism')
        pp = self._require_parallel_int('pipeline_parallelism')
        ep = self._require_parallel_int('expert_parallelism')
        attn_dp = self._require_parallel_int('attn_data_parallelism')
        if pp != 1:
            raise ValueError(f"sglang_wide_ep requires pipeline_parallelism 1, got {pp}")
        if ep != tp:
            raise ValueError(
                "sglang_wide_ep requires expert_parallelism == tensor_parallelism "
                f"(MORI all-to-all), got expert_parallelism={ep} tensor_parallelism={tp}"
            )
        if attn_dp != ep:
            raise ValueError(
                "sglang_wide_ep requires attn_data_parallelism == expert_parallelism "
                f"so attention stays data-parallel, got attn_data_parallelism={attn_dp} expert_parallelism={ep}"
            )
        if not str(self.bp_dict.get('moe_a2a_backend') or '').strip():
            raise ValueError("sglang_wide_ep requires moe_a2a_backend")

    def _server_env_body(self):
        body = super()._server_env_body()
        return (
            f"{body}"
            f"export EP={self.bp_dict['expert_parallelism']}\n"
            f"export ATTN_DP={self.bp_dict['attn_data_parallelism']}\n"
            f"export MOE_A2A_BACKEND={self.bp_dict['moe_a2a_backend']}\n"
        )

    def _configured_gpu_count(self):
        return int(self.bp_dict['expert_parallelism'])

    def launch_server(self, dtype='auto', kv_cache_dtype='auto'):
        """Launch one wide-EP ``sglang.launch_server`` group (no PD disagg)."""
        log.info(
            'Launch wide-EP SGLang on %d nodes (rank0=%s:%s ep=%s attn_dp=%s backend=%s)',
            self.nnodes,
            self.rank0_node,
            self.router_serv_port,
            self.bp_dict['expert_parallelism'],
            self.bp_dict['attn_data_parallelism'],
            self.bp_dict['moe_a2a_backend'],
        )
        flags_block = add_cli_flags_block(self.bp_dict, indent='    ')
        for i, node in enumerate(self.server_node_list):
            host_flag = '0.0.0.0' if i == 0 else node
            launch_body = (
                f"export NNODES={self.nnodes}\n"
                f"export NODE_RANK={i}\n"
                f"python3 -m sglang.launch_server --model {self.bp_dict['model']} \\\n"
                f"    --host {host_flag} \\\n"
                f"    --port {self.router_serv_port} \\\n"
                f"    --dtype {dtype} \\\n"
                f"    --kv-cache-dtype {kv_cache_dtype} \\\n"
                f"    --trust-remote-code \\\n"
                f"    --tp-size {self.bp_dict['tensor_parallelism']} \\\n"
                f"    --pp-size {self.bp_dict['pipeline_parallelism']} \\\n"
                f"    --ep-size {self.bp_dict['expert_parallelism']} \\\n"
                f"    --attn-dp-size {self.bp_dict['attn_data_parallelism']} \\\n"
                f"    --moe-a2a-backend {self.bp_dict['moe_a2a_backend']} \\\n"
                f"    --deepep-mode {self.bp_dict['deepep_mode']} \\\n"
                f"    --nnodes {self.nnodes} \\\n"
                f"    --node-rank {i} \\\n"
                f"    --dist-init-addr {self.dist_init_addr} \\\n"
                f"    --disable-radix-cache --disable-cuda-graph \\\n"
                f"    --mem-fraction-static {self.bp_dict['memory_fraction']} \\\n"
                f"{flags_block}\n"
                f"    --log-level {self.inf_dict['log_level']}\n"
            )
            stage_and_start_launch_script(
                self._container_exec,
                node,
                '/tmp/server_launch_script.sh',
                launch_body,
                '/tmp/server_env_script.sh',
                self.server_log_path(i),
            )
        time.sleep(5)

    def sglang_wide_ep_gpu_counts(self, mem_threshold_mb=5000):
        tp = int(self.bp_dict['tensor_parallelism'])
        pp = int(self.bp_dict['pipeline_parallelism'])
        ep = int(self.bp_dict['expert_parallelism'])
        attn_dp = int(self.bp_dict['attn_data_parallelism'])

        topo = collect_sglang_gpu_topology(
            self._host_exec,
            {"server": self.server_node_list},
            mem_threshold_mb=mem_threshold_mb,
        )
        server = topo["groups"]["server"]
        result = {
            "configured_tp": tp,
            "configured_pp": pp,
            "configured_ep": ep,
            "configured_attn_dp": attn_dp,
            "configured_nnodes": self.nnodes,
            "moe_a2a_backend": self.bp_dict['moe_a2a_backend'],
            "server_per_node": server["per_node"],
            "total_occupied_gpus": topo["total_occupied_gpus"],
        }
        log.info(
            "\n".join(
                format_sglang_gpu_topology_lines(
                    configured_tp=tp,
                    configured_pp=pp,
                    configured_ep=ep,
                    configured_attn_dp=attn_dp,
                    configured_nnodes=self.nnodes,
                    groups={"Server nodes": server},
                )
            )
        )
        return result
