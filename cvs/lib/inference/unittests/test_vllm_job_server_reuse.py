'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.vllm_job.VllmJob server-command construction:
  - the duplicate --max-model-len fix (config-pin suppresses the derived value)
  - server_signature(), which gates cross-cell server reuse
'''

import unittest
from types import SimpleNamespace

from cvs.lib.inference.vllm_job import VllmJob

_TP = 8
_PP = 2
_NNODES = 2


class FakeOrch:
    hosts = ["10.0.0.1", "10.0.0.2"]

    def exec(self, *a, **k):
        return {}

    def exec_on_head(self, *a, **k):
        return {}


def _variant(serve_args=None):
    params = SimpleNamespace(
        tensor_parallelism=str(_TP),
        pipeline_parallel_size=str(_PP),
        master_addr="10.0.0.1",
        master_port="29501",
        nnodes=str(_NNODES),
        port_no="8000",
        random_range_ratio="0.8",
        random_prefix_len="0",
        burstiness="1.0",
        seed="0",
        request_rate="inf",
        tokenizer_mode="auto",
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="50,90,95,99",
        base_url="http://0.0.0.0",
        dataset_name="random",
        backend="vllm",
    )
    return SimpleNamespace(
        params=params,
        model=SimpleNamespace(id="/models/Kimi-K2.5-W4A8"),
        paths=SimpleNamespace(log_dir="/logs", models_dir="/models"),
        roles=SimpleNamespace(
            server=SimpleNamespace(
                serve_args=dict(serve_args or {}),
                env={"VLLM_ROCM_USE_AITER": "1"},
                ib_netdev="enp159s0np0",
            )
        ),
    )


def _job(isl, osl, conc, serve_args=None):
    return VllmJob(
        orch=FakeOrch(),
        variant=_variant(serve_args),
        hf_token="tok",
        isl=isl,
        osl=osl,
        concurrency=conc,
        num_prompts="640",
    )


class TestMaxModelLenNoDuplicate(unittest.TestCase):
    def test_config_pin_wins_and_no_duplicate(self):
        argv = _job("1024", "1024", 16, serve_args={"max-model-len": "16384"})._server_argv(0)
        idxs = [i for i, a in enumerate(argv) if a == "--max-model-len"]
        self.assertEqual(len(idxs), 1, "config-pinned max-model-len must appear exactly once")
        self.assertEqual(argv[idxs[0] + 1], "16384", "config value must win")

    def test_derived_emitted_when_not_pinned(self):
        argv = _job("1024", "1024", 16, serve_args={})._server_argv(0)
        idxs = [i for i, a in enumerate(argv) if a == "--max-model-len"]
        self.assertEqual(len(idxs), 1, "derived max-model-len must still be emitted when unpinned")
        # 1024+1024 worst-case derived value, definitely not the 16384 config value
        self.assertNotEqual(argv[idxs[0] + 1], "16384")


class TestServerSignatureReuse(unittest.TestCase):
    def test_invariant_to_concurrency(self):
        # Pinned max-model-len: cells differing only in concurrency share a server.
        sa = {"max-model-len": "16384"}
        self.assertEqual(
            _job("1024", "1024", 4, sa).server_signature(),
            _job("1024", "1024", 64, sa).server_signature(),
        )

    def test_pinned_mml_shares_across_isl_osl(self):
        # With a fixed max-model-len, ISL/OSL never reach the server argv, so all
        # cells legitimately share one server (ISL/OSL are client-only knobs).
        sa = {"max-model-len": "16384"}
        self.assertEqual(
            _job("1024", "1024", 16, sa).server_signature(),
            _job("8192", "1024", 16, sa).server_signature(),
        )

    def test_derived_mml_distinguishes_osl(self):
        # Without a pin, max-model-len is derived per (isl+osl); different OSL must
        # change the signature so a real restart happens.
        self.assertNotEqual(
            _job("1024", "1024", 16, serve_args={}).server_signature(),
            _job("1024", "8192", 16, serve_args={}).server_signature(),
        )

    def test_signature_strips_node_rank_and_is_hashable(self):
        job = _job("1024", "1024", 16, serve_args={"max-model-len": "16384"})
        self.assertIn("--node-rank", job._server_argv(0))
        sig = job.server_signature()
        self.assertNotIn("--node-rank", sig[0])
        # hashable + stable
        self.assertEqual(hash(sig), hash(job.server_signature()))


if __name__ == "__main__":
    unittest.main()
