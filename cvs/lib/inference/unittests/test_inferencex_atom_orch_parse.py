'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for InferenceXAtomJob.parse_results (stock ``results`` artifact -> client.*).
No hardware: a fake orch returns committed fixture text.
'''

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.inference.inferencex_atom_orch import InferenceXAtomJob

_HERE = Path(__file__).parent
_FIXTURES = _HERE / "fixtures"
_ISL = 7168
_OSL = 1024
_TP = 8


class FakeOrch:
    def __init__(self, exec_return=None):
        self.exec_return = exec_return if exec_return is not None else {}

    def exec(self, cmd, **kwargs):
        return self.exec_return


def _fake_variant():
    params = SimpleNamespace(
        tensor_parallelism=str(_TP),
        port_no="8000",
        random_range_ratio="0.8",
        random_prefix_len="0",
        burstiness="1.0",
        seed="0",
        request_rate="inf",
        tokenizer_mode="auto",
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="99",
        base_url="http://0.0.0.0",
        dataset_name="random",
        backend="vllm",
        max_model_length="8192",
    )
    roles = SimpleNamespace(server=SimpleNamespace(serve_args={}, env={}))
    paths = SimpleNamespace(log_dir="/LOGS", models_dir="/models")
    model = SimpleNamespace(id="openai/gpt-oss-120b")
    return SimpleNamespace(params=params, roles=roles, paths=paths, model=model)


class TestInferenceXAtomOrchParse(unittest.TestCase):
    def test_parse_results_maps_client_metrics(self):
        raw = json.loads((_FIXTURES / "vllm_results_sample.json").read_text())
        job = InferenceXAtomJob(
            orch=FakeOrch(exec_return={"node0": json.dumps(raw)}),
            variant=_fake_variant(),
            hf_token="tok",
            isl=_ISL,
            osl=_OSL,
            concurrency=64,
            num_prompts=100,
        )
        out = job.parse_results()
        metrics = out["node0"]
        self.assertIn("client.output_throughput", metrics)
        self.assertIn("client.mean_ttft_ms", metrics)


if __name__ == "__main__":
    unittest.main()
