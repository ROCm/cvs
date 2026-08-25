'''Unit tests for atom NIAH job helpers.'''

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from cvs.lib.inference.atom.atom_niah_job import run_niah_cell
from cvs.lib.inference.utils.long_context_accuracy_config import LongContextAccCell
from cvs.lib.utils.model_query_lib import LongContextNiahBenchmark


def _variant():
    params = SimpleNamespace(port_no="8000", base_url="http://0.0.0.0")
    model = SimpleNamespace(id="moonshotai/Kimi-K2.7-Code")
    paths = SimpleNamespace(log_dir="/tmp/logs")
    return SimpleNamespace(params=params, model=model, paths=paths)


def _niah_payload(*, pass_rate: float) -> str:
    total = 4
    correct = round(pass_rate * total)
    payload = {
        "task": "long_ctx_niah",
        "metric_key": "pass_rate",
        "isl": 8192,
        "osl": 32,
        "correct": correct,
        "total": total,
        "pass_rate": pass_rate,
        "results": [],
    }
    return json.dumps(payload)


class TestAtomNiahJob(unittest.TestCase):
    def test_run_niah_cell_sources_server_env_and_uses_local_tokenizer(self):
        orch = MagicMock()
        captured = {}

        def _exec(cmd, timeout):
            captured["cmd"] = cmd
            captured["timeout"] = timeout
            return {"head": _niah_payload(pass_rate=0.5)}

        orch.exec_on_head.side_effect = _exec
        cell = LongContextAccCell(id="niah_8k", isl=8192, osl=32, num_prompts=4, seed=42)

        actuals = run_niah_cell(
            orch=orch,
            variant=_variant(),
            cell=cell,
            expected_pass_rate=0.0,
            output_dir="/tmp/logs/long_context_accuracy",
        )

        self.assertIn("source /tmp/server_env_script.sh &&", captured["cmd"])
        probe_src = LongContextNiahBenchmark.probe_script(
            port=8000,
            host="127.0.0.1",
            model="moonshotai/Kimi-K2.7-Code",
            isl=8192,
            osl=32,
            num_prompts=4,
            seed=42,
            local_files_only=True,
        )
        self.assertIn("local_files_only=True", probe_src)
        self.assertAlmostEqual(actuals["accuracy.niah_pass_rate__niah_8k"], 0.5)

    def test_run_niah_cell_record_only_accepts_low_pass_rate(self):
        orch = MagicMock()
        orch.exec_on_head.return_value = {"head": _niah_payload(pass_rate=0.25)}
        cell = LongContextAccCell(id="niah_8k", isl=8192, osl=32, num_prompts=4, seed=42)

        actuals = run_niah_cell(
            orch=orch,
            variant=_variant(),
            cell=cell,
            expected_pass_rate=0.0,
            output_dir="/tmp/logs/long_context_accuracy",
        )

        self.assertAlmostEqual(actuals["accuracy.niah_pass_rate__niah_8k"], 0.25)

    def test_run_niah_cell_gated_mode_fails_below_floor(self):
        orch = MagicMock()
        orch.exec_on_head.return_value = {"head": _niah_payload(pass_rate=0.25)}
        cell = LongContextAccCell(id="niah_8k", isl=8192, osl=32, num_prompts=4, seed=42)

        with self.assertRaises(RuntimeError):
            run_niah_cell(
                orch=orch,
                variant=_variant(),
                cell=cell,
                expected_pass_rate=0.75,
                output_dir="/tmp/logs/long_context_accuracy",
            )


if __name__ == "__main__":
    unittest.main()
