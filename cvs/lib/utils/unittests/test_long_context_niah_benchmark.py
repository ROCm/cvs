'''Unit tests for LongContextNiahBenchmark helpers.'''

import unittest

from cvs.lib.utils.model_query_lib import LongContextNiahBenchmark


class TestLongContextNiahBenchmark(unittest.TestCase):
    def test_probe_script_local_files_only_kwarg(self):
        src = LongContextNiahBenchmark.probe_script(
            port=8000,
            model="openai/gpt-oss-120b",
            isl=8192,
            osl=32,
            num_prompts=4,
            seed=42,
            local_files_only=True,
        )
        self.assertIn("local_files_only=True", src)

    def test_probe_script_omits_local_files_only_by_default(self):
        src = LongContextNiahBenchmark.probe_script(
            port=8000,
            model="openai/gpt-oss-120b",
            isl=8192,
            osl=32,
            num_prompts=4,
            seed=42,
        )
        self.assertNotIn("local_files_only", src)

    def test_prepare_passes_local_files_only_to_probe_kwargs(self):
        _, scoring = LongContextNiahBenchmark.prepare(
            {
                "num_prompts": 4,
                "seed": 42,
                "local_files_only": True,
                "expected_results": {"auto": {"pass_rate": 0.0}},
            },
            port=8000,
            host="127.0.0.1",
            model_id="openai/gpt-oss-120b",
            isl=8192,
            osl=32,
            log_dir="/tmp/logs",
            log_basename="niah.log",
        )
        self.assertTrue(scoring["probe_kwargs"]["local_files_only"])


if __name__ == "__main__":
    unittest.main()
