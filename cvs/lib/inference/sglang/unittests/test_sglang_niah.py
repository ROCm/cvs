'''Unit tests for SGLang long_ctx_niah probe flags.'''

import unittest

from cvs.lib.utils.model_query_lib import LongContextNiahBenchmark


class TestSglangLongCtxNiah(unittest.TestCase):
    def test_prepare_parses_enable_thinking_false(self):
        _, scoring = LongContextNiahBenchmark.prepare(
            {
                "num_prompts": 6,
                "seed": 42,
                "enable_thinking": False,
                "expected_results": {"auto": {"pass_rate": 0.0}},
            },
            port=8000,
            host="127.0.0.1",
            model_id="/models/qwen",
            isl=8192,
            osl=32,
            log_dir="/tmp/logs",
            log_basename="niah.log",
        )
        self.assertIs(scoring["probe_kwargs"]["enable_thinking"], False)
        src = LongContextNiahBenchmark.probe_script(**scoring["probe_kwargs"])
        self.assertIn("ENABLE_THINKING = False", src)
        self.assertIn("/no_think", src)
        self.assertIn("chat_template_kwargs", src)

    def test_probe_omits_thinking_knobs_when_enable_thinking_unset(self):
        src = LongContextNiahBenchmark.probe_script(
            port=8000,
            model="/models/llama",
            isl=8192,
            osl=32,
            num_prompts=6,
            seed=42,
        )
        self.assertIn("ENABLE_THINKING = None", src)


if __name__ == "__main__":
    unittest.main()
