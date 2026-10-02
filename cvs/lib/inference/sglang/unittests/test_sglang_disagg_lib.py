'''Unit tests for cvs/lib/inference/sglang/sglang_disagg_lib.py.'''

import unittest
from unittest import mock

from cvs.lib.inference.sglang import sglang_disagg_lib
from cvs.lib.inference.sglang.sglang_disagg_lib import SglangDisaggPD


def _bare_disagg(exec_text_side_effect):
    obj = SglangDisaggPD.__new__(SglangDisaggPD)
    obj._container_exec_text = mock.Mock(side_effect=exec_text_side_effect)
    return obj


@mock.patch.object(sglang_disagg_lib.time, 'sleep')
class TestPollRoleLogReady(unittest.TestCase):
    def test_grep_returns_only_first_match_without_context(self, _sleep):
        obj = _bare_disagg(['The server is fired up and ready to roll'])
        with mock.patch.object(sglang_disagg_lib, 'fail_test') as fail:
            obj._poll_role_log_ready('/logs/decode_node0/decode_server.log', ['h1'], 'Decode node 0')
        cmd = obj._container_exec_text.call_args.args[0]
        self.assertIn('grep -a -o -m 1 -E', cmd)
        self.assertNotIn('-B', cmd)
        self.assertNotIn('-A', cmd)
        self.assertEqual(obj._container_exec_text.call_args.kwargs['hosts'], ['h1'])
        fail.assert_not_called()

    def test_keeps_polling_until_banner_appears(self, _sleep):
        obj = _bare_disagg(['', '', 'The server is fired up and ready to roll'])
        with mock.patch.object(sglang_disagg_lib, 'fail_test') as fail:
            obj._poll_role_log_ready('/logs/p.log', ['h1'], 'Prefill node 0')
        self.assertEqual(obj._container_exec_text.call_count, 3)
        fail.assert_not_called()

    def test_fails_when_banner_never_appears(self, _sleep):
        obj = _bare_disagg(lambda *a, **k: 'ABORT: Output Truncated by agent on Host: h1')
        with mock.patch.object(sglang_disagg_lib, 'fail_test') as fail:
            obj._poll_role_log_ready('/logs/p.log', ['h1'], 'Prefill node 0', no_of_iterations=3)
        self.assertEqual(obj._container_exec_text.call_count, 2)
        fail.assert_called_once()
        self.assertIn('Prefill node 0', fail.call_args.args[0])


class TestDisaggOpenAIProbe(unittest.TestCase):
    def test_skips_structured_output(self):
        obj = SglangDisaggPD.__new__(SglangDisaggPD)
        obj.inf_dict = {
            "proxy_router_serv_port": "8000",
            "client_host": "127.0.0.1",
            "proxy_router_node": ["proxy-0"],
            "benchmark_serv_node": ["bench-0"],
        }
        obj.bp_dict = {"model": "meta-llama/Llama-3.1-70B-Instruct"}
        obj.log_dir = "/logs"
        obj.benchmark_serv_node = ["bench-0"]
        obj._container_exec = mock.Mock()
        obj.log_kv_transfer_logs = mock.Mock()
        with mock.patch.object(
            sglang_disagg_lib,
            "verify_openai_compatible_endpoints_common",
            return_value=([], False),
        ) as verify:
            obj.verify_openai_compatible_endpoints()
        self.assertFalse(verify.call_args.kwargs["include_structured_output"])


if __name__ == '__main__':
    unittest.main()
