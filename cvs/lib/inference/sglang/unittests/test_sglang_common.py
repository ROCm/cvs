'''Unit tests for cvs/lib/inference/sglang/sglang_common.py.'''

import io
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from cvs.lib.inference.sglang import sglang_common


_SAMPLE_BENCH_LOG = """
Successful requests: 100
Benchmark duration (s): 120.5
Total input tokens: 819200
Total generated tokens: 51200
Request throughput (req/s): 0.833333
Output token throughput (tok/s): 426.666667
Mean TTFT (ms): 45.5
Median TTFT (ms): 40.0
P99 TTFT (ms): 120.0
Mean TPOT (ms): 12.5
Median TPOT (ms): 12.9
P99 TPOT (ms): 18.0
Mean ITL (ms): 11.0
Median ITL (ms): 10.5
P99 ITL (ms): 16.0
Mean E2E Latency (ms): 250.0
Median E2E Latency (ms): 240.0
P90 E2E Latency (ms): 300.0
P95 E2E Latency (ms): 320.0
P99 E2E Latency (ms): 350.0
Serving Benchmark Result
"""


class _FakeSubtests:
    def __init__(self):
        self.failures = []

    class _Ctx:
        def __init__(self, outer):
            self._outer = outer

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is not None:
                self._outer.failures.append(exc)
                return True
            return False

    def test(self, **kwargs):
        return self._Ctx(self)


class TestSglangCommonHelpers(unittest.TestCase):
    def test_first_output(self):
        self.assertEqual(sglang_common.first_output({'a': 'x'}), 'x')
        self.assertEqual(sglang_common.first_output({}), '')

    def test_normalize_hosts(self):
        self.assertEqual(sglang_common.normalize_hosts(None), [])
        self.assertEqual(sglang_common.normalize_hosts('host1'), ['host1'])

    def test_thresholds_from_expected_latency(self):
        specs = sglang_common.thresholds_from_expected({'mean_ttft_ms': 100.0})
        self.assertEqual(specs['mean_ttft_ms']['kind'], 'max_ms')

    def test_perf_enforce_thresholds_defaults_true(self):
        self.assertTrue(sglang_common.perf_enforce_thresholds({}))

    def test_perf_enforce_thresholds_reads_bench_serv_random(self):
        bp = {
            'inference_tests': {
                'bench_serv_random': {'enforce_thresholds': False},
            }
        }
        self.assertFalse(sglang_common.perf_enforce_thresholds(bp))

    def test_node_threshold_actuals_filters_metrics(self):
        inference = {'node1': {'mean_ttft_ms': '50.0', 'other': '1'}}
        thresholds = {'mean_ttft_ms': {'kind': 'max_ms', 'value': 100.0}}
        actuals = sglang_common.node_threshold_actuals(inference, 'node1', thresholds)
        self.assertEqual(actuals, {'mean_ttft_ms': 50.0})

    def test_metric_threshold_violation_missing_metric(self):
        violation = sglang_common.metric_threshold_violation(
            'mean_ttft_ms',
            {},
            {'kind': 'max_ms', 'value': 100.0},
        )
        self.assertIn('missing from actuals', violation)

    def test_metric_threshold_violation_passes_within_threshold(self):
        violation = sglang_common.metric_threshold_violation(
            'mean_ttft_ms',
            {'mean_ttft_ms': 50.0},
            {'kind': 'max_ms', 'value': 100.0},
        )
        self.assertIsNone(violation)

    def test_parse_inference_bench_results_basic(self):
        parsed = sglang_common.parse_inference_bench_results(
            {'node1': _SAMPLE_BENCH_LOG},
            bench_num_prompts=100,
        )
        node = parsed['node1']
        self.assertEqual(node['successful_requests'], '100')
        self.assertEqual(node['benchmark_duration'], '120.5')
        self.assertEqual(node['median_tpot_ms'], '12.9')
        self.assertEqual(node['goodput'], '1.000000')

    def test_parse_inference_bench_results_disagg_extras(self):
        parsed = sglang_common.parse_inference_bench_results(
            {'node1': _SAMPLE_BENCH_LOG},
            num_gpus_for_per_gpu_throughput=16,
            include_itl=True,
            include_extended_e2e_percentiles=True,
        )
        node = parsed['node1']
        self.assertEqual(node['mean_itl_ms'], '11.0')
        self.assertEqual(node['p90_e2e_latency_ms'], '300.0')
        self.assertEqual(node['output_throughput_per_gpu_per_sec'], '26.666667')

    def test_finalize_inference_verification(self):
        host_exec = mock.Mock(return_value={'head': 'Mon Aug 12 18:00'})
        with mock.patch.object(sglang_common.time, 'sleep'):
            end = sglang_common.finalize_inference_verification(host_exec)
        host_exec.assert_called_once()
        self.assertEqual(end, {'head': 'Mon Aug 12 18:00'})

    def test_verify_inference_results_passes(self):
        host_exec = mock.Mock(return_value={'head': 'time'})
        with mock.patch.object(sglang_common.time, 'sleep'):
            end = sglang_common.verify_inference_results(
                {'node1': {'mean_ttft_ms': '50.0'}},
                {'mean_ttft_ms': 100.0},
                host_exec,
            )
        self.assertEqual(end, {'head': 'time'})

    def test_verify_inference_results_subtests(self):
        host_exec = mock.Mock(return_value={'head': 'time'})
        lifecycle = mock.Mock()
        lifecycle.perf_metric_rows = {}
        with mock.patch.object(sglang_common.time, 'sleep'):
            passed, end = sglang_common.verify_inference_results_subtests(
                {'node1': {'mean_ttft_ms': '50.0'}},
                {'mean_ttft_ms': 100.0},
                host_exec,
                _FakeSubtests(),
                'bench_serv',
                lifecycle=lifecycle,
                report_nodeid='test::node',
            )
        self.assertTrue(passed)
        self.assertEqual(end, {'head': 'time'})
        self.assertEqual(lifecycle.perf_metric_rows['test::node'][0]['status'], 'pass')

    def test_verify_inference_results_subtests_gates_violation(self):
        host_exec = mock.Mock(return_value={'head': 'time'})
        lifecycle = mock.Mock()
        lifecycle.perf_metric_rows = {}
        subtests = _FakeSubtests()
        with mock.patch.object(sglang_common.time, 'sleep'):
            passed, _end = sglang_common.verify_inference_results_subtests(
                {'node1': {'mean_ttft_ms': '500.0', 'p99_ttft_ms': '10.0'}},
                {'mean_ttft_ms': 100.0, 'p99_ttft_ms': 100.0},
                host_exec,
                subtests,
                'bench_serv',
                lifecycle=lifecycle,
                report_nodeid='nid',
            )
        self.assertFalse(passed)
        rows = {r['metric']: r['status'] for r in lifecycle.perf_metric_rows['nid']}
        self.assertEqual(rows, {'mean_ttft_ms': 'fail', 'p99_ttft_ms': 'pass'})
        self.assertEqual(len(subtests.failures), 1)

    def test_verify_inference_results_subtests_record_only_passes_on_violation(self):
        host_exec = mock.Mock(return_value={'head': 'time'})
        lifecycle = mock.Mock()
        lifecycle.perf_metric_rows = {}
        subtests = _FakeSubtests()
        with mock.patch.object(sglang_common.time, 'sleep'):
            passed, _end = sglang_common.verify_inference_results_subtests(
                {'node1': {'mean_ttft_ms': '500.0', 'p99_ttft_ms': '10.0'}},
                {'mean_ttft_ms': 100.0, 'p99_ttft_ms': 100.0},
                host_exec,
                subtests,
                'bench_serv',
                lifecycle=lifecycle,
                report_nodeid='nid',
                enforce_thresholds=False,
            )
        self.assertTrue(passed)
        rows = {r['metric']: r['status'] for r in lifecycle.perf_metric_rows['nid']}
        self.assertEqual(rows, {'mean_ttft_ms': 'pass', 'p99_ttft_ms': 'pass'})
        self.assertEqual(subtests.failures, [])

    def test_verify_inference_results_subtests_record_only_fails_without_results(self):
        host_exec = mock.Mock(return_value={'head': 'time'})
        lifecycle = mock.Mock()
        lifecycle.perf_metric_rows = {}
        subtests = _FakeSubtests()
        with mock.patch.object(sglang_common.time, 'sleep'):
            passed, _end = sglang_common.verify_inference_results_subtests(
                {},
                {'mean_ttft_ms': 100.0},
                host_exec,
                subtests,
                'bench_serv',
                lifecycle=lifecycle,
                report_nodeid='nid',
                enforce_thresholds=False,
            )
        self.assertFalse(passed)
        self.assertEqual(subtests.failures, [])

    def test_poll_for_inference_completion_success(self):
        log_text = {'bench': _SAMPLE_BENCH_LOG}

        def fetch_log_tail():
            return log_text

        with mock.patch.object(sglang_common.time, 'sleep'):
            result = sglang_common.poll_for_inference_completion(
                fetch_log_tail,
                sglang_common.parse_inference_bench_results,
                iterations=2,
                waittime_between_iters=0,
            )
        self.assertEqual(result['status'], 'success')
        self.assertIn('bench', result['results'])


class TestHfTokenRedaction(unittest.TestCase):
    def test_export_line_reads_file_and_omits_secret(self):
        secret = 'hf_supersecretvalue'
        line = sglang_common.hf_token_export_line({'hf_token_file': '/home/user/.hf_token', 'HF_TOKEN': secret})
        self.assertNotIn(secret, line)
        self.assertIn('/home/user/.hf_token', line)
        self.assertIn('export HF_TOKEN', line)
        self.assertNotIn('export HF_TOKEN=', line)

    def test_export_line_empty_without_path(self):
        self.assertEqual(sglang_common.hf_token_export_line({}), '')
        self.assertEqual(sglang_common.hf_token_export_line(None), '')

    def test_hub_model_export_line_fails_closed(self):
        secret = 'hf_supersecretvalue'
        line = sglang_common.hf_token_export_line(
            {'hf_token_file': '/home/user/.hf_token', 'HF_TOKEN': secret},
            'meta-llama/Llama-3.1-70B-Instruct',
        )
        self.assertNotIn(secret, line)
        self.assertIn('[ ! -s /home/user/.hf_token ]', line)
        self.assertIn('[ ! -r /home/user/.hf_token ]', line)
        self.assertIn('exit 1', line)
        self.assertIn('[ -z "$HF_TOKEN" ]', line)
        self.assertIn('export HF_TOKEN', line)
        self.assertNotIn('export HF_TOKEN=', line)

    def test_local_model_export_line_stays_optional(self):
        line = sglang_common.hf_token_export_line(
            {'hf_token_file': '/root/models/.hf_token'},
            '/root/models/DeepSeek-R1-0528',
        )
        self.assertNotIn('exit 1', line)
        self.assertIn('[ -s /root/models/.hf_token ]', line)

    def test_hub_model_without_token_path_exits(self):
        line = sglang_common.hf_token_export_line({}, 'meta-llama/Llama-3.1-70B-Instruct')
        self.assertIn('exit 1', line)
        self.assertIn('hf_token_file is unset', line)

    def test_model_needs_hf_token(self):
        self.assertTrue(sglang_common.model_needs_hf_token('meta-llama/Llama-3.1-70B-Instruct'))
        self.assertFalse(sglang_common.model_needs_hf_token('/root/models/DeepSeek-R1-0528'))
        self.assertFalse(sglang_common.model_needs_hf_token(''))
        self.assertFalse(sglang_common.model_needs_hf_token(None))

    def test_sourced_hub_token_script_blocks_launch(self):
        bash = shutil.which('bash')
        if bash is None:
            self.fail('bash is required to check the sourced token script')
        hub = 'meta-llama/Llama-3.1-70B-Instruct'
        secret = 'hf_supersecretvalue'
        script = sglang_common.hf_token_export_line({'hf_token_file': '.hf_token'}, hub)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self._assert_source_blocks(bash, tmp, script)
            with open(os.path.join(tmp, '.hf_token'), 'w', encoding='utf-8'):
                pass
            self._assert_source_blocks(bash, tmp, script)
            with open(os.path.join(tmp, '.hf_token'), 'w', encoding='utf-8') as handle:
                handle.write(' \n\t ')
            self._assert_source_blocks(bash, tmp, script)
            with open(os.path.join(tmp, '.hf_token'), 'w', encoding='utf-8') as handle:
                handle.write(secret + '\n')
            subprocess.run([bash, '-c', 'chmod 000 .hf_token'], cwd=tmp, check=True)
            try:
                readable = subprocess.run([bash, '-c', 'test -r .hf_token'], cwd=tmp, check=False)
                if readable.returncode != 0:
                    self._assert_source_blocks(bash, tmp, script)
                else:
                    # drvfs still reports mode 000 as readable. [ ! -r ] is what
                    # rejects that file when the filesystem enforces the mode.
                    self.assertIn('[ ! -r ', script)
            finally:
                subprocess.run([bash, '-c', 'chmod 644 .hf_token'], cwd=tmp, check=False)
            with open(os.path.join(tmp, '.hf_token'), 'w', encoding='utf-8') as handle:
                handle.write('  ' + secret + ' \n')
            proc = self._source_script(bash, tmp, script)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('LAUNCHED', proc.stdout)
            self.assertEqual(proc.stdout.splitlines()[0], secret)
            self.assertNotIn(secret, script)

    def test_sourced_local_model_script_allows_missing_token(self):
        bash = shutil.which('bash')
        if bash is None:
            self.fail('bash is required to check the sourced token script')
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            proc = self._source_script(
                bash,
                tmp,
                sglang_common.hf_token_export_line(
                    {'hf_token_file': '.hf_token'},
                    '/root/models/DeepSeek-R1-0528',
                ),
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('LAUNCHED', proc.stdout)
        self.assertIn('UNSET', proc.stdout)

    def test_record_container_exec_failure(self):
        with mock.patch.object(sglang_common, 'fail_test') as fail:
            sglang_common.record_container_exec_failure(
                {'node-a': {'exit_code': 1, 'output': 'HF token file missing, empty, or unreadable: .hf_token'}},
                'SGLang env script failed inside the container',
            )
        fail.assert_called_once()
        message = fail.call_args.args[0]
        self.assertIn('node-a', message)
        self.assertIn('exit 1', message)
        self.assertIn('missing, empty, or unreadable', message)
        with mock.patch.object(sglang_common, 'fail_test') as fail:
            sglang_common.record_container_exec_failure({'node-a': {'exit_code': 0, 'output': ''}}, 'ok')
        fail.assert_not_called()

    def _source_script(self, bash, cwd, script):
        # The launch shell has no set -e. exit from the sourced script is what
        # stops the following command. Keep the launch body in a file so a
        # Windows bash stub cannot expand $HF_TOKEN before bash runs it.
        with open(os.path.join(cwd, 'env.sh'), 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(script)
        launch = (
            'source ./env.sh\n'
            'if [ -n "${HF_TOKEN+x}" ]; then printf %s "$HF_TOKEN"; else echo UNSET; fi\n'
            'echo\n'
            'echo LAUNCHED\n'
        )
        with open(os.path.join(cwd, 'launch.sh'), 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(launch)
        return subprocess.run(
            [bash, 'launch.sh'],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )

    def _assert_source_blocks(self, bash, cwd, script):
        proc = self._source_script(bash, cwd, script)
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn('LAUNCHED', proc.stdout)
        self.assertNotIn('hf_supersecretvalue', proc.stdout + proc.stderr)

    def test_scrub_drops_token_and_keeps_path(self):
        secret = 'plain-token-value'
        cleaned = sglang_common.scrub_sglang_secrets(
            {
                'hf_token_file': '/home/user/.hf_token',
                'HF_TOKEN': secret,
                'container_config': {'env_dict': {'HUGGING_FACE_HUB_TOKEN': secret, 'NCCL_DEBUG': 'ERROR'}},
                'add_export_env': [f'HF_TOKEN={secret}', 'export hf_token={secret}', 'SGLANG_USE_AITER=1'],
            }
        )
        blob = str(cleaned)
        self.assertNotIn(secret, blob)
        self.assertEqual(cleaned['hf_token_file'], '/home/user/.hf_token')
        self.assertEqual(cleaned['container_config']['env_dict'], {'NCCL_DEBUG': 'ERROR'})
        self.assertEqual(cleaned['add_export_env'], ['SGLANG_USE_AITER=1'])

    def test_add_export_env_block_skips_token_assignment(self):
        secret = 'hf_supersecretvalue'
        block = sglang_common.add_export_env_block(
            {'add_export_env': [f'HF_TOKEN={secret}', 'SGLANG_USE_AITER=1']},
            indent='',
        )
        self.assertNotIn(secret, block)
        self.assertIn('export SGLANG_USE_AITER=1', block)

    def test_logged_env_command_does_not_contain_token(self):
        secret = 'hf_supersecretvalue'
        inf = sglang_common.scrub_sglang_secrets({'hf_token_file': '/home/user/.hf_token', 'HF_TOKEN': secret})
        bp = sglang_common.scrub_sglang_secrets({'add_export_env': [f'HF_TOKEN={secret}', 'SGLANG_USE_AITER=1']})
        body = sglang_common.hf_token_export_line(inf) + sglang_common.add_export_env_block(bp, indent='')
        cmd = 'bash -c ' + shlex.quote(f"cat > /tmp/server_env_script.sh <<'EOF'\n{body}EOF\n")
        self.assertNotIn(secret, body)
        self.assertNotIn(secret, cmd)
        self.assertIn('.hf_token', cmd)

    def test_redact_text_keeps_token_file_path(self):
        secret = 'hf_supersecretvalue'
        text = (
            f"cmd = export HF_TOKEN={secret} path=/home/user/.hf_token "
            f"{{'HF_TOKEN': '{secret}', 'hf_token_file': '/home/user/.hf_token'}}"
        )
        redacted = sglang_common.redact_hf_token_text(text)
        self.assertNotIn(secret, redacted)
        self.assertIn('/home/user/.hf_token', redacted)
        self.assertIn('<redacted>', redacted)

    def test_log_filter_redacts_formatted_dict(self):
        secret = 'hf_supersecretvalue'
        logger = logging.getLogger('sglang-hf-token-redaction-test')
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addFilter(sglang_common.HfTokenLogFilter())
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        try:
            logger.info("cmd = export HF_TOKEN=%s", secret)
            logger.info("%s", {'HF_TOKEN': 'not-an-hf-prefix', 'hf_token_file': '/home/user/.hf_token'})
        finally:
            logger.removeHandler(handler)
            logger.removeFilter(logger.filters[-1])
        text = stream.getvalue()
        self.assertNotIn(secret, text)
        self.assertNotIn('not-an-hf-prefix', text)
        self.assertIn('/home/user/.hf_token', text)
        self.assertIn('<redacted>', text)


if __name__ == '__main__':
    unittest.main()
