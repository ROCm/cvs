'''Unit tests for cvs/lib/inference/sglang/sglang_common.py.'''

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

    def test_launch_script_start_cmd_quotes_paths_and_nohups_with_bash(self):
        cmd = sglang_common.launch_script_start_cmd(
            '/tmp/server_launch_script.sh',
            'python3 -m sglang.launch_server --host 0.0.0.0\n',
            '/tmp/server_env_script.sh',
            '/logs/node a/server.log',
        )
        self.assertTrue(cmd.startswith('bash -c '))
        self.assertIn('cat > /tmp/server_launch_script.sh', cmd)
        self.assertIn('python3 -m sglang.launch_server --host 0.0.0.0', cmd)
        self.assertIn('chmod 755 /tmp/server_launch_script.sh', cmd)
        self.assertIn('/logs/node a', cmd)
        self.assertIn('source /tmp/server_env_script.sh', cmd)
        self.assertIn('nohup bash /tmp/server_launch_script.sh', cmd)
        self.assertIn('2>&1 &', cmd)

    def test_stage_and_start_launch_script_execs_on_normalized_hosts(self):
        seen = {}

        def exec_in_container(cmd, hosts=None):
            seen['cmd'] = cmd
            seen['hosts'] = hosts
            return {'n0': ''}

        sglang_common.stage_and_start_launch_script(
            exec_in_container,
            'n0',
            '/tmp/prefill_launch_script.sh',
            'export NNODES=1\n',
            '/tmp/prefill_env_script.sh',
            '/logs/prefill_node0/prefill_server.log',
        )
        self.assertEqual(seen['hosts'], ['n0'])
        self.assertIn('nohup bash /tmp/prefill_launch_script.sh', seen['cmd'])

    def test_resolve_single_execution_hosts_uses_first_cluster_node(self):
        hosts = sglang_common.resolve_single_execution_hosts(
            {
                'node_dict': {
                    'node-a': {'mgmt_ip': 'node-a'},
                    'node-b': {'mgmt_ip': 'node-b'},
                }
            }
        )
        self.assertEqual(hosts, ['node-a'])

    def test_resolve_single_execution_hosts_rejects_empty_cluster(self):
        with self.assertRaisesRegex(ValueError, 'at least one host'):
            sglang_common.resolve_single_execution_hosts({'node_dict': {}})

    def test_resolve_distributed_execution_hosts_takes_first_nnodes(self):
        cluster = {
            'node_dict': {
                'node-a': {'mgmt_ip': 'node-a'},
                'node-b': {'mgmt_ip': 'node-b'},
                'node-c': {'mgmt_ip': 'node-c'},
            }
        }
        hosts = sglang_common.resolve_distributed_execution_hosts(cluster, {'nnodes': 2})
        self.assertEqual(hosts, ['node-a', 'node-b'])

    def test_resolve_distributed_execution_hosts_ignores_pinned_server_list(self):
        cluster = {
            'node_dict': {
                'node-a': {'mgmt_ip': 'node-a'},
                'node-b': {'mgmt_ip': 'node-b'},
            }
        }
        hosts = sglang_common.resolve_distributed_execution_hosts(
            cluster,
            {
                'nnodes': 2,
                'server_node_list': ['node-b', 'node-a'],
                'benchmark_serv_node': 'node-b',
            },
        )
        self.assertEqual(hosts, ['node-a', 'node-b'])

    def test_resolve_distributed_execution_hosts_rejects_nnodes_less_than_2(self):
        with self.assertRaisesRegex(ValueError, 'nnodes >= 2'):
            sglang_common.resolve_distributed_execution_hosts(
                {'node_dict': {'node-a': {}, 'node-b': {}}},
                {'nnodes': 1},
            )

    def test_resolve_distributed_execution_hosts_rejects_nnodes_larger_than_cluster(self):
        with self.assertRaisesRegex(ValueError, 'requests 4 nodes'):
            sglang_common.resolve_distributed_execution_hosts(
                {'node_dict': {'node-a': {}, 'node-b': {}}},
                {'nnodes': 4},
            )

    def test_assign_disagg_pd_roles_two_nodes(self):
        roles = sglang_common.assign_disagg_pd_roles(['n0', 'n1', 'n2'], 2)
        self.assertEqual(roles['prefill_node_list'], ['n0'])
        self.assertEqual(roles['decode_node_list'], ['n1'])
        self.assertEqual(roles['proxy_router_node'], 'n0')
        self.assertEqual(roles['benchmark_serv_node'], 'n0')
        self.assertEqual(roles['prefill_coordinator_addr'], 'n0')
        self.assertEqual(roles['decode_coordinator_addr'], 'n1')

    def test_assign_disagg_pd_roles_four_and_six_nodes(self):
        four = sglang_common.assign_disagg_pd_roles(['n0', 'n1', 'n2', 'n3'], 4)
        self.assertEqual(four['prefill_node_list'], ['n0', 'n2'])
        self.assertEqual(four['decode_node_list'], ['n1', 'n3'])
        six = sglang_common.assign_disagg_pd_roles(['n0', 'n1', 'n2', 'n3', 'n4', 'n5'], 6)
        self.assertEqual(six['prefill_node_list'], ['n0', 'n2', 'n3'])
        self.assertEqual(six['decode_node_list'], ['n1', 'n4', 'n5'])

    def test_assign_disagg_pd_roles_rejects_odd_and_too_small(self):
        with self.assertRaisesRegex(ValueError, 'even nnodes'):
            sglang_common.assign_disagg_pd_roles(['n0', 'n1', 'n2'], 3)
        with self.assertRaisesRegex(ValueError, 'nnodes >= 2'):
            sglang_common.assign_disagg_pd_roles(['n0', 'n1'], 1)
        with self.assertRaisesRegex(ValueError, 'at least 2 cluster.json hosts'):
            sglang_common.assign_disagg_pd_roles(['n0'], 2)
        with self.assertRaisesRegex(ValueError, 'requests 4 nodes'):
            sglang_common.assign_disagg_pd_roles(['n0', 'n1'], 4)

    def test_resolve_disagg_execution_roles_rejects_cluster_smaller_than_two(self):
        with self.assertRaisesRegex(ValueError, 'at least 2 cluster.json hosts'):
            sglang_common.resolve_disagg_execution_roles(
                {'node_dict': {'n0': {}}},
                {'nnodes': 2},
            )

    def test_disagg_bind_roles_requires_stamped_pd_lists(self):
        from cvs.lib.inference.sglang.sglang_disagg_lib import SglangDisaggPD

        obj = SglangDisaggPD.__new__(SglangDisaggPD)
        obj.inf_dict = {'_execution_hosts': ['n0', 'n1'], 'nnodes': 2}
        with self.assertRaisesRegex(ValueError, 'load_variant\\(\\) to stamp PD roles'):
            obj._bind_roles()

    def test_disagg_bind_roles_consumes_stamped_lists(self):
        from cvs.lib.inference.sglang.sglang_disagg_lib import SglangDisaggPD

        obj = SglangDisaggPD.__new__(SglangDisaggPD)
        obj.inf_dict = {
            'prefill_node_list': ['n0'],
            'decode_node_list': ['n1'],
            'proxy_router_node': 'n0',
            'benchmark_serv_node': 'n0',
        }
        obj._bind_roles()
        self.assertEqual(obj.prefill_node_list, ['n0'])
        self.assertEqual(obj.decode_node_list, ['n1'])
        self.assertEqual(obj.proxy_node, ['n0'])
        self.assertEqual(obj.benchmark_serv_node, ['n0'])

    def test_single_host_exec_uses_exec_on_host_first_node_only(self):
        from cvs.lib.inference.sglang.sglang_single_lib import SglangSingle

        obj = SglangSingle.__new__(SglangSingle)
        obj.execution_hosts = ['node-a']
        obj.orch = mock.Mock()
        obj.orch.exec_on_host.return_value = {'node-a': 'ok'}
        out = obj._host_exec('amd-smi', timeout=30)
        self.assertEqual(out, {'node-a': 'ok'})
        obj.orch.exec_on_host.assert_called_once_with('amd-smi', hosts=['node-a'], timeout=30)

    def test_distributed_host_exec_defaults_to_rank0(self):
        from cvs.lib.inference.sglang.sglang_distributed_lib import SglangDistributed

        obj = SglangDistributed.__new__(SglangDistributed)
        obj.benchmark_serv_node = 'rank0'
        obj.orch = mock.Mock()
        obj._host_exec('date')
        obj.orch.exec_on_host.assert_called_once_with('date', hosts=['rank0'], timeout=None)
        obj.orch.exec_on_host.reset_mock()
        obj._host_exec('amd-smi', hosts=['n0', 'n1'], timeout=15)
        obj.orch.exec_on_host.assert_called_once_with('amd-smi', hosts=['n0', 'n1'], timeout=15)

    def test_disagg_host_exec_defaults_to_head(self):
        from cvs.lib.inference.sglang.sglang_disagg_lib import SglangDisaggPD

        obj = SglangDisaggPD.__new__(SglangDisaggPD)
        obj.orch = mock.Mock()
        obj.orch.head_node = 'n0'
        obj._host_exec('dmesg')
        obj.orch.exec_on_host.assert_called_once_with('dmesg', hosts=['n0'], timeout=None)

    def test_add_cli_flags_block_includes_activated_long_context_flags(self):
        block = sglang_common.add_cli_flags_block(
            {
                'add_flags': ['--attention-backend aiter'],
                'lng_ctx_activate': True,
                'context_length': '262144',
                'chunked_prefill_size': '8192',
                'max_prefill_tokens': '8192',
            },
            indent='',
        )

        self.assertEqual(
            block.splitlines(),
            [
                '--attention-backend aiter \\',
                '--context-length 262144 \\',
                '--chunked-prefill-size 8192 \\',
                '--max-prefill-tokens 8192 \\',
            ],
        )

    def test_add_cli_flags_block_omits_long_context_flags_when_disabled(self):
        block = sglang_common.add_cli_flags_block(
            {
                'add_flags': ['--attention-backend aiter'],
                'lng_ctx_activate': False,
                'context_length': '262144',
                'chunked_prefill_size': '8192',
            },
            indent='',
        )

        self.assertEqual(block, '--attention-backend aiter \\')

    def test_add_cli_flags_block_omits_chunked_prefill_for_decode(self):
        block = sglang_common.add_cli_flags_block(
            {
                'add_flags': ['--attention-backend aiter'],
                'lng_ctx_activate': True,
                'context_length': '262144',
                'chunked_prefill_size': '8192',
                'max_prefill_tokens': '8192',
            },
            indent='',
            include_chunked_prefill=False,
        )

        self.assertEqual(
            block.splitlines(),
            [
                '--attention-backend aiter \\',
                '--context-length 262144 \\',
            ],
        )

    def test_add_cli_flags_block_requires_long_context_parameters(self):
        with self.assertRaisesRegex(ValueError, 'chunked_prefill_size'):
            sglang_common.add_cli_flags_block(
                {
                    'lng_ctx_activate': True,
                    'context_length': '262144',
                }
            )

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

    def test_poll_for_inference_completion_uses_config_iteration_cap(self):
        calls = {'n': 0}

        def fetch_log_tail():
            calls['n'] += 1
            return {'bench': 'still running'}

        with mock.patch.object(sglang_common.time, 'sleep'):
            result = sglang_common.poll_for_inference_completion(
                fetch_log_tail,
                sglang_common.parse_inference_bench_results,
                iterations=50,
                waittime_between_iters=0,
                total_timeout=None,
                inference_poll_iterations='2',
            )
        self.assertEqual(calls['n'], 2)
        self.assertEqual(result['status'], 'stuck_in_progress')

    def test_scan_sglang_error_logs_clean(self):
        commands = []

        def execute(host, command):
            commands.append((host, command))
            return {host: ''}

        with mock.patch.object(sglang_common, 'fail_test') as fail:
            clean = sglang_common.scan_sglang_error_logs(
                [('node1', '/logs/server.log', 'server')],
                execute,
            )

        self.assertTrue(clean)
        fail.assert_not_called()
        self.assertIn('grep -niE -C 2', commands[0][1])
        self.assertIn('/logs/server.log', commands[0][1])

    def test_scan_sglang_error_logs_reports_error_and_missing_log(self):
        outputs = {
            'node1': '42-RuntimeError: worker failed\n',
            'node2': '__CVS_MISSING_LOG__:/logs/benchmark.log\n',
        }

        def execute(host, _command):
            return {host: outputs[host]}

        with mock.patch.object(sglang_common, 'fail_test') as fail:
            clean = sglang_common.scan_sglang_error_logs(
                [
                    ('node1', '/logs/server.log', 'server'),
                    ('node2', '/logs/benchmark.log', 'benchmark'),
                ],
                execute,
            )

        self.assertFalse(clean)
        self.assertEqual(fail.call_count, 2)
        messages = ' '.join(call.args[0] for call in fail.call_args_list)
        self.assertIn('application exception', messages)
        self.assertIn('is missing', messages)

    def test_scan_sglang_error_logs_ignores_only_matching_line(self):
        output = '10:RuntimeError: expected shutdown noise\n11:RuntimeError: real worker failure\n'

        with mock.patch.object(sglang_common, 'fail_test') as fail:
            clean = sglang_common.scan_sglang_error_logs(
                [('node1', '/logs/server.log', 'server')],
                lambda host, _command: {host: output},
                error_patterns={'runtime': r'RuntimeError'},
                ignore_error_patterns={'shutdown': r'expected shutdown noise'},
            )

        self.assertFalse(clean)
        fail.assert_called_once()

    def test_log_sglang_log_matches_writes_hits_without_fail_test(self):
        output = '88:Mooncake TransferEngine: QP failed\n'

        with mock.patch.object(sglang_common, 'fail_test') as fail:
            with mock.patch.object(sglang_common.log, 'info') as info:
                sglang_common.log_sglang_log_matches(
                    [('prefill', '/logs/prefill_server.log', 'prefill node 0')],
                    lambda host, _command: {host: output},
                    sglang_common.SGLANG_KV_TRANSFER_PATTERNS,
                    heading='KV / PD transfer log matches',
                )

        fail.assert_not_called()
        logged = ' '.join(str(call) for call in info.call_args_list)
        self.assertIn('Mooncake TransferEngine', logged)
        self.assertIn('prefill node 0', logged)

    def test_single_log_scan_targets_server_only(self):
        from cvs.lib.inference.sglang.sglang_single_lib import SglangSingle

        obj = SglangSingle.__new__(SglangSingle)
        obj.execution_hosts = ['node1']
        obj.log_dir = '/logs'
        obj._container_exec = mock.Mock(return_value={'node1': ''})

        self.assertTrue(obj.scan_for_inference_errors())
        commands = [call.args[0] for call in obj._container_exec.call_args_list]
        self.assertTrue(any('/logs/node1/server.log' in command for command in commands))
        self.assertFalse(any('benchmark_results.log' in command for command in commands))

    def test_single_openai_logs_server_error_lines(self):
        from cvs.lib.inference.sglang.sglang_single_lib import SglangSingle

        obj = SglangSingle.__new__(SglangSingle)
        obj.execution_hosts = ['node1']
        obj.log_dir = '/logs'
        obj._container_exec = mock.Mock(return_value={'node1': '42:RuntimeError: worker failed\n'})

        with mock.patch.object(sglang_common.log, 'info'):
            obj.log_server_error_logs()

        commands = [call.args[0] for call in obj._container_exec.call_args_list]
        self.assertTrue(any('/logs/node1/server.log' in command for command in commands))
        self.assertFalse(any('benchmark_results.log' in command for command in commands))

    def test_distributed_log_scan_targets_every_rank_server(self):
        from cvs.lib.inference.sglang.sglang_distributed_lib import SglangDistributed

        obj = SglangDistributed.__new__(SglangDistributed)
        obj.server_node_list = ['node1', 'node2']
        obj.log_dir = '/logs'
        obj._container_exec = mock.Mock(side_effect=lambda _cmd, hosts, timeout: {hosts[0]: ''})

        self.assertTrue(obj.scan_for_inference_errors())
        commands = [call.args[0] for call in obj._container_exec.call_args_list]
        self.assertTrue(any('/logs/server_node0/server.log' in command for command in commands))
        self.assertTrue(any('/logs/server_node1/server.log' in command for command in commands))
        self.assertFalse(any('benchmark_results.log' in command for command in commands))

    def test_distributed_openai_logs_every_rank_server_error_lines(self):
        from cvs.lib.inference.sglang.sglang_distributed_lib import SglangDistributed

        obj = SglangDistributed.__new__(SglangDistributed)
        obj.server_node_list = ['node1', 'node2']
        obj.log_dir = '/logs'
        obj._container_exec = mock.Mock(side_effect=lambda _cmd, hosts, timeout: {hosts[0]: '42:NCCL ERROR: boom\n'})

        with mock.patch.object(sglang_common.log, 'info'):
            obj.log_server_error_logs()

        commands = [call.args[0] for call in obj._container_exec.call_args_list]
        self.assertTrue(any('/logs/server_node0/server.log' in command for command in commands))
        self.assertTrue(any('/logs/server_node1/server.log' in command for command in commands))
        self.assertFalse(any('benchmark_results.log' in command for command in commands))

    def test_disagg_log_scan_targets_prefill_and_decode_servers(self):
        from cvs.lib.inference.sglang.sglang_disagg_lib import SglangDisaggPD

        obj = SglangDisaggPD.__new__(SglangDisaggPD)
        obj.prefill_node_list = ['prefill']
        obj.decode_node_list = ['decode']
        obj.log_dir = '/logs'
        obj._container_exec = mock.Mock(side_effect=lambda _cmd, hosts, timeout: {hosts[0]: ''})

        self.assertTrue(obj.scan_for_inference_errors())
        commands = [call.args[0] for call in obj._container_exec.call_args_list]
        self.assertTrue(any('/logs/prefill_node0/prefill_server.log' in command for command in commands))
        self.assertTrue(any('/logs/decode_node0/decode_server.log' in command for command in commands))
        self.assertFalse(any('proxy_router.log' in command for command in commands))
        self.assertFalse(any('benchmark_results.log' in command for command in commands))

    def test_disagg_openai_logs_kv_transfer_from_pd_and_router(self):
        from cvs.lib.inference.sglang.sglang_disagg_lib import SglangDisaggPD

        obj = SglangDisaggPD.__new__(SglangDisaggPD)
        obj.prefill_node_list = ['prefill']
        obj.decode_node_list = ['decode']
        obj.proxy_node = ['router']
        obj.log_dir = '/logs'
        obj._container_exec = mock.Mock(side_effect=lambda _cmd, hosts, timeout: {hosts[0]: '10:kv_transfer timeout\n'})

        with mock.patch.object(sglang_common.log, 'info'):
            obj.log_kv_transfer_logs()

        commands = [call.args[0] for call in obj._container_exec.call_args_list]
        self.assertTrue(any('/logs/prefill_node0/prefill_server.log' in command for command in commands))
        self.assertTrue(any('/logs/decode_node0/decode_server.log' in command for command in commands))
        self.assertTrue(any('/logs/proxy_router_node/proxy_router.log' in command for command in commands))

    def test_openai_completions_5xx_or_hang_uses_completions_not_chat(self):
        chat_5xx = {
            'model_endpoint': (200, {'data': [{'id': 'm'}]}),
            'chat_completion_endpoint': (500, {'error': 'chat boom'}),
            'completion_endpoint': (200, {'choices': [{'text': 'Paris'}]}),
        }
        self.assertFalse(sglang_common.openai_completions_5xx_or_hang(chat_5xx))

        completions_5xx = {
            'model_endpoint': (200, {'data': [{'id': 'm'}]}),
            'chat_completion_endpoint': (200, {'choices': [{'message': {'content': 'OK'}}]}),
            'completion_endpoint': (503, {'error': 'No decode workers'}),
        }
        self.assertTrue(sglang_common.openai_completions_5xx_or_hang(completions_5xx))

        completions_4xx = {
            'model_endpoint': (200, {'data': [{'id': 'm'}]}),
            'chat_completion_endpoint': (200, {'choices': [{'message': {'content': 'OK'}}]}),
            'completion_endpoint': (400, {'error': 'bad request'}),
        }
        self.assertFalse(sglang_common.openai_completions_5xx_or_hang(completions_4xx))

        self.assertTrue(sglang_common.openai_completions_5xx_or_hang({}, probe_err='timeout'))

    def test_openai_completions_missing_probe_key_fails_test(self):
        missing = {'chat_completion_endpoint': (200, {})}
        with mock.patch.object(sglang_common, 'fail_test') as fail:
            hung = sglang_common.openai_completions_5xx_or_hang(missing)
        self.assertTrue(hung)
        fail.assert_called_once()
        self.assertIn('completion_endpoint', fail.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
