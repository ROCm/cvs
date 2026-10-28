'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cvs.lib import ibperf_lib
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.tests.ibperf import conftest


def _cluster(nodes):
    return {
        'username': 'user',
        'priv_key_file': '/keys/id',
        'node_dict': {n: {'vpc_ip': n} for n in nodes},
    }


class TestOrchFixture(unittest.TestCase):
    def _start(self, nodes):
        orch = MagicMock()
        patcher = patch.object(conftest.OrchestratorFactory, 'create_orchestrator', return_value=orch)
        create = patcher.start()
        self.addCleanup(patcher.stop)
        gen = conftest.orch.__wrapped__(_cluster(nodes), conftest.lifecycle.__wrapped__())
        self.assertIs(next(gen).exec, orch.exec)
        return gen, orch, create.call_args.args[1]

    def test_install_orch_covers_nodes_used_by_vpod_pairs_with_odd_count(self):
        cluster_nodes = ['A', 'B', 'C']
        install_gen, install_orch, cfg = self._start(cluster_nodes)
        install_nodes = set(cfg.node_dict)
        pairings = (
            ('inter_vpod', {'A': '1', 'B': '1', 'C': '2'}),
            ('intra_vpod', {'A': '1', 'B': '2', 'C': '2'}),
        )
        for mode, vpod_map in pairings:
            with self.subTest(mode=mode):
                pairs, _ = ibperf_lib.build_ibperf_node_pairs(cluster_nodes, mode, vpod_map)
                paired_nodes = {node for pair in pairs for node in pair}
                self.assertLessEqual(paired_nodes, install_nodes)
                self.assertIn('C', paired_nodes)
        self.assertEqual(install_nodes, set(cluster_nodes))
        with self.assertRaises(StopIteration):
            next(install_gen)
        install_orch.close.assert_called_once_with()

    def test_teardown_closes_the_orchestrator(self):
        gen, orch, _ = self._start(['n1', 'n2'])
        orch.close.assert_not_called()
        with self.assertRaises(StopIteration):
            next(gen)
        orch.close.assert_called_once_with()

    @patch('cvs.core.orchestrators.container.RuntimeFactory')
    @patch('cvs.core.orchestrators.baremetal.MultiProcessParallelHandle')
    def test_container_mode_launch_runs_inside_then_teardown(self, _pssh, runtime_factory):
        runtime = runtime_factory.create.return_value
        cluster = {**_cluster(['n1', 'n2']), 'orchestrator': 'container', 'container': {'image': 'img', 'name': 'ct'}}
        lifecycle = conftest.lifecycle.__wrapped__()
        gen = conftest.orch.__wrapped__(cluster, lifecycle)

        orch = next(gen)
        runtime.setup_containers.assert_not_called()
        conftest.launch_containers(orch, lifecycle)
        runtime.setup_containers.assert_called_once()
        orch.exec('rdma link')
        self.assertEqual(runtime.exec.call_args.args[:2], ('ct', 'rdma link'))
        orch.exec_cmd_list(['a', 'b'])
        self.assertEqual(runtime.exec_cmd_list.call_args.args[:2], ('ct', ['a', 'b']))
        orch.exec_on_head('make')
        self.assertEqual(runtime.exec_on_head.call_args.args[:2], ('ct', 'make'))
        conftest.teardown_containers(orch, lifecycle)
        with self.assertRaises(StopIteration):
            next(gen)
        runtime.teardown_containers.assert_called_once_with('ct')

    @patch('cvs.core.orchestrators.baremetal.get_passwordless_sudo_status', return_value={'n1': True, 'n2': True})
    @patch('cvs.core.orchestrators.container.RuntimeFactory')
    @patch('cvs.core.orchestrators.baremetal.MultiProcessParallelHandle')
    def test_per_run_teardown_also_removes_the_container_on_pruned_nodes(self, pssh, _runtime_factory, _sudo):
        pssh.return_value.prune_nodes.return_value = ['n2']
        cluster = {**_cluster(['n1', 'n2']), 'orchestrator': 'container', 'container': {'image': 'img', 'name': 'ct'}}
        lifecycle = conftest.lifecycle.__wrapped__()
        orch = next(conftest.orch.__wrapped__(cluster, lifecycle))
        conftest.launch_containers(orch, lifecycle)

        orch.prune_nodes(['n2'])
        conftest.teardown_containers(orch, lifecycle)
        self.assertEqual(pssh.call_args.args[1], ['n2'])
        pssh.return_value.exec.assert_any_call(
            'sudo -n docker rm -f ct 2>/dev/null || true',
            timeout=None,
            detailed=False,
            print_console=True,
            inactivity_timeout=None,
        )

    def test_exec_on_head_without_a_launched_container_raises(self):
        orch = conftest.IbperfOrch(MagicMock(orchestrator_type='container', container_id=None))
        with self.assertRaisesRegex(RuntimeError, 'No containers running'):
            orch.exec_on_head('mkdir -p /shared')
        orch._orch.exec_on_head.assert_not_called()

    def test_failed_launch_skips_later_tests_and_leak_guard_tears_down(self):
        orch = MagicMock(orchestrator_type='container')
        orch.setup_containers.return_value = False
        patcher = patch.object(conftest.OrchestratorFactory, 'create_orchestrator', return_value=orch)
        patcher.start()
        self.addCleanup(patcher.stop)
        lifecycle = conftest.lifecycle.__wrapped__()
        gen = conftest.orch.__wrapped__(_cluster(['n1', 'n2']), lifecycle)

        with self.assertRaises(pytest.fail.Exception):
            conftest.launch_containers(next(gen), lifecycle)
        with self.assertRaises(pytest.skip.Exception):
            conftest.skip_after_failed_launch.__wrapped__(
                SimpleNamespace(node=SimpleNamespace(name='test_install_ib_perf')), lifecycle
            )
        conftest.skip_after_failed_launch.__wrapped__(
            SimpleNamespace(node=SimpleNamespace(name='test_teardown_containers')), lifecycle
        )
        with self.assertRaises(StopIteration):
            next(gen)
        orch.teardown_containers.assert_called_once_with()
        orch.close.assert_called_once_with()

    @patch('cvs.core.orchestrators.baremetal.MultiProcessParallelHandle')
    def test_baremetal_mode_skips_launch_and_runs_cmd_lists_on_the_host_handle(self, pssh):
        lifecycle = conftest.lifecycle.__wrapped__()
        orch = next(conftest.orch.__wrapped__(_cluster(['n1', 'n2']), lifecycle))
        with self.assertRaises(pytest.skip.Exception):
            conftest.launch_containers(orch, lifecycle)
        orch.exec_cmd_list(['a', 'b'])
        pssh.return_value.exec_cmd_list.assert_called_once_with(['a', 'b'], timeout=None, print_console=True)


class TestRunDeckFixtures(unittest.TestCase):
    def test_variant_reads_config_and_orch(self):
        config = {
            'ib_bw_test_list': ['ib_write_bw'],
            'msg_size_list': [2, 8192],
            'qp_count_list': ['8'],
            'duration': '10',
            'expected_results': {'ib_write_bw': {}},
            'verify_bw': 'True',
        }
        variant = conftest.ibperf_variant_config.__wrapped__(config, SimpleNamespace(hosts=['n1', 'n2']))
        self.assertEqual(variant['hosts'], ['n1', 'n2'])
        self.assertEqual(variant['node_count'], 2)
        self.assertEqual(variant['ib_bw_test_list'], ['ib_write_bw'])
        self.assertEqual(variant['ib_lat_test_list'], ibperf_lib.configured_lat_tests({}))
        self.assertEqual(variant['verify_bw'], 'True')


class TestFailedTestHook(unittest.TestCase):
    def _report(self, config, name, outcome, when='call', message=None):
        item = SimpleNamespace(config=config, name=name)
        call = SimpleNamespace(excinfo=SimpleNamespace(exconly=lambda: message) if message else None)
        report = SimpleNamespace(failed=outcome == 'failed', outcome=outcome, when=when)
        hook = conftest.pytest_runtest_makereport(item, call)
        next(hook)
        with self.assertRaises(StopIteration):
            hook.send(SimpleNamespace(get_result=lambda: report))

    def test_failures_before_the_results_fixture_exists_still_fail_the_deck(self):
        # A setup error in the first test (e.g. its orch fixture) comes before cvs_results_dict is created.
        config = SimpleNamespace(stash=pytest.Stash())
        self._report(config, 'test_ib_bw_perf[ib_write_bw]', 'failed', when='setup', message='KeyError: rocm_dir')
        results = conftest.cvs_results_dict.__wrapped__(SimpleNamespace(config=config))
        self._report(config, 'test_ib_lat_perf[ib_read_lat]', 'passed')
        self._report(config, 'test_build_ib_lat_perf_chart', 'failed', message='Failed: No latency results')
        self.assertEqual(
            results['_meta']['failed_tests']['rows'],
            [
                ['test_ib_bw_perf[ib_write_bw]', 'setup', 'KeyError: rocm_dir'],
                ['test_build_ib_lat_perf_chart', 'call', 'Failed: No latency results'],
            ],
        )
        payload = build_rundeck_payload(
            profile=load_json_profile('ib_perf_bw_test'), store={'cvs_results_dict': results}, cvs_version='1.0.0'
        )
        self.assertEqual(payload['overall_status'], 'fail')


if __name__ == '__main__':
    unittest.main()
