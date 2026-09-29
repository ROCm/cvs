'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import copy
import re
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import OrchestratorConfig
from cvs.lib import globals
from cvs.lib.mori_lib import (
    IO_EXIT_MARKER,
    MoriBenchmark,
    build_env_prefix,
    build_orch_testsuite_config,
    legacy_container_block,
    parse_pretty_tables_multi_rank,
    resolve_orchestrator_mode,
)


def _table(avg_bw_header='Avg Bw (GB/s)', avg_bw='44.00', avg_lat='1525.20'):
    sep = '+-------------+-----------+----------------+---------------+---------------+--------------+--------------+'
    return '\n'.join(
        [
            'RDMA Benchmark: Initiator Rank 0',
            sep,
            f'| MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | {avg_bw_header} | Min Lat (us) | Avg Lat (us) |',
            sep,
            f'|    524288   |    128    |     67.11      |     48.00     |     {avg_bw}     |   1398.10    |   {avg_lat}    |',
            sep,
        ]
    )


def _mori_dict(**overrides):
    d = {
        'master_addr': '',
        'master_port': '1234',
        'oob_port': 'eno0',
        'torchlib_dir': '/torch/lib',
        'mori_dir': '/sgl-workspace/mori',
        'mori_device_list': 'rdma0,rdma1',
        'nic_type': 'thor2',
        'log_dir': '/home/u/LOGS/mori',
        'expected_results': {'ibgda_write': {}, 'io_read': {}, 'io_write': {}},
    }
    d.update(overrides)
    return d


def _default_responder(cmd, host):
    if 'ibv_devinfo' in cmd:
        return 'hca_id: rdma0\nhca_id: rdma1\n'
    if IO_EXIT_MARKER in cmd and 'grep' in cmd:
        return f'{IO_EXIT_MARKER}0'
    if cmd.startswith('cat '):
        return _table()
    if 'shmem/test_api.py' in cmd:
        return '5 PASSED'
    if 'concurrent_put' in cmd:
        return 'PASSED'
    if 'dist_write' in cmd:
        return 'Index Size(B) bw(GB) Time(ms) Rate(Mpps)\n0 33554432 46.10 0.73 0.01'
    return ''


class _BaremetalFakeOrch:
    """Records commands. Like BaremetalOrchestrator it has no ``exec_cmd_list``."""

    orchestrator_type = 'baremetal'

    def __init__(self, hosts=('n0', 'n1'), responder=None, exit_code=0):
        self.hosts = list(hosts)
        self.head_node = self.hosts[0]
        self.responder = responder or _default_responder
        self.exit_code = exit_code
        self.exec_calls = []
        self.cmd_list_calls = []
        self.all = MagicMock()
        self.all.exec_cmd_list.side_effect = self._run_cmd_list

    def exec(self, cmd, hosts=None, timeout=None, detailed=False, print_console=True):
        self.exec_calls.append({'cmd': cmd, 'hosts': hosts, 'timeout': timeout})
        out = {}
        for host in hosts or self.hosts:
            text = self.responder(cmd, host)
            out[host] = {'output': text, 'exit_code': self.exit_code} if detailed else text
        return out

    def _run_cmd_list(self, cmd_list, timeout=None, print_console=True):
        self.cmd_list_calls.append({'cmds': list(cmd_list), 'timeout': timeout})
        return {host: self.responder(cmd, host) for host, cmd in zip(self.hosts, cmd_list)}

    def all_commands(self):
        cmds = [c['cmd'] for c in self.exec_calls]
        for call in self.cmd_list_calls:
            cmds.extend(call['cmds'])
        return cmds


class _ContainerFakeOrch(_BaremetalFakeOrch):
    orchestrator_type = 'container'

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        return self._run_cmd_list(cmd_list, timeout=timeout, print_console=print_console)


def _run_every_step(bench):
    bench.create_run_log_dir()
    bench.check_ibv_devices()
    bench.install_packages()
    bench.run_shmem_apitest()
    bench.run_concurrent_put_threads()
    bench.run_concurrent_put_imm_threads()
    bench.run_concurrent_put_signal_thread()
    bench.run_ibgda_dist_write()
    bench.run_mori_torch_io_test(op_type='read', buffer_size=16384, transfer_batch_size=128)


class TestResolveOrchestratorMode(unittest.TestCase):
    def test_mori_config_orchestrator_wins_over_legacy_and_cluster(self):
        mori = {'orchestrator': 'baremetal', 'container_image': 'img'}
        self.assertEqual(resolve_orchestrator_mode(mori, {'orchestrator': 'container'}), 'baremetal')

    def test_legacy_config_forces_container_on_spur_style_baremetal_cluster(self):
        # SPUR-generated cluster files say baremetal; a legacy mori config must still launch its container.
        self.assertEqual(
            resolve_orchestrator_mode({'container_image': 'img'}, {'orchestrator': 'baremetal'}), 'container'
        )

    def test_explicit_container_orchestrator_on_baremetal_cluster(self):
        mori = {'orchestrator': 'container', 'container': {'image': 'img'}}
        self.assertEqual(resolve_orchestrator_mode(mori, {'orchestrator': 'baremetal'}), 'container')

    def test_new_style_container_block_defers_to_cluster(self):
        mori = {'container': {'image': 'img'}, 'container_image': 'img'}
        self.assertEqual(resolve_orchestrator_mode(mori, {'orchestrator': 'baremetal'}), 'baremetal')
        self.assertEqual(resolve_orchestrator_mode(mori, {'orchestrator': 'container'}), 'container')

    def test_empty_container_image_is_not_legacy(self):
        self.assertEqual(resolve_orchestrator_mode({'container_image': ''}, {'orchestrator': 'baremetal'}), 'baremetal')

    def test_default_is_baremetal(self):
        self.assertEqual(resolve_orchestrator_mode({}, {}), 'baremetal')


class TestLegacyContainerBlock(unittest.TestCase):
    def test_translates_every_legacy_key(self):
        mori = {
            'container_image': 'rocm/mori:x',
            'container_name': 'mori_container',
            'container_config': {
                'device_list': ['/dev/dri', '/dev/kfd'],
                'volume_dict': {'/home/u': '/home/u', '/a.so': '/b.so:ro'},
                'env_dict': {'FOO': '1'},
            },
        }
        self.assertEqual(
            legacy_container_block(mori),
            {
                'image': 'rocm/mori:x',
                'name': 'mori_container',
                'runtime': {
                    'args': {'devices': ['/dev/dri', '/dev/kfd'], 'volumes': ['/home/u:/home/u', '/a.so:/b.so:ro']}
                },
                'env': {'FOO': '1'},
            },
        )

    def test_empty_config_yields_empty_block(self):
        self.assertEqual(legacy_container_block({'container_config': {'volume_dict': {}, 'env_dict': {}}}), {})


class TestBuildOrchTestsuiteConfig(unittest.TestCase):
    def setUp(self):
        self.cluster = {
            'orchestrator': 'baremetal',
            'node_dict': {'n0': {}, 'n1': {}},
            'username': 'u',
            'priv_key_file': '/k',
            'container': {'image': 'cluster/img', 'runtime': {'name': 'docker', 'args': {'ipc': 'host'}}},
        }

    def test_baremetal_passes_empty_container_block(self):
        # A cluster-level container block must not be validated/used for a baremetal run.
        cfg = build_orch_testsuite_config({'orchestrator': 'baremetal', 'container_image': 'x'}, self.cluster)
        self.assertEqual(cfg, {'orchestrator': 'baremetal', 'container': {}})

    def test_precedence_cluster_then_legacy_then_mori_block(self):
        mori = {'container_image': 'legacy/img', 'container_name': 'mori_container'}
        cfg = build_orch_testsuite_config(mori, self.cluster)
        self.assertEqual(cfg['orchestrator'], 'container')
        self.assertEqual(cfg['container']['image'], 'legacy/img')
        self.assertEqual(cfg['container']['runtime'], {'name': 'docker', 'args': {'ipc': 'host'}})

        mori['orchestrator'] = 'container'
        mori['container'] = {'image': 'block/img'}
        cfg = build_orch_testsuite_config(mori, self.cluster)
        self.assertEqual(cfg['container']['image'], 'block/img')
        self.assertEqual(cfg['container']['name'], 'mori_container')

    def test_devices_deduplicated_against_orch_defaults(self):
        mori = {
            'container_image': 'img',
            'container_config': {'device_list': ['/dev/dri', '/dev/kfd', '/dev/foo', '/dev/foo']},
        }
        cfg = build_orch_testsuite_config(mori, self.cluster)
        self.assertEqual(cfg['container']['runtime']['args']['devices'], ['/dev/foo'])

    def test_inputs_not_mutated(self):
        mori = {
            'orchestrator': 'container',
            'container': {'image': 'i', 'runtime': {'args': {'devices': ['/dev/kfd']}}},
        }
        mori_before = copy.deepcopy(mori)
        cluster_before = copy.deepcopy(self.cluster)
        build_orch_testsuite_config(mori, self.cluster)
        self.assertEqual(mori, mori_before)
        self.assertEqual(self.cluster, cluster_before)

    def test_result_is_accepted_by_orchestrator_config(self):
        cfg = OrchestratorConfig.from_configs(
            self.cluster, build_orch_testsuite_config({'container_image': 'img'}, self.cluster)
        )
        self.assertEqual(cfg.orchestrator, 'container')
        self.assertEqual(cfg.container['image'], 'img')
        self.assertEqual(cfg.container['lifetime'], 'per_run')


class TestBuildEnvPrefix(unittest.TestCase):
    def _run(self, prefix):
        env = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': '/pre', 'LD_LIBRARY_PATH': '/ld'}
        script = (
            prefix + 'printf "%s|%s|%s|%s" "$PYTHONPATH" "$LD_LIBRARY_PATH" "$MORI_RDMA_DEVICES" "$GLOO_SOCKET_IFNAME"'
        )
        return subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, check=True).stdout

    def test_path_vars_expand_in_the_executing_shell(self):
        # The legacy docker-exec quoting let the host shell expand $PYTHONPATH; it must stay literal until run.
        prefix = build_env_prefix('/opt/mori dir', '/torch/lib', 'eno0', 'rdma0,rdma1')
        self.assertIn('$PYTHONPATH', prefix)
        self.assertEqual(self._run(prefix), '/opt/mori dir:/pre|/torch/lib:/ld|rdma0,rdma1|eno0')

    def test_empty_torchlib_dir_leaves_ld_library_path_alone(self):
        prefix = build_env_prefix('/m', '', 'eno0', 'rdma0')
        self.assertNotIn('LD_LIBRARY_PATH', prefix)
        self.assertEqual(self._run(prefix), '/m:/pre|/ld|rdma0|eno0')


class TestParsePrettyTables(unittest.TestCase):
    def test_both_avg_bw_header_spellings_parse(self):
        for header in ('Avg Bw (GB/s)', 'Avg BW (GB/s)'):
            rows = parse_pretty_tables_multi_rank(_table(avg_bw_header=header))['ranks'][0]['rows']
            self.assertEqual(rows[0]['Avg_BW_GBps'], 44.0)
            self.assertEqual(rows[0]['MsgSize_B'], 524288)

    def test_unknown_header_still_raises(self):
        with self.assertRaises(KeyError):
            parse_pretty_tables_multi_rank(_table(avg_bw_header='Mean BW (GB/s)'))


class TestMoriBenchmarkCommands(unittest.TestCase):
    def setUp(self):
        globals.error_list = []
        self._sleep = patch('cvs.lib.mori_lib.time.sleep')
        self._sleep.start()

    def tearDown(self):
        self._sleep.stop()
        globals.error_list = []

    def test_no_command_uses_docker_in_either_mode(self):
        # Container mode must go through orch.exec (which wraps docker exec itself), never a hand-built wrapper.
        for orch in (_ContainerFakeOrch(), _BaremetalFakeOrch()):
            _run_every_step(MoriBenchmark(orch, _mori_dict()))
            for cmd in orch.all_commands():
                self.assertNotIn('docker', cmd)
                self.assertNotIn('mori_env_script', cmd)

    def test_baremetal_never_installs_packages_or_touches_host_ibverbs(self):
        orch = _BaremetalFakeOrch()
        bench = MoriBenchmark(orch, _mori_dict(nic_type='thor2'))
        self.assertFalse(bench.install_packages())
        self.assertEqual(orch.exec_calls, [])
        _run_every_step(bench)
        for cmd in orch.all_commands():
            self.assertNotIn('pip', cmd)
            self.assertNotIn('libbnxt', cmd)
            self.assertNotIn('sudo', cmd)

    def test_container_installs_prettytable_inside_container(self):
        orch = _ContainerFakeOrch()
        self.assertTrue(MoriBenchmark(orch, _mori_dict()).install_packages())
        self.assertEqual([c['cmd'] for c in orch.exec_calls], ['pip3 install prettytable'])

    def test_every_exec_has_a_timeout(self):
        orch = _ContainerFakeOrch()
        _run_every_step(MoriBenchmark(orch, _mori_dict()))
        self.assertTrue(orch.exec_calls)
        for call in orch.exec_calls + orch.cmd_list_calls:
            self.assertIsNotNone(call['timeout'], call)

    def test_benchmark_commands_carry_env_cd_and_per_host_log(self):
        orch = _ContainerFakeOrch()
        bench = MoriBenchmark(orch, _mori_dict())
        bench.run_shmem_apitest()
        cmd = orch.exec_calls[0]['cmd']
        self.assertTrue(cmd.startswith(bench.env_prefix))
        self.assertIn('cd /sgl-workspace/mori && pytest -vvv ./tests/python/shmem/test_api.py', cmd)
        self.assertIn(f'| tee {bench.run_log_dir}/shmem_api_$(hostname).log', cmd)
        self.assertTrue(bench.run_log_dir.startswith('/home/u/LOGS/mori/'))

    def test_log_dir_mkdir_failure_is_reported_per_node(self):
        orch = _ContainerFakeOrch(exit_code=1)
        MoriBenchmark(orch, _mori_dict()).create_run_log_dir()
        self.assertEqual(len(globals.error_list), 2)
        self.assertIn('node n0', globals.error_list[0])

    def test_missing_ibv_device_fails_with_node(self):
        orch = _ContainerFakeOrch(
            responder=lambda cmd, host: 'hca_id: rdma0' if host == 'n1' else 'hca_id: rdma0 rdma1'
        )
        MoriBenchmark(orch, _mori_dict()).check_ibv_devices()
        self.assertEqual(len(globals.error_list), 1)
        self.assertIn('rdma1', globals.error_list[0])
        self.assertIn('node n1', globals.error_list[0])


class TestMoriIoLaunch(unittest.TestCase):
    def setUp(self):
        globals.error_list = []
        self._sleep = patch('cvs.lib.mori_lib.time.sleep')
        self.sleep = self._sleep.start()

    def tearDown(self):
        self._sleep.stop()
        globals.error_list = []

    def test_master_addr_defaults_to_head_node(self):
        bench = MoriBenchmark(_ContainerFakeOrch(hosts=['a', 'b']), _mori_dict(master_addr=''))
        self.assertEqual(bench.master_addr, 'a')
        self.assertEqual(bench.io_rank_hosts(), ['a', 'b'])

    def test_launch_cmds_in_host_order_with_master_as_rank_zero(self):
        # exec_cmd_list maps by position, so cmd i must target hosts[i] while master gets node_rank 0.
        bench = MoriBenchmark(_ContainerFakeOrch(hosts=['a', 'b']), _mori_dict(master_addr='b'))
        cmd_list, log_paths = bench.build_io_launch_cmds('io_read_1_2_3', '--x')
        self.assertEqual(len(cmd_list), 2)
        self.assertIn('--node_rank=1', cmd_list[0])
        self.assertIn('--host=a', cmd_list[0])
        self.assertIn('--node_rank=0', cmd_list[1])
        self.assertIn('--host=b', cmd_list[1])
        for cmd in cmd_list:
            self.assertIn('--nnodes=2', cmd)
            self.assertIn('--master_addr=b --master_port=1234', cmd)
        self.assertTrue(log_paths['a'].endswith('/io_read_1_2_3_node1.log'))
        self.assertTrue(log_paths['b'].endswith('/io_read_1_2_3_node0.log'))

    def test_launch_cmd_detaches_and_records_exit_code(self):
        bench = MoriBenchmark(_ContainerFakeOrch(), _mori_dict())
        cmd_list, log_paths = bench.build_io_launch_cmds('case', '--x')
        cmd = cmd_list[0]
        self.assertIn(f'> {log_paths["n0"]} 2>&1 < /dev/null &', cmd)
        self.assertTrue(cmd.rstrip().endswith('&'))
        self.assertIn(f"echo {IO_EXIT_MARKER}$?", cmd)
        self.assertIn('nohup bash -c', cmd)

    def test_master_addr_outside_hosts_fails_without_launching(self):
        orch = _ContainerFakeOrch()
        MoriBenchmark(orch, _mori_dict(master_addr='10.0.0.9')).run_mori_torch_io_test()
        self.assertEqual(len(globals.error_list), 1)
        self.assertIn('10.0.0.9', globals.error_list[0])
        self.assertEqual(orch.all_commands(), [])

    def test_baremetal_uses_host_handle_for_cmd_list(self):
        orch = _BaremetalFakeOrch()
        MoriBenchmark(orch, _mori_dict()).run_mori_torch_io_test()
        self.assertTrue(orch.all.exec_cmd_list.called)
        self.assertEqual(globals.error_list, [])

    def test_poll_waits_until_every_host_reports(self):
        polls = {'n': 0}

        def responder(cmd, host):
            if host == 'n0':
                return f'{IO_EXIT_MARKER}0'
            polls['n'] += 1
            return f'{IO_EXIT_MARKER}3' if polls['n'] >= 2 else ''

        bench = MoriBenchmark(_ContainerFakeOrch(responder=responder), _mori_dict())
        codes = bench.poll_io_completion({'n0': '/l0', 'n1': '/l1'}, timeout=600, interval=7)
        self.assertEqual(codes, {'n0': 0, 'n1': 3})
        self.sleep.assert_called_once_with(7)

    def test_poll_times_out_with_partial_result(self):
        responder = lambda cmd, host: f'{IO_EXIT_MARKER}0' if host == 'n0' else ''  # noqa: E731
        bench = MoriBenchmark(_ContainerFakeOrch(responder=responder), _mori_dict())
        with patch('cvs.lib.mori_lib.time.monotonic', side_effect=[0, 5, 700]):
            codes = bench.poll_io_completion({'n0': '/l0', 'n1': '/l1'}, timeout=600, interval=1)
        self.assertEqual(codes, {'n0': 0})
        self.assertEqual(self.sleep.call_count, 1)

    def test_unfinished_host_fails_and_processes_are_killed(self):
        responder = lambda cmd, host: '' if IO_EXIT_MARKER in cmd else _default_responder(cmd, host)  # noqa: E731
        orch = _ContainerFakeOrch(responder=responder)
        with patch.object(MoriBenchmark, 'IO_POLL_TIMEOUT', 0):
            MoriBenchmark(orch, _mori_dict()).run_mori_torch_io_test()
        self.assertTrue(any('did not finish' in e and 'n0' in e for e in globals.error_list))
        self.assertTrue(any('pkill' in c for c in orch.all_commands()))

    def test_processes_killed_even_if_poll_raises(self):
        orch = _ContainerFakeOrch()
        bench = MoriBenchmark(orch, _mori_dict())
        with patch.object(MoriBenchmark, 'poll_io_completion', side_effect=RuntimeError('ssh lost')):
            with self.assertRaises(RuntimeError):
                bench.run_mori_torch_io_test()
        self.assertIn('pkill', orch.exec_calls[-1]['cmd'])

    def test_nonzero_exit_code_fails(self):
        def responder(cmd, host):
            if IO_EXIT_MARKER in cmd:
                return f'{IO_EXIT_MARKER}1' if host == 'n1' else f'{IO_EXIT_MARKER}0'
            return _default_responder(cmd, host)

        MoriBenchmark(_ContainerFakeOrch(responder=responder), _mori_dict()).run_mori_torch_io_test()
        self.assertEqual(len(globals.error_list), 1)
        self.assertIn('exited with code 1 on node n1', globals.error_list[0])

    def test_results_read_from_master_log_and_gated(self):
        # Only the master's (rank 0) log has the table; reading any other host's log must fail.
        def responder(cmd, host):
            if cmd.startswith('cat '):
                return _table(avg_bw='44.00', avg_lat='1525.20') if host == 'n1' else ''
            return _default_responder(cmd, host)

        expected = {
            'ibgda_write': {},
            'io_read': {
                'BUFF_SIZE:16384,TRANSFER_SIZE:128,QP_COUNT:1': {'524288': {'max_bw': '45.0', 'avg_lat': '1500'}}
            },
            'io_write': {},
        }
        orch = _ContainerFakeOrch(responder=responder)
        MoriBenchmark(orch, _mori_dict(master_addr='n1', expected_results=expected)).run_mori_torch_io_test(
            op_type='read', buffer_size=16384, transfer_batch_size=128
        )
        cat_call = [c for c in orch.exec_calls if c['cmd'].startswith('cat ')][0]
        self.assertEqual(cat_call['hosts'], ['n1'])
        self.assertTrue(cat_call['cmd'].endswith('_node0.log'))
        self.assertEqual(len(globals.error_list), 2)
        self.assertIn('BW is lower than expected', globals.error_list[0])
        self.assertIn('Latency is higher than expected', globals.error_list[1])

    def test_kill_patterns_match_benchmark_but_not_the_kill_shell(self):
        # pkill -f would otherwise kill the bash running it (docker exec / ssh), losing the result.
        orch = _ContainerFakeOrch()
        MoriBenchmark(orch, _mori_dict()).kill_io_processes()
        cmd = orch.exec_calls[0]['cmd']
        patterns = re.findall(r"pkill -9 -f '([^']+)'", cmd)
        self.assertEqual(len(patterns), 2)
        benchmark_proc = (
            'python3 /usr/local/bin/torchrun --nnodes=2 --node_rank=0 --master_port=1234 '
            '/sgl-workspace/mori/tests/python/io/benchmark.py --host=n0 --all'
        )
        for pattern in patterns:
            self.assertIsNone(re.search(pattern, cmd), pattern)
            self.assertIsNotNone(re.search(pattern, benchmark_proc), pattern)


if __name__ == '__main__':
    unittest.main()
