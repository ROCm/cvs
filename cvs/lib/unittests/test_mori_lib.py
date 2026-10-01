'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/lib/mori_lib.py: inline env from the config, command bounds
# (remote timeout beats the CVS timeout), baremetal safety, and the I/O
# launch/poll/kill flow against a recording fake orch.

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cvs.lib import globals
from cvs.lib.mori_lib import (
    IO_EXIT_MARKER,
    REMOTE_TIMEOUT_MARKER,
    MoriBenchmark,
    parse_pretty_tables_multi_rank,
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
        'mori_dir': '/sgl-workspace/mori',
        'env': {
            'NCCL_SOCKET_IFNAME': 'eno0',
            'GLOO_SOCKET_IFNAME': 'eno0',
            'MORI_RDMA_DEVICES': 'rdma0,rdma1',
            'LD_LIBRARY_PATH': '/torch/lib:$LD_LIBRARY_PATH',
        },
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


class _MoriBenchCase(unittest.TestCase):
    """Resets ``globals.error_list`` and patches out ``time.sleep`` around each test."""

    def setUp(self):
        globals.error_list = []
        self._sleep = patch('cvs.lib.mori_lib.time.sleep')
        self.sleep = self._sleep.start()

    def tearDown(self):
        self._sleep.stop()
        globals.error_list = []

    def _run_every_step(self, bench):
        bench.create_run_log_dir()
        bench.check_ibv_devices()
        bench.install_packages()
        bench.run_shmem_apitest()
        bench.run_concurrent_put_threads()
        bench.run_concurrent_put_imm_threads()
        bench.run_concurrent_put_signal_thread()
        bench.run_ibgda_dist_write()
        bench.run_mori_torch_io_test(op_type='read', buffer_size=16384, transfer_batch_size=128)
        # fail_test() doesn't raise, so without this a step that stopped passing would go unnoticed.
        self.assertEqual(globals.error_list, [])


class TestEnvPrefix(unittest.TestCase):
    def _run(self, prefix):
        env = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': '/pre', 'LD_LIBRARY_PATH': '/ld'}
        script = (
            prefix + 'printf "%s|%s|%s|%s" "$PYTHONPATH" "$LD_LIBRARY_PATH" "$MORI_RDMA_DEVICES" "$GLOO_SOCKET_IFNAME"'
        )
        return subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, check=True).stdout

    def _bench(self, mori_dir, env):
        return MoriBenchmark(_BaremetalFakeOrch(), _mori_dict(mori_dir=mori_dir, env=env))

    def test_config_env_expands_in_the_executing_shell(self):
        # $PYTHONPATH / $LD_LIBRARY_PATH must stay literal until the remote shell runs the command.
        env = {
            'LD_LIBRARY_PATH': '/torch/lib:$LD_LIBRARY_PATH',
            'MORI_RDMA_DEVICES': 'rdma0,rdma1',
            'GLOO_SOCKET_IFNAME': 'eno0',
        }
        prefix = self._bench('/opt/mori dir', env).env_prefix
        self.assertEqual(self._run(prefix), '/opt/mori dir:/pre|/torch/lib:/ld|rdma0,rdma1|eno0')

    def test_code_adds_only_pythonpath(self):
        prefix = self._bench('/m', {}).env_prefix
        self.assertEqual(self._run(prefix), '/m:/pre|/ld||')
        self.assertEqual(re.findall(r'export (\w+)=', prefix), ['PYTHONPATH'])

    def test_env_pythonpath_does_not_drop_mori_dir(self):
        prefix = self._bench('/m', {'PYTHONPATH': '/x:$PYTHONPATH'}).env_prefix
        self.assertEqual(self._run(prefix).split('|')[0], '/m:/x:/pre')

    def test_missing_env_block_still_builds_a_runnable_prefix(self):
        d = _mori_dict()
        del d['env']
        bench = MoriBenchmark(_BaremetalFakeOrch(), d)
        self.assertEqual(self._run(bench.env_prefix), '/sgl-workspace/mori:/pre|/ld||')


class TestParsePrettyTables(unittest.TestCase):
    def test_both_avg_bw_header_spellings_parse(self):
        for header in ('Avg Bw (GB/s)', 'Avg BW (GB/s)'):
            rows = parse_pretty_tables_multi_rank(_table(avg_bw_header=header))['ranks'][0]['rows']
            self.assertEqual(rows[0]['Avg_BW_GBps'], 44.0)
            self.assertEqual(rows[0]['MsgSize_B'], 524288)

    def test_unknown_header_still_raises(self):
        with self.assertRaises(KeyError):
            parse_pretty_tables_multi_rank(_table(avg_bw_header='Mean BW (GB/s)'))


class TestMoriBenchmarkCommands(_MoriBenchCase):
    def test_no_command_uses_docker_in_either_mode(self):
        # Container mode must go through orch.exec (which wraps docker exec itself), never a hand-built wrapper.
        for orch in (_ContainerFakeOrch(), _BaremetalFakeOrch()):
            self._run_every_step(MoriBenchmark(orch, _mori_dict()))
            for cmd in orch.all_commands():
                self.assertNotIn('docker', cmd)
                self.assertNotIn('mori_env_script', cmd)

    def test_baremetal_never_installs_packages_or_touches_host_ibverbs(self):
        orch = _BaremetalFakeOrch()
        bench = MoriBenchmark(orch, _mori_dict(nic_type='thor2'))
        self.assertFalse(bench.install_packages())
        self.assertEqual(orch.exec_calls, [])
        self._run_every_step(bench)
        for cmd in orch.all_commands():
            self.assertNotIn('pip', cmd)
            self.assertNotIn('libbnxt', cmd)
            self.assertNotIn('sudo', cmd)

    def test_container_installs_prettytable_inside_container(self):
        orch = _ContainerFakeOrch()
        self.assertTrue(MoriBenchmark(orch, _mori_dict()).install_packages())
        self.assertEqual(len(orch.exec_calls), 1)
        self.assertIn('pip3 install prettytable', orch.exec_calls[0]['cmd'])

    def test_every_exec_has_a_timeout(self):
        orch = _ContainerFakeOrch()
        self._run_every_step(MoriBenchmark(orch, _mori_dict()))
        self.assertTrue(orch.exec_calls)
        for call in orch.exec_calls + orch.cmd_list_calls:
            self.assertIsNotNone(call['timeout'], call)

    def test_benchmark_commands_carry_env_cd_and_per_host_log(self):
        orch = _ContainerFakeOrch()
        bench = MoriBenchmark(orch, _mori_dict())
        bench.run_shmem_apitest()
        cmd = orch.exec_calls[0]['cmd']
        self.assertTrue(cmd.startswith(bench.env_prefix))
        self.assertIn('cd /sgl-workspace/mori && ', cmd)
        self.assertIn('pytest -vvv ./tests/python/shmem/test_api.py', cmd)
        # Per-host log names keep hosts from overwriting each other on a shared log_dir.
        self.assertTrue(cmd.endswith(f' 2>&1 | tee {bench.run_log_dir}/shmem_api_$(hostname).log'))
        self.assertTrue(bench.run_log_dir.startswith('/home/u/LOGS/mori/'))

    def test_remote_bound_expires_before_cvs_timeout(self):
        # A CVS-side timeout alone leaves the remote command running (seen with the baremetal shmem pytest).
        orch = _ContainerFakeOrch()
        self._run_every_step(MoriBenchmark(orch, _mori_dict()))
        bounded = {}
        for call in orch.exec_calls:
            m = re.search(r'timeout -k (\d+) (\d+) bash -c', call['cmd'])
            if m:
                kill_after, remote = int(m.group(1)), int(m.group(2))
                self.assertLess(remote + kill_after, call['timeout'], call['cmd'][:120])
                bounded[call['cmd']] = remote
        for name in (
            'shmem/test_api.py',
            'concurrent_put_thread',
            'concurrent_put_imm_thread',
            'concurrent_put_signal_thread',
            'dist_write',
            'pip3 install prettytable',
        ):
            self.assertTrue(any(name in cmd for cmd in bounded), name)

    def test_timeout_not_above_margin_is_rejected(self):
        # Otherwise the remote bound could outlive the CVS-side timeout it exists to beat.
        bench = MoriBenchmark(_ContainerFakeOrch(), _mori_dict())
        for bad in (0, 5, MoriBenchmark.REMOTE_TIMEOUT_MARGIN):
            with self.assertRaises(ValueError):
                bench._bounded('true', bad)

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

    def test_unset_mori_rdma_devices_fails_without_running_ibv_devinfo(self):
        # An empty device list would otherwise pass the check vacuously.
        orch = _ContainerFakeOrch()
        MoriBenchmark(orch, _mori_dict(env={'NCCL_SOCKET_IFNAME': 'eno0'})).check_ibv_devices()
        self.assertEqual(len(globals.error_list), 1)
        self.assertIn('MORI_RDMA_DEVICES', globals.error_list[0])
        self.assertEqual(orch.exec_calls, [])


class TestRemoteBoundInBash(unittest.TestCase):
    """Runs the generated command in a real bash to prove the bound kills the whole tree."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, 'mori'))
        self.bench = MoriBenchmark(
            _BaremetalFakeOrch(),
            _mori_dict(mori_dir=os.path.join(self.tmp, 'mori'), log_dir=os.path.join(self.tmp, 'logs'), env={}),
        )
        os.makedirs(self.bench.run_log_dir)
        self.tag = f'{os.getpid()}{id(self) % 100000}'

    def tearDown(self):
        subprocess.run(['pkill', '-9', '-f', f'[s]leep 300.{self.tag}'], check=False)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, inner, timeout):
        cmd = self.bench._mori_cmd(inner, timeout=timeout, log_name='probe')
        return subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, check=False).stdout

    def _alive(self):
        out = subprocess.run(['pgrep', '-f', f'[s]leep 300.{self.tag}'], capture_output=True, text=True).stdout
        return out.split()

    def test_expired_bound_kills_children_and_keeps_output(self):
        # A child and a detached grandchild in the same process group, like pytest's spawned ranks.
        # Their fds are redirected so a regression fails fast instead of holding the tee open.
        inner = f'echo started; (sleep 300.{self.tag} >/dev/null 2>&1 &) ; sleep 300.{self.tag} >/dev/null 2>&1'
        margin = MoriBenchmark.REMOTE_TIMEOUT_MARGIN
        out = self._run(inner, timeout=margin + 1)
        self.assertIn('started', out)
        self.assertIn(f'{REMOTE_TIMEOUT_MARKER}1s rc=124', out)
        self.assertEqual(self._alive(), [])
        logs = os.listdir(self.bench.run_log_dir)
        self.assertEqual(len(logs), 1)
        with open(os.path.join(self.bench.run_log_dir, logs[0])) as fp:
            self.assertIn(REMOTE_TIMEOUT_MARKER, fp.read())

    def test_command_within_bound_is_unmarked(self):
        out = self._run('echo PASSED', timeout=600)
        self.assertIn('PASSED', out)
        self.assertNotIn(REMOTE_TIMEOUT_MARKER, out)


class TestMoriIoLaunch(_MoriBenchCase):
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
        self.assertIn(f"echo {IO_EXIT_MARKER}$?", cmd)

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
