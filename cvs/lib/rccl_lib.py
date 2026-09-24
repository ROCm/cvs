'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Standard libraries
import os
import re
import json
import shlex
import tempfile
from typing import List
from pathlib import Path

# Third party libraries
import pandas as pd
from pydantic import ValidationError

from cvs.core.scheduler import Scheduler, detect_scheduler, is_managed_compute, scheduler_hosts
from cvs.lib import globals
from cvs.schema.rccl import RcclTests, RcclTestsAggregated, RcclTestsMultinodeRaw
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

log = globals.log

TOPOLOGY_FIELDS = ('nodes', 'ranks', 'ranksPerNode', 'gpusPerRank')


def compare_topology(expected, observed):
    """Return requested-vs-reported mismatches without modifying either mapping."""
    return [
        f'{field}: requested {expected[field]}, reported {observed.get(field, "<missing>")}'
        for field in TOPOLOGY_FIELDS
        if field in expected and (field not in observed or expected[field] != observed[field])
    ]


def _result_model(row):
    """Allow rccl-tests output that omits the multinode topology block."""
    return RcclTestsMultinodeRaw if all(field in row for field in TOPOLOGY_FIELDS) else RcclTests


rccl_err_dict = {
    'orte': 'ORTE does not know how to route|ORTE was unable to reliably start',
    'nccl': 'NCCL ERROR|Test failure',
    'fs_err': 'No such file or directory',
}


class RcclVerifier:
    """Pass/fail checks on one RCCL result set. Log scan is separate (stdout)."""

    def __init__(self, test_name, results, expected=None, cvs_params=None):
        self.test_name = test_name
        self.results = results
        self.expected = expected
        self.cvs_params = cvs_params or {}

    @staticmethod
    def scan_logs(output):
        """Scan RCCL stdout for ORTE/NCCL/FS errors and require an avg-bus-bw marker."""
        error_list = []
        warn_list = []
        for line in output.split("\n"):
            for err_key in rccl_err_dict.keys():
                if re.search(f'{rccl_err_dict[err_key]}', line):
                    error_list.append(line)
                    fail_test(f'ERROR - {line}')
            if re.search('NCCL WARN', line):
                warn_list.append(line)
        if len(warn_list) > 0:
            log.warning('Following warnings were observed in the RCCL test')
            log.warning('#============#')
            log.warning('%s', warn_list)
            log.warning('#============#')
        if not re.search(r'#\sAvg bus bandwidth', output):
            fail_test('RCCL test did not complete successfully, no bandwidth numbers printed - pls check')

    def check_bus_bw(self):
        test_name = self.test_name
        expected = self.expected
        log.info(f'exp_res_dict = {expected}')
        tolerance = 0.95
        msg_size_list = list(expected.keys())
        log.info("%s", test_name)
        in_place = 0 if re.search('alltoall|all_to_all', test_name, re.I) else 1
        place_label = 'out-of-place' if in_place == 0 else 'in-place'
        for act_dict in self.results:
            if act_dict['inPlace'] != in_place:
                continue
            for msg_size in msg_size_list:
                if str(msg_size) != str(act_dict['size']):
                    continue
                expected_bw = float(expected[msg_size]['bus_bw'])
                actual_bw = float(act_dict['busBw'])
                threshold = expected_bw * tolerance
                log.info(f"Comparing: actual={actual_bw}, expected={expected_bw}, threshold={threshold:.2f}")
                if actual_bw < threshold:
                    fail_test(
                        f"The actual {place_label} bus BW {actual_bw} for msg size {act_dict['size']} is lower than expected bus BW {expected_bw} (threshold with 5% tolerance: {threshold:.2f})"
                    )

    def check_bw_dip(self):
        test_name = self.test_name
        expected = self.expected
        tolerance = 0.95
        if not expected:
            log.warning(f"No reference data provided for BW dip check, skipping validation for {test_name}")
            return
        ref_msg_sizes = set(str(size) for size in expected.keys())
        log.info(f"Validating BW dip only for reference message sizes: {ref_msg_sizes}")
        in_place = 0 if re.search('alltoall|all_to_all', test_name, re.I) else 1
        last_bw = 0.0
        last_msg_size = self.results[0]['size']
        for act_dict in self.results:
            if act_dict['inPlace'] != in_place:
                continue
            if str(act_dict['size']) not in ref_msg_sizes:
                continue
            current_bw = float(act_dict['busBw'])
            threshold = float(last_bw) * tolerance
            if last_bw > 0 and current_bw < threshold:
                fail_test(
                    f"The BusBW for msg size {act_dict['size']} = {current_bw} is less than the earlier msg size {last_msg_size} = BW {last_bw} (threshold with 5% tolerance: {threshold:.2f})"
                )
            last_bw = act_dict['busBw']
            last_msg_size = act_dict['size']

    def check_lat_dip(self):
        test_name = self.test_name
        expected = self.expected
        tolerance = 0.95
        if not expected:
            log.warning(f"No reference data provided for latency dip check, skipping validation for {test_name}")
            return
        ref_msg_sizes = set(str(size) for size in expected.keys())
        log.info(f"Validating latency dip only for reference message sizes: {ref_msg_sizes}")
        in_place = 0 if re.search('alltoall|all_to_all', test_name, re.I) else 1
        last_time = 0.0
        last_msg_size = self.results[0]['size']
        for act_dict in self.results:
            if act_dict['inPlace'] != in_place:
                continue
            if str(act_dict['size']) not in ref_msg_sizes:
                continue
            current_time = float(act_dict['time'])
            threshold = float(last_time) * tolerance
            if last_time > 0 and current_time < threshold:
                fail_test(
                    f"The latency for msg size {act_dict['size']} = {current_time} is less than the earlier msg size {last_msg_size} = latency {last_time} (threshold with 5% tolerance: {threshold:.2f})"
                )
            last_time = act_dict['time']
            last_msg_size = act_dict['size']

    def check(self):
        if re.search('True', self.cvs_params.get('verify_bus_bw', 'False'), re.I) and self.expected:
            self.check_bus_bw()
        if re.search('True', self.cvs_params.get('verify_bw_dip', 'True'), re.I):
            self.check_bw_dip()
        if re.search('True', self.cvs_params.get('verify_lat_dip', 'True'), re.I):
            self.check_lat_dip()

    @staticmethod
    def is_severe_wrong_corruption_error(err: ValidationError) -> bool:
        """Detect rccl-tests '#wrong' corruption from a pydantic ValidationError."""
        try:
            for item in err.errors():
                msg = item.get('msg', '') or ''
                if 'SEVERE DATA CORRUPTION' in msg or "'#wrong'" in msg or 'wrong=' in msg:
                    return True
        except Exception:
            pass
        s = str(err)
        return 'SEVERE DATA CORRUPTION' in s or "'#wrong'" in s


def convert_to_graph_dict(result_dict):
    graph_dict = {}
    for graph_series_name in result_dict.keys():
        log.info("%s", graph_series_name)
        graph_dict[graph_series_name] = {}
        dict_list = result_dict[graph_series_name]
        log.info("%s", dict_list)
        for dict_item in dict_list:
            msg_size = dict_item['size']
            graph_dict[graph_series_name][msg_size] = {}
            if re.search('alltoall', dict_item['name'], re.I) and dict_item['inPlace'] == 1:
                graph_dict[graph_series_name][msg_size]['bus_bw'] = dict_item['busBw']
                graph_dict[graph_series_name][msg_size]['alg_bw'] = dict_item['algBw']
                graph_dict[graph_series_name][msg_size]['time'] = dict_item['time']
            else:
                graph_dict[graph_series_name][msg_size]['bus_bw'] = dict_item['busBw']
                graph_dict[graph_series_name][msg_size]['alg_bw'] = dict_item['algBw']
                graph_dict[graph_series_name][msg_size]['time'] = dict_item['time']
    log.info("%s", graph_dict)
    return graph_dict


class OpenMPI:
    """Open MPI configuration consumed by rccl-tests during MPI_Init."""

    def __init__(self, mpi_params):
        self.mpi_params = mpi_params
        self.prefix = self._install_prefix(mpi_params.get('mpi_dir', '/usr/local'))
        self.oob_port = mpi_params.get('mpi_oob_port', 'eth0') or 'eth0'
        self.pml = 'ob1'
        self.ucx_env = {}
        self.prepared = False

    @staticmethod
    def _install_prefix(mpi_dir):
        path = os.path.normpath(str(mpi_dir or '/usr/local'))
        return os.path.dirname(path) if os.path.basename(path) == 'bin' else path

    def prepare(self, phdl, shdl, head_node):
        if self.prepared:
            return self
        self.pml, self.ucx_env = self._determine_pml_config(phdl, shdl, head_node)
        self.prepared = True
        return self

    def _ucx_available(self, shdl, head_node):
        check_ucx_cmd = f'{self.prefix}/bin/ompi_info | grep "pml: ucx" | wc -l'
        try:
            ucx_check_out = shdl.exec(check_ucx_cmd)
            ucx_available = int(ucx_check_out[head_node].strip()) > 0
            log.info("UCX available in OpenMPI build" if ucx_available else "UCX not available in OpenMPI build")
            return ucx_available
        except Exception as e:
            log.warning(f"ompi_info check failed: {e}; falling back to ldd check")

        libmpi_path = f'{self.prefix}/lib/libmpi.so'
        check_ldd_cmd = f'ldd {libmpi_path} | grep ucx | wc -l'
        try:
            ldd_out = shdl.exec(check_ldd_cmd)
            ucx_available = int(ldd_out[head_node].strip()) > 0
            log.info("UCX linked in libmpi.so" if ucx_available else "UCX not linked in libmpi.so")
            return ucx_available
        except Exception as e:
            log.warning(f"ldd check failed: {e}; assuming UCX is not available")
            return False

    def _determine_pml_config(self, phdl, shdl, head_node):
        """Choose PML from mpi_params.mpi_pml, falling back to ob1 if UCX is unusable."""
        mpi_pml = str(self.mpi_params.get('mpi_pml', 'auto')).lower()
        ucx_tls = self.mpi_params.get('ucx_tls', 'rc,self,sm,tcp') or 'rc,self,sm,tcp'

        if mpi_pml not in ('ucx', 'auto'):
            if mpi_pml == 'ob1':
                log.info('mpi_pml val in config is ob1')
            else:
                log.warning(f'mpi_pml val in config is {mpi_pml} (incorrect), falling back to pml ob1')
            log.info('PML: ob1  UCX params: ')
            return 'ob1', {}

        log.info(f'mpi_pml val in config is {mpi_pml}')
        if not self._ucx_available(shdl, head_node):
            log.warning('UCX not detected — falling back to pml ob1')
            log.info('PML: ob1  UCX params: ')
            return 'ob1', {}

        log.info('UCX detected in libmpi.so — using pml ucx')
        net_dev_list = self.mpi_params.get('net_dev_list', '')
        if not net_dev_list:
            log.warning("'net_dev_list' missing or empty — auto-detecting from backend NICs...")
            try:
                net_dev_list = linux_utils.get_ucx_net_devices(phdl)
            except ValueError as exc:
                log.error(f'UCX net device auto-detection failed: {exc}')
                log.warning('Falling back to pml ob1 due to auto-detection failure')
                log.info('PML: ob1  UCX params: ')
                return 'ob1', {}
        else:
            log.info(f'Using net_dev_list from mpi_params: {net_dev_list}')

        ucx_env = {
            'UCX_UNIFIED_MODE': 'y',
            'UCX_NET_DEVICES': net_dev_list,
            'UCX_TLS': ucx_tls,
        }
        log.info(f'PML: ucx  UCX params: {ucx_env}')
        return 'ucx', ucx_env

    def mpi_init_settings(self):
        """MCA and env consumed by MPI_Init. Launchers format this as --mca/-x or export."""
        return {
            'mca': {
                'btl': '^vader,openib',
                'btl_tcp_if_include': self.oob_port,
                'oob_tcp_if_include': self.oob_port,
                'pml': self.pml or 'ob1',
            },
            'env': dict(self.ucx_env),
        }


class MpiRun:
    """Launch rccl-tests through Open MPI's mpirun on bare metal."""

    def __init__(self, openmpi, shdl, cluster_nodes, vpc_nodes, global_ranks):
        self.openmpi = openmpi
        self.shdl = shdl
        self.cluster_nodes = cluster_nodes
        self.vpc_nodes = vpc_nodes
        self.global_ranks = global_ranks
        self.hosts_file_path = ''

    def topology(self):
        """Return the topology requested by -np and the uniform hostfile slots."""
        nodes = len(self.cluster_nodes)
        return {'nodes': nodes, 'ranks': self.global_ranks, 'ranksPerNode': self.global_ranks // nodes}

    def prepare(self):
        if self.hosts_file_path:
            return self
        slots = self.topology()['ranksPerNode']
        host_file = ''.join(f'{node} slots={slots}\n' for node in self.vpc_nodes)
        self.hosts_file_path = f'/tmp/rccl_hosts_file_{os.environ.get("USER", "cvs")}.txt'
        self.shdl.exec(f'rm -f {self.hosts_file_path}')
        self.shdl.exec(f'echo "{host_file}" > {self.hosts_file_path}')
        return self

    def cleanup(self):
        """No per-node session directory to remove; mpirun creates its own."""

    def command(self, binary_cmd, env_file=None, env_overrides=None):
        self.prepare()
        mpi_init = self.openmpi.mpi_init_settings()
        rank_cmds = []
        if env_file and str(env_file).lower() != 'none':
            rank_cmds.append(f'source {shlex.quote(str(env_file))}')
        if env_overrides:
            rank_cmds.extend(f'export {k}={shlex.quote(str(v))}' for k, v in env_overrides.items())
        rank_cmds.append(binary_cmd)
        rank_cmd = f'bash -c {shlex.quote(" && ".join(rank_cmds))}'
        mpirun_args = [
            f'{self.openmpi.prefix}/bin/mpirun',
            '--allow-run-as-root',
            f'-np {self.global_ranks}',
            f'--hostfile {self.hosts_file_path}',
            '--bind-to numa',
        ]
        mpirun_args.extend(f'-x {key}={value}' for key, value in mpi_init['env'].items())
        mpirun_args.extend(f'--mca {key} {value}' for key, value in mpi_init['mca'].items())
        mpirun_args.append(rank_cmd)
        return ' '.join(mpirun_args)


class Srun:
    """Launch a nested PMIx step through Slurm srun or Spur's spur run."""

    def __init__(self, openmpi, phdl, cluster_nodes, no_of_nodes, local_ranks, global_ranks):
        self.openmpi = openmpi
        self.phdl = phdl
        self.cluster_nodes = cluster_nodes
        self.no_of_nodes = no_of_nodes
        self.local_ranks = local_ranks
        self.global_ranks = global_ranks
        self.session_dir = ''

    def topology(self):
        """Return the topology requested by -N, -n and --ntasks-per-node."""
        return {'nodes': self.no_of_nodes, 'ranks': self.global_ranks, 'ranksPerNode': self.local_ranks}

    def prepare(self):
        """Create the ORTE session directory once per node.

        Ranks run with orte_create_session_dirs=0, so this path must exist before
        MPI_Init. PMIx's own rendezvous directory is separate and slurmstepd
        creates it. Resolving the path here instead of in the rank shell keeps
        every rank on a node in agreement even when SLURM_JOB_ID is absent.
        """
        if self.session_dir:
            return self
        parent = f'/tmp/{os.environ.get("USER", "cvs")}'
        session_dir = f'{parent}/ompi-{os.environ.get("SLURM_JOB_ID") or f"cvs-{os.getpid()}"}'
        self.phdl.exec(
            f'mkdir -p -m 700 {shlex.quote(parent)} 2>/dev/null || true; '
            f'mkdir -p {shlex.quote(session_dir)} && chmod 700 {shlex.quote(session_dir)}'
        )
        self.session_dir = session_dir
        return self

    def cleanup(self):
        """Remove the session directory created by prepare()."""
        if not self.session_dir:
            return
        try:
            self.phdl.exec(f'rm -rf {shlex.quote(self.session_dir)}', timeout=30)
        except Exception as exc:
            log.warning(f'Session dir cleanup for {self.session_dir} raised {exc!r} (continuing)')
        self.session_dir = ''

    def command(self, binary_cmd, env_file=None, env_overrides=None):
        self.prepare()
        rank_cmds = list(self.mpi_init_exports())
        if env_file and str(env_file).lower() != 'none':
            rank_cmds.append(f'source {shlex.quote(str(env_file))}')
        if env_overrides:
            rank_cmds.extend(f'export {k}={shlex.quote(str(v))}' for k, v in env_overrides.items())
        rank_cmds.append(binary_cmd)
        rank_cmd = f'bash -c {shlex.quote(" && ".join(rank_cmds))}'
        launcher = 'spur run' if detect_scheduler() == Scheduler.SPUR else 'srun'
        srun_args = [
            launcher,
            '--overlap',
            '--mpi=pmix',
            f'-N {self.no_of_nodes}',
            f'-n {self.global_ranks}',
            f'--ntasks-per-node {self.local_ranks}',
            '--gpu-bind=none',
        ]
        cpus = self._cpus_per_nested_task(self.local_ranks)
        if cpus is not None:
            srun_args.append(f'-c {cpus}')
        nodelist_flag = self._nodelist_flag(self.cluster_nodes)
        if nodelist_flag:
            srun_args.append(nodelist_flag)
        srun_args.extend(['--', rank_cmd])
        return ' '.join(srun_args)

    def mpi_init_exports(self):
        """Rank env for a PMIx step: prefix/session, then MCA/UCX from OpenMPI.mpi_init_settings."""
        prefix = self.openmpi.prefix
        session_dir = shlex.quote(self.session_dir)
        commands = [
            'export PMIX_MCA_gds=hash',
            f'export OPAL_PREFIX={shlex.quote(prefix)}',
            f'export PATH={shlex.quote(prefix + "/bin")}:$PATH',
            'export LD_LIBRARY_PATH="${OPAL_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
            f'export OMPI_MCA_orte_tmpdir_base={session_dir}',
            f'export OMPI_MCA_orte_top_session_dir={session_dir}',
            'export OMPI_MCA_orte_create_session_dirs=0',
        ]
        settings = self.openmpi.mpi_init_settings()
        commands.extend(f'export OMPI_MCA_{key}={shlex.quote(str(value))}' for key, value in settings['mca'].items())
        commands.extend(f'export {key}={shlex.quote(str(value))}' for key, value in settings['env'].items())
        return commands

    @staticmethod
    def _cpus_per_nested_task(local_ranks):
        """CPU count per inner PMIx task. Matches rccl-sn-ar.sh: SLURM_CPUS_ON_NODE / local ranks."""
        override = os.environ.get('RCCL_CPUS_PER_TASK')
        if override:
            return max(1, int(override))
        cpus_on_node = os.environ.get('SLURM_CPUS_ON_NODE')
        if cpus_on_node:
            return max(1, int(cpus_on_node) // max(1, int(local_ranks)))
        return None

    @staticmethod
    def _nodelist_flag(cluster_node_list):
        """Slurm: -w for subset steps. Spur 0.11: omit -w on the full allocation, else raise."""
        if not cluster_node_list:
            return None
        if detect_scheduler() != Scheduler.SPUR:
            return f'-w {",".join(cluster_node_list)}'
        allocated = scheduler_hosts()
        if set(cluster_node_list) != set(allocated):
            raise RuntimeError(
                'SPUR 0.11 does not apply --nodelist to job steps; '
                f'cannot restrict RCCL to {list(cluster_node_list)} inside allocation {allocated}. '
                'Pairwise/incremental RCCL is not supported on this Spur version.'
            )
        return None


class RcclJob:
    """One rccl-tests job composed from OpenMPI and a rank launcher.

    OpenMPI owns the settings consumed by MPI_Init. MpiRun or Srun owns rank
    creation. RcclJob owns the rccl-tests binary, NCCL environment, execution,
    and result collection. RcclVerifier owns pass/fail checks.
    """

    def __init__(
        self,
        phdl,
        shdl,
        test_name,
        env_file,
        mpi_params,
        rccl_test_params,
        cvs_params,
        cluster_node_list,
        vpc_node_list,
        env_overrides=None,
    ):
        if not cluster_node_list:
            raise ValueError('cluster_node_list must contain at least one node')

        self.phdl = phdl
        self.shdl = shdl
        self.test_name = test_name
        self.env_file = env_file
        self.mpi_params = mpi_params
        self.rccl_test_params = rccl_test_params
        self.cvs_params = cvs_params
        self.cluster_node_list = cluster_node_list
        self.vpc_node_list = vpc_node_list
        self.env_overrides = env_overrides

        self.no_of_nodes = int(mpi_params.get('no_of_nodes', 2))
        self.no_of_local_ranks = int(mpi_params.get('no_of_local_ranks', 8))
        self.no_of_global_ranks = self.no_of_nodes * self.no_of_local_ranks
        self.head_node = cluster_node_list[0]
        self.rccl_tests_dir = rccl_test_params.get('rccl_tests_dir', '/usr/local/rccl-tests/build')
        self.cvs_exec_timeout = int(cvs_params.get('cvs_exec_timeout', 2400))
        self.managed = is_managed_compute()
        self.openmpi = OpenMPI(mpi_params)
        if self.managed:
            self.launcher = Srun(
                self.openmpi,
                phdl,
                cluster_node_list,
                self.no_of_nodes,
                self.no_of_local_ranks,
                self.no_of_global_ranks,
            )
        else:
            self.launcher = MpiRun(
                self.openmpi,
                shdl,
                cluster_node_list,
                vpc_node_list,
                self.no_of_global_ranks,
            )
        self.output_flag = ''
        self.prepared = False
        self.topology_checks = []
        self.topology_check_mode = str(cvs_params.get('topology_check', 'warn')).strip().lower()
        if self.topology_check_mode not in ('off', 'warn', 'strict'):
            log.warning('Unrecognised topology_check mode %r; falling back to warn', self.topology_check_mode)
            self.topology_check_mode = 'warn'

    @classmethod
    def from_config(
        cls,
        phdl,
        shdl,
        test_name,
        config_dict,
        cluster_node_list,
        vpc_node_list,
        *,
        env_overrides=None,
    ):
        """Build a job directly from the grouped RCCL configuration."""
        return cls(
            phdl=phdl,
            shdl=shdl,
            test_name=test_name,
            env_file=config_dict.get('env_source_script', '/dev/null'),
            mpi_params=config_dict['mpi_params'],
            rccl_test_params=config_dict['rccl_test_params'],
            cvs_params=config_dict['cvs_params'],
            cluster_node_list=cluster_node_list,
            vpc_node_list=vpc_node_list,
            env_overrides=env_overrides,
        )

    def prepare(self):
        """Prepare OpenMPI, launcher resources, and output support once."""
        if self.prepared:
            return self

        self._require_spur_job_step()
        log.info(f'Starting RCCL Test ..........................................{self.test_name}')
        log.info(f'%% VPC Node IPs {self.vpc_node_list}')

        self.openmpi.prepare(self.phdl, self.shdl, self.head_node)
        self.launcher.prepare()

        binary_path = f'{self.rccl_tests_dir}/{self.test_name}'
        self.output_flag = self._detect_output_flag(binary_path)
        self.prepared = True
        return self

    @staticmethod
    def _require_spur_job_step():
        """Prevent a bare Spur allocation from silently falling back to SSH/mpirun."""
        in_job = os.environ.get('SPUR_JOB_ID') or os.environ.get('SLURM_JOB_ID')
        if detect_scheduler() == Scheduler.SPUR and in_job and not is_managed_compute():
            raise RuntimeError(
                'CVS RCCL requires a SPUR job step. '
                'Launch with `spur run --mpi=none` (one task per node). '
                'An allocation shell alone does not start the per-node CVS agents.'
            )

    def test_flags(self):
        """Flags shared by perf and regression rccl-tests commands."""
        flags = ''
        rccl_timeout = self.rccl_test_params.get('rccl_timeout')
        if rccl_timeout is not None:
            flags += f' -T {rccl_timeout}'
        if bool(self.rccl_test_params.get('output_algo_proto_channels', False)):
            flags += ' -A 1'
        return flags

    def launch_command(self, binary_cmd, env_overrides=None):
        """Ask the launcher to spawn this rccl-tests binary."""
        self.prepare()
        overrides = self.env_overrides if env_overrides is None else env_overrides
        return self.launcher.command(binary_cmd, self.env_file, overrides)

    def execute(self, binary_cmd, reason, env_overrides=None):
        """Build and execute one collective command, returning launch output."""
        cmd = self.launch_command(binary_cmd, env_overrides)
        log.info('%%%%%%%%%%%%%%%%')
        log.info("%s", cmd)
        log.info('%%%%%%%%%%%%%%%%')
        error_count_before = len(globals.error_list)
        try:
            out_dict = self.shdl.exec(cmd, timeout=self.cvs_exec_timeout, detailed=True)
            if not out_dict or self.head_node not in out_dict:
                return self._fail(reason, f'RCCL launch produced no result from {self.head_node}: {cmd}')
            output, exit_code = self._result_payload(out_dict[self.head_node])
            if exit_code != 0:
                return self._fail(reason, f'RCCL launch failed with exit code {exit_code}: {cmd}')
            RcclVerifier.scan_logs(output)
            if len(globals.error_list) > error_count_before:
                return self._fail(reason, f'RCCL launch output failed checks: {cmd}')
            return output
        except Exception as error:
            return self._fail(reason, f'Hit Exceptions with rccl cmd {cmd} - exception {error!r}')

    def _fail(self, reason, message):
        log.error("%s", message)
        self._cleanup_stale_processes(reason)
        fail_test(message)
        return None

    def _cleanup_stale_processes(self, reason):
        """Best-effort kill of leftover mpirun/prterun/rccl-tests processes.

        Pssh.exec()'s timeout is a client-side SSH read timeout. It does not
        signal remote ranks, so a timed-out launch can leave mpirun and the
        rccl-tests binary holding GPUs on every host phdl can reach. Cleanup
        runs on job failure so the next pairwise/incremental sub-test does not
        oversubscribe those GPUs. Patterns cover both MpiRun leftovers and the
        job binary; they must not match srun or spur.
        """
        log.warning(f'Cleaning up stale RCCL/mpirun processes on all reachable hosts ({reason})')
        for pattern in ('prterun', 'mpirun.*rccl_hosts_file', self.test_name):
            try:
                self.phdl.exec(f"pkill -9 -f '{pattern}' || true", timeout=30)
            except Exception as kill_exc:
                log.warning(f'Cleanup pkill for pattern {pattern!r} raised {kill_exc!r} (continuing)')
        self.launcher.cleanup()

    @staticmethod
    def _result_payload(result):
        """Normalize detailed exec output to (stdout, exit_code). Missing codes are failures."""
        if isinstance(result, dict) and ('output' in result or 'exit_code' in result):
            output = result.get('output') or ''
            exit_code = result.get('exit_code')
            if exit_code is None:
                return output, -1
            return output, exit_code
        return result or '', -1

    def _detect_output_flag(self, binary_path):
        """Return -X if the binary supports --rccl_output_file, else legacy -x."""
        try:
            check_new_cmd = f'strings {binary_path} | grep -q "\\-\\-rccl_output_file"'
            result = self.shdl.exec(f'{check_new_cmd} && echo "NEW" || echo "OLD"')
            output = result[self.head_node].strip()
            if output == "NEW":
                log.debug(f"Detected new RCCL test format: using -X/--rccl_output_file for {binary_path}")
                return '-X'
            log.debug(f"Detected legacy RCCL test format: using -x/--output_file for {binary_path}")
            return '-x'
        except Exception as e:
            log.warning(f"Failed to detect RCCL output flag format for {binary_path}: {e}. Defaulting to legacy -x")
            return '-x'

    def read_results(self, result_file, label=None):
        log_label = label or self.test_name
        with tempfile.TemporaryDirectory(prefix='cvs_rccl_dl_') as tmpdir:
            local_prefix = os.path.join(tmpdir, os.path.basename(result_file))
            paths = self.shdl.download_file(result_file, local_prefix)
            local_path = paths[self.head_node]
            log.info('SFTP download succeeded for %s <- %s:%s', log_label, self.head_node, result_file)
            with open(local_path, 'r', encoding='utf-8') as f:
                raw_output = f.read()
            try:
                results = json.loads(raw_output)
            except json.JSONDecodeError:
                msg = (
                    f'Unable to parse RCCL JSON result file {result_file} on {self.head_node}. '
                    f'Raw content: {raw_output.strip() or "<empty>"}'
                )
                log.error(msg)
                fail_test(msg)
                return []
            if not isinstance(results, list) or any(not isinstance(row, dict) for row in results):
                fail_test(
                    f'Invalid RCCL JSON result structure in {result_file} on {self.head_node}: '
                    'expected an array of result objects'
                )
                return []
            return results

    def save_results(self, result_file, payload, label):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                delete=False,
                prefix='cvs_rccl_json_',
                suffix='.json',
                dir='/tmp',
            ) as tf:
                json.dump(payload, tf, indent=2)
                tmp_path = tf.name
            try:
                self.shdl.upload_file(tmp_path, result_file)
                log.info('SFTP upload succeeded for %s -> %s:%s', label, self.head_node, result_file)
                return True
            except IOError as e:
                log.error('SFTP upload failed for %s -> %s:%s: %r', label, self.head_node, result_file, e)
                return False
            except Exception as e:
                log.error(
                    'SFTP upload unexpected error for %s -> %s:%s: %r',
                    label,
                    self.head_node,
                    result_file,
                    e,
                )
                return False
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError as e:
                    log.warning('Failed to remove temp file %s: %r', tmp_path, e)

    def collect_gpu_info(self):
        smi_out_dict = self.shdl.exec('rocm-smi -a | head -30')
        get_model_from_rocm_smi_output(smi_out_dict[self.head_node])

    def expected_topology(self, gpus_per_rank):
        """Combine launcher rank placement with rccl-tests' total GPUs per rank (-t times -g)."""
        return {**self.launcher.topology(), 'gpusPerRank': gpus_per_rank}

    def _check_reported_topology(self, observed_rows, label, gpus_per_rank):
        """Record topology discrepancies and apply the configured warning/failure policy."""
        mode = self.topology_check_mode
        if mode == 'off':
            log.debug('Topology check disabled for %s', label)
            return

        requested = self.expected_topology(gpus_per_rank)
        record = {
            'label': label,
            'requested': requested,
            'reported': {},
            'mismatches': [],
            'mode': mode,
            'verdict': 'skipped',
        }
        self.topology_checks.append(record)
        if not isinstance(observed_rows, list) or any(not isinstance(row, dict) for row in observed_rows):
            record['reason'] = 'Invalid result structure: expected an array of result objects'
            return
        if not observed_rows:
            record['reason'] = 'No result rows available'
            return
        reported_rows = [{field: row[field] for field in TOPOLOGY_FIELDS if field in row} for row in observed_rows]
        reported = reported_rows[0]
        record['reported'] = reported
        if not any(reported_rows):
            record['reason'] = 'Producer did not report topology fields'
            return
        if isinstance(self.launcher, MpiRun) and requested['ranks'] % requested['nodes']:
            record['reason'] = (
                'MPI ranks are not evenly divisible by the number of cluster nodes; only uniform launches are supported'
            )
            log.warning('Topology check skipped for %s: %s', label, record['reason'])
            return

        mismatches = compare_topology(requested, reported)
        for index, row_topology in enumerate(reported_rows[1:], start=1):
            if row_topology != reported:
                mismatches.append(f'row {index}: topology differs from row 0 (reported {row_topology})')
        record['mismatches'] = mismatches
        record['verdict'] = 'mismatch' if mismatches else 'pass'
        if not mismatches:
            return

        details = '\n'.join(mismatches)
        message = (
            '\n==================== RCCL TOPOLOGY MISMATCH ====================\n'
            f'{label}\n{details}\n'
            'Check the rccl-tests output and the requested launcher/binary flags.\n'
        )
        if (
            requested['nodes'] > 1
            and reported.get('nodes') == 1
            and reported.get('ranks') == requested['ranks']
            and reported.get('ranksPerNode') == requested['ranks']
        ):
            message += (
                'rccl-tests may report the global MPI size as ranksPerNode in common.cu, '
                'which also produces an incorrect node count. The upstream fix must pass localSize.\n'
                'See AIMVT-334: https://amd-hub.atlassian.net/browse/AIMVT-334\n'
            )
        message += (
            'Producer topology is preserved in the result files.\n'
            '===============================================================\n'
        )
        if mode == 'strict':
            fail_test(message)
        else:
            log.warning('%s', message)

    def _save_topology_checks(self, result_file):
        """Persist the audit beside the producer results, including checks before an aborted run."""
        base_path = Path(result_file)
        path = str(base_path.with_name(f'{base_path.stem}_topology_check.json'))
        payload = {'mode': self.topology_check_mode, 'checks': self.topology_checks}
        if not self.save_results(path, payload, 'rccl_topology_check'):
            log.error('Failed to save topology checks to %s on head node %s', path, self.head_node)

    def run_regression(self):
        self.prepare()
        self.topology_checks = []
        params = self.rccl_test_params
        threads = int(params.get('threads_per_gpu', 1))
        result_file = self.cvs_params.get('rccl_result_file', '/tmp/rccl_result_output.json')
        binary_cmd = (
            f'{self.rccl_tests_dir}/{self.test_name} '
            f'-b {params.get("start_msg_size", "1024")} '
            f'-e {params.get("end_msg_size", "16g")} '
            f'-f {params.get("step_function", 2)} '
            f'-t {threads} '
            f'-w {params.get("warmup_iterations", 10)} '
            f'-n {params.get("no_of_iterations", 20)} '
            f'-N {params.get("no_of_cycles", 1)} '
            f'-c {params.get("check_iteration_count", 1)}{self.test_flags()} '
            f'-Z json {self.output_flag} {result_file}'
        )
        result_out = []
        try:
            if self.execute(binary_cmd, f'exception in rccl_regression: {self.test_name}') is not None:
                result_out = self.read_results(result_file)
            self._check_reported_topology(result_out, self.test_name, gpus_per_rank=threads)
        finally:
            self._save_topology_checks(result_file)
        if not result_out:
            return []
        self.collect_gpu_info()
        expected = self.cvs_params.get('results', {})
        test_expected = expected.get(self.test_name) if expected else None
        self._verify_results(result_out, test_expected)
        return result_out

    def run_perf(self):
        self.prepare()
        self.topology_checks = []
        params = self.rccl_test_params
        data_types = params.get('data_types', ['float'])
        result_file = self.cvs_params.get('rccl_result_file', '/tmp/rccl_result_output.json')
        base_path = Path(result_file)
        raw_results = []
        validated_results = []

        try:
            for dtype in data_types:
                dtype_file = f'{base_path.parent}/{base_path.stem}_{dtype}.json'
                results, validated = self._run_perf_dtype(dtype, dtype_file)
                if results is None:
                    return raw_results
                raw_results.extend(results)
                validated_results.extend(validated)

            if self.save_results(result_file, raw_results, 'combined_rccl_results'):
                log.info(f'Saved combined results from all data types to {result_file}')
            else:
                log.error('Failed to save combined results to %s on head node %s', result_file, self.head_node)
        finally:
            self._save_topology_checks(result_file)

        if not raw_results:
            return []
        aggregated = self._aggregate_perf_results(validated_results, base_path)
        self.collect_gpu_info()
        verification_results = self._verification_results(aggregated, raw_results)
        expected = self._perf_expected_results(data_types)
        self._verify_results(verification_results, expected)
        return raw_results

    def _run_perf_dtype(self, dtype, result_file):
        params = self.rccl_test_params
        gpus_per_rank = int(params.get('threads_per_gpu', 1))
        label = f'{self.test_name}_{dtype}'
        log.info(f'Running {self.test_name} with dtype={dtype}')
        binary_cmd = (
            f'{self.rccl_tests_dir}/{self.test_name} '
            f'-b {params.get("start_msg_size", "1024")} '
            f'-e {params.get("end_msg_size", "16g")} '
            f'-f {params.get("step_function", 2)} '
            f'-g {gpus_per_rank} '
            f'-c {params.get("check_iteration_count", 1)} '
            f'-w {params.get("warmup_iterations", 10)} '
            f'-d {dtype} '
            f'-n {params.get("no_of_iterations", 20)} '
            f'-N {params.get("no_of_cycles", 1)}{self.test_flags()} '
            f'-Z json {self.output_flag} {result_file}'
        )
        if self.execute(binary_cmd, f'exception in rccl_perf ({dtype})') is None:
            self._check_reported_topology([], label, gpus_per_rank)
            return None, None

        results = self.read_results(result_file, label)
        try:
            validated = [_result_model(result).model_validate(result) for result in results]
            log.info(f'{dtype}: {len(validated)} rccl-tests row(s) passed schema validation')
            self._check_reported_topology(results, label, gpus_per_rank)
            return results, validated
        except ValidationError as error:
            if RcclVerifier.is_severe_wrong_corruption_error(error):
                message = (
                    "\n"
                    "==================== SEVERE DATA CORRUPTION ====================\n"
                    "RCCL rccl-tests JSON schema validation failed due to '#wrong' > 0.\n"
                    "This indicates invalid/corrupted rccl-tests results.\n\n"
                    f"data_type: {dtype}\n"
                    f"result_file: {result_file}\n\n"
                    "Action: aborting further RCCL iterations/data types.\n"
                    "Please inspect the rccl-tests stdout/stderr and re-run.\n"
                    "================================================================\n"
                )
                log.error("%s", message)
                fail_test(message)
            else:
                log.error(f'Validation Failed: {error}')
                fail_test(f'RCCL Test {dtype} schema validation failed: {error}')
            raise RuntimeError(f'RCCL Test {dtype} schema validation failed') from error

    def _aggregate_perf_results(self, validated_results, base_path):
        if not validated_results:
            log.warning('Aggregation skipped: no validated results found')
            return None
        try:
            aggregated = self.aggregate_results(validated_results)
            log.info(f'Aggregation passed: {len(aggregated)} RcclTestsAggregated schema validation passed')
            path = f'{base_path.parent}/{base_path.stem}_aggregated.json'
            payload = [result.model_dump() for result in aggregated]
            if self.save_results(path, payload, 'aggregated_rccl_results'):
                log.info(f'Saved aggregated results to {path}')
            else:
                log.error('Failed to save aggregated results to %s on head node %s', path, self.head_node)
            return aggregated
        except (ValidationError, ValueError) as error:
            log.error(f'Aggregation failed: {error}')
            fail_test(f'RCCL Test aggregation failed: {error}')
            return None

    @staticmethod
    def _verification_results(aggregated, raw_results):
        if not aggregated:
            log.info('Using raw results for verification (no aggregation performed)')
            return raw_results
        results = [
            {
                'name': result.name,
                'size': result.size,
                'type': result.type,
                'inPlace': result.inPlace,
                'busBw': result.busBw_mean,
                'algBw': result.algBw_mean,
                'time': result.time_mean,
            }
            for result in aggregated
        ]
        log.info(f'Converted {len(results)} aggregated results for verification')
        return results

    def _perf_expected_results(self, data_types):
        nic_model = self.cvs_params.get('nic_model', 'ainic')
        if re.search('ainic|pensando|amd', nic_model, re.I):
            nic_type = 'ainic'
        elif re.search('broadcom|thor|bnxt', nic_model, re.I):
            nic_type = 'thor'
        elif re.search('mellanox|connectx|cx|nvidia', nic_model, re.I):
            nic_type = 'connectx'
        else:
            nic_type = 'ainic'
        log.info(f'Detected NIC type: {nic_type} from nic_model: {nic_model}')

        result_key = f'{self.test_name}-{"_".join(data_types)}-{self.no_of_global_ranks}'
        log.info(f'Looking up results with key: {result_key} in nic_type: {nic_type}')
        expected = self.cvs_params.get('results', {})
        nic_results = expected.get(nic_type, {}) if isinstance(expected, dict) else {}
        if result_key in nic_results:
            log.info(f'Found expected results: {nic_type}/{result_key}')
        return nic_results.get(result_key)

    def _verify_results(self, results, expected):
        RcclVerifier(self.test_name, results, expected, self.cvs_params).check()

    @staticmethod
    def aggregate_results(validated_results: List[RcclTests]) -> List[RcclTestsAggregated]:
        """Aggregate rccl-test rows into mean/std per (name, size, type, inPlace)."""
        if not validated_results:
            raise ValueError("validated_results list cannot be empty")

        multinode = isinstance(validated_results[0], RcclTestsMultinodeRaw)
        for i, result in enumerate(validated_results):
            if isinstance(result, RcclTestsMultinodeRaw) != multinode:
                raise ValueError(f"Mixed single-node and multi-node results at index {i}")

        multinode_config = None
        if multinode:
            first = validated_results[0]
            multinode_config = {
                'nodes': first.nodes,
                'ranks': first.ranks,
                'ranksPerNode': first.ranksPerNode,
                'gpusPerRank': first.gpusPerRank,
            }
            for i, result in enumerate(validated_results):
                if (
                    result.nodes != multinode_config['nodes']
                    or result.ranks != multinode_config['ranks']
                    or result.ranksPerNode != multinode_config['ranksPerNode']
                    or result.gpusPerRank != multinode_config['gpusPerRank']
                ):
                    raise ValueError(
                        f"Inconsistent cluster config at index {i}: "
                        f"expected {multinode_config}, got "
                        f"nodes={result.nodes}, ranks={result.ranks}, "
                        f"ranksPerNode={result.ranksPerNode}, gpusPerRank={result.gpusPerRank}"
                    )
            log.info(f"Validated consistent multinode config: {multinode_config}")

        log.info(f"Aggregating {len(validated_results)} RCCL test results")
        data = [result.model_dump() for result in validated_results]
        df = pd.DataFrame(data)
        agg_df = df.groupby(['name', 'size', 'type', 'inPlace'], as_index=False).agg(
            busBw_mean=('busBw', 'mean'),
            busBw_std=('busBw', 'std'),
            algBw_mean=('algBw', 'mean'),
            algBw_std=('algBw', 'std'),
            time_mean=('time', 'mean'),
            time_std=('time', 'std'),
            num_runs=('numCycle', 'count'),
        )
        if multinode_config:
            for key, value in multinode_config.items():
                agg_df[key] = value

        agg_results = []
        errors = []
        for row_dict in agg_df.to_dict('records'):
            try:
                agg_results.append(RcclTestsAggregated.model_validate(row_dict))
            except ValidationError as e:
                error_msg = f"Validation failed for row {row_dict}: {e}"
                log.error("%s", error_msg)
                errors.append(error_msg)
        if errors:
            fail_test("Aggregation validation failed:\n" + "\n".join(errors))
        log.info(f"Successfully validated {len(agg_results)} aggregated results")
        return agg_results
