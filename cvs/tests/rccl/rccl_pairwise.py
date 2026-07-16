import pytest
import json

from cvs.lib import rccl_lib
from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


# ─────────────────────────────────────────────
# Fixtures  (identical pattern to rccl_perf.py)
# ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as f:
        cluster_dict = json.load(f)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as f:
        config_dict_t = json.load(f)
    config_dict = config_dict_t['rccl']
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    log.info("%s", cluster_dict)
    env_vars = cluster_dict.get("env_vars")
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
    return phdl


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    node_list = list(cluster_dict['node_dict'].keys())
    env_vars = cluster_dict.get("env_vars")
    head_node = node_list[0]
    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
    return shdl


@pytest.fixture(scope="module")
def vpc_node_list(cluster_dict):
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])
    return vpc_node_list


# ─────────────────────────────────────────────
# Helper: run one pairwise RCCL test
# ─────────────────────────────────────────────


def run_pairwise_rccl(phdl, shdl, node_pair_vpc, node_pair_mgmt, config_dict, phase_label):
    """
    Run all_reduce_perf between exactly two nodes and return the raw result dict.

    Args:
        phdl:             Parallel SSH handle (all cluster nodes).
        shdl:             SSH handle to head node.
        node_pair_vpc:    List of two VPC IPs to pass to mpirun.
        node_pair_mgmt:   List of two mgmt hostnames (used as cluster_node_list).
        config_dict:      RCCL config dict (mpi_params, rccl_test_params, cvs_params).
        phase_label:      Human-readable label for logging (e.g. 'Phase1 nodeA<->nodeB').

    Returns:
        result_dict from rccl_lib.rccl_perf, or None on exception.
    """
    log.info('─' * 60)
    log.info('Running pairwise RCCL: %s', phase_label)
    log.info('Nodes (VPC): %s', node_pair_vpc)

    # Override no_of_nodes to 2 for the pairwise run without mutating the
    # original dict — create a shallow copy of mpi_params only.
    mpi_params_pair = dict(config_dict['mpi_params'])
    mpi_params_pair['no_of_nodes'] = '2'

    env_script = config_dict.get('env_source_script', '/dev/null')

    try:
        result_dict = rccl_lib.rccl_perf(
            phdl,
            shdl,
            'all_reduce_perf',
            env_script,
            mpi_params_pair,
            config_dict['rccl_test_params'],
            config_dict['cvs_params'],
            node_pair_mgmt,  # cluster_node_list  (first entry = head node)
            node_pair_vpc,  # vpc_node_list       (passed to mpirun -H)
        )
        log.info('Pairwise result for %s: %s', phase_label, result_dict)
        return result_dict
    except Exception as exc:
        log.error('Pairwise RCCL failed for %s: %s', phase_label, exc)
        return None


# ─────────────────────────────────────────────
# Helper: evaluate pass / fail from result_dict
# ─────────────────────────────────────────────


def _is_pairwise_pass(result_dict, min_bw_gbps):
    """
    Return True when result_dict contains at least one result entry and the
    reported bus bandwidth meets or exceeds min_bw_gbps.

    A result_dict of None or an empty list is treated as a failure.
    min_bw_gbps <= 0 skips the bandwidth check (pass if test ran at all).
    """
    if not result_dict:
        return False

    # rccl_lib.rccl_perf returns a list of per-dtype dicts.
    # Each dict is expected to contain 'bus_bw' (float, GB/s) at the largest
    # message size tested.  We take the maximum across all dtype results.
    if not isinstance(result_dict, list) or len(result_dict) == 0:
        return False

    if min_bw_gbps <= 0:
        return True

    best_bw = 0.0
    for entry in result_dict:
        try:
            bw_values = list(entry.get('bus_bw', {}).values())
            if bw_values:
                best_bw = max(best_bw, max(float(v) for v in bw_values))
        except (TypeError, ValueError):
            pass

    log.info('Best bus BW observed: %.2f GB/s  (threshold: %.2f GB/s)', best_bw, min_bw_gbps)
    return best_bw >= min_bw_gbps


# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────


def test_collect_hostinfo(phdl):
    """Collect basic ROCm / host info from all nodes."""
    globals.error_list = []
    phdl.exec('cat /opt/rocm/.info/version')
    phdl.exec('hipconfig')
    phdl.exec('rocm_agent_enumerator')
    update_test_result()


def test_collect_networkinfo(phdl):
    """Collect basic RDMA / verbs info from all nodes."""
    globals.error_list = []
    phdl.exec('rdma link')
    phdl.exec('ibv_devinfo')
    update_test_result()


def test_rccl_pairwise(phdl, shdl, cluster_dict, config_dict, vpc_node_list):
    """
    Phase 0 + Phase 1: reference-node sanity check then pairwise validation.

    Phase 0 — single-node sanity check on the reference (first) node.
    Phase 1 — for every remaining node, run all_reduce_perf between the
               reference node and the candidate node.

    Passes if:
      - Phase 0 succeeds on the reference node, AND
      - Every pairwise test in Phase 1 succeeds.

    Failures are accumulated and reported at the end so that all pairs are
    always tested regardless of individual failures.

    Config knobs consumed from config_dict['cvs_params']:
      - pairwise_min_bw   (float, GB/s, default 0 → no BW check)
    """
    globals.error_list = []

    node_list = list(cluster_dict['node_dict'].keys())  # mgmt hostnames
    min_bw = float(config_dict.get('cvs_params', {}).get('pairwise_min_bw', 0))

    if len(node_list) < 1:
        fail_test('No nodes found in cluster_dict — cannot run pairwise test.')
        update_test_result()
        return

    # ── Phase 0: single-node sanity on the reference node ────────────────────
    ref_mgmt = node_list[0]
    ref_vpc = vpc_node_list[0]

    log.info('=' * 60)
    log.info('PHASE 0 — Single-node sanity check on reference node: %s', ref_mgmt)
    log.info('=' * 60)

    sanity_result = run_pairwise_rccl(
        phdl,
        shdl,
        node_pair_vpc=[ref_vpc],
        node_pair_mgmt=[ref_mgmt],
        config_dict=config_dict,
        phase_label=f'Phase0 sanity {ref_mgmt}',
    )

    if not sanity_result:
        fail_test(
            f'PHASE 0 FAILED: Reference node {ref_mgmt} did not pass the '
            f'single-node sanity check.  Aborting pairwise phase.'
        )
        update_test_result()
        return

    log.info('PHASE 0 PASSED: Reference node %s is healthy.', ref_mgmt)

    # ── Phase 1: pairwise tests ───────────────────────────────────────────────
    log.info('=' * 60)
    log.info('PHASE 1 — Pairwise validation (reference node: %s)', ref_mgmt)
    log.info('=' * 60)

    phase1_pass = []
    phase1_fail = []

    for idx in range(1, len(node_list)):
        cand_mgmt = node_list[idx]
        cand_vpc = vpc_node_list[idx]
        label = f'Phase1 {ref_mgmt} <-> {cand_mgmt}'

        result = run_pairwise_rccl(
            phdl,
            shdl,
            node_pair_vpc=[ref_vpc, cand_vpc],
            node_pair_mgmt=[ref_mgmt, cand_mgmt],
            config_dict=config_dict,
            phase_label=label,
        )

        if _is_pairwise_pass(result, min_bw):
            log.info('PHASE 1 PASS: %s', label)
            phase1_pass.append(cand_mgmt)
        else:
            log.error('PHASE 1 FAIL: %s', label)
            phase1_fail.append(cand_mgmt)
            fail_test(f'Pairwise test failed: {label}')

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info('=' * 60)
    log.info('PAIRWISE SUMMARY')
    log.info('Reference node  : %s', ref_mgmt)
    log.info('Passed (%d)     : %s', len(phase1_pass), phase1_pass)
    log.info('Failed (%d)     : %s', len(phase1_fail), phase1_fail)
    log.info('=' * 60)

    update_test_result()


def test_rccl_incremental(phdl, shdl, cluster_dict, config_dict, vpc_node_list):
    """
    Phase 2: incremental cluster build.

    Starting from the reference node, add one candidate at a time.
    A candidate is accepted into the valid cluster only when all_reduce_perf
    across [current valid nodes + candidate] meets the bandwidth threshold.

    Config knobs consumed from config_dict['cvs_params']:
      - pairwise_min_bw   (float, GB/s, default 0 → no BW check for Phase 2)
    """
    globals.error_list = []

    node_list = list(cluster_dict['node_dict'].keys())
    min_bw = float(config_dict.get('cvs_params', {}).get('pairwise_min_bw', 0))

    if len(node_list) < 2:
        log.warning('Only %d node(s) in cluster — skipping incremental phase.', len(node_list))
        update_test_result()
        return

    log.info('=' * 60)
    log.info('PHASE 2 — Incremental cluster build')
    log.info('=' * 60)

    # Seed the valid set with the reference node
    valid_mgmt = [node_list[0]]
    valid_vpc = [vpc_node_list[0]]

    phase2_pass = []
    phase2_fail = []

    for idx in range(1, len(node_list)):
        cand_mgmt = node_list[idx]
        cand_vpc = vpc_node_list[idx]

        trial_mgmt = valid_mgmt + [cand_mgmt]
        trial_vpc = valid_vpc + [cand_vpc]
        label = f'Phase2 {len(trial_mgmt)}-node cluster (adding {cand_mgmt})'

        log.info('Attempting to add node: %s  (cluster size would be %d)', cand_mgmt, len(trial_mgmt))

        # Override no_of_nodes to match the trial cluster size
        mpi_params_trial = dict(config_dict['mpi_params'])
        mpi_params_trial['no_of_nodes'] = str(len(trial_mgmt))
        trial_config = dict(config_dict)
        trial_config['mpi_params'] = mpi_params_trial

        env_script = config_dict.get('env_source_script', '/dev/null')

        try:
            result = rccl_lib.rccl_perf(
                phdl,
                shdl,
                'all_reduce_perf',
                env_script,
                mpi_params_trial,
                config_dict['rccl_test_params'],
                config_dict['cvs_params'],
                trial_mgmt,
                trial_vpc,
            )
        except Exception as exc:
            log.error('Phase 2 RCCL call raised exception for %s: %s', label, exc)
            result = None

        if _is_pairwise_pass(result, min_bw):
            log.info('PHASE 2 PASS: %s', label)
            valid_mgmt.append(cand_mgmt)
            valid_vpc.append(cand_vpc)
            phase2_pass.append(cand_mgmt)
        else:
            log.error('PHASE 2 FAIL: Node %s degraded cluster — excluding.', cand_mgmt)
            phase2_fail.append(cand_mgmt)
            fail_test(f'Incremental test failed when adding {cand_mgmt}: {label}')

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info('=' * 60)
    log.info('PHASE 2 SUMMARY')
    log.info('Final valid cluster (%d nodes) : %s', len(valid_mgmt), valid_mgmt)
    log.info('Excluded nodes      (%d)       : %s', len(phase2_fail), phase2_fail)
    log.info('=' * 60)

    update_test_result()
