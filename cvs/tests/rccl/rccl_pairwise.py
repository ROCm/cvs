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
    Run all_reduce_perf across exactly len(node_pair_mgmt) nodes and report
    whether the run was clean (no new entries in globals.error_list).

    Used for Phase 0 (1 node), Phase 1 (2 nodes: reference + candidate), and
    Phase 2 (N nodes: current valid group + candidate) — no_of_nodes always
    matches the number of nodes actually passed in, so callers do not need to
    special-case the node count.

    Cluster-specific NCCL/UCX tuning that the original bash tool hardcoded
    inline (NCCL_IB_HCA, NCCL_IB_GID_INDEX, NCCL_SOCKET_IFNAME, NCCL_DMABUF_ENABLE,
    forced UCX PML) is expected to come from config_dict['env_source_script']
    here — it is not reproduced inline, so that script must be kept in sync
    with the target cluster's fabric for results to be comparable to the bash
    tool's.

    Args:
        phdl:             Parallel SSH handle (all cluster nodes).
        shdl:             SSH handle to head node.
        node_pair_vpc:    List of VPC IPs to pass to mpirun.
        node_pair_mgmt:   List of mgmt hostnames (used as cluster_node_list).
        config_dict:      RCCL config dict (mpi_params, rccl_test_params, cvs_params).
        phase_label:      Human-readable label for logging (e.g. 'Phase1 nodeA<->nodeB').

    Returns:
        (result_dict, clean_run) tuple.
        result_dict is the raw list returned by rccl_lib.rccl_perf, or None on exception.
        clean_run is True only if the call completed without exception AND no
        new failures were recorded in globals.error_list during the call.
    """
    log.info('─' * 60)
    log.info('Running pairwise RCCL: %s', phase_label)
    log.info('Nodes (VPC): %s', node_pair_vpc)

    # Override no_of_nodes to match this call's node count without mutating
    # the original dict — create a shallow copy of mpi_params only.
    mpi_params_pair = dict(config_dict['mpi_params'])
    mpi_params_pair['no_of_nodes'] = str(len(node_pair_mgmt))

    env_script = config_dict.get('env_source_script', '/dev/null')

    error_count_before = len(globals.error_list)
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
    except Exception as exc:
        log.error('Pairwise RCCL failed for %s: %s', phase_label, exc)
        return None, False

    clean_run = len(globals.error_list) == error_count_before
    if not clean_run:
        log.error('Pairwise RCCL for %s reported errors: %s', phase_label, globals.error_list[error_count_before:])
    return result_dict, clean_run


# ─────────────────────────────────────────────
# Helper: evaluate pass / fail from result_dict
# ─────────────────────────────────────────────


def _extract_best_bw(result_dict):
    """
    Return the measured busBw (GB/s) for the in-place all_reduce measurement
    at the largest message size tested, or None if unavailable.

    rccl_lib.rccl_perf returns a flat list of raw rccl-tests JSON entries
    (schema cvs/schema/rccl.py:RcclTests), each keyed 'busBw' (float),
    'size' (int, bytes), and 'inPlace' (0 or 1) — NOT a 'bus_bw' dict.
    Selecting the largest-size, in-place row mirrors rccl_lib.check_bus_bw's
    in-place branch and matches bash's get_final_bw_only (last/largest
    message-size row of a two-size 8G/16G sweep).
    """
    if not result_dict:
        return None

    inplace_entries = [
        entry for entry in result_dict if isinstance(entry, dict) and entry.get('inPlace') == 1 and 'busBw' in entry
    ]
    if not inplace_entries:
        return None

    largest = max(inplace_entries, key=lambda entry: entry.get('size', 0))
    try:
        return float(largest['busBw'])
    except (TypeError, ValueError):
        return None


def _is_pairwise_pass(result_dict, clean_run, min_bw_gbps, require_bw_check):
    """
    Pass criterion for a pairwise/incremental RCCL run:
      - the rccl_perf call completed without raising, AND
      - clean_run is True (no new entries were added to globals.error_list
        during the call — i.e. no ORTE/NCCL error, no missing-bandwidth-marker
        failure, no schema validation failure), AND
      - result_dict is non-empty, AND
      - (only when require_bw_check and min_bw_gbps > 0) the measured busBw at
        the largest message size meets or exceeds min_bw_gbps.

    require_bw_check=False reproduces bash's Phase 1 semantics (clean exit
    only, no bandwidth gate). require_bw_check=True reproduces bash's Phase 2
    semantics (clean exit AND BusBW >= MIN_BW). min_bw_gbps <= 0 always skips
    the bandwidth comparison regardless of require_bw_check.
    """
    if not (clean_run and result_dict):
        return False

    if not require_bw_check or min_bw_gbps <= 0:
        return True

    best_bw = _extract_best_bw(result_dict)
    if best_bw is None:
        log.error('No in-place busBw measurement found in result — treating as failure.')
        return False

    log.info('Best bus BW observed: %.2f GB/s  (threshold: %.2f GB/s)', best_bw, min_bw_gbps)
    return best_bw >= min_bw_gbps


def _persist_pairwise_artifact(config_dict, phase_key, data):
    """
    Persist minimal pass/fail bookkeeping for a phase to a local JSON file so
    the outcome of a run survives beyond the log. Merges into any existing
    file under the same path so Phase 1 and Phase 2 results end up together.
    """
    results_file = config_dict.get('cvs_params', {}).get('pairwise_results_file', '/tmp/rccl_pairwise_results.json')
    try:
        try:
            with open(results_file) as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {}
        existing[phase_key] = data
        with open(results_file, 'w') as f:
            json.dump(existing, f, indent=2)
        log.info('Persisted %s results to %s', phase_key, results_file)
    except OSError as exc:
        log.error('Failed to persist %s results to %s: %s', phase_key, results_file, exc)


# Phase 1 survivors (mgmt hostnames, reference node excluded), populated by
# test_rccl_pairwise for test_rccl_incremental to consume. None until Phase 1
# has run in this process.
_phase1_survivors = None


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

    sanity_result, sanity_clean = run_pairwise_rccl(
        phdl,
        shdl,
        node_pair_vpc=[ref_vpc],
        node_pair_mgmt=[ref_mgmt],
        config_dict=config_dict,
        phase_label=f'Phase0 sanity {ref_mgmt}',
    )

    if not (sanity_clean and sanity_result):
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

        result, clean_run = run_pairwise_rccl(
            phdl,
            shdl,
            node_pair_vpc=[ref_vpc, cand_vpc],
            node_pair_mgmt=[ref_mgmt, cand_mgmt],
            config_dict=config_dict,
            phase_label=label,
        )

        # Phase 1 pass criterion matches bash: clean exit only, no bandwidth gate.
        if _is_pairwise_pass(result, clean_run, min_bw, require_bw_check=False):
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

    global _phase1_survivors
    _phase1_survivors = list(phase1_pass)
    _persist_pairwise_artifact(
        config_dict,
        'phase1',
        {'reference_node': ref_mgmt, 'passed': phase1_pass, 'failed': phase1_fail},
    )

    update_test_result()


def test_rccl_incremental(phdl, shdl, cluster_dict, config_dict, vpc_node_list):
    """
    Phase 2: incremental cluster build.

    Starting from the reference node, add one candidate at a time.
    A candidate is accepted into the valid cluster only when all_reduce_perf
    across [current valid nodes + candidate] runs cleanly AND meets the
    bandwidth threshold. Candidates are Phase 1 survivors only (matching
    bash, which reads phase1_successful_hosts.txt) — a node that failed
    Phase 1 is not retried here.

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

    ref_mgmt = node_list[0]
    ref_vpc = vpc_node_list[0]
    mgmt_to_vpc = dict(zip(node_list, vpc_node_list))

    global _phase1_survivors
    if _phase1_survivors is not None:
        candidate_mgmt_list = _phase1_survivors
        log.info('Phase 2 consuming %d Phase 1 survivor(s): %s', len(candidate_mgmt_list), candidate_mgmt_list)
    else:
        candidate_mgmt_list = node_list[1:]
        log.warning(
            'Phase 1 survivor list unavailable (test_rccl_pairwise did not run first in this '
            'session) — falling back to testing all %d candidate node(s).',
            len(candidate_mgmt_list),
        )

    # Seed the valid set with the reference node
    valid_mgmt = [ref_mgmt]
    valid_vpc = [ref_vpc]

    phase2_pass = []
    phase2_fail = []

    for cand_mgmt in candidate_mgmt_list:
        cand_vpc = mgmt_to_vpc[cand_mgmt]

        trial_mgmt = valid_mgmt + [cand_mgmt]
        trial_vpc = valid_vpc + [cand_vpc]
        label = f'Phase2 {len(trial_mgmt)}-node cluster (adding {cand_mgmt})'

        log.info('Attempting to add node: %s  (cluster size would be %d)', cand_mgmt, len(trial_mgmt))

        result, clean_run = run_pairwise_rccl(
            phdl,
            shdl,
            node_pair_vpc=trial_vpc,
            node_pair_mgmt=trial_mgmt,
            config_dict=config_dict,
            phase_label=label,
        )

        # Phase 2 pass criterion matches bash: clean exit AND BusBW >= pairwise_min_bw.
        if _is_pairwise_pass(result, clean_run, min_bw, require_bw_check=True):
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

    _persist_pairwise_artifact(
        config_dict,
        'phase2',
        {'final_valid_cluster': valid_mgmt, 'excluded_nodes': phase2_fail},
    )

    update_test_result()
