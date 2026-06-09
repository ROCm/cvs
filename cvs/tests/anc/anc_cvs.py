'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# NOTE: This file presumes that the ANC installation has already been completed
# on every node at "<cvs_home>/<anc_root_dir>" (see anc_installation.py). These
# tests only execute the ANC validation groups and collect their artifacts; they
# do not install or update ANC.
#
# Per-node artifacts are downloaded to:
#     <runner_log_folder>/anc/<ip>_<hostname>/<test_name>/
# (runner_log_folder defaults to /tmp/cvs_results).

import os
import re
import pytest
import json

from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import (
    fail_test,
    update_test_result,
    print_test_output,
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)
from cvs.lib import globals
from cvs.lib.run_config_paths import resolve_runner_results_base

log = globals.log

# ANC prints "Log Directory: <path>" near the top of its stdout; every artifact
# (journal.log, console.log, summary.json, ...) is written under this directory.
LOG_DIRECTORY_RE = re.compile(r"Log Directory:\s*(\S+)")

# ANC records its program return code near the end of console.log, e.g.
#   "Program exiting with return code ANC_SUCCESS [0]"
#   "Program exiting with return code ANC_PROG_VALIDATION_ERROR [5]".
# The FINAL such line is authoritative: a return code of [0] (ANC_SUCCESS) is the
# only PASS; anything else FAILS.
ANC_RETURN_CODE_RE = re.compile(r"return code\s+(\S+)\s*\[(-?\d+)\]")

# journal.log and console.log MUST be produced by every run; their absence (or a
# download failure) is treated as an infrastructure failure. console.log is also
# scanned for the final return code. The rest are best-effort.
MANDATORY_ARTIFACTS = ("journal.log", "console.log")
OPTIONAL_ARTIFACTS = ("summary.json", "errors.json", "system_monitor.json")

# Default per-group execution timeout (2 hours). Overridable via
# config_dict["anc"]["test_timeout"].
DEFAULT_ANC_TEST_TIMEOUT = 7200


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    '''
    Retrieve the --cluster_file CLI option value provided to pytest.

    Returns:
      str: Path to the ANC cluster JSON file.
    '''
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    '''
    Retrieve the --config_file CLI option value provided to pytest.

    Returns:
      str: Path to the ANC test configuration JSON file.
    '''
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    '''
    Load and resolve the ANC cluster configuration from JSON.

    Returns:
      dict: Parsed and resolved cluster configuration.
    '''
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("ANC cluster config: %s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    '''
    Load and resolve the ANC test configuration from JSON.

    Placeholders such as {home} are resolved using cluster_dict
    (e.g. cvs_home "{home}/cvs" -> "/home/<user>/cvs").

    Returns:
      dict: Parsed and resolved test configuration.
    '''
    with open(config_file) as json_file:
        config_dict = json.load(json_file)

    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("ANC test config: %s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    '''
    Parallel SSH handle for all ANC cluster nodes.

    Returns:
      Pssh: Handle targeting every node in cluster_dict["node_dict"].
    '''
    node_list = list(cluster_dict["node_dict"].keys())

    return Pssh(
        log,
        node_list,
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
    )


def _sanitize_path_component(value):
    '''Make a string safe to use as a single filesystem path component.'''
    return re.sub(r"[^\w.\-]", "_", value)


def _node_label(single, host, cluster_dict):
    '''
    Build the "<ip>_<hostname>" folder label for a node.

    The IP is taken from node_dict[host]["vpc_ip"] when available (falling back
    to the SSH target key), and the hostname is queried live from the node.

    Parameters:
      single:       Single-node Pssh handle for host.
      host:         SSH target key for the node.
      cluster_dict: Resolved cluster configuration.

    Returns:
      str: Sanitized "<ip>_<hostname>" path component.
    '''
    node_info = cluster_dict.get("node_dict", {}).get(host, {}) or {}
    ip = node_info.get("vpc_ip")
    if not ip or ip == "NA":
        ip = host

    hostname = host
    try:
        out = single.exec("hostname", timeout=30)
        resolved = (out.get(host, "") or "").strip()
        if resolved:
            hostname = resolved
    except Exception as exc:  # label is best-effort
        log.warning("Node %s: could not resolve hostname: %s", host, exc)

    return _sanitize_path_component(f"{ip}_{hostname}")


def _claim_remote_logs(single, host, user, log_dir):
    '''
    Take ownership of ANC's log directory so it can be pulled over SFTP.

    ANC runs under sudo, so the files it writes are typically root-owned and
    unreadable by the plain SSH user used for download_file (SFTP). Recursively
    chown the directory back to the SSH user. Best-effort: any failure is logged
    and surfaced later as a download failure if the files are truly unreadable.

    Parameters:
      single:   Single-node Pssh handle for host.
      host:     SSH target key for the node.
      user:     SSH user that must own the files for SFTP download.
      log_dir:  ANC log directory path on the node.
    '''
    cmd = f"sudo chown -R {user} '{log_dir}'"
    try:
        single.exec(cmd, timeout=60)
    except Exception as exc:  # best-effort; download will report if still locked
        log.warning("Node %s: could not chown log dir %s: %s", host, log_dir,
                    exc)


def _list_remote_files(single, host, log_dir):
    '''
    List the file names directly under log_dir on host (single round-trip).

    Returns:
      set | None: Set of file names, or None if the directory could not be
      listed (e.g. SSH error).
    '''
    try:
        out = single.exec(f"ls -1 '{log_dir}' 2>/dev/null", timeout=30)
    except Exception as exc:
        log.warning("Node %s: could not list log dir %s: %s", host, log_dir, exc)
        return None
    return set((out.get(host, "") or "").split())


def _download_artifact(single, host, remote_path, dest_dir, name):
    '''
    Download a single artifact.

    Returns:
      str | None: Local path on success, or None on download failure.
    '''
    try:
        result = single.download_file(remote_path, os.path.join(dest_dir, name))
        local_path = result.get(host, os.path.join(dest_dir, name))
        log.info("Node %s: downloaded %s -> %s", host, name, local_path)
        return local_path
    except Exception as exc:  # copy failure must fail the test
        log.error("Node %s: failed to download %s: %s", host, remote_path, exc)
        return None


def _download_artifacts(single, host, log_dir, dest_dir):
    '''
    Download mandatory and optional ANC artifacts for a node.

    Returns:
      tuple[dict, str | None]: (local_paths_by_name, infra_failure_reason).
        infra_failure_reason is None when both mandatory artifacts were
        downloaded successfully.
    '''
    remote_names = _list_remote_files(single, host, log_dir)
    if remote_names is None:
        return {}, f"could not list log directory: {log_dir}"

    local_paths = {}

    for name in MANDATORY_ARTIFACTS:
        if name not in remote_names:
            return local_paths, f"mandatory artifact missing: {log_dir}/{name}"
        local = _download_artifact(single, host, f"{log_dir}/{name}", dest_dir,
                                   name)
        if local is None:
            return local_paths, (
                f"failed to download mandatory artifact: {log_dir}/{name}"
            )
        local_paths[name] = local

    for name in OPTIONAL_ARTIFACTS:
        if name not in remote_names:
            log.info("Node %s: optional artifact not present: %s", host, name)
            continue
        local = _download_artifact(single, host, f"{log_dir}/{name}", dest_dir,
                                   name)
        if local is not None:
            local_paths[name] = local

    return local_paths, None


def _evaluate_node(cluster_dict, host, output, base, test_name):
    '''
    Collect artifacts for one node and decide whether it passed.

    A node FAILS when any of the following occur:
      - the Log Directory could not be parsed from ANC's output (did not run);
      - an SSH handle for collection could not be opened;
      - the log directory could not be listed, or a mandatory artifact
        (journal.log/console.log) is missing or could not be downloaded
        (cannot copy logs);
      - console.log cannot be read, has no return code, or its final return
        code is non-zero (non ANC_SUCCESS).

    Returns:
      str | None: A failure reason for this node, or None if the node passed.
    '''
    ld_match = LOG_DIRECTORY_RE.search(output)
    if not ld_match:
        log.error("Node %s: could not determine Log Directory from output", host)
        return "could not determine Log Directory from output"
    log_dir = ld_match.group(1)
    log.info("Node %s: ANC %s log directory: %s", host, test_name, log_dir)

    try:
        single = Pssh(
            log,
            [host],
            user=cluster_dict["username"],
            pkey=cluster_dict["priv_key_file"],
        )
    except Exception as exc:  # infra failure must fail the test
        return f"could not open SSH handle for artifact collection: {exc}"

    label = _node_label(single, host, cluster_dict)
    dest_dir = os.path.join(base, "anc", label, test_name)
    os.makedirs(dest_dir, exist_ok=True)

    # ANC ran under sudo; reclaim ownership so SFTP (plain user) can read.
    _claim_remote_logs(single, host, cluster_dict["username"], log_dir)

    local_paths, infra_reason = _download_artifacts(single, host, log_dir,
                                                     dest_dir)
    if infra_reason:
        return infra_reason

    console_path = local_paths["console.log"]
    try:
        with open(console_path, encoding="utf-8", errors="replace") as fh:
            console_text = fh.read()
    except Exception as exc:
        return f"could not read downloaded console.log: {exc}"

    # The final "return code <NAME> [<int>]" line in console.log is authoritative.
    rc_matches = ANC_RETURN_CODE_RE.findall(console_text)
    if not rc_matches:
        log.error("Node %s: ANC return code not found in console.log", host)
        return "ANC return code not found in console.log"

    rc_name, rc_value = rc_matches[-1][0], int(rc_matches[-1][1])
    log.info("Node %s: ANC %s test return code is %s [%s]", host, test_name,
             rc_name, rc_value)
    if rc_value != 0:
        return f"non-zero ANC return code {rc_name} [{rc_value}]"

    return None


def _run_anc_group(phdl, cluster_dict, config_dict, group, test_name):
    '''
    Execute an ANC group on every node, collect artifacts, and judge the result.

    Pass/fail policy:
      - PASS only when every expected node reports a final ANC return code [0]
        (ANC_SUCCESS) with journal.log/console.log collected.
      - A node FAILS on a non-zero final return code, an inability to run
        (SSH/exec/permission error, no output, or missing Log Directory), or an
        inability to copy mandatory logs.
      - Failures across multiple parallel nodes are aggregated into a SINGLE
        test failure (not one per node).

    Parameters:
      phdl:         Parallel SSH handle for all nodes.
      cluster_dict: Resolved cluster configuration.
      config_dict:  Resolved test configuration.
      group:        ANC group to run (e.g. "cpu_all", "gpu_mfg_l10").
      test_name:    Test name used for the artifact path (e.g. "test_cpu").
    '''
    globals.error_list = []

    anc_cfg = config_dict["anc"]
    cvs_home = anc_cfg["cvs_home"]
    anc_root_dir = anc_cfg["anc_root_dir"]
    anc_dir = f"{cvs_home}/{anc_root_dir}"
    timeout = anc_cfg.get("test_timeout", DEFAULT_ANC_TEST_TIMEOUT)
    base = resolve_runner_results_base(config_dict.get("run_config", {}))
    expected_nodes = list(cluster_dict["node_dict"].keys())

    cmd = f"cd '{anc_dir}' && sudo ./anc.py --group {group}"
    log.info("ANC '%s': running '%s' (timeout=%ss, artifacts under %s)",
             test_name, cmd, timeout, os.path.join(base, "anc"))

    try:
        out_dict = phdl.exec(cmd, timeout=timeout)
    except Exception as exc:  # infra failure must fail the test
        fail_test(f"ANC {test_name}: execution failed (SSH/exec error): {exc}")
        update_test_result()
        return

    print_test_output(log, out_dict)

    if not out_dict:
        fail_test(f"ANC {test_name}: no output / no reachable nodes")
        update_test_result()
        return

    failed_nodes = {}
    for host in expected_nodes:
        if host not in out_dict:
            reason = "node produced no output (did not run / unreachable)"
            log.error("Node %s: ANC %s FAILED - %s", host, test_name, reason)
            failed_nodes[host] = reason
            continue
        reason = _evaluate_node(cluster_dict, host, out_dict[host] or "", base,
                                test_name)
        if reason:
            log.error("Node %s: ANC %s FAILED - %s", host, test_name, reason)
            failed_nodes[host] = reason

    # Aggregate all node failures into a single test failure.
    if failed_nodes:
        details = "; ".join(f"{h}: {r}" for h, r in failed_nodes.items())
        fail_test(
            f"ANC {test_name} failed on {len(failed_nodes)}/{len(expected_nodes)} "
            f"node(s): {details}"
        )

    update_test_result()


class TestAncPreTasks:
    '''Pre-tasks: validate connectivity and gather host information.'''
    pass


class TestAncCoreTasks:
    '''Core ANC test execution tasks.'''

    def test_cpu(self, phdl, cluster_dict, config_dict):
        '''
        Run the ANC CPU validation group (cpu_all) on all nodes.

        Executes "sudo ./anc.py --group cpu_all" under the ANC install
        directory and collects journal.log/console.log (mandatory) plus
        summary.json/errors.json/system_monitor.json (when present). PASS only
        when every node reports a final ANC_SUCCESS [0] return code.
        '''
        log.info("ANC Core Task: CPU validation (group=cpu_all)")
        _run_anc_group(phdl, cluster_dict, config_dict, "cpu_all", "test_cpu")

    def test_gpu(self, phdl, cluster_dict, config_dict):
        '''
        Run the ANC GPU validation group (gpu_mfg_l10) on all nodes.

        Executes "sudo ./anc.py --group gpu_mfg_l10" under the ANC install
        directory and collects journal.log/console.log (mandatory) plus
        summary.json/errors.json/system_monitor.json (when present). PASS only
        when every node reports a final ANC_SUCCESS [0] return code.
        '''
        log.info("ANC Core Task: GPU validation (group=gpu_mfg_l10)")
        _run_anc_group(phdl, cluster_dict, config_dict, "gpu_mfg_l10",
                       "test_gpu")


class TestAncPostTasks:
    '''Post-tasks: cleanup and result collection.'''
    pass
