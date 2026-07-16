'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

'''
Shared ANC (Automated Node Checkout) machinery for the ANC CVS suites.

This module holds the logic reused across the ANC suites so each suite file
stays thin:

  - Package install (``install_anc``) dispatched by release-archive flavour
    (``-deb-`` / ``-rpm-`` / ``-tar-``), with an optional version precheck /
    post-verify driven by ``config["anc"]["anc_version"]``.
  - Group execution (``run_anc_groups``): run one or more ANC groups in a
    single ``anc.py -g <groups...>`` invocation on all nodes, collect the
    per-run artifacts, and judge pass/fail from the final ANC return code.

The CPU/GPU group sets consumed by the ``anc_test_cpu`` / ``anc_test_gpu``
suites live here (``CPU_GROUPS`` / ``GPU_GROUPS``).
'''

import os
import re

from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import (
    fail_test,
    update_test_result,
    print_test_output,
)
from cvs.lib import globals
from cvs.lib.run_config_paths import resolve_runner_results_base

log = globals.log

# --- Install locations ---------------------------------------------------
ANC_TOOLS_PREFIX = "/opt/amdtools"
# ANC directory and entrypoint laid down by the deb/rpm packages.
ANC_DIR = f"{ANC_TOOLS_PREFIX}/anc"
ANC_BIN = f"{ANC_DIR}/anc.py"

# validate_exe_paths.py ships beside the anc test suites.
RESOURCES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "anc", "resources",
)

# --- Group sets ----------------------------------------------------------
# CPU validation groups run in a single anc.py -g invocation.
CPU_GROUPS = [
    "ampttk_full",
    "cachewalker_full",
    "cpu_all",
    "cpu_content_check",
    "cpu_mfg_l10",
    "cpu_sanity",
    "difect_full",
    "fpdeluge_full",
    "hdrt_full",
    "maxcorestim_full",
    "memtest_full",
    "miidct_full",
    "mithac_full",
    "weighted_sanity",
]

# GPU validation groups run in a single anc.py -g invocation.
GPU_GROUPS = [
    "gpu_content_check",
    "gpu_mfg_l10",
    "hbm_lvl1",
    "hbm_lvl2",
    "hbm_lvl3",
    "hbm_lvl4",
    "hbm_lvl5",
]

# --- Artifact / return-code parsing --------------------------------------
# ANC prints "Log Directory: <path>" near the top of stdout; every artifact
# (journal.log, console.log, summary.json, ...) is written under this dir.
LOG_DIRECTORY_RE = re.compile(r"Log Directory:\s*(\S+)")

# ANC records its return code near the end of console.log, e.g.
#   "Program exiting with return code ANC_SUCCESS [0]".
# The FINAL such line is authoritative: [0] (ANC_SUCCESS) is the only PASS.
ANC_RETURN_CODE_RE = re.compile(r"return code\s+(\S+)\s*\[(-?\d+)\]")

MANDATORY_ARTIFACTS = ("journal.log", "console.log")
OPTIONAL_ARTIFACTS = ("summary.json", "errors.json", "system_monitor.json")

# Default per-run execution timeout (2 hours); override via
# config["anc"]["test_timeout"].
DEFAULT_ANC_TEST_TIMEOUT = 7200

# --- ROCm shared-library resolution --------------------------------------
# ANC tools (e.g. computerocker) link against ROCm libs. On some nodes the .so
# files are present under these dirs but not in the dynamic-linker cache, so
# tools fail with "librocblas.so.5: cannot open shared object file". The
# ldconfig pre-task registers whichever of these dirs exist and refreshes the
# cache. Ordered so the truncating write targets the first existing dir.
ROCM_LIB_DIRS = ["/opt/rocm/lib", "/opt/rocm/lib64"]

# Library used to decide whether the ROCm dirs still need to be registered.
LDCONFIG_TARGET_LIB = "librocblas"


# =========================================================================
# Package installation
# =========================================================================
def detect_package_type(anc_release_url):
    '''
    Infer the ANC package flavour from the release archive name.

    Archive names embed the packaging kind as a ``-<type>-`` token, e.g.
    ``anc-release-helios-nda-1.4.7-deb-linux-x64.tar.gz`` -> ``deb``.

    Returns:
      str: one of ``"deb"``, ``"rpm"``, ``"tar"``.

    Raises:
      ValueError: if no known package token is present in the name.
    '''
    name = anc_release_url.rsplit("/", 1)[-1].lower()
    for pkg_type in ("deb", "rpm", "tar"):
        if f"-{pkg_type}-" in name or f"-{pkg_type}." in name:
            return pkg_type
    raise ValueError(
        f"Cannot determine ANC package type from archive name '{name}'; "
        f"expected a '-deb-', '-rpm-', or '-tar-' token"
    )


def node_version_matches(phdl, expected_version):
    '''
    Query ``ANC_BIN --version`` on every node and report version matches.

    Returns:
      dict[str, bool]: host -> True when ``expected_version`` appears in the
      node's ``anc.py --version`` output (False if ANC is absent or reports a
      different version).
    '''
    out_dict = phdl.exec(f"{ANC_BIN} --version 2>&1 || true", timeout=60)
    print_test_output(log, out_dict)
    return {
        host: (expected_version in (output or ""))
        for host, output in out_dict.items()
    }


def install_anc(phdl, config_dict):
    '''
    Install ANC on all nodes, dispatching by release-archive flavour.

    The packaging kind is inferred from the ``anc_release_url`` archive name
    (``-deb-`` / ``-rpm-`` / ``-tar-``) and the matching installer is invoked.
    When ``anc_version`` is set in config, a precheck runs first: if every node
    already reports that version via ``anc.py --version``, installation is
    skipped. After installing, the same version check runs as the final step to
    confirm the target version is present on all nodes.

    Reports pass/fail via fail_test / update_test_result (call from a test).
    '''
    globals.error_list = []

    anc_cfg = config_dict["anc"]
    anc_version = anc_cfg.get("anc_version")
    anc_release_url = anc_cfg["anc_release_url"]
    pkg_type = detect_package_type(anc_release_url)
    log.info("ANC package type detected from archive: %s", pkg_type)

    # Precheck: skip install if all nodes already run the target version.
    if anc_version:
        log.info("ANC precheck: expecting version %s", anc_version)
        matches = node_version_matches(phdl, anc_version)
        if matches and all(matches.values()):
            log.info(
                "ANC %s already installed on all nodes; skipping install",
                anc_version,
            )
            update_test_result()
            return

    if pkg_type == "deb":
        _install_anc_deb(phdl, config_dict)
    elif pkg_type == "rpm":
        _install_anc_rpm(phdl, config_dict)
    elif pkg_type == "tar":
        _install_anc_tar(phdl, config_dict)
    else:
        fail_test(f"ANC '{pkg_type}' package installation is not yet supported")
        update_test_result()
        return

    # Bail if the install phase already recorded failures.
    if globals.error_list:
        update_test_result()
        return

    # Final verification: confirm the target version is now on all nodes.
    if anc_version:
        log.info("ANC final verification: expecting version %s", anc_version)
        globals.error_list = []
        matches = node_version_matches(phdl, anc_version)
        for host, ok in matches.items():
            if ok:
                log.info("Node %s: ANC version %s confirmed", host, anc_version)
            else:
                fail_test(
                    f"ANC version verification failed on {host}: "
                    f"expected {anc_version} from '{ANC_BIN} --version'"
                )
        update_test_result()


def _install_anc_rpm(phdl, config_dict):
    '''
    Install ANC from .rpm package(s) on remote nodes (fresh install each run).

    Downloads and extracts the release tarball, installs the resulting
    ``anc*.rpm`` files with ``dnf install`` (deps resolved from configured
    repos), then smoke-tests with ``ANC_BIN --help``.
    '''
    globals.error_list = []

    anc_cfg = config_dict["anc"]
    cvs_home = anc_cfg["cvs_home"]
    anc_release_url = anc_cfg["anc_release_url"]
    tarball = anc_release_url.rsplit("/", 1)[-1]

    log.info("ANC: install .rpm package(s) on remote nodes (cvs_home=%s)",
             cvs_home)

    install_cmd = (
        f"set -e && "
        f"mkdir -p '{cvs_home}' && cd '{cvs_home}' && "
        f"echo 'Downloading ANC release...' && "
        f"wget -q '{anc_release_url}' -O '{tarball}' && "
        f"echo 'Extracting outer archive...' && "
        f"tar -xzf '{tarball}' && "
        f"rm -f '{tarball}' && "
        # Install via dnf so declared deps are resolved from the repos.
        f"echo 'Installing ANC .rpm package(s)...' && "
        f"rpms=$(ls anc*.rpm 2>/dev/null) && "
        f"if [ -z \"$rpms\" ]; then echo 'NO_RPM_FOUND' && exit 1; fi && "
        f"sudo dnf install -y $rpms && "
        f"rm -f $rpms && "
        f"echo '--- Verification ---' && "
        f"{ANC_BIN} --help && echo 'ANC_INSTALL_SUCCESS'"
    )

    out_dict = phdl.exec(install_cmd, timeout=300)
    print_test_output(log, out_dict)

    for host, output in out_dict.items():
        if "NO_RPM_FOUND" in output:
            fail_test(
                f"ANC .rpm installation failed on {host}: "
                f"no anc*.rpm found after extracting '{tarball}'"
            )
        elif "ANC_INSTALL_SUCCESS" not in output:
            fail_test(
                f"ANC .rpm installation failed on {host}: "
                f"'{ANC_BIN} --help' did not succeed"
            )
        else:
            log.info("Node %s: ANC .rpm installed and verified", host)

    update_test_result()


def _install_anc_deb(phdl, config_dict):
    '''
    Install ANC from .deb package(s) on remote nodes (fresh install each run).

    Downloads and extracts the release tarball, installs the resulting
    ``anc*.deb`` files with ``dpkg -i``, then smoke-tests with
    ``ANC_BIN --help``.
    '''
    globals.error_list = []

    anc_cfg = config_dict["anc"]
    cvs_home = anc_cfg["cvs_home"]
    anc_release_url = anc_cfg["anc_release_url"]
    tarball = anc_release_url.rsplit("/", 1)[-1]

    log.info("ANC: install .deb package(s) on remote nodes (cvs_home=%s)",
             cvs_home)

    install_cmd = (
        f"set -e && "
        f"mkdir -p '{cvs_home}' && cd '{cvs_home}' && "
        f"echo 'Downloading ANC release...' && "
        f"wget -q '{anc_release_url}' -O '{tarball}' && "
        f"echo 'Extracting outer archive...' && "
        f"tar -xzf '{tarball}' && "
        f"rm -f '{tarball}' && "
        # python3 is present on these nodes but not tracked in dpkg's database
        # (installed outside apt/dpkg), so a plain `dpkg -i` aborts on the
        # unsatisfiable "depends on python3" check. Force past dependency
        # problems and then configure so the tool is set up regardless.
        f"echo 'Installing ANC .deb package(s)...' && "
        f"debs=$(ls anc*.deb 2>/dev/null) && "
        f"if [ -z \"$debs\" ]; then echo 'NO_DEB_FOUND' && exit 1; fi && "
        f"sudo dpkg -i --force-depends $debs && "
        f"sudo dpkg --configure --force-depends -a && "
        f"rm -f $debs && "
        f"echo '--- Verification ---' && "
        f"{ANC_BIN} --help && echo 'ANC_INSTALL_SUCCESS'"
    )

    out_dict = phdl.exec(install_cmd, timeout=300)
    print_test_output(log, out_dict)

    for host, output in out_dict.items():
        if "NO_DEB_FOUND" in output:
            fail_test(
                f"ANC .deb installation failed on {host}: "
                f"no anc*.deb found after extracting '{tarball}'"
            )
        elif "ANC_INSTALL_SUCCESS" not in output:
            fail_test(
                f"ANC .deb installation failed on {host}: "
                f"'{ANC_BIN} --help' did not succeed"
            )
        else:
            log.info("Node %s: ANC .deb installed and verified", host)

    update_test_result()


def _install_anc_tar(phdl, config_dict):
    '''
    Install ANC from the tar release on remote nodes (fresh install each run).

    Extracts anc-tool and anc-content into cvs_home, removes
    ``anc_root_dir/content/base``, then validates exe_path entries with the
    bundled validate_exe_paths.py script.
    '''
    globals.error_list = []

    anc_cfg = config_dict["anc"]
    cvs_home = anc_cfg["cvs_home"]
    anc_root_dir = anc_cfg["anc_root_dir"]
    anc_release_url = anc_cfg["anc_release_url"]
    tarball = anc_release_url.rsplit("/", 1)[-1]

    log.info("ANC: install tar release on remote nodes (cvs_home=%s, root=%s)",
             cvs_home, anc_root_dir)

    install_cmd = (
        f"set -e && "
        f"mkdir -p '{cvs_home}' && cd '{cvs_home}' && "
        f"if [ -d '{anc_root_dir}' ]; then "
        f"echo 'Removing existing {anc_root_dir}...' && "
        f"rm -rf '{anc_root_dir}'; "
        f"fi && "
        f"echo 'Downloading ANC release...' && "
        f"wget -q '{anc_release_url}' -O '{tarball}' && "
        f"echo 'Extracting outer archive...' && "
        f"tar -xzf '{tarball}' && "
        f"rm -f '{tarball}' && "
        f"echo 'Extracting anc-tool archive to {cvs_home}...' && "
        f"tar -xzf anc-tool*.tar.gz && "
        f"rm -f anc-tool*.tar.gz && "
        f"echo 'Extracting anc-content archive to {cvs_home}...' && "
        f"tar -xzf anc-content*.tar.gz && "
        f"rm -f anc-content*.tar.gz && "
        f"echo 'Removing {anc_root_dir}/content/base...' && "
        f"rm -rf '{anc_root_dir}/content/base' && "
        f"echo '--- Directory listing (2 levels) ---' && "
        f"find . -maxdepth 2 -type d | sort && "
        f"echo '--- Verification ---' && "
        f"test -d '{anc_root_dir}' && echo 'ANC_INSTALL_SUCCESS'"
    )

    out_dict = phdl.exec(install_cmd, timeout=300)
    print_test_output(log, out_dict)

    for host, output in out_dict.items():
        if "ANC_INSTALL_SUCCESS" not in output:
            fail_test(
                f"ANC installation failed on {host}: "
                f"'{anc_root_dir}' not found under {cvs_home}"
            )
        else:
            log.info("Node %s: ANC directory installed successfully", host)

    if globals.error_list:
        update_test_result()
        return

    # Validate exe_path entries with the bundled script.
    content_dir = f"{cvs_home}/{anc_root_dir}/content"
    local_script = os.path.join(RESOURCES_DIR, "validate_exe_paths.py")
    remote_script = "/tmp/validate_exe_paths.py"

    log.info("Uploading validation script to remote nodes...")
    phdl.upload_file(local_script, remote_script)

    log.info("Validating exe_path entries from %s", content_dir)
    validate_dict = phdl.exec(
        f"python3 {remote_script} '{content_dir}'", timeout=60
    )
    print_test_output(log, validate_dict)

    log.info("Removing validation script from remote nodes...")
    phdl.exec(f"rm -f {remote_script}", timeout=10)

    for host, output in validate_dict.items():
        if "VALIDATION_FAILED" in output:
            fail_test(
                f"exe_path validation failed on {host}: "
                f"one or more tool paths are missing"
            )
        elif "VALIDATION_SUCCESS" in output:
            log.info("Node %s: All exe_path entries validated", host)
        elif "ERROR:" in output:
            fail_test(f"Validation error on {host}: {output.strip()}")
        else:
            fail_test(f"Unexpected validation output on {host}: {output.strip()}")

    update_test_result()


# =========================================================================
# ROCm shared-library resolution (ldconfig)
# =========================================================================
def ensure_rocm_ldconfig(phdl):
    '''
    Make ROCm shared libraries resolvable on every node before ANC runs.

    ANC tools link against ROCm libs (e.g. librocblas). On some nodes the .so
    files exist under ROCM_LIB_DIRS but are not in the dynamic-linker cache, so
    tools fail with "librocblas.so.5: cannot open shared object file". This
    pre-task, per node:

      1. If the target lib (librocblas) already resolves via ``ldconfig -p``,
         nothing to do -> OK.
      2. Otherwise locate the lib under ROCM_LIB_DIRS. If it is NOT present on
         disk, this is a real missing-library error (not a cache issue) -> FAIL.
      3. If present, register the existing ROCM_LIB_DIRS in
         /etc/ld.so.conf.d/rocm.conf, run ``sudo ldconfig``, and re-check that
         the lib now resolves -> OK, else FAIL.

    Reports pass/fail via fail_test / update_test_result (call from a test).
    '''
    globals.error_list = []
    log.info("ANC Pre-Task: ensure ROCm libs resolvable (%s)",
             LDCONFIG_TARGET_LIB)

    # Build the conf file from whichever ROCM_LIB_DIRS exist (first write
    # truncates, rest append), then ldconfig and re-verify.
    conf_lines = " && ".join(
        (
            f"if [ -d '{d}' ]; then echo '{d}' | "
            f"sudo tee {'-a ' if i else ''}/etc/ld.so.conf.d/rocm.conf "
            f">/dev/null; fi"
        )
        for i, d in enumerate(ROCM_LIB_DIRS)
    )

    check = (
        f"if ldconfig -p 2>/dev/null | grep -q '{LDCONFIG_TARGET_LIB}'; then "
        f"echo 'LDCONFIG_OK_ALREADY'; "
        f"else "
        # Not cached: is it on disk anywhere under the ROCm lib dirs?
        f"found=''; "
        f"for d in {' '.join(ROCM_LIB_DIRS)}; do "
        f"if ls \"$d\"/{LDCONFIG_TARGET_LIB}* >/dev/null 2>&1; then found=$d; fi; "
        f"done; "
        f"if [ -z \"$found\" ]; then echo 'LDCONFIG_LIB_MISSING'; else "
        f"{conf_lines} && sudo ldconfig && "
        f"if ldconfig -p 2>/dev/null | grep -q '{LDCONFIG_TARGET_LIB}'; then "
        f"echo 'LDCONFIG_FIXED'; else echo 'LDCONFIG_STILL_MISSING'; fi; "
        f"fi; "
        f"fi"
    )

    out_dict = phdl.exec(check, timeout=120)
    print_test_output(log, out_dict)

    for host, output in out_dict.items():
        text = output or ""
        if "LDCONFIG_OK_ALREADY" in text:
            log.info("Node %s: %s already resolvable", host, LDCONFIG_TARGET_LIB)
        elif "LDCONFIG_FIXED" in text:
            log.info("Node %s: registered ROCm lib dirs and ran ldconfig; "
                     "%s now resolvable", host, LDCONFIG_TARGET_LIB)
        elif "LDCONFIG_LIB_MISSING" in text:
            fail_test(
                f"ROCm library {LDCONFIG_TARGET_LIB} not found under "
                f"{ROCM_LIB_DIRS} on {host}: library is genuinely missing "
                f"(not an ldconfig cache issue)"
            )
        elif "LDCONFIG_STILL_MISSING" in text:
            fail_test(
                f"ROCm library {LDCONFIG_TARGET_LIB} still unresolved on "
                f"{host} after ldconfig"
            )
        else:
            fail_test(
                f"Unexpected ldconfig check output on {host}: {text.strip()}"
            )

    update_test_result()


# =========================================================================
# Group execution + artifact collection
# =========================================================================
def _sanitize_path_component(value):
    '''Make a string safe to use as a single filesystem path component.'''
    return re.sub(r"[^\w.\-]", "_", value)


def _node_label(single, host, cluster_dict):
    '''Build the "<ip>_<hostname>" artifact folder label for a node.'''
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
    '''chown ANC's (root-owned) log dir back to the SSH user for SFTP.'''
    try:
        single.exec(f"sudo chown -R {user} '{log_dir}'", timeout=60)
    except Exception as exc:  # best-effort; download reports if still locked
        log.warning("Node %s: could not chown log dir %s: %s", host, log_dir,
                    exc)


def _list_remote_files(single, host, log_dir):
    '''List file names directly under log_dir (None on SSH error).'''
    try:
        out = single.exec(f"ls -1 '{log_dir}' 2>/dev/null", timeout=30)
    except Exception as exc:
        log.warning("Node %s: could not list log dir %s: %s", host, log_dir, exc)
        return None
    return set((out.get(host, "") or "").split())


def _download_artifact(single, host, remote_path, dest_dir, name):
    '''Download a single artifact (None on failure).'''
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

    rc_matches = ANC_RETURN_CODE_RE.findall(console_text)
    if not rc_matches:
        log.error("Node %s: ANC return code not found in console.log", host)
        return "ANC return code not found in console.log"

    rc_name, rc_value = rc_matches[-1][0], int(rc_matches[-1][1])
    log.info("Node %s: ANC %s return code is %s [%s]", host, test_name,
             rc_name, rc_value)
    if rc_value != 0:
        return f"non-zero ANC return code {rc_name} [{rc_value}]"

    return None


def run_anc_groups(phdl, cluster_dict, config_dict, groups, test_name):
    '''
    Run one or more ANC groups in a single invocation on all nodes.

    Executes ``cd <ANC_DIR> && sudo ./anc.py -g <groups...>`` on every node,
    collects journal.log/console.log (mandatory) plus summary.json/errors.json/
    system_monitor.json (when present), and PASSES only when every node reports
    a final ANC_SUCCESS [0] return code. Failures across parallel nodes are
    aggregated into a SINGLE test failure.

    Parameters:
      groups:    list of ANC group names passed to ``anc.py -g``.
      test_name: name used for the artifact subpath (e.g. "test_cpu").
    '''
    globals.error_list = []

    anc_cfg = config_dict["anc"]
    timeout = anc_cfg.get("test_timeout", DEFAULT_ANC_TEST_TIMEOUT)
    base = resolve_runner_results_base(config_dict.get("run_config", {}))
    expected_nodes = list(cluster_dict["node_dict"].keys())
    groups_arg = " ".join(groups)

    cmd = f"cd '{ANC_DIR}' && sudo ./anc.py -g {groups_arg}"
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

    if failed_nodes:
        details = "; ".join(f"{h}: {r}" for h, r in failed_nodes.items())
        fail_test(
            f"ANC {test_name} failed on {len(failed_nodes)}/"
            f"{len(expected_nodes)} node(s): {details}"
        )

    update_test_result()
