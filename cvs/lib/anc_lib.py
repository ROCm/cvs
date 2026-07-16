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
import tarfile
from datetime import datetime

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
# ANC prints "Log directory: <path>" (case varies) near the end of its run;
# every artifact (journal.log, console.log, summary.json, ...) lives under it.
LOG_DIRECTORY_RE = re.compile(r"Log directory:\s*(\S+)", re.IGNORECASE)

# ANC records its final program return code, e.g.
#   "Program exiting with return code ANC_SUCCESS [0]".
# The FINAL such line is authoritative: [0] (ANC_SUCCESS) is the only PASS.
ANC_RETURN_CODE_RE = re.compile(r"return code\s+(\S+)\s*\[(-?\d+)\]")

# The "Items: N Total | ... PASSED, ... FAILED, ..." summary line in the ANC
# run summary; surfaced verbatim so the CVS result shows the pass/fail counts.
ANC_ITEMS_SUMMARY_RE = re.compile(r"^\s*Items:\s*\d+\s*Total\b.*$", re.MULTILINE)

# A per-item FAILED row in the ANC Results Details table, e.g.
#   "FAILED      i305 mithac      VENICE ES  ANC_ITEM_TIMEOUT".
ANC_FAILED_ITEM_RE = re.compile(r"^\s*FAILED\s+\S+.*$", re.MULTILINE)

# console.log is the only artifact we must have (holds the verdict). Everything
# else is pulled best-effort as part of the whole-directory copy.
CONSOLE_LOG = "console.log"

# --- Config-driven console + log-path behaviour --------------------------
# When config anc.print_all_to_console is falsey, the (potentially huge) ANC
# group-run output is suppressed from the console; install/ldconfig/version
# diagnostics always print. Default: print everything.
PRINT_ALL_TO_CONSOLE_KEY = "print_all_to_console"

# config anc.log_folder_path is where the ANC log directory is copied. It may
# contain "{runner_log_folder}" (resolved to run_config's runner_log_folder),
# "<test_name>" (the suite's test name, e.g. test_cpu), and "<timestamp>" (a
# per-run stamp so repeated runs of the same test do not overwrite each other).
LOG_FOLDER_PATH_KEY = "log_folder_path"
DEFAULT_LOG_FOLDER_PATH = "{runner_log_folder}/anc_logs/<test_name>/<timestamp>"


def _as_bool(value, default=True):
    '''Interpret a config value (str/bool/None) as a boolean.'''
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def new_run_timestamp():
    '''Return a filesystem-safe timestamp (YYYYmmdd-HHMMSS) for a run folder.'''
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_anc_log_folder(config_dict, test_name, timestamp):
    '''
    Resolve config anc.log_folder_path into an absolute destination directory.

    Substitutes "{runner_log_folder}" (from run_config, via
    resolve_runner_results_base), "<test_name>", and "<timestamp>". Falls back
    to DEFAULT_LOG_FOLDER_PATH when the key is unset. When the template has no
    "<timestamp>" token, the stamp is appended so repeated runs stay separate.
    '''
    template = config_dict.get("anc", {}).get(
        LOG_FOLDER_PATH_KEY, DEFAULT_LOG_FOLDER_PATH
    )
    runner_base = resolve_runner_results_base(config_dict.get("run_config", {}))
    path = template.replace("{runner_log_folder}", runner_base)
    path = path.replace("<test_name>", test_name)
    if "<timestamp>" in path:
        path = path.replace("<timestamp>", timestamp)
    else:
        path = os.path.join(path, timestamp)
    if "~" in path:
        path = os.path.expanduser(path)
    return os.path.abspath(path)

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


def _pull_log_dir(single, host, user, log_dir, dest_dir):
    '''
    Copy the ENTIRE ANC log directory from the node into dest_dir.

    ANC writes every artifact (console.log, journal.log, per-item logs, json
    summaries, ...) under its "Log directory". Rather than cherry-picking
    files, tar the whole directory on the node, download the tarball, and
    extract it locally so the full ANC log tree is preserved.

    Returns:
      tuple[str | None, str | None]: (console_log_path, failure_reason).
      console_log_path is the local path to the extracted console.log on
      success; failure_reason is set (and path None) on any collection error.
    '''
    # ANC ran under sudo; reclaim ownership so the plain SSH user can read/tar.
    _claim_remote_logs(single, host, user, log_dir)

    log_dir = log_dir.rstrip("/")
    base = os.path.basename(log_dir)
    parent = os.path.dirname(log_dir)
    remote_tar = f"/tmp/anc_{base}.tar.gz"

    try:
        single.exec(f"tar -czf '{remote_tar}' -C '{parent}' '{base}'",
                    timeout=600)
    except Exception as exc:
        return None, f"could not archive log dir {log_dir}: {exc}"

    local_tar = os.path.join(dest_dir, f"{base}.tar.gz")
    try:
        single.download_file(remote_tar, local_tar)
    except Exception as exc:
        return None, f"could not download log archive from {log_dir}: {exc}"
    finally:
        try:
            single.exec(f"rm -f '{remote_tar}'", timeout=30)
        except Exception as exc:  # best-effort cleanup
            log.warning("Node %s: could not remove remote tar %s: %s", host,
                        remote_tar, exc)

    try:
        with tarfile.open(local_tar) as tf:
            tf.extractall(dest_dir)
    except Exception as exc:
        return None, f"could not extract log archive for {log_dir}: {exc}"
    finally:
        try:
            os.remove(local_tar)
        except OSError:
            pass

    extracted_dir = os.path.join(dest_dir, base)
    log.info("Node %s: ANC logs copied to %s", host, extracted_dir)

    console_path = os.path.join(extracted_dir, CONSOLE_LOG)
    if not os.path.isfile(console_path):
        return None, f"{CONSOLE_LOG} not found under {log_dir}"
    return console_path, None


def _summarize_failure(console_text):
    '''
    Build a concise failure detail string from ANC's console.log summary.

    Pulls the "Items: N Total | ..." summary line plus any per-item FAILED
    rows so the CVS failure message shows both the counts and which items
    failed.
    '''
    parts = []
    items = ANC_ITEMS_SUMMARY_RE.search(console_text)
    if items:
        parts.append(items.group(0).strip())
    failed_rows = [m.strip() for m in ANC_FAILED_ITEM_RE.findall(console_text)]
    if failed_rows:
        parts.append("failed items: " + " | ".join(failed_rows))
    return "; ".join(parts)


def _evaluate_node(cluster_dict, host, output, log_base, test_name):
    '''
    Collect the ANC log directory for one node and decide whether it passed.

    ``output`` is the captured ANC console text (full when
    print_all_to_console, else just the "Log directory:" line) used only to
    locate the log-directory path. The whole log directory is then copied under
    ``<log_base>/<ip>_<hostname>/`` and the verdict is taken from console.log's
    final "Program exiting with return code ANC_SUCCESS [0]" line. On failure,
    the item summary and FAILED rows are surfaced.

    Returns:
      str | None: A failure reason for this node, or None if the node passed.
    '''
    ld_match = LOG_DIRECTORY_RE.search(output or "")
    if not ld_match:
        log.error("Node %s: could not determine Log directory from output", host)
        return "could not determine Log directory from ANC output"
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
    dest_dir = os.path.join(log_base, label)
    os.makedirs(dest_dir, exist_ok=True)

    console_path, infra_reason = _pull_log_dir(
        single, host, cluster_dict["username"], log_dir, dest_dir
    )
    if infra_reason:
        return infra_reason

    try:
        with open(console_path, encoding="utf-8", errors="replace") as fh:
            console_text = fh.read()
    except Exception as exc:
        return f"could not read collected console.log: {exc}"

    rc_matches = ANC_RETURN_CODE_RE.findall(console_text)
    if not rc_matches:
        log.error("Node %s: ANC return code not found in console.log", host)
        return "ANC return code not found in console.log"

    rc_name, rc_value = rc_matches[-1][0], int(rc_matches[-1][1])
    log.info("Node %s: ANC %s program return code is %s [%s]", host, test_name,
             rc_name, rc_value)
    if rc_value != 0:
        detail = _summarize_failure(console_text)
        reason = f"ANC returned {rc_name} [{rc_value}]"
        if detail:
            reason = f"{reason}; {detail}"
        return reason

    return None


def run_anc_groups(phdl, cluster_dict, config_dict, groups, test_name):
    '''
    Run one or more ANC groups in a single invocation on all nodes.

    Executes ``cd <ANC_DIR> && sudo ./anc.py -g <groups...>`` on every node,
    copies the ENTIRE ANC log directory to the configured log_folder_path
    (``{runner_log_folder}/anc_logs/<test_name>`` by default, under a per-node
    ``<ip>_<hostname>/`` subdir), and PASSES only when every node's console.log
    ends with ANC_SUCCESS [0]. On failure the item summary and FAILED rows are
    surfaced. Failures across parallel nodes are aggregated into a SINGLE test
    failure.

    Console behaviour is config-driven: when anc.print_all_to_console is truthy
    (default), the full ANC group output is echoed to the console; when falsey,
    ANC output is suppressed and only the "Log directory:" line is captured
    (the verdict always comes from the collected console.log either way).

    Parameters:
      groups:    list of ANC group names passed to ``anc.py -g``.
      test_name: name used for the log path and messages (e.g. "test_cpu").
    '''
    globals.error_list = []

    anc_cfg = config_dict["anc"]
    timeout = anc_cfg.get("test_timeout", DEFAULT_ANC_TEST_TIMEOUT)
    print_all = _as_bool(anc_cfg.get(PRINT_ALL_TO_CONSOLE_KEY), default=True)
    timestamp = new_run_timestamp()
    log_base = resolve_anc_log_folder(config_dict, test_name, timestamp)
    os.makedirs(log_base, exist_ok=True)
    expected_nodes = list(cluster_dict["node_dict"].keys())
    groups_arg = " ".join(groups)

    # Announce the resolved (substituted) log directory up front so the user
    # knows exactly where this run's ANC logs will land.
    log.info("=" * 72)
    log.info("ANC '%s': logs for this run -> %s", test_name, log_base)
    log.info("=" * 72)

    if print_all:
        # Run ANC normally so its full output streams back, and print it.
        cmd = f"cd '{ANC_DIR}' && sudo ./anc.py -g {groups_arg}"
    else:
        # Suppress the (potentially huge) ANC output: redirect stdout/stderr to
        # a per-run file on the node and echo ONLY the "Log directory:" line.
        # The verdict is sourced from the collected console.log regardless.
        remote_stdout = "/tmp/anc_run_$$.out"
        cmd = (
            f"cd '{ANC_DIR}' && "
            f"sudo ./anc.py -g {groups_arg} > '{remote_stdout}' 2>&1; "
            f"grep -i 'Log directory:' '{remote_stdout}' | tail -1; "
            f"rm -f '{remote_stdout}'"
        )

    log.info("ANC '%s': running %d group(s) (print_all_to_console=%s, "
             "timeout=%ss, logs under %s)", test_name, len(groups), print_all,
             timeout, log_base)
    log.info("ANC '%s': groups=%s", test_name, groups_arg)

    try:
        out_dict = phdl.exec(cmd, timeout=timeout)
    except Exception as exc:  # infra failure must fail the test
        fail_test(f"ANC {test_name}: execution failed (SSH/exec error): {exc}")
        update_test_result()
        return

    # Echo full ANC output only when print_all_to_console is enabled.
    if print_all:
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
        reason = _evaluate_node(cluster_dict, host, out_dict[host] or "",
                                log_base, test_name)
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


# =========================================================================
# Per-group test scaffolding
# =========================================================================
class AncGroupTest:
    '''
    Base class for a single-ANC-group suite.

    A per-group suite subclasses this and sets ``GROUP`` to the ANC group name.
    Running the suite installs/verifies ANC and fixes ROCm ldconfig as
    pre-tasks, then runs just that one group. The test name (and therefore the
    log subpath ``.../anc_logs/test_<GROUP>/<timestamp>``) is ``test_<GROUP>``.

    The per-group suite files under cpu/ and gpu/ are GENERATED from the group
    lists in this module by build_tools/gen_anc_suites.py (``make
    gen-anc-suites``); do not hand-edit them. To add/remove a group, edit
    CPU_GROUPS / GPU_GROUPS here and regenerate.

    Fixtures cluster_dict / config_dict / phdl come from the ANC conftest.py.
    '''

    GROUP = None  # set by subclass, e.g. "cpu_all"

    def test_install_anc(self, phdl, config_dict):
        '''Pre-task: install/verify ANC before running the group.'''
        log.info("ANC group '%s' Pre-Task: install/verify ANC", self.GROUP)
        install_anc(phdl, config_dict)

    def test_rocm_ldconfig(self, phdl):
        '''Pre-task: ensure ROCm libs are resolvable before running the group.'''
        log.info("ANC group '%s' Pre-Task: ensure ROCm ldconfig", self.GROUP)
        ensure_rocm_ldconfig(phdl)

    def test_group(self, phdl, cluster_dict, config_dict):
        '''Run this suite's single ANC group and collect/judge its logs.'''
        assert self.GROUP, "AncGroupTest subclass must set GROUP"
        run_anc_groups(
            phdl, cluster_dict, config_dict, [self.GROUP], f"test_{self.GROUP}"
        )


def anc_group_class_name(group):
    '''Return the CamelCase test-class name for a group (e.g. cpu_all->CpuAll).'''
    return "".join(part.capitalize() for part in group.split("_"))
