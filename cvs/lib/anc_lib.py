'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

'''
Shared ANC (AMD Node Check) machinery for the ANC CVS suites.

This module holds the logic reused across the ANC suites so each suite file
stays thin:

  - Package install (``install_anc``) dispatched by release-archive flavour
    (``-deb-`` / ``-rpm-`` / ``-tar-``), with an optional version precheck /
    post-verify driven by ``config["anc"]["anc_version"]``.
  - Group execution (``run_anc_groups``): run one or more ANC groups in a
    single ``anc.py -g <groups...>`` invocation on all nodes, collect the
    per-run artifacts, and judge pass/fail from the final ANC return code.
  - Setup guard (``ensure_anc_ready``): session-cached install + ldconfig that
    each group test calls first, so a single-group run self-installs and a
    full-suite run pays the setup cost once.

The CPU/GPU group sets consumed by the ``anc_test_cpu`` / ``anc_test_gpu``
suites live here (``CPU_GROUPS`` / ``GPU_GROUPS``). The per-group test
functions in those two suite files are GENERATED from these lists by
build_tools/gen_anc_suites.py (``make gen-anc-suites``); do not hand-edit them.
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
    "tests",
    "anc",
    "resources",
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

# ANC reports an unknown group both as a FATAL line and a dedicated return code:
#   "FATAL: Group 'foo' not found"
#   "Program exiting with return code ANC_PROG_NOT_FOUND [13]"
# Either is sufficient to conclude the requested group is not installed on that
# node; matching both covers print_all_to_console on (raw FATAL line streams
# back) and off (only console.log is collected, which carries the return code).
ANC_GROUP_NOT_FOUND_RE = re.compile(r"FATAL:\s*Group\s+'([^']*)'\s+not found", re.IGNORECASE)
ANC_PROG_NOT_FOUND_NAME = "ANC_PROG_NOT_FOUND"

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
# "<node>" (the per-node "<ip>_<hostname>" label), "<test_name>" (the suite's
# test name, e.g. test_cpu), and "<timestamp>" (a per-run stamp so repeated runs
# of the same test do not overwrite each other). "<node>" comes first by default
# so a multi-node run groups every test/timestamp under each node's own folder.
LOG_FOLDER_PATH_KEY = "log_folder_path"
DEFAULT_LOG_FOLDER_PATH = "{runner_log_folder}/anc_logs/<node>/<test_name>/<timestamp>"

# config anc.ADD_ANC_LOGS_TO_HTML_REPORTS controls whether the collected ANC
# log tree is bundled into the pytest-html report zip. When True, always attach.
# When False (default), attach ONLY when the test failed (so passing runs stay
# lean but failures always carry their evidence).
ADD_ANC_LOGS_TO_HTML_KEY = "ADD_ANC_LOGS_TO_HTML_REPORTS"

# config anc.COLLECT_HTML_REPORTS ("True" by default) makes the ANC suites
# generate a pytest-html report even when no --html is passed on the command
# line. config anc.html_report_path is the destination template; it accepts the
# same tokens as log_folder_path ("{runner_log_folder}", "<node>",
# "<test_name>", "<timestamp>") plus "{home}" (already resolved by the config
# placeholder pass). Because pytest-html creates ONE report per session before
# any SSH connection exists, "<node>" here resolves to the FIRST node in the
# cluster file's node_dict (label built from the cluster file alone, no SSH). An
# explicit --html on the command line always wins over this.
COLLECT_HTML_REPORTS_KEY = "COLLECT_HTML_REPORTS"
HTML_REPORT_PATH_KEY = "html_report_path"
DEFAULT_HTML_REPORT_PATH = "{runner_log_folder}/html_reports/<node>/<test_name>/<timestamp>"


def cluster_node_label_from_file(cluster_dict):
    '''
    Build the "<ip>_<hostname>" label for the FIRST node using only the cluster
    file (no SSH). Used for the session-level HTML report path, which must be
    resolved before any node connection exists. Returns "unknown_node" when the
    cluster file has no node_dict entries.
    '''
    node_dict = cluster_dict.get("node_dict", {}) or {}
    hosts = list(node_dict.keys())
    if not hosts:
        return "unknown_node"
    host = hosts[0]
    info = node_dict.get(host, {}) or {}
    ip = info.get("vpc_ip")
    if not ip or ip == "NA":
        ip = host
    # No SSH here, so hostname == the cluster-file host key (best-effort match to
    # the per-node log folder label, which appends the SSH-resolved hostname).
    return _sanitize_path_component(f"{ip}_{host}")


def resolve_anc_html_report_path(config_dict, cluster_dict, test_name, timestamp):
    '''
    Resolve config anc.html_report_path into an absolute *.html file path for the
    auto-collected pytest-html report. Substitutes "{runner_log_folder}",
    "<node>" (first cluster node's label, from the file only), "<test_name>", and
    "<timestamp>". The resolved template is treated as a directory; the report
    file "<test_name>.html" is placed inside it.
    '''
    template = config_dict.get("anc", {}).get(HTML_REPORT_PATH_KEY, DEFAULT_HTML_REPORT_PATH)
    runner_base = resolve_runner_results_base(config_dict.get("run_config", {}))
    node = cluster_node_label_from_file(cluster_dict)
    path = template.replace("{runner_log_folder}", runner_base)
    path = path.replace("<node>", node)
    path = path.replace("<test_name>", test_name)
    path = path.replace("<timestamp>", timestamp)
    # expanduser only expands a LEADING ~ (a mid-path ~ is a literal dir name,
    # not a home reference, and must not expand); it is a no-op otherwise.
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return os.path.join(path, f"{test_name}.html")


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


def resolve_anc_log_folder(config_dict, test_name, timestamp, node=None):
    '''
    Resolve config anc.log_folder_path into an absolute destination directory.

    Substitutes "{runner_log_folder}" (from run_config, via
    resolve_runner_results_base), "<node>" (the per-node "<ip>_<hostname>"
    label), "<test_name>", and "<timestamp>". Falls back to
    DEFAULT_LOG_FOLDER_PATH when the key is unset. When the template has no
    "<timestamp>" token, the stamp is appended so repeated runs stay separate.

    When ``node`` is None the "<node>" token is left intact (used for the
    run-start banner, which shows the pattern before any node is known). When a
    node label is given but the template lacks a "<node>" token, the label is
    appended so per-node artifacts never collide.
    '''
    template = config_dict.get("anc", {}).get(LOG_FOLDER_PATH_KEY, DEFAULT_LOG_FOLDER_PATH)
    runner_base = resolve_runner_results_base(config_dict.get("run_config", {}))
    path = template.replace("{runner_log_folder}", runner_base)
    path = path.replace("<test_name>", test_name)
    if node is not None:
        if "<node>" in path:
            path = path.replace("<node>", node)
        else:
            path = os.path.join(path, node)
    if "<timestamp>" in path:
        path = path.replace("<timestamp>", timestamp)
    else:
        path = os.path.join(path, timestamp)
    # expanduser only expands a LEADING ~ (a mid-path ~ is a literal dir name,
    # not a home reference, and must not expand); it is a no-op otherwise.
    path = os.path.expanduser(path)
    return os.path.abspath(path)


# Default per-run execution timeout (2 hours); override via
# config["anc"]["test_timeout"].
DEFAULT_ANC_TEST_TIMEOUT = 7200

# --- Package-install timeout + download progress -------------------------
# The install exec's timeout is parallel-ssh's read_timeout: an INACTIVITY
# (per-read) timeout, not a total wall-clock budget. It fires only when the SSH
# channel produces no output for this many seconds. ANC release archives are
# large (the 1.4.x content deb is ~550 MB) and download speed is unpredictable,
# so the download is the slowest, most variable step. Two things keep it from
# tripping this timeout spuriously:
#   1. The download runs with a periodic progress heartbeat (see
#      _download_with_progress_snippet) so the channel is never silent while
#      bytes are flowing, no matter how slow the link is. A genuine stall is
#      caught by wget's own --timeout/--tries (NOT by this exec's inactivity
#      timeout, which the heartbeat deliberately keeps from firing): a hung
#      read aborts wget non-zero and fails the install.
#   2. This default is generous (30 min) so the silent phases that follow
#      (dpkg/dnf install, tar extract of the large content payload) also fit.
# Override via config["anc"]["install_timeout"]. This is SCOPED TO INSTALL ONLY;
# the ANC group-run timeout (test_timeout) is unaffected.
DEFAULT_ANC_INSTALL_TIMEOUT = 1800
INSTALL_TIMEOUT_KEY = "install_timeout"

# How often (seconds) the download heartbeat prints partial-file size. Must be
# comfortably smaller than DEFAULT_ANC_INSTALL_TIMEOUT so the heartbeat keeps
# the channel active well within the inactivity window.
ANC_DOWNLOAD_PROGRESS_INTERVAL = 15


def _install_timeout(anc_cfg):
    '''Return the install exec inactivity timeout (config-overridable).'''
    return anc_cfg.get(INSTALL_TIMEOUT_KEY, DEFAULT_ANC_INSTALL_TIMEOUT)


def _download_with_progress_snippet(url, dest, interval=ANC_DOWNLOAD_PROGRESS_INTERVAL):
    '''
    Build a shell snippet that downloads ``url`` to ``dest`` while emitting a
    periodic progress line, and exits non-zero if the download fails.

    ``wget -q`` is silent for the entire (potentially multi-minute) download,
    which would trip the exec's inactivity timeout on a slow link. Instead run
    wget in the background and, every ``interval`` seconds, echo the partial
    file size so the SSH channel keeps producing output while bytes flow. The
    whole thing is wrapped in a subshell whose exit status is wget's (via
    ``wait``), so it composes with the surrounding ``set -e && ...`` chain: a
    failed download aborts the install exactly as a plain ``wget && `` would.

    Because the heartbeat keeps the SSH channel active, the exec's inactivity
    timeout no longer catches a hung download. wget's own ``--timeout`` /
    ``--tries`` do that instead: a stalled read aborts after a few minutes,
    wget exits non-zero, and the ``set -e`` chain fails rather than hanging
    until the (large) install_timeout elapses.

    ``url`` and ``dest`` are single-quoted here; callers validate config values
    for embedded single quotes (see _assert_shell_safe) before interpolation.
    '''
    return (
        f"( wget -q --timeout=60 --tries=3 '{url}' -O '{dest}' & "
        f"wpid=$!; "
        f"while kill -0 $wpid 2>/dev/null; do "
        f"sleep {interval}; "
        f"if [ -f '{dest}' ]; then "
        f"echo \"ANC download progress: $(du -h '{dest}' 2>/dev/null | cut -f1) downloaded\"; "
        f"fi; "
        f"done; "
        f"wait $wpid )"
    )


# --- ROCm shared-library resolution --------------------------------------
# ANC tools (e.g. computerocker) link against ROCm libs. On some nodes the .so
# files are present under these dirs but not in the dynamic-linker cache, so
# tools fail with "librocblas.so.5: cannot open shared object file". The
# ldconfig pre-task registers whichever of these dirs exist and refreshes the
# cache (the conf file is truncated once, then each existing dir is appended).
ROCM_LIB_DIRS = ["/opt/rocm/lib", "/opt/rocm/lib64"]

# Library used to decide whether the ROCm dirs still need to be registered.
LDCONFIG_TARGET_LIB = "librocblas"


# =========================================================================
# Package installation
# =========================================================================
def _assert_shell_safe(anc_cfg, keys):
    '''
    Reject config values that would break the single-quoted remote shell strings.

    The install commands wrap config values (cvs_home, anc_release_url, ...) in
    single quotes when interpolating them into the remote command. A value that
    itself contains a single quote would break that quoting. Config is trusted,
    so rather than shell-escape everywhere we validate at this boundary and fail
    fast with a clear message.
    '''
    for key in keys:
        value = anc_cfg.get(key)
        if value is not None and "'" in str(value):
            raise ValueError(
                f"ANC config value {key}={value!r} contains a single quote, "
                f"which is not supported (it would break the remote shell "
                f"command quoting). Remove the quote from the value."
            )


def _expected_nodes(cluster_dict):
    '''Return the full list of expected hosts from the cluster file.'''
    return list(cluster_dict.get("node_dict", {}).keys())


def _fail_unreachable_nodes(cluster_dict, out_dict, action):
    '''
    fail_test every expected node absent from an exec result (unreachable).

    ``phdl.exec`` only returns REACHABLE hosts, so a node that is down / did not
    respond simply drops out of ``out_dict``. Callers that judge pass/fail from
    ``out_dict`` alone would silently treat such a node as a success. This flags
    every expected host missing from ``out_dict`` as a failed ``action`` (e.g.
    "install", "ldconfig").

    Returns:
      list[str]: the expected hosts that were missing from ``out_dict``.
    '''
    expected = _expected_nodes(cluster_dict)
    missing = [host for host in expected if host not in (out_dict or {})]
    for host in missing:
        fail_test(f"ANC {action} failed on {host}: node produced no output (unreachable / did not run)")
    return missing


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
        f"Cannot determine ANC package type from archive name '{name}'; expected a '-deb-', '-rpm-', or '-tar-' token"
    )


def node_version_matches(phdl, expected_version):
    '''
    Query ``ANC_BIN --version`` on every node and report version matches.

    The version is matched as a whole token (bounded by non-version characters),
    not a substring, so expecting "1.4.7" does NOT match "1.4.70" or "1.4.7-rc".

    Returns:
      dict[str, bool]: host -> True when ``expected_version`` appears as a
      standalone version token in the node's ``anc.py --version`` output (False
      if ANC is absent or reports a different version).
    '''
    out_dict = phdl.exec(f"{ANC_BIN} --version 2>&1 || true", timeout=60)
    print_test_output(log, out_dict)
    # Boundaries are any char that isn't part of a version token (digits, dots,
    # hyphens, alphanumerics) so "1.4.7" won't match inside "1.4.70"/"1.4.7-rc".
    version_re = re.compile(r"(?<![\w.\-])" + re.escape(expected_version) + r"(?![\w.\-])")
    return {host: bool(version_re.search(output or "")) for host, output in out_dict.items()}


def install_anc(phdl, cluster_dict, config_dict):
    '''
    Install ANC on all nodes, dispatching by release-archive flavour.

    The packaging kind is inferred from the ``anc_release_url`` archive name
    (``-deb-`` / ``-rpm-`` / ``-tar-``) and the matching installer is invoked.
    When ``anc_version`` is set in config, a precheck runs first: install is
    skipped ONLY if every expected node already reports that version via
    ``anc.py --version``. After installing, the same version check runs as the
    final step to confirm the target version is present on all nodes.

    Node coverage: ``phdl.exec`` returns only reachable hosts, so any expected
    node (from ``cluster_dict["node_dict"]``) that is unreachable is treated as
    a failed install rather than silently passing.

    Reports pass/fail via fail_test / update_test_result (call from a test).
    '''
    globals.error_list = []

    anc_cfg = config_dict["anc"]
    # Values interpolated into single-quoted remote shell strings by the
    # sub-installers; reject quotes up front (config is trusted, boundary check).
    _assert_shell_safe(anc_cfg, ("cvs_home", "anc_release_url"))
    anc_version = anc_cfg.get("anc_version")
    anc_release_url = anc_cfg["anc_release_url"]
    pkg_type = detect_package_type(anc_release_url)
    log.info("ANC package type detected from archive: %s", pkg_type)

    expected = _expected_nodes(cluster_dict)

    # Precheck: skip install only when EVERY expected node already runs the
    # target version. A missing (unreachable) node yields matches.get(host)
    # == None -> falsy, so we never skip on incomplete coverage.
    if anc_version:
        log.info("ANC precheck: expecting version %s", anc_version)
        matches = node_version_matches(phdl, anc_version)
        if expected and all(matches.get(host) for host in expected):
            log.info(
                "ANC %s already installed on all nodes; skipping install",
                anc_version,
            )
            update_test_result()
            return

    # The sub-installers only record failures via fail_test (they do NOT report
    # the result themselves); install_anc is the single owner of
    # update_test_result so the whole install is reported exactly once. Each
    # sub-installer also flags any expected node missing from its output.
    if pkg_type == "deb":
        _install_anc_deb(phdl, cluster_dict, config_dict)
    elif pkg_type == "rpm":
        _install_anc_rpm(phdl, cluster_dict, config_dict)
    elif pkg_type == "tar":
        _install_anc_tar(phdl, cluster_dict, config_dict)
    else:
        fail_test(f"ANC '{pkg_type}' package installation is not yet supported")
        update_test_result()
        return

    # Final verification: confirm the target version is now on all expected
    # nodes. Skip it if the install phase already recorded failures (a bad
    # install can't pass version verification, and this keeps the failure
    # detail focused).
    if anc_version and not globals.error_list:
        log.info("ANC final verification: expecting version %s", anc_version)
        matches = node_version_matches(phdl, anc_version)
        for host in expected:
            if host not in matches:
                fail_test(
                    f"ANC version verification failed on {host}: node produced no output (unreachable / did not run)"
                )
            elif matches[host]:
                log.info("Node %s: ANC version %s confirmed", host, anc_version)
            else:
                fail_test(
                    f"ANC version verification failed on {host}: expected {anc_version} from '{ANC_BIN} --version'"
                )

    update_test_result()


def _install_anc_rpm(phdl, cluster_dict, config_dict):
    '''
    Install ANC from .rpm package(s) on remote nodes (fresh install each run).

    Downloads and extracts the release tarball, installs the resulting
    ``anc*.rpm`` files with ``dnf install`` (deps resolved from configured
    repos), then smoke-tests with ``ANC_BIN --help``.

    Records failures via fail_test; the caller (install_anc) owns the single
    update_test_result reporting call.
    '''
    anc_cfg = config_dict["anc"]
    cvs_home = anc_cfg["cvs_home"]
    anc_release_url = anc_cfg["anc_release_url"]
    tarball = anc_release_url.rsplit("/", 1)[-1]

    log.info("ANC: install .rpm package(s) on remote nodes (cvs_home=%s)", cvs_home)

    install_cmd = (
        f"set -e && "
        f"mkdir -p '{cvs_home}' && cd '{cvs_home}' && "
        f"echo 'Downloading ANC release...' && "
        f"{_download_with_progress_snippet(anc_release_url, tarball)} && "
        f"echo 'Extracting outer archive...' && "
        f"tar -xzf '{tarball}' && "
        f"rm -f '{tarball}' && "
        # Install via dnf so declared deps are resolved from the repos.
        f"echo 'Installing ANC .rpm package(s)...' && "
        # `|| true` so an empty glob doesn't abort the whole script under
        # `set -e` (failed command substitution); the -z check below reports it.
        f"rpms=$(ls anc*.rpm 2>/dev/null || true) && "
        f"if [ -z \"$rpms\" ]; then echo 'NO_RPM_FOUND' && exit 1; fi && "
        f"sudo dnf install -y $rpms && "
        f"rm -f $rpms && "
        f"echo '--- Verification ---' && "
        f"{ANC_BIN} --help && echo 'ANC_INSTALL_SUCCESS'"
    )

    out_dict = phdl.exec(install_cmd, timeout=_install_timeout(anc_cfg))
    print_test_output(log, out_dict)

    for host, output in out_dict.items():
        if "NO_RPM_FOUND" in output:
            fail_test(f"ANC .rpm installation failed on {host}: no anc*.rpm found after extracting '{tarball}'")
        elif "ANC_INSTALL_SUCCESS" not in output:
            fail_test(f"ANC .rpm installation failed on {host}: '{ANC_BIN} --help' did not succeed")
        else:
            log.info("Node %s: ANC .rpm installed and verified", host)

    _fail_unreachable_nodes(cluster_dict, out_dict, ".rpm install")


def _install_anc_deb(phdl, cluster_dict, config_dict):
    '''
    Install ANC from .deb package(s) on remote nodes (fresh install each run).

    Downloads and extracts the release tarball, installs the resulting
    ``anc*.deb`` files with ``dpkg -i``, then smoke-tests with
    ``ANC_BIN --help``.

    Records failures via fail_test; the caller (install_anc) owns the single
    update_test_result reporting call.
    '''
    anc_cfg = config_dict["anc"]
    cvs_home = anc_cfg["cvs_home"]
    anc_release_url = anc_cfg["anc_release_url"]
    tarball = anc_release_url.rsplit("/", 1)[-1]

    log.info("ANC: install .deb package(s) on remote nodes (cvs_home=%s)", cvs_home)

    install_cmd = (
        f"set -e && "
        f"mkdir -p '{cvs_home}' && cd '{cvs_home}' && "
        f"echo 'Downloading ANC release...' && "
        f"{_download_with_progress_snippet(anc_release_url, tarball)} && "
        f"echo 'Extracting outer archive...' && "
        f"tar -xzf '{tarball}' && "
        f"rm -f '{tarball}' && "
        # python3 is present on these nodes but not tracked in dpkg's database
        # (installed outside apt/dpkg), so a plain `dpkg -i` aborts on the
        # unsatisfiable "depends on python3" check. Force past dependency
        # problems and then configure so the tool is set up regardless.
        f"echo 'Installing ANC .deb package(s)...' && "
        # `|| true` so an empty glob doesn't abort the whole script under
        # `set -e` (failed command substitution); the -z check below reports it.
        f"debs=$(ls anc*.deb 2>/dev/null || true) && "
        f"if [ -z \"$debs\" ]; then echo 'NO_DEB_FOUND' && exit 1; fi && "
        f"sudo dpkg -i --force-depends $debs && "
        # Configure ONLY the ANC packages (by name), not `-a` (which would
        # configure every pending package on the node). Names come from the
        # .deb control field so we never touch unrelated half-installed pkgs.
        # `dpkg -i --force-depends` above already configures the packages in the
        # same step (force-depends lets its configure pass), so this is only a
        # fallback for the (rare) case where that configure was skipped and a
        # package is left half-configured. Guard on each package's dpkg status so
        # we DON'T re-configure already-configured packages: doing so prints
        # "already installed and configured" to stderr and exits non-zero (noisy
        # even when swallowed). Only packages not in "install ok installed" get a
        # configure pass; the `anc.py --help` check below is the real gate.
        f"pkgs=$(for d in $debs; do dpkg-deb -f \"$d\" Package; done) && "
        f"for p in $pkgs; do "
        f"if ! dpkg-query -W -f='${{Status}}' \"$p\" 2>/dev/null | grep -q 'install ok installed'; then "
        f"sudo dpkg --configure --force-depends \"$p\"; fi; done && "
        f"rm -f $debs && "
        f"echo '--- Verification ---' && "
        f"{ANC_BIN} --help && echo 'ANC_INSTALL_SUCCESS'"
    )

    out_dict = phdl.exec(install_cmd, timeout=_install_timeout(anc_cfg))
    print_test_output(log, out_dict)

    for host, output in out_dict.items():
        if "NO_DEB_FOUND" in output:
            fail_test(f"ANC .deb installation failed on {host}: no anc*.deb found after extracting '{tarball}'")
        elif "ANC_INSTALL_SUCCESS" not in output:
            fail_test(f"ANC .deb installation failed on {host}: '{ANC_BIN} --help' did not succeed")
        else:
            log.info("Node %s: ANC .deb installed and verified", host)

    _fail_unreachable_nodes(cluster_dict, out_dict, ".deb install")


def _install_anc_tar(phdl, cluster_dict, config_dict):
    '''
    Install ANC from the tar release on remote nodes (fresh install each run).

    The tar release's ``anc-tool`` / ``anc-content`` payloads are relocatable
    (top-level ``anc/`` and ``anc/content/`` + sibling tool dirs). Extracting
    BOTH into ``ANC_TOOLS_PREFIX`` (/opt/amdtools) reproduces exactly the layout
    the deb/rpm packages install (verified against the 1.4.7 packages), so ANC
    ends up at ``ANC_BIN`` (/opt/amdtools/anc/anc.py) and the content YAMLs'
    hard-coded ``exe_path: /opt/amdtools/...`` entries resolve. The release
    archive is staged in ``cvs_home`` only for the download/unpack; the tools
    themselves always live under /opt/amdtools regardless of flavour.

    ``content/base`` (a no_op placeholder) is deliberately kept, matching the
    deb/rpm install. Extraction into /opt/amdtools needs sudo, so the runner must
    have passwordless sudo (already required to run ANC).

    Records failures via fail_test; the caller (install_anc) owns the single
    update_test_result reporting call.
    '''
    anc_cfg = config_dict["anc"]
    cvs_home = anc_cfg["cvs_home"]
    anc_release_url = anc_cfg["anc_release_url"]
    tarball = anc_release_url.rsplit("/", 1)[-1]

    log.info("ANC: install tar release on remote nodes (cvs_home=%s, install prefix=%s)", cvs_home, ANC_TOOLS_PREFIX)

    install_cmd = (
        f"set -e && "
        f"mkdir -p '{cvs_home}' && cd '{cvs_home}' && "
        f"echo 'Downloading ANC release...' && "
        f"{_download_with_progress_snippet(anc_release_url, tarball)} && "
        f"echo 'Extracting outer archive...' && "
        f"tar -xzf '{tarball}' && "
        f"rm -f '{tarball}' && "
        # Fresh install: drop any prior ANC tree, then extract both payloads
        # into /opt/amdtools so the layout matches the deb/rpm packages.
        f"echo 'Removing existing {ANC_DIR}...' && "
        f"sudo rm -rf '{ANC_DIR}' && "
        f"sudo mkdir -p '{ANC_TOOLS_PREFIX}' && "
        f"echo 'Extracting anc-tool archive to {ANC_TOOLS_PREFIX}...' && "
        f"sudo tar -xzf anc-tool*.tar.gz -C '{ANC_TOOLS_PREFIX}' && "
        f"rm -f anc-tool*.tar.gz && "
        f"echo 'Extracting anc-content archive to {ANC_TOOLS_PREFIX}...' && "
        f"sudo tar -xzf anc-content*.tar.gz -C '{ANC_TOOLS_PREFIX}' && "
        f"rm -f anc-content*.tar.gz && "
        f"echo '--- Directory listing (2 levels) ---' && "
        f"find '{ANC_TOOLS_PREFIX}' -maxdepth 2 -type d | sort && "
        f"echo '--- Verification ---' && "
        f"test -f '{ANC_BIN}' && echo 'ANC_INSTALL_SUCCESS'"
    )

    out_dict = phdl.exec(install_cmd, timeout=_install_timeout(anc_cfg))
    print_test_output(log, out_dict)

    for host, output in out_dict.items():
        if "ANC_INSTALL_SUCCESS" not in output:
            fail_test(
                f"ANC installation failed on {host}: '{ANC_BIN}' not found after extracting into {ANC_TOOLS_PREFIX}"
            )
        else:
            log.info("Node %s: ANC directory installed successfully", host)

    _fail_unreachable_nodes(cluster_dict, out_dict, "tar install")

    # Skip exe_path validation if the extraction/reachability already failed
    # (nothing to validate); the caller reports the recorded failure.
    if globals.error_list:
        return

    # Validate exe_path entries with the bundled script.
    content_dir = f"{ANC_DIR}/content"
    local_script = os.path.join(RESOURCES_DIR, "validate_exe_paths.py")
    remote_script = "/tmp/validate_exe_paths.py"

    log.info("Uploading validation script to remote nodes...")
    phdl.upload_file(local_script, remote_script)

    log.info("Validating exe_path entries from %s", content_dir)
    validate_dict = phdl.exec(f"python3 {remote_script} '{content_dir}'", timeout=60)
    print_test_output(log, validate_dict)

    log.info("Removing validation script from remote nodes...")
    phdl.exec(f"rm -f {remote_script}", timeout=10)

    for host, output in validate_dict.items():
        if "VALIDATION_FAILED" in output:
            fail_test(f"exe_path validation failed on {host}: one or more tool paths are missing")
        elif "VALIDATION_SUCCESS" in output:
            log.info("Node %s: All exe_path entries validated", host)
        elif "ERROR:" in output:
            fail_test(f"Validation error on {host}: {output.strip()}")
        else:
            fail_test(f"Unexpected validation output on {host}: {output.strip()}")

    _fail_unreachable_nodes(cluster_dict, validate_dict, "exe_path validation")


# =========================================================================
# ROCm shared-library resolution (ldconfig)
# =========================================================================
def ensure_rocm_ldconfig(phdl, cluster_dict):
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

    Node coverage: any expected node (from ``cluster_dict["node_dict"]``) that
    is unreachable (absent from the exec output) is failed rather than silently
    passing.

    Reports pass/fail via fail_test / update_test_result (call from a test).
    '''
    globals.error_list = []
    log.info("ANC Pre-Task: ensure ROCm libs resolvable (%s)", LDCONFIG_TARGET_LIB)

    # Build the conf file from whichever ROCM_LIB_DIRS exist. Truncate ONCE up
    # front, then append every existing dir; this keeps the decision independent
    # of which dirs happen to exist (a plain index-based first-write-truncates
    # scheme skips the truncate when the first dir is absent, so lines would
    # accumulate on repeated runs). At least one dir exists when this runs
    # ("found" is set below), so an empty conf file is never left behind.
    conf_file = "/etc/ld.so.conf.d/rocm.conf"
    conf_lines = " && ".join(
        [f"echo -n '' | sudo tee {conf_file} >/dev/null"]
        + [f"if [ -d '{d}' ]; then echo '{d}' | sudo tee -a {conf_file} >/dev/null; fi" for d in ROCM_LIB_DIRS]
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
            log.info("Node %s: registered ROCm lib dirs and ran ldconfig; %s now resolvable", host, LDCONFIG_TARGET_LIB)
        elif "LDCONFIG_LIB_MISSING" in text:
            fail_test(
                f"ROCm library {LDCONFIG_TARGET_LIB} not found under "
                f"{ROCM_LIB_DIRS} on {host}: library is genuinely missing "
                f"(not an ldconfig cache issue)"
            )
        elif "LDCONFIG_STILL_MISSING" in text:
            fail_test(f"ROCm library {LDCONFIG_TARGET_LIB} still unresolved on {host} after ldconfig")
        else:
            fail_test(f"Unexpected ldconfig check output on {host}: {text.strip()}")

    _fail_unreachable_nodes(cluster_dict, out_dict, "ldconfig")

    update_test_result()


# Session-scoped guard so a whole-suite run (many group functions) installs and
# fixes ldconfig ONCE, while a single-group run still gets ANC installed. The
# flag lives for the pytest process lifetime; it is only set after install +
# ldconfig succeed (both raise via update_test_result on failure), so a failed
# attempt is retried by the next group rather than silently skipped.
_ANC_READY = False


def _anc_installed(phdl, cluster_dict, config_dict):
    '''
    Report whether ANC is already installed on every expected node.

    When ``anc.anc_version`` is set, "installed" means every node reports that
    version (reuses ``node_version_matches``). Otherwise "installed" means
    ``ANC_BIN`` is present on every node.
    '''
    expected = _expected_nodes(cluster_dict)
    if not expected:
        return False

    anc_version = config_dict.get("anc", {}).get("anc_version")
    if anc_version:
        matches = node_version_matches(phdl, anc_version)
        return all(matches.get(host) for host in expected)

    out_dict = phdl.exec(f"test -f '{ANC_BIN}' && echo ANC_PRESENT || true", timeout=60)
    return all("ANC_PRESENT" in (out_dict.get(host) or "") for host in expected)


def ensure_anc_ready(phdl, cluster_dict, config_dict):
    '''
    Ensure ANC is installed and ROCm libs are resolvable before a group runs.

    Idempotent and session-cached: the first call installs ANC (unless already
    present, per ``_anc_installed``) and runs ``ensure_rocm_ldconfig``; later
    calls in the same pytest session return immediately. Each ANC group test
    calls this first so a single-group run self-installs while a full-suite run
    pays the setup cost once.

    Install / ldconfig failures propagate as pytest failures (via their own
    ``update_test_result``), which aborts the calling group test.
    '''
    global _ANC_READY
    if _ANC_READY:
        log.info("ANC already ensured this session; skipping install/ldconfig")
        return

    if _anc_installed(phdl, cluster_dict, config_dict):
        log.info("ANC already installed on all nodes; skipping install")
    else:
        log.info("ANC not present on all nodes; installing")
        install_anc(phdl, cluster_dict, config_dict)

    ensure_rocm_ldconfig(phdl, cluster_dict)
    _ANC_READY = True


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


def _chown_tree_to_runner(host, root_path):
    '''Recursively chown an extracted log tree to the current process's UID/GID.

    Best-effort: if the runner lacks privileges to chown (e.g. some entry is
    root-owned and the process is unprivileged), log a warning rather than
    failing the whole log-collection step.
    '''
    uid = os.getuid()
    gid = os.getgid()
    try:
        os.chown(root_path, uid, gid)
        for dirpath, dirnames, filenames in os.walk(root_path):
            for name in dirnames + filenames:
                try:
                    os.chown(os.path.join(dirpath, name), uid, gid)
                except OSError as exc:
                    log.warning("Node %s: could not chown %s: %s", host, os.path.join(dirpath, name), exc)
    except OSError as exc:
        log.warning("Node %s: could not chown log tree %s: %s", host, root_path, exc)


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
    log_dir = log_dir.rstrip("/")
    base = os.path.basename(log_dir)
    parent = os.path.dirname(log_dir)
    remote_tar = f"/tmp/anc_{base}.tar.gz"

    # ANC ran under sudo, so log_dir (and its parents, e.g. /root/logs) are
    # root-owned and unreadable by the plain SSH user. Build the archive under
    # sudo, then chown it back so SFTP can pull it. `&&` so a tar failure aborts
    # before the (silent-on-nonzero) exec proceeds to download a missing file.
    archive_cmd = f"sudo tar -czf '{remote_tar}' -C '{parent}' '{base}' && sudo chown {user} '{remote_tar}'"
    try:
        single.exec(archive_cmd, timeout=600)
    except Exception as exc:
        return None, f"could not archive log dir {log_dir}: {exc}"

    local_tar = os.path.join(dest_dir, f"{base}.tar.gz")
    # download_file suffixes the local path with '_<host>' (per-host collision
    # avoidance) and returns {host: actual_path}; use that, not local_tar.
    try:
        downloaded = single.download_file(remote_tar, local_tar)
    except Exception as exc:
        return None, f"could not download log archive from {log_dir}: {exc}"
    finally:
        try:
            single.exec(f"rm -f '{remote_tar}'", timeout=30)
        except Exception as exc:  # best-effort cleanup
            log.warning("Node %s: could not remove remote tar %s: %s", host, remote_tar, exc)

    local_tar = None
    if isinstance(downloaded, dict) and downloaded:
        # single-node handle: prefer the host key, else the only entry.
        local_tar = downloaded.get(host) or next(iter(downloaded.values()))
    if not local_tar or not os.path.isfile(local_tar):
        return None, (
            f"could not archive log dir {log_dir}: remote archive was not created (check sudo/tar permissions on node)"
        )

    try:
        with tarfile.open(local_tar) as tf:
            # filter="data" sanitizes members (no absolute paths / "../" escapes)
            # and silences the 3.12+ extractall deprecation. The param was added
            # in 3.12 / backported to 3.9.17+, 3.10.12+, 3.11.4+; fall back for
            # older interpreters (repo targets python>=3.9).
            try:
                tf.extractall(dest_dir, filter="data")
            except TypeError:
                tf.extractall(dest_dir)
    except Exception as exc:
        return None, f"could not extract log archive for {log_dir}: {exc}"
    finally:
        try:
            os.remove(local_tar)
        except OSError:
            pass

    extracted_dir = os.path.join(dest_dir, base)
    # ANC archived its logs as root, so entries may carry root ownership. Give
    # the whole extracted tree back to the user running the test so the logs are
    # readable/removable without sudo on this box.
    _chown_tree_to_runner(host, extracted_dir)
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


def _evaluate_node(cluster_dict, config_dict, host, output, test_name, timestamp):
    '''
    Collect the ANC log directory for one node and decide whether it passed.

    ``output`` is the captured ANC console text (full when
    print_all_to_console, else just the "Log directory:" line) used only to
    locate the log-directory path. The per-node destination is resolved from
    config anc.log_folder_path with the "<node>" token set to this node's
    "<ip>_<hostname>" label (default layout
    ``.../anc_logs/<node>/<test_name>/<timestamp>/``); the whole log directory is
    copied there and the verdict is taken from console.log's final "Program
    exiting with return code ANC_SUCCESS [0]" line. On failure, the item summary
    and FAILED rows are surfaced.

    Returns:
      tuple[str | None, str | None, str | None]: (failure reason or None if the
      node passed, the per-node destination directory or None if it was never
      created, the "<ip>_<hostname>" node label or None if SSH never opened).
    '''
    try:
        single = Pssh(
            log,
            [host],
            user=cluster_dict["username"],
            pkey=cluster_dict["priv_key_file"],
        )
    except Exception as exc:  # infra failure must fail the test
        return f"could not open SSH handle for artifact collection: {exc}", None, None

    label = _node_label(single, host, cluster_dict)

    # ANC prints "FATAL: Group '<x>' not found" (and exits ANC_PROG_NOT_FOUND)
    # when the requested group is not installed on this node. There are no useful
    # logs to collect in that case, so report it plainly and skip collection.
    if ANC_GROUP_NOT_FOUND_RE.search(output or ""):
        log.error("Node %s: ANC group '%s' not found on remote system", host, test_name)
        return f"This test is not available on the remote system [{label}]", None, label

    ld_match = LOG_DIRECTORY_RE.search(output or "")
    if not ld_match:
        log.error("Node %s: could not determine Log directory from output", host)
        return "could not determine Log directory from ANC output", None, label
    log_dir = ld_match.group(1)
    log.info("Node %s: ANC %s log directory: %s", host, test_name, log_dir)

    dest_dir = resolve_anc_log_folder(config_dict, test_name, timestamp, node=label)
    os.makedirs(dest_dir, exist_ok=True)

    console_path, infra_reason = _pull_log_dir(single, host, cluster_dict["username"], log_dir, dest_dir)
    if infra_reason:
        return infra_reason, dest_dir, label

    try:
        with open(console_path, encoding="utf-8", errors="replace") as fh:
            console_text = fh.read()
    except Exception as exc:
        return f"could not read collected console.log: {exc}", dest_dir, label

    rc_matches = ANC_RETURN_CODE_RE.findall(console_text)
    if not rc_matches:
        log.error("Node %s: ANC return code not found in console.log", host)
        return "ANC return code not found in console.log", dest_dir, label

    rc_name, rc_value = rc_matches[-1][0], int(rc_matches[-1][1])
    log.info("Node %s: ANC %s program return code is %s [%s]", host, test_name, rc_name, rc_value)
    if rc_value != 0:
        # Fallback path: ANC still writes a Log directory on a missing group, so
        # if the FATAL line was not surfaced in the streamed output it is caught
        # here via the dedicated return code, with the same friendly message.
        if rc_name == ANC_PROG_NOT_FOUND_NAME or ANC_GROUP_NOT_FOUND_RE.search(console_text):
            log.error("Node %s: ANC group '%s' not found on remote system", host, test_name)
            return f"This test is not available on the remote system [{label}]", dest_dir, label
        detail = _summarize_failure(console_text)
        reason = f"ANC returned {rc_name} [{rc_value}]"
        if detail:
            reason = f"{reason}; {detail}"
        return reason, dest_dir, label

    return None, dest_dir, label


def _attach_anc_logs_to_html(request, config_dict, test_name, node_entries, timestamp, failed):
    '''
    Bundle this run's collected ANC log trees into the pytest-html report zip.

    ``node_entries`` is a list of ``(node_dir, node_label)`` pairs. Every node's
    collected directory is tarred into ONE archive under its own
    ``<node_label>/`` arcname (the "<ip>_<hostname>" label, which is unique per
    node), copied into the report log dir via add_html_to_report (so it lands in
    the zip), and linked from the test row via a stashed pytest-html url extra
    (merged by the repo-root pytest_runtest_makereport hook).

    The node label is used as the arcname (rather than a runner-base-relative
    path) because the on-disk log_folder_path may live anywhere (the shipped
    default is under {home}, not the runner base), so a relative path is not a
    reliable per-node discriminator and would collapse to a shared component.

    Gating (per config anc.ADD_ANC_LOGS_TO_HTML_REPORTS):
      - flag True  -> always attach.
      - flag False -> attach only when the test failed.

    Best-effort: never turn a reporting problem into a test failure.
    '''
    if request is None:
        return
    mgr = getattr(request.config, "_html_report_manager", None)
    if mgr is None or not getattr(mgr, "is_enabled", False):
        return

    always = _as_bool(config_dict.get("anc", {}).get(ADD_ANC_LOGS_TO_HTML_KEY), default=False)
    if not always and not failed:
        log.info(
            "ANC '%s': skipping HTML log attach (test passed and %s is False)", test_name, ADD_ANC_LOGS_TO_HTML_KEY
        )
        return

    present = [(d, label) for d, label in node_entries if d and os.path.isdir(d) and os.listdir(d)]
    if not present:
        log.warning("ANC '%s': no collected logs to attach to HTML report", test_name)
        return

    # Write the archive under the runner base, named by test + timestamp; each
    # node's tree goes in under its own unique "<ip>_<hostname>/" arcname so
    # multi-node runs never overwrite each other inside the tarball.
    runner_base = resolve_runner_results_base(config_dict.get("run_config", {}))
    os.makedirs(runner_base, exist_ok=True)
    tar_path = os.path.join(runner_base, f"{test_name}_{timestamp}_anc_logs.tar.gz")
    try:
        with tarfile.open(tar_path, "w:gz") as tf:
            seen = {}
            for node_dir, label in present:
                node_dir = node_dir.rstrip(os.sep)
                # Unique per-node arcname from the node label; fall back to the
                # leaf dir name if a label is somehow missing, and disambiguate
                # any duplicate arcname so no node's tree is overwritten.
                arcname = label or os.path.basename(node_dir)
                if arcname in seen:
                    seen[arcname] += 1
                    arcname = f"{arcname}_{seen[arcname]}"
                else:
                    seen[arcname] = 0
                tf.add(node_dir, arcname=arcname)
    except Exception as exc:  # best-effort; do not fail the test
        log.warning("ANC '%s': could not archive logs for HTML report: %s", test_name, exc)
        return

    link_name = f"ANC logs: {test_name}" + (" (FAILED)" if failed else "")
    try:
        # Copies the archive into the report's log dir (so it lands in the zip)
        # and returns its path relative to the main report for linking.
        rel_path = mgr.add_html_to_report(tar_path, request=request)
        if rel_path:
            # pytest-html 4.x renders links from report.extras (not
            # user_properties); stash the extra so the repo-root
            # pytest_runtest_makereport hook can merge it in for this test.
            try:
                import pytest_html

                extra = pytest_html.extras.url(rel_path, name=link_name)
                pending = getattr(request.node, "_anc_html_extras", [])
                pending.append(extra)
                request.node._anc_html_extras = pending
            except Exception as exc:  # link is best-effort; file is already in zip
                log.warning("ANC '%s': archived logs added to zip but could not add report link: %s", test_name, exc)
        log.info("ANC '%s': attached logs to HTML report as '%s'", test_name, link_name)
    except Exception as exc:  # best-effort
        log.warning("ANC '%s': could not attach logs to HTML report: %s", test_name, exc)


def run_anc_groups(phdl, cluster_dict, config_dict, groups, test_name, request=None):
    '''
    Run one or more ANC groups in a single invocation on all nodes.

    Executes ``cd <ANC_DIR> && sudo ./anc.py -g <groups...>`` on every node,
    copies the ENTIRE ANC log directory to the configured log_folder_path
    (``{runner_log_folder}/anc_logs/<node>/<test_name>/<timestamp>`` by default,
    where ``<node>`` is that node's ``<ip>_<hostname>`` label), and PASSES only
    when every node's console.log ends with ANC_SUCCESS [0]. On failure the item
    summary and FAILED rows are surfaced. Failures across parallel nodes are
    aggregated into a SINGLE test failure.

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
    # Per-node destinations are resolved individually in _evaluate_node (each
    # substitutes its own "<node>" label). Here we only compute the pattern with
    # "<node>" left intact, purely to announce where logs will land.
    log_pattern = resolve_anc_log_folder(config_dict, test_name, timestamp)
    expected_nodes = list(cluster_dict["node_dict"].keys())
    groups_arg = " ".join(groups)

    # Announce the resolved (substituted) log directory up front so the user
    # knows exactly where this run's ANC logs will land.
    log.info("=" * 72)
    log.info("ANC '%s': logs for this run -> %s", test_name, log_pattern)
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
            # Echo the "Log directory:" line (used to locate artifacts) and any
            # "FATAL: Group ... not found" line so a missing group is detectable
            # even when full ANC output is suppressed.
            f"grep -iE 'Log directory:|FATAL: Group' '{remote_stdout}'; "
            f"rm -f '{remote_stdout}'"
        )

    log.info(
        "ANC '%s': running %d group(s) (print_all_to_console=%s, timeout=%ss, logs under %s)",
        test_name,
        len(groups),
        print_all,
        timeout,
        log_pattern,
    )
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
    node_entries = []
    for host in expected_nodes:
        if host not in out_dict:
            reason = "node produced no output (did not run / unreachable)"
            log.error("Node %s: ANC %s FAILED - %s", host, test_name, reason)
            failed_nodes[host] = reason
            continue
        reason, dest_dir, label = _evaluate_node(
            cluster_dict,
            config_dict,
            host,
            out_dict[host] or "",
            test_name,
            timestamp,
        )
        if dest_dir:
            # Fall back to the host key for the arcname label if SSH-based
            # labelling failed, so every collected dir stays distinct.
            node_entries.append((dest_dir, label or _sanitize_path_component(host)))
        if reason:
            log.error("Node %s: ANC %s FAILED - %s", host, test_name, reason)
            failed_nodes[host] = reason

    if failed_nodes:
        details = "; ".join(f"{h}: {r}" for h, r in failed_nodes.items())
        fail_test(f"ANC {test_name} failed on {len(failed_nodes)}/{len(expected_nodes)} node(s): {details}")

    # Attach collected logs to the HTML report before update_test_result() may
    # raise: always when the flag is set, otherwise only on failure.
    _attach_anc_logs_to_html(request, config_dict, test_name, node_entries, timestamp, bool(failed_nodes))

    update_test_result()
