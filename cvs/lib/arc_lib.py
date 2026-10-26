'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

ARC installation, controller execution, artifact collection, and verdicts.
'''

import json
import os
import posixpath
import re
from collections import namedtuple

from cvs.lib import amdtools_lib, anc_rundeck, arc_rundeck, globals
from cvs.lib.amdtools_lib import pull_remote_dir
from cvs.lib.run_config_paths import resolve_runner_results_base
from cvs.lib.utils_lib import fail_test, print_test_output, update_test_result

log = globals.log
ARC_TOOLS_PREFIX = amdtools_lib.AMDTOOLS_PREFIX
ARC_DIR = f"{ARC_TOOLS_PREFIX}/arc"
ARC_BIN = f"{ARC_DIR}/arc.py"
ARC_INSTALL_PATH_KEY = "ARC_INSTALL_PATH"
CONTROLLER_NODES_KEY = "controller_nodes"
RACK_CONFIG_KEY = "rack_config"
SPEC_MODE_KEY = "spec_mode"
MOCK_KEY = "mock"
CONTINUE_ON_FAILURE_KEY = "continue_on_failure"
ALLOW_DISRUPTIVE_KEY = "allow_disruptive"
INACTIVITY_TIMEOUT_KEY = "inactivity_timeout"
RUN_TIMEOUT_KEY = "run_timeout"
INSTALL_TIMEOUT_KEY = "install_timeout"
PRINT_ALL_TO_CONSOLE_KEY = "print_all_to_console"
ADD_ARC_LOGS_TO_HTML_KEY = "ADD_ARC_LOGS_TO_HTML_REPORTS"
SPEC_MODES = ("full", "partial")
DEFAULT_SPEC_MODE = "full"
DEFAULT_ARC_INACTIVITY_TIMEOUT = 900
DEFAULT_ARC_RUN_TIMEOUT = 14400
DEFAULT_ARC_INSTALL_TIMEOUT = 1800
# ARC enforces item limits itself and may stay silent for a whole item, so CVS reports liveness.
ARC_HEARTBEAT_INTERVAL = 60
ARC_POLL_INTERVAL = 5
ARC_KILL_GRACE = 30
ARC_TIMEOUT_RC = 124
MIN_ARC_INACTIVITY_TIMEOUT = 2 * ARC_HEARTBEAT_INTERVAL
ARC_LOG_SUBPATH = "arc_logs/<node>/<test_name>/<timestamp>"
REMOTE_LOG_ROOT = "$HOME/arc_logs"
ARC_HTML_EXTRAS_ATTR = "_arc_html_extras"
SUMMARY_JSON = "summary.json"
ERRORS_JSON = "errors.json"
PASS_STATUS_CODES = ("SUCCESS", "WARNING")
EXIT_SENTINEL = "CVS_ARC_EXIT_CODE"
LOG_DIR_SENTINEL = "CVS_ARC_LOG_DIR"
TIMED_OUT_SENTINEL = "CVS_ARC_TIMED_OUT"
EXIT_CODE_RE = re.compile(rf"^{EXIT_SENTINEL}=(-?\d+)\s*$", re.M)
LOG_DIR_RE = re.compile(rf"^{LOG_DIR_SENTINEL}=(\S+)\s*$", re.M)
TIMED_OUT_RE = re.compile(rf"^{TIMED_OUT_SENTINEL}=(\d+)\s*$", re.M)
ARC_NOT_FOUND_RE = re.compile(r"Test flow or test item '([^']*)' not found")
ARC_VERSION_RE = re.compile(
    r"^\s*(?i:arc(?:\.py)?)\s+(" + amdtools_lib.VERSION_BODY + ")" + amdtools_lib.VERSION_END, re.M
)
SELECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
ArcPaths = namedtuple("ArcPaths", ["prefix", "arc_dir", "arc_bin"])
ControllerResult = namedtuple(
    "ControllerResult", ["reason", "dest_dir", "label", "summary_json", "errors_json", "summary"]
)

# These names follow ARC development at 1.8.1 across heliosr, heliosp, and heliosm.
# Content absent from an installed platform reports n/a.
EXAMPLE_FLOWS = ["example_pass", "example_failure"]
# These flows clear AMC log stores as PRE/POST conditions.
DIAG_FLOWS = ["diags_lvl1", "diags_lvl2", "wld2_verify"]
DISRUPTIVE_FLOWS = [
    "heliosm_mfg_l11",
    "heliosp_mfg_l11",
    "heliosr_mfg_l11",
    "reboot_all",
    "scale_up_config",
    "scale_up_verify",
]
DIAG_ITEMS = [
    "ComputeLeakTest",
    "GetRackState",
    "PingTest",
    "PortFlapTest",
    "PowerShelfDiagnostic",
    "PowerShelfMeshHeliosM",
    "RackCablingCheck",
    "RackCablingCheckHeliosM",
    "RmcDiagnostic",
    "RmcLeakTest",
    "RSCMDiagnostic",
    "SwitchTrayDiagnostic",
    "TorSwitchDiagnostic",
    "Wld2ZoneCheck",
]
TELEMETRY_ITEMS = [
    "AmcManagerTelemetry",
    "AmcSystemTelemetry",
    "BmcManagerTelemetry",
    "ComputeFwInventory",
    "FecDataCollection",
]
ANC_ITEMS = ["ANCCPUTest", "ANCGPUTest"]
WORKLOAD_ITEMS = [
    "FrontEndPerformanceTest",
    "ScaleUpConnectivityTest_ComputeOnly",
    "ScaleUpTransferBench_CD",
    "ScaleUpTransferBench_ID",
    "ScaleUpTransferBench_ID_ComputeOnly",
    "ScaleUpTransferBench_Targeted",
]
# These reboot or power-cycle hardware, toggle ports, reconfigure fabric, run
# PRBS/SBR or switch loopback, or alter host namespaces/routes.
DISRUPTIVE_ITEMS = [
    "ComputeDCCycle",
    "PortToggle",
    "PRBSCheck",
    "PRBSCheckEAMFirst",
    "RebootComputeTest",
    "RebootRmcTest",
    "RebootSwitchTest",
    "SBRTest",
    "ScaleOutConnectivityTestID",
    "ScaleOutConnectivityTestL2",
    "ScaleOutConnectivityTestL3",
    "ScaleUpConnectivityTest",
    "ScaleupEAMConfig",
    "ScaleupSwitchConfig",
    "SwitchInternalLoopback",
]
ARC_SUITES = (
    {"key": "examples", "names": EXAMPLE_FLOWS, "unit": "flow", "title": "example flow", "disruptive": False},
    {"key": "flows", "names": DIAG_FLOWS, "unit": "flow", "title": "diagnostic flow", "disruptive": False},
    {
        "key": "disruptive_flows",
        "names": DISRUPTIVE_FLOWS,
        "unit": "flow",
        "title": "DISRUPTIVE flow",
        "disruptive": True,
    },
    {"key": "diag", "names": DIAG_ITEMS, "unit": "item", "title": "diagnostic item", "disruptive": False},
    {"key": "telemetry", "names": TELEMETRY_ITEMS, "unit": "item", "title": "telemetry item", "disruptive": False},
    {"key": "anc", "names": ANC_ITEMS, "unit": "item", "title": "ANC item", "disruptive": False},
    {"key": "workload", "names": WORKLOAD_ITEMS, "unit": "item", "title": "workload item", "disruptive": False},
    {"key": "disruptive", "names": DISRUPTIVE_ITEMS, "unit": "item", "title": "DISRUPTIVE item", "disruptive": True},
)
_ARC_READY = False
_ARC_INSTALL_PATHS = None


def resolve_controllers(cluster_dict, config_dict):
    '''Return configured controller host keys, defaulting to the first node.'''
    nodes = list((cluster_dict.get("node_dict") or {}).keys())
    raw = (config_dict.get("arc") or {}).get(CONTROLLER_NODES_KEY) or []
    if isinstance(raw, str):
        raw = [raw]
    return list(raw) if raw else nodes[:1]


def resolve_arc_install_prefix(config_dict, pkg_type):
    '''Resolve a tar relocation or the fixed package prefix, rejecting root.'''
    prefix = ARC_TOOLS_PREFIX
    if pkg_type == "tar":
        raw = (config_dict.get("arc") or {}).get(ARC_INSTALL_PATH_KEY)
        if raw and str(raw).strip():
            amdtools_lib.assert_shell_safe(config_dict.get("arc", {}), (ARC_INSTALL_PATH_KEY,), "ARC")
            prefix = os.path.expanduser(str(raw).strip())
    resolved = os.path.abspath(prefix)
    if not resolved.strip("/"):
        raise ValueError(
            f"config arc.{ARC_INSTALL_PATH_KEY} resolves to filesystem root ({resolved!r}), which is not a valid ARC install prefix"
        )
    return resolved


def resolve_arc_paths(config_dict, pkg_type):
    '''Build paths for a package flavour.'''
    prefix = resolve_arc_install_prefix(config_dict, pkg_type)
    arc_dir = f"{prefix}/arc"
    return ArcPaths(prefix, arc_dir, f"{arc_dir}/arc.py")


def resolve_arc_paths_from_config(config_dict):
    '''Resolve paths using the configured release URL.'''
    url = (config_dict.get("arc") or {}).get("arc_release_url") or ""
    try:
        pkg_type = amdtools_lib.detect_package_flavour(url, "ARC").pkg_type if url.strip() else ""
    except ValueError:
        pkg_type = ""
    return resolve_arc_paths(config_dict, pkg_type)


def check_version_matches_url(config_dict):
    '''Return a problem when the requested release exceeds the URL release.'''
    cfg = config_dict.get("arc") or {}
    version = cfg.get("arc_version")
    if version is None or (isinstance(version, str) and not version.strip()):
        return None
    if not amdtools_lib.is_strict_version(version):
        return f"config arc.arc_version ({version}) is not a valid ARC version"
    url_version = amdtools_lib.parse_version_from_url(cfg.get("arc_release_url"))
    if url_version and amdtools_lib.compare_versions(str(version).strip(), url_version) > 0:
        return f"config arc.arc_version ({version}) is newer than arc.arc_release_url ({url_version})"
    return None


def _config_seconds(cfg, key, default):
    '''Read a whole, non-negative number of seconds from the arc config.'''
    value = cfg.get(key, default)
    text = str(value).strip()
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"arc.{key} must be a whole number of seconds (got {value!r})")
    return int(text)


def validate_arc_config(config_dict, cluster_dict):
    '''Collect configuration problems before contacting controllers.'''
    cfg = config_dict.get("arc") or {}
    problems = []
    url = cfg.get("arc_release_url")
    if not url or not str(url).strip():
        problems.append("arc.arc_release_url must be set")
    else:
        try:
            amdtools_lib.assert_shell_safe(cfg, ("arc_release_url",), "ARC")
            flavour = amdtools_lib.detect_package_flavour(url, "ARC")
            if not flavour.is_direct:
                problems.append("ARC releases are direct .deb/.rpm/.tar.gz artifacts")
        except ValueError as exc:
            problems.append(str(exc))
    mismatch = check_version_matches_url(config_dict)
    if mismatch:
        problems.append(mismatch)
    try:
        resolve_arc_install_prefix(config_dict, "tar")
    except ValueError as exc:
        problems.append(str(exc))
    nodes = cluster_dict.get("node_dict") or {}
    if not nodes:
        problems.append("cluster node_dict must contain a controller")
    for host in resolve_controllers(cluster_dict, config_dict):
        if host not in nodes:
            problems.append(f"arc.controller_nodes host {host!r} is not in node_dict")
    rack = cfg.get(RACK_CONFIG_KEY)
    if not rack or not str(rack).strip():
        problems.append("arc.rack_config must be set")
    else:
        try:
            amdtools_lib.assert_shell_safe(cfg, (RACK_CONFIG_KEY,), "ARC")
        except ValueError as exc:
            problems.append(str(exc))
    if cfg.get(SPEC_MODE_KEY, DEFAULT_SPEC_MODE) not in SPEC_MODES:
        problems.append(f"arc.spec_mode must be one of {SPEC_MODES}")
    username_problem = amdtools_lib.validate_cluster_username(cluster_dict, "ARC")
    if username_problem:
        problems.append(username_problem)
    try:
        _config_seconds(cfg, RUN_TIMEOUT_KEY, DEFAULT_ARC_RUN_TIMEOUT)
    except ValueError as exc:
        problems.append(str(exc))
    try:
        inactivity_timeout = _config_seconds(cfg, INACTIVITY_TIMEOUT_KEY, DEFAULT_ARC_INACTIVITY_TIMEOUT)
        if inactivity_timeout < MIN_ARC_INACTIVITY_TIMEOUT:
            problems.append(
                f"arc.inactivity_timeout must be at least {MIN_ARC_INACTIVITY_TIMEOUT} s "
                f"(CVS prints an ARC heartbeat every {ARC_HEARTBEAT_INTERVAL} s)"
            )
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def disruptive_skip_reason(config_dict, name):
    '''Return a skip reason unless state-changing ARC content is opted in.'''
    if amdtools_lib.as_bool((config_dict.get("arc") or {}).get(ALLOW_DISRUPTIVE_KEY), default=False):
        return None
    return f"ARC '{name}' changes hardware state (reboot / power-cycle / port or fabric reconfiguration / PRBS-SBR); set arc.allow_disruptive to \"True\" to run it"


def parse_arc_version(output):
    '''Parse ARC's labelled --version line.'''
    match = ARC_VERSION_RE.search(output or "")
    return match.group(1) if match else None


def controller_installed_versions(orch, controllers, arc_bin=ARC_BIN):
    '''Read installed ARC versions on controllers only.'''
    out = orch.exec(f"python3 '{arc_bin}' --version 2>&1 || true", hosts=controllers, timeout=60)
    return {host: parse_arc_version(value) for host, value in out.items()}


def controller_version_matches(orch, controllers, expected_version, arc_bin=ARC_BIN):
    '''Return satisfying-version flags for responding controllers.'''
    versions = controller_installed_versions(orch, controllers, arc_bin)
    return {host: amdtools_lib.version_satisfies(version, expected_version) for host, version in versions.items()}


def _install_arc_package(orch, controllers, config_dict, pkg_type, paths):
    '''Install a direct ARC artifact on the selected controllers.'''
    cfg = config_dict["arc"]
    url = cfg["arc_release_url"]
    archive = f"arc.{pkg_type if pkg_type != 'tar' else 'tar.gz'}"
    parts = ["set -e; "]
    if pkg_type == "tar":
        parts.append(amdtools_lib.sudo_prefix_snippet(paths.prefix, "ARC") + "; ")
    parts.extend(
        [
            'STAGE=$(mktemp -d); ',
            'trap \'rm -rf "$STAGE"\' EXIT; ',
            'cd "$STAGE"; ',
            amdtools_lib.download_with_progress_snippet(url, archive, "ARC") + "; ",
        ]
    )
    if pkg_type == "tar":
        parts.append(amdtools_lib.replace_tops_and_extract_snippet(archive, paths.prefix))
        parts.append(f"test -f '{paths.arc_bin}' && python3 '{paths.arc_bin}' --version && echo 'ARC_INSTALL_SUCCESS'")
    elif pkg_type == "deb":
        parts.extend(
            [
                "sudo dpkg -i --force-depends ./arc.deb; ",
                "pkg=$(dpkg-deb -f ./arc.deb Package); ",
                'if ! dpkg-query -W -f=\'${Status}\' "$pkg" 2>/dev/null | grep -q \'install ok installed\'; then ',
                'sudo dpkg --configure --force-depends "$pkg"; fi; ',
                f"python3 '{paths.arc_bin}' --version && echo 'ARC_INSTALL_SUCCESS'",
            ]
        )
    else:
        parts.append("sudo dnf install -y ./arc.rpm; ")
        parts.append(f"python3 '{paths.arc_bin}' --version && echo 'ARC_INSTALL_SUCCESS'")
    out = orch.exec(
        "".join(parts), hosts=controllers, timeout=cfg.get(INSTALL_TIMEOUT_KEY, DEFAULT_ARC_INSTALL_TIMEOUT)
    )
    print_test_output(log, out)
    for host, output in out.items():
        if "ARC_INSTALL_SUCCESS" not in (output or ""):
            fail_test(f"ARC {pkg_type} installation failed on {host}: {paths.arc_bin} was not verified")
    amdtools_lib.fail_unreachable_nodes(controllers, out, f"{pkg_type} install", "ARC")


def _install_arc_deb(orch, controllers, config_dict, paths):
    '''Install a direct deb artifact.'''
    _install_arc_package(orch, controllers, config_dict, "deb", paths)


def _install_arc_rpm(orch, controllers, config_dict, paths):
    '''Install a direct rpm artifact.'''
    _install_arc_package(orch, controllers, config_dict, "rpm", paths)


def _install_arc_tar(orch, controllers, config_dict, paths):
    '''Install a direct tar artifact.'''
    _install_arc_package(orch, controllers, config_dict, "tar", paths)


def install_arc(orch, cluster_dict, config_dict):
    '''Install ARC, with a controller-only version precheck and postcheck.'''
    globals.error_list = []
    cfg = config_dict["arc"]
    amdtools_lib.assert_shell_safe(cfg, ("arc_release_url", ARC_INSTALL_PATH_KEY), "ARC")
    controllers = resolve_controllers(cluster_dict, config_dict)
    try:
        flavour = amdtools_lib.detect_package_flavour(cfg["arc_release_url"], "ARC")
    except ValueError as exc:
        fail_test(str(exc))
        update_test_result()
        return
    if not flavour.is_direct:
        fail_test("ARC releases are direct .deb/.rpm/.tar.gz artifacts")
        update_test_result()
        return
    paths = resolve_arc_paths(config_dict, flavour.pkg_type)
    version = cfg.get("arc_version")
    if version:
        matches = controller_version_matches(orch, controllers, version, paths.arc_bin)
        if controllers and all(matches.get(host) for host in controllers):
            log.info("All controllers already satisfy ARC %s (installed >= requested); skipping install", version)
            update_test_result()
            return
    {"deb": _install_arc_deb, "rpm": _install_arc_rpm, "tar": _install_arc_tar}[flavour.pkg_type](
        orch, controllers, config_dict, paths
    )
    if version and not globals.error_list:
        matches = controller_version_matches(orch, controllers, version, paths.arc_bin)
        for host in controllers:
            if host not in matches:
                fail_test(f"ARC version verification failed on {host}: node produced no output")
            elif not matches[host]:
                fail_test(f"ARC version verification failed on {host}: expected {version} at '{paths.arc_bin}'")
            else:
                log.info("Node %s: ARC version %s confirmed", host, version)
    update_test_result()


def resolve_arc_install_location(orch, cluster_dict, config_dict):
    '''Verify and cache the configured ARC entrypoint on controllers.'''
    global _ARC_INSTALL_PATHS
    paths = resolve_arc_paths_from_config(config_dict)
    controllers = resolve_controllers(cluster_dict, config_dict)
    out = orch.exec(f"test -f '{paths.arc_bin}' && echo ARC_PRESENT || true", hosts=controllers, timeout=60)
    missing = [host for host in controllers if "ARC_PRESENT" not in (out.get(host) or "")]
    for host in missing:
        log.error("Node %s: ARC installation not found in the path %s", host, paths.arc_bin)
        fail_test(f"ARC installation not found in the path {paths.arc_bin} on {host}")
    if missing:
        _ARC_INSTALL_PATHS = None
        return None
    for host in controllers:
        log.info("Node %s: Using ARC installation at %s", host, paths.arc_bin)
    _ARC_INSTALL_PATHS = paths
    return paths


def check_rack_config_present(orch, cluster_dict, config_dict):
    '''Check each controller's rack JSON at the configured common path.'''
    controllers = resolve_controllers(cluster_dict, config_dict)
    rack = config_dict["arc"][RACK_CONFIG_KEY]
    out = orch.exec(f"test -f '{rack}' && echo ARC_RACK_PRESENT || true", hosts=controllers, timeout=60)
    missing = [host for host in controllers if "ARC_RACK_PRESENT" not in (out.get(host) or "")]
    for host in missing:
        fail_test(f"ARC rack config '{rack}' not found on {host}")
    return not missing


def _arc_installed(orch, cluster_dict, config_dict):
    '''Check ARC presence or version on every controller.'''
    controllers = resolve_controllers(cluster_dict, config_dict)
    if not controllers:
        return False
    paths = resolve_arc_paths_from_config(config_dict)
    version = config_dict["arc"].get("arc_version")
    if version:
        matches = controller_version_matches(orch, controllers, version, paths.arc_bin)
        return all(matches.get(host) for host in controllers)
    out = orch.exec(f"test -f '{paths.arc_bin}' && echo ARC_PRESENT || true", hosts=controllers, timeout=60)
    return all("ARC_PRESENT" in (out.get(host) or "") for host in controllers)


def ensure_arc_ready(orch, cluster_dict, config_dict):
    '''Install ARC once per session, then verify the binary and rack JSON.'''
    global _ARC_READY
    if _ARC_READY:
        return
    if not _arc_installed(orch, cluster_dict, config_dict):
        install_arc(orch, cluster_dict, config_dict)
    globals.error_list = []
    paths = resolve_arc_install_location(orch, cluster_dict, config_dict)
    rack_present = check_rack_config_present(orch, cluster_dict, config_dict)
    if paths is None or not rack_present:
        update_test_result()
        return
    _ARC_READY = True


def validate_rack_config(orch, cluster_dict, config_dict):
    '''Run ARC's rack validation without run-only flags.'''
    globals.error_list = []
    ensure_arc_ready(orch, cluster_dict, config_dict)
    cfg = config_dict["arc"]
    controllers = resolve_controllers(cluster_dict, config_dict)
    arc_bin = (_ARC_INSTALL_PATHS or resolve_arc_paths_from_config(config_dict)).arc_bin
    timestamp = amdtools_lib.new_run_timestamp()
    cmd = (
        f"python3 '{arc_bin}' --rack '{cfg[RACK_CONFIG_KEY]}' --validate "
        f"--spec-mode {cfg.get(SPEC_MODE_KEY, DEFAULT_SPEC_MODE)} "
        f"-o \"{REMOTE_LOG_ROOT}/validate/{timestamp}\"; rc=$?; echo; echo \"{EXIT_SENTINEL}=$rc\""
    )
    out = orch.exec(
        cmd, hosts=controllers, inactivity_timeout=cfg.get(INACTIVITY_TIMEOUT_KEY, DEFAULT_ARC_INACTIVITY_TIMEOUT)
    )
    print_test_output(log, out)
    missing = amdtools_lib.fail_unreachable_nodes(controllers, out, "rack validation", "ARC")
    for host in controllers:
        if host in missing:
            continue
        codes = EXIT_CODE_RE.findall(out.get(host) or "")
        if not codes or int(codes[-1]) != 0:
            fail_test(f"ARC rack validation failed on {host}: exit code {codes[-1] if codes else 'missing'}")
    update_test_result()


def build_arc_invocation(arc_bin, selection, rack_config, spec_mode, mock, continue_on_failure):
    '''Build a shell-safe ARC run command for validated content names.'''
    if any(not SELECTION_NAME_RE.fullmatch(name) for name in selection):
        raise ValueError("ARC selection names must contain only letters, digits, or underscores")
    prefix = "MOCK=1 " if mock else ""
    flags = " --continue-on-failure" if continue_on_failure else ""
    return (
        f"{prefix}python3 '{arc_bin}' {' '.join(selection)} --run --rack '{rack_config}' "
        f"--spec-mode {spec_mode} --no-progress{flags}"
    )


def build_arc_run_command(invocation, base, test_name, run_timeout, stdout_file=None):
    '''
    Wrap an ARC invocation in a controller-side liveness heartbeat and wall-clock watchdog.

    ARC enforces its own per-item limits and can stay silent for a whole item, so the
    shell reports liveness on a fixed cadence instead of on output growth, leaving the
    SSH inactivity timer for genuine channel stalls. ARC runs in its own session so a
    run_timeout expiry, or a heartbeat that can no longer reach CVS, stops its whole
    process group. A falsy run_timeout disables the watchdog. stdout_file captures ARC
    output on the controller instead of streaming it.
    '''
    redirect = f' > "{stdout_file}" 2>&1' if stdout_file else ""
    scratch = f' "{posixpath.dirname(stdout_file)}"' if stdout_file else ""
    size = f', $(wc -c < "{stdout_file}" 2>/dev/null || echo 0) bytes of output' if stdout_file else ""
    watchdog = timed_out = not_found = ""
    if run_timeout:
        watchdog = (
            f'if [ $((now - start)) -ge {run_timeout} ]; then timed_out=1; '
            f'echo "ARC {test_name} timed out after {run_timeout}s (arc.run_timeout); '
            'stopping ARC process group $arc_pid"; cvs_arc_stop; break; fi; '
        )
        timed_out = (
            f'if [ "$timed_out" = 1 ]; then echo "{TIMED_OUT_SENTINEL}={run_timeout}"; rc={ARC_TIMEOUT_RC}; fi; '
        )
    if stdout_file:
        not_found = f"grep -E \"Test flow or test item '.*' not found\" \"{stdout_file}\"; rm -f \"{stdout_file}\"; "
    trailer = (
        f'echo; logdir=$(ls -d "$base"/*/ 2>/dev/null | head -n 1); '
        f'echo "{LOG_DIR_SENTINEL}=${{logdir:-$base}}"; echo "{EXIT_SENTINEL}=$rc"'
    )
    return (
        f'base="{base}"; mkdir -p "$base"{scratch} || exit 1; '
        'cvs_arc_stop() { kill -s TERM -- "-$arc_pid" 2>/dev/null; g=0; '
        f'while [ "$g" -lt {ARC_KILL_GRACE} ] && kill -0 -- "-$arc_pid" 2>/dev/null; do sleep 1; g=$((g + 1)); done; '
        'kill -s KILL -- "-$arc_pid" 2>/dev/null; }; '
        f'setsid env PYTHONUNBUFFERED=1 {invocation} -o "$base"{redirect} & arc_pid=$!; '
        "trap '' PIPE; "
        f'echo "ARC {test_name} started: pid $arc_pid"; '
        f'start=$(date +%s); next=$((start + {ARC_HEARTBEAT_INTERVAL})); timed_out=0; '
        f'while kill -0 "$arc_pid" 2>/dev/null; do sleep {ARC_POLL_INTERVAL}; now=$(date +%s); '
        f"{watchdog}"
        f'if [ "$now" -ge "$next" ]; then '
        f'echo "ARC {test_name} running: $((now - start))s elapsed{size}" || {{ cvs_arc_stop; break; }}; '
        f'next=$((now + {ARC_HEARTBEAT_INTERVAL})); fi; done; '
        "trap - PIPE; "
        'wait "$arc_pid"; rc=$?; '
        f"{timed_out}{not_found}{trailer}"
    )


def run_arc_flows(orch, cluster_dict, config_dict, flows, test_name, request=None):
    '''Run selected ARC flows.'''
    run_arc_selection(orch, cluster_dict, config_dict, flows, test_name, unit="flow", request=request)


def run_arc_items(orch, cluster_dict, config_dict, items, test_name, request=None):
    '''Run selected ARC items.'''
    run_arc_selection(orch, cluster_dict, config_dict, items, test_name, unit="item", request=request)


def _find_arc_json(root, filename):
    '''Find an ARC JSON artifact, preferring the ARC session directory.'''
    matches = []
    if not root or not os.path.isdir(root):
        return None
    for parent, _dirs, files in os.walk(root):
        if filename in files:
            matches.append(os.path.join(parent, filename))
    matches.sort(key=lambda path: os.path.basename(os.path.dirname(path)) != "ARC")
    return matches[0] if matches else None


def _evaluate_controller(orch, cluster_dict, host, output, test_name, timestamp):
    '''Judge one controller using CVS sentinels and collected summary.json.'''
    label = amdtools_lib.node_label_from_file(cluster_dict, host)
    result = ControllerResult(None, None, label, None, None, None)
    if ARC_NOT_FOUND_RE.search(output):
        return result._replace(reason=f"This test is not available on the remote system [{label}]")
    timed_out = TIMED_OUT_RE.findall(output)
    timeout_reason = (
        f"ARC run timed out after {timed_out[-1]} s (arc.run_timeout); CVS stopped ARC's process group on the controller"
        if timed_out
        else None
    )
    codes = EXIT_CODE_RE.findall(output)
    if not codes:
        return result._replace(
            reason="ARC run did not complete (no exit code captured: SSH channel silent for arc.inactivity_timeout, or dropped)"
        )
    rc = int(codes[-1])
    dirs = LOG_DIR_RE.findall(output)
    if not dirs:
        return result._replace(reason=f"ARC exit code {rc} had no log directory sentinel")
    log_dir = dirs[-1]
    dest_dir = amdtools_lib.resolve_run_log_folder(ARC_LOG_SUBPATH, test_name, timestamp, node=label)
    os.makedirs(dest_dir, exist_ok=True)
    extracted_dir, pull_reason = pull_remote_dir(orch, host, cluster_dict["username"], log_dir, dest_dir, "ARC")
    result = result._replace(dest_dir=dest_dir)
    if pull_reason:
        return result._replace(reason=f"{timeout_reason}; {pull_reason}" if timeout_reason else pull_reason)
    summary_path = _find_arc_json(extracted_dir, SUMMARY_JSON)
    errors_path = _find_arc_json(extracted_dir, ERRORS_JSON)
    result = result._replace(summary_json=summary_path, errors_json=errors_path)
    if timeout_reason:
        return result._replace(reason=timeout_reason)
    if not summary_path:
        return result._replace(reason=f"ARC exited {rc} without writing {SUMMARY_JSON} under {log_dir}")
    try:
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
    except (OSError, ValueError) as exc:
        return result._replace(reason=f"could not read collected summary.json: {exc}")
    result = result._replace(summary=summary)
    overall = summary.get("overall_summary") or {}
    status = overall.get("status_code") or "UNKNOWN"
    if rc == 0 and status in PASS_STATUS_CODES:
        if status == "WARNING":
            log.warning("ARC %s on %s returned non-blocking WARNING", test_name, host)
        return result
    if rc != 0 and status in PASS_STATUS_CODES:
        return result._replace(reason=f"ARC exit code {rc} disagrees with summary.json status_code {status}")
    numeric = overall.get("status_code_numeric", overall.get("code", "?"))
    counts = [
        overall.get(key, summary.get(key, 0))
        for key in ("total_passed", "total_completed", "total_failed", "total_skipped", "total_not_run")
    ]
    reason = (
        f"ARC returned {status} [{numeric}] (exit code {rc}); {counts[0]} passed, {counts[1]} completed, "
        f"{counts[2]} failed, {counts[3]} skipped, {counts[4]} not run"
    )
    return result._replace(reason=reason)


def _record_arc_verdict(request, test_name, results):
    '''Add one pass/fail line to the HTML Reports section.'''
    if request is None:
        return
    mgr = getattr(request.config, "_html_report_manager", None)
    if mgr is None or not getattr(mgr, "is_enabled", False):
        return
    failed = [r for r in results if r.reason]
    detail = "; ".join(f"FAILED on {r.label} ({r.reason})" for r in failed)
    if not failed:
        detail = "PASS on all controllers [" + ", ".join(r.label for r in results) + "]"
    try:
        mgr.record_anc_verdict(test_name, not failed, detail)
    except Exception as exc:
        log.warning("ARC '%s': could not record verdict: %s", test_name, exc)


def _attach_arc_logs_to_html(request, config_dict, test_name, results, timestamp, failed):
    '''Link JSON artifacts and selected log tarballs to pytest HTML.'''
    if request is None:
        return {}, {}
    mgr = getattr(request.config, "_html_report_manager", None)
    if mgr is None or not getattr(mgr, "is_enabled", False):
        return {}, {}
    hrefs = {}
    for result in results:
        for path, suffix, link_name in (
            (result.summary_json, "summary", f"ARC summary: {result.label}"),
            (result.errors_json, "errors", f"errors_occured: {result.label}"),
        ):
            if not path:
                continue
            try:
                rel = mgr.add_html_to_report(
                    path,
                    request=request,
                    dest_name=f"{result.label}_{test_name}_{timestamp}_{suffix}.json",
                    track_in_reports=False,
                )
                amdtools_lib.stash_report_link(request, rel, link_name, ARC_HTML_EXTRAS_ATTR)
                if rel and (suffix == "errors" or result.label not in hrefs):
                    hrefs[result.label] = os.path.basename(rel)
            except Exception as exc:
                log.warning("ARC '%s': could not attach %s for %s: %s", test_name, suffix, result.label, exc)
    always = amdtools_lib.as_bool(config_dict.get("arc", {}).get(ADD_ARC_LOGS_TO_HTML_KEY), default=False)
    tar_hrefs = {}
    if always or failed:
        tar_hrefs = amdtools_lib.attach_node_log_tarballs(
            request,
            mgr,
            results,
            test_name,
            timestamp,
            resolve_runner_results_base(config_dict.get("run_config", {})),
            "ARC",
            ARC_HTML_EXTRAS_ATTR,
        )
    return hrefs, tar_hrefs


def _capture_rundeck_results(
    request, cluster_dict, config_dict, selection, test_name, timestamp, results, json_hrefs, tarball_hrefs
):
    '''Add per-controller records to the ARC Run Deck fixture.'''
    if request is None:
        return
    import pytest

    try:
        arc_res_dict = request.getfixturevalue("arc_res_dict")
    except pytest.FixtureLookupError:
        return
    try:
        suite_name = getattr(request.config, "_suite_name", "") or "arc"
        records = {
            r.label: arc_rundeck.build_node_record(
                status=amdtools_lib.deck_status(r.reason),
                summary=r.summary,
                errors_json_href=json_hrefs.get(r.label, ""),
                log_tarball_href=tarball_hrefs.get(r.label, ""),
            )
            for r in results
        }
        anc_rundeck.record_group(
            arc_res_dict,
            " ".join(selection),
            records,
            meta=arc_rundeck.make_meta(cluster_dict, config_dict, suite_name, timestamp),
        )
    except Exception as exc:
        log.warning("ARC '%s': could not capture Run Deck results: %s", test_name, exc)


def run_arc_selection(orch, cluster_dict, config_dict, selection, test_name, unit="flow", request=None):
    '''Run ARC on configured controllers and aggregate structured verdicts.'''
    globals.error_list = []
    cfg = config_dict["arc"]
    controllers = resolve_controllers(cluster_dict, config_dict)
    arc_bin = (_ARC_INSTALL_PATHS or resolve_arc_paths_from_config(config_dict)).arc_bin
    inactivity_timeout = _config_seconds(cfg, INACTIVITY_TIMEOUT_KEY, DEFAULT_ARC_INACTIVITY_TIMEOUT)
    run_timeout = _config_seconds(cfg, RUN_TIMEOUT_KEY, DEFAULT_ARC_RUN_TIMEOUT)
    print_all = amdtools_lib.as_bool(cfg.get(PRINT_ALL_TO_CONSOLE_KEY), default=True)
    timestamp = amdtools_lib.new_run_timestamp()
    pattern = amdtools_lib.resolve_run_log_folder(ARC_LOG_SUBPATH, test_name, timestamp)
    log.info("=" * 72)
    log.info("ARC '%s': logs for this run -> %s", test_name, pattern)
    log.info("=" * 72)
    invocation = build_arc_invocation(
        arc_bin,
        selection,
        cfg[RACK_CONFIG_KEY],
        cfg.get(SPEC_MODE_KEY, DEFAULT_SPEC_MODE),
        amdtools_lib.as_bool(cfg.get(MOCK_KEY), default=False),
        amdtools_lib.as_bool(cfg.get(CONTINUE_ON_FAILURE_KEY), default=False),
    )
    base = f"{REMOTE_LOG_ROOT}/{test_name}/{timestamp}"
    stdout_file = None
    if not print_all:
        stdout_file = f"{amdtools_lib.remote_user_tmp(cluster_dict.get('username', ''))}/arc_run_$$.out"
    cmd = build_arc_run_command(invocation, base, test_name, run_timeout, stdout_file)
    log.info(
        "ARC '%s': running %d %s(s) on %d controller(s) (run_timeout=%ss, inactivity_timeout=%ss)",
        test_name,
        len(selection),
        unit,
        len(controllers),
        run_timeout,
        inactivity_timeout,
    )
    try:
        out = orch.exec(cmd, hosts=controllers, inactivity_timeout=inactivity_timeout)
    except Exception as exc:
        fail_test(f"ARC {test_name}: execution failed (SSH/exec error): {exc}")
        update_test_result()
        return
    if print_all:
        print_test_output(log, out)
    if not out:
        fail_test(f"ARC {test_name}: no output / no reachable controllers")
        update_test_result()
        return
    results = []
    for host in controllers:
        if host not in out:
            label = amdtools_lib.node_label_from_file(cluster_dict, host)
            results.append(
                ControllerResult("node produced no output (did not run / unreachable)", None, label, None, None, None)
            )
        else:
            results.append(_evaluate_controller(orch, cluster_dict, host, out[host] or "", test_name, timestamp))
    failures = [(host, result.reason) for host, result in zip(controllers, results) if result.reason]
    if failures:
        detail = "; ".join(f"{host}: {reason}" for host, reason in failures)
        fail_test(f"ARC {test_name} failed on {len(failures)}/{len(controllers)} controller(s): {detail}")
    _record_arc_verdict(request, test_name, results)
    json_hrefs, tar_hrefs = _attach_arc_logs_to_html(
        request, config_dict, test_name, results, timestamp, bool(failures)
    )
    _capture_rundeck_results(
        request, cluster_dict, config_dict, selection, test_name, timestamp, results, json_hrefs, tar_hrefs
    )
    update_test_result()
