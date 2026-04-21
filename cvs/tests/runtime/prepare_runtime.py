"""prepare_runtime (CVS docker-mode P6).

Bring a cluster of CVS-runner containers up:

  1. preflight (per node)        -- amdgpu live, /dev/kfd, /dev/dri count, docker daemon
  2. image_stage (per node)      -- docker pull (or load) runtime.image
  3. container_start (per node)  -- docker rm -f; docker run -d ... cvs-runner
  4. arch_detect (cluster)       -- assert all nodes report same gfx arch
  5. agfhc_stage_only            -- if runtime.agfhc_tarball: scp tarball to every node
  6. component_installs          -- for name in runtime.installs:
                                       cvs run <name> ... (in docker mode)
  7. in_container_smoke          -- rocminfo + per-component verifications

Per-node artifacts at /tmp/cvs/prepare_runtime/<host>.json record gfx_arch,
manifest, per-install pass/fail, and (when AGFHC absent) agfhc_skip_reason.
Downstream test fixtures (P7's agfhc_cvs.py module-skip, P9's rvs check)
read these to skip cleanly when an install failed.

Idempotent: re-running prepare_runtime tears down any existing container and
re-installs from scratch.

Skipped entirely in host mode (no `runtime` block in cluster.json).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from cvs.lib.arch_detect import detect_cluster_gfx_arch
from cvs.lib.driver_recovery import verify_or_recover_driver
from cvs.lib.exclusivity import (
    check_exclusivity,
    render_violations,
    violation_count,
)
from cvs.lib.parallel_ssh_lib import NoOpWrapper, Pssh, wrapper_for_cluster
from cvs.lib.runtime_config import parse_runtime
from cvs.lib.utils_lib import resolve_cluster_config_placeholders
from cvs.lib import globals as cvs_globals

log = cvs_globals.log

ARTIFACT_DIR = "/tmp/cvs/prepare_runtime"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as f:
        d = json.load(f)
    return resolve_cluster_config_placeholders(d)


@pytest.fixture(scope="module")
def runtime_cfg(cluster_dict):
    cfg = parse_runtime(cluster_dict)
    if not cfg.is_docker():
        pytest.skip(
            "prepare_runtime is a no-op in host mode "
            "(cluster.json has no runtime block or runtime.mode == 'host')"
        )
    return cfg


@pytest.fixture(scope="module")
def phdl_host(cluster_dict):
    """Always uses NoOpWrapper -- commands run on the HOST, not in the container.
    Required for preflight (before any container exists), image staging, and
    `docker exec` invocations from the orchestrator side.
    """
    node_list = list(cluster_dict["node_dict"].keys())
    env_vars = cluster_dict.get("env_vars")
    return Pssh(
        log,
        node_list,
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=env_vars,
        wrapper=NoOpWrapper(),
    )


@pytest.fixture(scope="module")
def phdl_container(cluster_dict):
    """Routes commands into the running container via wrapper_for_cluster.
    Only valid AFTER container_start has run. Use for in-container smoke checks.
    """
    node_list = list(cluster_dict["node_dict"].keys())
    env_vars = cluster_dict.get("env_vars")
    return Pssh(
        log,
        node_list,
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=env_vars,
        wrapper=wrapper_for_cluster(cluster_dict),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_artifact(host: str, data: dict):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, f"{host}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _write_all_artifacts(artifacts: dict):
    paths = {}
    for host, data in artifacts.items():
        paths[host] = _write_artifact(host, data)
    return paths


def _docker(phdl_host, args: str, timeout: int = 60):
    """Run a `sudo docker <args>` command on every node via phdl_host."""
    return phdl_host.exec(f"sudo docker {args}", timeout=timeout)


def _cvs_bin() -> str:
    """Resolve the `cvs` binary the orchestrator should invoke for install_*.
    Defaults to `sys.executable`'s neighbor (so we use the same conda/venv
    `cvs` that's running this test). Override via CVS_BIN env var.
    """
    override = os.environ.get("CVS_BIN", "").strip()
    if override:
        return override
    # sys.executable is e.g. /opt/cvs_env/bin/python; cvs lives next to it
    candidate = os.path.join(os.path.dirname(sys.executable), "cvs")
    if os.path.exists(candidate):
        return candidate
    return "cvs"  # fall back to PATH lookup


# ---------------------------------------------------------------------------
# The single test_prepare_runtime function
# ---------------------------------------------------------------------------
def test_prepare_runtime(cluster_dict, runtime_cfg, phdl_host, phdl_container,
                         cluster_file, config_file):
    """End-to-end prepare flow. Pytest test so it integrates with `cvs run`.

    Each step writes progress to per-node artifacts. Hard-fail steps (preflight,
    container_start, arch_detect) abort the whole flow with artifacts saved.
    Soft-fail steps (per-install) record the failure but continue, so a missing
    AGFHC tarball or a flaky install doesn't block other tests from running.
    """
    nodes = list(cluster_dict["node_dict"].keys())
    artifacts = {
        node: {
            "phase": "preparing",
            "started_at": _now_iso(),
            "cluster_file": cluster_file,
            "config_file": config_file,
            "image": runtime_cfg.image,
            "container_name": runtime_cfg.container_name,
            "checks": {},
            "installs": {},
            "errors": [],
        }
        for node in nodes
    }

    def _abort(msg: str):
        for n in nodes:
            artifacts[n]["phase"] = "failed"
            artifacts[n]["finished_at"] = _now_iso()
        _write_all_artifacts(artifacts)
        pytest.fail(msg)

    # ---------- 1. preflight ----------
    log.info("[P6] preflight on %d node(s)", len(nodes))
    # P8: self-heal once via verify_or_recover_driver before failing. Picks up
    # the common case where Conductor reservation refresh dropped amdgpu since
    # the last session.
    try:
        recovery = verify_or_recover_driver(phdl_host)
        for node in nodes:
            if recovery.get(node, {}).get("attempted"):
                log.info("[P6] self-healed amdgpu on %s via modprobe", node)
        amdgpu_live_per_node = {n: r["after"] for n, r in recovery.items()}
    except RuntimeError as e:
        # All nodes attempted; some still dead -- record and continue to the
        # explicit per-node check below so the artifact captures everything.
        log.error("[P6] driver recovery failed: %s", e)
        amdgpu_live_per_node = {n: False for n in nodes}

    for node in nodes:
        ok = amdgpu_live_per_node.get(node, False)
        artifacts[node]["checks"]["amdgpu_live"] = ok
        if not ok:
            artifacts[node]["errors"].append(
                "amdgpu kernel module not live or /dev/kfd missing even after `sudo modprobe amdgpu`"
            )

    out = phdl_host.exec("ls /dev/dri/renderD* 2>/dev/null | wc -l", timeout=30)
    for node in nodes:
        try:
            count = int(out.get(node, "0").strip().splitlines()[-1])
        except (ValueError, IndexError):
            count = 0
        artifacts[node]["checks"]["dri_render_count"] = count
        if count == 0:
            artifacts[node]["errors"].append("no /dev/dri/renderD* nodes -- amdgpu not enumerated")

    out = phdl_host.exec("docker --version 2>&1 || sudo docker --version 2>&1", timeout=30)
    for node in nodes:
        has_docker = "Docker version" in out.get(node, "")
        artifacts[node]["checks"]["docker_present"] = has_docker
        if not has_docker:
            artifacts[node]["errors"].append("docker CLI missing on this node")

    failed_nodes = [n for n in nodes if artifacts[n]["errors"]]
    if failed_nodes:
        _abort(f"preflight failed on: {failed_nodes}; see /tmp/cvs/prepare_runtime/<host>.json")

    # ---------- 1b. exclusivity check (P10) ----------
    log.info("[P10] exclusivity check (mode=%s)", runtime_cfg.exclusivity)
    excl_summary = check_exclusivity(phdl_host, runtime_cfg)
    n_violations = violation_count(excl_summary)
    for node in nodes:
        artifacts[node]["exclusivity"] = {
            "mode": runtime_cfg.exclusivity,
            "stray_containers": excl_summary["stray_containers"].get(node, []),
            "kfd_holders": excl_summary["kfd_holders"].get(node, []),
            "reserved_ports": excl_summary["reserved_ports"].get(node, []),
        }
    if n_violations:
        msg = render_violations(excl_summary)
        if runtime_cfg.exclusivity == "strict":
            log.error("[P10] STRICT exclusivity violations: %s", msg)
            _abort(f"exclusivity (strict): {msg}")
        else:
            log.warning("[P10] WARN exclusivity violations: %s", msg)
    else:
        log.info("[P10] exclusivity: clean (no violations)")

    # ---------- 2. image_stage ----------
    log.info("[P6] image_stage runtime.image=%s ensure=%s",
             runtime_cfg.image, runtime_cfg.ensure_image)
    if runtime_cfg.ensure_image == "pull":
        out = _docker(phdl_host, f"pull {runtime_cfg.image}", timeout=1800)
        for node in nodes:
            ok = (
                "Status: Image is up to date" in out.get(node, "")
                or "Status: Downloaded newer image" in out.get(node, "")
                or "Pulled" in out.get(node, "")
            )
            # Also accept "image is already cached" -- check via inspect
            artifacts[node]["checks"]["image_pulled"] = ok or _image_present_on_node(phdl_host, runtime_cfg, node)
    elif runtime_cfg.ensure_image.startswith("load:"):
        tarball = runtime_cfg.ensure_image[len("load:"):]
        if not os.path.exists(tarball):
            _abort(f"runtime.ensure_image='load:{tarball}' but file not found on orchestrator")
        # scp + docker load on each node
        for node in nodes:
            log.info("[P6] scp %s -> %s:/tmp/cvs/_image.tar", tarball, node)
            subprocess.run(
                ["scp", "-q", tarball, f"{cluster_dict['username']}@{node}:/tmp/cvs/_image.tar"],
                check=True,
            )
        _docker(phdl_host, "load -i /tmp/cvs/_image.tar", timeout=1800)
        for node in nodes:
            artifacts[node]["checks"]["image_loaded"] = True
    else:
        _abort(f"unsupported runtime.ensure_image: {runtime_cfg.ensure_image!r}")

    # Sanity: image present on every node?
    for node in nodes:
        if not _image_present_on_node(phdl_host, runtime_cfg, node):
            _abort(f"image {runtime_cfg.image} not present on {node} after stage step")

    # ---------- 3. container_start (idempotent: rm -f then run -d) ----------
    log.info("[P6] container_start name=%s", runtime_cfg.container_name)
    _docker(phdl_host, f"rm -f {runtime_cfg.container_name}", timeout=60)

    docker_run = (
        f"run -d --name {runtime_cfg.container_name} --privileged "
        f"--network=host --ipc=host "
        f"--device=/dev/kfd --device=/dev/dri "
        f"--group-add video --group-add render "
        f"--cap-add=SYS_PTRACE --cap-add=IPC_LOCK "
        f"--ulimit memlock=-1 --shm-size=16g "
        f"-v /sys:/sys:ro -v /tmp/cvs:/tmp/cvs "
        f"{runtime_cfg.image} sleep infinity"
    )
    _docker(phdl_host, docker_run, timeout=120)

    out = _docker(
        phdl_host,
        f"ps --filter name=^{runtime_cfg.container_name}$ --filter status=running --format '{{{{.Names}}}}'",
    )
    for node in nodes:
        running = runtime_cfg.container_name in out.get(node, "")
        artifacts[node]["checks"]["container_running"] = running
        if not running:
            artifacts[node]["errors"].append(f"container '{runtime_cfg.container_name}' failed to start")
    failed_nodes = [n for n in nodes if not artifacts[n]["checks"].get("container_running")]
    if failed_nodes:
        _abort(f"container_start failed on: {failed_nodes}")

    # ---------- 4. arch_detect (cluster invariant) ----------
    log.info("[P6] arch_detect")
    try:
        gfx_arch = detect_cluster_gfx_arch(phdl_host, runtime_cfg)
    except RuntimeError as e:
        for node in nodes:
            artifacts[node]["errors"].append(f"arch_detect: {e}")
        _abort(f"arch_detect failed: {e}")
    log.info("[P6] gfx_arch=%s", gfx_arch)
    for node in nodes:
        artifacts[node]["gfx_arch"] = gfx_arch

    # ---------- 5. agfhc_stage_only ----------
    if runtime_cfg.agfhc_tarball:
        if not os.path.exists(runtime_cfg.agfhc_tarball):
            for node in nodes:
                artifacts[node]["agfhc_staged"] = False
                artifacts[node]["agfhc_skip_reason"] = (
                    f"tarball not found on orchestrator: {runtime_cfg.agfhc_tarball}"
                )
        else:
            # P7: prior container runs may have left /tmp/cvs/ root-owned (bind
            # mount is created as root by docker). Re-chown to the SSH user
            # before scp so the tarball drop succeeds without sudo.
            log.info("[P6] chown /tmp/cvs to %s on each node (pre-scp)",
                     cluster_dict["username"])
            phdl_host.exec(
                f"sudo chown -R {cluster_dict['username']}:{cluster_dict['username']} /tmp/cvs 2>/dev/null; "
                f"sudo chmod 0755 /tmp/cvs 2>/dev/null; true",
                timeout=30,
            )
            log.info("[P6] agfhc_stage scp %s to %d node(s)",
                     runtime_cfg.agfhc_tarball, len(nodes))
            staged_ok = True
            for node in nodes:
                target = f"{cluster_dict['username']}@{node}:/tmp/cvs/agfhc.tar.bz2"
                rc = subprocess.run(
                    ["scp", "-q", runtime_cfg.agfhc_tarball, target],
                    capture_output=True, text=True,
                )
                if rc.returncode == 0:
                    artifacts[node]["agfhc_staged"] = True
                    artifacts[node]["agfhc_tarball_remote_path"] = "/tmp/cvs/agfhc.tar.bz2"
                else:
                    artifacts[node]["agfhc_staged"] = False
                    artifacts[node]["agfhc_skip_reason"] = (
                        f"scp failed: {rc.stderr.strip()[:200]}"
                    )
                    staged_ok = False
            if not staged_ok:
                log.warning("[P6] AGFHC staging failed on some nodes; install_agfhc will likely skip")
    else:
        for node in nodes:
            artifacts[node]["agfhc_staged"] = False
            artifacts[node]["agfhc_skip_reason"] = "no agfhc_tarball configured"

    # ---------- 6. component_installs ----------
    installs = runtime_cfg.resolved_installs()
    log.info("[P6] resolved installs: %s", installs)
    cvs_bin = _cvs_bin()
    for name in installs:
        log.info("[P6] running %s %s", cvs_bin, name)
        rc = subprocess.run(
            [cvs_bin, "run", name,
             "--cluster_file", cluster_file,
             "--config_file", config_file],
            capture_output=True, text=True, timeout=1800,
        )
        for node in nodes:
            artifacts[node]["installs"][name] = {
                "exit": rc.returncode,
                "ok": rc.returncode == 0,
                "stderr_tail": rc.stderr[-500:] if rc.stderr else "",
            }
        if rc.returncode == 0:
            log.info("[P6]   %s: PASS", name)
        else:
            log.warning("[P6]   %s: FAIL (exit=%d)", name, rc.returncode)

    # ---------- 7. in_container_smoke ----------
    log.info("[P6] in_container_smoke")
    out = phdl_container.exec("/opt/rocm/bin/rocminfo > /dev/null 2>&1 && echo ok", timeout=60)
    for node in nodes:
        artifacts[node]["checks"]["in_container_rocminfo"] = "ok" in out.get(node, "")

    # Per-install verification: only if the install reported PASS.
    def _per_install_smoke(install_name: str, probe_cmd: str, key: str):
        any_ok = any(
            artifacts[n]["installs"].get(install_name, {}).get("ok") for n in nodes
        )
        if not any_ok:
            return
        out = phdl_container.exec(probe_cmd, timeout=60)
        for node in nodes:
            if not artifacts[node]["installs"].get(install_name, {}).get("ok"):
                artifacts[node]["checks"][key] = "skipped (install failed)"
                continue
            text = out.get(node, "").strip()
            artifacts[node]["checks"][key] = text[-200:] if text else "no output"

    _per_install_smoke("install_rvs",
                       "/opt/rocm/bin/rvs --version 2>&1 | head -1",
                       "rvs_version")
    _per_install_smoke("install_agfhc",
                       "/opt/amd/agfhc/agfhc --version 2>&1 | head -3",
                       "agfhc_version")
    _per_install_smoke("install_transferbench",
                       "ls /opt/INSTALL/TransferBench/TransferBench 2>&1 | head -1",
                       "transferbench_path")

    # ---------- finalize ----------
    for node in nodes:
        artifacts[node]["phase"] = "ready"
        artifacts[node]["finished_at"] = _now_iso()
    paths = _write_all_artifacts(artifacts)
    for node, p in paths.items():
        log.info("[P6] artifact %s -> %s", node, p)


# ---------------------------------------------------------------------------
# Helpers used inside the test
# ---------------------------------------------------------------------------
def _image_present_on_node(phdl_host, runtime_cfg, node) -> bool:
    """Quick `docker images <tag>` check on a single node."""
    out = _docker(
        phdl_host,
        f"images --filter reference={runtime_cfg.image} --format '{{{{.Repository}}}}:{{{{.Tag}}}}'",
        timeout=30,
    )
    return runtime_cfg.image in out.get(node, "")
