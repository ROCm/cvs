"""verify_runtime (CVS docker-mode P8).

Standalone health-check + driver-recovery test, callable as
`cvs run verify_runtime --cluster_file ... --config_file ...`.

Two intended use cases:

  1. Between destructive tests (e.g. after `rvs_cvs` LEVEL-4 unloads amdgpu),
     before running `agfhc_cvs` or `rccl_*` so the driver is back to a known
     state and the container's stale device handles are refreshed.

  2. As a standalone health check after a Conductor reservation refresh /
     silent reboot dropped the amdgpu module.

Usable in BOTH host mode and docker mode:

  * Host mode: just runs the driver-recovery sequence on the host. Same value
    cvs-config-gen Phase 2 provides today, but invocable as a CVS command.
  * Docker mode: also `docker restart`s the cvs-runner container so its
    device-fd cache is refreshed against the (potentially newly-loaded) host
    amdgpu module, then verifies in-container `rocminfo` works.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from cvs.lib.driver_recovery import restart_container, verify_or_recover_driver
from cvs.lib.parallel_ssh_lib import NoOpWrapper, Pssh, wrapper_for_cluster
from cvs.lib.runtime_config import parse_runtime
from cvs.lib.utils_lib import resolve_cluster_config_placeholders
from cvs.lib import globals as cvs_globals

log = cvs_globals.log

ARTIFACT_DIR = "/tmp/cvs/verify_runtime"


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as f:
        d = json.load(f)
    return resolve_cluster_config_placeholders(d)


@pytest.fixture(scope="module")
def runtime_cfg(cluster_dict):
    return parse_runtime(cluster_dict)


@pytest.fixture(scope="module")
def phdl_host(cluster_dict):
    """Always uses NoOpWrapper -- driver_recovery commands run on the host."""
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
    """Routes commands into the container via wrapper_for_cluster.
    Only meaningful in docker mode; in host mode resolves to NoOp.
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


def test_verify_runtime(cluster_dict, runtime_cfg, phdl_host, phdl_container):
    """End-to-end host-side driver verify + (docker mode only) container restart."""
    artifact = {
        "started_at": _now_iso(),
        "runtime_mode": runtime_cfg.mode,
    }

    # --- Step 1: host-side driver verify + recovery (always) -----------
    log.info("[P8] verify_or_recover_driver()")
    try:
        artifact["driver"] = verify_or_recover_driver(phdl_host)
    except RuntimeError as e:
        artifact["status"] = "driver_recovery_failed"
        artifact["error"] = str(e)
        artifact["finished_at"] = _now_iso()
        _write_artifact(artifact)
        pytest.fail(f"driver_recovery failed: {e}")

    # --- Step 2: docker-mode-only steps ------------------------------
    if runtime_cfg.is_docker():
        log.info("[P8] docker restart %s", runtime_cfg.container_name)
        artifact["container_restart"] = restart_container(
            phdl_host, runtime_cfg.container_name
        )
        not_running = [n for n, ok in artifact["container_restart"].items() if not ok]
        if not_running:
            artifact["status"] = "container_restart_failed"
            artifact["finished_at"] = _now_iso()
            _write_artifact(artifact)
            pytest.fail(
                f"container '{runtime_cfg.container_name}' not running after restart on: {not_running}"
            )

        log.info("[P8] in-container rocminfo smoke")
        out = phdl_container.exec(
            "/opt/rocm/bin/rocminfo > /dev/null 2>&1 && echo ok",
            timeout=60,
        )
        artifact["in_container_rocminfo"] = {n: "ok" in v for n, v in out.items()}
        not_ok = [n for n, ok in artifact["in_container_rocminfo"].items() if not ok]
        if not_ok:
            artifact["status"] = "in_container_smoke_failed"
            artifact["finished_at"] = _now_iso()
            _write_artifact(artifact)
            pytest.fail(f"in-container rocminfo failed on: {not_ok}")

    artifact["status"] = "ok"
    artifact["finished_at"] = _now_iso()
    _write_artifact(artifact)
    log.info("[P8] verify_runtime completed: status=ok")


def _write_artifact(data):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, "verify.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("[P8] artifact -> %s", path)
