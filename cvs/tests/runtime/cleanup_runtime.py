"""cleanup_runtime (CVS docker-mode P6).

Tear down the per-node CVS-runner containers brought up by prepare_runtime.

Symmetric pair to prepare_runtime; safe to invoke at any time. Idempotent --
if no container is running, the test still succeeds and writes a cleanup
artifact noting the no-op.

Skipped entirely in host mode (no `runtime` block in cluster.json).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from cvs.lib.parallel_ssh_lib import NoOpWrapper, Pssh
from cvs.lib.runtime_config import parse_runtime
from cvs.lib.utils_lib import resolve_cluster_config_placeholders
from cvs.lib import globals as cvs_globals

log = cvs_globals.log

ARTIFACT_DIR = "/tmp/cvs/cleanup_runtime"


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
    cfg = parse_runtime(cluster_dict)
    if not cfg.is_docker():
        pytest.skip("cleanup_runtime is a no-op in host mode")
    return cfg


@pytest.fixture(scope="module")
def phdl_host(cluster_dict):
    """Always uses NoOpWrapper -- we run docker on the host, not in container."""
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


def test_cleanup_runtime(cluster_dict, runtime_cfg, phdl_host):
    """Remove the cvs-runner container on every node and record the outcome."""
    nodes = list(cluster_dict["node_dict"].keys())
    artifact = {
        "phase": "cleaning",
        "started_at": _now_iso(),
        "container_name": runtime_cfg.container_name,
        "image": runtime_cfg.image,
        "per_node": {},
    }

    # Snapshot pre-state: was the container running?
    before = phdl_host.exec(
        f"sudo docker ps -a --filter name=^{runtime_cfg.container_name}$ "
        f"--format '{{{{.Status}}}}'",
        timeout=30,
    )
    for node in nodes:
        artifact["per_node"][node] = {
            "before_status": before.get(node, "").strip() or "absent",
        }

    # docker rm -f always; ignore errors (treat container-absent as success).
    rm_out = phdl_host.exec(
        f"sudo docker rm -f {runtime_cfg.container_name} 2>&1",
        timeout=60,
    )
    for node in nodes:
        artifact["per_node"][node]["rm_output"] = rm_out.get(node, "").strip()

    # Verify gone.
    after = phdl_host.exec(
        f"sudo docker ps -a --filter name=^{runtime_cfg.container_name}$ --format '{{{{.Names}}}}'",
        timeout=30,
    )
    for node in nodes:
        still_there = runtime_cfg.container_name in after.get(node, "")
        artifact["per_node"][node]["removed"] = not still_there

    artifact["phase"] = "done"
    artifact["finished_at"] = _now_iso()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    out_path = os.path.join(ARTIFACT_DIR, "cleanup.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    log.info("[P6] cleanup artifact -> %s", out_path)

    # Hard-fail only if any node still has the container after rm.
    not_removed = [
        n for n, d in artifact["per_node"].items() if not d["removed"]
    ]
    if not_removed:
        pytest.fail(f"container still present on: {not_removed}")
