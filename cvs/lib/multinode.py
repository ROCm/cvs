"""Multi-node MPI bootstrap inside cvs-runner containers (CVS docker-mode P13).

Provides ephemeral SSH key distribution + container-side `~/.ssh/config` so
that `mpirun` launching from one container can reach `sshd` (port 2222) in
the cvs-runner containers on peer nodes via host networking.

Lifecycle (called from prepare_runtime):
  1. setup_multinode_ssh(phdl_host, phdl_container, runtime_cfg, nodes)
       - Generate one ephemeral RSA keypair on the orchestrator (/tmp/cvs/<container>_id_rsa).
       - Push the pubkey into `/root/.ssh/authorized_keys` of every container.
       - Push `~/.ssh/config` into every container that maps each peer node to port 2222
         and disables strict host key checking (ephemeral, single-cluster scope).
  2. teardown_multinode_ssh()  -- best-effort delete of orchestrator-side key files.

We assume:
  - cvs-runner containers run on the host network (`--network=host`), so
    `<peer-hostname>:2222` is reachable directly. (cvs-config-gen Phase 6
    confirms this is the standard CVS deployment.)
  - Peer-to-peer SSH inside the cluster is allowed -- cluster.json's
    `node_dict` already expresses the peer set.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List

from cvs.lib import globals as cvs_globals

log = cvs_globals.log

EPHEMERAL_KEY_DIR = "/tmp/cvs"


def _orchestrator_key_path(container_name: str) -> str:
    return os.path.join(EPHEMERAL_KEY_DIR, f"{container_name}_id_rsa")


def generate_ephemeral_key(container_name: str) -> tuple[str, str]:
    """Generate a fresh RSA keypair on the orchestrator. Returns (priv_path, pub_str)."""
    Path(EPHEMERAL_KEY_DIR).mkdir(parents=True, exist_ok=True)
    priv = _orchestrator_key_path(container_name)
    pub = priv + ".pub"
    if os.path.exists(priv):
        os.remove(priv)
    if os.path.exists(pub):
        os.remove(pub)
    log.info("[P13] generating ephemeral keypair at %s", priv)
    subprocess.run(
        ["ssh-keygen", "-t", "rsa", "-b", "2048", "-N", "", "-q", "-f", priv,
         "-C", f"cvs-runner-ephemeral-{container_name}"],
        check=True,
    )
    pub_str = Path(pub).read_text().strip()
    return priv, pub_str


def _push_file_to_container(
    phdl_host,
    container_name: str,
    contents: str,
    target_path: str,
    mode: str = "600",
) -> None:
    """Write `contents` to `target_path` inside `container_name` on every node.

    Uses base64 + docker exec stdin to avoid shell-quoting hazards (newlines,
    single/double quotes, special chars in keys).
    """
    import base64
    b64 = base64.b64encode(contents.encode()).decode()
    target_dir = "/".join(target_path.split("/")[:-1]) or "/"
    cmd = (
        f"sudo docker exec -i {container_name} bash -c "
        f"'mkdir -p {target_dir} && chmod 700 {target_dir} && "
        f"echo {b64} | base64 -d > {target_path} && chmod {mode} {target_path}'"
    )
    phdl_host.exec(cmd, timeout=30)


def push_authorized_key(phdl_host, container_name: str, pub_key: str) -> None:
    """Append the orchestrator's pubkey to /root/.ssh/authorized_keys in every container."""
    log.info("[P13] pushing ephemeral pubkey into container '%s'", container_name)
    _push_file_to_container(
        phdl_host, container_name, pub_key + "\n", "/root/.ssh/authorized_keys", "600"
    )


def push_ssh_config(phdl_host, container_name: str, peer_nodes: Iterable[str]) -> None:
    """Write ~/.ssh/config inside every container so mpirun targets port 2222."""
    config_lines = [
        "Host *",
        "    StrictHostKeyChecking no",
        "    UserKnownHostsFile /dev/null",
        "    Port 2222",
        "    User root",
        "    IdentityFile /root/.ssh/id_rsa",
        "",
    ]
    for node in peer_nodes:
        config_lines += [
            f"Host {node}",
            f"    HostName {node}",
            "    Port 2222",
            "    User root",
            "",
        ]
    config_str = "\n".join(config_lines)
    _push_file_to_container(
        phdl_host, container_name, config_str, "/root/.ssh/config", "600"
    )


def push_private_key(phdl_host, container_name: str, priv_key_path: str) -> None:
    """Copy the orchestrator's private key into every container at /root/.ssh/id_rsa."""
    priv_str = Path(priv_key_path).read_text()
    _push_file_to_container(
        phdl_host, container_name, priv_str, "/root/.ssh/id_rsa", "600"
    )


def verify_container_sshd(phdl_host, container_name: str) -> dict:
    """Confirm sshd is listening on port 2222 inside every container.

    Uses the kernel's /proc/net/tcp interface (always available, no extra
    package install required). Port 2222 in hex is 0x08AE; a listening
    socket has st=0A. The probe also falls back to a simple `pgrep sshd`
    in case /proc parsing fails.
    """
    probe = (
        f"sudo docker exec {container_name} bash -c "
        f"\"awk 'NR>1 && \\$2 ~ /:08AE\\$/ && \\$4 == \\\"0A\\\" {{print}}' "
        f"/proc/net/tcp; pgrep -x sshd | head -1\""
    )
    out = phdl_host.exec(probe, timeout=15)
    return {node: bool(raw.strip()) for node, raw in out.items()}


_HOSTNAME_RE = __import__("re").compile(r"^[A-Za-z0-9._-]+$")


def verify_container_to_container_ssh(
    phdl_host, container_name: str, source_node: str, target_node: str
) -> bool:
    """Confirm that container on source_node can SSH into container on target_node.

    Success criterion is strict: the LAST line of stdout must look like a
    hostname (alphanumeric + `.` + `-` + `_`). Anything else -- error tokens,
    empty output, banner text, etc. -- is treated as failure. This prevents
    false-positives when sshd isn't actually running but ssh fails silently.
    """
    cmd = (
        f"sudo docker exec {container_name} bash -c "
        f"'ssh -o ConnectTimeout=10 -o BatchMode=yes {target_node} hostname 2>&1' "
        f"| tail -1"
    )
    out = phdl_host.exec(cmd, timeout=30)
    raw = out.get(source_node, "")
    if not raw:
        return False
    line = raw.strip().splitlines()[-1] if raw.strip() else ""
    if not line or not _HOSTNAME_RE.match(line):
        return False
    if any(tok in line.lower() for tok in ("error", "denied", "refused", "timed", "unreachable")):
        return False
    return True


def setup_multinode_ssh(phdl_host, runtime_cfg, nodes: List[str]) -> dict:
    """End-to-end ephemeral key + ssh_config push to every container."""
    container = runtime_cfg.container_name

    priv_path, pub_str = generate_ephemeral_key(container)
    push_authorized_key(phdl_host, container, pub_str)
    push_private_key(phdl_host, container, priv_path)
    push_ssh_config(phdl_host, container, nodes)
    sshd_status = verify_container_sshd(phdl_host, container)

    # Self-loopback test from each node to itself.
    self_loops = {}
    for node in nodes:
        # Only test the local node from its own container; aggregate per-node.
        single_phdl = type(phdl_host)(
            log,
            [node],
            user=phdl_host.user if hasattr(phdl_host, "user") else None,
            pkey=phdl_host.pkey if hasattr(phdl_host, "pkey") else None,
        )
        self_loops[node] = verify_container_to_container_ssh(
            single_phdl, container, node, node
        )

    return {
        "container": container,
        "key_path": priv_path,
        "sshd_listening": sshd_status,
        "self_loopback_ssh_ok": self_loops,
    }


def teardown_multinode_ssh(container_name: str) -> None:
    """Best-effort delete of orchestrator-side ephemeral key files."""
    priv = _orchestrator_key_path(container_name)
    for p in (priv, priv + ".pub"):
        try:
            if os.path.exists(p):
                os.remove(p)
                log.info("[P13] removed orchestrator ephemeral key: %s", p)
        except Exception as exc:
            log.warning("[P13] failed to remove %s: %s", p, exc)
