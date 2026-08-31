'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import base64
import ipaddress
import os
import shlex

from cvs.lib import globals

log = globals.log

DEFAULT_KEY_NAME = "cluster_id"
DEFAULT_REMOTE_SSH_DIR = "~/.ssh"
SSH_CONFIG_BEGIN = "# BEGIN CVS cluster_key_distribution (managed)"
SSH_CONFIG_END = "# END CVS cluster_key_distribution (managed)"

_KNOWN_DEFAULT_KEY_NAMES = {"id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}


# ---------------------------------------------------------------------------
# Pure / logic functions
# ---------------------------------------------------------------------------


def validate_key_distribution_config(config_dict):
    """Validate config subsection and return a normalized dict with defaults applied."""
    norm = dict(config_dict)

    for field in ("cluster_key_private_path", "cluster_key_public_path"):
        val = norm.get(field, "")
        if not val:
            raise ValueError(f"ssh_key_distribution.{field} is required and must be non-empty")
        if not os.path.isfile(val):
            raise ValueError(f"ssh_key_distribution.{field}: file not found: {val!r}")

    controlling = norm.get("controlling_station_pubkey_path", "")
    if controlling and not os.path.isfile(controlling):
        raise ValueError(f"ssh_key_distribution.controlling_station_pubkey_path: file not found: {controlling!r}")

    norm.setdefault("key_name", DEFAULT_KEY_NAME)
    norm.setdefault("remote_ssh_dir", DEFAULT_REMOTE_SSH_DIR)
    norm.setdefault("ssh_config_host_pattern", "")
    norm.setdefault("verify_connectivity", True)
    norm.setdefault("verify_mode", "ring")
    norm.setdefault("verify_timeout", 20)
    norm.setdefault("ssh_config_write_mode", "managed_block")
    norm.setdefault("controlling_station_pubkey_path", "")

    key_name = norm["key_name"]
    if key_name in _KNOWN_DEFAULT_KEY_NAMES:
        log.warning(
            "ssh_key_distribution.key_name=%r matches a well-known SSH default identity; "
            "distribution may overwrite an existing key on remote nodes",
            key_name,
        )

    return norm


def collect_cluster_hostnames(cluster_dict):
    """Return ordered, de-duplicated SSH-reachable identifiers: node keys + distinct vpc_ips."""
    seen = set()
    result = []
    for node_name, node_info in cluster_dict.get("node_dict", {}).items():
        if node_name not in seen:
            seen.add(node_name)
            result.append(node_name)
        vpc_ip = node_info.get("vpc_ip", "") if isinstance(node_info, dict) else ""
        if vpc_ip and vpc_ip != node_name and vpc_ip not in seen:
            seen.add(vpc_ip)
            result.append(vpc_ip)
    return result


def _longest_common_prefix(strings):
    """Return the longest common leading substring across a non-empty list."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def derive_ssh_host_pattern(hostnames, override=""):
    """Derive the SSH Host line token(s) covering all cluster nodes.

    Resolution order:
      1. Non-empty override → use verbatim.
      2. All IPv4 → collapse on longest shared octet boundary.
      3. Non-IP names share a non-trivial common alphanumeric prefix → prefix*.
      4. Fallback: space-joined explicit list.
    """
    if override:
        return override

    if not hostnames:
        return "*"

    if len(hostnames) == 1:
        return hostnames[0]

    # Step 2: all IPv4?
    parsed_ips = []
    for h in hostnames:
        try:
            parsed_ips.append(ipaddress.ip_address(h))
        except ValueError:
            parsed_ips = []
            break

    all_ips = bool(parsed_ips)

    if all_ips:
        octets = [str(ip).split(".") for ip in parsed_ips]
        shared = 0
        for i in range(3):
            if len({o[i] for o in octets}) == 1:
                shared = i + 1
            else:
                break
        if shared == 3:
            prefix_octets = octets[0][:3]
            return ".".join(prefix_octets) + ".*"
        if shared == 2:
            prefix_octets = octets[0][:2]
            return ".".join(prefix_octets) + ".*"
        if shared == 1:
            return octets[0][0] + ".*"
        # all IPs but no shared octet prefix → explicit list (skip string-prefix step)
        return " ".join(hostnames)

    # Step 3: common alphanumeric prefix (non-IP names only)
    prefix = _longest_common_prefix(hostnames)
    if prefix and any(len(h) > len(prefix) for h in hostnames):
        return prefix + "*"

    # Step 4: explicit list
    return " ".join(hostnames)


def render_ssh_config_block(host_pattern, username, identity_file):
    """Return the ~/.ssh/config text block with BEGIN/END markers."""
    lines = [
        SSH_CONFIG_BEGIN,
        f"Host {host_pattern}",
        f"    User {username}",
        f"    IdentityFile {identity_file}",
        "    StrictHostKeyChecking no",
        "    UserKnownHostsFile /dev/null",
        "    LogLevel ERROR",
        SSH_CONFIG_END,
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Remote-command builders
# ---------------------------------------------------------------------------


def build_ensure_ssh_dir_cmd(remote_ssh_dir):
    """mkdir -p + chmod 700 on the remote ssh dir (~ preserved for remote shell expansion)."""
    if remote_ssh_dir.startswith("~"):
        dir_arg = remote_ssh_dir
    else:
        dir_arg = shlex.quote(remote_ssh_dir)
    return f"mkdir -p {dir_arg} && chmod 700 {dir_arg}"


def build_key_perms_cmd(remote_ssh_dir, key_name):
    """chmod 600 on private key, chmod 644 on public key."""
    if remote_ssh_dir.startswith("~"):
        d = remote_ssh_dir
    else:
        d = shlex.quote(remote_ssh_dir)
    priv = shlex.quote(key_name)
    pub = shlex.quote(key_name + ".pub")
    return f"chmod 600 {d}/{priv} && chmod 644 {d}/{pub}"


def build_authorize_pubkey_cmd(remote_ssh_dir, pubkey_remote_path):
    """Idempotent append of a remote pubkey file into authorized_keys using grep -qxF."""
    if remote_ssh_dir.startswith("~"):
        d = remote_ssh_dir
    else:
        d = shlex.quote(remote_ssh_dir)
    ak = f"{d}/authorized_keys"
    pub = pubkey_remote_path if pubkey_remote_path.startswith("~") else shlex.quote(pubkey_remote_path)
    # grep -qxF uses fixed-string exact whole-line match — never duplicates; key never on the cmd line
    return f"touch {ak} && chmod 600 {ak} && grep -qxF -- \"$(cat {pub})\" {ak} || cat {pub} >> {ak}"


def build_write_ssh_config_cmd(remote_ssh_dir, block_text, mode):
    """Return a shell command that installs block_text into ~/.ssh/config.

    Uses base64 encoding so the multi-line block survives over a single exec channel
    without heredoc quoting issues or libssh2's ~30 KB exec_request limit.
    """
    if remote_ssh_dir.startswith("~"):
        d = remote_ssh_dir
    else:
        d = shlex.quote(remote_ssh_dir)
    cfg = f"{d}/config"
    b64 = base64.b64encode(block_text.encode()).decode()

    if mode == "overwrite":
        return f"echo {shlex.quote(b64)} | base64 -d > {cfg} && chmod 600 {cfg}"

    # managed_block: delete old CVS block (if any) then append fresh one
    begin_q = shlex.quote(SSH_CONFIG_BEGIN)
    end_q = shlex.quote(SSH_CONFIG_END)
    sed_del = f"sed -i '/{begin_q}/,/{end_q}/d' {cfg} 2>/dev/null || true"
    append = f"echo {shlex.quote(b64)} | base64 -d >> {cfg} && chmod 600 {cfg}"
    return f"touch {cfg} && {sed_del} && {append}"


# ---------------------------------------------------------------------------
# Orchestration drivers
# ---------------------------------------------------------------------------


def upload_cluster_keys(orch, norm_config):
    """SFTP private+public cluster key to every node; apply permissions. Returns {node: bool}."""
    priv_local = norm_config["cluster_key_private_path"]
    pub_local = norm_config["cluster_key_public_path"]
    key_name = norm_config["key_name"]
    remote_ssh_dir = norm_config["remote_ssh_dir"]

    remote_priv = f"{remote_ssh_dir}/{key_name}"
    remote_pub = f"{remote_ssh_dir}/{key_name}.pub"

    results = {node: True for node in orch.all.hosts}

    for local, remote in ((priv_local, remote_priv), (pub_local, remote_pub)):
        try:
            orch.all.upload_file(local, remote)
        except IOError as e:
            log.error("SFTP upload %s -> %s failed: %r", local, remote, e)
            for node in results:
                results[node] = False

    perms_cmd = build_key_perms_cmd(remote_ssh_dir, key_name)
    out = orch.exec(perms_cmd, timeout=30, detailed=True)
    for node, detail in out.items():
        if isinstance(detail, dict):
            if detail.get("exit_code", 0) != 0:
                log.error("chmod keys failed on %s: %s", node, detail.get("output", ""))
                results[node] = False
        else:
            if "error" in str(detail).lower():
                results[node] = False

    return results


def authorize_cluster_pubkey(orch, norm_config):
    """Append cluster pubkey to authorized_keys on all nodes. Returns {node: bool}."""
    key_name = norm_config["key_name"]
    remote_ssh_dir = norm_config["remote_ssh_dir"]
    pubkey_remote = f"{remote_ssh_dir}/{key_name}.pub"

    cmd = build_authorize_pubkey_cmd(remote_ssh_dir, pubkey_remote)
    out = orch.exec(cmd, timeout=30, detailed=True)
    return _detailed_to_bool(out)


def authorize_controlling_station(orch, norm_config):
    """Upload controlling station pubkey and append to authorized_keys. No-op if path unset."""
    controlling_local = norm_config.get("controlling_station_pubkey_path", "")
    if not controlling_local:
        return {}

    remote_ssh_dir = norm_config["remote_ssh_dir"]
    remote_tmp = f"{remote_ssh_dir}/.cvs_controlling_station.pub"

    results = {node: True for node in orch.all.hosts}
    try:
        orch.all.upload_file(controlling_local, remote_tmp)
    except IOError as e:
        log.error("SFTP upload controlling station key failed: %r", e)
        for node in results:
            results[node] = False
        return results

    cmd = build_authorize_pubkey_cmd(remote_ssh_dir, remote_tmp)
    out = orch.exec(cmd, timeout=30, detailed=True)
    for node, ok in _detailed_to_bool(out).items():
        results[node] = results.get(node, True) and ok

    return results


def install_ssh_config(orch, cluster_dict, norm_config):
    """Derive host pattern, render config block, install on all nodes. Returns {node: bool}."""
    remote_ssh_dir = norm_config["remote_ssh_dir"]
    key_name = norm_config["key_name"]
    override = norm_config.get("ssh_config_host_pattern", "")
    mode = norm_config.get("ssh_config_write_mode", "managed_block")
    username = cluster_dict.get("username", "")
    identity_file = f"{remote_ssh_dir}/{key_name}"

    hostnames = collect_cluster_hostnames(cluster_dict)
    pattern = derive_ssh_host_pattern(hostnames, override=override)
    block = render_ssh_config_block(pattern, username, identity_file)

    cmd = build_write_ssh_config_cmd(remote_ssh_dir, block, mode)
    out = orch.exec(cmd, timeout=30, detailed=True)
    return _detailed_to_bool(out)


def verify_passwordless_ssh(orch, cluster_dict, norm_config):
    """Probe passwordless SSH between node pairs. Returns {(src, dst): bool}."""
    nodes = list(cluster_dict.get("node_dict", {}).keys())
    if len(nodes) < 2:
        return {}

    remote_ssh_dir = norm_config["remote_ssh_dir"]
    timeout = norm_config.get("verify_timeout", 20)
    mode = norm_config.get("verify_mode", "ring")

    ssh_config_path = f"{remote_ssh_dir}/config"

    results = {}

    if mode == "ring":
        # Build one probe command per node (node i -> node i+1 mod n), run via exec_cmd_list
        cmd_list = []
        pairs = []
        for i, src in enumerate(nodes):
            dst = nodes[(i + 1) % len(nodes)]
            if src == dst:
                continue
            pairs.append((src, dst))
            cmd_list.append(f"ssh -F {ssh_config_path} -o BatchMode=yes -o ConnectTimeout={timeout} {dst} true")
        out = orch.all.exec_cmd_list(cmd_list, timeout=timeout + 10)
        for (src, dst), output in zip(
            pairs, [out.get(node, "") for node in nodes if node != nodes[0] or len(nodes) < 2]
        ):
            results[(src, dst)] = "error" not in str(output).lower() and output is not None

        # exec_cmd_list returns {node: output}; map back by position
        node_outputs = [out.get(node, "") for node in nodes]
        results = {}
        for i, (src, dst) in enumerate(pairs):
            raw = node_outputs[i] if i < len(node_outputs) else ""
            results[(src, dst)] = raw is not None and "error" not in str(raw).lower()

    else:
        # full_mesh: O(n*(n-1)) probes, one exec per source node
        for src in nodes:
            peers = [n for n in nodes if n != src]
            if not peers:
                continue
            cmd_list = [
                f"ssh -F {ssh_config_path} -o BatchMode=yes -o ConnectTimeout={timeout} {dst} true" for dst in peers
            ]
            # Run all probes from src via a temporary single-host handle
            from cvs.lib.parallel_ssh_lib import Pssh

            tmp = Pssh(
                log,
                [src],
                user=cluster_dict.get("username"),
                pkey=cluster_dict.get("priv_key_file"),
                host_key_check=False,
            )
            try:
                src_out = tmp.exec_cmd_list(cmd_list, timeout=timeout + 10)
                outputs = [src_out.get(src, "")] if len(peers) == 1 else list(src_out.values())
                for dst, raw in zip(peers, outputs):
                    results[(src, dst)] = raw is not None and "error" not in str(raw).lower()
            finally:
                tmp.destroy_clients()

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detailed_to_bool(out):
    """Convert a detailed=True exec result dict to {node: bool}."""
    result = {}
    for node, detail in out.items():
        if isinstance(detail, dict):
            result[node] = detail.get("exit_code", 0) == 0
        else:
            result[node] = True
    return result


def _upload_local_file(orch, local_path, remote_path):
    """SFTP upload with IOError handling. Returns True on success."""
    try:
        orch.all.upload_file(local_path, remote_path)
        return True
    except IOError as e:
        log.error("SFTP upload %s -> %s failed: %r", local_path, remote_path, e)
        return False
