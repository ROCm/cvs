#!/bin/bash
# Copy the SSH private key the container will actually use.
# Called by full-rebuild.sh after `docker compose up -d`.
#
#   jump_host.enabled != true  ->  copy cluster.ssh.key_file
#   jump_host.enabled == true  ->  copy jump_host.key_file
#   never copy jump_host.node_key_file (that file lives on the jump host)
#
# Host lookup is by basename under the home directory of cluster.ssh.username:
#   ~/.ssh/<key>  or  /root/.ssh/<key>  ->  <home>/.ssh/<key>  into  /root/.ssh/<key>
#
# Overrides:
#   CONTAINER_NAME   container to copy into   (default: cvs-cluster-monitor)
#   CONFIG_FILE      cluster config to read   (default: config/cluster.yaml)
#   SSH_USER         host account to read keys from
#   HOST_SSH         host directory holding the keys

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONTAINER_NAME="${CONTAINER_NAME:-cvs-cluster-monitor}"
DEST="/root/.ssh"

if [ -z "${CONFIG_FILE:-}" ]; then
    if [ -f config/cluster.yaml ]; then
        CONFIG_FILE="config/cluster.yaml"
    else
        CONFIG_FILE="config/cluster.yaml.example"
    fi
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: cluster config not found: $CONFIG_FILE"
    exit 1
fi

# Read a key from the cluster.ssh block only, stopping before jump_host.
yaml_ssh_value() {
    awk -v want="$1" '
        /^[[:space:]]*ssh:[[:space:]]*$/ { in_ssh = 1; next }
        in_ssh && /^[[:space:]]*jump_host:/ { exit }
        in_ssh {
            key = $1
            sub(/:$/, "", key)
            if (key == want) {
                gsub(/["'\'']/, "", $2)
                if ($2 != "") { print $2; exit }
            }
        }
    ' "$CONFIG_FILE"
}

# Whose ~/.ssh to read on the host. Template placeholders fall back to the
# invoking account (SUDO_USER when run through sudo).
if [ -z "${SSH_USER:-}" ]; then
    SSH_USER="$(yaml_ssh_value username)"
fi
case "$SSH_USER" in
    ""|your_username|user|username)
        SSH_USER="${SUDO_USER:-$USER}"
        ;;
esac

HOST_HOME="$(getent passwd "$SSH_USER" | cut -d: -f6)"
if [ -z "$HOST_HOME" ]; then
    echo "Error: no home directory for user '$SSH_USER' on this host."
    echo "Set SSH_USER or HOST_SSH explicitly."
    exit 1
fi
HOST_SSH="${HOST_SSH:-${HOST_HOME}/.ssh}"

if [ ! -d "$HOST_SSH" ]; then
    echo "Error: SSH directory not found: $HOST_SSH"
    exit 1
fi

if ! sudo docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Error: container '$CONTAINER_NAME' is not running."
    echo "Start it first (full-rebuild.sh does this before this step)."
    exit 1
fi

echo "Config:    $CONFIG_FILE"
echo "SSH user:  $SSH_USER"
echo "Host keys: $HOST_SSH"
echo "Target:    ${CONTAINER_NAME}:${DEST}"
echo ""

# Exactly one key: ssh.key_file, or jump_host.key_file when jump is enabled.
mapfile -t KEY_PATHS < <(awk '
    function indent(s) { match(s, /^[[:space:]]*/); return RLENGTH }
    /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
    {
        ind = indent($0)
        if (in_jump && ind <= jump_indent) in_jump = 0
    }
    /^[[:space:]]*jump_host:/ { in_jump = 1; jump_indent = ind; next }
    /node_key_file:/ { next }
    {
        sub(/#.*/, "")
        key = $1; sub(/:$/, "", key)
        val = $2; gsub(/["'\'']/, "", val)
        if (key == "enabled" && in_jump) jump_enabled = val
        else if (key == "key_file" && val != "") {
            if (in_jump) jump_key = val; else ssh_key = val
        }
    }
    END {
        if (jump_enabled == "true") {
            if (jump_key != "") print jump_key
        } else if (ssh_key != "") {
            print ssh_key
        }
    }
' "$CONFIG_FILE")

if [ "${#KEY_PATHS[@]}" -eq 0 ]; then
    echo "Error: no key_file entries in $CONFIG_FILE"
    exit 1
fi

sudo docker exec "$CONTAINER_NAME" mkdir -p "$DEST"
sudo docker exec "$CONTAINER_NAME" chmod 700 "$DEST"

copy_one() {
    local src="$1" dest_name="$2"
    sudo docker cp "$src" "${CONTAINER_NAME}:${DEST}/${dest_name}"
    # docker cp keeps the host uid/gid; the container runs as root.
    sudo docker exec "$CONTAINER_NAME" chown 0:0 "${DEST}/${dest_name}"
    sudo docker exec "$CONTAINER_NAME" chmod 600 "${DEST}/${dest_name}"
}

copied=0
missing=0

for keypath in "${KEY_PATHS[@]}"; do
    name="$(basename "$keypath")"
    src="${HOST_SSH}/${name}"
    echo "  ${keypath} <- ${src}"
    if [ ! -f "$src" ]; then
        echo "    not found on host"
        missing=$((missing + 1))
        continue
    fi
    copy_one "$src" "$name"
    copied=$((copied + 1))
    echo "    copied"
done

echo ""
echo "Contents of ${CONTAINER_NAME}:${DEST}:"
sudo docker exec "$CONTAINER_NAME" ls -la "$DEST" || true

if [ "$copied" -eq 0 ]; then
    echo ""
    echo "Error: none of the key_file paths in $CONFIG_FILE were found under $HOST_SSH"
    exit 1
fi

if [ "$missing" -ne 0 ]; then
    echo ""
    echo "$missing key file(s) missing under $HOST_SSH."
    echo "Add them with the same basename and re-run, or upload them in the Configuration UI."
fi
