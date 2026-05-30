#!/usr/bin/env bash
# Default CVS container provisioning script.
#
# Run inside each freshly-launched container (via `docker exec`) before
# setup_sshd. CVS's container exec model needs an in-container sshd on port
# 2224; many base images do not ship one. This installs only the sshd binary
# (openssh-server) so the existing setup_sshd can start `/usr/sbin/sshd -p2224`.
#
# Override with container.setup_script in the cluster file to install other
# packages (or to support non-apt base images -- this default is apt-only).
set -euo pipefail

if command -v sshd >/dev/null 2>&1 || [ -x /usr/sbin/sshd ]; then
    echo "default_container_setup: sshd already present, nothing to install"
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends openssh-server
