#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CVS_ROOT="${SCRIPT_DIR}"
WORKSPACE="$(cd "${CVS_ROOT}/.." && pwd)"
PKG_ROOT="${CVS_ROOT}/cvs"
CLUSTER_FILE="${PKG_ROOT}/input/cluster_file/cluster_runtime.json"
MODEL_BASE="${WORKSPACE}/models"
FLUX_MODEL_DIR="${MODEL_BASE}/black-forest-labs/FLUX.1-dev"
WAN_MODEL_DIR="${MODEL_BASE}/Wan-AI/Wan2.2-I2V-A14B"
HF_HOME="${WORKSPACE}"

export PYTHONPATH="${CVS_ROOT}"
export HF_HOME="${HF_HOME}"

# Prefer HF_TOKEN, fall back to HF_KEY or .env if provided.
HF_TOKEN_VALUE="${HF_TOKEN:-${HF_KEY:-}}"
if [ -z "${HF_TOKEN_VALUE}" ] && [ -f "${WORKSPACE}/.env" ]; then
  IFS= read -r HF_TOKEN_VALUE < "${WORKSPACE}/.env" || true
fi
if [ -n "${HF_TOKEN_VALUE}" ]; then
  export HF_TOKEN="${HF_TOKEN_VALUE}"
fi

# Pick a non-loopback IPv4 if available; otherwise fall back to hostname.
node_ip="$(hostname -I 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i ~ /^[0-9.]+$/ && $i !~ /^127\./) {print $i; exit}}' || true)"
node_host="${node_ip:-$(hostname)}"
export NODE_HOST="${node_host}"
export CLUSTER_FILE="${CLUSTER_FILE}"

python3 - <<'PY'
import json
import os

node = os.environ.get("NODE_HOST") or os.environ.get("HOSTNAME") or os.uname().nodename
user = os.environ.get("USER", "")
cluster = {
    "username": user,
    "priv_key_file": f"/home/{user}/.ssh/id_rsa",
    "head_node_dict": {"mgmt_ip": node},
    "node_dict": {
        node: {
            "bmc_ip": "NA",
            "vpc_ip": node,
        }
    },
}
path = os.environ["CLUSTER_FILE"]
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as fp:
    json.dump(cluster, fp, indent=4)
print(f"Wrote cluster file: {path} (node={node})")
PY

has_model_files() {
  python3 - "$1" <<'PY'
import os
import sys

path = sys.argv[1]
if not os.path.isdir(path):
    sys.exit(1)

for root, _, files in os.walk(path):
    for name in files:
        if name == "model_index.json" or name.endswith((".safetensors", ".bin", ".pt")):
            sys.exit(0)
sys.exit(1)
PY
}

mkdir -p "${MODEL_BASE}"

if ! has_model_files "${FLUX_MODEL_DIR}"; then
  echo "Downloading FLUX.1-dev model to ${FLUX_MODEL_DIR}..."
  mkdir -p "${FLUX_MODEL_DIR}"
  rm -f "${FLUX_MODEL_DIR}/.cache/huggingface/"*.lock 2>/dev/null || true
  docker run --rm \
    --mount type=bind,source="${HF_HOME}",target=/hf_home \
    -e HF_HOME=/hf_home \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    amdsiloai/pytorch-xdit:v25.11.2 \
    hf download black-forest-labs/FLUX.1-dev \
      --local-dir /hf_home/models/black-forest-labs/FLUX.1-dev
fi

if ! has_model_files "${WAN_MODEL_DIR}"; then
  echo "Downloading WAN 2.2 I2V-A14B model to ${WAN_MODEL_DIR}..."
  mkdir -p "${WAN_MODEL_DIR}"
  rm -f "${WAN_MODEL_DIR}/.cache/huggingface/"*.lock 2>/dev/null || true
  docker run --rm \
    --mount type=bind,source="${HF_HOME}",target=/hf_home \
    -e HF_HOME=/hf_home \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    amdsiloai/pytorch-xdit:v25.11.2 \
    hf download Wan-AI/Wan2.2-I2V-A14B \
      --revision 206a9ee1b7bfaaf8f7e4d81335650533490646a3 \
      --local-dir /hf_home/models/Wan-AI/Wan2.2-I2V-A14B
fi

# Clean up any stale containers before running tests.
if command -v docker >/dev/null 2>&1; then
  ids="$(docker ps -aq || true)"
  if [ -n "${ids}" ]; then
    docker rm -f ${ids} || true
  fi
fi

# Best-effort cleanup of GPU processes owned by this user.
if command -v fuser >/dev/null 2>&1; then
  pids="$(fuser /dev/kfd 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+$/')"
  for pid in ${pids}; do
    owner="$(ps -o user= -p "${pid}" | awk '{print $1}' || true)"
    if [ "${owner}" = "${USER}" ]; then
      kill -9 "${pid}" || true
    fi
  done
fi

cd "${CVS_ROOT}"

python3 -m pytest -vvv \
  --log-file=/tmp/flux_cvs.log \
  -s cvs/tests/inference/pytorch_xdit/pytorch_xdit_flux1_dev_t2i.py \
  --cluster_file "${CLUSTER_FILE}" \
  --config_file "${PKG_ROOT}/input/config_file/inference/pytorch_xdit/mi300x_flux1_dev_t2i.json" \
  --capture=tee-sys

python3 -m pytest -vvv \
  --log-file=/tmp/wan_cvs.log \
  -s cvs/tests/inference/pytorch_xdit/pytorch_xdit_wan22_i2v_a14b.py \
  --cluster_file "${CLUSTER_FILE}" \
  --config_file "${PKG_ROOT}/input/config_file/inference/pytorch_xdit/mi300x_wan22_i2v_a14b.json" \
  --capture=tee-sys
