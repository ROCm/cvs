#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
bootstrap_xdit_models.sh

Bootstrap (download + stage) WAN 2.2 and FLUX.1-dev model files into local directories
in the exact layout expected by CVS pytorch-xdit benchmarks.

This script is meant to be run ONCE per cluster to pre-stage models on shared storage.
Benchmarks then run fully offline by pointing config JSON model_repo to the staged path.

Requirements:
- docker available on the machine running this script
- network access to Hugging Face for the initial staging (gated models require accepted license)

Token handling:
- Provide either --hf-token-file (preferred) OR --hf-token.
- The token is passed to docker via a temporary --env-file (not printed).

Examples:
  ./bootstrap_xdit_models.sh \
    --hf-token-file "$HOME/.hf_token" \
    --models-root "/models" \
    --hf-home "/hf_cache"

  ./bootstrap_xdit_models.sh \
    --hf-token "$(cat /path/to/token.txt)" \
    --flux-dir "/models/black-forest-labs/FLUX.1-dev" \
    --wan-dir  "/models/Wan-AI/Wan2.2-I2V-A14B"

EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }

HF_TOKEN=""
HF_TOKEN_FILE=""
MODELS_ROOT="/models"
HF_HOME_DIR="/hf_cache"
FLUX_DIR=""
WAN_DIR=""
IMAGE="amdsiloai/pytorch-xdit:v25.11.2"

# Pinned revisions used by CVS configs / known-good runs
WAN_REPO_ID="Wan-AI/Wan2.2-I2V-A14B"
WAN_REV="206a9ee1b7bfaaf8f7e4d81335650533490646a3"

FLUX_REPO_ID="black-forest-labs/FLUX.1-dev"
# If you want to pin FLUX, set this to a commit hash. Leave empty for latest default branch.
FLUX_REV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --hf-token) HF_TOKEN="${2:-}"; shift 2 ;;
    --hf-token-file) HF_TOKEN_FILE="${2:-}"; shift 2 ;;
    --models-root) MODELS_ROOT="${2:-}"; shift 2 ;;
    --hf-home) HF_HOME_DIR="${2:-}"; shift 2 ;;
    --flux-dir) FLUX_DIR="${2:-}"; shift 2 ;;
    --wan-dir) WAN_DIR="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    --flux-rev) FLUX_REV="${2:-}"; shift 2 ;;
    --wan-rev) WAN_REV="${2:-}"; shift 2 ;;
    *) die "Unknown arg: $1 (use --help)" ;;
  esac
done

if [[ -n "$HF_TOKEN_FILE" ]]; then
  [[ -f "$HF_TOKEN_FILE" ]] || die "--hf-token-file does not exist: $HF_TOKEN_FILE"
  HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi
HF_TOKEN="${HF_TOKEN//$'\r'/}"
HF_TOKEN="${HF_TOKEN//$'\n'/}"
[[ -n "$HF_TOKEN" ]] || die "Provide --hf-token-file or --hf-token"

[[ -n "$FLUX_DIR" ]] || FLUX_DIR="${MODELS_ROOT}/black-forest-labs/FLUX.1-dev"
[[ -n "$WAN_DIR"  ]] || WAN_DIR="${MODELS_ROOT}/Wan-AI/Wan2.2-I2V-A14B"

mkdir -p "$HF_HOME_DIR" "$FLUX_DIR" "$WAN_DIR"

# Create temp env file so token is not on process argv.
ENVFILE="$(mktemp)"
cleanup() { rm -f "$ENVFILE"; }
trap cleanup EXIT
chmod 600 "$ENVFILE"
printf "HF_TOKEN=%s\n" "$HF_TOKEN" > "$ENVFILE"
printf "HF_HOME=/hf_home\n" >> "$ENVFILE"

docker_download() {
  local repo_id="$1"; shift
  local local_dir="$1"; shift
  local cache_dir="$1"; shift

  docker run --rm \
    --user "$(id -u):$(id -g)" \
    --env-file "$ENVFILE" \
    --mount "type=bind,source=${cache_dir},target=/hf_home" \
    --mount "type=bind,source=${local_dir},target=/out" \
    "$IMAGE" \
    hf download "$repo_id" \
      --cache-dir /hf_home \
      --local-dir /out \
      --max-workers 1 \
      "$@"
}

echo "Staging FLUX into: $FLUX_DIR"
flux_args=()
if [[ -n "$FLUX_REV" ]]; then
  flux_args+=(--revision "$FLUX_REV")
fi

# For FLUX we stage only what diffusers pipeline needs for these benchmarks.
# (This avoids downloading unrelated files while still ensuring /model is loadable offline.)
docker_download "$FLUX_REPO_ID" "$FLUX_DIR" "$HF_HOME_DIR" \
  "${flux_args[@]}" \
  --include "model_index.json" \
  --include "scheduler/*" \
  --include "tokenizer/*" \
  --include "tokenizer_2/*" \
  --include "text_encoder/*" \
  --include "text_encoder_2/*" \
  --include "transformer/*" \
  --include "vae/*"

echo "Staging WAN into: $WAN_DIR"
# For WAN we download the full snapshot at the pinned revision to avoid missing any needed subfolders.
docker_download "$WAN_REPO_ID" "$WAN_DIR" "$HF_HOME_DIR" \
  --revision "$WAN_REV"

cat <<EOF

Done.

Update CVS configs to point to these staged paths:
- Flux model_repo: $FLUX_DIR
- Wan  model_repo: $WAN_DIR

HF cache used (can be shared across runs): $HF_HOME_DIR
Container image used: $IMAGE

EOF

