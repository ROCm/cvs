#!/usr/bin/env bash
# Build (and optionally ship) the CVS docker-mode P0 spike image.
#
# Usage:
#   contrib/docker/build-spike.sh [--remote user@host] [--tag <tag>]
#
# Defaults:
#   --tag      cvs-spike:latest
#   (no --remote)  build only; do not ship
#
# The spike image is intentionally minimal -- public rocm/dev-ubuntu base
# plus TransferBench. P5 replaces this with the production multi-stage
# Dockerfile.

set -euo pipefail

TAG="cvs-spike:latest"
REMOTE=""
ROCM_BASE_TAG="latest"
TRANSFERBENCH_REF="develop"
OFFLOAD_ARCH="gfx942"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)              TAG="$2";                shift 2 ;;
        --remote)           REMOTE="$2";             shift 2 ;;
        --rocm-base-tag)    ROCM_BASE_TAG="$2";      shift 2 ;;
        --transferbench-ref) TRANSFERBENCH_REF="$2"; shift 2 ;;
        --offload-arch)     OFFLOAD_ARCH="$2";       shift 2 ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)  echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

DOCKERFILE_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE="${DOCKERFILE_DIR}/Dockerfile.spike"

# Pick a docker invocation that actually works in this shell. Prefer the
# user's docker if they're in the docker group; fall back to sudo.
if docker info >/dev/null 2>&1; then
    DOCKER="docker"
elif sudo -n docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
else
    DOCKER="sudo docker"
    echo "[build-spike] note: docker requires sudo; password prompt may follow" >&2
fi

build_local() {
    echo "[build-spike] building ${TAG} locally (rocm-base=${ROCM_BASE_TAG}, transferbench=${TRANSFERBENCH_REF})"
    ${DOCKER} build \
        --build-arg "ROCM_BASE_TAG=${ROCM_BASE_TAG}" \
        --build-arg "TRANSFERBENCH_REF=${TRANSFERBENCH_REF}" \
        --build-arg "OFFLOAD_ARCH=${OFFLOAD_ARCH}" \
        -t "${TAG}" \
        -f "${DOCKERFILE}" \
        "${DOCKERFILE_DIR}"
}

build_remote() {
    echo "[build-spike] building ${TAG} on ${REMOTE} via SCP'd Dockerfile"
    # Stage the Dockerfile remotely so we don't need a build context dir.
    local stage="/tmp/cvs_spike_build_$$"
    ssh "${REMOTE}" "mkdir -p ${stage}"
    scp "${DOCKERFILE}" "${REMOTE}:${stage}/Dockerfile.spike"
    # Conductor hosts: docker usually needs sudo. The build is long-running,
    # wrap in screen so a dropped SSH does not abort it.
    local screen_name="cvs_p0_build"
    local logfile="${stage}/build.log"
    ssh "${REMOTE}" "
        set -e
        screen -X -S ${screen_name} quit 2>/dev/null || true
        screen -dmS ${screen_name} bash -c '
            sudo docker build \
                --build-arg ROCM_BASE_TAG=${ROCM_BASE_TAG} \
                --build-arg TRANSFERBENCH_REF=${TRANSFERBENCH_REF} \
                --build-arg OFFLOAD_ARCH=${OFFLOAD_ARCH} \
                -t ${TAG} \
                -f ${stage}/Dockerfile.spike \
                ${stage} > ${logfile} 2>&1
            echo BUILD_EXIT=\$? >> ${logfile}
        '
    "
    echo "[build-spike] build started on ${REMOTE} in screen ${screen_name}; tail with:"
    echo "  ssh ${REMOTE} 'tail -f ${logfile}'"
    echo "[build-spike] poll for completion with:"
    echo "  ssh ${REMOTE} 'grep BUILD_EXIT ${logfile}'"
}

if [[ -n "${REMOTE}" ]]; then
    build_remote
else
    build_local
fi
