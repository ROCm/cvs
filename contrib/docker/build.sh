#!/usr/bin/env bash
# Production CVS-runner image build (P5).
#
# Substrate-only image: TheRock + apt prereqs + CVS source. P6's
# prepare_runtime installs TransferBench/RVS/AGFHC/ibperf-tools at
# container-start time via `cvs run install_*` (single source of truth).
#
# Usage:
#   contrib/docker/build.sh \
#       --rocm <therock-tarball.tar.gz> \
#       --offload-arches <gfx-targets> \
#       --tag <image:tag> \
#       [--remote user@host]
#
# Required:
#   --rocm           path on the orchestrator to a TheRock dist tarball
#   --offload-arches GPU_TARGETS string (e.g. "gfx942" or "gfx90a;gfx942")
#                    No default -- must be explicit, to avoid silent
#                    "wrong arch on this hardware" failures later.
#   --tag            image tag (e.g. cvs-runner:7.13.0a-gfx942)
#
# Optional:
#   --remote         user@host -- build on the remote node in a screen
#                    session, avoiding a multi-GB image transfer.

set -euo pipefail

ROCM=""
OFFLOAD_ARCHES=""
TAG=""
REMOTE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rocm)           ROCM="$2";           shift 2 ;;
        --offload-arches) OFFLOAD_ARCHES="$2"; shift 2 ;;
        --tag)            TAG="$2";            shift 2 ;;
        --remote)         REMOTE="$2";         shift 2 ;;
        -h|--help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

# --- arg validation ---------------------------------------------------
[[ -n "${ROCM}" ]] || { echo "ERROR: --rocm <tarball> required" >&2; exit 2; }
[[ -f "${ROCM}" ]] || { echo "ERROR: --rocm tarball not found at ${ROCM}" >&2; exit 2; }
[[ -n "${OFFLOAD_ARCHES}" ]] || {
    echo "ERROR: --offload-arches required (e.g. 'gfx942')." >&2
    echo "       No default is provided to avoid silent wrong-arch builds." >&2
    exit 2
}
[[ -n "${TAG}" ]] || { echo "ERROR: --tag required (e.g. cvs-runner:dev)" >&2; exit 2; }

DOCKERFILE_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE="${DOCKERFILE_DIR}/Dockerfile"
REPO_ROOT="$(cd "${DOCKERFILE_DIR}/../.." && pwd)"
TARBALL_NAME="$(basename "${ROCM}")"

build_local() {
    # Hard-link the (multi-GB) tarball into a temp build context so docker
    # can COPY it without a literal copy. CVS source is hard-linked too.
    local ctx
    ctx=$(mktemp -d)
    trap 'rm -rf "${ctx}"' EXIT

    cp -al "${ROCM}" "${ctx}/${TARBALL_NAME}"
    cp "${DOCKERFILE}" "${ctx}/Dockerfile"
    cp -al "${REPO_ROOT}/cvs" "${ctx}/cvs"

    # Pick docker invocation (sudo if user not in docker group).
    local DOCKER="docker"
    if ! docker info >/dev/null 2>&1; then
        if sudo -n docker info >/dev/null 2>&1; then
            DOCKER="sudo docker"
        else
            DOCKER="sudo docker"
            echo "[build] note: docker requires sudo; password prompt may follow" >&2
        fi
    fi

    echo "[build] building ${TAG} locally (offload-arches=${OFFLOAD_ARCHES})"
    ${DOCKER} build \
        --build-arg "OFFLOAD_ARCHES=${OFFLOAD_ARCHES}" \
        --build-arg "THEROCK_TARBALL_NAME=${TARBALL_NAME}" \
        -t "${TAG}" \
        -f "${ctx}/Dockerfile" \
        "${ctx}"
}

build_remote() {
    local stage="/tmp/cvs_p5_build_$$"
    local screen_name="cvs_p5_build"

    echo "[build] staging build context to ${REMOTE}:${stage}"
    ssh "${REMOTE}" "rm -rf ${stage} && mkdir -p ${stage}"

    echo "[build] copying TheRock tarball ($(du -h "${ROCM}" | cut -f1))..."
    scp -q "${ROCM}" "${REMOTE}:${stage}/${TARBALL_NAME}"

    echo "[build] copying Dockerfile + CVS source..."
    scp -q "${DOCKERFILE}" "${REMOTE}:${stage}/Dockerfile"
    # Use rsync if available for the cvs/ tree; fall back to scp -r
    if command -v rsync >/dev/null && ssh "${REMOTE}" 'command -v rsync' >/dev/null 2>&1; then
        rsync -aq --exclude '__pycache__' --exclude '*.pyc' --exclude '.cvs_venv' \
            "${REPO_ROOT}/cvs/" "${REMOTE}:${stage}/cvs/"
    else
        scp -qr "${REPO_ROOT}/cvs" "${REMOTE}:${stage}/cvs"
    fi

    # Pick remote docker invocation: sudo if needed
    local remote_docker
    if ssh "${REMOTE}" 'docker info >/dev/null 2>&1'; then
        remote_docker="docker"
    else
        remote_docker="sudo docker"
    fi

    echo "[build] starting build in screen ${screen_name} on ${REMOTE}"
    ssh "${REMOTE}" "
        screen -X -S ${screen_name} quit 2>/dev/null || true
        screen -dmS ${screen_name} bash -c '
            ${remote_docker} build \
                --build-arg OFFLOAD_ARCHES=${OFFLOAD_ARCHES} \
                --build-arg THEROCK_TARBALL_NAME=${TARBALL_NAME} \
                -t ${TAG} \
                -f ${stage}/Dockerfile \
                ${stage} > ${stage}/build.log 2>&1
            echo BUILD_EXIT=\$? >> ${stage}/build.log
        '
    "
    echo "[build] build started; tail with:"
    echo "  ssh ${REMOTE} 'tail -f ${stage}/build.log'"
    echo "[build] poll for completion with:"
    echo "  ssh ${REMOTE} 'grep BUILD_EXIT ${stage}/build.log'"
    echo "[build] cleanup staging dir AFTER build with:"
    echo "  ssh ${REMOTE} 'rm -rf ${stage}'"
}

if [[ -n "${REMOTE}" ]]; then
    build_remote
else
    build_local
fi
