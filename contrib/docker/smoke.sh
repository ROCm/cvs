#!/usr/bin/env bash
# Substrate smoke checks for the CVS-runner production image (P5).
#
# Run AFTER `docker run -d --name <container>` against the new image.
# Verifies the substrate has everything install_* scripts (P6) need, but
# does NOT check for RVS/TransferBench/AGFHC/ibperf binaries -- those land
# at prepare_runtime time, not at image-build time.
#
# Usage:
#   contrib/docker/smoke.sh <container_name>
#
# Exit code: 0 if every substrate check passes, 1 on first failure.

set -uo pipefail

CONTAINER="${1:-cvs-runner}"

if [[ -z "${CONTAINER}" ]]; then
    echo "Usage: $0 <container_name>" >&2
    exit 2
fi

# Prefer caller-provided DOCKER override; else autodetect.
DOCKER="${DOCKER:-docker}"
if ! command -v ${DOCKER} >/dev/null 2>&1; then
    echo "ERROR: docker CLI not found" >&2
    exit 2
fi
if ! ${DOCKER} info >/dev/null 2>&1; then
    DOCKER="sudo ${DOCKER}"
fi

# Confirm container is running.
if ! ${DOCKER} ps --filter "name=^${CONTAINER}$" --filter "status=running" --format '{{.Names}}' \
        | grep -q "^${CONTAINER}\$"; then
    echo "ERROR: container '${CONTAINER}' is not running" >&2
    exit 2
fi

PASS=0
FAIL=0
FAILED_CHECKS=()

run_check() {
    local label="$1"; shift
    if ${DOCKER} exec "${CONTAINER}" bash -lc "$*" >/dev/null 2>&1; then
        printf '  PASS  %s\n' "${label}"
        PASS=$((PASS+1))
    else
        printf '  FAIL  %s\n' "${label}"
        FAIL=$((FAIL+1))
        FAILED_CHECKS+=("${label}")
    fi
}

echo "=== CVS-runner substrate smoke (${CONTAINER}) ==="

# Substrate: ROCm
run_check "rocminfo runs"                 "/opt/rocm/bin/rocminfo > /dev/null"
run_check "rocminfo finds GPU agents"     "/opt/rocm/bin/rocminfo | grep -q 'Marketing Name'"
run_check "/opt/rocm/.info/version"       "test -f /opt/rocm/.info/version"
run_check "/opt/rocm/.info/version-utils (AGFHC shim)" \
                                          "test -f /opt/rocm/.info/version-utils"
run_check "/opt/rocm/env_source_file.sh (RCCL shim)" \
                                          "test -x /opt/rocm/env_source_file.sh"
run_check "librocm-style libs in /opt/rocm/lib" \
                                          "ls /opt/rocm/lib/libamdhip64.so* 2>/dev/null | head -1"

# Substrate: rccl-tests (TheRock dist ships these pre-built)
run_check "rccl-tests all_reduce_perf present" \
                                          "test -x /opt/rocm/bin/all_reduce_perf"

# Substrate: build prereqs (so install_* can run inside the container)
run_check "cmake present"                 "command -v cmake"
run_check "make present"                  "command -v make"
run_check "git present"                   "command -v git"
run_check "sudo present"                  "command -v sudo"
run_check "openssh-client present"        "command -v ssh"
run_check "hipcc present (TheRock)"       "test -x /opt/rocm/bin/hipcc"

# Substrate: python deps for AGFHC
run_check "python3 + pyyaml + psutil"     "python3 -c 'import yaml, psutil'"

# Substrate: NUMA libs (TransferBench)
run_check "libnuma + numactl"             "command -v numactl && ldconfig -p | grep -q libnuma"

# Substrate: ibverbs (ibperf-tools)
run_check "libibverbs runtime"            "ldconfig -p | grep -q libibverbs"

# Substrate: OpenMPI (rccl-tests + ibperf)
run_check "mpirun present"                "command -v mpirun"

# Substrate: CVS source for install_* scripts (Dockerfile COPYs `cvs/`
# repo dir into `/opt/cvs/`, so the package layout is /opt/cvs/tests/...
# not /opt/cvs/cvs/tests/...).
run_check "CVS source baked in"           "test -d /opt/cvs/tests/health/install"
run_check "install_rvs.py present"        "test -f /opt/cvs/tests/health/install/install_rvs.py"
run_check "install_transferbench.py present" \
                                          "test -f /opt/cvs/tests/health/install/install_transferbench.py"
run_check "install_agfhc.py present"      "test -f /opt/cvs/tests/health/install/install_agfhc.py"

# Substrate: AGFHC dir empty (install_agfhc populates it later)
run_check "/opt/amd/agfhc/ exists, empty" \
                                          "test -d /opt/amd/agfhc && [ -z \"\$(ls -A /opt/amd/agfhc 2>/dev/null)\" ]"

# Substrate: manifest readable
run_check "manifest at /etc/cvs-runner-manifest.json" \
                                          "test -f /etc/cvs-runner-manifest.json && python3 -c 'import json; json.load(open(\"/etc/cvs-runner-manifest.json\"))'"

echo ""
echo "=== Summary: ${PASS} pass / ${FAIL} fail ==="
if (( FAIL > 0 )); then
    printf 'Failed checks:\n'
    printf '  - %s\n' "${FAILED_CHECKS[@]}"
    exit 1
fi
