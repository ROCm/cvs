#!/usr/bin/env bash
# Assert that RCCL resolves ROCr correctly and that BOTH A/B sides agree on the
# transport capabilities they will use.
#
# WHY
# ---
# RCCL probes ROCr at init via dlopen("libhsa-runtime64.so"). If the ROCm dist
# has a flattened layout (several inodes for one SONAME — see
# normalize_rocm_dist.sh), that dlopen returns a SECOND, uninitialised copy of
# ROCr. hsa_system_get_info then returns 4107, RCCL jumps to its error path,
# pfn_hsa_amd_portable_export_dmabuf is never resolved, and DMA-BUF export is
# silently disabled. Multi-node collectives hang for the full rccl_timeout.
#
# Whether a given librccl is affected depends on how it was BUILT: a build that
# links ROCr via DT_NEEDED is immune, one that relies on dlopen is not. That is
# the dangerous part for an A/B gate — reference and candidate can differ. When
# reference has DMA-BUF off and candidate has it on, the comparison is not
# merely slow, it is INVALID: the candidate scores as a huge fake improvement.
#
# So this script checks two things:
#   1. liveness  — each side actually has DMA-BUF enabled
#   2. symmetry  — both sides resolved the SAME capabilities, so the A/B
#                  measurement is comparing library changes and nothing else
#
# USAGE
#   check_dmabuf.sh --lib DIR              # probe one side by lib directory
#   check_dmabuf.sh --ldpath 'A:B:C'       # probe one side by full LD_LIBRARY_PATH
#   check_dmabuf.sh --config ci_detect.json  # probe reference AND candidate, compare
#
# Requires GPUs: run inside an allocation with a GRES request (e.g.
# --exclusive --gres=gpu:8). Without a gres request Slurm's device cgroup hides
# /dev/kfd and the probe reports "no ROCm-capable device is detected".
#
# Exit 0 = healthy (and symmetric, in --config mode)
#      1 = DMA-BUF disabled on a side, or the two sides disagree
#      2 = inconclusive / harness failure

set -uo pipefail

RCCL_CI_ROOT="${RCCL_CI_ROOT:-/it-share/rccl-ci}"
ROCM_DIST="${ROCM_DIST:-${RCCL_CI_ROOT}/rocm_devel}"
PERF_BIN="${RCCL_CI_ROOT}/rccl-tests-2.30.4/bin/all_reduce_perf"
NGPU="${DMABUF_CHECK_NGPU:-2}"

LIB_DIR=""; LD_PATH=""; CONFIG=""; ROCR_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lib)       LIB_DIR="$2"; shift 2 ;;
    --ldpath)    LD_PATH="$2"; shift 2 ;;
    --config)    CONFIG="$2"; shift 2 ;;
    --rocr-path) ROCR_PATH="$2"; shift 2 ;;
    --ngpu)      NGPU="$2"; shift 2 ;;
    -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -x "${PERF_BIN}" ]] || { echo "[ERROR] missing ${PERF_BIN}" >&2; exit 2; }

# --- probe one side -----------------------------------------------------------
# Echoes "<dmabuf> <rocr_version> <hsa_needed> <resolved_lib>" on stdout;
# human-readable detail goes to stderr so callers can capture the verdict alone.
# dmabuf is one of: enabled | disabled | unknown
probe_side() {
  local label="$1" ldpath="$2"
  local log resolved hsa_needed dmabuf rocr rc

  log="$(mktemp)"

  resolved="$(LD_LIBRARY_PATH="${ldpath}" ldd "${PERF_BIN}" 2>/dev/null | awk '/librccl/ {print $3}')"
  hsa_needed="$(readelf -d "${resolved}" 2>/dev/null | grep -c 'hsa-runtime')"
  [[ -z "${hsa_needed}" ]] && hsa_needed=0

  {
    echo "--- ${label} ---"
    echo "  LD_LIBRARY_PATH : ${ldpath}"
    echo "  resolved librccl: ${resolved:-<unresolved>}"
    echo "  hsa DT_NEEDED   : ${hsa_needed} (0 = dlopen path, sensitive to dist layout)"
  } >&2

  env -i \
    PATH="/usr/bin:/bin" \
    HOME="${HOME}" \
    LD_LIBRARY_PATH="${ldpath}" \
    NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=INIT \
    HSA_NO_SCRATCH_RECLAIM=1 \
    NCCL_IGNORE_CPU_AFFINITY=1 \
    ${ROCR_PATH:+RCCL_ROCR_PATH="${ROCR_PATH}"} \
    timeout 180s "${PERF_BIN}" -b 8 -e 8 -f 2 -g "${NGPU}" > "${log}" 2>&1
  rc=$?

  rocr="$(grep -oiE 'ROCr version [0-9.]+' "${log}" | head -1 | awk '{print $3}')"
  [[ -z "${rocr}" ]] && rocr="none"

  if grep -qi 'DMA_BUF Support Enabled' "${log}"; then
    dmabuf="enabled"
  elif grep -qE '4107|DMA_BUF Support Disabled|Could not find .*dmabuf' "${log}"; then
    dmabuf="disabled"
  else
    dmabuf="unknown"
    { echo "  [!] no ROCr verdict (perf exit=${rc}); tail:"; tail -12 "${log}" | sed 's/^/      /'; } >&2
  fi

  grep -iE 'rocr|dma.?buf|4107' "${log}" | sed 's/^/  /' >&2
  echo "  => dmabuf=${dmabuf} rocr=${rocr}" >&2
  echo >&2

  rm -f "${log}"
  echo "${dmabuf} ${rocr} ${hsa_needed} ${resolved:-none}"
}

echo "=========================================================================="
echo "RCCL transport capability preflight"
echo "  node      : $(hostname)"
echo "  rocm dist : ${ROCM_DIST}"
echo "  rocr path : ${ROCR_PATH:-<unset — default dlopen resolution>}"
echo "=========================================================================="

# --- single-side mode ---------------------------------------------------------
if [[ -z "${CONFIG}" ]]; then
  if [[ -n "${LIB_DIR}" ]]; then
    LD_PATH="${LIB_DIR}:${ROCM_DIST}/lib:/it-share/ompi-5.0.8/lib"
  fi
  [[ -n "${LD_PATH}" ]] || { echo "[ERROR] need --lib, --ldpath or --config" >&2; exit 2; }

  read -r dmabuf rocr _hsa _lib <<< "$(probe_side "side" "${LD_PATH}")"
  case "${dmabuf}" in
    enabled)  echo "[PASS] DMA-BUF enabled (ROCr ${rocr})."; exit 0 ;;
    disabled) echo "[FAIL] DMA-BUF DISABLED — RCCL loaded a second, uninitialised ROCr." >&2
              echo "       Check dist layout: ${RCCL_CI_ROOT}/sbatch/lib/normalize_rocm_dist.sh --check" >&2
              exit 1 ;;
    *)        echo "[INCONCLUSIVE] no ROCr verdict." >&2; exit 2 ;;
  esac
fi

# --- A/B mode: probe both sides and compare -----------------------------------
[[ -f "${CONFIG}" ]] || { echo "[ERROR] config not found: ${CONFIG}" >&2; exit 2; }

read -r REF_LD CAND_LD <<< "$(python3 -c "
import json,sys
d=json.load(open('${CONFIG}'))
ab=d.get('rccl',{}).get('ab_regression',{})
r=ab.get('reference',{}).get('ld_library_path','')
c=ab.get('candidate',{}).get('ld_library_path','')
if not r or not c: sys.exit(3)
print(r,c)
" 2>/dev/null)" || { echo "[ERROR] could not read ld_library_path for both sides from ${CONFIG}" >&2; exit 2; }

read -r R_DMABUF R_ROCR R_HSA R_LIB <<< "$(probe_side "reference" "${REF_LD}")"
read -r C_DMABUF C_ROCR C_HSA C_LIB <<< "$(probe_side "candidate" "${CAND_LD}")"

echo "=== capability summary ==="
printf '  %-10s dmabuf=%-9s rocr=%-6s hsa_needed=%s\n' "reference" "${R_DMABUF}" "${R_ROCR}" "${R_HSA}"
printf '  %-10s dmabuf=%-9s rocr=%-6s hsa_needed=%s\n' "candidate" "${C_DMABUF}" "${C_ROCR}" "${C_HSA}"
echo

status=0

# 1. Liveness. A side without DMA-BUF will hang the multi-node collectives.
for side in reference candidate; do
  v="R_DMABUF"; [[ "${side}" == "candidate" ]] && v="C_DMABUF"
  case "${!v}" in
    enabled) ;;
    disabled)
      echo "[FAIL] ${side}: DMA-BUF is DISABLED — multi-node collectives will hang." >&2
      echo "       Fix the dist layout: ${RCCL_CI_ROOT}/sbatch/lib/normalize_rocm_dist.sh --check" >&2
      status=1 ;;
    *)
      echo "[WARN] ${side}: capability probe inconclusive." >&2
      [[ ${status} -eq 0 ]] && status=2 ;;
  esac
done

# 2. Symmetry. This is the correctness gate: differing capabilities mean the A/B
# result measures the environment, not the code change under test.
if [[ "${R_DMABUF}" != "${C_DMABUF}" ]]; then
  echo "[FAIL] A/B ASYMMETRY: reference dmabuf=${R_DMABUF} but candidate dmabuf=${C_DMABUF}." >&2
  echo "       The two sides would not use the same transport path, so any" >&2
  echo "       measured delta reflects the environment, not the code change." >&2
  echo "       Refusing to report a verdict from an invalid comparison." >&2
  status=1
fi

if [[ ${status} -eq 0 ]]; then
  echo "[PASS] both sides: DMA-BUF enabled, ROCr ${R_ROCR} — capabilities symmetric."
fi
exit "${status}"
