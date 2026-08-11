#!/usr/bin/env bash
# Shared RCCL A/B regression runner (no #SBATCH directives).
# Invoked by sbatch/rccl_ab.sbatch or manually via srun inside an allocation.
#
# Optional (per-run workspace isolation — see sbatch/lib/workspace.sh):
#   RCCL_CI_WORKSPACE=1  run out of runs/<run_key>/ (own cvs worktree, own
#                        cvs-sbatch copy, own artifacts and logs) instead of the
#                        shared trees. Default 0 = legacy shared behaviour.

set -euo pipefail

readonly RCCL_CI_ROOT="${RCCL_CI_ROOT:-/it-share/rccl-ci}"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly JOB_TAG="${SLURM_JOB_ID:-local}"

# Orchestrator must run on the login/submit host and SSH to compute nodes.
# Running inside `srun` on a compute node ties the pytest process to that Slurm
# step; a node blip or heavy MPI load kills the whole pipeline before artifacts
# are written.  Refuse that launch mode with a clear message.
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_STEP_ID:-}" ]]; then
  echo "[ERROR] Do not launch run_rccl_ab.sh via srun on a compute node." >&2
  echo "        Use sbatch, or from the login node with the wrap-job env:" >&2
  echo "          export SLURM_JOB_ID=<id> SLURM_NODELIST='<nodes>' SLURM_NNODES=4" >&2
  echo "          CONFIG_JSON=... bash ${RCCL_CI_ROOT}/sbatch/run_rccl_ab.sh" >&2
  exit 1
fi

# --- per-run workspace (no-op unless RCCL_CI_WORKSPACE=1) ---------------------
# shellcheck source=/dev/null
source "${RCCL_CI_ROOT}/sbatch/lib/workspace.sh"

# ws_init is idempotent: the build job created this workspace, we re-enter it.
# It is also safe if detect runs without a preceding build (manual submission) —
# the workspace is created fresh and the config's lib paths are whatever the
# caller set.
if ws_enabled; then
  ws_init || { echo "[ERROR] workspace init failed" >&2; exit 1; }
  ws_begin
  trap 'ws_end' EXIT
fi

if ws_enabled; then
  export RUN_LOG_DIR="$(ws_root)/logs"
  export CVS_DIR="$(ws_root)/cvs"
  export CVS_SBATCH_DIR="$(ws_root)/cvs-sbatch"
  AB_ARTIFACT_DIR="$(ws_root)/artifacts"
else
  export RUN_LOG_DIR="${RCCL_CI_ROOT}/logs/run_${TIMESTAMP}_${JOB_TAG}"
  export CVS_DIR="${RCCL_CI_ROOT}/cvs"
  export CVS_SBATCH_DIR="${RCCL_CI_ROOT}/cvs-sbatch"
  AB_ARTIFACT_DIR="${RCCL_CI_ROOT}/ab_artifacts"
fi
mkdir -p "${RUN_LOG_DIR}" "${AB_ARTIFACT_DIR}"

readonly SLURM_LOG_DIR="${RCCL_CI_ROOT}/logs"
mkdir -p "${SLURM_LOG_DIR}"

export SKIP_CVS_SETUP=1
export CONFIG_JSON="${CONFIG_JSON:-${RCCL_CI_ROOT}/configs/ab_robustness.json}"
# Normalise to an absolute path. run.sh runs from cvs-sbatch/ and resolves the
# config with realpath, so a path relative to RCCL_CI_ROOT (or the submit cwd)
# would otherwise fail there. Accept either form.
if [[ "${CONFIG_JSON}" != /* ]]; then
  if [[ -f "${RCCL_CI_ROOT}/${CONFIG_JSON}" ]]; then
    CONFIG_JSON="${RCCL_CI_ROOT}/${CONFIG_JSON}"
  elif [[ -f "${CONFIG_JSON}" ]]; then
    CONFIG_JSON="$(cd "$(dirname "${CONFIG_JSON}")" && pwd)/$(basename "${CONFIG_JSON}")"
  fi
  export CONFIG_JSON
fi
[[ -f "${CONFIG_JSON}" ]] || { echo "[ERROR] CONFIG_JSON not found: ${CONFIG_JSON}" >&2; exit 1; }

# Point output_dir and the /tmp sweep scratch at this run. Idempotent — the
# build job already did this, but detect may run standalone.
ws_config_retarget "${CONFIG_JSON}" || true

export TEST_PATH="${TEST_PATH:-./cvs/tests/rccl/rccl_ab_regression.py}"
export LOG_FILE="${RUN_LOG_DIR}/pytest.log"

# Seed librocm_smi64.so.1 compat symlink on every allocated node before MPI.
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  echo "[INFO] Seeding /tmp/rocm_smi_fix on allocation nodes..."
  while IFS= read -r _node; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${_node}" \
      'mkdir -p /tmp/rocm_smi_fix && ln -sf /opt/rocm/lib/librocm_smi64.so.7 /tmp/rocm_smi_fix/librocm_smi64.so.1' \
      2>/dev/null || echo "[WARN] could not seed rocm_smi fix on ${_node}" >&2
  done < <(scontrol show hostnames "${SLURM_NODELIST}")
fi

echo "========================================================================"
echo "RCCL A/B regression CI"
echo "  Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "  Nodes        : ${SLURM_NNODES:-N/A} (${SLURM_NODELIST:-N/A})"
echo "  Config       : ${CONFIG_JSON}"
echo "  Run log dir  : ${RUN_LOG_DIR}"
if ws_enabled; then
echo "  Workspace    : $(ws_root)"
echo "  Artifacts    : ${AB_ARTIFACT_DIR}"
echo "  CVS worktree : ${CVS_DIR}"
fi
echo "========================================================================"

# Per-PR builds of reference + candidate librccl.so now happen in a separate,
# earlier sbatch (sbatch/rccl_build.sbatch, single build-node allocation) so the
# compile doesn't hold this 4-node detection allocation idle. CONFIG_JSON is
# expected to already point at the correct reference/candidate libs by the time
# this script runs — see sbatch/run_rccl_build.sh.

# Warm up PRTE daemon connections on all nodes before the test loop.
# First mpirun in a fresh allocation races the daemon on one node (cold-start);
# this throwaway hostname ping stabilises all connections before real sweeps begin.
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  echo "[INFO] Warming up PRTE daemon connections across all nodes..."
  _hostspec=$(scontrol show hostnames "${SLURM_NODELIST}" | awk '{print $1":8"}' | paste -sd,)
  _np=$(( $(scontrol show hostnames "${SLURM_NODELIST}" | wc -l) * 8 ))
  timeout 60s /it-share/ompi-5.0.8/bin/mpirun \
    --allow-run-as-root \
    -np "${_np}" -H "${_hostspec}" \
    --bind-to numa \
    --mca pml ob1 --mca btl tcp,self \
    --mca oob_tcp_if_include eno0,eno1 \
    --mca btl_tcp_if_include eno0,eno1 \
    hostname >/dev/null 2>&1 \
    && echo "[INFO] PRTE warmup complete." \
    || echo "[WARN] PRTE warmup timed out or failed — continuing anyway."
fi

cd "${CVS_SBATCH_DIR}" || { echo "[ERROR] cannot cd ${CVS_SBATCH_DIR}" >&2; exit 1; }

if [[ ! -x run.sh ]]; then chmod +x run.sh 2>/dev/null || true; fi

./run.sh
pytest_exit=$?

# In workspace mode artifacts already live inside the workspace next to logs/,
# so there is nothing to copy. In legacy mode snapshot the shared dir into the
# per-run log dir before the next run overwrites it.
if ! ws_enabled && [[ -d "${AB_ARTIFACT_DIR}" ]]; then
  cp -a "${AB_ARTIFACT_DIR}" "${RUN_LOG_DIR}/"
fi

if [[ -f "${SLURM_LOG_DIR}/sp_tests-${JOB_TAG}.out" ]]; then
  cp -f "${SLURM_LOG_DIR}/sp_tests-${JOB_TAG}.out" "${RUN_LOG_DIR}/slurm.out" 2>/dev/null || true
  cp -f "${SLURM_LOG_DIR}/sp_tests-${JOB_TAG}.err" "${RUN_LOG_DIR}/slurm.err" 2>/dev/null || true
fi

ln -sfn "${RUN_LOG_DIR}" "${RCCL_CI_ROOT}/logs/latest"

# Gate on confirmed regressions, not on pytest exit code.
# Pytest exits 1 on intermittent harness failures (empty output, SSH blips) even
# when the regression detector found 0 confirmed regressions — those are cluster
# noise, not actual regressions. Re-read the report and derive the gate exit code
# from the confirmed-regression count so the SLURM job only fails on real issues.
_REPORT="${AB_ARTIFACT_DIR}/ab_regression_report.json"
if [[ -f "${_REPORT}" ]]; then
  _confirmed=$(python3 -c "
import json, sys
try:
    d = json.load(open('${_REPORT}'))
    n = sum(r.get('summary', {}).get('regressions', 0) for r in d.get('reports', {}).values())
    print(n)
except Exception as e:
    print('ERR', file=sys.stderr); sys.exit(2)
" 2>/dev/null)
  if [[ "${_confirmed}" == "0" ]]; then
    echo "[INFO] Verdict: PASS — 0 confirmed regressions (pytest_exit=${pytest_exit})"
    exit_code=0
  elif [[ "${_confirmed}" =~ ^[0-9]+$ ]]; then
    echo "[INFO] Verdict: FAIL — ${_confirmed} confirmed regression(s)"
    exit_code=1
  else
    echo "[WARN] Could not parse report; falling back to pytest exit code ${pytest_exit}"
    exit_code="${pytest_exit}"
  fi
else
  echo "[WARN] Report not found at ${_REPORT}; using pytest exit code ${pytest_exit}"
  exit_code="${pytest_exit}"
fi

ws_record verdict_exit_code "${exit_code}" || true

echo "[INFO] A/B run finished with exit code ${exit_code} (pytest_exit=${pytest_exit})"
echo "[INFO] Artifacts: ${RUN_LOG_DIR}"
exit "${exit_code}"
