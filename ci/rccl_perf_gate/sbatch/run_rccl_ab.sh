#!/usr/bin/env bash
# Shared RCCL A/B regression runner (no #SBATCH directives).
# Invoked by sbatch/rccl_ab.sbatch or manually via srun inside an allocation.
#
# Optional (per-run workspace isolation — see sbatch/lib/workspace.sh):
#   RCCL_CI_WORKSPACE=1  run out of runs/<run_key>/ (own cvs worktree, own
#                        cvs-sbatch copy, own artifacts and logs) instead of the
#                        shared trees. Default 0 = legacy shared behaviour.
#
#   RCCL_CI_CAPABILITY_GATE=0  skip the pre-flight transport capability check
#                        (lib/check_dmabuf.sh). Default 1 = check.

set -euo pipefail

# Everything this job writes lands on a shared NFS tree that several people
# operate by hand. A default umask of 022 (or 077 under some runner setups) left
# artifacts and per-run configs that only their creator could delete, so cleanup
# by anyone else failed silently and the files accumulated. Group-writable is the
# correct mode for shared CI state.
umask 002

readonly RCCL_CI_ROOT="${RCCL_CI_ROOT:-/it-share/rccl-ci}"

# Helper libs ship next to this script in ROCm/cvs. Prefer that copy so a fresh
# checkout is self-contained; fall back to the legacy NFS deploy path for the
# hand-maintained /it-share/rccl-ci/sbatch/ copies.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${_SELF_DIR}/lib" ]]; then
  readonly RCCL_CI_LIB="${_SELF_DIR}/lib"
else
  readonly RCCL_CI_LIB="${RCCL_CI_ROOT}/sbatch/lib"
fi
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
source "${RCCL_CI_LIB}/workspace.sh"

# ws_init is idempotent: the build job created this workspace, we re-enter it.
# It is also safe if detect runs without a preceding build (manual submission) —
# the workspace is created fresh and the config's lib paths are whatever the
# caller set.
if ws_enabled; then
  ws_init || { echo "[ERROR] workspace init failed" >&2; exit 1; }
  ws_begin
  # EXIT alone is not enough. Bash runs EXIT traps when the script returns or
  # calls exit -- not when it is killed by an untrapped signal, which is precisely
  # how a job ends when it hits --time or someone runs scancel. Those are the runs
  # that leak a .active marker and pin their workspace against gc forever.
  # Trapping the signals explicitly makes the EXIT trap fire on those paths too.
  trap 'ws_end' EXIT INT TERM HUP
fi

if ws_enabled; then
  export RUN_LOG_DIR="$(ws_root)/logs"
  export CVS_DIR="$(ws_root)/cvs"
  export CVS_SBATCH_DIR="$(ws_root)/cvs-sbatch"
  AB_ARTIFACT_DIR="$(ws_root)/artifacts"
  # Stamp the report with the code that produced it. The shared cvs checkout is
  # edited between runs, so "the detector" is not a fixed thing -- without the SHA
  # a report on NFS cannot be attributed to a version afterwards.
  RCCL_CI_RUN_KEY="$(ws_run_key)"; export RCCL_CI_RUN_KEY
  RCCL_CI_CVS_SHA="$(git -C "${CVS_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  export RCCL_CI_CVS_SHA
  echo "[INFO] detector code: ${CVS_DIR} @ ${RCCL_CI_CVS_SHA}"
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

# --- transport capability pre-flight ------------------------------------------
# A ~10s check that both A/B sides resolve ROCr and end up with DMA-BUF. It
# catches two failure modes that are otherwise completely silent:
#
#   1. DMA-BUF disabled on a side -> multi-node collectives fall back to a path
#      that hangs on this fabric, so every sweep burns the full rccl_timeout and
#      a 25min run becomes hours. This is precisely what a flattened ROCm dist
#      caused before lib/normalize_rocm_dist.sh existed.
#   2. reference/candidate ASYMMETRY -> worse than slow. The two sides use
#      different transports, so the measured delta reflects the environment
#      rather than the code change, and the candidate scores a large fake
#      improvement. A green verdict from an asymmetric pair is a false negative,
#      which is the one outcome a regression gate must never produce.
#
# Runs directly in the batch-script process, which holds the job's GPU cgroup
# (verified: the .batch step is allocated gres/gpu=8 under plain --exclusive).
# Do NOT wrap this in srun — an srun step without its own --gres gets no
# /dev/kfd and the probe reports "no ROCm-capable device is detected".
if [[ "${RCCL_CI_CAPABILITY_GATE:-1}" == "1" && -x "${RCCL_CI_LIB}/check_dmabuf.sh" ]]; then
  echo "[INFO] Pre-flight: RCCL transport capability check..."
  set +e
  timeout 600s bash "${RCCL_CI_LIB}/check_dmabuf.sh" --config "${CONFIG_JSON}"
  _gate_rc=$?
  set -e
  case "${_gate_rc}" in
    0)
      echo "[INFO] Capability gate PASSED."
      ws_record capability_gate pass || true
      ;;
    1)
      echo "[ERROR] Capability gate FAILED — refusing to start a 4-node A/B that" >&2
      echo "        would either hang or produce an invalid comparison." >&2
      echo "        Check the ROCm dist layout first:" >&2
      echo "          ${RCCL_CI_LIB}/normalize_rocm_dist.sh --check" >&2
      ws_record capability_gate fail || true
      exit 1
      ;;
    *)
      # Inconclusive (2), or the timeout tripped (124). Do not block on a probe
      # that could not reach a verdict: the detector has its own hang
      # protection, and a flaky pre-flight must never be able to fail a good PR.
      echo "[WARN] Capability gate inconclusive (exit ${_gate_rc}); continuing." >&2
      ws_record capability_gate "inconclusive:${_gate_rc}" || true
      ;;
  esac
else
  echo "[INFO] Capability gate skipped (RCCL_CI_CAPABILITY_GATE=${RCCL_CI_CAPABILITY_GATE:-1})."
fi

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

# `set -e` is on for this whole script, which means a bare `./run.sh` that exits
# non-zero terminates the script THERE -- `pytest_exit=$?` never runs, and neither
# does anything below it: no artifact snapshot, no slurm.out copy, no logs/latest,
# no verdict recount, no ws_record. Every failing run since this was written has
# silently skipped all of it (165 job logs, only 39 ever reached the tail, and all
# 39 had pytest_exit=0 -- the recount block below has never once executed on a
# failure, which is the only case it exists for). Disable errexit across the call
# so a non-zero exit is data, not a control-flow event.
set +e
./run.sh
pytest_exit=$?
set -e

# In workspace mode artifacts already live inside the workspace next to logs/,
# so there is nothing to copy. In legacy mode snapshot the shared dir into the
# per-run log dir before the next run overwrites it.
if ! ws_enabled && [[ -d "${AB_ARTIFACT_DIR}" ]]; then
  cp -a "${AB_ARTIFACT_DIR}" "${RUN_LOG_DIR}/"
fi

# Link, don't copy. These are the job's own stdout/stderr and they are still open
# and being appended to right now, so a copy captures a truncated prefix and then
# diverges from the real file forever -- the "slurm.out" in the run dir was always
# missing its own tail, including the verdict lines printed below. A symlink into
# logs/ always reads current, costs nothing, and cannot go stale.
for _s in out err; do
  _src="${SLURM_LOG_DIR}/sp_tests-${JOB_TAG}.${_s}"
  [[ -e "${_src}" ]] && ln -sfn "${_src}" "${RUN_LOG_DIR}/slurm.${_s}"
done

# logs/latest is a global "most recent run" convenience pointer, and with runs no
# longer serialised by a single GitHub lock, two jobs can reach this line at once.
# ln -sfn on an existing symlink is not atomic (unlink + symlink), so a reader can
# catch the gap and see no such file. Create it under a temp name and rename over:
# rename(2) on the same filesystem is atomic, so a reader sees old or new, never
# nothing. It is still last-writer-wins as to WHICH run it points at -- that is
# inherent to a single global pointer, which is why nothing in the gate reads it.
_latest_tmp="${RCCL_CI_ROOT}/logs/.latest.$$"
ln -sfn "${RUN_LOG_DIR}" "${_latest_tmp}" 2>/dev/null \
  && mv -Tf "${_latest_tmp}" "${RCCL_CI_ROOT}/logs/latest" 2>/dev/null \
  || rm -f "${_latest_tmp}" 2>/dev/null || true

# Gate on confirmed regressions, not on pytest exit code.
# Pytest exits 1 on intermittent harness failures (empty output, SSH blips) even
# when the regression detector found 0 confirmed regressions — those are cluster
# noise, not actual regressions. Re-read the report and derive the gate exit code
# from the confirmed-regression count so the SLURM job only fails on real issues.
#
# The dangerous half of that reasoning is "0 confirmed regressions => PASS". It is
# only true if the detector actually looked. A run that measured nothing, dropped
# half its keys, or died at repeat 4 of 7 also reports 0 confirmed regressions, and
# this block would happily convert its pytest failure into a green gate. So the
# report now carries an explicit `trustworthy` flag and we refuse to override
# pytest's verdict unless it is set.
_REPORT="${AB_ARTIFACT_DIR}/ab_regression_report.json"
if [[ -f "${_REPORT}" ]]; then
  _verdict=$(python3 -c "
import json, sys
try:
    d = json.load(open('${_REPORT}'))
    reports = d.get('reports', {}) or {}
    n = sum(r.get('summary', {}).get('regressions', 0) for r in reports.values())
    # Trust the top-level flag when present. Fall back to the per-group flags so a
    # report written by an older detector still can't claim more than it knows.
    top = d.get('trustworthy')
    if top is None:
        top = bool(reports) and all(
            r.get('summary', {}).get('trustworthy', False) for r in reports.values())
    print('%d %s' % (n, 'trustworthy' if top else 'untrustworthy'))
    print(' | '.join(str(x) for x in (d.get('untrustworthy_reasons') or [])[:5]), file=sys.stderr)
except Exception:
    print('ERR ERR')
")
  _confirmed="${_verdict%% *}"
  _trust="${_verdict##* }"
  if [[ "${_trust}" != "trustworthy" ]]; then
    echo "[ERROR] Verdict: NO VERDICT — the report is not trustworthy (${_trust})."
    echo "[ERROR] 0 confirmed regressions here means 'we could not look', not 'nothing broke'."
    echo "[ERROR] Failing the job rather than reporting a green gate. See ${_REPORT}."
    exit_code="${pytest_exit}"
    (( exit_code == 0 )) && exit_code=1
  elif [[ "${_confirmed}" == "0" ]]; then
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
  # No report at all is an infra failure, not a pass — never let a missing report
  # exit 0 just because run.sh happened to return 0.
  (( exit_code == 0 )) && { echo "[ERROR] run.sh exited 0 but produced no report."; exit_code=1; }
fi

ws_record verdict_exit_code "${exit_code}" || true

# Opportunistic cleanup. The cron janitor is the real mechanism (see ws_janitor),
# but a detect run is the one thing guaranteed to happen on this cluster, so it
# also sweeps -- that way the disk stays bounded even if nobody ever installs the
# cron entry. Best-effort: a gc failure must never change the verdict.
ws_gc || true

echo "[INFO] A/B run finished with exit code ${exit_code} (pytest_exit=${pytest_exit})"
echo "[INFO] Artifacts: ${RUN_LOG_DIR}"
exit "${exit_code}"
