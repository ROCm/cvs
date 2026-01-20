#!/usr/bin/env bash

##############################################################
# SBATCH configuration for CVS benchmarking
# Author: surya.periaswamy@amd.com
# Date: 09/11/2025
#
# Submit with: sbatch /apps/cvs_tests/cvs-sbatch/sbatch/default.sbatch
# Debug mode:  DEBUG_MODE=1 sbatch sbatch/default.sbatch
##############################################################

#SBATCH --job-name=aorta-automation-tests
#SBATCH --output=/scratch/users/speriasw/tmp/slurm/%x-%j.out
#SBATCH --error=/scratch/users/speriasw/tmp/slurm/%x-%j.err
#SBATCH --reservation=rccl_415
#SBATCH --account=rccl
#SBATCH --partition=rccl
#SBATCH --exclusive

# Enable strict error handling
set -euo pipefail

# Enable debug mode if requested
if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  set -x  # Print commands as they execute
  echo "========================================" >&2
  echo "[DEBUG] SBATCH script starting" >&2
  echo "[DEBUG] Job ID: ${SLURM_JOB_ID}" >&2
  echo "[DEBUG] Submit directory: ${SLURM_SUBMIT_DIR}" >&2
  echo "[DEBUG] Current directory: $(pwd)" >&2
  echo "[DEBUG] Hostname: $(hostname)" >&2
  echo "========================================" >&2
fi

# Always run from the shared cvs-sbatch root directory (so `run.sh`, configs, and relative paths resolve).
# Note: Slurm stages the script under /var/spool/slurmd/... on compute nodes, so BASH_SOURCE is not a
# reliable way to find the repo. Use a shared path instead.
RUN_DIR="${CVS_SBATCH_DIR:-/apps/cvs_tests/cvs-sbatch}"
if [[ ! -d "${RUN_DIR}" ]]; then
  # Fallback: if someone has a different checkout location, try SLURM_SUBMIT_DIR.
  RUN_DIR="${SLURM_SUBMIT_DIR:-${RUN_DIR}}"
fi

if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  echo "[DEBUG] SBATCH script path: ${BASH_SOURCE[0]}" >&2
  echo "[DEBUG] Run dir: ${RUN_DIR}" >&2
fi

cd "${RUN_DIR}" || {
  echo "[ERROR] Failed to cd to ${RUN_DIR}" >&2
  exit 1
}

# Optional pre-flight: run setup_nodes.py across allocated nodes.
# Enable with: SETUP_NODES=1 sbatch ...
if [[ "${SETUP_NODES:-0}" == "1" ]]; then
  echo "[INFO] SETUP_NODES=1: running setup_nodes.py pre-flight" >&2

  # setup_nodes.py depends on `paramiko`; default to TheRock venv if available.
  py_cmd="python3"
  if [[ -f "${SETUP_NODES_VENV_ACTIVATE:-/apps/cvs_tests/TheRock/.venv/bin/activate}" ]]; then
    # shellcheck disable=SC1091
    source "${SETUP_NODES_VENV_ACTIVATE:-/apps/cvs_tests/TheRock/.venv/bin/activate}"
    # Prefer the venv python (usually `python`), rather than the system python3.
    py_cmd="python"
  fi

  # Ensure paramiko is available; if not, fall back to a one-off uv isolated run.
  if "${py_cmd}" -c "import paramiko" >/dev/null 2>&1; then
    "${py_cmd}" "${RUN_DIR}/setup_nodes.py"
  else
    echo "[WARNING] paramiko not available in ${py_cmd}; running setup_nodes.py via uv one-off env" >&2
    if command -v uv >/dev/null 2>&1; then
      uv run --isolated --no-project --with paramiko python3 "${RUN_DIR}/setup_nodes.py"
    else
      echo "[ERROR] uv not found and paramiko missing; cannot run setup_nodes.py" >&2
      exit 1
    fi
  fi

  deactivate >/dev/null 2>&1 || true
fi

# Verify run.sh exists
if [[ ! -f "run.sh" ]]; then
  echo "[ERROR] run.sh not found in $(pwd)" >&2
  echo "[ERROR] Directory contents:" >&2
  ls -la >&2
  exit 1
fi

# Verify run.sh is executable
if [[ ! -x "run.sh" ]]; then
  echo "[WARNING] run.sh is not executable, making it executable" >&2
  chmod +x run.sh
fi

if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  echo "[DEBUG] Executing run.sh..." >&2
fi

# Execute run.sh (DEBUG_MODE will be inherited)
./run.sh
exit_code=$?

# Collect artifacts (best-effort):
# - Archive *all* per-job artifacts under: /apps/cvs_tests/artifacts_archive/<job_id>/
# - Publish only the latest 3 HTML artifacts (heatmap, perf report, pytest report) to:
#   /apps/cvs_tests/test_reports/ with stable names for serving (no job id / timestamp).
artifact_src="/tmp/cvs_rccl_${USER}_${SLURM_JOB_ID}"
archive_dir="/apps/cvs_tests/artifacts_archive/${SLURM_JOB_ID}"
publish_dir="/apps/cvs_tests/test_reports"

mkdir -p "${archive_dir}" "${publish_dir}" 2>/dev/null || true

if [[ -d "${artifact_src}" ]]; then
  # 1) Archive everything for this job id
  cp -a "${artifact_src}/." "${archive_dir}/" 2>/dev/null || true

  # 2) Publish latest HTMLs with stable filenames
  # Support both timestamped and stable filenames.
  perf_html="$(ls -1t "${artifact_src}"/rccl_perf_report_*.html 2>/dev/null | head -n 1 || true)"
  [[ -z "${perf_html}" && -f "${artifact_src}/rccl_perf_report.html" ]] && perf_html="${artifact_src}/rccl_perf_report.html"

  heatmap_html="$(ls -1t "${artifact_src}"/rccl_heatmap_*.html 2>/dev/null | head -n 1 || true)"
  [[ -z "${heatmap_html}" && -f "${artifact_src}/rccl_heatmap.html" ]] && heatmap_html="${artifact_src}/rccl_heatmap.html"

  pytest_html="${artifact_src}/pytest_report.html"

  if [[ -n "${perf_html}" && -f "${perf_html}" ]]; then
    cp -f "${perf_html}" "${publish_dir}/rccl_perf_report.html" 2>/dev/null || true
  fi
  if [[ -n "${heatmap_html}" && -f "${heatmap_html}" ]]; then
    cp -f "${heatmap_html}" "${publish_dir}/rccl_heatmap.html" 2>/dev/null || true
  fi
  if [[ -f "${pytest_html}" ]]; then
    cp -f "${pytest_html}" "${publish_dir}/pytest_report.html" 2>/dev/null || true
  fi
else
  echo "[WARNING] Artifact source directory not found: ${artifact_src}" >&2
fi

chmod -R g+rwX "${archive_dir}" "${publish_dir}" 2>/dev/null || true

if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  echo "========================================" >&2
  echo "[DEBUG] run.sh completed with exit code: $exit_code" >&2
  echo "========================================" >&2
fi

exit $exit_code
