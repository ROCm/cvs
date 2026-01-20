#!/usr/bin/env bash

##############################################################
# SBATCH configuration for Aorta distributed training benchmark
# Author: surya.periaswamy@amd.com
# Date: 01/20/2026
#
# Submit with: sbatch /apps/cvs_tests/cvs-sbatch/sbatch/aorta.sbatch
# Debug mode:  DEBUG_MODE=1 sbatch sbatch/aorta.sbatch
#
# Requirements:
#   - Docker daemon running on compute nodes
#   - User in 'docker' group on compute nodes
#   - Shared filesystem accessible from all nodes (for aorta_path)
#   - Docker image pullable or pre-cached on nodes
##############################################################

#SBATCH --job-name=Aorta-Distributed-Training
#SBATCH --output=/apps/cvs_tests/logs/%x-%j.out
#SBATCH --error=/apps/cvs_tests/logs/%x-%j.err
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=17nodexgmi38
#SBATCH --reservation=rccl
#SBATCH --exclusive
#SBATCH --chdir=/apps/cvs_tests/cvs-sbatch

# Enable strict error handling
set -euo pipefail

##############################################################
# Aorta-specific Configuration
##############################################################
# Override default test path and config for Aorta benchmark
export TEST_PATH="./cvs/tests/benchmark/test_aorta.py"
export CONFIG_FILE="config_aorta.yaml"

# Optional: Set Aorta-specific environment variables
# export AORTA_PATH="/shared/path/to/aorta"  # Override aorta_path in config
# export SKIP_RCCL_BUILD=true  # Skip RCCL build if already built

##############################################################
# Standard SBATCH Execution (shared with RCCL tests)
##############################################################

# Enable debug mode if requested
if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  set -x
  echo "========================================" >&2
  echo "[DEBUG] Aorta SBATCH script starting" >&2
  echo "[DEBUG] Job ID: ${SLURM_JOB_ID}" >&2
  echo "[DEBUG] Submit directory: ${SLURM_SUBMIT_DIR}" >&2
  echo "[DEBUG] Current directory: $(pwd)" >&2
  echo "[DEBUG] Hostname: $(hostname)" >&2
  echo "[DEBUG] TEST_PATH: ${TEST_PATH}" >&2
  echo "[DEBUG] CONFIG_FILE: ${CONFIG_FILE}" >&2
  echo "========================================" >&2
fi

# Run from the shared cvs-sbatch root directory
RUN_DIR="${CVS_SBATCH_DIR:-/apps/cvs_tests/cvs-sbatch}"
if [[ ! -d "${RUN_DIR}" ]]; then
  RUN_DIR="${SLURM_SUBMIT_DIR:-${RUN_DIR}}"
fi

if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  echo "[DEBUG] Run dir: ${RUN_DIR}" >&2
fi

cd "${RUN_DIR}" || {
  echo "[ERROR] Failed to cd to ${RUN_DIR}" >&2
  exit 1
}

# Verify config file exists
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] Aorta config file not found: ${CONFIG_FILE}" >&2
  echo "[ERROR] Please create ${RUN_DIR}/${CONFIG_FILE}" >&2
  echo "[ERROR] Example: cvs copy-config aorta/aorta_benchmark.yaml ${RUN_DIR}/${CONFIG_FILE}" >&2
  exit 1
fi

# Verify run.sh exists and is executable
if [[ ! -f "run.sh" ]]; then
  echo "[ERROR] run.sh not found in $(pwd)" >&2
  exit 1
fi

if [[ ! -x "run.sh" ]]; then
  chmod +x run.sh
fi

if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  echo "[DEBUG] Executing run.sh for Aorta benchmark..." >&2
fi

# Execute run.sh (will pick up TEST_PATH and CONFIG_FILE from environment)
./run.sh
exit_code=$?

##############################################################
# Collect Aorta-specific artifacts
##############################################################
artifact_src="/tmp/aorta_${USER}_${SLURM_JOB_ID}"
archive_dir="/apps/cvs_tests/artifacts_archive/${SLURM_JOB_ID}"
publish_dir="/apps/cvs_tests/test_reports"

mkdir -p "${archive_dir}" "${publish_dir}" 2>/dev/null || true

# Aorta outputs are typically in the aorta_path directory
# Copy TraceLens and GEMM analysis results if they exist
if [[ -d "${artifact_src}" ]]; then
  cp -a "${artifact_src}/." "${archive_dir}/" 2>/dev/null || true
fi

# Also look for tracelens_analysis in aorta output directory
# (The specific path depends on aorta_path in config)
aorta_output=$(find /tmp -maxdepth 2 -type d -name "nodes*_rccl_*" -user "${USER}" 2>/dev/null | head -1 || true)
if [[ -n "${aorta_output}" && -d "${aorta_output}" ]]; then
  echo "[INFO] Found Aorta output: ${aorta_output}" >&2
  
  # Copy tracelens analysis if exists
  if [[ -d "${aorta_output}/tracelens_analysis" ]]; then
    cp -a "${aorta_output}/tracelens_analysis" "${archive_dir}/" 2>/dev/null || true
  fi
  
  # Copy torch_profiler traces
  if [[ -d "${aorta_output}/torch_profiler" ]]; then
    cp -a "${aorta_output}/torch_profiler" "${archive_dir}/" 2>/dev/null || true
  fi
fi

# Copy pytest report
pytest_html="/tmp/cvs_aorta_${USER}_${SLURM_JOB_ID}/pytest_report.html"
if [[ -f "${pytest_html}" ]]; then
  cp -f "${pytest_html}" "${publish_dir}/aorta_pytest_report.html" 2>/dev/null || true
fi

chmod -R g+rwX "${archive_dir}" "${publish_dir}" 2>/dev/null || true

if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  echo "========================================" >&2
  echo "[DEBUG] run.sh completed with exit code: $exit_code" >&2
  echo "[DEBUG] Artifacts archived to: ${archive_dir}" >&2
  echo "========================================" >&2
fi

exit $exit_code
