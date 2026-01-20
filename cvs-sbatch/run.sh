#!/usr/bin/env bash

##############################################################
# CVS Benchmark Automation - Main Orchestrator
# Author: surya.periaswamy@amd.com
# Date: 09/11/2025
#
# Description: Thin orchestrator that coordinates:
#   - CVS repository management
#   - Cluster configuration generation
#   - Test execution with Python
#
# Usage:
#   Direct: ./run.sh (for local testing)
#   SBATCH: sbatch sbatch/default.sbatch
##############################################################

set -euo pipefail

# Get script directory for reliable path resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source library modules
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/git_ops.sh"
source "${SCRIPT_DIR}/lib/cluster_config.sh"
source "${SCRIPT_DIR}/lib/python_env.sh"

##############################################################
# CONFIGURATION
##############################################################

# CVS Repository settings
readonly CVS_REPO_URL="${CVS_REPO_URL:-https://github.com/ROCm/cvs.git}"
readonly CVS_DIR="${CVS_DIR:-/apps/cvs_tests/cvs}"
readonly CVS_BRANCH="${CVS_BRANCH:-main}"

# Test settings
# TEST_PATH can be overridden for different benchmarks (RCCL, Aorta, etc.)
readonly TEST_PATH="${TEST_PATH:-./cvs/tests/rccl/rccl_heatmap_cvs.py}"
readonly LOG_FILE="${LOG_FILE:-/tmp/cvs_${USER}_${SLURM_JOB_ID:-$$}.log}"

# Configuration files
# CONFIG_FILE supports both .json and .yaml formats
readonly CLUSTER_JSON="cluster.json"
readonly CONFIG_JSON="${CONFIG_FILE:-config.json}"

# SSH key locations to check (in order of preference)
readonly SSH_KEY_PATHS=(
  "$HOME/.ssh/id_ed25519"
  "$HOME/.ssh/id_rsa"
  "$HOME/.ssh/id_ecdsa"
)

##############################################################
# MAIN EXECUTION
##############################################################

main() {
  log_info "========================================"
  log_info "CVS Benchmark Automation"
  log_info "========================================"
  log_info "Job ID: ${SLURM_JOB_ID:-N/A}"
  log_info "Nodes: ${SLURM_NNODES:-N/A}"
  log_info "Working directory: $(pwd)"
  log_info "CVS directory: $CVS_DIR"
  log_info "CVS branch: $CVS_BRANCH"
  log_info "Test path: $TEST_PATH"
  log_info "Config file: $CONFIG_JSON"
  log_info "========================================"

  # Setup error handling and cleanup
  setup_traps

  # Execute workflow
  #setup_cvs_repo
  generate_cluster_config
  run_tests "$TEST_PATH"

  log_info "========================================"
  log_success "CVS benchmark automation completed successfully!"
  log_info "Cleanup will happen on exit..."
  log_info "========================================"
}

# Only run main if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
