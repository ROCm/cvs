#!/usr/bin/env bash

##############################################################
# Test SBATCH configuration for Aorta benchmark
# Uses local paths for testing
##############################################################

#SBATCH --job-name=Aorta-Test
#SBATCH --output=/scratch/users/speriasw/tmp/slurm/%x-%j.out
#SBATCH --error=/scratch/users/speriasw/tmp/slurm/%x-%j.err
#SBATCH --reservation=rccl_92
#SBATCH --account=rccl
#SBATCH --partition=rccl
#SBATCH --nodes=1
#SBATCH --exclusive

set -euo pipefail

##############################################################
# Configuration
##############################################################
export TEST_PATH="./cvs/tests/benchmark/test_aorta.py"
export CONFIG_FILE="config_aorta_test.yaml"
export DEBUG_MODE=1

# Use our local CVS checkout
export CVS_DIR="/home/AMD/speriasw/aorta-cvs-integration"
export CVS_SBATCH_DIR="/home/AMD/speriasw/aorta-cvs-integration/cvs-sbatch"

##############################################################
# Execution
##############################################################
echo "========================================" 
echo "[DEBUG] Aorta Test SBATCH starting"
echo "[DEBUG] Job ID: ${SLURM_JOB_ID}"
echo "[DEBUG] Node: $(hostname)"
echo "[DEBUG] TEST_PATH: ${TEST_PATH}"
echo "[DEBUG] CONFIG_FILE: ${CONFIG_FILE}"
echo "[DEBUG] CVS_DIR: ${CVS_DIR}"
echo "========================================"

# Create log directory if it doesn't exist
mkdir -p /home/AMD/speriasw/logs

# Change to cvs-sbatch directory
cd "${CVS_SBATCH_DIR}" || exit 1

# Verify files exist
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_FILE}"
  exit 1
fi

if [[ ! -f "run.sh" ]]; then
  echo "[ERROR] run.sh not found"
  exit 1
fi

chmod +x run.sh

echo "[DEBUG] Executing run.sh..."
./run.sh
exit_code=$?

echo "========================================"
echo "[DEBUG] Completed with exit code: $exit_code"
echo "========================================"

exit $exit_code
