#!/usr/bin/env bash

##############################################################
# CVS Benchmark - Direct Execution (Non-SLURM)
# 
# Use this script when running benchmarks outside of SLURM.
# It requires a manual cluster configuration file.
#
# Usage:
#   ./run_direct.sh                              # Uses default cluster_config.json
#   MANUAL_CLUSTER_CONFIG=my_cluster.json ./run_direct.sh
#   ./run_direct.sh --cluster my_cluster.json   # Alternative syntax
#
# Environment Variables:
#   MANUAL_CLUSTER_CONFIG - Path to cluster configuration JSON
#   CONFIG_FILE           - Benchmark configuration (JSON or YAML)
#   TEST_PATH             - Test file to run
#   CVS_DIR               - Path to CVS repository
#
# Example:
#   # Run Aorta benchmark on custom cluster
#   MANUAL_CLUSTER_CONFIG=examples/cluster_config_chi.json \
#   CONFIG_FILE=config_aorta.yaml \
#   TEST_PATH=./cvs/tests/benchmark/test_aorta.py \
#   ./run_direct.sh
##############################################################

set -euo pipefail

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --cluster)
      export MANUAL_CLUSTER_CONFIG="$2"
      shift 2
      ;;
    --config)
      export CONFIG_FILE="$2"
      shift 2
      ;;
    --test)
      export TEST_PATH="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --cluster FILE   Path to cluster configuration JSON"
      echo "  --config FILE    Path to benchmark configuration (JSON/YAML)"
      echo "  --test PATH      Test file to run"
      echo "  --help           Show this help message"
      echo ""
      echo "Environment Variables:"
      echo "  MANUAL_CLUSTER_CONFIG  Same as --cluster"
      echo "  CONFIG_FILE            Same as --config"
      echo "  TEST_PATH              Same as --test"
      echo "  CVS_DIR                Path to CVS repository"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Verify cluster config exists
CLUSTER_CONFIG="${MANUAL_CLUSTER_CONFIG:-cluster_config.json}"
if [[ ! -f "$CLUSTER_CONFIG" ]]; then
  echo "Error: Cluster configuration not found: $CLUSTER_CONFIG"
  echo ""
  echo "For non-SLURM environments, you must provide a cluster configuration file."
  echo ""
  echo "Quick start:"
  echo "  1. Copy the template:"
  echo "     cp cluster_config.json.template cluster_config.json"
  echo ""
  echo "  2. Edit cluster_config.json with your node information"
  echo ""
  echo "  3. Run again:"
  echo "     ./run_direct.sh"
  echo ""
  echo "Or use an example config:"
  echo "  MANUAL_CLUSTER_CONFIG=examples/cluster_config_chi.json ./run_direct.sh"
  exit 1
fi

echo "=========================================="
echo "CVS Benchmark - Direct Execution Mode"
echo "=========================================="
echo "Cluster config: $CLUSTER_CONFIG"
echo ""

# Export the cluster config path and run main script
export MANUAL_CLUSTER_CONFIG="$CLUSTER_CONFIG"

# Run the main orchestrator
exec ./run.sh "$@"
