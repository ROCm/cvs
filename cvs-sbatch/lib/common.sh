#!/usr/bin/env bash

##############################################################
# Common utility functions
# - Logging
# - Cleanup
# - Error handling
##############################################################

# Log level control
# Set DEBUG_MODE=1 to enable debug logging
readonly DEBUG_MODE="${DEBUG_MODE:-0}"

log_debug() {
  if [[ "${DEBUG_MODE}" == "1" ]]; then
    echo "[DEBUG] $*" >&2
  fi
}

log_info() {
  echo "[INFO] $*" >&2
}

log_error() {
  echo "[ERROR] $*" >&2
}

log_success() {
  echo "[SUCCESS] $*" >&2
}

cleanup_on_error() {
  log_error "Script failed on line $1"
  popd &>/dev/null || true
  cleanup_files
  exit 1
}

cleanup_files() {
  log_info "Cleaning up files..."

  # Clean up generated cluster.json
  if [[ -f "$CLUSTER_JSON" ]]; then
    rm -f "$CLUSTER_JSON"
  fi

  # Clean up any lingering venv in CVS directory (from cancelled runs)
  if [[ -d "$CVS_DIR/.venv" ]]; then
    log_info "Cleaning up lingering .venv"
    rm -rf "$CVS_DIR/.venv"
  fi
}

setup_traps() {
  # Trap for errors
  trap 'cleanup_on_error $LINENO' ERR

  # Trap for normal exit
  trap 'cleanup_files' EXIT
}
