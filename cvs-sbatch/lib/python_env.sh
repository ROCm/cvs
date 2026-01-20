#!/usr/bin/env bash

##############################################################
# Python environment and test execution
# - UV-based execution
# - System pytest execution
# - Virtual environment creation
# - Automatic fallback mechanism
##############################################################
# HTML report configuration (can be overridden by environment)
# Default to per-job output directory so the report is collected as an artifact automatically.
ENABLE_HTML_REPORT="${ENABLE_HTML_REPORT:-true}"
REPORT_DIR="${REPORT_DIR:-/tmp/cvs_${USER:-user}_${SLURM_JOB_ID:-$$}}"
REPORT_NAME="${REPORT_NAME:-pytest_report.html}"

build_html_args() {
  HTML_ARGS=()

  if [[ "$ENABLE_HTML_REPORT" == "true" ]]; then
    mkdir -p "$REPORT_DIR"
    HTML_ARGS+=(
      --html="$REPORT_DIR/$REPORT_NAME"
      --self-contained-html
      --capture=tee-sys
    )
  fi
}


run_with_uv() {
  local test_path="$1"
  local cluster_file="$2"
  local config_file="$3"

  log_info "Using uv to run tests with system Python"
  log_info "Test path: $test_path"
  log_info "Log file: $LOG_FILE"

  # Use a per-job virtual environment in local /tmp to avoid shared-workdir `.venv`
  # collisions (and failures like "Stale file handle" when removing `.venv`).
  local job_id="${SLURM_JOB_ID:-$$}"
  local user_name="${USER:-user}"
  local uv_venv_dir="${UV_VENV_DIR:-/tmp/uv_venv_${user_name}_${job_id}}"

  log_info "Creating per-job uv venv: $uv_venv_dir"
  # --clear ensures we can reuse the same path across retries for the same SLURM job id
  uv venv --python python3 --clear "$uv_venv_dir"

  # shellcheck disable=SC1091
  source "$uv_venv_dir/bin/activate"

  build_html_args

  # Use --with-requirements to install dependencies from requirements.txt
  # This ensures pytest and all other test dependencies are available.
  # `--active` forces uv to use the activated venv above (instead of project `.venv`).
  local uv_args=("--active")

  if [[ -f "requirements.txt" ]]; then
    log_info "Installing dependencies from requirements.txt"
    uv_args+=("--with-requirements" "requirements.txt")
  else
    # Fallback: at minimum we need pytest
    log_info "No requirements.txt found, installing pytest only"
    uv_args+=("--with" "pytest")
  fi

  uv run "${uv_args[@]}" python -m pytest -vv --log-file="$LOG_FILE" -s --tb=short \
    "${HTML_ARGS[@]}" \
    "$test_path" \
    --cluster_file "$cluster_file" \
    --config_file "$config_file"

  deactivate || true
  # Best-effort cleanup; don't fail the job if /tmp has filesystem quirks.
  rm -rf "$uv_venv_dir" 2>/dev/null || true
}

run_with_system_pytest() {
  local test_path="$1"
  local cluster_file="$2"
  local config_file="$3"

  log_info "Using system pytest"
  log_info "Test path: $test_path"
  log_info "Log file: $LOG_FILE"

  build_html_args

  pytest -vv --log-file="$LOG_FILE" -s --tb=short \
    "${HTML_ARGS[@]}" \
    "$test_path" \
    --cluster_file "$cluster_file" \
    --config_file "$config_file"
}

run_with_venv() {
  local test_path="$1"
  local cluster_file="$2"
  local config_file="$3"

  log_info "Setting up Python virtual environment"

  # Always recreate venv for idempotency (prevents corrupted state from cancelled runs)
  if [[ -d ".venv" ]]; then
    log_info "Removing existing .venv to ensure clean state"
    rm -rf .venv
  fi

  log_info "Creating fresh virtual environment"
  python3 -m venv .venv

  # shellcheck disable=SC1091
  source .venv/bin/activate

  # Upgrade pip quietly
  pip install --quiet --upgrade pip

  # Install dependencies only if requirements.txt exists
  if [[ -f "requirements.txt" ]]; then
    log_info "Installing from requirements.txt"
    pip install --quiet -r requirements.txt
  elif [[ -f "pyproject.toml" ]]; then
    log_info "Found pyproject.toml but skipping editable install"
    # Install dependencies only
    pip install --quiet pytest
  fi

  # Ensure pytest is installed
  log_info "Installing pytest"
  pip install --quiet pytest

  log_info "Running tests..."
  log_info "Test path: $test_path"
  log_info "Log file: $LOG_FILE"

  build_html_args

  pytest -vv --log-file="$LOG_FILE" -s --tb=short \
    "${HTML_ARGS[@]}" \
    "$test_path" \
    --cluster_file "$cluster_file" \
    --config_file "$config_file"

  deactivate

  # Clean up venv after successful run
  log_info "Cleaning up virtual environment"
  rm -rf .venv
}

run_tests() {
  local test_path="$1"
  
  log_info "Preparing to run tests..."
  log_debug "Test path: $test_path"
  log_debug "CVS directory: $CVS_DIR"

  # Store absolute paths
  local cluster_json_abs
  local config_json_abs
  cluster_json_abs="$(realpath "$CLUSTER_JSON")"
  config_json_abs="$(realpath "$CONFIG_JSON")"

  log_debug "Cluster config (abs): $cluster_json_abs"
  log_debug "Benchmark config (abs): $config_json_abs"

  # Verify files exist
  if [[ ! -f $cluster_json_abs ]]; then
    log_error "$CLUSTER_JSON not found!"
    exit 1
  fi
  if [[ ! -f $config_json_abs ]]; then
    log_error "$CONFIG_JSON not found!"
    exit 1
  fi

  # Change to CVS directory
  log_debug "Changing to CVS directory: $CVS_DIR"
  pushd "$CVS_DIR" >/dev/null || exit 1

  # Run tests with available Python environment
  log_debug "Detecting Python environment..."
  if command -v uv &>/dev/null; then
    log_debug "Found: uv"
    run_with_uv "$test_path" "$cluster_json_abs" "$config_json_abs"
  elif command -v pytest &>/dev/null; then
    log_debug "Found: pytest"
    run_with_system_pytest "$test_path" "$cluster_json_abs" "$config_json_abs"
  elif python3 -m venv --help &>/dev/null; then
    log_debug "Found: python3 venv"
    run_with_venv "$test_path" "$cluster_json_abs" "$config_json_abs"
  else
    log_error "No Python package manager found (tried uv, pytest, python3 venv)"
    popd >/dev/null || exit 1
    exit 1
  fi

  popd >/dev/null || exit 1

  log_success "All tests completed!"
}
