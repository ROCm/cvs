#!/usr/bin/env bash

##############################################################
# Git operations for CVS repository management
# - Clone repository
# - Update to latest branch
# - Idempotent operations
##############################################################

setup_cvs_repo() {
  log_info "Setting up CVS repository..."
  log_debug "CVS_DIR: $CVS_DIR"
  log_debug "CVS_REPO_URL: $CVS_REPO_URL"
  log_debug "CVS_BRANCH: $CVS_BRANCH"

  if [[ -d "$CVS_DIR/.git" ]]; then
    log_info "CVS exists, ensuring it's up to date on $CVS_BRANCH"
  else
    log_info "Cloning CVS from $CVS_REPO_URL..."
    if ! git clone "$CVS_REPO_URL" "$CVS_DIR"; then
      log_error "Failed to clone CVS repository"
      exit 1
    fi
  fi

  # Fetch latest (idempotent - always get latest refs)
  log_debug "Fetching from origin..."
  git -C "$CVS_DIR" fetch origin "$CVS_BRANCH" &>/dev/null || {
    log_error "Failed to fetch from origin"
    exit 1
  }

  # Checkout branch (idempotent - safe to run multiple times)
  log_debug "Checking out branch $CVS_BRANCH..."
  git -C "$CVS_DIR" checkout "$CVS_BRANCH" &>/dev/null || {
    log_error "Failed to checkout branch $CVS_BRANCH"
    exit 1
  }

  # Pull latest (idempotent - no-op if already up to date)
  log_debug "Pulling latest changes..."
  git -C "$CVS_DIR" pull origin "$CVS_BRANCH" &>/dev/null || {
    log_error "Failed to pull latest changes from $CVS_BRANCH"
    exit 1
  }

  log_success "CVS repository ready on branch $CVS_BRANCH"
}
