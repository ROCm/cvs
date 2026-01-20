#!/usr/bin/env bash

##############################################################
# Cluster configuration generation
# - SSH key detection
# - Node IP resolution
# - Generate cluster.json from SLURM allocation
##############################################################

detect_ssh_key() {
  local key_file=""

  for key_path in "${SSH_KEY_PATHS[@]}"; do
    if [[ -f $key_path ]]; then
      key_file="$key_path"
      log_info "Detected SSH key: $key_file"
      break
    fi
  done

  # Fallback to default
  if [[ -z $key_file ]]; then
    log_info "No SSH key found, using default: $HOME/.ssh/id_rsa"
    key_file="$HOME/.ssh/id_rsa"
  fi

  echo "$key_file"
}

get_node_ip() {
  local node="$1"
  local ip

  log_debug "Resolving hostname: $node"
  
  # Try to resolve the hostname
  if ! ip=$(host "$node" 2>/dev/null | awk '/has address/ {print $4}'); then
    log_error "Failed to resolve hostname: $node"
    return 1
  fi

  # Validate that we got an IP address
  if [[ -z $ip ]] || [[ ! $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    log_error "Invalid or empty IP address for node: $node"
    return 1
  fi

  log_debug "Resolved $node -> $ip"
  echo "$ip"
}

build_node_dict_json() {
  local nodes=("$@")
  local node_dict="{}"

  log_info "Building node dictionary for ${#nodes[@]} compute nodes..."

  for node in "${nodes[@]}"; do
    local node_ip
    if ! node_ip="$(get_node_ip "$node")"; then
      log_error "Skipping node $node due to resolution failure"
      continue
    fi

    log_info "  - Node: $node -> IP: $node_ip"

    # Build JSON incrementally with error checking
    if ! node_dict=$(echo "$node_dict" | jq --arg ip "$node_ip" \
      '. + {($ip): {"bmc_ip": "NA", "vpc_ip": $ip}}' 2>&1); then
      log_error "Failed to add node $node_ip to JSON"
      return 1
    fi
  done

  # Validate final JSON
  if ! echo "$node_dict" | jq empty 2>/dev/null; then
    log_error "Generated invalid JSON for node dictionary"
    return 1
  fi

  echo "$node_dict"
}

generate_cluster_config() {
  log_info "Generating cluster configuration..."

  # Get current user
  local current_user="${USER:-$(whoami)}"

  # Detect SSH key
  local priv_key
  priv_key="$(detect_ssh_key)"

  # Get login/management node IP
  local login_node_ip
  if ! login_node_ip="$(get_node_ip "$SLURM_SUBMIT_HOST")"; then
    log_error "Failed to resolve login node: $SLURM_SUBMIT_HOST"
    exit 1
  fi
  log_info "Login node resolved: $SLURM_SUBMIT_HOST -> $login_node_ip"

  # Build node dictionary
  local nodes
  mapfile -t nodes < <(scontrol show hostnames "$SLURM_NODELIST")

  log_info "SLURM_NODELIST: $SLURM_NODELIST"
  log_info "Parsed nodes: ${nodes[*]}"

  if [[ ${#nodes[@]} -eq 0 ]]; then
    log_error "No compute nodes found in SLURM_NODELIST"
    exit 1
  fi

  local node_dict_json
  if ! node_dict_json="$(build_node_dict_json "${nodes[@]}")"; then
    log_error "Failed to build node dictionary"
    exit 1
  fi

  log_info "Node dictionary JSON: $node_dict_json"

  # Generate cluster.json with error checking
  if ! jq -n --arg username "$current_user" \
    --arg key_file "$priv_key" \
    --arg head_ip "$login_node_ip" \
    --argjson nodes "$node_dict_json" \
    '{
    "_user_comment": "Dynamically generated from SLURM job allocation",
    "username": $username,
    "_key_comment": "Auto-detected SSH private key",
    "priv_key_file": $key_file,
    "_node_comment": "Generated from SLURM_NODELIST",
    "_vpc_comment": "VPC IP set to same as node IP for this cluster",
    "head_node_dict": {
      "mgmt_ip": $head_ip
    },
    "node_dict": $nodes
  }' >"$CLUSTER_JSON" 2>&1; then
    log_error "Failed to generate cluster.json"
    exit 1
  fi

  log_success "Generated $CLUSTER_JSON:"
  log_info "  - Username: $current_user"
  log_info "  - SSH Key: $priv_key"
  log_info "  - Head/Login Node IP: $login_node_ip"
  log_info "  - Compute Nodes: ${#nodes[@]} nodes (${nodes[*]})"
}
