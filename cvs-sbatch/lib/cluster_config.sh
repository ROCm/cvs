#!/usr/bin/env bash

##############################################################
# Cluster configuration generation
# - Supports both SLURM and non-SLURM environments
# - SSH key detection
# - Node IP resolution
# - Generate cluster.json from SLURM allocation or manual config
##############################################################

# Non-SLURM configuration file (set via environment or use default)
readonly MANUAL_CLUSTER_CONFIG="${MANUAL_CLUSTER_CONFIG:-cluster_config.json}"

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
  
  # First check if it's already an IP address
  if [[ $node =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    log_debug "Node is already an IP address: $node"
    echo "$node"
    return 0
  fi

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

##############################################################
# Check if running under SLURM
##############################################################
is_slurm_environment() {
  # Check for SLURM-specific environment variables
  if [[ -n "${SLURM_JOB_ID:-}" ]] && [[ -n "${SLURM_NODELIST:-}" ]]; then
    return 0  # true - running under SLURM
  fi
  return 1  # false - not running under SLURM
}

##############################################################
# Generate cluster config from SLURM allocation
##############################################################
generate_cluster_config_slurm() {
  log_info "Generating cluster configuration from SLURM allocation..."

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

##############################################################
# Generate cluster config from manual configuration file
# Expected format:
# {
#   "username": "user",
#   "priv_key_file": "/home/user/.ssh/id_ed25519",
#   "head_node_dict": { "mgmt_ip": "node1" },
#   "node_dict": {
#     "node1": { "bmc_ip": "NA", "vpc_ip": "node1" },
#     "node2": { "bmc_ip": "NA", "vpc_ip": "node2" }
#   }
# }
##############################################################
generate_cluster_config_manual() {
  local config_file="${1:-$MANUAL_CLUSTER_CONFIG}"
  
  log_info "Generating cluster configuration from manual config: $config_file"

  # Check if manual config file exists
  if [[ ! -f "$config_file" ]]; then
    log_error "Manual cluster config file not found: $config_file"
    log_error "Please create a cluster configuration file or run under SLURM."
    log_error ""
    log_error "Expected format:"
    log_error '  {'
    log_error '    "username": "your-username",'
    log_error '    "priv_key_file": "/home/user/.ssh/id_ed25519",'
    log_error '    "head_node_dict": { "mgmt_ip": "head-node-hostname" },'
    log_error '    "node_dict": {'
    log_error '      "node1-hostname": { "bmc_ip": "NA", "vpc_ip": "node1-hostname" },'
    log_error '      "node2-hostname": { "bmc_ip": "NA", "vpc_ip": "node2-hostname" }'
    log_error '    }'
    log_error '  }'
    log_error ""
    log_error "Set MANUAL_CLUSTER_CONFIG environment variable to specify a different file."
    exit 1
  fi

  # Validate JSON
  if ! jq empty "$config_file" 2>/dev/null; then
    log_error "Invalid JSON in config file: $config_file"
    exit 1
  fi

  # Extract values from manual config
  local username priv_key head_node_mgmt
  username=$(jq -r '.username // empty' "$config_file")
  priv_key=$(jq -r '.priv_key_file // empty' "$config_file")
  head_node_mgmt=$(jq -r '.head_node_dict.mgmt_ip // empty' "$config_file")

  # Use defaults if not specified
  if [[ -z "$username" ]]; then
    username="${USER:-$(whoami)}"
    log_info "Username not specified in config, using current user: $username"
  fi

  if [[ -z "$priv_key" ]]; then
    priv_key="$(detect_ssh_key)"
    log_info "SSH key not specified in config, auto-detected: $priv_key"
  fi

  if [[ -z "$head_node_mgmt" ]]; then
    log_error "head_node_dict.mgmt_ip is required in config file"
    exit 1
  fi

  # Resolve head node IP (handle both hostname and IP)
  local head_node_ip
  if ! head_node_ip="$(get_node_ip "$head_node_mgmt")"; then
    log_error "Failed to resolve head node: $head_node_mgmt"
    exit 1
  fi
  log_info "Head node resolved: $head_node_mgmt -> $head_node_ip"

  # Extract node_dict and resolve hostnames to IPs
  local node_dict_raw node_keys
  node_dict_raw=$(jq '.node_dict // {}' "$config_file")
  
  # Get all node keys (hostnames)
  mapfile -t node_keys < <(echo "$node_dict_raw" | jq -r 'keys[]')

  if [[ ${#node_keys[@]} -eq 0 ]]; then
    log_error "No nodes found in node_dict"
    exit 1
  fi

  log_info "Found ${#node_keys[@]} nodes in config: ${node_keys[*]}"

  # Build node dictionary with resolved IPs
  local node_dict_resolved="{}"
  for node_key in "${node_keys[@]}"; do
    local vpc_ip node_ip
    
    # Get the vpc_ip from the config
    vpc_ip=$(echo "$node_dict_raw" | jq -r --arg key "$node_key" '.[$key].vpc_ip // $key')
    
    # Resolve to IP address
    if ! node_ip="$(get_node_ip "$vpc_ip")"; then
      log_warn "Failed to resolve node $node_key (vpc_ip: $vpc_ip), using hostname as-is"
      node_ip="$vpc_ip"
    fi

    log_info "  - Node: $node_key -> IP: $node_ip"

    # Add to resolved dict (using resolved IP as key)
    node_dict_resolved=$(echo "$node_dict_resolved" | jq \
      --arg ip "$node_ip" \
      '. + {($ip): {"bmc_ip": "NA", "vpc_ip": $ip}}')
  done

  # Generate cluster.json
  if ! jq -n --arg username "$username" \
    --arg key_file "$priv_key" \
    --arg head_ip "$head_node_ip" \
    --argjson nodes "$node_dict_resolved" \
    '{
    "_user_comment": "Generated from manual cluster configuration",
    "username": $username,
    "_key_comment": "SSH private key from config or auto-detected",
    "priv_key_file": $key_file,
    "_node_comment": "Nodes from manual configuration",
    "_vpc_comment": "VPC IP resolved from hostnames",
    "head_node_dict": {
      "mgmt_ip": $head_ip
    },
    "node_dict": $nodes
  }' >"$CLUSTER_JSON" 2>&1; then
    log_error "Failed to generate cluster.json"
    exit 1
  fi

  log_success "Generated $CLUSTER_JSON from manual config:"
  log_info "  - Username: $username"
  log_info "  - SSH Key: $priv_key"
  log_info "  - Head Node IP: $head_node_ip"
  log_info "  - Compute Nodes: ${#node_keys[@]} nodes"
}

##############################################################
# Main entry point - auto-detect environment and generate config
##############################################################
generate_cluster_config() {
  if is_slurm_environment; then
    log_info "SLURM environment detected"
    generate_cluster_config_slurm
  else
    log_info "Non-SLURM environment detected"
    generate_cluster_config_manual
  fi
}
