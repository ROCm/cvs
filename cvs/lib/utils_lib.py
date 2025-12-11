'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
import os
import sys
import json

import pytest
from cvs.lib import globals

log = globals.log


def fail_test(msg):
    """
    Record and report a test failure without immediately raising an exception.
    This will enable Pytest to continue running further steps in the test case
    without returning back on first failure.

    Parameters:
      msg (str): Human-readable failure description to log and store.

    Behavior:
      - Prints a standardized "FAIL - ..." message to stdout for quick visibility.
      - Logs the same message at error level via the global `log` logger.
      - Appends the raw message to `globals.error_list` for later aggregation/reporting.
      - Does NOT raise/abort by default. The pytest failure line is present but commented out.

    Notes:
      - If you want to stop execution on first failure (e.g., in pytest),
        uncomment the pytest.fail(...) line.
      - Assumes:
          * `log` is a configured logger available in module scope.
          * `globals.error_list` is an initialized list collecting error messages.
    """
    print('FAIL - {}'.format(msg))
    log.error('FAIL - {}'.format(msg))
    # We append the error messages to the global error list which will be checked at the
    # end of every test case in Pytest to determine test PASS/FAIL
    globals.error_list.append(msg)


def update_test_result():
    # For every Pytest test case, we will initialize the global.error_list to empty and
    # whenever fail_test is executed, it would add the error messages and if this is
    # a non-zero list, then the test case will be marked as a failure in the end of the
    # test case
    if len(globals.error_list) > 0:
        pytest.fail('Following FAILURES seen - {}'.format(globals.error_list))


def print_test_output(log, out_dict):
    print('#========================================================#')
    print('\t\t ** Test Output **')
    print('#========================================================#')
    for node in out_dict.keys():
        print(f'==== {node} ====')
        print(out_dict[node])


def scan_test_results(out_dict):
    """
    Scan test outputs from multiple hosts for failure indicators and report context.

    Parameters:
      out_dict (dict): Mapping of host -> output string (e.g., test logs or stdout/stderr).

    Behavior:
      - For each host, searches the output for common failure patterns:
          'test FAIL', 'test ERROR', 'ABORT', 'Traceback', 'No such file', 'FATAL' (case-insensitive).
      - When a match is found, splits the full output into words and locates the first token
        containing any of: 'FAIL', 'ERR', 'ABORT', 'Traceback'.
      - Prints up to 5 words before and after the matched token to provide context.
      - Calls fail_test with a message that includes the before/actual/after tokens.

    Notes:
      - Assumes `re` is imported, and `fail_test` is available in scope.
      - word_count controls the number of contextual words (currently set to 5).
      - Potential issue: `fail_test` is called unconditionally at the end of the loop iteration,
        which will raise even when no match is found, and before_words/actual_word/after_words
        might be undefined if no target word was located. Consider moving the fail_test call
        inside the `if target_word_index != -1:` block and guarding for no match cases.
    """

    word_count = 5  # Number of words to include before/after the matched token for context

    # Iterate over each host's output
    for host in out_dict.keys():
        # Search for any high-level failure pattern in the raw output
        match = re.search('test FAIL |test ERROR |ABORT|Traceback|No such file|FATAL', out_dict[host], re.I)
        if match:
            # Record the span of the first match (currently unused; could help with slicing)
            # start_index = match.start()
            # end_index = match.end()
            # Tokenize the entire output into words for contextual extraction
            words = out_dict[host].split()
            # Find the index of the target word in the list of words
            target_word_index = -1
            for i, word in enumerate(words):
                # Check if this token contains any of the target substrings
                if re.search('FAIL|ERR|ABORT|Traceback', word, re.I):  # Check if the pattern is part of the word
                    target_word_index = i
                    actual_word = word
                    break

            if target_word_index != -1:
                # Collect up to 'word_count' words immediately before the matched token
                # Get the words before the match
                before_words = words[max(0, target_word_index - word_count) : target_word_index]
                # Get the words after the match
                after_words = words[target_word_index + 1 : min(len(words), target_word_index + word_count + 1)]
                print(f"Words before: {before_words}")
                print(f"Words after: {after_words}")
                print(f'Test failed in scan_result on node {host} due to pattern ')
            fail_test(
                f'Test failed in scan_result on node {host} due to pattern {before_words} {actual_word} {after_words}'
            )


def json_to_dict(json_string):
    print('^^^^^^^^^')
    print(json_string)
    return json.loads(json_string)


def convert_phdl_json_to_dict(dict_json):
    """
    Convert a per-node JSON payload map into a per-node Python dict map.

    Args:
      dict_json (dict): Mapping of node identifier -> JSON string (or JSON-like object)
                        that can be parsed by json_to_dict.

    Returns:
      dict: Mapping of node identifier -> parsed Python dictionary.
            If parsing fails for a node, its value will be an empty dict and a failure
            will be recorded via fail_test.

    Behavior:
      - Iterates over each node key in dict_json.
      - Attempts to parse the node's JSON payload using json_to_dict.
      - On parse error, prints/logs a failure message, calls fail_test, and assigns {} for that node.

    Assumptions:
      - json_to_dict is available in scope and raises an Exception on invalid JSON.
      - fail_test is available in scope for reporting failures.
      - dict_json keys are node identifiers; values are JSON strings or compatible objects.
    """

    out_dict = {}  # Will hold node -> parsed dict

    # Process each node's JSON blob and convert to a Python dict
    for node in dict_json.keys():
        try:
            # Parse the JSON content for this node
            out_dict[node] = json_to_dict(dict_json[node])
        except Exception:
            print(f'ERROR converting Json output to dict for node {node}')
            fail_test(f'ERROR converting Json output to dict for node {node}')
            out_dict[node] = {}
    return out_dict


def get_model_from_rocm_smi_output(smi_output):
    """
    Infer the GPU model identifier from a rocm-smi output snippet.

    Args:
      smi_output (str): Text output captured from `rocm-smi -a` or similar.

    Returns:
      str: A normalized model string, one of:
           - 'mi300x' (default if no known model is matched)
           - 'mi325'
           - 'mi350'
           - 'mi355'

    Behavior:
      - Performs case-insensitive regex searches for specific model tokens in the provided
        rocm-smi output (e.g., "MI300X", "MI325", "MI350", "MI355").
      - Returns the corresponding normalized lowercase token (e.g., 'mi300x').
      - Falls back to 'mi300x' if no pattern matches (conservative default).

    """
    if re.search('MI300X', smi_output, re.I):
        model = 'mi300x'
    elif re.search('MI325', smi_output, re.I):
        model = 'mi325'
    elif re.search('MI350', smi_output, re.I):
        model = 'mi350'
    elif re.search('MI355', smi_output, re.I):
        model = 'mi355'
    else:
        model = 'mi300x'
    return model


def convert_hms_to_secs(time_string):
    """
    Convert a time string in 'HH:MM:SS' format to total seconds.

    Args:
      time_string (str): Time formatted as 'hours:minutes:seconds'.
                         - Hours and minutes should be integers.
                         - Seconds may be integer or float (e.g., '12.5').

    Returns:
      float | None:
        - Total seconds as a float (to support fractional seconds) when parsing succeeds.
        - None if the input format is invalid or contains non-numeric components.

    Behavior:
      - Splits the input on ':' and expects exactly 3 parts.
      - Parses hours and minutes as integers, seconds as float.
      - Computes total_seconds = hours*3600 + minutes*60 + seconds.
      - Prints a user-friendly error message and returns None on format/parse errors.

    """
    try:
        # Break the time string into components by ':'
        parts = time_string.split(':')

        # Expect exactly three components: HH:MM:SS
        if len(parts) == 3:
            # Convert hours and minutes to int; seconds to float to allow decimal seconds
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])  # Use float to handle potential decimal seconds
            # Compute total seconds, preserving fractional seconds
            total_seconds = (hours * 3600) + (minutes * 60) + seconds
            return total_seconds
        else:
            print("Invalid time format. Please use 'hr:min:sec'.")
            return None
    except ValueError:
        print("Invalid time format. Please ensure hours, minutes, and seconds are numeric.")
        return None


def _resolve_placeholders_in_dict(target_dict, replacements, context_name=""):
    """
    Internal helper function to recursively resolve placeholders in a dictionary.
    Also checks for unresolved manual placeholders like <changeme> and exits immediately.

    Args:
      target_dict: Dictionary (or nested dict/list/str) where placeholders should be replaced
      replacements: Dictionary mapping placeholder strings to their replacement values
                   Example: {'{user-id}': 'master', '{home}': '/home/master'}
      context_name: Optional context name for logging (e.g., "cluster", "config")

    Returns:
      dict/list/str: Input structure with all placeholders replaced

    Raises:
      SystemExit: If unresolved <changeme> or similar patterns are found

    Example:
      target = {"path": "/home/{user-id}/files", "user": "{user-id}"}
      replacements = {"{user-id}": "john"}
      result = _resolve_placeholders_in_dict(target, replacements)
      # Returns: {"path": "/home/john/files", "user": "john"}
    """

    def replace_in_string(value, path=""):
        """Replace all placeholders in a string and check for unresolved patterns."""
        if not isinstance(value, str):
            return value

        # Check for unresolved <changeme> pattern BEFORE replacement
        has_changeme = '<changeme>' in value.lower() or '<changeme>' in path.lower()
        if has_changeme:
            error_msg = f"\n{'=' * 70}\n"
            error_msg += f"ERROR: Unresolved placeholder found in {context_name}!\n"
            error_msg += f"{'=' * 70}\n\n"
            error_msg += f"  {path}: {value}\n\n"
            error_msg += f"{'=' * 70}\n"
            error_msg += "ACTION REQUIRED:\n"
            error_msg += "Please edit your configuration file and replace all the '<changeme>' placeholders\n"
            error_msg += "with an appropriate value before running the tests.\n"
            error_msg += f"{'=' * 70}\n"

            log.error(error_msg)
            # print(error_msg, file=sys.stderr)
            sys.exit(1)

        # Perform placeholder replacement
        result = value
        for placeholder, replacement in replacements.items():
            result = result.replace(placeholder, replacement)
        return result

    def replace_recursive(obj, path=""):
        """Recursively replace placeholders in nested structures."""
        if isinstance(obj, dict):
            resolved_dict = {}
            for k, v in obj.items():
                # Resolve placeholders in the key
                resolved_key = replace_in_string(k, f"{path}.{k}" if path else k)
                # Resolve placeholders in the value
                resolved_value = replace_recursive(v, f"{path}.{resolved_key}" if path else resolved_key)
                resolved_dict[resolved_key] = resolved_value
            return resolved_dict
        elif isinstance(obj, list):
            return [replace_recursive(item, f"{path}[{idx}]") for idx, item in enumerate(obj)]
        elif isinstance(obj, str):
            return replace_in_string(obj, path)
        else:
            return obj

    # Log the resolution operation
    if context_name:
        placeholder_summary = ', '.join([f'{k}={v}' for k, v in replacements.items()])
        log.info(f'Resolving {context_name} placeholders: {placeholder_summary}')

    resolved_dict = replace_recursive(target_dict)

    return resolved_dict


def resolve_cluster_config_placeholders(cluster_dict):
    """
    Resolve path placeholders in cluster configuration dictionary.
    This is called when loading cluster_dict before username is available from config.

    Supported placeholders:
      - {user-id}: Replaced with current system user from environment

    Args:
      cluster_dict: Cluster configuration dictionary (can be nested dict/list/str)

    Returns:
      dict: Cluster configuration with resolved path placeholders

    Example:
      Input username:  "{user-id}"
      Output username: "master" (from system environment)

      Input priv_key_file: "/home/{user-id}/.ssh/id_rsa"
      Output priv_key_file: "/home/master/.ssh/id_rsa"
    """
    # Get username from environment (fallback chain: USER -> LOGNAME -> USERNAME -> 'root')
    username = os.getenv('USER') or os.getenv('LOGNAME') or os.getenv('USERNAME') or 'root'

    log.info(f'Resolving cluster path placeholders with system username: {username}')

    # Define replacement mapping - only resolve {user-id} in cluster config
    replacements = {
        '{user-id}': username,
    }

    resolved_cluster = _resolve_placeholders_in_dict(cluster_dict, replacements, context_name="cluster config")

    return resolved_cluster


def resolve_test_config_placeholders(config_dict, cluster_dict):
    """
    Resolve path placeholders in test configuration dictionary.

    Supported placeholders:
      - {user-id} or {user}: Replaced with username from cluster_dict or current system user
      - {home}: Replaced with home directory of the user
      - {home-mount-dir}: Replaced with home mount directory name from cluster_dict
      - {node-dir-name}: Replaced with node directory name from cluster_dict

    Args:
      config_dict: Configuration dictionary (can be nested dict/list/str)
      cluster_dict: Cluster dictionary containing username and other cluster info

    Returns:
      dict/list/str: Configuration with resolved path placeholders

    Raises:
      SystemExit: If any <changeme> patterns are found

    Example:
      Input:  "/home/{user-id}/INSTALL"
      Output: "/home/master/INSTALL"

      Input:  "{home}/cvs/INSTALL"
      Output: "/home/master/cvs/INSTALL"

      Input:  "/{home-mount-dir}/{user-id}/cvs_cache"
      Output: "/home/master/cvs_cache"
    """
    # Get username from cluster config or fallback to environment
    username = cluster_dict.get('username', os.getenv('USER', 'root'))
    home_mount_dir_name = cluster_dict.get('home_mount_dir_name', 'home')
    node_dir_name = cluster_dict.get('node_dir_name', 'root')

    # Get home directory
    home_dir = os.path.expanduser(f'~{username}')

    log.info(
        f'Resolving config path placeholders with username: {username}, home: {home_dir}, home_mount_dir: {home_mount_dir_name}'
    )

    # Define replacement mapping - resolve all placeholders in config
    replacements = {
        '{user-id}': username,
        '{user}': username,
        '{home}': home_dir,
        '{home-mount-dir}': home_mount_dir_name,
        '{node-dir-name}': node_dir_name,
    }

    resolved_config = _resolve_placeholders_in_dict(config_dict, replacements, context_name="test config")

    return resolved_config


def collect_system_metadata(phdl, cluster_dict, config_dict, test_command=None, env_vars=None):
    """
    Collect comprehensive system metadata from compute nodes for test reporting.

    Args:
        phdl: Parallel SSH handle to execute commands on all nodes
        cluster_dict: Cluster configuration dictionary
        config_dict: Test configuration dictionary
        test_command: Optional test command string that was run
        env_vars: Optional list of environment variable names to capture

    Returns:
        dict: Metadata dictionary with system information
    """
    from datetime import datetime

    metadata = {}

    # Get first node for representative info
    node_list = list(cluster_dict['node_dict'].keys())
    head_node = node_list[0] if node_list else None

    # Capture date/time
    metadata['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata['timestamp'] = datetime.now().isoformat()

    # Capture hostname(s)
    try:
        hostname_dict = phdl.exec('hostname')
        metadata['hostnames'] = {node: hostname_dict[node].strip() for node in hostname_dict.keys()}
    except Exception as e:
        log.warning(f'Failed to get hostnames: {e}')

    # Capture ROCm version
    try:
        rocm_dict = phdl.exec(
            'cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm*/share/doc/rocm/version 2>/dev/null || echo "unknown"'
        )
        if head_node and head_node in rocm_dict:
            metadata['rocm_version'] = rocm_dict[head_node].strip()
    except Exception as e:
        log.warning(f'Failed to get ROCm version: {e}')

    # Capture OS info
    try:
        os_dict = phdl.exec('cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION)=" | head -2')
        if head_node and head_node in os_dict:
            os_info = os_dict[head_node].strip()
            metadata['os'] = os_info.replace('\n', ', ')
    except Exception as e:
        log.warning(f'Failed to get OS info: {e}')

    # Capture kernel version
    try:
        kernel_dict = phdl.exec('uname -r')
        if head_node and head_node in kernel_dict:
            metadata['kernel'] = kernel_dict[head_node].strip()
    except Exception as e:
        log.warning(f'Failed to get kernel version: {e}')

    # Capture GPU info
    try:
        gpu_dict = phdl.exec('rocm-smi --showproductname 2>/dev/null | grep "GPU" | head -1')
        if head_node and head_node in gpu_dict:
            gpu_line = gpu_dict[head_node].strip()
            # Extract just the GPU model name (everything after "Card Series:" or similar)
            # Format: "GPU[0]      : Card Series:       AMD Instinct MI300X"
            if ':' in gpu_line:
                # Split by colon and get the last part
                parts = gpu_line.split(':')
                if len(parts) >= 2:
                    # Get everything after the last colon and clean it up
                    gpu_model = parts[-1].strip()
                    if gpu_model:
                        metadata['gpu_model'] = gpu_model
                    else:
                        # Fallback: try second-to-last part
                        gpu_model = parts[-2].strip() if len(parts) >= 3 else gpu_line
                        metadata['gpu_model'] = gpu_model
                else:
                    metadata['gpu_model'] = gpu_line
            else:
                metadata['gpu_model'] = gpu_line

            # Also get GPU count
            gpu_count_dict = phdl.exec('rocm-smi --showproductname 2>/dev/null | grep "GPU\\[" | wc -l')
            if head_node and head_node in gpu_count_dict:
                try:
                    gpu_count = int(gpu_count_dict[head_node].strip())
                    if gpu_count > 0:
                        metadata['gpu_count'] = gpu_count
                except:
                    pass
    except Exception as e:
        log.warning(f'Failed to get GPU info: {e}')

    # Capture RDMA NIC info (InfiniBand/RoCE adapters used for RCCL)
    try:
        # Get list of IB devices
        ib_devices_dict = phdl.exec('ibv_devinfo -l 2>/dev/null')
        if head_node and head_node in ib_devices_dict:
            ib_devices = [dev.strip() for dev in ib_devices_dict[head_node].strip().split('\n') if dev.strip()]
            if ib_devices:
                metadata['rdma_devices'] = ib_devices

                # Get detailed info for each device
                nic_details = []
                for device in ib_devices[:8]:  # Limit to first 8 devices
                    try:
                        # Get device info including board_id (model), fw_ver, and node_guid
                        dev_info_dict = phdl.exec(
                            f'ibv_devinfo -d {device} 2>/dev/null | grep -E "board_id|fw_ver|node_guid|sys_image_guid" | head -4'
                        )
                        if head_node and head_node in dev_info_dict:
                            dev_info = dev_info_dict[head_node].strip()
                            if dev_info:
                                # Parse and clean the output
                                lines = [line.strip() for line in dev_info.split('\n') if line.strip()]
                                dev_dict = {'device': device}
                                for line in lines:
                                    if ':' in line:
                                        key, value = line.split(':', 1)
                                        dev_dict[key.strip()] = value.strip()
                                nic_details.append(dev_dict)
                    except Exception as e:
                        log.warning(f'Failed to get details for device {device}: {e}')

                if nic_details:
                    metadata['rdma_nic_details'] = nic_details
    except Exception as e:
        log.warning(f'Failed to get RDMA NIC info: {e}')

    # Get NIC model from lspci for RDMA devices (Mellanox, AMD Thor2, InfiniBand)
    try:
        # Expanded grep to catch Mellanox, Thor2/RDMA, and other RDMA devices
        nic_model_dict = phdl.exec(
            'lspci 2>/dev/null | grep -iE "mellanox|infiniband|network.*amd|rdma.*amd|thor" | grep -vE "usb|audio"'
        )
        if head_node and head_node in nic_model_dict:
            nic_models = nic_model_dict[head_node].strip()
            if nic_models:
                # Extract and categorize NIC models
                models = []
                mellanox_nics = []
                amd_nics = []
                other_nics = []

                for line in nic_models.split('\n'):
                    if line.strip():
                        # Extract device description (everything after the last colon)
                        if ':' in line:
                            parts = line.split(':', 2)
                            if len(parts) >= 3:
                                model = parts[2].strip()
                            elif len(parts) == 2:
                                model = parts[1].strip()
                            else:
                                continue

                            models.append(model)

                            # Categorize by vendor
                            model_lower = model.lower()
                            if 'mellanox' in model_lower or 'connectx' in model_lower:
                                mellanox_nics.append(model)
                            elif 'amd' in model_lower or 'thor' in model_lower or 'rdma' in model_lower:
                                amd_nics.append(model)
                            else:
                                other_nics.append(model)

                if models:
                    # Provide categorized view
                    nic_summary = {}
                    if mellanox_nics:
                        nic_summary['mellanox'] = mellanox_nics
                    if amd_nics:
                        nic_summary['amd'] = amd_nics
                    if other_nics:
                        nic_summary['other'] = other_nics

                    metadata['rdma_nic_models'] = nic_summary

                    # Also provide flat list for backward compatibility
                    metadata['rdma_nic_models_list'] = models
    except Exception as e:
        log.warning(f'Failed to get NIC models from lspci: {e}')

    # Get RDMA driver information
    try:
        # Get loaded RDMA modules
        rdma_modules_dict = phdl.exec('lsmod 2>/dev/null | grep -E "^mlx|^ib_" | awk \'{print $1}\' | sort | head -10')
        if head_node and head_node in rdma_modules_dict:
            modules = rdma_modules_dict[head_node].strip()
            if modules:
                module_list = [m.strip() for m in modules.split('\n') if m.strip()]
                if module_list:
                    metadata['rdma_drivers'] = module_list

        # Get driver version for mlx5_core (primary Mellanox driver)
        mlx5_version_dict = phdl.exec('modinfo mlx5_core 2>/dev/null | grep "^version:" | head -1')
        if head_node and head_node in mlx5_version_dict:
            mlx5_ver = mlx5_version_dict[head_node].strip()
            if mlx5_ver and ':' in mlx5_ver:
                version = mlx5_ver.split(':', 1)[1].strip()
                if version:
                    metadata['mlx5_driver_version'] = version
    except Exception as e:
        log.warning(f'Failed to get RDMA driver info: {e}')

    # Capture BIOS/IFWI version
    try:
        bios_dict = phdl.exec('sudo dmidecode -s bios-version 2>/dev/null || echo "unknown"')
        if head_node and head_node in bios_dict:
            metadata['bios_version'] = bios_dict[head_node].strip()
    except Exception as e:
        log.warning(f'Failed to get BIOS version: {e}')

    # Capture RCCL info if available
    try:
        if 'rccl_dir' in config_dict and config_dict['rccl_dir']:
            rccl_dir = config_dict['rccl_dir']

            # Get RCCL commit hash
            rccl_hash_dict = phdl.exec(f'cd {rccl_dir} 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo "unknown"')
            if head_node and head_node in rccl_hash_dict:
                rccl_hash = rccl_hash_dict[head_node].strip()
                if 'unknown' not in rccl_hash and len(rccl_hash) >= 7:  # Valid git hash (short or long)
                    metadata['rccl_commit'] = rccl_hash

            # Get RCCL branch name
            rccl_branch_dict = phdl.exec(
                f'cd {rccl_dir} 2>/dev/null && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"'
            )
            if head_node and head_node in rccl_branch_dict:
                rccl_branch = rccl_branch_dict[head_node].strip()
                if 'unknown' not in rccl_branch and rccl_branch:
                    metadata['rccl_branch'] = rccl_branch
    except Exception as e:
        log.warning(f'Failed to get RCCL info: {e}')

    # Capture RCCL-tests info if available
    try:
        if 'rccl_tests_dir' in config_dict and config_dict['rccl_tests_dir']:
            rccl_tests_dir = config_dict['rccl_tests_dir']

            # Get RCCL-tests commit hash
            tests_hash_dict = phdl.exec(
                f'cd {rccl_tests_dir} 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo "unknown"'
            )
            if head_node and head_node in tests_hash_dict:
                tests_hash = tests_hash_dict[head_node].strip()
                if 'unknown' not in tests_hash and len(tests_hash) >= 7:
                    metadata['rccl_tests_commit'] = tests_hash

            # Get RCCL-tests branch name
            tests_branch_dict = phdl.exec(
                f'cd {rccl_tests_dir} 2>/dev/null && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"'
            )
            if head_node and head_node in tests_branch_dict:
                tests_branch = tests_branch_dict[head_node].strip()
                if 'unknown' not in tests_branch and tests_branch:
                    metadata['rccl_tests_branch'] = tests_branch
    except Exception as e:
        log.warning(f'Failed to get RCCL-tests info: {e}')

    # Capture MPI info if available
    try:
        if 'mpi_dir' in config_dict and config_dict['mpi_dir']:
            mpi_dir = config_dict['mpi_dir']
            # Try multiple ways to get MPI version
            mpi_version_dict = phdl.exec(
                f'{mpi_dir}/bin/mpirun --version 2>/dev/null | head -1 || {mpi_dir}/mpirun --version 2>/dev/null | head -1 || mpirun --version 2>/dev/null | head -1 || echo "unknown"'
            )
            if head_node and head_node in mpi_version_dict:
                mpi_ver = mpi_version_dict[head_node].strip()
                if mpi_ver and 'unknown' not in mpi_ver:
                    metadata['mpi_version'] = mpi_ver
    except Exception as e:
        log.warning(f'Failed to get MPI version: {e}')

    # Capture test command if provided
    if test_command:
        metadata['test_command'] = test_command

    # Capture RCCL/NCCL environment variables from config (these are set via mpirun -x)
    rccl_env_vars = {}
    env_var_mapping = {
        'debug_level': 'NCCL_DEBUG',
        'ib_hca_list': 'NCCL_IB_HCA',
        'net_dev_list': 'UCX_NET_DEVICES',
        'ucx_tls': 'UCX_TLS',
        'nccl_net_plugin': 'NCCL_NET_PLUGIN',
        'gid_index': 'NCCL_IB_GID_INDEX',
        'oob_port': 'NCCL_SOCKET_IFNAME',
        'rocm_path_var': 'ROCM_PATH',
        'mpi_path_var': 'MPI_PATH',
        'rccl_path_var': 'RCCL_PATH',
    }

    for config_key, env_name in env_var_mapping.items():
        if config_key in config_dict and config_dict[config_key]:
            value = config_dict[config_key]
            if value and str(value).lower() not in ['none', 'null', '']:
                rccl_env_vars[env_name] = str(value)

    # Also capture shell environment variables if requested
    if env_vars:
        for var_name in env_vars:
            # Skip if already captured from config
            if var_name in rccl_env_vars:
                continue
            try:
                var_dict = phdl.exec(f'echo ${var_name}')
                if head_node and head_node in var_dict:
                    value = var_dict[head_node].strip()
                    if value and value != var_name:  # Not just echoing the variable name
                        rccl_env_vars[var_name] = value
            except Exception as e:
                log.warning(f'Failed to get env var {var_name}: {e}')

    if rccl_env_vars:
        metadata['environment_variables'] = rccl_env_vars

    # Add cluster configuration summary
    metadata['cluster_info'] = {'num_nodes': len(node_list), 'node_names': node_list}

    # Add test configuration summary
    if config_dict:
        test_config_summary = {}
        # Include key test parameters
        for key in [
            'data_type_list',
            'gpu_count_list',
            'start_msg_size',
            'end_msg_size',
            'no_of_cycles',
            'warmup_iterations',
            'no_of_iterations',
        ]:
            if key in config_dict:
                test_config_summary[key] = config_dict[key]

        if test_config_summary:
            metadata['test_config'] = test_config_summary

    log.info(f'Collected metadata: {list(metadata.keys())}')
    return metadata
