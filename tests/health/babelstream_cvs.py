import pytest

import re
import sys
import os
import sys
import time
import json
import logging
import time


sys.path.insert( 0, './lib' )
from parallel_ssh_lib import *
from utils_lib import *

import globals

log = globals.log


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Retrieve the --cluster_file CLI option provided to pytest.

    Args:
      pytestconfig: Built-in pytest fixture exposing command-line options.

    Returns:
      str: Path to the cluster configuration JSON file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Retrieve the --config_file CLI option provided to pytest.

    Args:
      pytestconfig: Built-in pytest fixture exposing command-line options.

    Returns:
      str: Path to the test configuration JSON file.

    Notes:
      - Ensure your pytest invocation includes: --config_file=/path/to/config.json
      - Module scope ensures this runs once per module to avoid repeated lookups.
    """ 
    return pytestconfig.getoption("config_file")





@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and return the entire cluster configuration from JSON.

    Args:
    cluster_file (str): Path to the cluster JSON file.

    Returns:
    dict: Parsed cluster configuration (nodes, credentials, etc.).
    """
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
    log.info(cluster_dict)
    return cluster_dict





@pytest.fixture(scope="module")
def config_dict(config_file):
    """
    Load and return the 'babelstream' subsection from the test configuration JSON.

    Args:
      config_file (str): Path to the test configuration JSON.

    Returns:
      dict: The 'babelstream' configuration block, expected to include:
        - path: location where hip-stream (BabelStream HIP binary) will live
        - git_install_path: directory to clone and build BabelStream
        - git_url: BabelStream repository URL
        - results: expected performance thresholds for kernels (copy/add/mul/triad/dot)
    """ 
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['babelstream']
    log.info(config_dict)
    return config_dict



def parse_babelstream_results( out_dict, exp_dict ):
    """
    Parse BabelStream outputs per node and validate kernel bandwidths vs expected thresholds.

    Args:
      out_dict (dict[str, str]): Mapping: node -> the full stdout/stderr of BabelStream runs.
      exp_dict (dict): Expected thresholds like:
        {
          "copy": "<float-like>", "add": "<float-like>", "mul": "<float-like>",
          "triad": "<float-like>", "dot": "<float-like>"
        }

    Behavior:
      - Uses regex to extract measured GB/s for kernels (Copy, Add, Mul, Triad, Dot).
      - For each occurrence (multiple ranks), compares actual vs expected; fails if actual < expected.

    Notes:
      - Regex assumes the standard BabelStream output layout:
        "<Kernel> <GB/s> <some_time> <some_other_value>"
      - Values are interpreted as floats; ensure the configuration provides numeric-like strings.

    """
    for node in out_dict.keys():
        pattern = r"Copy\s+([0-9\.]+)\s+[0-9\.]+\s+[0-9\.]+\s+"
        copy_list = re.findall( pattern, out_dict[node] )
        for copy_val in copy_list:
            if float(copy_val) < float(exp_dict['copy']):
                fail_test(f"Copy value {copy_val} less than expected {exp_dict['copy']} on node {node}")
        pattern = r"Add\s+([0-9\.]+)\s+[0-9\.]+\s+[0-9\.]+\s+"
        add_list = re.findall( pattern, out_dict[node] )
        for add_val in add_list:
            if float(add_val) < float(exp_dict['add']):
                fail_test(f"Add value {add_val} less than expected {exp_dict['add']} on node {node}")
        pattern = r"Mul\s+([0-9\.]+)\s+[0-9\.]+\s+[0-9\.]+\s+"
        mul_list = re.findall( pattern, out_dict[node] )
        for mul_val in mul_list:
            if float(mul_val) < float(exp_dict['mul']):
                fail_test(f"Mul value {mul_val} less than expected {exp_dict['mul']} on node {node}")
        pattern = r"Triad\s+([0-9\.]+)\s+[0-9\.]+\s+[0-9\.]+\s+"
        triad_list = re.findall( pattern, out_dict[node] )
        for triad_val in triad_list:
            if float(triad_val) < float(exp_dict['triad']):
                fail_test(f"Triad value {triad_val} less than expected {exp_dict['triad']} on node {node}")
        pattern = r"Dot\s+([0-9\.]+)\s+[0-9\.]+\s+[0-9\.]+\s+"
        dot_list = re.findall( pattern, out_dict[node] )
        for dot_val in dot_list:
            if float(dot_val) < float(exp_dict['dot']):
                fail_test(f"Triad value {dot_val} less than expected {exp_dict['dot']} on node {node}")




@pytest.fixture(scope="module")
def shdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for the head node only.

    Args:
      cluster_dict (dict): Cluster metadata fixture (see phdl docstring).
    
    Returns:
      Pssh: Handle configured for the first node (head node) in node_dict.

    Notes:
      - Useful when commands should be executed only from a designated head node.
      - Module scope ensures a single connection context for the duration of the module.
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
    """
    nhdl_dict = {}
    node_list = list(cluster_dict['node_dict'].keys())
    head_node = node_list[0]
    shdl = Pssh( log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return shdl


        

@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Create a parallel SSH handle (Pssh) for executing commands across all cluster nodes.

    Args:
      cluster_dict (dict): Cluster metadata containing at least:
        - node_dict: mapping of node name/IP -> details
        - username: SSH username for nodes
        - priv_key_file: path to SSH private key

    Returns:
      Pssh: A handle that runs commands in parallel and returns a dict of node -> output.

    """
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return phdl


 
 
def test_create_wrapper_script( phdl, shdl, config_dict ):
    """
    Create a wrapper script to run hip-stream with device bound to MPI rank.

    Script content:
      #!/bin/bash
      <path>/hip-stream --device $OMPI_COMM_WORLD_RANK -n 50 -s 268435456

    Steps:
      1) Create wrapper.sh at the configured path.
      2) Verify the file exists.
      3) chmod +x wrapper.sh
      4) Update test result.

    Args:
      hdl: Single-node SSH handler used to write files and update permissions on the head node.
      config_dict (dict): Contains 'path' for the wrapper location.

    Notes:
      - -n and -s parameters can be tuned via config to adjust runtime and problem size.
    """ 

    globals.error_list = []
    log.info('Testcase create hip-stream wrapper-script')
    path = config_dict['path']

    if config_dict['nfs_install'] is True:
        hdl = shdl
    else:
        hdl = phdl

    out_dict = hdl.exec(f'cd {path};ls -l')

    print(f"echo -e '#!/bin/bash\n{path}/hip-stream --device $OMPI_COMM_WORLD_RANK -n 50 -s 268435456' > {path}/wrapper.sh")
    out_dict = hdl.exec(f"echo -e '#!/bin/bash\\n{path}/hip-stream --device $OMPI_COMM_WORLD_RANK -n 50 -s 268435456' > {path}/wrapper.sh" )
    for node in out_dict.keys():
        print(out_dict[node])

    time.sleep(1)

    out_dict = hdl.exec(f'cat {path}/wrapper.sh')
    out_dict = phdl.exec(f'ls -l {path}/wrapper.sh')
    for node in out_dict.keys():
        if re.search('No such file', out_dict[node], re.I ):
            fail_test(f'Creation of wrapper script failed, file not found or content missing on node {node}' )
    out_dict = hdl.exec(f'chmod 755 {path}/wrapper.sh')
    update_test_result()




def test_run_babelstream(phdl, config_dict, ):
    """
    Run BabelStream across 8 MPI ranks (GPUs) and validate output/error patterns and performance presence.

    Args:
      phdl: Parallel SSH handle to execute commands on nodes.
      config_dict (dict): BabelStream configuration with:
        - 'path': Directory containing wrapper.sh (created earlier).
        - 'results': Expected performance thresholds per kernel for post-parse validation.

    Behavior:
      - Resets global error list and logs the start.
      - Changes to the configured path and launches the wrapper over 8 ranks using mpiexec.
      - Scans outputs per node for generic failure patterns (fail|error|fatal|core|crash).
      - Ensures expected performance lines (e.g., 'Triad') are present to confirm proper run.
      - Invokes parse_babelstream_results to compare measured bandwidths against thresholds.
      - Finalizes test status with update_test_result.

    Assumptions:
      - wrapper.sh exists, is executable, and binds device selection to MPI rank.
      - parse_babelstream_results and update_test_result utilities are available.
      - Timeout (120s) is sufficient for your platform/workload; adjust as needed.
    """
    globals.error_list = []
    log.info('Testcase Run babelstream on all 8 GPUs')
    path = config_dict['path']
    exp_dict = config_dict['results']
    out_dict = phdl.exec(f'cd {path};mpiexec --allow-run-as-root -n 8 ./wrapper.sh', timeout=(60*2))
    for node in out_dict.keys():
        if re.search( 'fail|error|fatal|core|crash', out_dict[node], re.I ):
            fail_test(f'Failure error patterns seen in babelstream test on node {node}')
        if not re.search( 'Triad', out_dict[node], re.I ):
            fail_test(f'Expected performance number outputs not printed in babelstream out on node {node} - Test Failed')
    parse_babelstream_results( out_dict, exp_dict )
    update_test_result()
