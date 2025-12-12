RCCL (ROCm Communication Collectives Library) tests are comprehensive benchmarks that validate distributed GPU communication performance across AMD GPU clusters. These tests ensure optimal performance for AI training, HPC workloads, and distributed computing

# How to run the tests

This Pytest script can be run in the following fashion (for the details on arguments and their purpose, please refer the main README under the CVS parent folder

```
(myenv) [user@host]~/cvs:(main)$
(myenv) [user@host]~/cvs:(main)$pwd
/home/user/cvs
(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/rccl/rccl_multinode_cvs.py --cluster_file input/cluster_file/cluster.json  --config_file input/config_file/rccl/rccl_config.json --html=/var/www/html/cvs/rccl.html --capture=tee-sys --self-contained-html

```

# Additional RCCL env setting to use AINIC RCCL net plugin

## Overview

The `ainic_env_script.sh` script (located at `input/config_file/rccl/ainic_env_script.sh`) sets recommended environment variables for optimal performance when using RCCL (ROCm Communication Collectives Library) with AMD AI Network Interface Cards (AINIC). These settings are designed to enhance communication efficiency in multinode GPU clusters.

This script is part of the CVS (Cluster Validation Suite) and is used to configure the environment for RCCL-based collective operations, such as all-reduce, all-gather, etc.

## Prerequisites

- AMD ANP (AMD Network Plugin) built/installed

## Usage

1. **Edit the Script**: Update the `ANP_HOME_DIR` variable in `ainic_env_script.sh` to point to your AMD ANP installation directory.

   ```bash
   export ANP_HOME_DIR=/path/to/your/amd-anp/installation
   ```

   Replace `/path/to/your/amd-anp/installation` with the actual absolute path.

2. **Configure rccl_config.json**: Ensure the `env_source_script` field in `rccl_config.json` points to the correct path of `ainic_env_script.sh`. For example:

   ```json
   "env_source_script": "/path/to/cvs/input/config_file/rccl/ainic_env_script.sh"
   ```

   CVS will automatically source this script before running the tests.

3. **Run RCCL Tests**: Use the pytest command as described in the "How to run the tests" section. The environment variables will be set automatically from the script.
