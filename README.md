# Cluster Validation Suite

> [!NOTE]

> The published Cluster Validation Suite documentation is available [here](https://rocm.docs.amd.com/projects/cvs/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

CVS is a collection of tests scripts that can validate AMD AI clusters end to end from running single node burn in health tests to cluster wide distributed training and inferencing tests. CVS can be used by AMD customers to verify the health of the cluster as a whole which includes verifying the GPU/CPU node health, Host OS configuratin checks, NIC validations etc. CVS test suite collection comprises of the following set of tests

1. Platform Tests - Host OS config checks, BIOS checks, Firmware/Driver checks, Network config checks.
2. Burn in Health Tests - AGFHC, Transferbench, RocBLAS, rocHPL, Single node RCCL
3. Network Tests - Ping checks, Multi node RCCL validations for different collectives
4. Distributed Training Tests - Run Llama 70B and 405B model distributed trainings with JAX and Megatron frameworks.
5. Distributed Inferencing Tests - Work in Progress

CVS leverages the PyTest open source framework to run the tests and generate reports and can be launched from a head-node or any linux management station which has connectivity to the cluster nodes via SSH. The single node tests are run in parallel cluster wide using the parallel-ssh open source python modules to optimize the time for running them. Currently CVS has been validated only on Ubuntu based Linux distro clusters. 

CVS Repository is organized as the following directories

1. tests directory - This comprises of the actual pytest scripts that would be run which internally will be calling the library functions under the ./lib directory which are in native python and can be invoked from any python scripts for reusability. The tests directory has sub folder based on the nature of the tests like health, rccl, training etc.
2. lib directory - This is a collection of python modules which offer a wide range of utility functions and can be reused in other python scripts as well.
3. input directory - This is a collection of the input json files that are provided to the pytest scripts using the 2 arguments --cluster_file and the --config_file. The cluster_file is a JSON file which captures all the aspects of the cluster testbed, things like the IP address/hostnames, username, keyfile etc. We avoid putting a lot of other information like linux net devices names or rdma device names etc to keep it user friendly and auto-discover them.
4. utils directory - This is a collection of standalone scripts which can be run natively without pytest and offer different utility functions.

# How to install

CVS is packaged as a proper Python package and can be installed using pip. There are two main ways to install CVS:

## Method 1: Install from Source

1. **Clone the repository and build cvs python pkg:**
```bash
git clone https://github.com/ROCm/cvs
cd cvs
python setup.py sdist
```

2. **Create and activate a virtual environment (recommended):**
```bash
python3 -m venv cvs_env
source cvs_env/bin/activate  # On Windows: cvs_env\Scripts\activate
```

3. **Install cvs python pkg:**
```bash
pip install dist/cvs*.tar.gz
```

This installs CVS from source, allowing you to use the latest developement version of the software.

## Method 2: Install from Released Distribution

If CVS is available as a released package as a distribution file:

1. **Create and activate a virtual environment (recommended):**
```bash
python3 -m venv cvs_env
source cvs_env/bin/activate  # On Windows: cvs_env\Scripts\activate
```

2. **Install from a local distribution file:**
```bash
pip install cvs-0.1.0.tar.gz
```

## Verification

After installation, verify that CVS is working:

```bash
cvs --version  # Should show version information
cvs list       # Should list available test suites
```

The `cvs` command will now be available globally in your environment. You can run tests from anywhere, not just from the CVS source directory.

# How to upgrade

To upgrade CVS to the latest version:

## If installed from source:
```bash
cd /path/to/cvs/source
git pull  # Get latest changes
python setup.py sdist
pip install --upgrade dist/cvs*.tar.gz
```

## If installed from local distribution:
```bash
pip install --upgrade cvs-0.1.1.tar.gz
```

After upgrading, verify the installation:
```bash
cvs --version
```

# How to run CVS Tests

## Setting up Configuration Files

Before running tests, you need to set up your cluster and test configuration files. Sample configuration files are included with the CVS installation.

### Find and Copy Sample Configuration Files

```bash
# Find the location of installed CVS input files
CVS_INPUT_DIR=$(python -c "import cvs.input; print(cvs.input.__path__[0])")
echo "CVS input files are located at: $CVS_INPUT_DIR"

# Copy sample files to your home directory
mkdir -p ~/CVS/INPUT
cp -r $CVS_INPUT_DIR/* ~/CVS/INPUT/

# List the copied files
ls -la ~/CVS/INPUT/
```

### Modify Configuration Files

Edit the copied files to match your cluster setup:

```bash
# Edit cluster configuration
vi ~/CVS/INPUT/cluster_file/cluster.json

# Edit test-specific configuration (example for RCCL)
vi ~/CVS/INPUT/config_file/rccl/rccl_config.json
```

**Important**: Update the following in your configuration files:
- **Cluster file**: IP addresses, hostnames, SSH credentials for your cluster nodes
- **Config files**: Test-specific parameters like network interfaces, GPU settings, etc.

### Example Configuration File Locations

After setup, your files will be at:
- Cluster config: `~/CVS/INPUT/cluster_file/cluster.json`
- RCCL config: `~/CVS/INPUT/config_file/rccl/rccl_config.json`
- Other configs: `~/CVS/INPUT/config_file/*.json`

## Running Tests

Once your configuration files are set up, you can run CVS tests using the convenient `cvs` command.

```bash
# List all available test suites
cvs list

# Run all tests from a specific test suite
cvs run rccl_multinode_cvs --cluster_file ./input/cluster_file/cluster.json --config_file ./input/config_file/rccl_config.json --html=/var/www/html/cvs/rccl_test_report.html --self-contained-html --capture=tee-sys

# Run a specific test from a test suite
cvs run rccl_multinode_cvs test_collect_hostinfo --cluster_file ./input/cluster_file/cluster.json --config_file ./input/config_file/rccl_config.json --html=/var/www/html/cvs/rccl_test_report.html --self-contained-html --capture=tee-sys

# Run without HTML reporting
cvs run rccl_multinode_cvs --cluster_file ./input/cluster_file/cluster.json --config_file ./input/config_file/rccl_config.json
```

## Command Line Options

The `cvs run` command supports common pytest options directly:

- `--html`: Create HTML report file at given path
- `--self-contained-html`: Create a self-contained HTML file containing all the HTML report
- `--log-file`: Path to file for logging output (default: /tmp/test.log)
- `--log-level`: Level of messages to catch/display (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--capture`: Per-test capturing method for stdout/stderr (no, tee-sys, tee-merged, fd, sys)

All other pytest arguments are supported and passed transparently to pytest. For the complete list, run: `pytest --help`

## CVS-Specific Options

- `--cluster_file`: Path to cluster configuration JSON file
- `--config_file`: Path to test configuration JSON file

## Example with full options

```bash
cvs run rccl_multinode_cvs \
  --cluster_file ./input/cluster_file/cluster.json \
  --config_file ./input/config_file/rccl_config.json \
  --html=/var/www/html/cvs/rccl_test_report.html \
  --self-contained-html \
  --log-file=/tmp/rccl_test.log \
  --log-level=INFO \
  --capture=tee-sys
```

You can also create wrapper shell scripts to run multiple test suites by putting different `cvs run` commands in a bash script.
