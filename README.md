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

CVS is packaged as a proper Python package and can be installed using pip.

## Prerequisites

- Python 3.9 or later
- Git

### Debian/Ubuntu Systems

On Debian and Ubuntu distributions, the `venv` module is not included in the base Python package. Install it before proceeding:

```bash
sudo apt install python3-venv
```

## Method 1: Install from Source (Quick Method using Makefile)

1. **Clone the repository and install using make:**
```bash
git clone https://github.com/ROCm/cvs
cd cvs
make install
```

This will automatically:
- Build the source distribution
- Create a virtual environment named `.cvs_venv/` under repository root `cvs/`
- Install CVS in the virtual environment

2. **Activate the virtual environment:**
```bash
source .cvs_venv/bin/activate
```

This is the quickest way to install CVS from source, allowing you to use the latest development version of the software.

## Method 2: Install from Source (Manual Method)

For users who want to install CVS in a custom virtual environment:

1. **Clone the repository and build cvs python pkg:**
```bash
git clone https://github.com/ROCm/cvs
cd cvs
python setup.py sdist
```

2. **Create and activate a virtual environment (recommended):**
```bash
python3 -m venv cvs_env  # or any custom name
source cvs_env/bin/activate  # On Windows: cvs_env\Scripts\activate
```

3. **Install cvs python pkg:**
```bash
pip install dist/cvs*.tar.gz
```

This method gives you more control over the virtual environment name and location.

## Verification

After installation, verify that CVS is working:

```bash
cvs --version  # Should show version information
cvs list       # Should list available test suites
```

The `cvs` command will now be available globally in your environment. You can run tests from anywhere, not just from the CVS source directory.

# How to upgrade

To upgrade CVS to the latest version:

## If installed from source using make:
```bash
cd /path/to/cvs/source
git pull  # Get latest changes
make install
source .cvs_venv/bin/activate
```

## If installed from source manually:
```bash
cd /path/to/cvs/source
git pull  # Get latest changes
python setup.py sdist
pip install --upgrade dist/cvs*.tar.gz
```

After upgrading, verify the installation:
```bash
cvs --version
```

# How to run CVS Tests

## Setting up Configuration Files

Before running tests, you need to set up your cluster and test configuration files. Sample configuration files are included with the CVS installation.

### Copy Sample Configuration Files

CVS provides a convenient `cvs copy-config` command to copy sample configuration files. This is the recommended method for setting up your configuration files.

First, list available configuration files:

```bash
cvs copy-config --list
```

Then copy specific files as needed:

```bash
# Copy cluster configuration
cvs copy-config cluster.json --output /tmp/cvs/input/cluster_file/cluster.json

# Alternatively, generate cluster configuration for multiple hosts (see 'Generate Cluster Configuration File' section below)

# Copy test-specific configurations
cvs copy-config rccl/rccl_config.json --output /tmp/cvs/input/config_file/rccl_config.json
cvs copy-config health/mi300_health_config.json --output /tmp/cvs/input/config_file/health_config.json
```

Or copy all configuration files at once:

```bash
cvs copy-config --all --output /tmp/cvs/input/
```

To force overwrite existing files:

```bash
cvs copy-config --all --output /tmp/cvs/input/ --force
```

**Note**: The `cvs copy-config` command automatically creates output directories as needed, preserves the original directory structure when copying all files, and can overwrite existing files with the `--force` option.

### Generate Cluster Configuration File

Alternatively, you can generate the cluster JSON file for N number of hosts using the `cvs generate cluster_json` command. This is useful for automatically creating cluster configurations from a list of host IPs.

#### Option 1: Using a hosts file

Create a hosts file with one IP address or hostname per line (supports IP ranges like `192.168.1.10-20` and bracket notation like `hostname[1-10]`, comments with `#`, and blank lines are ignored):

```bash
# Example hosts file: /tmp/hosts.txt
# Single host
192.168.1.10

# IP range (expands to 192.168.1.11 through 192.168.1.15)
192.168.1.11-15

# Hostname bracket range (expands to server01 through server05)
server[01-05]

# Another single host
192.168.1.20

# Additional hosts can be added here
```

Then generate the cluster JSON:

```bash
cvs generate cluster_json --input_hosts_file /tmp/hosts.txt --output_json_file /tmp/cvs/input/cluster_file/cluster.json --username myuser --key_file ~/.ssh/id_rsa --head_node 192.168.1.10
```

#### Option 2: Using comma-separated hosts

Alternatively, specify hosts directly on the command line using comma-separated values:

```bash
cvs generate cluster_json --hosts "192.168.1.10,192.168.1.11-15,server[01-05]" --output_json_file /tmp/cvs/input/cluster_file/cluster.json --username myuser --key_file ~/.ssh/id_rsa --head_node 192.168.1.10
```

Example with mixed formats:

```bash
cvs generate cluster_json --hosts "mia1-p01-g20,mia1-p01-g22,mia1-p01-g[24-30],192.168.1.10-12" --output_json_file cluster.json --username myuser --key_file ~/.ssh/id_rsa
```

**Command options**:
- `--input_hosts_file`: Path to file with host IPs/hostnames (one per line, supports ranges and bracket notation)
- `--hosts`: Comma-separated list of host IPs/hostnames (supports ranges and bracket notation)
  - **Note**: Use either `--input_hosts_file` OR `--hosts`, not both
- `--output_json_file`: Path to output cluster JSON file
- `--username`: SSH username for the hosts
- `--key_file`: Path to SSH private key file
- `--head_node`: IP of the head node (optional, defaults to first host in the list; can be different from the hosts in the file)

**Supported range formats**:
- IP ranges: `192.168.1.10-20` expands to `192.168.1.10` through `192.168.1.20`
- Hostname bracket ranges: `server[1-5]` expands to `server1` through `server5`
- Leading zeros preserved: `node[01-10]` expands to `node01` through `node10`
- With suffix: `node[1-3].example.com` expands to `node1.example.com`, `node2.example.com`, `node3.example.com`

### Modify Configuration Files

Edit the copied files to match your cluster setup. If you generated the cluster.json using the `cvs generate cluster_json` command above, it should already be properly configured and may require no editing.

```bash
# Edit cluster configuration (skip if generated above)
vi /tmp/cvs/input/cluster_file/cluster.json

# Edit test-specific configuration (example for RCCL)
vi /tmp/cvs/input/config_file/rccl/rccl_config.json
```

**Important**: Update the following in your configuration files:
- **Cluster file**: IP addresses, hostnames, SSH credentials for your cluster nodes
- **Config files**: Test-specific parameters like network interfaces, GPU settings, etc.

### Example Configuration File Locations

After setup, your files will be at:
- Cluster config: `/tmp/cvs/input/cluster_file/cluster.json`
- RCCL config: `/tmp/cvs/input/config_file/rccl/rccl_config.json`
- Other configs: `/tmp/cvs/input/config_file/*/*.json`

## Running Tests

Once your configuration files are set up, you can run CVS tests using the convenient `cvs` command.

```bash
# List all available test suites
cvs list

# List sub-tests within a specific test suite
cvs list rccl_multinode_cvs

# Run all tests from a specific test suite
cvs run rccl_multinode_cvs --cluster_file /tmp/cvs/input/cluster_file/cluster.json --config_file /tmp/cvs/input/config_file/rccl/rccl_config.json --html=/var/www/html/cvs/rccl_test_report.html --self-contained-html --capture=tee-sys

# Run a specific test from a test suite
cvs run rccl_multinode_cvs test_collect_hostinfo --cluster_file /tmp/cvs/input/cluster_file/cluster.json --config_file /tmp/cvs/input/config_file/rccl/rccl_config.json --html=/var/www/html/cvs/rccl_test_report.html --self-contained-html --capture=tee-sys

# Run multiple specific tests from a test suite
cvs run rccl_multinode_cvs test_collect_hostinfo test_basic_ring --cluster_file /tmp/cvs/input/cluster_file/cluster.json --config_file /tmp/cvs/input/config_file/rccl/rccl_config.json --html=/var/www/html/cvs/rccl_test_report.html --self-contained-html --capture=tee-sys

# Run without HTML reporting
cvs run rccl_multinode_cvs --cluster_file /tmp/cvs/input/cluster_file/cluster.json --config_file /tmp/cvs/input/config_file/rccl/rccl_config.json
```

## Executing Commands on Cluster Nodes

CVS provides an `exec` command to execute arbitrary shell commands on all nodes in the cluster simultaneously using parallel SSH.

```bash
# Execute a command on all nodes using --cluster_file
cvs exec --cmd "hostname" --cluster_file /tmp/cvs/input/cluster_file/cluster.json

# Execute a command using CLUSTER_FILE environment variable
CLUSTER_FILE=/tmp/cvs/input/cluster_file/cluster.json cvs exec --cmd "hostname"

# Execute other commands
cvs exec --cmd "uptime" --cluster_file /tmp/cvs/input/cluster_file/cluster.json
cvs exec --cmd "rocm-smi --showproductname --showmeminfo vram" --cluster_file /tmp/cvs/input/cluster_file/cluster.json
```

The `exec` command supports the following options:
- `--cmd`: The shell command to execute on all nodes (required)
- `--cluster_file`: Path to cluster configuration JSON file (optional if CLUSTER_FILE env var is set)

## Command Help

```bash
$ cvs --help
usage: cvs [-h] [--version] {run,list,generate,monitor,exec} ...

Cluster Validation Suite (CVS)

positional arguments:
  {run,list,generate,monitor,exec}
    run                 Run a specific test (wrapper over pytest)
    list                List available tests
    generate            Generate configuration files or templates
    monitor             Run cluster monitoring scripts
    exec                Execute a command on all nodes in the cluster

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

```bash
$ cvs run --help
usage: cvs run [-h] --cluster_file CLUSTER_FILE --config_file
               CONFIG_FILE [--html HTML] [--self-contained-html]
               [--log-file LOG_FILE]
               [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               [--capture {no,tee-sys,tee-merged,fd,sys}]
               [test] [function]

positional arguments:
  test                  Name of the test file to run (omit to list
                        available tests)
  function              Optional: specific test function to run

options:
  -h, --help            show this help message and exit
  --cluster_file CLUSTER_FILE
                        Path to cluster configuration JSON file
                        (required)
  --config_file CONFIG_FILE
                        Path to test configuration JSON file (required)
  --html HTML           Pytest: Create HTML report file at given path
  --self-contained-html
                        Pytest: Create a self-contained HTML file
                        containing all the HTML report
  --log-file LOG_FILE   Pytest: Path to file for logging output
                        (default: /tmp/cvs/test.log)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Pytest: Level of messages to catch/display
  --capture {no,tee-sys,tee-merged,fd,sys}
                        Per-test capturing method for stdout/stderr
```

## CVS Run Command Options

The `cvs run` command supports common pytest options directly:

- `--html`: Create HTML report file at given path
- `--self-contained-html`: Create a self-contained HTML file containing all the HTML report
- `--log-file`: Path to file for logging output (default: /tmp/cvs/test.log)
- `--log-level`: Level of messages to catch/display (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--capture`: Per-test capturing method for stdout/stderr (no, tee-sys, tee-merged, fd, sys)

All other pytest arguments are supported and passed transparently to pytest. For the complete list, run: `pytest --help`

## CVS-Specific Options

**Note**: The `--cluster_file` and `--config_file` arguments are **mandatory** for running tests. These files contain the cluster configuration and test-specific settings required by CVS tests.

- `--cluster_file`: Path to cluster configuration JSON file (required)
- `--config_file`: Path to test configuration JSON file (required)

## Complete CVS Run Example

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