# CVS Benchmark Automation

Automated benchmark execution for CVS (Cluster Validation Suite).

**Supports both SLURM and non-SLURM environments.**

## Structure

```
cvs-sbatch/
├── run.sh                      # Main orchestrator script
├── run_direct.sh               # Direct execution (non-SLURM)
├── sbatch/
│   ├── default.sbatch          # SBATCH configuration for RCCL tests
│   └── aorta.sbatch            # SBATCH configuration for Aorta benchmarks
├── lib/
│   ├── common.sh               # Logging, cleanup, error handling
│   ├── git_ops.sh              # CVS repository management
│   ├── cluster_config.sh       # Cluster JSON generation (SLURM + manual)
│   └── python_env.sh           # Python environment & test execution
├── examples/
│   └── cluster_config_chi.json # Example cluster config for CHI cluster
├── cluster_config.json.template # Template for manual cluster config
├── config.json                 # RCCL benchmark configuration
├── config_aorta.yaml           # Aorta benchmark configuration
└── README.md                   # This file
```

## Features

- **Dual Environment Support**: Works with SLURM and non-SLURM clusters
- **Modular Design**: Separated concerns for maintainability
- **Idempotent**: Safe to run multiple times
- **Auto-Cleanup**: Removes generated files on exit
- **Python Environment Fallback**: UV → System pytest → Venv
- **Dynamic Cluster Configuration**: Auto-generates from SLURM or manual config

---

## Quick Start

### SLURM Environment

```bash
# RCCL benchmark
sbatch sbatch/default.sbatch

# Aorta benchmark
sbatch sbatch/aorta.sbatch
```

### Non-SLURM Environment

```bash
# 1. Create cluster config from template
cp cluster_config.json.template cluster_config.json

# 2. Edit with your node information
vim cluster_config.json

# 3. Run benchmark
./run_direct.sh
```

---

## Usage

### SLURM Mode

When running under SLURM, the cluster configuration is automatically generated from SLURM environment variables (`SLURM_NODELIST`, `SLURM_SUBMIT_HOST`, etc.).

```bash
# Submit RCCL benchmark
sbatch sbatch/default.sbatch

# Submit Aorta benchmark
CONFIG_FILE=config_aorta.yaml sbatch sbatch/aorta.sbatch
```

### Non-SLURM Mode (Manual Cluster Config)

For clusters without SLURM, you must provide a cluster configuration file.

#### Step 1: Create Cluster Configuration

```bash
# Option A: Copy template
cp cluster_config.json.template cluster_config.json

# Option B: Use example
cp examples/cluster_config_chi.json cluster_config.json
```

#### Step 2: Edit Configuration

```json
{
    "username": "your-username",
    "priv_key_file": "/home/user/.ssh/id_ed25519",
    "head_node_dict": {
        "mgmt_ip": "head-node-hostname"
    },
    "node_dict": {
        "node1-hostname": {
            "bmc_ip": "NA",
            "vpc_ip": "node1-hostname"
        },
        "node2-hostname": {
            "bmc_ip": "NA",
            "vpc_ip": "node2-hostname"
        }
    }
}
```

**Fields:**
- `username`: SSH username (leave empty for current user)
- `priv_key_file`: Path to SSH private key (leave empty for auto-detection)
- `head_node_dict.mgmt_ip`: Head/management node hostname or IP
- `node_dict`: Map of compute nodes with their VPC IPs

#### Step 3: Run Benchmark

```bash
# Using run_direct.sh (recommended)
./run_direct.sh

# Or with custom cluster config
./run_direct.sh --cluster my_cluster.json

# Or with environment variable
MANUAL_CLUSTER_CONFIG=my_cluster.json ./run.sh
```

### Running Specific Benchmarks

```bash
# RCCL benchmark (non-SLURM)
MANUAL_CLUSTER_CONFIG=cluster_config.json \
TEST_PATH=./cvs/tests/rccl/rccl_heatmap_cvs.py \
./run.sh

# Aorta benchmark (non-SLURM)
MANUAL_CLUSTER_CONFIG=cluster_config.json \
TEST_PATH=./cvs/tests/benchmark/test_aorta.py \
CONFIG_FILE=config_aorta.yaml \
./run.sh
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CVS_REPO_URL` | CVS Git repository URL | `https://github.com/ROCm/cvs.git` |
| `CVS_DIR` | Local path to CVS repository | `/apps/cvs_tests/cvs` |
| `CVS_BRANCH` | CVS branch to use | `main` |
| `TEST_PATH` | Test file to execute | `./cvs/tests/rccl/rccl_heatmap_cvs.py` |
| `CONFIG_FILE` | Benchmark config (JSON/YAML) | `config.json` |
| `LOG_FILE` | Path to log file | `/tmp/cvs_$USER_$$.log` |
| `MANUAL_CLUSTER_CONFIG` | Manual cluster config (non-SLURM) | `cluster_config.json` |
| `DEBUG_MODE` | Enable verbose logging | `0` |

### Running Aorta Benchmarks

**Prerequisites:**
- Docker daemon running on compute nodes
- User in 'docker' group on compute nodes
- Aorta repository on shared filesystem
- Docker image pullable or pre-cached

```bash
# SLURM
sbatch sbatch/aorta.sbatch

# Non-SLURM
MANUAL_CLUSTER_CONFIG=cluster_config.json \
CONFIG_FILE=config_aorta.yaml \
TEST_PATH=./cvs/tests/benchmark/test_aorta.py \
./run_direct.sh
```

### Custom Configuration

```bash
# Copy and modify config
cp config_aorta.yaml my_config.yaml
vim my_config.yaml

# Run with custom config
CONFIG_FILE=my_config.yaml ./run_direct.sh
```

---

## Debug Mode

Enable verbose logging for troubleshooting:

```bash
# SLURM
DEBUG_MODE=1 sbatch sbatch/default.sbatch

# Non-SLURM
DEBUG_MODE=1 ./run_direct.sh
```

Debug mode provides:
- Detailed execution traces (`set -x`)
- Debug-level log messages
- Directory and path verification
- Step-by-step progress indicators

---

## Library Modules

### `lib/common.sh`
- Logging functions (`log_info`, `log_error`, `log_success`, `log_debug`)
- Cleanup functions
- Error handling and traps

### `lib/git_ops.sh`
- Clone CVS repository
- Update to latest branch
- Idempotent git operations

### `lib/cluster_config.sh`
- Auto-detect SLURM vs non-SLURM environment
- SSH key detection
- Node IP resolution
- Generate `cluster.json` from SLURM allocation **or** manual config

### `lib/python_env.sh`
- UV-based execution
- System pytest execution
- Virtual environment management
- Automatic fallback mechanism

---

## Generated Files

The following files are generated during execution and cleaned up automatically:

- `cluster.json` - Generated from SLURM allocation or manual config
- `.venv/` - Python virtual environment (if created)

---

## Development

### Adding New SBATCH Configurations

```bash
cp sbatch/default.sbatch sbatch/multi_node.sbatch
# Edit sbatch/multi_node.sbatch as needed
sbatch sbatch/multi_node.sbatch
```

### Adding a New Benchmark

1. Create a test file in `cvs/tests/` (e.g., `cvs/tests/benchmark/test_mybench.py`)
2. Create a config file (e.g., `config_mybench.yaml`)
3. Create an sbatch file:

```bash
cp sbatch/default.sbatch sbatch/mybench.sbatch
# Edit to set:
#   export TEST_PATH="./cvs/tests/benchmark/test_mybench.py"
#   export CONFIG_FILE="config_mybench.yaml"
sbatch sbatch/mybench.sbatch
```

### Testing Individual Modules

```bash
# Source a module and test its functions
source lib/common.sh
log_info "Testing logging"

# Test cluster config generation (non-SLURM)
source lib/cluster_config.sh
MANUAL_CLUSTER_CONFIG=my_cluster.json generate_cluster_config_manual
```

---

## Troubleshooting

### Check SLURM Output

```bash
# Check output
cat /scratch/users/$USER/tmp/slurm/cvs-benchmark-<JOB_ID>.out

# Check errors
cat /scratch/users/$USER/tmp/slurm/cvs-benchmark-<JOB_ID>.err
```

### Non-SLURM: "Cluster config not found"

```bash
# Ensure you have a cluster config file
cp cluster_config.json.template cluster_config.json
# Edit with your nodes
vim cluster_config.json
```

### Non-SLURM: "Failed to resolve hostname"

- Ensure hostnames in your config are resolvable
- Use IP addresses instead of hostnames if DNS is not available
- Verify SSH connectivity to all nodes

### Manual Cleanup

```bash
rm -f cluster.json
rm -rf /path/to/cvs/.venv
```

---

## Planned Features

- TODO: Dynamic benchmark configuration generation
    - Level 1: collectives [All_reduce, Reduce_gather, Reduce_scatter, AlltoAll], dtypes [float, bfloat16]
    - Level 2: check functionality of all collectives & dtypes
    - Level 3: sweep across algo, proto combination matrix
    - Level 4: user defined sweeps
- TODO: Bats unit tests for individual modules
- TODO: Bats integration tests

---

## Author

surya.periaswamy@amd.com

## License

Internal use only.
