# CVS Benchmark Automation

Automated SLURM-based config generation for CVS (Cluster Validation Suite).

## Structure

```
cvs-sbatch/
├── run.sh                      # Main orchestrator script
├── sbatch/
│   ├── default.sbatch          # SBATCH configuration for RCCL tests
│   └── aorta.sbatch            # SBATCH configuration for Aorta benchmarks
├── lib/
│   ├── common.sh               # Logging, cleanup, error handling
│   ├── git_ops.sh              # CVS repository management
│   ├── cluster_config.sh       # Cluster JSON generation
│   └── python_env.sh           # Python environment & test execution
├── config.json                 # RCCL benchmark configuration
├── config_aorta.yaml           # Aorta benchmark configuration
└── README.md                   # This file
```

## Features

- **Modular Design**: Separated concerns for maintainability
- **Idempotent**: Safe to run multiple times
- **Auto-Cleanup**: Removes generated files on exit
- **Python Environment Fallback**: UV → System pytest → Venv
- **Dynamic Cluster Configuration**: Generates cluster.json from SLURM allocation

## Usage

### Submit RCCL Benchmark (Default)

```bash
sbatch sbatch/default.sbatch 
```

### Submit Aorta Benchmark

```bash
# Ensure config_aorta.yaml has correct aorta_path
sbatch sbatch/aorta.sbatch
```

**Prerequisites for Aorta:**
- Docker daemon running on compute nodes
- User in 'docker' group on compute nodes
- Aorta repository on shared filesystem
- Docker image pullable or pre-cached

### Local Testing

```bash
./run.sh
```

### Configuration via Environment Variables

```bash
# Override CVS repository
export CVS_REPO_URL="https://github.com/your/fork.git"
export CVS_BRANCH="your-branch"

# Override CVS directory
export CVS_DIR="/path/to/cvs"

# Override test settings
export TEST_PATH="./tests/your_test.py"
export CONFIG_FILE="config_custom.yaml"  # Supports .json and .yaml
export LOG_FILE="/tmp/your_test.log"

# Enable debug mode (verbose logging)
export DEBUG_MODE=1

sbatch sbatch/default.sbatch
```

### Running Aorta with Custom Config

```bash
# Copy and modify config
cp config_aorta.yaml my_aorta_config.yaml
# Edit my_aorta_config.yaml (set aorta_path, etc.)

# Run with custom config
export CONFIG_FILE="my_aorta_config.yaml"
sbatch sbatch/aorta.sbatch
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# One-time debug mode
DEBUG_MODE=1 sbatch sbatch/default.sbatch

# Or export for multiple runs
export DEBUG_MODE=1
sbatch sbatch/default.sbatch
```

Debug mode provides:
- Detailed execution traces (`set -x`)
- Debug-level log messages
- Directory and path verification
- Step-by-step progress indicators
- Exit code reporting

**Production use:** Keep `DEBUG_MODE=0` (default) for clean logs

## Library Modules

### `lib/common.sh`
- Logging functions (`log_info`, `log_error`, `log_success`)
- Cleanup functions
- Error handling and traps

### `lib/git_ops.sh`
- Clone CVS repository
- Update to latest branch
- Idempotent git operations

### `lib/cluster_config.sh`
- SSH key detection
- Node IP resolution
- Generate `cluster.json` from SLURM allocation

### `lib/python_env.sh`
- UV-based execution
- System pytest execution
- Virtual environment management
- Automatic fallback mechanism

## SBATCH Configuration

Edit `sbatch/default.sbatch` to modify SLURM parameters:
- Node count
- Tasks per node
- GPUs per node
- Partition
- Reservation
- Account

## Generated Files

The following files are generated during execution and cleaned up automatically:

- `cluster.json` - Generated from SLURM allocation
- `.venv/` - Python virtual environment (if created)

## Development

# Planned Features

- TODO: Dynamic benchmark configuration generation
    - Level 1: collectives [All_reduce, Reduce_gather, Reduce_scatter, AlltoAll], dtypes [float, bloat16]
    - Level 2: check functionality of all collectives & dtypes
    - Level 3: sweep across algo, proto combination matrix
    - Level 4: user defined sweeps
- TODO: Bats unit tests for the individual modules
- TODO: Bats integration tests for the individual modules 

### Adding New SBATCH Configurations

Create additional files in `sbatch/`:

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

### Adding New Library Functions

1. Create or edit files in `lib/`
2. Source them in `run.sh`
3. Use functions in the main workflow

### Testing Individual Modules

```bash
# Source a module and test its functions
source lib/common.sh
log_info "Testing logging"
```

## Troubleshooting

### Check SLURM Output

```bash
# Check output
cat /scratch/users/speriasw/tmp/slurm/cvs-benchmark-<JOB_ID>.out

# Check errors
cat /scratch/users/speriasw/tmp/slurm/cvs-benchmark-<JOB_ID>.err
```

### Enable Debug Mode

Submit with verbose logging:

```bash
DEBUG_MODE=1 sbatch sbatch/default.sbatch
```

This enables:
- Command tracing (`set -x`)
- Detailed debug messages
- Path verification steps
- Exit code reporting

### Manual Cleanup

If cleanup didn't run:

```bash
rm -f cluster.json
rm -rf /path/to/cvs/.venv
```

## Author

surya.periaswamy@amd.com

## License

Internal use only.

