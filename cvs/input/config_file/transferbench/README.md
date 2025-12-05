# TransferBench Configuration Files

This directory contains configuration files for TransferBench testing in CVS.

## Config Files

### `transferbench_singlenode_config.json`
- **Purpose**: Single-node TransferBench performance testing
- **Supported Executors**:
  - `G0->G0->G1`: GPU-to-GPU via GPU executor (intra-GPU)
  - `C0->D0->G0`: CPU-to-GPU via DMA executor
  - `F0->F0->F1`: Memory type F transfers (local)
  - `C0->C0->C1`: CPU-to-CPU transfers (inter-socket)
- **Excluded**: NIC executor (`G0->N0->G1`) - requires multi-node setup
- **Usage**: `pytest transferbench_singlenode_cvs.py --config_file transferbench_singlenode_config.json`

### `transferbench_multinode_config.json`
- **Purpose**: Multi-node TransferBench performance testing
- **Supported Executors**: All single-node executors plus:
  - `G0->N0->G1`: GPU-to-GPU via NIC executor (networking)
- **Usage**: `pytest transferbench_multinode_cvs.py --config_file transferbench_multinode_config.json`

## Configuration Parameters

Both config files support:
- `transfer_scale`: Number of parallel transfers [1, 2, 4, 8]
- `executor_scale`: Number of execution units [1, 2, 4]
- `source_executor_destination`: Memory transfer patterns (varies by config)
- `start_num_bytes`/`end_num_bytes`/`step_function`: Message size sweep parameters
- Various runtime parameters (iterations, paths, verification settings)

## Test Execution

```bash
# Single-node testing
pytest cvs/tests/transferbench/transferbench_singlenode_cvs.py \
  --config_file cvs/input/config_file/transferbench/transferbench_singlenode_config.json \
  --cluster_file <cluster_config.json>

# Multi-node testing
pytest cvs/tests/transferbench/transferbench_multinode_cvs.py \
  --config_file cvs/input/config_file/transferbench/transferbench_multinode_config.json \
  --cluster_file <cluster_config.json>
```</content>
<parameter name="filePath">/home/ichristo/github/ROCm/cvs/cvs/input/config_file/transferbench/README.md