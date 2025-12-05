# TransferBench (ROCm Communication Collectives Library) tests are comprehensive benchmarks that validate distributed GPU communication performance across AMD GPU clusters. These tests ensure optimal performance for AI training, HPC workloads, and distributed computing

# How to run the tests

This Pytest script can be run in the following fashion (for the details on arguments and their purpose, please refer the main README under the CVS parent folder

```
(myenv) [user@host]~/cvs:(main)$
(myenv) [user@host]~/cvs:(main)$pwd
/home/user/cvs
(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/transferbench/transferbench_multinode_cvs.py --cluster_file input/cluster_file/cluster.json  --config_file input/config_file/transferbench/transferbench_config.json --html=/var/www/html/cvs/transferbench.html --capture=tee-sys --self-contained-html

```

# Configuration

TransferBench tests require a configuration file with the following structure:

```json
{
  "transferbench": {
    "rocm_path_var": "/opt/rocm",
    "mpi_dir": "/opt/ompi",
    "mpi_path_var": "/opt/ompi",
    "transferbench_dir": "/path/to/TransferBench",
    "transferbench_path_var": "/path/to/TransferBench",
    "transferbench_bin_dir": "/path/to/TransferBench/bin",
    "num_iterations": 10,
    "num_warmups": 5,
    "transferbench_result_file": "/tmp/transferbench_result_output.json",
    "verify_bus_bw": "False",
    "verify_bw_dip": "True",
    "verify_lat_dip": "True",
    "cluster_snapshot_debug": "False",
    "env_source_script": "None",
    "transfer_configs": [
      "1 4 (G0->G0->G1)",
      "1 1 (C0->D0->G0)"
    ],
    "num_bytes": ["1048576", "0"],
    "results": {}
  }
}
```

# TransferBench vs RCCL

- **TransferBench**: Low-level memory transfer benchmarking (GPU↔GPU, CPU↔GPU, NIC↔GPU)
- **RCCL**: High-level collective communication benchmarking (AllReduce, AllGather, etc.)

TransferBench focuses on the fundamental data movement operations that underlie collective communications.