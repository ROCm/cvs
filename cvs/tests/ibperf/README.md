# IB Perf (perftest) tests

`install_ibperf_tools` builds perftest with ROCm on the head node. `ib_perf_bw_test` runs the perftest bandwidth (`ib_write_bw`, `ib_read_bw`, `ib_send_bw`) and latency (`ib_write_lat`, `ib_read_lat`, `ib_send_lat`) tests between node pairs, one process per GPU and its PCIe-local NIC.

```bash
cvs config copy ibperf/ibperf_config.json --output ~/cvs_workspace/ibperf/ibperf_config.json
cvs run install_ibperf_tools --cluster_file ~/cvs_workspace/cluster.json --config_file ~/cvs_workspace/ibperf/ibperf_config.json
cvs run ib_perf_bw_test --cluster_file ~/cvs_workspace/cluster.json --config_file ~/cvs_workspace/ibperf/ibperf_config.json
```

Replace `rocm_dir`'s `<changeme>` before running. Prerequisites, pairing, results and troubleshooting are on the [IB Perf how-to](https://rocm.docs.amd.com/projects/cvs/en/latest/how-to/test-suites/network/ib-perf.html). Every config field is in the [IB Perf configuration reference](https://rocm.docs.amd.com/projects/cvs/en/latest/reference/configuration-files/network/ib.html).
