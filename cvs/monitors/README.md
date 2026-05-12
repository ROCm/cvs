# Utility scripts

This folder comprises of standalone native Python scripts that offer different utility functions.    

## Cluster Health Checker

This is a python script that generates a health report of your cluster by collecting various logs and metrics of the GPU nodes. It can help
identify any hardware failure/degradation signatures like RAS errors, PCIe/XGMI errors, network drop/error counters 

### Cluster Health Checker

This is a python script that generates an overall health report of your cluster by collecting various logs and metrics of the GPU nodes.

It can help identify various

1. Hardware failures/degradation signatures like RAS errors, PCIe/XGMI errors, network drop/error counters.
2. Software failures - By looking for failure signatures in demsg, journlctl logs

This also acts as triaging tool to troubleshoot any performance issues that may be related to the AI infrastructure. It allows you to take s
napshot of all counters (GPU/NIC) while your training/inference workloads are in progress and compare the counters and identify any increment of unexpected counters across all nodes in the cluster to find needle in a haystack.            


### Usage for Cluster Health Checker

The Cluster Health Checker consumes the same `--cluster_file` JSON used by
`cvs run`, `cvs exec`, and `cvs scp` (see [`cvs/input/cluster_file/README.md`](../input/cluster_file/README.md)
for the schema). The node list, SSH username, and private key are all read
from the cluster file - no separate hosts file or credentials need to be
passed on the CLI.

Just like `cvs exec` and `cvs scp`, the cluster file can be supplied either
via `--cluster_file <path>` or by exporting `CLUSTER_FILE=<path>` once per
shell. The env var takes precedence when both are set, so users can
`export CLUSTER_FILE=...` once and then run any `cvs exec`, `cvs scp`,
`cvs monitor check_cluster_health`, etc. without repeating the path.

```
(myenv) [ubuntu-host]~/cvs:(main)$ cvs monitor check_cluster_health --help
usage: cvs monitor check_cluster_health [-h] (--cluster_file CLUSTER_FILE | --hosts_file HOSTS_FILE)
                                        [--username USERNAME] [--password PASSWORD] [--key_file KEY_FILE]
                                        [--iterations ITERATIONS] [--time_between_iters TIME_BETWEEN_ITERS]
                                        [--report_file REPORT_FILE]

Check Cluster Health

options:
  -h, --help            show this help message and exit
  --cluster_file CLUSTER_FILE
                        Path to a CVS cluster JSON file (see cvs/input/cluster_file/cluster.json).
                        Provides node list, username, and SSH key. Recommended.
                        Falls back to the CLUSTER_FILE environment variable when omitted.
  --hosts_file HOSTS_FILE
                        [DEPRECATED] File with one host IP/hostname per line. Use --cluster_file instead.
  --username USERNAME   SSH username (required with --hosts_file)
  --password PASSWORD   SSH password (only valid with --hosts_file)
  --key_file KEY_FILE   SSH private key file (only valid with --hosts_file)
  --iterations ITERATIONS
                        Number of iterations to run the checks
  --time_between_iters TIME_BETWEEN_ITERS
                        Time duration to sleep between iterations
  --report_file REPORT_FILE
                        Output HTML report file path
```

#### Recommended: run with a cluster file

```
cvs monitor check_cluster_health \
    --cluster_file cvs/input/cluster_file/cluster.json \
    --iterations 2
```

Or set `CLUSTER_FILE` once and reuse it across CVS commands:

```
export CLUSTER_FILE=cvs/input/cluster_file/cluster.json
cvs monitor check_cluster_health --iterations 2
cvs exec --cmd "hostname"
cvs scp --file /path/to/file.txt
```

Or directly:

```
python3 ./cvs/monitors/check_cluster_health.py \
    --cluster_file cvs/input/cluster_file/cluster.json \
    --iterations 2
```

#### Deprecated: legacy hosts file

`--hosts_file` is still accepted for backward compatibility but emits a
deprecation warning. Lines starting with `#` and blank lines are ignored.

```
python3 ./cvs/monitors/check_cluster_health.py \
    --hosts_file /home/user1/hosts_file.txt \
    --username user1 \
    --key_file /home/user1/.ssh/id_rsa \
    --iterations 2
```

### Debugging using RDMA Statistics Table

<img width="992" height="687" alt="RDMA_Statistics_Table" src="https://github.com/user-attachments/assets/1efc20c8-5a96-4391-b6c1-7877f78ee901" />

### Debugging PCIe Errors

<img width="1028" height="461" alt="PCIe_NAK_errors_Table" src="https://github.com/user-attachments/assets/243bbd37-3a80-40de-b3e2-b17a060dd5ae" />

### Debugging GPU ECC Errors

<img width="953" height="676" alt="GPU_ECC_Errors_Table" src="https://github.com/user-attachments/assets/8e48e3ed-6565-441e-80c6-8c9224eb21f0" />

### Debugging a bad cable using FEC errors from ethtool stats

<img width="689" height="350" alt="FEC_Errors_bad_cable" src="https://github.com/user-attachments/assets/69aa01eb-7dd9-4d81-97ec-a0e885ebde01" />

### Debugging a Network Congestion using RDMA Snapshot feature

<img width="1193" height="627" alt="Snapshot_rdma_for_debugging" src="https://github.com/user-attachments/assets/dab21c4b-d8c1-4c63-afd5-f2b6dec7d3fe" />

### Scanning Dmesg/Journlctl cluster wide

<img width="953" height="477" alt="Journctl_snapshot" src="https://github.com/user-attachments/assets/e5ed08c0-69f1-4c53-88c3-f829540a841c" />






