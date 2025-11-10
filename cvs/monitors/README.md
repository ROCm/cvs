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


```
(myenv) [ubuntu-host]~/cvs:(main)$
(myenv) [ubuntu-host]~/cvs:(main)$python3 ./utils/check_cluster_health.py -h
usage: check_cluster_health.py [-h] --hosts_file HOSTS_FILE --username USERNAME (--password PASSWORD | --key_file KEY_FILE)
                               [--iterations ITERATIONS] [--time_between_iters TIME_BETWEEN_ITERS] [--report_file REPORT_FILE]

Check Cluster Health

options:
  -h, --help            show this help message and exit
  --hosts_file HOSTS_FILE
                        File name with list of IP address one per line
  --username USERNAME   Username to ssh to the hosts
  --password PASSWORD   Password for username
  --key_file KEY_FILE   Private Keyfile for username
  --iterations ITERATIONS
                        Number of iterations to run the checks
  --time_between_iters TIME_BETWEEN_ITERS
                        Time duration to sleep between iterations ..
  --report_file REPORT_FILE
(myenv) [ubuntu-host]~/cvs:(main)$
(myenv) [ubuntu-host]~/cvs:(main)$
(myenv) [ubuntu-host]~/cvs/utils:(main)$
(myenv) [ubuntu-host]~/cvs/utils:(main)$python3 ./utils/check_cluster_health.py --hosts_file /home/user1/hosts_file.txt --username user1  --key_file /home/user1/.ssh/id_rsa --iterations 2

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






