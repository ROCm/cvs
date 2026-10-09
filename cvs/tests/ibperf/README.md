IB (InfiniBand) perf and latency tests are tools used to measure network performance, with perf tests  measuring throughput (bandwidth) and latency tests measuring delay. Perf tests, such as ib_write_bw, evaluate the maximum data transfer rate under different message sizes, while latency tests, like ib_send_lat, measure the time it takes for a message to travel between two nodes, often reporting results like minimum, median, and maximum latency. 

Following are the currently supported test suites

1. IB Bandwidth 
2. IB Latency


# How to run the tests

This Pytest script can be run in the following fashion (for the details on arguments and their purpose, please refer the main README under the CVS parent folder

In the config file, cvs/input/config_file/ibperf/ibperf_config.json, change the value of parameter "install_dir": "/home/{user-id}/" to the desired location. Else {user-id} will be resolved as the current username at runtime.

Node pairing defaults to `sequential`, using consecutive nodes in cluster-file order. Set `ibperf.pairing_mode` to `inter_vpod` for pairs across vPODs, or `intra_vpod` for pairs within a vPOD. For explicit membership, set `ibperf.vpod_source` to `cluster_file` and add `vpod_id` to every `node_dict` entry. For example, labels A for n1/n2 and B for n3/n4 produce inter-vPOD pairs (n1,n3) and (n2,n4). The default AFM source uses `afmctl show device --json` and applies only within one scale-up domain; use cluster-file labels if accelerator IDs can repeat across domains. The suite still assumes 8 GPUs per node.


```
(myenv) [user@host]~/cvs:(main)$
(myenv) [user@host]~/cvs:(main)$pwd
/home/user/cvs/cvs
(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/ibperf/install_ibperf_tools.py --cluster_file input/cluster_file/cluster.json --config_file input/config_file/ibperf/ibperf_config.json --html=/var/www/html/cvs/ib.html --capture=tee-sys --self-contained-html

(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/ibperf/ib_perf_bw_test.py --cluster_file input/cluster_file/cluster.json --config_file input/config_file/ibperf/ibperf_config.json --html=/var/www/html/cvs/ib.html --capture=tee-sys --self-contained-html

```
