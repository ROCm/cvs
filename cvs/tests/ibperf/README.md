IB (InfiniBand) perf and latency tests are tools used to measure network performance, with perf tests  measuring throughput (bandwidth) and latency tests measuring delay. Perf tests, such as ib_write_bw, evaluate the maximum data transfer rate under different message sizes, while latency tests, like ib_send_lat, measure the time it takes for a message to travel between two nodes, often reporting results like minimum, median, and maximum latency. 

Following are the currently supported test suites

Before launching perftest, ibperf validates `gid_index` against `gid_type` (default `RoCE v2`) on every node and NIC it uses. Set `gid_index` to `auto`, leave it empty, or remove it to select the lowest common IPv4-mapped GID of that type. Use `gid_type: "any"` to disable type checking; InfiniBand link layers are exempt.

1. IB Bandwidth 
2. IB Latency


# How to run the tests

This Pytest script can be run in the following fashion (for the details on arguments and their purpose, please refer the main README under the CVS parent folder

In the config file, cvs/input/config_file/ibperf/ibperf_config.json, change the value of parameter "install_dir": "/home/{user-id}/" to the desired location. Else {user-id} will be resolved as the current username at runtime.


```
(myenv) [user@host]~/cvs:(main)$
(myenv) [user@host]~/cvs:(main)$pwd
/home/user/cvs/cvs
(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/ibperf/install_ibperf_tools.py --cluster_file input/cluster_file/cluster.json --config_file input/config_file/ibperf/ibperf_config.json --html=/var/www/html/cvs/ib.html --capture=tee-sys --self-contained-html

(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/ibperf/ib_perf_bw_test.py --cluster_file input/cluster_file/cluster.json --config_file input/config_file/ibperf/ibperf_config.json --html=/var/www/html/cvs/ib.html --capture=tee-sys --self-contained-html

```
