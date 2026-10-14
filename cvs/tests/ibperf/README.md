IB (InfiniBand) perf and latency tests are tools used to measure network performance, with perf tests  measuring throughput (bandwidth) and latency tests measuring delay. Perf tests, such as ib_write_bw, evaluate the maximum data transfer rate under different message sizes, while latency tests, like ib_send_lat, measure the time it takes for a message to travel between two nodes, often reporting results like minimum, median, and maximum latency. 

Following are the currently supported test suites

Before launching perftest, ibperf validates `gid_index` against `gid_type` (default `RoCE v2`) on every node and NIC it uses. Set `gid_index` to `auto`, leave it empty, or remove it to select the lowest common usable index: an IPv4-mapped GID of that type on RoCE, or the lowest populated GID (normally 0) on InfiniBand. Use `gid_type: "any"` to disable type checking; InfiniBand link layers are exempt.

1. IB Bandwidth 
2. IB Latency


# How to run the tests

This Pytest script can be run in the following fashion (for the details on arguments and their purpose, please refer the main README under the CVS parent folder

In the config file, cvs/input/config_file/ibperf/ibperf_config.json, change the value of parameter "install_dir": "/home/{user-id}/" to the desired location. Else {user-id} will be resolved as the current username at runtime.

Perftest must be built with ROCm DMA-BUF support. The tests probe the perftest binary on every node before launching. The `require_dmabuf` config setting defaults to `True` and fails the test if any node lacks support; set it to `False` to run without DMA-BUF on all nodes.

Each NIC uses its PCIe-local GPU. The tests find the HIP ordinal for `--use_rocm` by matching the GPU BDF against KFD topology. Setting `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, or `GPU_DEVICE_ORDINAL` in cluster `env_vars` makes the ibperf tests fail.


```
(myenv) [user@host]~/cvs:(main)$
(myenv) [user@host]~/cvs:(main)$pwd
/home/user/cvs/cvs
(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/ibperf/install_ibperf_tools.py --cluster_file input/cluster_file/cluster.json --config_file input/config_file/ibperf/ibperf_config.json --html=/var/www/html/cvs/ib.html --capture=tee-sys --self-contained-html

(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/ibperf/ib_perf_bw_test.py --cluster_file input/cluster_file/cluster.json --config_file input/config_file/ibperf/ibperf_config.json --html=/var/www/html/cvs/ib.html --capture=tee-sys --self-contained-html

```
