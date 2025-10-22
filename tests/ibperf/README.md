IB (InfiniBand) perf and latency tests are tools used to measure network performance, with perf tests  measuring throughput (bandwidth) and latency tests measuring delay. Perf tests, such as ib_write_bw, evaluate the maximum data transfer rate under different message sizes, while latency tests, like ib_send_lat, measure the time it takes for a message to travel between two nodes, often reporting results like minimum, median, and maximum latency. 

Following are the currently supported test suites

1. IB Bandwidth 
2. IB Latency


# How to run the tests

This Pytest script can be run in the following fashion (for the details on arguments and their purpose, please refer the main README under the CVS parent folder

```
(myenv) [user@host]~/cvs:(main)$
(myenv) [user@host]~/cvs:(main)$pwd
/home/user/cvs
(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/ibperf/ib_perf_bw_test.py --cluster_file input/cluster_file/cluster.json --config_file input/config_file/ibperf/ibperf_config.json --html=/var/www/html/cvs/ib.html --capture=tee-sys --self-contained-html

```
