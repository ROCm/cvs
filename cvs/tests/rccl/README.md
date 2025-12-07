RCCL (ROCm Communication Collectives Library) tests are comprehensive benchmarks that validate distributed GPU communication performance across AMD GPU clusters. These tests ensure optimal performance for AI training, HPC workloads, and distributed computing

# How to run the tests

This Pytest script can be run in the following fashion (for the details on arguments and their purpose, please refer the main README under the CVS parent folder

```
(myenv) [user@host]~/cvs:(main)$
(myenv) [user@host]~/cvs:(main)$pwd
/home/user/cvs/cvs
(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/rccl/rccl_multinode_cvs.py --cluster_file input/cluster_file/cluster.json  --config_file input/config_file/rccl/rccl_config.json --html=/var/www/html/cvs/rccl.html --capture=tee-sys --self-contained-html

```
