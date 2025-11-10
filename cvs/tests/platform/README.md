Host OS config checks, BIOS checks, Firmware/Driver checks, Network config checks
The host check scripts, which can validate anything on host side like new model load balancing enable, PCIE checks, kernel version, rocm version and so on.


# How to run the tests

This Pytest script can be run in the following fashion (for the details on arguments and their purpose, please refer the main README under the CVS parent folder

```
(myenv) [user@host]~/cvs:(main)$
(myenv) [user@host]~/cvs:(main)$pwd
/home/user/cvs
(myenv) [user@host]~/cvs:(main)$pytest -vvv --log-file=/tmp/test.log -s ./tests/platform/host_configs_cvs.py --cluster_file input/cluster_file/cluster.json  --config_file input/config_file/platform/host_config.json --html=/var/www/html/cvs/rochpl.html --capture=tee-sys --self-contained-html

```
