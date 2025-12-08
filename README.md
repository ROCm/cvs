# Cluster Validation Suite

> [!NOTE]
>
> The published Cluster Validation Suite documentation is available [here](https://rocm.docs.amd.com/projects/cvs/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

CVS is a collection of test scripts that validate AMD AI clusters end-to-end, from single-node burn-in health tests to cluster-wide distributed training and inference tests. CVS can be used by AMD customers to verify the health of the cluster as a whole which includes verifying the GPU/CPU node health, Host OS configuration checks, NIC validations, etc. The Cluster Validation Suite includes the following test categories:

1. Platform Tests - Host OS config checks, BIOS checks, Firmware/Driver checks, Network config checks.
2. Burn in Health Tests - AGFHC, Transferbench, RocBLAS, rocHPL, Single node RCCL
3. Network Tests - Ping checks, Multi node RCCL validations for different collectives
4. Distributed Training Tests - Run Llama 70B and 405B model distributed trainings with JAX and Megatron frameworks.
5. Distributed Inference Tests - Work in Progress

CVS leverages the PyTest open source framework to run the tests and generate reports and can be launched from a head-node or any Linux management station which has connectivity to the cluster nodes via SSH. The single node tests are run in parallel cluster wide using the parallel-ssh open source python modules to optimize the time for running them. Currently CVS has been validated only on Ubuntu based Linux distro clusters.

CVS Repository is organized as the following directories

| Directory | Summary |
|-|-|
| `input/` | This is a collection of the input JSON files that are provided to the pytest scripts using the 2 arguments --cluster_file and the `--config_file`. The cluster_file is a JSON file which captures all the aspects of the cluster testbed, things like the IP address/hostnames, username, keyfile, etc. We avoid putting a lot of other information like Linux network devices names or rdma device names to keep it user friendly and auto-discover them. |
| `lib/` | This is a collection of python modules which offer a wide range of utility functions and can be reused in other python scripts as well.|
| `tests/` | Contains the pytest test suites, which call into the shared library code under `lib/`. Each subdirectory corresponds to a test domain (health, rccl, training, etc.). |
| `utils/` | This is a collection of standalone scripts which can be run natively without pytest and offer different utility functions.|

## How to install

To install, simply clone the CVS repository:

```sh
git clone https://github.com/rocm/cvs
```

It is recommended to run CVS from a runner machine - Ubuntu VM or bare metal and in the absence of a dedicated runner machine, you can also install and run it from the first host in your cluster. It is recommended to run from a dedicated runner machine. Some errors during burn-in health tests can cause a host to reboot, which will abort the test and prevent the report from being generated.

## Setting up your environment for CVS

Enter your venv environment and from there install the required python packages using the following commands

```sh
python -m venv myenv
source myenv/bin/activate
cd cvs
pip install -r requirements.txt 
```

## How to run CVS Tests

All the pytest scripts from `cvs/tests` folder must be run from the cvs root folder as shown below as the system lib paths have been set accordingly. Run the following from the CVS repo directory root.

```sh
pytest -vvv \
  -s \
  --cluster_file ./input/mi325_cluster.json \
  --config_file ./input/rccl/rccl_config.json \
  --html=/var/www/html/cvs/rccl_test_report.html \
  --capture=tee-sys \
  --self-contained-html \
  -log-file=/tmp/rccl_test.log \
  ./tests/rccl/rccl_multinode_cvs.py
```

To configure the tests, you can use the following arguments:

| Flag | Description |
|-|-|
| -log-file | The text log file where all the python logger outputs are captured |
| -s | Disable output capturing (stdout and stderr). Allows print/log output to stream to console. |
| --cluster_file | Location of the cluster file which has the details of the cluster - IPs, access details |
| --config_file | This is the configuration file used for the test. Depending on the test suite that is being run, the configuration will vary like the rccl test in the above case uses the `rccl_config.json` file which captures all relevant information related to RCCL like the environment variables, configuration options etc. The sample input files are organized as sub-directories under the cvs/input folder in similar fashion as the pytests |
| --html | This is the output HTML report file that will be generated by pytest at the end of the script completion. It will have a summary of the number of test cases that have passed/failed etc and one can also navigate the logs directly in the browser from this report |
| --capture=tee-sys | capture all std.out and std.err writes from your tests |
| --self-contained-html | Generate as a single HTML report including the styling and embedded images for all test cases |

See more configuration options by running `pytest -h`.

You can also create a wrapper shell script to run multiple test suites one after the other by putting the different pytest run commands in a bash script as described in the README under the `cvs/tests/health` folder.

FOR MORE DETAILS on running individual test suites and details on the configuration files, please refer the individual folder `README.md` files.
