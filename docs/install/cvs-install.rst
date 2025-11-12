.. meta::
  :description: Component install 
  :keywords: Component, ROCm, install

*******************************************
Cluster Validation Suite (CVS) installation
*******************************************

System requirements
===================

CVS supports these GPUs:

- AMD Instinct™ MI325X
- AMD Instinct™ MI300X	

CVS supports these Linux distributions:

.. list-table::
   :widths: 3 3 4 4
   :header-rows: 1

   * - Operating system
     - Kernel
     - ROCm version (tested on)
     - Python version (tested on)
   * - Ubuntu 24.04.3
     - 6.8 [GA], 6.14 [HWE]
     - 7.0.2
     - 3.10
   * - Ubuntu 22.04.5
     - 5.15 [GA], 6.8 [HWE]
     - 7.0.2
     - 3.10

Install CVS
===========

Run CVS from a node (head node), such as an Ubuntu virtual machine/bare metal with or without GPU in that node. 
It's recommended to run CVS from head node that is not a part of the test cluster. 
This is to avoid loss of data if the node requires a reboot (such as during a system failure).

1. Git clone the package:

   .. code:: bash

     git clone https://github.com/ROCm/cvs

   The CVS GitHub repository is organized in these directories:

   -	``tests``: This folder contains the PyTest scripts which internally call the library functions under the ``./lib`` directory. They're in native Python and can be invoked from any Python scripts for reusability. The ``tests`` directory contains a subfolder based on the nature of the tests, such as health, RCCL, training, and more.
   -	``lib``: This is a collection of Python modules with utility functions that can be reused in other Python scripts.
   -	``input``: This is a collection of the input JSON files that are provided to the PyTest scripts using the two arguments ``--cluster_file`` and the ``--config_file``. The ``--cluster_file`` is a JSON file which captures all the aspects of the cluster test bed, such as the IP address/hostnames, username, keyfile, and more. 
   -	``utils``: This is a collection of standalone scripts that can be run natively without PyTest. They offer different utility functions.

2. Navigate to the extracted directory and run the installation script:

   .. code:: bash

     cd cvs

3. Set the environment:

   .. code:: b3ash

     python3 -m venv myenv

     source myenv/bin/activate

     pip3 install -r requirements.txt  


Configure the CVS cluster file
==============================

The cluster file is a JSON file containing the cluster's IP addresses. You must configure the cluster file before you run any CVS tests. 

1. Go to ``cvs/input/cluster_file/cluster.json`` in your cloned repo.
2. Edit the management IP (``"mgmt_ip"``) and node dictionary (``"node_dict"``) with the list of IPs of the available cluster.
3. Ensure the user-id (``"{user-id}"``) and ``priv_key_file`` match.

Here's a code snippet of the ``cluster.json`` file for reference:

.. dropdown:: ``cluster.json``

  .. code:: json

    {
        "_user_comment": " user-id will be resolved to current username in runtime. You can also change to your user-id here.",
        "username": "{user-id}",
    
        "_key_comment": " Change <priv_key_file> to your private key if it is different.",
        "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
    
        "_node_comment": " Change to your node IPs. The Public IPs of the nodes will be the keys of the node_dict",
        "_vpc_comment": "If your cluster has a dedicated VPC IP that is reachable from other nodes in the cluster, set it to that (or) else set the same as the main host IP/Name",
    
        "head_node_dict":
        {
            "mgmt_ip": "{xx.xx.xx.xx|hostname}"
        },
        "node_dict":
        {
            "{xx.xx.xx.xx|hostname}":
            {
                "bmc_ip": "NA",
                "vpc_ip": "{xx.xx.xx.xx|hostname}"
            },
            "{xx.xx.xx.xx|hostname}":
            {
                "bmc_ip": "NA",
                "vpc_ip": "{xx.xx.xx.xx|hostname}"
            },
            "{xx.xx.xx.xx|hostname}":
            {
                "bmc_ip": "NA",
                "vpc_ip": "{xx.xx.xx.xx|hostname}"
            }
    
        }
    
    
    }

Set up your tests
=================

There are JSON configuration files for each CVS test. You must configure the JSON file for each test you want to run in CVS.
The test configuration files are in the ``cvs/input/config_file`` directory of the cloned repo. 

.. tip::

  See :doc:`Test configuration files <../reference/configuration-files/configure-config>` for code snippets and parameters of each configuration file.

Follow these instructions for each test you'd like to conduct.

Platform
--------

In the ``cvs/input/config_file/platform/host_config.json`` file, modify these parameters to suit your use case: 

- ``os_version``
- ``kernel_version``
- ``rocm_version``
- ``bios_version``

Health
------

In the ``cvs/input/config_file/health/mi300_health_config.json`` file, edit the paths to your desired location in these parameters:

- Under ``agfhc``: 

  - ``path``
  - ``package_tar_ball``
  - ``install_dir``

-  Under ``transferbench``: 

   - ``example_tests_path`` 
   - ``git_install_path`` 

- Under ``rvs``:

  - ``git_install_path`` 

InfiniBand (IB Perf)
--------------------

In the ``cvs/input/config_file/ibperf/ibperf_config.json`` file, update the ``install_dir`` parameter to your desired location.
Change any other parameters in the configuration file relevant to your testing requirements.

 
ROCm Communication Collectives Library (RCCL)
---------------------------------------------

In the ``cvs/input/config_file/rccl/rccl_config.json`` and ``cvs/input/config_file/rccl/single_node_mi355_rccl.json`` files, change the directory path to your desired location in these variables: 

- ``rccl_dir``
- ``rccl_tests_dir``
- ``mpi_dir``
- ``mpi_path_var`` 
- ``rccl_path_var``
- ``rocm_path_var``

JAX / Megatron training configuration files
--------------------------------

Parameters with the ``<changeme>`` value must have that value modified to your specifications.
Change any other parameters in the configuration file relevant to your testing requirements. 



