.. meta::
  :description: Component install 
  :keywords: Component, ROCm, install

*******************************************
Cluster Validation Suite (CVS) installation
*******************************************

Before you can use CVS, you must clone the `CVS GitHub repository <https://github.com/ROCm/cvs>`_ and set your environment. 
Then, you need to configure the JSON cluster file and the script test configuration files according to your use case.

System requirements
===================

CVS supports these GPUs:

- AMD Instinct™ MI325X
- AMD Instinct™ MI300X	

CVS supports these Linux distributions:

- Ubuntu 24.04.2

  - Kernel: 6.8 [GA], 6.14 [HWE]

- Ubuntu 22.04.5

  - Kernel: 5.15 [GA], 6.8 [HWE]

Install CVS
===========

You should run CVS from a runner machine such as an Ubuntu virtual machine or bare metal. Alternatively, you can install and run CVS from the first host in your cluster. 
It's recommended to run CVS from a dedicated runner machine to avoid loss of data if the host requires a reboot (such as during a system failure).

1. Git clone the package:

   .. code:: bash

     adc@def:~/ git clone https://github.com/ROCm/cvs

   The CVS GitHub repository is organized in these directories:

   -	``tests``: This folder has the PyTest scripts which internally call the library functions under the ``./lib`` directory. They're in native Python and can be invoked from any Python scripts for reusability. The ``tests`` directory has a subfolder based on the nature of the tests like health, RCCL, training, and more.
   -	``lib``: This is a collection of Python modules with utility functions that can be reused in other Python scripts.
   -	``input``: This is a collection of the input JSON files that are provided to the PyTest scripts using the two arguments ``--cluster_file`` and the ``--config_file``. The ``--cluster_file`` is a JSON file which captures all the aspects of the cluster test bed, such as the IP address/hostnames, username, keyfile, and more. 
   -	``utils``: This is a collection of standalone scripts which can be run natively without PyTest. They offer different utility functions.

2. Navigate to the extracted directory and run the installation script:

   .. code:: bash

     adc@def:~/cd cvs

3. Set the environment:

   .. code:: bash

     adc@def:~/cvs $  python3 -m venv myenv

     adc@def:~/cvs $  source myenv/bin/activate

     adc@def:~/cvs $  pip3 install -r requirements.txt  


Configure the CVS cluster file
==============================

The cluster file is a JSON file which has the details about the IPs of the cluster. You must configure the cluster file before you run any CVS tests. 

1. Go to ``cvs/input/cluster_file/cluster.json`` in your cloned repo.
2. Edit the management IP and node dictionary with the list of IPs of the available cluster:

Here's a code snippet of the ``cluster.json`` file for reference:

.. code:: json

  {
    "_comment": "change to your user-id and your private key and your node IPs. The Public IPs of the nodes will be the keys of the node_dict",
    "username": "username",
    "priv_key_file": "/home/abc/.ssh/id_rsa",
    "head_node_dict":
    {
        "mgmt_ip": "10.10.10.1"
    },
    "node_dict":
    {
        "10.10.10.1":
        {
            "bmc_ip": "NA",
            "vpc_ip": "10.10.10.1"
        },
        "10.10.10.2":
        {
            "bmc_ip": "NA",
            "vpc_ip": "10.10.10.2"
        }

    },

    "bmc_mapping_dict":
    {
    },

    "backend_nw_dict":
    {
    },
  
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
 
-  Under ``rocblas``, ``rochpl``, and ``babelstream``: 
  
   - ``path`` 
   - ``git_install_path``

ROCm Communication Collectives Library (RCCL)
---------------------------------------------

In the ``cvs/input/config_file/rccl/rccl_config.json`` file, change the directory path to your desired location in these variables: 

- ``rccl_dir``
- ``rccl_tests_dir``
- ``mpi_dir``
- ``mpi_path_var`` 
- ``rccl_path_var``
- ``rocm_path_var``

InfiniBand (IB Perf)
--------------------

Change the parameters in the configuration file as needed.
