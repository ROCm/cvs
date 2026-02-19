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

Prerequisites
-------------

- Python 3.9 or later
- Git

Debian/Ubuntu Systems
~~~~~~~~~~~~~~~~~~~~~~

On Debian and Ubuntu distributions, the ``venv`` module is not included in the base Python package. Install it before proceeding:

.. code:: bash

  sudo apt install python3-venv

Method 1: Quick installation using Makefile (Recommended)
----------------------------------------------------------

This is the quickest way to install CVS from source.

1. Clone the repository and install using make:

   .. code:: bash

     git clone https://github.com/ROCm/cvs
     cd cvs
     make install

   This will automatically:

   - Build the source distribution
   - Create a virtual environment in ``.cvs_venv/``
   - Install CVS in the virtual environment

   The CVS GitHub repository is organized in these directories:

   -   ``tests``: This folder contains the PyTest scripts which internally call the library functions under the ``./lib`` directory. They're in native Python and can be invoked from any Python scripts for reusability. The ``tests`` directory contains a subfolder based on the nature of the tests, such as health, RCCL, training, and more.
   -   ``lib``: This is a collection of Python modules with utility functions that can be reused in other Python scripts.
   -   ``input``: This is a collection of the input JSON files that are provided to the PyTest scripts using the two arguments ``--cluster_file`` and the ``--config_file``. The ``--cluster_file`` is a JSON file which captures all the aspects of the cluster test bed, such as the IP address/hostnames, username, keyfile, and more.
   -   ``utils``: This is a collection of standalone scripts that can be run natively without PyTest. They offer different utility functions.

2. Activate the virtual environment:

   .. code:: bash

     source .cvs_venv/bin/activate

After activation, verify CVS is available:

   .. code:: bash

     cvs list

If you see a list of available test suites, CVS is installed correctly.

Method 2: Manual installation
------------------------------

For users who want to install CVS in a custom virtual environment:

1. Clone the repository:

   .. code:: bash

     git clone https://github.com/ROCm/cvs
     cd cvs

2. Build CVS:

   .. code:: bash

     python setup.py sdist

3. Create and activate a Python virtual environment, then install CVS:

   .. code:: bash

     python3 -m venv cvs_env  # or any custom name
     source cvs_env/bin/activate
     pip install dist/cvs*.tar.gz

This method gives you more control over the virtual environment name and location.

After installation, verify CVS is available:

   .. code:: bash

     cvs list

If you see a list of available test suites, CVS is installed correctly.


Configure the CVS cluster file
==============================

The cluster file is a JSON file containing the cluster's IP addresses. You must configure the cluster file before you run any CVS tests. 

1. Copy the cluster file template to your desired location:

   .. code:: bash

     cvs copy-config cluster.json --output ~/my_cluster.json

2. Edit the management IP (``"mgmt_ip"``) and node dictionary (``"node_dict"``) with the list of IPs of your cluster.
3. Ensure the user-id (``"{user-id}"``) and ``priv_key_file`` match your setup.

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

You can list all available configuration files using:

.. code:: bash

  cvs copy-config --list

.. tip::

  See :doc:`Test configuration files <../reference/configuration-files/configure-config>` for code snippets and parameters of each configuration file.

For each test you'd like to conduct, copy the relevant configuration file and modify it for your use case.

Platform
--------

1. Copy the platform configuration file:

   .. code:: bash

     cvs copy-config platform/host_config.json --output ~/my_host_config.json

2. Edit the file and modify these parameters to suit your use case:

   - ``os_version``
   - ``kernel_version``
   - ``rocm_version``
   - ``bios_version``

Health
------

1. Copy the health configuration file:

   .. code:: bash

     cvs copy-config health/mi300_health_config.json --output ~/my_health_config.json

2. Edit the file and modify the paths to your desired location in these parameters:

   - Under ``agfhc``: 

     - ``path``
     - ``package_tar_ball``
     - ``install_dir``

   - Under ``transferbench``: 

     - ``example_tests_path`` 
     - ``git_install_path`` 

   - Under ``rvs``:

     - ``git_install_path`` 

InfiniBand (IB Perf)
--------------------

1. Copy the IB performance configuration file:

   .. code:: bash

     cvs copy-config ibperf/ibperf_config.json --output ~/my_ibperf_config.json

2. Edit the file and update the ``install_dir`` parameter to your desired location.
3. Change any other parameters relevant to your testing requirements.

 
ROCm Communication Collectives Library (RCCL)
---------------------------------------------

1. Copy the RCCL configuration file(s) you need:

   .. code:: bash

     cvs copy-config rccl/rccl_config.json --output ~/my_rccl_config.json
     # Or for single node:
     cvs copy-config rccl/single_node_mi355_rccl.json --output ~/my_single_node_rccl.json

2. Edit the file(s) and change the directory paths to your desired location in these variables:

   - ``rccl_dir``
   - ``rccl_tests_dir``
   - ``mpi_dir``
   - ``mpi_path_var`` 
   - ``rccl_path_var``
   - ``rocm_path_var``

JAX / Megatron training configuration files
--------------------------------------------

1. List available training configuration files:

   .. code:: bash

     cvs copy-config training --list

2. Copy the configuration file you need:

   .. code:: bash

     cvs copy-config training/jax/mi300x_jax_llama3_1_70b_distributed.json --output ~/my_training_config.json
     # Or for Megatron:
     cvs copy-config training/megatron/mi3xx_megatron_llama_distributed.json --output ~/my_megatron_config.json

3. Edit the file and modify parameters with the ``<changeme>`` value to your specifications.
4. Change any other parameters relevant to your testing requirements. 



