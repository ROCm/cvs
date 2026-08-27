.. meta::
  :description: Run Megatron Llama training benchmarks
  :keywords: CVS, megatron

***********************
Megatron training tests
***********************

.. _megatron-set-up-config:

Set up config
=============

1. List available training configuration files:

   .. code:: bash

     cvs config list training

2. Copy the configuration file you need, for example:

   .. code:: bash

     cvs config copy training/megatron/mi3xx_megatron_llama_distributed.json --output ~/cvs_workspace/training/megatron/mi3xx_megatron_llama_distributed.json

3. Replace every ``<changeme>`` with cluster-specific values (container image, checkpoint paths, NCCL/RDMA fields on distributed configs).
4. Change any other parameters relevant to your testing requirements.

Full parameter list: :doc:`/reference/configuration-files/training/megatron`.

.. _megatron-run-tests:

Run tests
=========

Megatron training test scripts
------------------------------

You can list all available Megatron training test cases using the CLI:

.. code:: bash

  cvs list megatron_llama3_1_70b_distributed

.. code:: text

  Available tests in megatron_llama3_1_70b_distributed:
    - test_cleanup_stale_containers
    - test_disable_firewall  
    - test_launch_megatron_containers
    - test_llama_3_1_fp8_single_node

.. code:: bash

  cvs list megatron_llama3_1_70b_single

.. code:: text

  Available tests in megatron_llama3_1_70b_single:
    - test_cleanup_stale_containers
    - test_launch_megatron_containers
    - test_llama_3_1_fp8_single_node

.. code:: bash

  cvs list megatron_llama3_1_8b_distributed

.. code:: text

  Available tests in megatron_llama3_1_8b_distributed:
    - test_cleanup_stale_containers
    - test_disable_firewall
    - test_launch_megatron_containers
    - test_llama_3_1_fp8_single_node
Use these scripts to run the Megatron tests.

Single Node 8b MI3XX
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_llama3_1_8b_single --cluster_file input/cluster_file/cluster.json --config_file input/config_file/training/megatron/mi3xx_megatron_llama_single.json --html=/var/www/html/cvs/megatron.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

Single Node 8b MI35X
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_llama3_1_8b_single --cluster_file input/cluster_file/cluster.json --config_file input/config_file/training/megatron/mi35x_megatron_llama_single.json --html=/var/www/html/cvs/megatron.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

Single Node 70b MI3XX
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_llama3_1_70b_single --cluster_file input/cluster_file/cluster.json --config_file input/config_file/training/megatron/mi3xx_megatron_llama_single.json --html=/var/www/html/cvs/megatron.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

Single Node 70b MI35X
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_llama3_1_70b_single --cluster_file input/cluster_file/cluster.json --config_file input/config_file/training/megatron/mi35x_megatron_llama_single.json --html=/var/www/html/cvs/megatron.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

Distributed 8b
~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_llama3_1_8b_distributed --cluster_file input/cluster_file/cluster.json --config_file input/config_file/training/megatron/mi3xx_megatron_llama_distributed.json --html=/var/www/html/cvs/megatron.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

Distributed 70b
~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_llama3_1_70b_distributed --cluster_file input/cluster_file/cluster.json --config_file input/config_file/training/megatron/mi3xx_megatron_llama_distributed.json --html=/var/www/html/cvs/megatron.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s
