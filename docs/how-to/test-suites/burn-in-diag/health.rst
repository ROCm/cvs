.. meta::
  :description: Run AGFHC, TransferBench, and RVS burn-in tests
  :keywords: CVS, health

**********************
Health (burn-in) tests
**********************

.. _health-set-up-config:

Set up config
=============

1. Copy the health configuration file:

   .. code:: bash

     cvs config copy health/mi300_health_config.json --output ~/cvs_workspace/health/mi300_health_config.json

2. Edit paths and any remaining ``<changeme>`` placeholders:

   - Under ``agfhc``: ``path``, ``package_tar_ball``, ``install_dir``
   - Under ``transferbench``: ``git_install_path``, ``rocm_path`` (or leave ``<changeme>`` to auto-detect)
   - Under ``rvs``: ``git_install_path``, ``path``, ``rocm_path``

Full parameter list: :doc:`/reference/configuration-files/burn-in-diag/health`.

.. _health-run-tests:

Run tests
=========

The burn-in health tests are single node diagnostic tests that validate the hardware and firmware versions' functionality and performance. 
For the performance validation, they use the reference bandwidth or latency numbers provided as part of the input ``config_file`` for the relevant test. 

Use these scripts to run each health test. These CVS test scripts have two parts: installing the functionality and running the tests.

AGFHC
~~~~~

See the `AGFHC (AMD GPU Field Health Check) <https://instinct.docs.amd.com/projects/gpu-operator/en/latest/test/agfhc.html>`_ docs for more information.

You can list all available AGFHC test cases using the CLI:

.. code:: bash

  cvs list agfhc_cvs

.. code:: text

  Available tests in agfhc_cvs:
    - test_agfhc_hbm
    - test_agfhc_hbm1_lvl5
    - test_agfhc_hbm2_lvl5
    - test_agfhc_hbm3_lvl3
    - test_agfhc_dma_all_lvl1
    - test_agfhc_dma_lvl1
    - test_agfhc_gfx_lvl1
    - test_agfhc_pcie_lvl1
    - test_agfhc_pcie_lvl3
    - test_agfhc_xgmi_lvl1
    - test_agfhc_all_perf
    - test_agfhc_all_lvl5

Use these scripts to start the test:

1. Run the installation: 

   .. code:: bash 

     cvs run install_agfhc --cluster_file input/cluster_file/cluster.json --config_file input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/agfhc.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

2. Run the AGFHC test:

   .. code:: bash
    
     cvs run agfhc_cvs --cluster_file input/cluster_file/cluster.json --config_file input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/agfhc.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

3. Run the CSP qualification test:

   .. code:: bash
    
     cvs run csp_qual_agfhc --cluster_file input/cluster_file/cluster.json --config_file input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/agfhc.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

TransferBench
~~~~~~~~~~~~~

See the `TransferBench <https://rocm.docs.amd.com/projects/TransferBench/en/latest/install/install.html#install-transferbench>`_ docs for more information.

You can list all available TransferBench test cases using the CLI:

.. code:: bash

  cvs list transferbench_cvs

.. code:: text

  Available tests in transferbench_cvs:
    - test_transfer_bench_a2a
    - test_transfer_bench_p2p
    - test_transfer_bench_healthcheck
    - test_transfer_bench_a2asweep
    - test_transfer_bench_scaling
    - test_transfer_bench_schmoo

Use these scripts to start the test:

1. Run the installation: 

   .. code:: bash

     cvs run install_transferbench --cluster_file input/cluster_file/cluster.json --config_file input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/transferbench.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

2. Start the TransferBench test:

   .. code:: bash
    
     cvs run transferbench_cvs --cluster_file input/cluster_file/cluster.json --config_file input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/transferbench.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

On **Spur or Slurm managed compute**, launch CVS inside a **job step** (one task per node), for example ``spur run --mpi=none`` or ``srun --mpi=none``. A bare allocation or a ``spur submit`` / ``sbatch`` script that never starts a step is not managed CVS (``SLURM_STEP_ID`` stays unset). ``--cluster_file`` is optional; CVS builds the live cluster file from scheduler hosts and starts one HTTP agent per task.

Set ``transferbench.git_install_path`` and ``transferbench.path`` to a **shared, user-writable** tree. ``transferbench_cvs`` uses ``orch.sudo_prefix()`` around TransferBench; if passwordless sudo is unavailable the binary runs as the job user.

Spur example:

.. code:: bash

  spur run -A <account> -p <partition> \
    -N 1 --gpus-per-node 8 --exclusive -t 04:00:00 --mpi=none \
    bash -lc 'source ~/.cvs_venv/bin/activate &&
      cvs run install_transferbench --config_file <health-config.json> --html <report.html>'

  spur run -A <account> -p <partition> \
    -N 1 --gpus-per-node 8 --exclusive -t 04:00:00 --mpi=none \
    bash -lc 'source ~/.cvs_venv/bin/activate &&
      cvs run transferbench_cvs --config_file <health-config.json> --html <report.html>'


RVS
~~~

See the `ROCm Validation Suite (RVS) <https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/install/installation.html>`_ docs for more information.

You can list all available RVS test cases using the CLI:

.. code:: bash

  cvs list rvs_cvs

.. code:: text

  Available tests in rvs_cvs:
    - test_rvs_level_config
    - test_rvs_gpu_enumeration
    - test_rvs_gpup_single
    - test_rvs_mem_test
    - test_rvs_gst_single
    - test_rvs_iet_single
    - test_rvs_pebb_single
    - test_rvs_pbqt_single
    - test_rvs_peqt_single
    - test_rvs_rcqt_single
    - test_rvs_tst_single
    - test_rvs_babel_stream

Use these scripts to start the test:

1. Run the installation: 

   .. code:: bash

     cvs run install_rvs --cluster_file input/cluster_file/cluster.json --config_file input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/rvs.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

2. Start the RVS test: 

   .. code:: bash
    
     cvs run rvs_cvs --cluster_file input/cluster_file/cluster.json --config_file input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/rvs.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

.. note::

  Both ``cvs run install_rvs`` and ``cvs run rvs_cvs`` support running inside a per-host container instead of on the host filesystem. Pass a ``cluster_container.json`` cluster file with ``orchestrator: container`` to route invocations through the container backend. See :doc:`/how-to/run-with-containers`.
