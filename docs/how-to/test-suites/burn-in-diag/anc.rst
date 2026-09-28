.. meta::
  :description: Run AMD Node Check (ANC) CPU and GPU diagnostic tests across every node in a CVS-managed AMD Instinct GPU cluster to verify hardware health.
  :keywords: CVS, ANC, AMD Node Check, AMD, GPU, AMD Instinct, ROCm, diagnostic, burn-in, CPU, HBM

*****************************************************
Run AMD Node Check (ANC) CPU and GPU diagnostic tests
*****************************************************

AMD Node Check (ANC) runs CPU and GPU diagnostic groups on every node in the cluster. CVS installs ANC when needed, invokes each group as ``sudo ./anc.py -g <group>``, and collects logs and HTML reports.

ANC requires **root**. The runner must have passwordless SSH and passwordless ``sudo`` on every target node. Without passwordless ``sudo``, group runs and log collection fail.

.. _anc-set-up-config:

Set up config
=============

1. Copy the ANC configuration file:

   .. code:: bash

     cvs config copy anc/anc_config.json --output ~/cvs_workspace/anc/anc_config.json

2. Replace every ``<changeme>`` placeholder:

   - ``anc_release_url`` — URL of the ANC release archive to download and install
   - ``log_folder_path`` — controller-side directory prefix for collected logs and auto-generated HTML reports
   - ``anc_version`` — expected ANC version (must match the version in ``anc_release_url``)

3. Optionally set ``ANC_INSTALL_PATH`` for relocatable **tar** installs. Deb and rpm packages ignore this key and always install under ``/opt/amdtools``.

For the complete field reference including release URL, install path, and log collection options, see :doc:`/reference/configuration-files/burn-in-diag/anc`.

.. _anc-run-tests:

Run tests
=========

List the install suite and the CPU / GPU group suites:

.. code:: bash

  cvs list anc_installation
  cvs list anc_test_cpu
  cvs list anc_test_gpu

Install ANC
~~~~~~~~~~~

Every CPU and GPU group run installs ANC as a session-cached pre-task, so a separate install step is optional. Run ``anc_installation`` when you want to install or refresh ANC without running a validation group:

.. code:: bash

  cvs run anc_installation \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/anc/anc_config.json \
    --html=/var/www/html/cvs/anc.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/anc.log -vvv -s

CPU groups
~~~~~~~~~~

``cvs list anc_test_cpu`` reports one ``test_<group>`` function per CPU group:

.. code:: text

  Available tests in anc_test_cpu:
    - test_ampttk_full
    - test_cachewalker_full
    - test_cpu_all
    - test_cpu_content_check
    - test_cpu_mfg_l10
    - test_cpu_sanity
    - test_difect_full
    - test_fpdeluge_full
    - test_hdrt_full
    - test_maxcorestim_full
    - test_memtest_full
    - test_miidct_full
    - test_mithac_full
    - test_weighted_sanity

Run every CPU group (install + ldconfig once, then each group as its own test):

.. code:: bash

  cvs run anc_test_cpu \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/anc/anc_config.json \
    --html=/var/www/html/cvs/anc_cpu.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/anc_cpu.log -vvv -s

Run a single group by function name:

.. code:: bash

  cvs run anc_test_cpu test_cpu_all \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/anc/anc_config.json

GPU groups
~~~~~~~~~~

``cvs list anc_test_gpu`` reports one ``test_<group>`` function per GPU group:

.. code:: text

  Available tests in anc_test_gpu:
    - test_gpu_content_check
    - test_gpu_mfg_l10
    - test_hbm_lvl1
    - test_hbm_lvl2
    - test_hbm_lvl3
    - test_hbm_lvl4
    - test_hbm_lvl5

Run every GPU group:

.. code:: bash

  cvs run anc_test_gpu \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/anc/anc_config.json \
    --html=/var/www/html/cvs/anc_gpu.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/anc_gpu.log -vvv -s

Run a single GPU group:

.. code:: bash

  cvs run anc_test_gpu test_hbm_lvl1 \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/anc/anc_config.json

Pass and fail
=============

A node passes only when ANC started (a ``Log directory`` line is present), ``console.log`` was collected, and the **final** return-code line in ``console.log`` is ``ANC_SUCCESS [0]``. Failures on multiple nodes are aggregated into a single test failure.

Logs land under ``<log_folder_path>/anc_logs/<ip>_<hostname>/<test_name>/<timestamp>/``. When ``COLLECT_HTML_REPORTS`` is ``True`` (the default), CVS also writes a pytest-html report under ``log_folder_path`` even if you omit ``--html``. An explicit ``--html`` on the command line always wins.
