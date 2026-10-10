.. meta::
  :description: Run AMD Node Check (ANC) CPU and GPU diagnostic tests across every node in a CVS-managed AMD Instinct GPU cluster to verify hardware health.
  :keywords: CVS, ANC, AMD Node Check, AMD, GPU, AMD Instinct, ROCm, diagnostic, burn-in, CPU, HBM, DIMM

*****************************************************
Run AMD Node Check (ANC) CPU and GPU diagnostic tests
*****************************************************

AMD Node Check (ANC) runs CPU and GPU diagnostic groups on every node in the cluster (the CPU groups include the DIMM/UMC groups), plus individual ANC items. CVS installs ANC when needed, invokes each group as ``sudo ./anc.py -g <group>`` (and each individual item as ``sudo ./anc.py -i <item>``), and collects logs and HTML reports.

ANC requires **root**. The runner must have passwordless SSH and passwordless ``sudo`` on every target node. Without passwordless ``sudo``, group runs and log collection fail.

.. _anc-set-up-config:

Set up config
=============

1. Copy the ANC configuration file:

   .. code:: bash

     cvs config copy anc/anc_config.json --output ~/cvs_workspace/anc/anc_config.json

2. Replace every ``<changeme>`` placeholder:

   - ``anc_release_url`` — URL of the ANC release archive to download and install
   - ``anc_version`` — minimum required ANC version (the version in ``anc_release_url`` must be >= this). Install is skipped only when **all** expected nodes already satisfy it; if any node is below the minimum, the installer runs across all nodes.

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
  cvs list anc_test_gemm          # and the other per-family item suites

Install ANC
~~~~~~~~~~~

Every CPU / GPU group and individual-item run installs ANC as a session-cached pre-task, so a separate install step is optional. Run ``anc_installation`` when you want to install or refresh ANC without running a validation group:

.. code:: bash

  cvs run anc_installation \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/anc/anc_config.json \
    --capture=tee-sys -vvv -s

CPU groups
~~~~~~~~~~

``cvs list anc_test_cpu`` reports one ``test_<group>`` function per CPU group:

.. code:: text

  Available tests in anc_test_cpu:
    - test_cpu_content_check
    - test_cpu_mfg_l10
    - test_weighted_sanity
    - test_dimm_content_check
    - test_dimm_mfg_l10
    - test_dimm_weighted_sanity

The DIMM/UMC groups are part of ``anc_test_cpu`` because ANC reports them under the CPU device.

Run every CPU group (install + ldconfig once, then each group as its own test):

.. code:: bash

  cvs run anc_test_cpu \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/anc/anc_config.json \
    --capture=tee-sys -vvv -s

Run a single group by function name:

.. code:: bash

  cvs run anc_test_cpu test_cpu_mfg_l10 \
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
    --capture=tee-sys -vvv -s

Run a single GPU group:

.. code:: bash

  cvs run anc_test_gpu test_hbm_lvl1 \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/anc/anc_config.json

Individual items
~~~~~~~~~~~~~~~~

Individual ANC items are exposed as per-family suites, each item its own ``test_<item>`` function run as ``sudo ./anc.py -i <item>``: ``anc_test_computerocker`` (24), ``anc_test_memrocker`` (10), ``anc_test_oblex`` (8), ``anc_test_gemm`` (3), ``anc_test_xgmi`` (3), ``anc_test_ualink`` (3), ``anc_test_pcie`` (3), ``anc_test_babel`` (2), and ``anc_test_basic`` (18 one-off items such as ``test_ampttk``, ``test_hdrt``, ``test_no_op``, ``test_sdma_bidi_peak``). ``cvs list anc_test_<family>`` reports a suite's items; the per-family lists live in ``cvs/lib/anc_lib.py``.

Run a whole item family:

.. code:: bash

  cvs run anc_test_gemm \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/anc/anc_config.json \
    --capture=tee-sys -vvv -s

Run a single item:

.. code:: bash

  cvs run anc_test_gemm test_gemm_fp8_trig \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/anc/anc_config.json

Pass and fail
=============

A node passes only when ANC started (a ``Log directory`` line is present), ``console.log`` was collected, and the **final** return-code line in ``console.log`` is ``ANC_SUCCESS [0]``. Failures on multiple nodes are aggregated into a single test failure.

Logs land under ``<run_dir>/anc_logs/<ip>_<hostname>/<test_name>/<timestamp>/``.
``cvs run`` also writes a self-contained pytest HTML report and a text log under
``<run_dir>`` by default (see :doc:`/reference/cli/cvs-run`); pass ``--html`` /
``--log-file`` to redirect them or ``--no-html`` / ``--no-log-file`` to suppress
them. ``<run_dir>`` is ``<workspace>/cvs_runs/<run_id>/``.
