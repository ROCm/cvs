.. meta::
  :description: Run CVS test scripts
  :keywords: CVS, health, network, tests, RCCL

*********
Run tests
*********

To run a test suite you need two files: a **cluster file** (``--cluster_file``) that describes your nodes and SSH access, and a **test suite config** (``--config_file``) with suite-specific settings. Set those up first — see :doc:`/how-to/configure/cluster-config` and :doc:`/how-to/configure/test-suite-config/index`. To choose the right template, see :doc:`/how-to/configure/test-suite-config/pick-config-file`.

Then use ``cvs run`` on the head node to execute tests across the cluster.

List available suites
=============================

.. code:: bash

  cvs list

You can also run ``cvs run`` with no arguments to see the same catalog.

Run a suite
===================

Pass the cluster file and test suite config from your workspace:

.. code:: bash

  cvs run agfhc_cvs \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi300_health_config.json \
    --html=/var/www/html/cvs/agfhc.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/test.log -vvv -s

List test cases in a suite
==========================

``cvs list <suite>`` lists the test functions in that suite, including parameterized tests generated from your config. Pass the same ``--cluster_file`` and ``--config_file`` you use for ``cvs run``:

.. code:: bash

  cvs list agfhc_cvs \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi300_health_config.json

Run one test function
=============================

Add the test function name after the suite name:

.. code:: bash

  cvs run agfhc_cvs test_agfhc_hbm \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi300_health_config.json \
    --html=/var/www/html/cvs/agfhc.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/test.log -vvv -s

Run with containers
===================

CVS selects an execution backend in the cluster file:

- **Bare metal** — use when you can install or upgrade the ROCm stack on each host and run tests on the host filesystem.
- **Container** — use when you cannot install or change ROCm on the hosts, or when you have a Docker image with a pinned ROCm version and framework dependencies (for example PyTorch) that you run on each node.

See :doc:`/how-to/run-with-containers` for ``cluster_container.json``, image selection, and container run commands.

Common ``cvs run`` options
==========================

.. include:: /_includes/common-cvs-run-flags.rst

Test suites
===========

Per-suite **Set up config** and **Run tests** steps are grouped by category:

- :doc:`Burn-in / Diag tests </how-to/test-suites/burn-in-diag/index>` — platform, health, preflight
- :doc:`Network tests </how-to/test-suites/network/index>` — IB Perf, RCCL, MORI
- :doc:`Training tests </how-to/test-suites/training/index>` — Aorta, JAX MaxText, Megatron
- :doc:`Inference tests </how-to/test-suites/inference/index>` — vLLM, ATOM, SGLang, xDiT

Scalability
===========

For clusters with 32+ nodes, see :doc:`/concepts/cvs-at-scale`.

Test results
============

.. include:: /_includes/test-results.rst
