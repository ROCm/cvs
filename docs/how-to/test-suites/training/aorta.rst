.. meta::
  :description: Run the Aorta distributed training benchmark
  :keywords: CVS, aorta

***************
Aorta benchmark
***************

.. _aorta-set-up-config:

Set up config
=============

1. Copy the Aorta benchmark configuration file:

   .. code:: bash

     cvs config copy aorta/aorta_benchmark.yaml --output ~/cvs_workspace/aorta/aorta_benchmark.yaml

2. Edit the file for your environment:

   - ``aorta_path`` — prefer local or scratch storage (avoid NFS root-squash paths)
   - ``docker.image`` and RCCL build settings as needed
   - Any ``<changeme>`` placeholders

Full parameter list: :doc:`/reference/configuration-files/training/aorta`.

.. _aorta-run-tests:

Run tests
=========

Aorta benchmark
---------------

The Aorta benchmark runs an Aorta-based workload in a Docker container with RCCL, collects PyTorch profiler traces, and validates iteration time, compute ratio, overlap ratio, and rank balance against configurable thresholds in ``aorta_benchmark.yaml``.

**Where to put Aorta (``aorta_path``):** Prefer local or scratch storage (for example under ``/scratch/``). If ``aorta_path`` is on NFS (such as home directories under ``/home``), the container can hit *Permission denied* when creating ``artifacts/`` because many NFS exports use *root_squash*. Use a non-root-squashed path or adjust exports; set ``aorta_path`` accordingly in ``aorta_benchmark.yaml``.

List tests in this suite:

.. code:: bash

  cvs list test_aorta

.. code:: text

  Available tests in test_aorta:
    - test_validate_runner_config
    - test_run_benchmark
    - test_parse_results
    - test_validate_thresholds
    - test_generate_report

Run from the CVS package directory (the directory that contains ``input/``), for example:

.. code:: bash

  cd /path/to/your/cvs-checkout/cvs
  cvs run test_aorta --cluster_file input/cluster_file/cluster.json --config_file input/config_file/aorta/aorta_benchmark.yaml --html=/var/www/html/cvs/aorta.html --capture=tee-sys --self-contained-html --log-file=/tmp/aorta.log -vvv -s
