.. meta::
  :description: Run ATOM inference benchmarks
  :keywords: CVS, atom

********************
ATOM inference tests
********************

.. _atom-set-up-config:

Set up config
=============

1. List available ATOM configs:

   .. code:: bash

     cvs config list inference/atom

2. Copy the main ``*.json`` and matching ``*_threshold.json`` (keep both in the same directory):

   .. code:: bash

     cvs config copy inference/atom/mi300x_atom_gpt-oss-120b_mxfp4_single.json --output ~/cvs_workspace/inference/atom/mi300x_atom_gpt-oss-120b_mxfp4_single.json
     cvs config copy inference/atom/mi300x_atom_gpt-oss-120b_mxfp4_single_threshold.json --output ~/cvs_workspace/inference/atom/mi300x_atom_gpt-oss-120b_mxfp4_single_threshold.json

3. Edit the files — set ``container_image``, ``nnodes``, and any remaining ``<changeme>`` values.

Full parameter list: :doc:`/reference/configuration-files/inference/atom`.

.. _atom-run-tests:

Run tests
=========

ATOM test scripts
------------------------------

You can list all available ATOM test cases using the CLI:

.. code:: bash

  cvs list atom

.. code:: text

  Available tests in atom:
    - test_launch_container
    - test_atom_inference
    - test_print_results_table
    - test_teardown

Use these scripts to run the ATOM tests. Supply your own suite JSON
(``schema_version: 1`` variant config); see :doc:`/reference/configuration-files/inference/atom`.
After ``cvs config copy``, keep **one** ``*threshold.json`` in the same directory as the
``--config_file`` you pass (per-variant subdirs under ``~/input/.../atom/``).
Copy-paste lab commands: ``cvs/input/config_file/inference/atom/README.md``.

.. code:: bash

  TS=$(date +%Y%m%d_%H%M%S)
  cvs run atom \
    --cluster_file ~/input/cluster_file/atom_cluster.json \
    --config_file ~/input/config_file/inference/atom/single/mi300x_atom_deepseek-r1_fp8_single.json \
    --html=~/cvs_results/${TS}_atom-single_mi300x.html \
    --self-contained-html \
    --log-file=~/cvs_results/${TS}_atom-single_mi300x.log \
    -vvv -s
