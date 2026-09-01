.. meta::
  :description: Run Flux.1, Flux.2, and WAN 2.2 xDiT inference tests
  :keywords: CVS, flux1_t2i, flux2, wan22, xdit

******************
xDiT tests
******************

xDit suites live under ``cvs.tests.inference.xdit``. Config templates are in
``cvs/input/config_file/inference/xdit/``. Use a **single** config with ``*_single`` suites
and a **distributed** config (``nnodes >= 2``) with ``*_distributed`` suites. FLUX.1-dev and
FLUX.2-dev share the ``pytorch_xdit_flux_dev_*`` suites; pick the matching ``flux1`` or
``flux2`` JSON. Replace every ``<changeme>`` (and ``{user-id}``) before running.

.. _xdit-set-up-config:

Set up config
=============

1. List available xDiT configs:

   .. code:: bash

     cvs config list inference/xdit

2. Copy the templates you need. FLUX.1-dev and FLUX.2-dev share the same suites; copy
   the matching ``flux1`` or ``flux2`` JSON. For example:

   .. code:: bash

     cvs config copy inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json --output ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json
     cvs config copy inference/xdit/mi3xx_pytorch_xdit_wan22_14b_single.json --output ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_wan22_14b_single.json

3. Edit each file — set ``container_image``, ``nnodes`` for distributed runs, and any
   ``<changeme>`` placeholders.

Config references: :doc:`/reference/configuration-files/inference/flux1_t2i`, :doc:`/reference/configuration-files/inference/wan22_i2v`.

.. _xdit-run-tests:

Run tests
=========

Pytorch xdit test scripts
------------------------------

You can list all available Pytorch xdit test cases using the CLI:

.. code:: bash

  cvs list pytorch_xdit_flux_dev_single

.. code:: text

  Available tests in pytorch_xdit_flux_dev_single:
    - test_cleanup_stale_containers
    - test_parse_and_validate_results
    - test_run_flux1_benchmark
    - test_verify_hf_cache_or_download

.. code:: bash

  cvs list pytorch_xdit_flux_dev_distributed

.. code:: text

  Available tests in pytorch_xdit_flux_dev_distributed:
    - test_cleanup_stale_containers
    - test_parse_and_validate_results
    - test_run_flux1_benchmark
    - test_verify_hf_cache_or_download
    - test_verify_parallelism_config

.. code:: bash

  cvs list pytorch_xdit_wan22_14b_single

.. code:: text

  Available tests in pytorch_xdit_wan22_14b_single:
    - test_cleanup_stale_containers
    - test_parse_and_validate_results
    - test_run_wan22_benchmark
    - test_verify_hf_cache_or_download

.. code:: bash

  cvs list pytorch_xdit_wan22_14b_diffusers_single

.. code:: text

  Available tests in pytorch_xdit_wan22_14b_diffusers_single:
    - test_cleanup_stale_containers
    - test_parse_and_validate_results
    - test_run_wan22_diffusers_benchmark
    - test_verify_model_on_nodes

.. code:: bash

  cvs list pytorch_xdit_wan22_14b_diffusers_distributed

.. code:: text

  Available tests in pytorch_xdit_wan22_14b_diffusers_distributed:
    - test_cleanup_stale_containers
    - test_parse_and_validate_results
    - test_run_wan22_diffusers_benchmark
    - test_verify_model_on_nodes
    - test_verify_parallelism_config

Use these scripts to run the Pytorch xdit tests.

FLUX.1-dev single-node:

.. code:: bash

  cvs run pytorch_xdit_flux_dev_single --cluster_file input/cluster_file/cluster.json --config_file input/config_file/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json --html=/var/www/html/cvs/pytorch_xdit_flux1_single.html --capture=tee-sys --self-contained-html --log-file=/tmp/pytorch_xdit_flux1_single.log -vvv -s

FLUX.1-dev distributed:

.. code:: bash

  cvs run pytorch_xdit_flux_dev_distributed --cluster_file input/cluster_file/cluster.json --config_file input/config_file/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_distributed.json --html=/var/www/html/cvs/pytorch_xdit_flux1_distributed.html --capture=tee-sys --self-contained-html --log-file=/tmp/pytorch_xdit_flux1_distributed.log -vvv -s

FLUX.2-dev single-node (same suite, flux2 config; mounts ``scripts/flux2_example.py`` when the image lacks it):

.. code:: bash

  cvs run pytorch_xdit_flux_dev_single --cluster_file input/cluster_file/cluster.json --config_file input/config_file/inference/xdit/mi3xx_pytorch_xdit_flux2_dev_single.json --html=/var/www/html/cvs/pytorch_xdit_flux2_single.html --capture=tee-sys --self-contained-html --log-file=/tmp/pytorch_xdit_flux2_single.log -vvv -s

FLUX.2-dev distributed:

.. code:: bash

  cvs run pytorch_xdit_flux_dev_distributed --cluster_file input/cluster_file/cluster.json --config_file input/config_file/inference/xdit/mi3xx_pytorch_xdit_flux2_dev_distributed.json --html=/var/www/html/cvs/pytorch_xdit_flux2_distributed.html --capture=tee-sys --self-contained-html --log-file=/tmp/pytorch_xdit_flux2_distributed.log -vvv -s

WAN 2.2 I2V native single-node:

.. code:: bash

  cvs run pytorch_xdit_wan22_14b_single --cluster_file input/cluster_file/cluster.json --config_file input/config_file/inference/xdit/mi3xx_pytorch_xdit_wan22_14b_single.json --html=/var/www/html/cvs/pytorch_xdit_wan22.html --capture=tee-sys --self-contained-html --log-file=/tmp/pytorch_xdit_wan22.log -vvv -s

WAN 2.2 I2V Diffusers xFuser single-node:

.. code:: bash

  cvs run pytorch_xdit_wan22_14b_diffusers_single --cluster_file input/cluster_file/cluster.json --config_file input/config_file/inference/xdit/mi3xx_pytorch_xdit_wan22_14b_diffusers_single.json --html=/var/www/html/cvs/pytorch_xdit_wan22_diffusers_single.html --capture=tee-sys --self-contained-html --log-file=/tmp/pytorch_xdit_wan22_diffusers_single.log -vvv -s

WAN 2.2 I2V Diffusers xFuser distributed:

.. code:: bash

  cvs run pytorch_xdit_wan22_14b_diffusers_distributed --cluster_file input/cluster_file/cluster.json --config_file input/config_file/inference/xdit/mi3xx_pytorch_xdit_wan22_14b_diffusers_distributed.json --html=/var/www/html/cvs/pytorch_xdit_wan22_diffusers_distributed.html --capture=tee-sys --self-contained-html --log-file=/tmp/pytorch_xdit_wan22_diffusers_distributed.log -vvv -s
