.. meta::
  :description: Configure CVS test suite config files for AMD Instinct GPU clusters: browse templates, set placeholders, and pass configs to cvs run.
  :keywords: CVS, ROCm, configure, config_file, test suite, AMD Instinct, GPU, AMD, JSON, CLI, Linux, templates

*************************************************************************
Configure Cluster Validation Suite (CVS) test suite configuration files
*************************************************************************

The test suite config (``--config_file``) holds suite-specific settings: what to run, paths, thresholds, container images, and similar options. Pass it to ``cvs run``.

Configure each test suite config file with settings specific to your cluster. Shipped templates keep required input to a minimum: fields you must set are marked with the ``<changeme>`` placeholder.

Replace every ``<changeme>`` value before running — CVS exits with an error if any placeholder remains unresolved. For the complete field-by-field reference for each test suite's config schema, see :doc:`/reference/configuration-files/index`.

Browse and copy templates
=========================

CVS ships templates under ``cvs/input/config_file/``. Use ``cvs config`` to discover what is available and copy files into your workspace before editing them.

``list-dirs`` — browse directories grouped by category (start here when you are not sure what exists):

.. code:: bash

  cvs config list-dirs
  cvs config list-dirs training

``list`` — list template files grouped by parent directory:

.. code:: bash

  cvs config list
  cvs config list platform
  cvs config list rccl

``copy`` — copy one file or every bundled template (``--output`` is required; use ``--force`` to overwrite existing files):

.. code:: bash

  cvs config copy platform/host_config.json --output ~/cvs_workspace/host_config.json
  cvs config copy health/mi300_health_config.json --output ~/cvs_workspace/mi300_health_config.json
  cvs config copy rccl/rccl_config.json --output ~/cvs_workspace/rccl_config.json
  cvs config copy --all --output ~/cvs_workspace/
  cvs config copy --all --output ~/cvs_workspace/ --force

The ``copy`` command creates output directories as needed and preserves directory structure when using ``--all``.

For cluster file templates (``--cluster_file``), use the same ``cvs config copy`` command — see :doc:`/how-to/configure/cluster-config`.

For a suite-by-suite index of example config paths, see :doc:`/how-to/configure/test-suite-config/pick-config-file`.

Next steps
==========

- :doc:`/how-to/test-suites/index` — run a test suite. Each suite's page includes a **Set up config** section with the copy command and the fields you must edit.
- :doc:`/reference/configuration-files/index` — full field-by-field schema for every test suite config file.
