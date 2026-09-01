.. meta::
  :description: CVS command-line reference
  :keywords: CVS, CLI, cvs run, cvs exec, cvs scp

**************
CLI reference
**************

Top-level commands
==================

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``cvs generate``
     - Generate configuration files or templates
   * - ``cvs config``
     - Browse and copy bundled configuration templates
   * - ``cvs run``
     - Run a test suite (wrapper over pytest)
   * - ``cvs list``
     - List available tests
   * - ``cvs monitor``
     - Cluster health monitoring
   * - ``cvs exec``
     - Execute a command on all cluster nodes
   * - ``cvs scp``
     - Copy files to cluster nodes in parallel

``cvs generate``
================

Run ``cvs generate`` with no arguments to list available generators.

Subcommands: ``cluster_json``, ``heatmap``.

``cluster_json``
----------------

Generate a cluster JSON file from a host list.

Required options (one host source plus credentials):

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--input_hosts_file PATH`` **or** ``--hosts HOSTS``
     - Host list: file with one IP/hostname per line, or comma-separated list. Supports ranges such as ``192.168.1.10-20`` and ``hostname[1-10]``
   * - ``--output_json_file PATH``
     - Output cluster JSON path
   * - ``--username USER``
     - SSH username for cluster nodes
   * - ``--key_file PATH``
     - SSH private key file

Optional:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--head_node IP``
     - Head node IP (defaults to the first host)

See :doc:`/how-to/configure/cluster-config`.

``heatmap``
-----------

Generate an RCCL performance heatmap HTML report from actual vs reference JSON.

Required options:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``-a PATH``, ``--actual PATH``
     - Actual results JSON (RCCL graph format)
   * - ``-r PATH``, ``--reference PATH``
     - Golden reference JSON

Optional options:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``-o PATH``, ``--output PATH``
     - Output HTML path (default: ``/tmp/rccl_heatmap_<timestamp>.html``)
   * - ``-t TITLE``, ``--title TITLE``
     - Chart title (default: ``RCCL Performance Heatmap``)
   * - ``--metadata``
     - Include metadata table (actual JSON must have a ``metadata`` key)
   * - ``--no-data-table``
     - Omit the data table from the HTML report

See :doc:`/how-to/test-suites/network/rccl`.

``cvs config``
==============

Subcommands: ``list-dirs``, ``list``, ``copy``.

``list-dirs [path]``
--------------------

List config directories grouped by category (``config_file_dirs:``, ``cluster_file_dirs:``, ``env_file_dirs:``). Optional ``path`` scopes to a subtree (for example ``training``).

``list [path]``
---------------

List template files grouped by parent directory. Optional ``path`` scopes to a subdirectory or file prefix (for example ``platform``, ``rccl``).

``copy [path] --output PATH``
-----------------------------

Copy bundled templates into your workspace.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--output PATH``
     - Destination file or directory (**required**)
   * - ``--all``
     - Copy all templates, preserving directory structure
   * - ``--force``
     - Overwrite existing destination files

Positional ``path`` is optional: omit for bulk operations with ``--all``; specify a template path for a single-file copy (for example ``platform/host_config.json``).

See :doc:`/how-to/configure/test-suite-config/index` and :doc:`/how-to/configure/cluster-config`.

``cvs run``
===========

Positional arguments: ``test [function ...]`` — suite name and optional test function names.

Required options:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--cluster_file PATH``
     - Cluster JSON (nodes, SSH credentials, backend)
   * - ``--config_file PATH``
     - Suite-specific test configuration JSON

Optional options:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--workspace PATH``
     - Shared-filesystem root; run dir is ``<workspace>/cvs_runs/<run_id>``. Falls back to ``$CVS_WORKSPACE``, then the venv parent directory
   * - ``--html PATH``
     - Pytest HTML report output path
   * - ``--self-contained-html``
     - Embed CSS and images in the HTML report
   * - ``--log-file PATH``
     - Text log file (parent directories created automatically)
   * - ``--log-level LEVEL``
     - ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``
   * - ``--capture MODE``
     - ``no``, ``tee-sys``, ``tee-merged``, ``fd``, ``sys``

All other pytest flags pass through. Run ``pytest --help`` for the full list.

See :doc:`/how-to/run-tests/index`.

``cvs list``
============

Positional argument: ``[test]`` — optional suite name.

Optional options:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--cluster_file PATH``
     - Cluster file for parameterized test collection
   * - ``--config_file PATH``
     - Config file for parameterized test collection

With no arguments, lists all suites (same catalog as ``cvs run`` with no arguments). With a suite name, lists test functions in that suite.

See :doc:`/how-to/run-tests/index`.

``cvs monitor``
===============

Run ``cvs monitor`` with no arguments to list available monitors.

Subcommand: ``check_cluster_health``

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--cluster_file PATH``
     - Cluster JSON (recommended; takes precedence over ``CLUSTER_FILE``)
   * - ``--iterations N``
     - Number of check iterations
   * - ``--time_between_iters SECONDS``
     - Sleep between iterations
   * - ``--report_file PATH``
     - Output HTML report path (default: ``cluster_report.html`` in the current directory)

Deprecated (use ``--cluster_file`` instead): ``--hosts_file``, ``--username``, ``--password``, ``--key_file``.

See :doc:`Health reports </how-to/monitor/health-reports/index>` and :doc:`Live dashboards </how-to/monitor/live-dashboards/index>`.

``cvs exec``
============

Required options:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--cmd COMMAND``
     - Shell command to run on selected nodes

Optional options:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--cluster_file PATH``
     - Cluster JSON (or set ``CLUSTER_FILE``)
   * - ``--target {computes,switches,all}``
     - ``computes`` (default), ``switches``, or both
   * - ``--timeout SECONDS``
     - Per-node command output timeout (default: ``30``)
   * - ``--connect-timeout SECONDS``
     - Per-node SSH connect timeout (default: ``15``)
   * - ``--json``
     - Emit structured JSON on stdout
   * - ``--verbose``, ``-v``
     - Show SSH connection diagnostics

See :doc:`/how-to/execute-cluster-commands`.

``cvs scp``
===========

Required options:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--file PATH``
     - Local file or directory to copy

Optional options:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Description
   * - ``--dest PATH``
     - Remote destination (defaults to the same path as the source)
   * - ``--recurse``
     - Copy directories recursively
   * - ``--cluster_file PATH``
     - Cluster JSON (or set ``CLUSTER_FILE``)
   * - ``--parallel N``
     - Parallel SCP operations (default: ``20``)

See :doc:`/how-to/copy-to-cluster`.
