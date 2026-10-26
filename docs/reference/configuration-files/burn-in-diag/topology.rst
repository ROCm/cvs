.. meta::
  :description: JSON configuration schema for the CVS cluster topology discovery suite, covering probe timeout and ANC fallback fields.
  :keywords: CVS, topology, configuration, JSON, amd-smi, cluster, ROCm, GPU, AMD

**************************************************
Cluster topology discovery configuration schema
**************************************************

Configuration for the ``topology_discover`` suite, under the top-level ``topology`` key in ``cvs/input/config_file/topology/topology_config.json``. All fields are optional; the suite reads ``amd-smi`` and sysfs, which auto-detect the hardware, so the shipped defaults run as-is.

How to run this suite: :doc:`/how-to/test-suites/burn-in-diag/topology`.

Fields
======

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Meaning
   * - ``description``
     - ``Cluster Hardware Topology``
     - Human-readable label shown in reports.
   * - ``probe_timeout``
     - ``120``
     - Seconds allowed for the per-node probe to return on each node. The probe only reads ``amd-smi`` and sysfs, so the default is generous; raise it only for very large clusters.
   * - ``use_anc_fallback``
     - ``false``
     - Reserved. When true, a node without ``amd-smi`` falls back to ``anc.py --device-topology``. Not required when ``amd-smi`` is present on every node.
