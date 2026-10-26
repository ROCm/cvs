.. meta::
  :description: Reference for the CVS health test configuration file, covering AGFHC, TransferBench, and RVS burn-in parameters for AMD GPU cluster validation.
  :keywords: CVS, health, AGFHC, TransferBench, RVS, ROCm, GPU, burn-in, AMD Instinct, configuration, JSON, benchmark

***************************************************************************
Health test configuration file reference for Cluster Validation Suite (CVS)
***************************************************************************

The health configuration file controls AGFHC, `TransferBench <https://rocm.docs.amd.com/projects/TransferBench/en/latest/index.html>`_, and `ROCm Validation Suite (RVS) <https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/index.html>`_ burn-in parameters. Use these tests to validate GPU hardware, memory bandwidth, and compute throughput on each cluster node before production workloads.

See :doc:`/how-to/test-suites/burn-in-diag/health` for more information on running these tests.

.. note::

  In this configuration file, ``{user-id}`` resolves to the current username at runtime. You can also manually change this value to your username.

.. dropdown:: ``health_config.json``

  .. code:: json

    {

        "agfhc":
        {
            "path": "/opt/amd/agfhc",
            "package_tar_ball": "/home/{user-id}/PACKAGES/agfhc-mi300x_1.22.0_ub2204.tar.bz2",
            "install_dir": "/home/{user-id}/INSTALL/agfhc/",
            "_comment_env": "Companion-tool paths. Defaults match an /opt/amd install. Update every value when path changes. A missing key falls back to the directory next to path. FIREXS2_APP_PATH and LD_LIBRARY_PATH are derived from FIREXS_PATH.",
            "env": {
                "TXB_PATH": "/opt/amd/transferbench",
                "MINIHPL_PATH": "/opt/amd/minihpl",
                "MEMROCKER_PATH": "/opt/amd/memrocker",
                "FIREXS_PATH": "/opt/amd/firexs2",
                "AFIDSAG_PATH": "/opt/amd/afid"
            },
            "_comments_log_dir": "log_dir has to be a NON NFS local file system. CVS runs mkdir -p and test -w on each node before the recipes.",
            "log_dir": "/root/agfhc_logs",
            "hbm_test_duration": "00:01:30"
        },
        "transferbench":
        {
          "path": "/home/{user-id}/INSTALL/TransferBench",
          "git_install_path": "/home/{user-id}/INSTALL/",
          "git_url": "https://github.com/ROCm/TransferBench.git",
          "git_tag": "v1.67.00",
          "rocm_path": "<changeme>",
          "env": {},
          "auto_num_cpu_devices": true,
          "p2p_preset": "p2p",
          "scaling_preset": "scaling",
          "a2asweep_timeout": 1500,
          "results":
          {
              "bytes_to_transfer": "268435456",
              "path": "/home/{user-id}/INSTALL/TransferBench",
              "gpu_to_gpu_a2a_per_link_min": "32.9",
              "gpu_to_gpu_a2a_rtotal": "320.0",
              "avg_gpu_to_gpu_p2p_unidir_bw": "33.9",
              "avg_gpu_to_gpu_p2p_bidir_bw": "43.9",
              "best_gpu0_bw": "480.0",
              "32_cu_local_read": "1650",
              "32_cu_local_write": "1250.0",
              "32_cu_local_copy": "1250.0",
              "32_cu_rem_read": "48.0",
              "32_cu_rem_write": "48.0",
              "32_cu_rem_copy": "48.0"
          }
        },
        "rvs":
        {
            "path": "<changeme>/bin",
            "git_install_path": "/home/{user-id}/INSTALL/rvs",
            "git_url": "https://github.com/ROCm/ROCmValidationSuite.git",
            "nfs_install": "True",
            "_comment_rocm_path": "ROCm installation path.Set to placeholder changeme auto-detect from /opt/rocm or /opt/rocm/core-*",
            "rocm_path": "<changeme>",
            "config_path_mi300x": "/opt/rocm/share/rocm-validation-suite/conf/MI300X",
            "config_path_default": "/opt/rocm/share/rocm-validation-suite/conf",
            "_comment_rvs_test_level": "RVS test level configuration (0-5). 0: Run individual tests (skip level test), 1-5: Run LEVEL config test if RVS >= 1.3.0, else run individual tests. Default is 4.",
            "rvs_test_level": 4,
            "tests": [
                {
                    "name": "level_config",
                    "description": "RVS LEVEL Configuration Test - Runs all modules collectively",
                    "timeout": 14400,
                    "expected_pass": true,
                    "fail_regex_patterns": [
                        "met:\\s*FALSE",
                        "pass:\\s*FALSE",
                        "\\[ERROR\\s*\\]",
                        "FAIL",
                        "ERROR:",
                        "peqt false",
                        "RVS-ERROR",
                        "Missing packages\\s*:\\s*([1-9]\\d*)",
                        "Version mismatch packages\\s*:\\s*([1-9]\\d*)"
                    ]
                },
                {
                    "name": "mem_test",
                    "config_file": "mem.conf",
                    "description": "Memory Test",
                    "timeout": 10000,
                    "expected_pass": true,
                    "fail_regex_pattern": "FAIL|\\[ERROR\\s*\\]|RVS-ERROR"
                },
                {
                    "name": "gst_single",
                    "config_file": "gst_single.conf",
                    "description": "GPU Stress Test",
                    "timeout": 18000,
                    "expected_pass": true,
                    "fail_regex_pattern": "met:\\s*FALSE|RVS-ERROR"
                },
                {
                    "name": "iet_stress",
                    "config_file": "iet_stress.conf",
                    "description": "Peak Power Test",
                    "timeout": 3600,
                    "expected_pass": true,
                    "fail_regex_pattern": "pass:\\s*FALSE|RVS-ERROR"
                },
                {
                    "name": "pebb_single",
                    "config_file": "pebb_single.conf",
                    "description": "PCI Express Bandwidth Benchmark",
                    "timeout": 3600,
                    "expected_pass": true,
                    "fail_regex_pattern": "\\[ERROR\\s*\\]|RVS-ERROR"
                },
                {
                    "name": "pbqt_single",
                    "config_file": "pbqt_single.conf",
                    "description": "P2P Benchmark and Qualification Tool",
                    "timeout": 3600,
                    "expected_pass": true,
                    "fail_regex_pattern": "FAIL|ERROR:|RVS-ERROR"
                },
                {
                    "name": "babel_stream",
                    "config_file": "babel.conf",
                    "description": "BABEL Benchmark Test",
                    "timeout": 9000,
                    "expected_pass": true,
                    "fail_regex_pattern": "\\[ERROR\\s*\\]|RVS-ERROR"
                }
            ]
        }

    }



Parameters
==========

The following parameters are available in the health configuration file.

AGFHC
-----
.. |br| raw:: html

    <br />

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``path``
     - ``/opt/amd/agfhc``
     - Directory that contains the ``agfhc`` executable. Any other path runs ``./install --rocm-tar -o`` on the parent of ``path``, without sudo. That parent must be writable on the nodes.
   * - ``package_tar_ball``
     - ``/home/{user-id}/`` |br| ``PACKAGES/agfhc-mi300x`` |br| ``_1.22.0_ub2204.tar.bz2``
     - Path where the tar ball is downloaded
   * - ``install_dir``
     - ``/home/{user-id}/INSTALL/agfhc/``
     - Path where AGFHC runs
   * - ``env``
     - ``TXB_PATH=/opt/amd/transferbench``, ``MINIHPL_PATH=/opt/amd/minihpl``, ``MEMROCKER_PATH=/opt/amd/memrocker``, ``FIREXS_PATH=/opt/amd/firexs2``, ``AFIDSAG_PATH=/opt/amd/afid``
     - Companion-tool paths exported for every AGFHC run, including ``/opt/amd``. Update them when ``path`` changes. A missing key falls back to ``<parent of path>/transferbench``, ``minihpl``, ``memrocker``, ``firexs2``, or ``<path>/sag`` for ``AFIDSAG_PATH``. ``FIREXS2_APP_PATH`` and ``LD_LIBRARY_PATH`` are derived from ``FIREXS_PATH``.
   * - ``log_dir``
     - ``/root/agfhc_logs``
     - Local, non-NFS log directory. CVS runs ``mkdir -p`` and ``test -w`` on each node before the recipes.
   * - ``hbm_test_duration``
     - 00:01:30
     - HBM test duration

TransferBench
-------------

``install_transferbench`` clones ``git_url``, checks out ``git_tag``, and builds on every node. ``transferbench_cvs`` runs ``<path>/TransferBench <preset>`` on every node. ``rocm_path`` set to ``<changeme>`` auto-detects ``/opt/rocm`` or ``/opt/rocm/core-*``.

Top-level ``transferbench`` keys:

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Packaged value
     - Meaning
   * - ``path``
     - ``/home/{user-id}/INSTALL/TransferBench``
     - Directory that contains the ``TransferBench`` binary. The command is ``<path>/TransferBench``.
   * - ``git_install_path``
     - ``/home/{user-id}/INSTALL/``
     - Parent directory for the clone. The installer writes ``<git_install_path>/TransferBench``.
   * - ``git_url``
     - ``https://github.com/ROCm/TransferBench.git``
     - Repository cloned by ``install_transferbench``.
   * - ``git_tag``
     - ``v1.67.00``
     - Required tag. Install fails when the checkout does not match.
   * - ``rocm_path``
     - ``<changeme>``
     - ``<changeme>`` auto-detects. Any other value is used only when ``<rocm_path>/lib`` contains ``libamdhip64``.
   * - ``env``
     - ``{}``
     - Extra variables for the TransferBench command. ``NUM_CPU_DEVICES`` set here is kept. Cluster ``env_vars`` are not.
   * - ``auto_num_cpu_devices``
     - ``true``
     - When ``NUM_CPU_DEVICES`` is unset, count NUMA nodes that list CPUs and export that count.
   * - ``num_cpu_devices``
     - omitted
     - Optional explicit CPU NUMA count. Use it when p2p aborts while enumerating empty CPU NUMA nodes.
   * - ``p2p_preset``
     - ``p2p``
     - Preset for ``test_transfer_bench_p2p``.
   * - ``scaling_preset``
     - ``scaling``
     - Preset for ``test_transfer_bench_scaling``.
   * - ``a2a_preset``, ``healthcheck_preset``, ``a2asweep_preset``, ``schmoo_preset``
     - omitted
     - Optional preset names. Omitted keys use ``a2a``, ``healthcheck``, ``a2asweep``, and ``schmoo``.
   * - ``a2asweep_timeout``
     - ``1500``
     - Seconds allowed for ``test_transfer_bench_a2asweep``. This value wins when set.
   * - ``a2asweep_combos``
     - ``24`` when omitted
     - Combo count used only when ``a2asweep_timeout`` is omitted.
   * - ``a2asweep_sec_per_combo``
     - ``50`` when omitted
     - Seconds per combo used only when ``a2asweep_timeout`` is omitted.
   * - ``a2asweep_timeout_buffer``
     - ``300`` when omitted
     - Extra seconds added when ``a2asweep_timeout`` is omitted. The derived timeout is at least ``600``.

``results`` keys below are the pass/fail gates, except where the gate column says otherwise. Bandwidths are GB/s.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Packaged value
     - Gate
   * - ``bytes_to_transfer``
     - ``268435456``
     - Present in the packaged file. The suite does not read it.
   * - ``path``
     - same as ``transferbench.path``
     - Present in the packaged file. The suite does not read it. The binary path is the top-level ``path``.
   * - ``gpu_to_gpu_a2a_per_link_min``
     - ``32.9``
     - Required for ``test_transfer_bench_a2a``. Each GPU's ``RTotal / (N-1)`` must be at least this value. ``gpu_to_gpu_a2a_avg`` is the legacy name.
   * - ``gpu_to_gpu_a2a_rtotal``
     - ``320.0``
     - Recorded when set. Not a pass/fail gate.
   * - ``avg_gpu_to_gpu_p2p_unidir_bw``
     - ``33.9``
     - Minimum UniDir GPU-to-GPU average for ``test_transfer_bench_p2p``.
   * - ``avg_gpu_to_gpu_p2p_bidir_bw``
     - ``43.9``
     - Minimum BiDir GPU-to-GPU average for ``test_transfer_bench_p2p``.
   * - ``best_gpu0_bw``
     - ``480.0``
     - Minimum GPU 0 peak for ``test_transfer_bench_scaling``.
   * - ``32_cu_local_read``, ``32_cu_local_write``, ``32_cu_local_copy``
     - ``1650``, ``1250.0``, ``1250.0``
     - Minimum 32-CU local read, write, and copy for ``test_transfer_bench_schmoo``.
   * - ``32_cu_rem_read``, ``32_cu_rem_write``, ``32_cu_rem_copy``
     - ``48.0``
     - Minimum 32-CU remote read, write, and copy for ``test_transfer_bench_schmoo``.

ROCm Validation Suite (RVS)
---------------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``path``
     - ``<changeme>/bin``
     - Path where the RVS test is installed
   * - ``git_install_path``
     - ``/home/{user-id}/INSTALL/rvs``
     - Path to installed Git repo
   * - ``git_url``
     - `https://github.com/ROCm/ROCmValidationSuite.git <https://github.com/ROCm/ROCmValidationSuite.git>`_
     - URL for Git repo
   * - ``nfs_install``
     - True
     - Set the flag to install nfs
   * - ``rocm_path``
     - ``<changeme>``
     - Set the path of rocm
   * - ``config_path_mi300x``
     - ``/opt/rocm/share/`` |br| ``rocm-validation-suite`` |br| ``/conf/MI300X``
     - Path for Instinct MI300X configuration
   * - ``config_path_default``
     - ``/opt/rocm/share/`` |br| ``rocm-validation`` |br| ``-suite/conf``
     -  Default path for RVS
   * - ``rvs_test_level``
     - 4
     - Test level
   * - ``name``
     - ``level_config``
     - Test name
   * - ``description``
     - RVS LEVEL Configuration Test - Runs all modules collectively
     - Test description
   * - ``timeout``
     - 14400
     - Timeout in secs
   * - ``expected_pass``
     - True
     - Result
   * - ``fail_regex_pattern``
     - Values:
        - ``"met:\\s*FALSE",``
        - ``"pass:\\s*FALSE",``
        - ``"\\[ERROR\\s*\\]",``
        - ``"FAIL",``
        - ``"ERROR:",``
        - ``"peqt false",``
        - ``"RVS-ERROR",``
        - ``"Missing packages\\s*:\\s*([1-9]\\d*)",``
        - ``"Version mismatch packages\\s*:\\s*([1-9]\\d*)"``
     - Regular expressions
   * - ``name``
     - ``mem_test``
     - Test name
   * - ``config_file``
     - ``mem_test.conf``
     - Test config file
   * - ``description``
     - Memory test
     - Test description
   * - ``timeout``
     - 10000
     - Timeout in secs
   * - ``expected_pass``
     - True
     - Result
   * - ``fail_regex_pattern``
     - ``FAIL|\\[ERROR\\s*\\]|RVS-ERROR``
     - Failure expression
   * - ``name``
     - ``gst_single``
     - Test name
   * - ``config_file``
     - ``gst_single.conf``
     - Test config file
   * - ``description``
     - GPU Stress Test
     - Test description
   * - ``timeout``
     - 18000
     - Timeout in secs
   * - ``expected_pass``
     - True
     - Result
   * - ``fail_regex_pattern``
     - ``met:\\s*FALSE|RVS-ERROR``
     - Failure expression
   * - ``name``
     - ``iet_stress``
     - Test name
   * - ``config_file``
     - ``iet_stress.conf``
     - Test config file
   * - ``description``
     - Peak Power Test
     - Test description
   * - ``timeout``
     - 3600
     - Timeout in secs
   * - ``expected_pass``
     - True
     - Result
   * - ``fail_regex_pattern``
     - ``\\[ERROR\\s*\\]|RVS-ERROR``
     - Failure expression
   * - ``name``
     - ``pbqt_single``
     - Test name
   * - ``config_file``
     - ``pbqt_single.conf``
     - Test config file
   * - ``description``
     - P2P Benchmark and Qualification Tool
     - Test description
   * - ``timeout``
     - 3600
     - Timeout in secs
   * - ``expected_pass``
     - True
     - Result
   * - ``fail_regex_pattern``
     - ``FAIL|ERROR:|RVS-ERROR``
     - Failure expression
   * - ``name``
     - ``babel_stream``
     - Test name
   * - ``config_file``
     - ``babel_stream.conf``
     - Test config file
   * - ``description``
     - BABEL Benchmark Test
     - Test description
   * - ``timeout``
     - 9000
     - Timeout in secs
   * - ``expected_pass``
     - True
     - Result
   * - ``fail_regex_pattern``
     - ``\\[ERROR\\s*\\]|RVS-ERROR``
     - Failure expression
