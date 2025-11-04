.. meta::
  :description: Configure the Health configuration file variables
  :keywords: health, ROCm, install, cvs

******************************
Health test configuration file
******************************

Here's a code snippet of the ``mi300_health_config.json`` file for reference:

.. dropdown:: ``mi300_health_config.json``

  .. code:: json
    
    {

        "agfhc":
        {
            "path": "/opt/amd/agfhc",
            "package_tar_ball": "/root/cache/PACKAGES/agfhc-mi300x_1.22.0_ub2204.tar.bz2",
            "install_dir": "/root/cache/INSTALL/agfhc/",
            "nfs_install": "True",
            "hbm_test_duration": "00:01:30"
        },
        "transferbench":
        {
          "path": "/opt/amd/transferbench",
          "example_tests_path": "/root/cache/INSTALL/TransferBench/examples",
          "git_install_path": "/root/cache/INSTALL/",
          "git_url": "https://github.com/ROCm/TransferBench.git",
          "nfs_install": "True",
          "results":
          {
              "bytes_to_transfer": "268435456",
              "path": "/opt/amd/transferbench",
              "gpu_to_gpu_a2a_rtotal": "320.0",
              "avg_gpu_to_gpu_p2p_unidir_bw": "33.9",
              "avg_gpu_to_gpu_p2p_bidir_bw": "43.9",
              "best_gpu0_bw": "480.0",
              "32_cu_local_read": "1650",
              "32_cu_local_write": "1250.0",
              "32_cu_local_copy": "1250.0",
              "32_cu_rem_read": "48.0",
              "32_cu_rem_write": "48.0",
              "32_cu_rem_copy": "48.0",
              "example_results":
              {
                  "test1": "47.1",
                  "test2": "48.4",
                  "test3_0_to_1": "31.9",
                  "test3_1_to_0": "38.9",
                  "test4": "1264",
                  "test6": "48.6"
              }

          }
        },
        "rvs":
        {
            "path": "/opt/rocm/bin",
            "git_install_path": "/root/cache/INSTALL",
            "git_url": "https://github.com/ROCm/ROCmValidationSuite.git",
            "nfs_install": "True",
            "config_path_mi300x": "/opt/rocm/share/rocm-validation-suite/conf/MI300X",
            "config_path_default": "/opt/rocm/share/rocm-validation-suite/conf",
            "tests": [
            {
                "name": "gpup_single",
                "config_file": "gpup_single.conf",
                "description": "GPU Properties Test",
                "timeout": 1800,
                "expected_pass": true,
                "fail_regex_pattern": "FAIL|ERROR"
            },
            {
                "name": "mem_test",
                "config_file": "mem.conf",
                "description": "Memory Test",
                "timeout": 6000,
                "expected_pass": true,
                "fail_regex_pattern": "FAIL|\\[ERROR\\s*\\]"
            },
            {
                "name": "gst_single",
                "config_file": "gst_single.conf",
                "description": "GPU Stress Test",
                "timeout": 6000,
                "expected_pass": true,
                "fail_regex_pattern": "met:\\s*FALSE"
            },
            {
                "name": "iet_single",
                "config_file": "iet_single.conf",
                "description": "Input EDPp Test",
                "timeout": 1800,
                "expected_pass": true,
                "fail_regex_pattern": "pass:\\s*FALSE"
            },
            {
                "name": "pebb_single",
                "config_file": "pebb_single.conf",
                "description": "PCI Express Bandwidth Benchmark",
                "timeout": 1800,
                "expected_pass": true,
                "fail_regex_pattern": "\\[ERROR\\s*\\]"
            },
            {
                "name": "pbqt_single",
                "config_file": "pbqt_single.conf",
                "description": "P2P Benchmark and Qualification Tool",
                "timeout": 1800,
                "expected_pass": true,
                "fail_regex_pattern": "FAIL|ERROR:"
            },
            {
                "name": "peqt_single",
                "config_file": "peqt_single.conf",
                "description": "PCI Express Qualification Tool",
                "timeout": 1800,
                "expected_pass": true,
                "fail_regex_pattern": "peqt false"
            },
            {
                "name": "rcqt_single",
                "config_file": "rcqt_single.conf",
                "description": "ROCm Configuration Qualification Tool",
                "timeout": 1800,
                "expected_pass": true,
                "fail_regex_pattern": " not installed and no information is available"
            },
            {
                "name": "tst_single",
                "config_file": "tst_single.conf",
                "description": "Thermal Stress Test",
                "timeout": 1800,
                "expected_pass": true,
                "fail_regex_pattern": "pass: FLASE"
            },
            {
                "name": "babel_stream",
                "config_file": "babel.conf",
                "description": "BABEL Benchmark Test",
                "timeout": 6000,
                "expected_pass": true,
                "fail_regex_pattern": "\\[ERROR\\s*\\]"
            }
            ]
    }
  
  }

Parameters
==========

Here's an exhaustive list of the available parameters in the Health configuration file.

AGFHC
-----
.. |br| raw:: html

    <br />

.. list-table::
   :widths: 15 10 30
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``path``
     - ``/root/cache/PACKAGES`` |br| ``/agfhc-mi300x_1.22.0`` |br| ``_ub2204.tar.bz2``
     - Path where AGFHC is installed
   * - ``package_tar_ball``
     - ``/root/cache/INSTALL/agfhc/``
     - Path where the tar ball is downloaded   
   * - ``install_dir``
     - ``/root/cache/INSTALL/agfhc/``
     - Path where AGFHC runs
   * - ``nfs_install``
     - True
     - Set the flag to install nfs
   * - ``hbm_test_duration``
     - 00:01:30
     - HBM test duration

TransferBench
-------------

.. list-table::
   :widths: 15 10 30
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``path``
     - ``/opt/amd/transferbench``
     - Path where TransferBench is installed
   * - ``example_tests_path``
     - ``/root/cache/INSTALL`` |br| ``/TransferBench/examples``
     - Path where TransferBench examples are installed
   * - ``git_install_path``
     - ``/root/cache/INSTALL/``
     - Path where the Git repo is installed
   * - ``git_url``
     - `https://github.com/ROCm/TransferBench.git <https://github.com/ROCm/TransferBench.git>`_
     - URL for Git repo
   * - ``nfs_install``
     - True
     - Set the flag to install nfs
   * - ``bytes_to_transfer``
     - 268435456
     - Amount of data to transfer in bytes (256 MB); this is the payload size for bandwidth tests
   * - ``gpu_to_gpu_a2a_rtotal``
     - 320.0
     - All-to-all communication total bandwidth in GB/s across all GPUs 
   * - ``avg_gpu_to_gpu_p2p_unidir_bw``
     - 33.9
     - Average peer-to-peer unidirectional bandwidth (GB/s) between GPU pairs
   * - ``avg_gpu_to_gpu_p2p_bidir_bw``
     - 43.9
     - Average peer-to-peer bidirectional bandwidth (GB/s) between GPU pair
   * - ``best_gpu0_bw``
     - 480.0
     - Best measured bandwidth (GB/s) for GPU 0
   * - ``32_cu_local_read``
     - 1650
     - Local memory read bandwidth (GB/s) using 32 CUs
   * - ``32_cu_local_write``
     - 1250.0
     - Local memory write bandwidth (GB/s) using 32 CUs
   * - ``32_cu_local_copy``
     - 1250.0
     - Local memory copy bandwidth (GB/s) using 32 CUs
   * - ``32_cu_rem_read``
     - 48.0
     - Remote memory read bandwidth (GB/s) using 32 CUs
   * - ``32_cu_rem_write``
     - 48.0
     - Remote memory write bandwidth (GB/s) using 32 CUs
   * - ``32_cu_rem_copy``
     - 48.0
     - Remote memory copy bandwidth (GB/s) using 32 CUs
   * - ``test1``
     - 47.1
     - Specific benchmark result (likely bandwidth in GB/s)
   * - ``test2``
     - 48.4
     - Another benchmark result
   * - ``test3_0_to_1``
     - 31.9
     - Directional test from GPU 0 to GPU 1
   * - ``test3_1_to_0``
     - 38.9
     - Directional test from GPU 1 to GPU 0
   * - ``test4``
     - 1264
     - High-performance test result (possibly local memory)
   * - ``test6``
     - 48.6
     - Additional benchmark result

ROCm Validation Suite (RVS)
---------------------------

.. list-table::
   :widths: 15 10 30
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``path``
     - ``/opt/rocm/bin``
     - Path where the RVS test is installed
   * - ``git_install_path``
     - ``/root/cache/INSTALL``
     - Path to installed Git repo
   * - ``git_url``
     - `https://github.com/ROCm/ROCmValidationSuite.git <https://github.com/ROCm/ROCmValidationSuite.git>`_
     - URL for Git repo
   * - ``nfs_install``
     - True
     - Set the flag to install nfs
   * - ``config_path_mi300x``
     - ``/opt/rocm/share/`` |br| ``rocm-validation-suite`` |br| ``/conf/MI300X``
     - Path for Instinct MI300X configuration 
   * - ``config_path_default``
     - ``/opt/rocm/share/`` |br| ``rocm-validation`` |br| ``-suite/conf``
     -  Default path for RVS
   * - ``name``
     - ``gst_single``
     - Test name
   * - ``config_file``
     - ``gst_single.conf``
     - Test configuration file
   * - ``description``
     - GPU Stress Test - Single GPU validation
     - Test description
   * - ``timeout``
     - 1800
     - Timeout in seconds
   * - ``expected_pass``
     - True
     - Result
   * - ``name``
     - ``iet_single``
     -  Test name
   * - ``config_file``
     - ``iet_single.conf``
     - Test configuration file
   * - ``description``
     - Input EDPp Test - Single GPU validation
     - Test description
   * - ``config_file``
     - ``iet_single.conf``
     - Test configuration file
   * - ``name``
     - ``pebb_single``
     -  Test name
   * - ``config_file``
     - ``pebb_single.conf``
     - Test configuration file
 