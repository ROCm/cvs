.. meta::
  :description: Configure the Health configuration file variables
  :keywords: health, ROCm, install, cvs

******************************
Health test configuration file
******************************

The burn-in health tests are single-node diagnostic tests that validate the hardware and firmware versions' functionality and performance.

Here's a code snippet of the ``mi300_health_config.json`` file for reference:

.. note::

  In this configuration file, ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username. 

.. dropdown:: ``mi300_health_config.json``

  .. code:: json
    
    {
    
        "agfhc":
        {
            "path": "/opt/amd/agfhc",
            "package_tar_ball": "/home/{user-id}/PACKAGES/agfhc-mi300x_1.22.0_ub2204.tar.bz2",
            "install_dir": "/home/{user-id}/INSTALL/agfhc/",
            "_comments_log_dir": "log_dir has to be a NON NFS local file system",
            "log_dir": "/root/agfhc_logs",
            "nfs_install": "True",
            "hbm_test_duration": "00:01:30"
        },
        "transferbench":
        {
          "path": "/home/{user-id}/INSTALL/TransferBench",
          "example_tests_path": "/home/{user-id}/INSTALL/TransferBench/examples",
          "git_install_path": "/home/{user-id}/INSTALL/",
          "git_url": "https://github.com/ROCm/TransferBench.git",
          "nfs_install": "True",
          "results":
          {
              "bytes_to_transfer": "268435456",
              "path": "/home/{user-id}/INSTALL/TransferBench",
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
            "git_install_path": "/home/{user-id}/INSTALL/rvs",
            "git_url": "https://github.com/ROCm/ROCmValidationSuite.git",
            "nfs_install": "True",
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
                    "description": "Power Stress Test",
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

Here's an exhaustive list of the available parameters in the Health configuration file.

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
     - Path where AGFHC is installed
   * - ``package_tar_ball``
     - ``/home/{user-id}/`` |br| ``PACKAGES/agfhc-mi300x`` |br| ``_1.22.0_ub2204.tar.bz2``
     - Path where the tar ball is downloaded   
   * - ``install_dir``
     - ``/home/{user-id}/INSTALL/agfhc/``
     - Path where AGFHC runs
   * - ``_comments_log_dir``
     - ``/home/{user-id}/INSTALL/agfhc/``
     - Path where AGFHC runs
   * - ``log_dir``
     - ``/root/agfhc_logs``
     - Log directory
   * - ``nfs_install``
     - True
     - Set the flag to install nfs
   * - ``hbm_test_duration``
     - 00:01:30
     - HBM test duration

TransferBench
-------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``path``
     - ``/opt/amd/transferbench``
     - Path where TransferBench is installed
   * - ``example_tests_path``
     - ``/home/{user-id}/INSTALL`` |br| ``/TransferBench/examples``
     - Path where TransferBench examples are installed
   * - ``git_install_path``
     - ``/home/{user-id}/INSTALL/``
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
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``path``
     - ``/opt/rocm/bin``
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
   * - ``config_path_mi300x``
     - ``/opt/rocm/share/`` |br| ``rocm-validation-suite`` |br| ``/conf/MI300X``
     - Path for Instinct MI300X configuration 
   * - ``config_path_default``
     - ``/opt/rocm/share/`` |br| ``rocm-validation`` |br| ``-suite/conf``
     -  Default path for RVS
   * - ``_comment_rvs_test_level``
     - "RVS test level configuration (0-5). 0: Run individual tests (skip level test), 1-5: Run LEVEL config test if RVS >= 1.3.0, else run individual tests. Default is 4."
     -  RVS test comments
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
     - Power Stress Test
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


  
  
  
  
  
  
  
  
  
  
   
