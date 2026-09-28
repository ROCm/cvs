.. meta::
  :description: Run CVS platform host configuration check tests to verify OS, kernel, ROCm, BIOS, PCIe, and GPU firmware versions across AMD Instinct cluster nodes.
  :keywords: CVS, platform, AMD Instinct, ROCm, AMD, GPU, Linux, Ubuntu, BIOS, PCIe, kernel, host configuration

***********************************************
Run CVS platform host configuration check tests
***********************************************

Platform tests validate host OS configuration, BIOS version, firmware, driver, PCIe settings, and GPU health on each cluster node. Run platform tests before burn-in to catch configuration mismatches early.

.. _platform-set-up-config:

Set up config
=============

1. Copy the platform configuration file:

   .. code:: bash

     cvs config copy platform/host_config.json --output ~/cvs_workspace/platform/host_config.json

2. Edit the file and replace the values with your cluster's actual versions — leave no ``<changeme>`` placeholders:

   - ``os_version``
   - ``kernel_version``
   - ``rocm_version``
   - ``bios_version``

For the complete field reference and expected-value format, see :doc:`/reference/configuration-files/burn-in-diag/platform`.

.. _platform-run-tests:

Run tests
=========

Run host check scripts to validate host-side configurations, such as model load balancing enablement, PCIe checks, kernel version, and ROCm version.

You can list all available host check test cases using the CLI:

.. code:: bash

  cvs list host_configs_cvs

.. code:: text

  Available tests in host_configs_cvs:
    - test_check_os_release
    - test_check_kernel_version
    - test_check_bios_version
    - test_check_rocm_version
    - test_check_gpu_fw_version
    - test_check_pci_realloc
    - test_check_iommu_pt
    - test_check_numa_balancing
    - test_check_online_memory
    - test_check_pci_accelerators
    - test_check_gpu_pcie_speed_width
    - test_check_be_nic_pcie_speed_width
    - test_check_pci_acs
    - test_check_dmesg_driver_errors

Run the platform host check suite:

.. code:: bash

  cvs run host_configs_cvs \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/platform/host_config.json \
    --html=/var/www/html/cvs/host.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/host.log -vvv -s
