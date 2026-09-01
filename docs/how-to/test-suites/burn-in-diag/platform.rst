.. meta::
  :description: Run platform host configuration checks
  :keywords: CVS, platform

**************
Platform tests
**************

.. _platform-set-up-config:

Set up config
=============

1. Copy the platform configuration file:

   .. code:: bash

     cvs config copy platform/host_config.json --output ~/cvs_workspace/platform/host_config.json

2. Edit the file and set expected values for your cluster:

   - ``os_version``
   - ``kernel_version``
   - ``rocm_version``
   - ``bios_version``

Full parameter list: :doc:`/reference/configuration-files/burn-in-diag/platform`.

.. _platform-run-tests:

Run tests
=========

The host check scripts can validate various host-side configurations, such as model load balancing enablement, PCIe checks, kernel version, and ROCm version.

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

Here's the test script:

.. code:: bash

  cvs run host_configs_cvs --cluster_file input/cluster_file/cluster.json --config_file input/config_file/platform/host_config.json --html=/var/www/html/cvs/host.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s
