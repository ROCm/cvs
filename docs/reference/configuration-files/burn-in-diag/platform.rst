.. meta::
  :description: Reference for the CVS platform test configuration file, covering OS version, kernel, ROCm version, BIOS, PCIe, and firmware validation parameters.
  :keywords: CVS, platform, ROCm, OS version, kernel, BIOS, PCIe, firmware, GPU, AMD, configuration, JSON, host check

*******************************************************************************
Platform test configuration file reference for Cluster Validation Suite (CVS)
*******************************************************************************

Run host check scripts to validate host-side configurations, such as model load balancing enablement, PCIe checks, kernel version, and ROCm version.

The following sample shows the ``host_config.json`` structure. Replace the values with your cluster's actual versions before running:

See :doc:`/how-to/test-suites/burn-in-diag/platform` for more information on running these tests.

.. dropdown:: ``host_config.json``
     
  .. code:: json
    
    {
        "_comment": "Replace every <changeme> with this cluster's value before running. Keys whose names start with _ are examples or comments and are ignored.",
        "host":
        {
          "os_version": "<changeme>",
          "_example_os_version": "Ubuntu 24.04.1 LTS",
          "kernel_version": "<changeme>",
          "_example_kernel_version": "6.8.0-60-generic",
          "rocm_version": "<changeme>",
          "_example_rocm_version": "7.0.2",
          "bios_version": "<changeme>",
          "_example_bios_version": "20171212",
          "pci_realloc": "<changeme>",
          "_example_pci_realloc": "off",
          "online_memory": "<changeme>",
          "_example_online_memory": "1.3T",
          "gpu_count": "<changeme>",
          "_example_gpu_count": "8",
          "gpu_pcie_speed": "<changeme>",
          "_example_gpu_pcie_speed": "32",
          "gpu_pcie_width": "<changeme>",
          "_example_gpu_pcie_width": "16",
          "nic_pcie_speed": "<changeme>",
          "_example_nic_pcie_speed": "32",
          "_comment_nic_pcie_speed": "If the NIC is connected via UALink (for example, MI450) delete nic_pcie_speed. Otherwise replace <changeme> with the NIC PCIe speed.",
          "nic_pcie_width": "<changeme>",
          "_example_nic_pcie_width": "16",
          "_comment_nic_pcie_width": "If the NIC is connected via UALink (for example, MI450) delete nic_pcie_width. Otherwise replace <changeme> with the NIC PCIe width.",
          "nic_link_speed": "400000",
          "_comment_nic_link_speed": "Expected backend NIC link speed in Mb/s as reported by /sys/class/net/<iface>/speed. 400000 (400 Gb/s) is the AMD Instinct acceptance requirement; change it only for a different NIC generation.",
          "nic_link_interfaces": ["<changeme>"],
          "_example_nic_link_interfaces": ["enp28s0np0", "enp62s0np0", "enp79s0np0", "enp96s0np0", "enp158s0np0", "enp190s0np0", "enp206s0np0", "enp222s0np0"],
          "_comment_nic_link_interfaces": "Backend (scale-out) NIC netdev names as listed in /sys/class/net (not RDMA device names such as rocep28s0 or mlx5_0). Set to [] to auto-detect backend NICs on each node. Auto-detect only sees NICs that have a netdev and an RDMA device and fails a node with fewer NICs than another node; list the names here to also catch a NIC missing on every node or on a single-node run.",
          "fw_dict":
          {
              "CP_MEC1": "<changeme>",
              "_example_CP_MEC1": "32945",
              "CP_MEC2": "<changeme>",
              "_example_CP_MEC2": "32945",
              "RLC": "<changeme>",
              "_example_RLC": "65",
              "SDMA0": "<changeme>",
              "_example_SDMA0": "24",
              "SDMA1": "<changeme>",
              "_example_SDMA1": "24",
              "VCN": "<changeme>",
              "_example_VCN": "09.11.70.09",
              "RLC_RESTORE_LIST_GPM_MEM": "<changeme>",
              "_example_RLC_RESTORE_LIST_GPM_MEM": "4",
              "RLC_RESTORE_LIST_SRM_MEM": "<changeme>",
              "_example_RLC_RESTORE_LIST_SRM_MEM": "4",
              "RLC_RESTORE_LIST_CNTL": "<changeme>",
              "_example_RLC_RESTORE_LIST_CNTL": "4",
              "PSP_SOSDRV": "<changeme>",
              "_example_PSP_SOSDRV": "00.36.02.56",
              "TA_RAS": "<changeme>",
              "_example_TA_RAS": "1B.36.02.14",
              "TA_XGMI": "<changeme>",
              "_example_TA_XGMI": "20.00.00.14",
              "PM": "<changeme>",
              "_example_PM": "07.85.11.01"
            }
        }
      }       

Parameters
==========

The following parameters are available in the platform configuration file. Set each to the expected value for your cluster — the test compares the actual system state against these values.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``os_version``
     - Ubuntu 24.04.1 LTS
     - Version of OS
   * - ``kernel_version``
     - ``6.8.0-60-generic``
     - Version of kernel
   * - ``rocm_version``
     - ``<cluster-specific>``
     - ROCm version installed on the cluster nodes
   * - ``bios_version``
     - ``20171212``
     - BIOS version
   * - ``pci_realloc``
     - Off
     - PCI reallocation
   * - ``online_memory``
     - 1.3T
     - Available system RAM
   * - ``gpu_count``
     - 8
     - Number of GPUs
   * - ``gpu_pcie_speed``
     - 32
     - PCIe speed
   * - ``gpu_pcie_width``
     - 16
     - Width of PCIe
   * - ``nic_link_speed``
     - ``400000``
     - Expected backend NIC link speed in Mb/s (``/sys/class/net/<iface>/speed``)
   * - ``nic_link_interfaces``
     - ``<cluster-specific>``
     - Backend NIC netdev names; ``[]`` auto-detects and fails a node with fewer NICs than another node
   * - ``CP_MEC1``
     - 32945
     - Compute Pipeline MicroEngine Controller 1 firmware
   * - ``CP_MEC2``
     - 32945
     - Compute Pipeline MicroEngine Controller 2 firmware
   * - ``RLC``
     - 65
     - RunList Controller firmware
   * - ``SDMA0``
     - 24
     - System DMA Engine 0 firmware
   * - ``SDMA1``
     - 24
     - System DMA Engine 1 firmware
   * - ``VCN``
     - ``09.11.70.09``
     - Video Core Next firmware
   * - ``RLC_RESTORE_LIST_GPM_MEM``
     - 4
     - RunList Controller restore mechanisms for power state transitions
   * - ``RLC_RESTORE_LIST_SRM_MEM``
     - 4
     - RunList Controller restore mechanisms for power state transitions
   * - ``RLC_RESTORE_LIST_CNTL``
     - 4
     - RunList Controller restore mechanisms for power state transitions
   * - ``PSP_SOSDRV``
     - ``00.36.02.56``
     - Platform Security Processor SOS driver
   * - ``TA_RAS``
     - ``1B.36.02.14``
     - Trusted application for RAS (Reliability, Availability, and Serviceability)
   * - ``TA_XGMI``
     - ``20.00.00.14``
     - Trusted application for xGMI (External Global Memory Interconnect)
   * - ``PM``
     - ``07.85.11.01``
     - Power Management firmware

