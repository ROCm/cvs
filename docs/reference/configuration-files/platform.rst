.. meta::
  :description: Configure the Platform configuration file variables
  :keywords: platform, ROCm, install, cvs

********************************
Platform test configuration file
********************************

Here's a code snippet of the ``host_config.json`` file for reference:

.. dropdown:: ``host_config.json``
     
  .. code:: json
    
    {

        "host":
        {
          "os_version": "Ubuntu 24.04.1 LTS",
          "kernel_version": "6.8.0-60-generic",
          "rocm_version": "6.4.1",
          "bios_version": "20171212",
          "pci_realloc": "off",
          "online_memory": "1.3T",
          "gpu_count": "8",
          "gpu_pcie_speed": "32",
          "gpu_pcie_width": "16",
          "fw_dict":
          {
              "CP_MEC1": "32945",
              "CP_MEC2": "32945",
              "RLC": "65",
              "SDMA0": "24",
              "SDMA1": "24",
              "VCN": "09.11.70.09",
              "RLC_RESTORE_LIST_GPM_MEM": "4",
              "RLC_RESTORE_LIST_SRM_MEM": "4",
              "RLC_RESTORE_LIST_CNTL": "4",
              "PSP_SOSDRV": "00.36.02.56",
              "TA_RAS": "1B.36.02.14",
              "TA_XGMI": "20.00.00.14",
              "PM": "07.85.11.01"
            }
        }
      }       

Parameters
==========

Here's an exhaustive list of the available parameters in the Platform configuration file.

.. list-table::
   :widths: 15 10 30
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
     - 6.4.1
     - ROCm version   
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

