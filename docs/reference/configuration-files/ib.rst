.. meta::
  :description: Configure the InfiniBand configuration file variables
  :keywords: InfiniBand, ROCm, install, cvs

********************************************
InfiniBand (IB Perf) test configuration file
********************************************

IB Perf and latency tests measure network performance. Perf tests measure throughput (bandwidth), and latency tests measure delay. 

Here's a code snippet of the ``ibperf_config.json`` file for reference:

.. note::

  In this configuration file, ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username. 

.. dropdown:: ``ibperf_config.json``

  .. code:: json
      
    {
        "ibperf":
        {
          "install_perf_package": "True",
          "install_dir": "/home/{user-id}/",
          "rocm_dir": "/opt/rocm",
          "qp_count_list": [ "8", "16" ],
          "ib_bw_test_list": [ "ib_write_bw", "ib_send_bw"],
          "ib_lat_test_list": [ "ib_write_lat", "ib_send_lat", "ib_read_lat" ],
          "msg_size_list": [ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ],
          "gid_index": "3",
          "port_no": "1516",
          "duration": "30",
          "verify_bw": "True",
          "expected_results":
          {
          "ib_write_bw":
          {
              "8192":
                    {
            "8": "180.0",
            "16": "200.0"
                    },
              "8388608":
                    {
            "8": "280.0",
            "16": "300.0"
                    }
    
                }
          }
        }
    
    }

Parameters
==========

Here's an exhaustive list of the available parameters in the IB Perf configuration file.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``install_perf_package``
     - True
     - Enable automatic installation of InfiniBand performance tools
   * - ``install_dir``
     - ``/home/{user-id}/``
     - Installation directory for performance testing tools
   * - ``rocm_dir``
     - ``/opt/rocm``
     - ROCm installation path   
   * - ``qp_count_list``
     - Values:
        - 8 
        - 16
     - Queue Pair counts to test
   * - ``ib_bw_test_list``
     - Values:
        - ``ib_write_bw`` 
        - ``ib_send_bw``
     - IB bandwidth tests
   * - ``ib_lat_test_list``
     - Values:
        - ``ib_write_lat`` 
        - ``ib_send_lat`` 
        - ``ib_read_lat``
     - IB latency tests
   * - ``msg_size_list``
     - Values:
        - 2 
        - 4 
        - 8 
        - 16 
        - 32
        - 64 
        - 128 
        - 256 
        - 512
        - 1024 
        - 2048 
        - 4096 
        - 8192 
        - 16384
        - 32768
        - 65536 
     - Test message sizes in bytes
   * - ``gid_index``
     - 3
     - Global Identifier index for InfiniBand
   * - ``port_no``
     - 1516
     - Port number for test communication
   * - ``duration``
     - 30
     - Test duration in seconds
   * - ``verify_bw``
     - True
     - Bandwidth verification 

The ``expected_results`` section also contains the ``ib_write_bw`` parameter. It describes the bandwith expectation, and it has these default values in the JSON file:

.. dropdown:: ib_write_bw

  .. code:: json

    "8192":
                    {
            "8": "180.0",
            "16": "200.0"
                    },
              "8388608":
                    {
            "8": "280.0",
            "16": "300.0"
                    }




