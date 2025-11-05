.. meta::
  :description: Configure the RCCL configuration file variables
  :keywords: RCCL, ROCm, install, cvs

*********************************************************************
ROCm Communication Collectives Library (RCCL) test configuration file
*********************************************************************

These RCCL tests are comprehensive benchmarks that validate distributed GPU communication performance across AMD GPU clusters. 
These tests ensure optimal performance for AI training, HPC workloads, and distributed computing.

Here's a code snippet of the ``rccl_config.json`` file for reference:

.. dropdown:: ``rccl_config.json``

  .. code:: json

    {
      "rccl":
      {
        "no_of_nodes": "2",
        "no_of_global_ranks": "16",
        "no_of_local_ranks": "8",
        "ranks_per_node": "8",
        "rccl_dir": "/root/cache/INSTALL/rccl-tests",
        "rccl_tests_dir": "/root/cache/INSTALL/rccl-tests/build",
        "mpi_dir": "/root/cache/INSTALL/ompi-4.1.6/install/bin",
        "mpi_path_var": "/root/cache/INSTALL/ompi-4.1.6/install",
        "rocm_path_var": "/opt/rocm-6.4.1",
        "rccl_path_var": "/root/cache/INSTALL/rccl-tests",
        "cluster_snapshot_debug": "False",
        "env_source_script": "None",
        "rccl_collective": [ "all_reduce_perf", "all_gather_perf", "scatter_perf", "gather_perf", "reduce_scatter_perf", "sendrecv_perf", "alltoall_perf", "alltoallv_perf", "reduce_scatter_perf", "broadcast_perf" ],
        "rccl_algo": [ "ring" ],
        "rccl_protocol": [ "simple" ],
        "qp_scale": [ "1" ],
        "ib_hca_list": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7",
        "net_dev_list": "ens11np0,ens12np0,ens21np0,ens22np0,ens31np0,ens32np0,ens41np0,ens42np0",
        "oob_port": "ens51f0np0",
        "gid_index": "1",
        "qp_count": "1",
        "start_msg_size": "1024",
        "end_msg_size": "16g",
        "step_function": "2",
        "threads_per_gpu": "1",
        "warmup_iterations": "10",
        "no_of_iterations": "1",
        "check_iteration_count": "1",
        "nccl_ib_timeout": "30",
        "ib_rx_queue_len": "8192",
        "ucx_tls": "tcp",
        "hcoll_enable_mcast_all": "0",
        "nccl_cumem_enable": "0",
        "nccl_ib_sl": "0",
        "nccl_ib_tc": "0",
        "nccl_ib_split_data_on_qps": "0",
        "nccl_pxn_disable": "0",
        "nccl_net_plugin": "none",
        "verify_bus_bw": "False",
        "debug_level": "ERROR",
        "rccl_result_file": "/tmp/rccl_result_file.json",
        "_comments_results": "expected results below are for 2 node cluster, will vary based on cluster size",
        "results":
        {
            "all_reduce_perf":
            {
                "bus_bw":
                {
                    "8589934592": "330.00",
                    "17179869184": "350.00"
                }
            },
            "all_gather_perf":
            {
                "bus_bw":
                {
                    "8589934592": "330.00",
                    "17179869184": "350.00"
                }
            },
            "gather_perf":
            {
                "bus_bw":
                {
                    "8589934592": "330.00",
                    "17179869184": "350.00"
                }
            },
            "scatter_perf":
            {
                "bus_bw":
                {
                    "8589934592": "330.0",
                    "17179869184": "350.0"
                }
            },
            "reduce_perf":
            {
                "bus_bw":
                {
                    "8589934592": "300.0",
                    "17179869184": "310.00"
                }
            },
            "reduce_scatter_perf":
            {
                "bus_bw":
                {
                    "8589934592": "340.0",
                    "17179869184": "360.00"
                }
            },
            "alltoall_perf":
            {
                "bus_bw":
                {
                    "8589934592": "45.00",
                    "17179869184": "50.00"
                }
            },
            "alltoallv_perf":
            {
                "bus_bw":
                {
                    "8589934592": "45.00",
                    "17179869184": "50.00"
                }
            },
            "sendrecv_perf":
            {
                "bus_bw":
                {
                    "8589934592": "47.00",
                    "17179869184": "48.00"
                }
            },
            "broadcast_perf":
            {
                "bus_bw":
                {
                    "8589934592": "310.00",
                    "17179869184": "312.00"
                }
            }
          }
      }

    }

Parameters
==========

Here's an exhaustive list of the available parameters in the RCCL configuration file:

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``no_of_nodes``
     - 2
     - Number of nodes in a cluster
   * - ``no_of_global_ranks``
     - 16
     - Total MPI ranks across the entire cluster
   * - ``no_of_local_ranks``
     - 8
     - MPI ranks running on each individual node  
   * - ``ranks_per_node``
     - 8
     - Confirms GPUs per node
   * - ``rccl_dir``
     - ``/opt/rccl-tests/``
     - Directory where RCCL is installed
   * - ``rccl_tests_dir``
     - ``/opt/rccl-tests/build``
     - Directory where RCCL tests are installed
   * - ``mpi_dir``
     - ``/usr/bin``
     - Directory where mpi bin is located
   * - ``mpi_path_var``
     - ``/usr``
     - Directory where mpi is located
   * - ``rocm_path_var``
     - ``/opt/rocm-6.4.2/``
     - Path to ROCm installation
   * - ``rccl_path_var``
     - ``/opt/rccl-tests/``
     - Directory where RCCL tests are located
   * - ``cluster_snapshot_debug``
     - False
     - Enables/disables cluster debug snapshot
   * - ``env_source_script``
     - None
     - Path to environment setup script
   * - ``rccl_collective``
     - Values:
        * ``all_reduce_perf``
        * ``all_gather_perf``
        * ``scatter_perf``
        * ``gather_perf``
        * ``reduce_scatter_perf``
        * ``sendrecv_perf``
        * ``alltoall_perf``
        * ``alltoallv_perf``
        * ``reduce_scatter_perf``
        * ``broadcast_perf``
     - RCCL tests list
   * - ``rccl_algo``
     - Values
        * ``ring``
        * ``tree`` 
     - Communication algorithms
   * - ``rccl_protocol``
     - Simple
     - Communication protocol
   * - ``qp_scale``
     - 1, 2
     - Queue pair scaling factors
   * - ``ib_hca_list``
     - Values:
        * ``bnxt_re0``
        * ``bnxt_re1``
        * ``bnxt_re2``
        * ``bnxt_re3``
        * ``bnxt_re4``
        * ``bnxt_re5``
        * ``bnxt_re6``
        * ``bnxt_re7`` 
     - List of IB hosts
   * - ``net_dev_list``
     - Values:
        * ``ens28np0``
        * ``ens27np0``
        * ``ens25np0``
        * ``ens26np0``
        * ``ens24np0``
        * ``ens23np0``
        * ``ens21np0``
        * ``ens22np0``
     - List of network hosts
   * - ``oob_port``
     - eth0
     - Out of band port
   * - ``gid_index``
     - 1
     - Global ID index for InfiniBand
   * - ``start_msg_size``
     - 1024
     - Start with 1KB messages
   * - ``end_msg_size``
     - 16g
     - End with 16GB messages
   * - ``step_function``
     - 2
     - Double message size each step
   * - ``threads_per_gpu``
     - 1
     - One thread per GPU
   * - ``warmup_iterations``
     - 10
     - Warmup runs
   * - ``no_of_iterations``
     - 1
     - Number of iterations to run the RCCL tests
   * - ``check_iteration_count``
     - 1
     - Verification iteration
   * - ``nccl_ib_timeout``
     - 30
     -  InfiniBand timeout
   * - ``ib_rx_queue_len``
     - 8192
     - Receive queue length
   * - ``ucx_tls``
     - Tcp
     - UCX transport layer
   * - ``hcoll_enable_mcast_all``
     - 0
     - Disable multicast
   * - ``nccl_cumem_enable``
     - 0
     - CUDA memory optimizations
   * - ``nccl_ib_sl``
     - 0
     - InfiniBand service level
   * - ``nccl_ib_tc``
     - 0
     - InfiniBand traffic class
   * - ``nccl_ib_split_data_on_qps``
     - 0
     - Don't split data across queue pairs
   * - ``nccl_pxn_disable``
     - 0, 1
     - PXN disable options
   * - ``nccl_net_plugin``
     - None
     - Network plugin
   * - ``verify_bus_bw``
     - False
     - Verify bus bandwidth
   * - ``verify_bw_dip``
     - True
     - Check for bandwidth drops
   * - ``verify_lat_dip``
     - True
     - Check for latency spikes
   * - ``debug``
     - ERROR
     - Set the debug level
   * - ``rccl_result_file``
     - ``/tmp/rccl_result_file.json``
     - Path where RCCL results are captured
   * - ``_comments_results``
     - N/A
     - Expected results are for the two-node cluster and vary based on cluster size
   * - ``all_reduce_perf``
     - N/A
     - Global reduction: sum/max/min across all GPUs
   * - ``all_gather_perf``
     - N/A
     - All GPUs receive the complete combined dataset from all ranks
   * - ``gather_perf``
     - N/A
     - Collect data from all ranks to one root rank
   * - ``scatter_perf``
     - N/A
     - Distribute data from one rank to all ranks
   * - ``reduce_perf``
     - N/A
     - Reduce operation 
   * - ``reduce_scatter_perf``
     - N/A
     - Reduce operation followed by scatter
   * - ``alltoall_perf``
     - N/A
     - Every rank sends unique data to every other rank
   * - ``alltoallv_perf``
     - N/A
     - Variable-size all-to-all exchange
   * - ``sendrecv_perf``
     - N/A
     - Point-to-point communication between rank pairs
   * - ``broadcast_perf``
     - N/A
     - One-to-all communication
   * - ``bus_bw``
     - N/A
     - Bus bandwidth