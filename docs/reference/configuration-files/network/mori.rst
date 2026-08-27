.. meta::
  :description: Configure the variables in the MORI configuration file
  :keywords: communication, ROCm, install, cvs, MORI, RDMA, multi-node, MI35X

*************************************************
MORI configuration file
*************************************************

MORI (Memory-Oriented RDMA Interface) tests validate RDMA communication performance across multi-node AMD GPU clusters.
These tests ensure optimal bandwidth, latency, and reliability for distributed workloads that require high-speed inter-node communication.

The MORI tests check:

- **Container orchestration**: Docker setup with MORI libraries for RDMA communication
- **RDMA device configuration**: Proper setup of RDMA devices and network interfaces
- **Read/Write operations**: Bandwidth and latency metrics for RDMA read and write operations
- **Network interface types**: Support for various NICs (AINIC, Thor2, CX7)
- **Multi-node coordination**: Master-worker communication and synchronization
- **Result verification**: Expected bandwidth and latency thresholds

Change the parameters as needed in the MORI configuration file: ``mori_config.json`` for multi-node RDMA testing.

.. note::

  - ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username.
  - Replace all ``<changeme>`` placeholders with actual values for your cluster.

``mori_config.json``
====================

Here's a code snippet of the ``mori_config.json`` file for reference:

.. dropdown:: ``mori_config.json``

  .. code:: json

    {
        "no_of_nodes": "2",
        "container_image": "<changeme>",
        "container_name": "mori_container",
        "oob_port": "<changeme>",
        "mori_device_list": "<changeme>",
        "mori_dir": "<changeme>",
        "torchlib_dir": "/usr/local/lib/python3.12/dist-packages/torch/lib",
        "master_addr": "<changeme>",
        "master_port": "1234",
        "nic_type": "<changeme>",
        "log_dir": "/home/{user-id}/LOGS/mori",
        "container_config": {
            "device_list": [ "/dev/dri", "/dev/kfd" ],
            "volume_dict": {
                "/home/{user-id}": "/home/{user-id}",
                "/it-share/models": "/root/models",
                "/usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-164.g21c72dcad": 
                    "/usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-164.g21c72dcad",
                "/usr/lib/x86_64-linux-gnu/libionic.so.1": 
                    "/usr/lib/x86_64-linux-gnu/libionic.so.1",
                "/usr/lib/x86_64-linux-gnu/libionic.so": 
                    "/usr/lib/x86_64-linux-gnu/libionic.so",
                "/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so": 
                    "/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so",
                "/etc/libibverbs.d/ionic.driver": 
                    "/etc/libibverbs.d/ionic.driver"
            },
            "env_dict": {}
        },
        "expected_results": {
            "read": {
                "16384,128,1": {
                    "524288": {
                        "max_bw": "45.0",
                        "avg_lat": "1500"
                    },
                    "1048576": {
                        "max_bw": "46.0",
                        "avg_lat": "2500"
                    }
                }
            },
            "write": {
                "128": {
                    "524288": {
                        "max_bw": "45.0",
                        "avg_lat": "1500"
                    },
                    "1048576": {
                        "max_bw": "46.0",
                        "avg_lat": "2500"
                    }
                }
            }
        }
    }

Parameters
==========

Use the parameters in this table to configure the MORI configuration file.

.. |br| raw:: html

    <br />

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``no_of_nodes``
     - 2
     - Number of nodes in the MORI test cluster
   * - ``container_image``
     - ``<changeme>``
     - Docker container image with MORI libraries for RDMA testing (e.g., rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1224)
   * - ``container_name``
     - mori_container
     - Name of the Docker container instance
   * - ``oob_port``
     - ``<changeme>``
     - Out-of-band network interface for control plane communication (e.g., eno0, eno1, enp0s0)
   * - ``mori_device_list``
     - ``<changeme>``
     - Comma-separated list of RDMA devices to use for testing (e.g., rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7)
   * - ``mori_dir``
     - ``<changeme>``
     - Directory containing MORI binaries and libraries (e.g., /sgl-workspace/mori)
   * - ``torchlib_dir``
     - /usr/local/lib/python3.12/dist-packages/torch/lib
     - Directory containing PyTorch libraries (for dependencies)
   * - ``master_addr``
     - ``<changeme>``
     - IP address or hostname of the master node
   * - ``master_port``
     - 1234
     - Port for master-worker communication
   * - ``nic_type``
     - ``<changeme>``
     - Network interface card type: ainic (AMD Pensando AINIC), thor2 (Thor2 adapter), or cx7 (Mellanox ConnectX-7)
   * - ``log_dir``
     - ``/home/{user-id}/LOGS/mori``
     - Directory for MORI test logs
   * - ``container_config.`` |br| ``device_list``
     - Values: |br| - ``"/dev/dri"`` |br| - ``"/dev/kfd"``
     - List of device paths to mount in the container for GPU access
   * - ``container_config.`` |br| ``volume_dict``
     - Multiple mappings
     - Dictionary mapping host paths to container paths for volume mounts
   * - ``/home/{user-id}``
     - ``/home/{user-id}``
     - User home directory mount
   * - ``/it-share/models``
     - ``/root/models``
     - Models directory mount
   * - ``/usr/lib/x86_64-`` |br| ``linux-gnu/libionic.`` |br| ``so.1.0.54.0-164.`` |br| ``g21c72dcad``
     - ``/usr/lib/x86_64-`` |br| ``linux-gnu/libionic.`` |br| ``so.1.0.54.0-164.`` |br| ``g21c72dcad``
     - AMD Pensando AINIC library (specific version)
   * - ``/usr/lib/x86_64-`` |br| ``linux-gnu/libionic.`` |br| ``so.1``
     - ``/usr/lib/x86_64-`` |br| ``linux-gnu/libionic.`` |br| ``so.1``
     - AMD Pensando AINIC library (versioned symlink)
   * - ``/usr/lib/x86_64-`` |br| ``linux-gnu/libionic.so``
     - ``/usr/lib/x86_64-`` |br| ``linux-gnu/libionic.so``
     - AMD Pensando AINIC library (unversioned symlink)
   * - ``/usr/lib/x86_64-`` |br| ``linux-gnu/libibverbs/`` |br| ``libionic-rdmav34.so``
     - ``/usr/lib/x86_64-`` |br| ``linux-gnu/libibverbs/`` |br| ``libionic-rdmav34.so``
     - RDMA verbs provider for AMD Pensando AINIC
   * - ``/etc/libibverbs.d/`` |br| ``ionic.driver``
     - ``/etc/libibverbs.d/`` |br| ``ionic.driver``
     - Driver configuration file for AMD Pensando NICs
   * - ``container_config.`` |br| ``env_dict``
     - Empty
     - Dictionary of environment variables to set in the container
   * - ``expected_results.read``
     - Nested dictionary
     - Expected results for RDMA read operations with format: ``"num_threads,block_size,num_blocks"`` → ``transfer_size`` → ``max_bw``/``avg_lat``
   * - ``expected_results.`` |br| ``read.16384,128,1.`` |br| ``524288.max_bw``
     - 45.0
     - Expected maximum bandwidth (GB/s) for read operation with 16384 threads, block size 128, 1 block, and 524288 byte transfer
   * - ``expected_results.`` |br| ``read.16384,128,1.`` |br| ``524288.avg_lat``
     - 1500
     - Expected average latency (microseconds) for read operation with 16384 threads, block size 128, 1 block, and 524288 byte transfer
   * - ``expected_results.`` |br| ``read.16384,128,1.`` |br| ``1048576.max_bw``
     - 46.0
     - Expected maximum bandwidth (GB/s) for read operation with 16384 threads, block size 128, 1 block, and 1048576 byte transfer
   * - ``expected_results.`` |br| ``read.16384,128,1.`` |br| ``1048576.avg_lat``
     - 2500
     - Expected average latency (microseconds) for read operation with 16384 threads, block size 128, 1 block, and 1048576 byte transfer
   * - ``expected_results.write``
     - Nested dictionary
     - Expected results for RDMA write operations with format: ``"num_threads"`` → ``transfer_size`` → ``max_bw``/``avg_lat``
   * - ``expected_results.`` |br| ``write.128.`` |br| ``524288.max_bw``
     - 45.0
     - Expected maximum bandwidth (GB/s) for write operation with 128 threads and 524288 byte transfer
   * - ``expected_results.`` |br| ``write.128.`` |br| ``524288.avg_lat``
     - 1500
     - Expected average latency (microseconds) for write operation with 128 threads and 524288 byte transfer
   * - ``expected_results.`` |br| ``write.128.`` |br| ``1048576.max_bw``
     - 46.0
     - Expected maximum bandwidth (GB/s) for write operation with 128 threads and 1048576 byte transfer
   * - ``expected_results.`` |br| ``write.128.`` |br| ``1048576.avg_lat``
     - 2500
     - Expected average latency (microseconds) for write operation with 128 threads and 1048576 byte transfer
