.. meta::
  :description: Reference for the MORI network configuration file in CVS, covering RDMA read/write bandwidth, latency thresholds, container setup, and NIC device options.
  :keywords: CVS, MORI, RDMA, ROCm, network, bandwidth, latency, InfiniBand, AMD, GPU, JSON, configuration, multi-node

*************************************************************************
MORI network configuration file reference for Cluster Validation Suite (CVS)
*************************************************************************

MORI (Memory-Oriented RDMA Interface) tests validate RDMA communication performance across multi-node AMD GPU clusters.
These tests ensure optimal bandwidth, latency, and reliability for distributed workloads that require high-speed inter-node communication.

The MORI tests check:

- **Orchestration**: runs in a Docker container with the MORI libraries, or directly on hosts that have MORI installed
- **RDMA device configuration**: Proper setup of RDMA devices and network interfaces
- **Read/Write operations**: Bandwidth and latency metrics for RDMA read and write operations
- **Network interface types**: Support for various NICs (AINIC, Thor2, CX7)
- **Multi-node coordination**: Master-worker communication and synchronization
- **Result verification**: Expected bandwidth and latency thresholds

Change the parameters as needed in the MORI configuration file: ``mori_config.json`` for multi-node RDMA testing.

.. note::

  - ``{user-id}`` resolves to the current username at runtime. You can also manually change this value to your username.
  - Replace all ``<changeme>`` placeholders with actual values for your cluster.

``mori_config.json``
====================

Here's a code snippet of the ``mori_config.json`` file for reference:

.. dropdown:: ``mori_config.json``

  .. code:: json

    {
        "no_of_nodes": "2",
        "orchestrator": "container",
        "container": {
            "image": "<changeme>",
            "name": "mori_container",
            "lifetime": "per_run",
            "runtime": {
                "name": "docker",
                "args": {
                    "devices": [ "/dev/dri", "/dev/kfd" ],
                    "volumes": [
                        "/home/{user-id}:/home/{user-id}",
                        "/it-share/models:/root/models",
                        "/usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-164.g21c72dcad:/usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-164.g21c72dcad",
                        "/usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1",
                        "/usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so",
                        "/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so",
                        "/etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver"
                    ]
                }
            }
        },
        "env": {
            "NCCL_SOCKET_IFNAME": "<changeme>",
            "GLOO_SOCKET_IFNAME": "<changeme>",
            "MORI_RDMA_DEVICES": "<changeme>",
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH"
        },
        "mori_dir": "<changeme>",
        "master_addr": "<changeme>",
        "master_port": "1234",
        "nic_type": "<changeme>",
        "log_dir": "/home/{user-id}/LOGS/mori",
        "expected_results": {
            "ibgda_write": {
                "PROCS:2,CTAS:2,THREADS:256,QP_COUNT:4": {
                    "33554432": { "max_bw": "46.0", "avg_lat": "390" },
                    "67108864": { "max_bw": "46.0", "avg_lat": "780.0" }
                }
            },
            "io_read": {
                "BUFF_SIZE:16384,TRANSFER_SIZE:128,QP_COUNT:2": {
                    "524288": { "max_bw": "45.0", "avg_lat": "1500" },
                    "1048576": { "max_bw": "46.0", "avg_lat": "2500" }
                }
            },
            "io_write": {
                "BUFF_SIZE:16384,TRANSFER_SIZE:128,QP_COUNT:2": {
                    "524288": { "max_bw": "45.0", "avg_lat": "1500" },
                    "1048576": { "max_bw": "46.0", "avg_lat": "2500" }
                }
            }
        }
    }

For a baremetal run, set ``"orchestrator": "baremetal"``, omit the ``container`` block, and point ``mori_dir``
and ``env.LD_LIBRARY_PATH`` at the MORI checkout and PyTorch libraries installed on the hosts.

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
   * - ``orchestrator``
     - container
     - ``container`` runs every MORI command inside a container on each node; ``baremetal`` runs it directly on the hosts, which must already have MORI built and its Python dependencies installed. Set it explicitly: it overrides the cluster file's ``orchestrator``.
   * - ``container.image``
     - ``<changeme>``
     - Docker image with the MORI libraries for RDMA testing (for example, ``rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1224``). Container mode only.
   * - ``container.name``
     - mori_container
     - Name of the container on each node
   * - ``container.lifetime``
     - per_run
     - ``per_run`` launches the container at the start of the suite and removes it at the end
   * - ``container.runtime.name``
     - docker
     - Container runtime
   * - ``container.runtime.`` |br| ``args.devices``
     - Values: |br| - ``"/dev/dri"`` |br| - ``"/dev/kfd"``
     - Device paths passed to the container for GPU access
   * - ``container.runtime.`` |br| ``args.volumes``
     - Multiple mounts
     - Volume mounts as ``"host_path:container_path[:options]"`` strings. The sample mounts the user home directory, a models directory, and the AMD Pensando AINIC (``libionic``) user-space libraries and driver file. For Broadcom Thor2, mount the host's RDMA provider read-only instead, for example ``"/usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so:ro"``.
   * - ``env.NCCL_SOCKET_IFNAME``
     - ``<changeme>``
     - Out-of-band network interface for control-plane traffic (for example, ``eno0``)
   * - ``env.GLOO_SOCKET_IFNAME``
     - ``<changeme>``
     - Network interface for the PyTorch Gloo rendezvous; normally the same as ``NCCL_SOCKET_IFNAME``
   * - ``env.MORI_RDMA_DEVICES``
     - ``<changeme>``
     - Comma-separated RDMA devices to use (for example, ``rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7``). ``test_setup_ibv_devices`` checks that each one shows up in ``ibv_devinfo``.
   * - ``env.LD_LIBRARY_PATH``
     - ``/usr/local/lib/`` |br| ``python3.12/dist-packages/`` |br| ``torch/lib:$LD_LIBRARY_PATH``
     - PyTorch library directory prepended to ``LD_LIBRARY_PATH``
   * - ``env``
     - See above
     - Every entry is exported before each MORI command, in both modes. A value of the form ``prefix:$NAME`` or ``$NAME:suffix`` prepends or appends to the existing variable. The suite also prepends ``mori_dir`` to ``PYTHONPATH``.
   * - ``mori_dir``
     - ``<changeme>``
     - MORI checkout with the built examples and Python tests (for example, ``/sgl-workspace/mori`` in the container image)
   * - ``master_addr``
     - ``<changeme>``
     - Cluster host that acts as node rank 0 for the MORI-IO benchmark. If empty, the cluster's head node is used.
   * - ``master_port``
     - 1234
     - TCP port for the torchrun rendezvous on ``master_addr``
   * - ``nic_type``
     - ``<changeme>``
     - Network interface card type: ainic (AMD Pensando AINIC), thor2 (Broadcom Thor2), or cx7 (NVIDIA ConnectX-7)
   * - ``log_dir``
     - ``/home/{user-id}/LOGS/mori``
     - Host directory for MORI test logs; each run writes to a timestamped subdirectory. In container mode it must be inside a mounted volume.
   * - ``expected_results.`` |br| ``ibgda_write``
     - Nested dictionary
     - Minimum bandwidth for the IBGDA ``dist_write`` test, keyed by ``"PROCS:<n>,CTAS:<n>,THREADS:<n>,QP_COUNT:<n>"`` → message size in bytes → ``max_bw`` (GB/s)
   * - ``expected_results.`` |br| ``io_read``
     - Nested dictionary
     - Thresholds for the MORI-IO read test, keyed by ``"BUFF_SIZE:<n>,TRANSFER_SIZE:<n>,QP_COUNT:<n>"`` → message size in bytes → ``max_bw`` (minimum average bandwidth, GB/s) and ``avg_lat`` (maximum average latency, microseconds)
   * - ``expected_results.`` |br| ``io_write``
     - Nested dictionary
     - Thresholds for the MORI-IO write test, in the same format as ``io_read``

A test case whose key is not in ``expected_results`` runs and records its results without a threshold check.
