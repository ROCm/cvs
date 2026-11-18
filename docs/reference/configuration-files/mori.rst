.. meta::
  :description: Configure the variables in the MORI configuration file
  :keywords: communication, ROCm, install, cvs, MORI, RDMA, multi-node, MI35X

*************************************************
MORI configuration file
*************************************************

MORI (Multi-node RDMA Optimization and Remote Inference) tests validate RDMA communication performance across multi-node AMD GPU clusters.
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

Configuration Parameters
========================

Here's an exhaustive list of the available parameters in the ``mori_config.json`` configuration file:

General Configuration
---------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default/Example values
     - Description
   * - ``no_of_nodes``
     - 2
     - Number of nodes in the MORI test cluster
   * - ``container_image``
     - rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1224
     - Docker container image with MORI libraries for RDMA testing
   * - ``container_name``
     - mori_container
     - Name of the Docker container instance
   * - ``oob_port``
     - eno0
     - Out-of-band network interface for control plane communication
   * - ``mori_device_list``
     - rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
     - Comma-separated list of RDMA devices to use for testing
   * - ``mori_dir``
     - /sgl-workspace/mori
     - Directory containing MORI binaries and libraries
   * - ``torchlib_dir``
     - /usr/local/lib/python3.12/dist-packages/torch/lib
     - Directory containing PyTorch libraries (for dependencies)
   * - ``master_addr``
     - <changeme>
     - IP address or hostname of the master node
   * - ``master_port``
     - 1234
     - Port for master-worker communication
   * - ``nic_type``
     - ainic, thor2, or cx7
     - Network interface card type (AMD Pensando AINIC, Thor2, or Mellanox CX7)
   * - ``log_dir``
     - ``/home/{user-id}/LOGS/mori``
     - Directory for MORI test logs
   * - ``container_config.device_list``
     - ``[ "/dev/dri", "/dev/kfd" ]``
     - List of device paths to mount in the container for GPU access
   * - ``container_config.volume_dict``
     - Multiple mappings
     - Dictionary mapping host paths to container paths for volume mounts
   * - ``container_config.env_dict``
     - Empty
     - Dictionary of environment variables to set in the container

Expected Results Configuration
-------------------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Result parameters
     - Structure
     - Description
   * - ``expected_results.read``
     - Nested dictionary
     - Expected results for RDMA read operations
   * - ``expected_results.write``
     - Nested dictionary
     - Expected results for RDMA write operations
   * - ``max_bw``
     - Bandwidth in GB/s
     - Maximum expected bandwidth for the operation
   * - ``avg_lat``
     - Latency in microseconds
     - Average expected latency for the operation

Read Operation Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

The read operation results are organized by test configuration:

- **Key format**: ``"num_threads,block_size,num_blocks"``
  
  - ``num_threads``: Number of concurrent threads (e.g., 16384, 128)
  - ``block_size``: Block size parameter (e.g., 128)
  - ``num_blocks``: Number of blocks (e.g., 1)

- **Transfer sizes**: Nested under configuration key (e.g., 524288, 1048576 bytes)

Example:

.. code:: json

    "read": {
        "16384,128,1": {
            "524288": {
                "max_bw": "45.0",
                "avg_lat": "1500"
            }
        }
    }

Write Operation Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The write operation results are organized similarly:

- **Key format**: ``"num_threads"`` (e.g., "128")
- **Transfer sizes**: Nested under configuration key

Example:

.. code:: json

    "write": {
        "128": {
            "524288": {
                "max_bw": "45.0",
                "avg_lat": "1500"
            }
        }
    }

MORI Overview
=============

**What is MORI?**

MORI (Multi-node RDMA Optimization and Remote Inference) is a benchmarking and validation framework for RDMA communication:

- Tests RDMA read and write operations
- Validates bandwidth and latency performance
- Supports multiple RDMA devices per node
- Multi-threaded performance testing
- Configurable block sizes and transfer sizes

**Key Capabilities**

- **Multi-Device Support**: Test all RDMA devices simultaneously
- **Scalable Testing**: From 2 nodes to large clusters
- **Performance Metrics**: Bandwidth (GB/s) and latency (μs)
- **Flexible Configuration**: Adjustable thread counts, block sizes, transfer sizes
- **Network Validation**: Ensure RDMA setup is optimal

RDMA Communication Patterns
============================

**RDMA Read Operations**

Direct memory read from remote node:

- Source node reads data from target node's memory
- No CPU involvement on target node (zero-copy)
- Lower latency for small transfers
- Important for distributed inference and parameter serving

**RDMA Write Operations**

Direct memory write to remote node:

- Source node writes data to target node's memory
- No CPU involvement on target node (zero-copy)
- Higher bandwidth for large transfers
- Important for gradient synchronization and all-reduce

**Performance Characteristics**

- **Bandwidth**: Typically 40-50 GB/s per device with optimal configuration
- **Latency**: Sub-microsecond for small transfers, ~1-3ms for larger transfers
- **Scalability**: Linear scaling with number of RDMA devices

Network Interface Card Types
=============================

**AMD Pensando AINIC**

- AMD's accelerated network interface card
- Optimized for AMD GPUs
- Integrated offload engines
- Set ``nic_type: "ainic"``

**Thor2**

- High-performance network adapter
- Advanced RDMA capabilities
- Set ``nic_type: "thor2"``

**Mellanox ConnectX-7 (CX7)**

- Industry-standard InfiniBand/RoCE adapter
- 400Gb/s capability
- Wide compatibility
- Set ``nic_type: "cx7"``

RDMA Device Configuration
==========================

**Device List Format**

Comma-separated RDMA device names:

.. code:: json

    {
        "mori_device_list": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"
    }

**Device Naming**

- Standard format: ``rdmaX`` where X is 0-7
- Verify available devices: ``ibstat`` or ``rdma link``
- Ensure all devices are active and in ACTIVE state

**Device-GPU Affinity**

For optimal performance:

- Each RDMA device should be on the same NUMA node as its corresponding GPU
- Typically: rdma0 with GPU0, rdma1 with GPU1, etc.
- Check affinity: ``nvidia-smi topo -m`` or ``rocm-smi --showtopo``

Out-of-Band Network
====================

**Purpose**

The OOB (Out-of-Band) network is used for:

- Control plane communication
- Test coordination between nodes
- Separate from high-speed data plane (RDMA)

**Configuration**

Set the network interface name:

.. code:: json

    {
        "oob_port": "eno0"
    }

**Common Interface Names**

- ``eno0``, ``eno1``: Embedded Ethernet
- ``enp0s0``, ``enp1s0``: PCI Ethernet
- ``eth0``, ``eth1``: Legacy naming

**Verification**

Check available interfaces:

.. code:: bash

    ip addr show

Master-Worker Configuration
============================

**Master Node**

The master node coordinates the test:

- Set ``master_addr`` to the master node's IP or hostname
- Master initiates test sequences
- Collects and aggregates results

**Worker Nodes**

Worker nodes participate in the test:

- Connect to master via ``master_addr:master_port``
- Execute test operations as directed
- Report results back to master

**Port Configuration**

Default port is 1234, but can be changed:

.. code:: json

    {
        "master_addr": "10.0.1.100",
        "master_port": "1234"
    }

Ensure the port is:

- Not blocked by firewalls
- Not in use by other services
- Accessible from all nodes

Performance Metrics
===================

**Bandwidth (max_bw)**

Maximum sustained bandwidth in GB/s:

$$\text{Bandwidth} = \frac{\text{Transfer Size}}{\text{Transfer Time}}$$

Expected values:

- Single device: 40-50 GB/s
- Multiple devices: Near-linear scaling
- Depends on transfer size and configuration

**Latency (avg_lat)**

Average round-trip latency in microseconds (μs):

- Small transfers (< 1MB): 100-500 μs
- Medium transfers (1MB): 1000-2000 μs
- Large transfers (> 1MB): 2000-3000 μs

**Factors Affecting Performance**

- Transfer size: Larger transfers achieve higher bandwidth
- Number of threads: More threads improve bandwidth but may increase latency
- Block size: Optimal block size depends on workload
- Network topology: Direct connections vs. switches
- Concurrent operations: Multiple devices in parallel

Test Configuration Parameters
==============================

**Number of Threads**

Controls concurrency level:

- Low (128): Good for latency testing
- Medium (1024-4096): Balanced
- High (16384): Maximum bandwidth

**Block Size**

Size of each data block:

- Small (64-128): Lower latency overhead
- Medium (256-512): Balanced
- Large (1024+): Higher bandwidth

**Number of Blocks**

Total blocks to transfer:

- Affects total transfer size
- Higher values: More statistical significance
- Lower values: Faster test completion

**Transfer Sizes**

Common sizes (in bytes):

- 524288 (512 KB): Small transfer test
- 1048576 (1 MB): Medium transfer test
- 4194304 (4 MB): Large transfer test
- 16777216 (16 MB): Very large transfer test

Volume Mounts for AMD Pensando NICs
====================================

**Required Libraries**

For AMD Pensando AINIC support, mount these libraries:

.. code:: json

    {
        "volume_dict": {
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
        }
    }

**Why These Mounts?**

- ``libionic.so*``: Core Pensando AINIC library
- ``libionic-rdmav34.so``: RDMA verbs support for Pensando
- ``ionic.driver``: Driver configuration for Pensando NICs
- Essential for AINIC functionality with MORI

**Verification**

After container start, verify libraries are accessible:

.. code:: bash

    docker exec mori_container ls -l /usr/lib/x86_64-linux-gnu/libionic*

Performance Optimization Tips
==============================

**Maximize Bandwidth**

- Use all available RDMA devices
- Increase number of threads (up to 16384)
- Use larger transfer sizes (1MB+)
- Ensure direct node-to-node connectivity
- Optimize block size for your workload

**Minimize Latency**

- Reduce number of threads (128-512)
- Use smaller transfer sizes
- Minimize block size
- Ensure low-latency network switches
- Check for proper device-GPU affinity

**Network Optimization**

- Enable flow control on switches
- Configure proper MTU (typically 9000 for jumbo frames)
- Verify no packet drops: ``ethtool -S rdma0``
- Check for proper NUMA affinity
- Disable unnecessary network services

**System Tuning**

- Set CPU governor to performance mode
- Disable CPU frequency scaling
- Increase network buffer sizes
- Tune RDMA parameters (QP depth, CQ depth)
- Monitor system resources during tests

Troubleshooting
===============

**Container Issues**

- Verify container image is accessible on all nodes
- Check device mounts (``/dev/dri``, ``/dev/kfd``)
- Ensure RDMA devices are visible in container
- Verify volume mounts for Pensando libraries
- Check container logs for errors

**Network Issues**

- Verify RDMA devices are active: ``ibstat`` or ``rdma link``
- Check network connectivity: ``ping`` via OOB interface
- Test RDMA connectivity: ``ibv_devinfo``
- Verify all nodes can reach master
- Check firewall rules for master_port
- Ensure RDMA devices are not in error state

**Performance Issues**

- Check for packet drops: ``ethtool -S rdma0 | grep drop``
- Verify proper NUMA affinity
- Monitor CPU usage during tests
- Check for thermal throttling
- Ensure consistent performance across all devices
- Verify switch configuration

**Configuration Issues**

- Verify ``mori_dir`` path exists and contains MORI binaries
- Check ``torchlib_dir`` contains required PyTorch libraries
- Ensure ``oob_port`` interface exists on all nodes
- Verify ``mori_device_list`` matches actual devices
- Check ``master_addr`` is reachable from all workers

**Library Issues (Pensando AINIC)**

- Verify all ``libionic`` libraries exist on host
- Check library versions match
- Ensure proper symbolic links
- Verify driver is loaded: ``lsmod | grep ionic``
- Check dmesg for driver errors

Best Practices
==============

**Configuration Management**

- Use version control for configuration files
- Document actual values for ``<changeme>`` placeholders
- Test configuration on a small cluster first
- Maintain separate configs for different cluster sizes
- Keep a baseline configuration for reference

**Testing Methodology**

- Start with 2 nodes before scaling up
- Test individual RDMA devices first
- Gradually increase thread counts
- Validate results against expected thresholds
- Run multiple iterations for statistical significance
- Document any deviations from expected results

**Production Deployment**

- Use consistent hardware across all nodes
- Verify firmware versions on all NICs
- Ensure all nodes have identical software stacks
- Set up monitoring for RDMA performance
- Implement health checks for RDMA devices
- Plan for regular validation testing

**Maintenance**

- Regularly verify RDMA device health
- Monitor for firmware updates
- Check for library updates
- Validate after any network changes
- Document all configuration changes
- Maintain logs for troubleshooting

Multi-Node Scaling
===================

**2-Node Configuration**

Basic setup for initial testing:

.. code:: json

    {
        "no_of_nodes": "2"
    }

**4-Node Configuration**

Small cluster:

.. code:: json

    {
        "no_of_nodes": "4"
    }

**8+ Node Configuration**

Large cluster considerations:

.. code:: json

    {
        "no_of_nodes": "8"
    }

Additional considerations:

- Ensure switch capacity supports all-to-all communication
- Monitor aggregate bandwidth
- Consider network topology (fat-tree, dragonfly, etc.)
- Plan for potential bottlenecks
- Implement proper load balancing

Expected Results Interpretation
================================

**Read Operations**

Example configuration ``"16384,128,1"`` with 524288 byte transfer:

- 16384 threads for high concurrency
- 128 block size
- 1 block per operation
- Expected: 45 GB/s bandwidth, 1500 μs latency

**Write Operations**

Example configuration ``"128"`` with 524288 byte transfer:

- 128 threads for moderate concurrency
- Expected: 45 GB/s bandwidth, 1500 μs latency

**Validation**

Test results should meet or exceed expected values:

- Bandwidth within 10% of expected
- Latency within 20% of expected
- Consistent results across multiple runs
- No significant outliers

Advanced Configuration
======================

**Custom Transfer Sizes**

Add custom transfer sizes to expected_results:

.. code:: json

    {
        "expected_results": {
            "read": {
                "16384,128,1": {
                    "2097152": {
                        "max_bw": "48.0",
                        "avg_lat": "5000"
                    }
                }
            }
        }
    }

**Custom Thread Configurations**

Test different concurrency levels:

.. code:: json

    {
        "expected_results": {
            "read": {
                "512,128,1": {
                    "524288": {
                        "max_bw": "40.0",
                        "avg_lat": "1000"
                    }
                },
                "8192,128,1": {
                    "524288": {
                        "max_bw": "44.0",
                        "avg_lat": "1200"
                    }
                }
            }
        }
    }

**Multiple Block Configurations**

Test with different block counts:

.. code:: json

    {
        "expected_results": {
            "read": {
                "16384,128,2": {
                    "1048576": {
                        "max_bw": "47.0",
                        "avg_lat": "3000"
                    }
                }
            }
        }
    }

Integration with Distributed Workloads
=======================================

**Use Cases**

MORI testing validates RDMA for:

- Distributed LLM training
- Model parallel inference
- Parameter server architectures
- All-reduce operations in collective communication
- Disaggregated memory systems

**Correlation with Application Performance**

Good MORI results indicate:

- Efficient gradient synchronization
- Low-latency parameter access
- High-bandwidth model shard transfers
- Reliable multi-node communication
- Optimal collective operation performance

**Baseline Requirements**

Minimum MORI performance for distributed workloads:

- Bandwidth: > 35 GB/s per device
- Latency: < 2000 μs for 1MB transfers
- Consistent performance across all devices
- No packet loss or errors

Example Configurations
======================

**High-Bandwidth Testing**

.. code:: json

    {
        "no_of_nodes": "2",
        "mori_device_list": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
        "expected_results": {
            "read": {
                "16384,128,1": {
                    "4194304": {
                        "max_bw": "48.0",
                        "avg_lat": "10000"
                    }
                }
            }
        }
    }

**Low-Latency Testing**

.. code:: json

    {
        "no_of_nodes": "2",
        "mori_device_list": "rdma0",
        "expected_results": {
            "read": {
                "128,128,1": {
                    "524288": {
                        "max_bw": "40.0",
                        "avg_lat": "800"
                    }
                }
            }
        }
    }

**Production Validation**

.. code:: json

    {
        "no_of_nodes": "4",
        "mori_device_list": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
        "nic_type": "ainic",
        "expected_results": {
            "read": {
                "16384,128,1": {
                    "524288": {"max_bw": "45.0", "avg_lat": "1500"},
                    "1048576": {"max_bw": "46.0", "avg_lat": "2500"},
                    "4194304": {"max_bw": "48.0", "avg_lat": "10000"}
                }
            },
            "write": {
                "128": {
                    "524288": {"max_bw": "45.0", "avg_lat": "1500"},
                    "1048576": {"max_bw": "46.0", "avg_lat": "2500"}
                }
            }
        }
    }
