.. meta::
  :description: Configure the variables in the SGLang Disaggregated Prefill-Decode configuration file
  :keywords: inference, ROCm, install, cvs, SGLang, disaggregated, prefill, decode, LLM, MI35X

*************************************************
SGLang disaggregated prefill-decode inference configuration file
*************************************************

SGLang disaggregated prefill-decode tests validate distributed LLM inference performance using SGLang's disaggregated architecture on AMD MI35X GPU clusters.
These tests ensure optimal throughput, latency, and scalability for large-scale LLM serving workloads by separating prefill and decode phases across different node groups.

The SGLang disaggregated P-D tests check:

- **Container orchestration**: Docker setup with SGLang for multi-node inference
- **Disaggregated architecture**: Separate prefill and decode node groups with proxy router
- **Model serving**: LLM deployment with cache-aware policies and tensor parallelism
- **Performance metrics**: Throughput, TTFT, TPOT, and end-to-end latency
- **Distributed communication**: NCCL configuration for InfiniBand/RoCE networks
- **Result verification**: Expected throughput and latency metrics

Change the parameters as needed in the SGLang disaggregated configuration file: ``sglang_disagg_pd_mi35x.json`` for distributed LLM serving.

.. note::

  - ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username.
  - Replace all ``<changeme>`` placeholders with actual values for your cluster.

``sglang_disagg_pd_mi35x.json``
================================

Here's a code snippet of the ``sglang_disagg_pd_mi35x.json`` file for reference:

.. dropdown:: ``sglang_disagg_pd_mi35x.json``

  .. code:: json

    {
        "config": {
            "container_image": "lmsysorg/sglang:v0.5.7-rocm700-mi35x",
            "container_name": "sglang_container",
            "nnodes": "4",
            "hf_token_file": "/home/{user-id}/.hf_token",
            "shm_size": "128G",
            "log_dir": "/home/{user-id}/LOGS/sglang",
            "log_level": "info",
            "nic_type": "ainic",
            "nccl_ib_hca_list": "<changeme>",
            "nccl_ib_hca": "<changeme>",
            "nccl_socket_ifname": "<changeme>",
            "gloo_socket_ifname": "<changeme>",
            "gloo_tcp_ifname": "<changeme>",
            "nccl_ib_gid_index": "1",
            "nccl_debug": "ERROR",
            "prefill_node_list": [ "<changeme>", "<changeme>" ],
            "decode_node_list": [ "<changeme>", "<changeme>" ],
            "proxy_router_node": "<changeme>",
            "benchmark_serv_node": "<changeme>",
            "prefill_serv_port": "30001",
            "decode_serv_port": "30002",
            "proxy_router_port": "8000",
            "prefill_coordinator_addr": "<changeme>",
            "decode_coordinator_addr": "<changeme>",
            "prefill_coordinator_port": "40001",
            "decode_coordinator_port": "40002",
            "proxy_router_serv_port": "8000",
            "container_config": {
                "device_list": [ "/dev/dri", "/dev/kfd" ],
                "volume_dict": {
                    "/home/{user-id}": "/home/{user-id}",
                    "/it-share/models": "/root/models"
                },
                "env_dict": {}
            }
        },
        "benchmark_params": {
            "llama-70b": {
                "backend": "sglang",
                "max_concurrency": "64",
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "prefill_policy": "cache_aware",
                "decode_policy": "cache_aware",
                "tensor_parallelism": "8",
                "memory_fraction": "0.7",
                "tokenizer_mode": "auto",
                "inference_poll_iterations": "16",
                "inference_tests": {
                    "gsm8k": {
                        "backend": "sglang",
                        "num_questions": "1000",
                        "max_concurrency": "100",
                        "expected_results": {
                            "auto": {
                                "tokens_per_sec": "2300"
                            }
                        }
                    },
                    "bench_serv_random": {
                        "backend": "sglang",
                        "data_set_name": "random",
                        "num_prompts": "3000",
                        "input_length": "1024",
                        "output_length": "1024",
                        "random_range_ratio": "0.5",
                        "expected_results": {
                            "auto": {
                                "output_throughput_per_sec": "21000",
                                "mean_ttft_ms": "34500",
                                "mean_tpot_ms": "50"
                            }
                        }
                    }
                }
            },
            "deepseek-r1": {
                "backend": "sglang",
                "max_concurrency": "64",
                "model": "/root/models/deepseek-ai/DeepSeek-R1-0528/",
                "prefill_policy": "cache_aware",
                "decode_policy": "cache_aware",
                "tensor_parallelism": "8",
                "memory_fraction": "0.7",
                "tokenizer_mode": "auto",
                "inference_poll_iterations": "16",
                "inference_tests": {
                    "gsm8k": {
                        "backend": "sglang",
                        "num_questions": "1000",
                        "max_concurrency": "100",
                        "expected_results": {
                            "auto": {
                                "tokens_per_sec": "700"
                            }
                        }
                    },
                    "bench_serv_random": {
                        "backend": "sglang",
                        "data_set_name": "random",
                        "num_prompts": "3000",
                        "input_length": "1024",
                        "output_length": "1024",
                        "random_range_ratio": "0.5",
                        "expected_results": {
                            "auto": {
                                "output_throughput_per_sec": "10000",
                                "mean_ttft_ms": "60000",
                                "mean_tpot_ms": "110"
                            }
                        }
                    }
                }
            }
        }
    }

Configuration Parameters
========================

Here's an exhaustive list of the available parameters in the ``sglang_disagg_pd_mi35x.json`` configuration file:

General Configuration
---------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``container_image``
     - lmsysorg/sglang:v0.5.7-rocm700-mi35x
     - Docker container image with SGLang for MI35X GPUs
   * - ``container_name``
     - sglang_container
     - Name of the Docker container instance
   * - ``nnodes``
     - 4
     - Total number of nodes in the cluster (prefill + decode nodes)
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - Path to HuggingFace authentication token file
   * - ``shm_size``
     - 128G
     - Shared memory size for the container (critical for distributed workloads)
   * - ``log_dir``
     - ``/home/{user-id}/LOGS/sglang``
     - Directory for SGLang logs (must be accessible from all nodes)
   * - ``log_level``
     - info
     - Logging level (debug, info, warning, error)
   * - ``nic_type``
     - ainic
     - Network interface card type (ainic for AMD Pensando)
   * - ``container_config.device_list``
     - ``[ "/dev/dri", "/dev/kfd" ]``
     - List of device paths to mount in the container for GPU access
   * - ``container_config.volume_dict``
     - Multiple mappings
     - Dictionary mapping host paths to container paths for volume mounts
   * - ``container_config.env_dict``
     - Empty
     - Dictionary of environment variables to set in the container

Network Configuration
---------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Network parameters
     - Example values
     - Description
   * - ``nccl_ib_hca_list``
     - rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
     - Comma-separated list of RDMA devices for NCCL communication
   * - ``nccl_ib_hca``
     - rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
     - InfiniBand HCA devices for NCCL (typically same as nccl_ib_hca_list)
   * - ``nccl_socket_ifname``
     - eno0
     - Network interface for NCCL socket communication
   * - ``gloo_socket_ifname``
     - eno0
     - Network interface for Gloo backend socket communication
   * - ``gloo_tcp_ifname``
     - eno0
     - Network interface for Gloo TCP communication
   * - ``nccl_ib_gid_index``
     - 1
     - InfiniBand GID index for RDMA communication
   * - ``nccl_debug``
     - ERROR
     - NCCL debug level (ERROR, WARN, INFO, TRACE)

Disaggregated Architecture Configuration
-----------------------------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Architecture parameters
     - Values
     - Description
   * - ``prefill_node_list``
     - Array of node addresses
     - List of nodes dedicated to prefill phase processing
   * - ``decode_node_list``
     - Array of node addresses
     - List of nodes dedicated to decode phase processing
   * - ``proxy_router_node``
     - Single node address
     - Node running the proxy router for request routing
   * - ``benchmark_serv_node``
     - Single node address
     - Node running the benchmark server for testing
   * - ``prefill_serv_port``
     - 30001
     - Port for prefill service
   * - ``decode_serv_port``
     - 30002
     - Port for decode service
   * - ``proxy_router_port``
     - 8000
     - Port for proxy router service
   * - ``prefill_coordinator_addr``
     - Master node address
     - Coordinator address for prefill node synchronization
   * - ``decode_coordinator_addr``
     - Master node address
     - Coordinator address for decode node synchronization
   * - ``prefill_coordinator_port``
     - 40001
     - Port for prefill coordinator
   * - ``decode_coordinator_port``
     - 40002
     - Port for decode coordinator
   * - ``proxy_router_serv_port``
     - 8000
     - Service port for proxy router

Model Configuration (Llama-70B)
--------------------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Model parameters
     - Default values
     - Description
   * - ``backend``
     - sglang
     - Inference backend to use
   * - ``max_concurrency``
     - 64
     - Maximum number of concurrent requests
   * - ``model``
     - meta-llama/Llama-3.3-70B-Instruct
     - Model identifier or local path
   * - ``prefill_policy``
     - cache_aware
     - Scheduling policy for prefill phase (cache_aware, fcfs)
   * - ``decode_policy``
     - cache_aware
     - Scheduling policy for decode phase (cache_aware, fcfs)
   * - ``tensor_parallelism``
     - 8
     - Number of GPUs for tensor parallelism per node
   * - ``memory_fraction``
     - 0.7
     - Fraction of GPU memory to allocate for KV cache
   * - ``tokenizer_mode``
     - auto
     - Tokenizer mode (auto, slow, fast)
   * - ``inference_poll_iterations``
     - 16
     - Number of polling iterations for inference completion

Inference Test: GSM8K
---------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - GSM8K parameters
     - Default values
     - Description
   * - ``backend``
     - sglang
     - Inference backend
   * - ``num_questions``
     - 1000
     - Number of questions from GSM8K dataset to test
   * - ``max_concurrency``
     - 100
     - Maximum concurrent requests during testing
   * - ``expected_results.auto.tokens_per_sec``
     - 2300 (Llama-70B), 700 (DeepSeek-R1)
     - Expected token generation throughput

Inference Test: Random Benchmark
---------------------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Random benchmark parameters
     - Default values
     - Description
   * - ``backend``
     - sglang
     - Inference backend
   * - ``data_set_name``
     - random
     - Type of dataset (random, synthetic)
   * - ``num_prompts``
     - 3000
     - Number of prompts to generate and test
   * - ``input_length``
     - 1024
     - Input token length for each prompt
   * - ``output_length``
     - 1024
     - Output token length for each response
   * - ``random_range_ratio``
     - 0.5
     - Ratio for randomizing input/output lengths
   * - ``expected_results.auto.output_throughput_per_sec``
     - 21000 (Llama-70B), 10000 (DeepSeek-R1)
     - Expected output token throughput
   * - ``expected_results.auto.mean_ttft_ms``
     - 34500 (Llama-70B), 60000 (DeepSeek-R1)
     - Expected mean time to first token in milliseconds
   * - ``expected_results.auto.mean_tpot_ms``
     - 50 (Llama-70B), 110 (DeepSeek-R1)
     - Expected mean time per output token in milliseconds

Inference Test: Generated Shared Prefix
----------------------------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Shared prefix parameters
     - Default values
     - Description
   * - ``backend``
     - sglang
     - Inference backend
   * - ``gsp_num_groups``
     - 1
     - Number of groups with shared prefixes
   * - ``gsp_prompts_per_group``
     - 16
     - Number of prompts per group sharing the same prefix
   * - ``gsp_system_prompt_len``
     - 0
     - Length of system prompt tokens
   * - ``gsp_question_len``
     - 1024
     - Length of question tokens
   * - ``gsp_output_len``
     - 1024
     - Length of output tokens

SGLang Disaggregated Architecture Overview
===========================================

**What is SGLang?**

SGLang (SGLang Runtime) is a high-performance serving framework for large language models:

- Optimized for high throughput and low latency
- Supports disaggregated prefill-decode architecture
- Automatic prefix caching with RadixAttention
- Tensor parallelism for large models
- Cache-aware scheduling policies

**Disaggregated Prefill-Decode Architecture**

SGLang separates inference into two phases across different node groups:

1. **Prefill Nodes**: Process input prompts and generate KV cache
2. **Decode Nodes**: Generate output tokens using cached KV states
3. **Proxy Router**: Routes requests and coordinates between phases

**Benefits of Disaggregation**

- **Resource Optimization**: Different compute patterns for prefill vs. decode
- **Scalability**: Scale prefill and decode independently
- **Throughput**: Better GPU utilization and lower latency
- **Cost Efficiency**: Optimize hardware allocation per phase

Architecture Components
=======================

**Prefill Cluster**

Handles the initial prompt processing:

- Input tokenization and encoding
- Attention computation for input sequence
- KV cache generation
- Higher compute intensity, lower memory bandwidth

**Decode Cluster**

Handles incremental token generation:

- Autoregressive token generation
- Attends to cached KV states
- Lower compute intensity, higher memory bandwidth
- Optimized for throughput

**Proxy Router**

Coordinates request routing:

- Receives incoming requests
- Routes to prefill cluster
- Transfers KV cache to decode cluster
- Streams responses back to client
- Load balancing across nodes

**Coordinator Nodes**

Synchronize distributed operations:

- Master address for prefill/decode clusters
- Handles distributed initialization
- Manages collective communication
- Monitors cluster health

Performance Metrics Explained
==============================

**Tokens Per Second (TPS)**

Total token generation throughput:

$$\text{TPS} = \frac{\text{Total Output Tokens}}{\text{Total Time}}$$

Higher values indicate better performance.

**Time to First Token (TTFT)**

Latency from request to first generated token:

- Includes prefill time
- Network transfer to decode cluster
- Critical for user-perceived latency
- Measured in milliseconds

**Time Per Output Token (TPOT)**

Average time between consecutive output tokens:

$$\text{TPOT} = \frac{\text{Total Decode Time}}{\text{Number of Output Tokens}}$$

Lower values indicate faster generation.

**Output Throughput**

Output tokens generated per second:

$$\text{Output Throughput} = \frac{\text{Total Output Tokens}}{\text{Total Time}}$$

Key metric for disaggregated systems.

**Request Throughput**

Requests completed per second:

$$\text{Request Throughput} = \frac{\text{Num Requests}}{\text{Total Time}}$$

Indicates system capacity.

Cache-Aware Scheduling
=======================

**RadixAttention**

SGLang's automatic prefix caching mechanism:

- Automatically detects common prefixes
- Caches KV states for reuse
- Reduces redundant computation
- Improves throughput for similar requests

**Scheduling Policies**

Two main policies available:

1. **cache_aware**: Prioritizes requests with cached prefixes
   - Maximizes cache hit rate
   - Better throughput for similar prompts
   - Recommended for most workloads

2. **fcfs** (First-Come-First-Served): Simple FIFO scheduling
   - No cache optimization
   - Predictable latency
   - Use for latency-sensitive applications

Network Configuration Best Practices
=====================================

**InfiniBand/RoCE Setup**

For optimal multi-node performance:

- Use all available RDMA devices (typically 8 per node)
- Set ``nccl_ib_hca_list`` with all rdma devices
- Configure ``nccl_ib_gid_index`` appropriately (usually 1 or 3)
- Ensure consistent configuration across all nodes

**Network Interface Selection**

Choose appropriate interfaces:

- Primary network: Use for data transfer (eno0, enp0s0, etc.)
- RDMA devices: Use for GPU-GPU communication
- Separate management network if available

**NCCL Debug Levels**

Adjust based on needs:

- **ERROR**: Production (minimal overhead)
- **WARN**: Debugging network issues
- **INFO**: Detailed information
- **TRACE**: Very verbose (debugging only)

**Example Configuration**

.. code:: json

    {
        "nccl_ib_hca_list": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
        "nccl_ib_hca": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
        "nccl_socket_ifname": "eno0",
        "gloo_socket_ifname": "eno0",
        "nccl_ib_gid_index": "1",
        "nccl_debug": "ERROR"
    }

Cluster Planning and Node Assignment
=====================================

**Node Distribution**

Typical 4-node setup:

- 2 prefill nodes: Handle prompt processing
- 2 decode nodes: Handle token generation
- 1 proxy router node: Can be on prefill/decode node
- 1 benchmark node: Can be on any node or separate

**Scaling Considerations**

Adjust node counts based on workload:

- More prefill nodes: Long context, high request rate
- More decode nodes: Long outputs, higher concurrency
- Balanced: Similar input/output lengths

**Example 4-Node Setup**

.. code:: json

    {
        "prefill_node_list": ["node1", "node2"],
        "decode_node_list": ["node3", "node4"],
        "proxy_router_node": "node1",
        "benchmark_serv_node": "node1",
        "prefill_coordinator_addr": "node1",
        "decode_coordinator_addr": "node3"
    }

**Example 8-Node Setup**

.. code:: json

    {
        "prefill_node_list": ["node1", "node2", "node3"],
        "decode_node_list": ["node4", "node5", "node6", "node7", "node8"],
        "proxy_router_node": "node1",
        "benchmark_serv_node": "node8",
        "prefill_coordinator_addr": "node1",
        "decode_coordinator_addr": "node4"
    }

Model Configuration
===================

**Tensor Parallelism**

Splits model across multiple GPUs:

- ``tensor_parallelism = 8``: Use all 8 GPUs per node
- Each GPU holds 1/8 of model parameters
- Required for large models (70B+)
- Adds communication overhead

**Memory Fraction**

Controls KV cache size:

- ``memory_fraction = 0.7``: Use 70% of GPU memory for cache
- Higher values: More concurrent requests
- Lower values: More memory for model weights
- Adjust based on max_concurrency needs

**Model Paths**

Two options for model specification:

1. **HuggingFace**: ``"meta-llama/Llama-3.3-70B-Instruct"``
   - Automatic download
   - Requires HF token
   - Cached locally

2. **Local Path**: ``"/root/models/deepseek-ai/DeepSeek-R1-0528/"``
   - Pre-downloaded model
   - Faster startup
   - Use volume mounts to access

Benchmark Configuration
=======================

**GSM8K Test**

Mathematical reasoning benchmark:

- Tests model accuracy on math problems
- Measures end-to-end throughput
- High concurrency tests system limits
- Expected: 2300 TPS for Llama-70B, 700 TPS for DeepSeek-R1

**Random Benchmark**

Synthetic load testing:

- Generates random prompts
- Controlled input/output lengths
- Tests sustained throughput
- Measures TTFT and TPOT
- Random range ratio adds variability

**Generated Shared Prefix**

Tests prefix caching effectiveness:

- Multiple prompts share common prefix
- Validates RadixAttention performance
- Important for real-world workloads (e.g., system prompts)
- Shows cache hit rate benefits

Performance Optimization Tips
==============================

**Memory Optimization**

- Adjust ``memory_fraction`` based on concurrency needs
- Monitor GPU memory usage: ``rocm-smi``
- Reduce ``max_concurrency`` if OOM occurs
- Use smaller ``tensor_parallelism`` if possible

**Throughput Optimization**

- Enable cache-aware scheduling
- Increase ``max_concurrency`` up to memory limits
- Optimize network configuration (use all RDMA devices)
- Balance prefill/decode node ratio
- Ensure efficient shared storage for logs

**Latency Optimization**

- Reduce ``inference_poll_iterations`` for lower latency
- Use FCFS scheduling for predictable latency
- Co-locate proxy router with prefill nodes
- Minimize network hops
- Use low-latency interconnect

**Network Optimization**

- Use all available RDMA devices
- Enable RDMA for storage if possible
- Separate data and management networks
- Monitor network utilization
- Tune NCCL parameters for your topology

Troubleshooting
===============

**Container Issues**

- Verify container image is accessible on all nodes
- Check device mounts (``/dev/dri``, ``/dev/kfd``)
- Ensure sufficient shared memory (``shm_size = 128G``)
- Verify volume mounts are accessible from all nodes
- Check HuggingFace token validity

**Network Issues**

- Verify RDMA devices are available: ``ibstat`` or ``rdma link``
- Check network interface names: ``ip addr``
- Test connectivity between nodes
- Verify firewall rules for all ports
- Check NCCL environment variables
- Monitor with: ``NCCL_DEBUG=INFO``

**Memory Issues (OOM)**

- Reduce ``memory_fraction`` from 0.7 to 0.6
- Decrease ``max_concurrency``
- Use gradient checkpointing if available
- Verify GPU memory availability: ``rocm-smi``
- Check for memory leaks in logs

**Performance Issues**

- Verify all nodes are healthy and responsive
- Check GPU utilization on all nodes: ``rocm-smi -d 0 --showuse``
- Monitor network bandwidth utilization
- Check for stragglers in prefill/decode clusters
- Review NCCL communication patterns
- Ensure load balancing across nodes

**Coordinator Issues**

- Verify coordinator addresses are reachable
- Check coordinator ports are not blocked
- Ensure coordinator node is stable
- Review synchronization logs
- Test with single-node first, then scale

**Model Loading Issues**

- Verify model path or HuggingFace ID is correct
- Check HuggingFace token for private models
- Ensure sufficient disk space for model download
- Verify volume mounts for local models
- Check model compatibility with SGLang version

Hardware-Specific Tuning
=========================

**MI35X Configuration**

- 8 GPUs per node, 128GB HBM total
- Recommended: ``tensor_parallelism = 8`` for 70B+ models
- Expected throughput: 21K tokens/sec (Llama-70B), 10K tokens/sec (DeepSeek-R1)
- Optimal memory_fraction: 0.7
- Pensando AINIC support for efficient networking

**Multi-Node Scaling**

Adjust for cluster size:

- 2 nodes: 1 prefill, 1 decode
- 4 nodes: 2 prefill, 2 decode (recommended)
- 8 nodes: 3 prefill, 5 decode (decode-heavy)
- 16+ nodes: Custom ratio based on workload

**Port Allocation**

Ensure ports don't conflict:

- Prefill service: 30001
- Decode service: 30002
- Proxy router: 8000
- Prefill coordinator: 40001
- Decode coordinator: 40002
- Add offset per cluster if running multiple

Best Practices
==============

**Configuration Management**

- Use version control for configuration files
- Document node assignments and network topology
- Track performance across different configurations
- Maintain baseline benchmarks for comparison
- Test changes in staging before production

**Production Deployment**

- Use cache-aware scheduling for better throughput
- Set appropriate ``max_concurrency`` based on load tests
- Monitor all nodes for health and performance
- Implement automatic failover for coordinator nodes
- Set up alerting for performance degradation
- Use persistent logs on shared storage

**Model Updates**

- Test new models in single-node mode first
- Validate performance meets expectations
- Compare against baseline metrics
- Update expected_results based on testing
- Document model-specific tuning

**Monitoring and Logging**

- Centralize logs on shared storage
- Monitor key metrics: TPS, TTFT, TPOT, throughput
- Track GPU utilization across all nodes
- Monitor network bandwidth and latency
- Set up dashboards for real-time visibility
- Archive logs for post-mortem analysis

**Maintenance Windows**

- Schedule regular maintenance for node updates
- Test network configuration changes offline
- Validate after hardware/software updates
- Keep rollback configurations ready
- Document all changes and their impacts

Advanced Configuration
======================

**Custom Scheduling Policies**

Experiment with different policies:

- ``cache_aware``: Best for repeated prefixes
- ``fcfs``: Best for latency-sensitive apps
- Mix policies per workload type

**Dynamic Node Allocation**

Adjust node assignments based on load:

- Scale prefill nodes during high request rate
- Scale decode nodes for long-form generation
- Use Kubernetes for dynamic scaling
- Implement auto-scaling based on queue depth

**Multi-Tenant Deployment**

Run multiple model instances:

- Separate port ranges per tenant
- Dedicated node groups if needed
- Resource quotas and limits
- Isolated logging and monitoring

**Custom Models**

Fine-tuned or custom models:

- Use local model paths
- Ensure tokenizer compatibility
- Validate model format (safetensors, pytorch)
- Test thoroughly before production
- Document model-specific parameters

Volume Mounts for AMD Pensando NICs
====================================

**Required Libraries**

For AMD Pensando AINIC support, mount:

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

- ``libionic.so``: Core Pensando library
- ``libionic-rdmav34.so``: RDMA verbs support
- ``ionic.driver``: Driver configuration
- Essential for AINIC functionality

Example Usage Scenarios
========================

**High-Throughput Serving**

.. code:: json

    {
        "max_concurrency": "128",
        "memory_fraction": "0.8",
        "prefill_policy": "cache_aware",
        "decode_policy": "cache_aware"
    }

**Low-Latency Serving**

.. code:: json

    {
        "max_concurrency": "32",
        "memory_fraction": "0.6",
        "prefill_policy": "fcfs",
        "decode_policy": "fcfs",
        "inference_poll_iterations": "8"
    }

**Batch Processing**

.. code:: json

    {
        "max_concurrency": "256",
        "memory_fraction": "0.9",
        "prefill_policy": "cache_aware"
    }

**Development/Testing**

.. code:: json

    {
        "max_concurrency": "16",
        "memory_fraction": "0.5",
        "nccl_debug": "INFO"
    }

DeepSeek-R1 Specific Considerations
====================================

**Model Characteristics**

- Much larger than Llama-70B
- Reasoning-focused architecture
- Longer generation sequences
- Higher latency, lower throughput

**Recommended Settings**

- Lower ``max_concurrency``: 32-64
- Higher ``memory_fraction``: 0.75-0.8
- More decode nodes relative to prefill
- Longer timeouts for requests
- Monitor token generation carefully

**Expected Performance**

- TPS: ~700 (vs. 2300 for Llama-70B)
- TTFT: ~60s (vs. 34.5s for Llama-70B)
- TPOT: ~110ms (vs. 50ms for Llama-70B)
- Higher variance due to reasoning steps

Comparison: Llama-70B vs DeepSeek-R1
=====================================

.. list-table::
   :widths: 2 2 2 4
   :header-rows: 1

   * - Metric
     - Llama-70B
     - DeepSeek-R1
     - Notes
   * - Tokens/sec
     - 2300
     - 700
     - 3.3x difference
   * - Output throughput
     - 21000
     - 10000
     - 2.1x difference
   * - Mean TTFT
     - 34.5s
     - 60s
     - Longer prefill for R1
   * - Mean TPOT
     - 50ms
     - 110ms
     - 2.2x slower decode
   * - Concurrency
     - 64-128
     - 32-64
     - R1 needs lower concurrency
   * - Use case
     - General chat, Q&A
     - Complex reasoning
     - Choose based on needs
