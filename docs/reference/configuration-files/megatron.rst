.. meta::
  :description: Configure the variables in the Megatron training configuration files
  :keywords: training, ROCm, install, cvs, Megatron,

*************************************
Megatron training configuration files
*************************************

Megatron training enables scaling transformer models from millions to trillions of parameters by efficiently utilizing hundreds or thousands of GPUs across multiple nodes.

The Megatron tests check:

- **Container orchestration**: Docker setup with ROCm/RDMA
- **Multi-node communication**: NCCL/RCCL initialization
- **Model convergence**: Loss decreases and no NaN/Inf values
- **Performance targets**: Throughput and memory usage within expected ranges
- **Result verification**: Expected tokens/sec and TFLOPS metrics

Change the parameters as needed in the Megatron training configuration files: ``mi3xx_singlenode_megatron_llama.json`` and ``mi3xx_distributed_megatron_llama.json`` for single node and multiple node configurations, respectively.

Further, you can configure the ``mi35x_singlenode_megatron_llama.json`` file to run Megatron on a single node MI35x.

.. note::

  Parameters with the ``<changeme>`` value must have that value modified to your specifications.

Single node configuration
=========================

This is the ``mi3xx_singlenode_megatron_llama.json`` configuration file:

.. dropdown:: ``mi3xx_singlenode_megatron_llama.json``

  .. code:: json
      
    {
    
      "config":
      {
          "container_image": "rocm/megatron-lm:v25.5_py310",
          "container_name": "megatron_llama3.1_310",
          "_example_nnodes": "4",
          "nnodes": "<changeme>-no of nodes to run singlenode training",
          "master_address": "<changeme>",
            "_example_training_iterations": "30",
          "training_iterations": "<changeme>",
          "hf_token_file": "/home/{user-id}/.hf_token",
          "shm_size": "128G",
          "_comments_data_cache_dir": "This path should be accessible from all nodes like a common FS like NFS for distributed training",
          "data_cache_dir": "/home/{user-id}/cache",
          "mock_data": "True",
          "log_dir": "/home/{user-id}/LOG_DIR",
          "dataset_source": 
          {
          },
          "container_config":
          {
              "device_list": [ "/dev/dri", "/dev/kfd" ],
              "volume_dict":
              {
              "/home/{user-id}": "/home/{user-id}"
              }
          }
      },
      "model_params":
      {
          "single_node":
          {
                "llama3_1_8b":
                {
                    "mi300x":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-8B",
                        "model_size": "8",
                        "batch_size": "128",
                        "micro_batch_size": "2",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
                            "throughput_per_gpu": "380.0",
                            "tokens_per_gpu": "6500.0",
                            "elapsed_time_per_iteration": "12000.0"
                        }
                    },
                    "mi325":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-8B",
                        "model_size": "8",
                        "batch_size": "128",
                        "micro_batch_size": "2",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
                            "throughput_per_gpu": "380.0",
                            "tokens_per_gpu": "6500.0",
                            "elapsed_time_per_iteration": "12000.0"
                        }
                    }
                },
                "llama3_1_70b":
                {
                    "mi300x":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-70B",
                        "model_size": "70",
                        "batch_size": "128",
                        "micro_batch_size": "1",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                        "tensor_parallelism": "8",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
                            "throughput_per_gpu": "500.0",
                            "tokens_per_gpu": "1000.0"
                        }
                    },
                    "mi325":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-70B",
                        "model_size": "70",
                        "batch_size": "128",
                        "micro_batch_size": "1",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                        "tensor_parallelism": "8",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
                            "throughput_per_gpu": "520.0",
                            "tokens_per_gpu": "1100.0"
                        }
                    }
                }
          }

        }
    
    }


Parameters
----------

.. |br| raw:: html

    <br />

Use the parameters in these tables to configure the training file.

``config``
~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``container_image``
     - ``rocm/megatron-lm:v25.5_py310`` 
     - Docker image used to run Megatron-LM
   * - ``container_name``
     - ``megatron_llama3.1_310`` 
     - Name assigned to the container instance
   * - ``_example_nnodes``
     - 4
     - Example of number of cluster nodes participating in the job 
   * - ``Nnodes``
     - "``<changeme>``-no of nodes to run singlenode training"
     - Number of nodes in the distributed job 
   * - ``master_address``
     - ``<changeme>``
     - IP of the master/coordinator node
   * - ``_example_training_iterations``
     - 30
     - Example of number of training iterations/steps to run in this test
   * - ``training_iterations``
     - ``<changeme>``
     - Number of training iterations/steps to run in this test
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - Path to a Hugging Face token file used to download tokenized models/datasets requiring authorization
   * - ``shm_size``
     - 256G
     - Docker shared memory size mounted into container
   * - ``_comments_data_cache_dir``
     - "This path should be accessible from all nodes like a common FS like NFS for distributed training"
     - Comment explaining ``data_cache_dir`` must be accessible from all nodes
   * - ``data_cache_dir``
     - ``/home/{user-id}/cache``
     - Dataset/cache directory
   * - ``mock_data``
     - True
     - "True"/"False": Use synthetic data (True) to avoid real dataset downloads in CI/smoke tests 
   * - ``log_dir``
     - ``/home/{user-id}/LOG_DIR``
     - Path where training logs should be written on the host

``model_params/single_node/llama3_1_8b/mi300x``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``device_list``
     - Values:
        - ``"/dev/dri"``
        - ``"/dev/kfd"``
        - ``"/dev/infiniband/rdma_cm"``
     - Kernel devices exposed inside container
   * - ``volume_dict``
     - N/A
     - Host-to-container mounts: map host paths (home, RDMA libs, ``/lib/libibverbs.d``, log output) into the container so code and drivers are accessible
   * - ``/home/<changeme>``
     - ``/home/{user-id}``
     - The directory
   * - ``model_params``
     - N/A
     - Model parameters
   * - ``single_node``
     - N/A
     - The structure (single node)
   * - ``llama3_1_8b``
     - N/A
     - The model being used
   * - ``mi300X``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-8B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 8
     - The abbreviated model size
   * - ``batch_size``
     - 128
     - Global batch size
   * - ``micro_batch_size``
     - 2
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 1
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 0
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 0
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "throughput_per_gpu": "380.0",
          "tokens_per_gpu": "6500.0",
          "elapsed_time_per_iteration": "12000.0"
      }

``model_params/single_node/llama3_1_8b/mi325``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi325``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-8B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 8
     - The abbreviated model size
   * - ``batch_size``
     - 128
     - Global batch size
   * - ``micro_batch_size``
     - 2
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 1
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 0
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 0
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "throughput_per_gpu": "380.0",
          "tokens_per_gpu": "6500.0",
          "elapsed_time_per_iteration": "12000.0"
      }

``model_params/single_node/llama3_1_70b/mi300X``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi300X``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-70B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 70
     - The abbreviated model size
   * - ``batch_size``
     - 128
     - Global batch size
   * - ``micro_batch_size``
     - 1
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 8
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 0
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 0
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "throughput_per_gpu": "500.0",
          "tokens_per_gpu": "1000.0",
      }


``model_params/single_node/llama3_1_70b/mi325``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi325``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-70B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 70
     - The abbreviated model size
   * - ``batch_size``
     - 128
     - Global batch size
   * - ``micro_batch_size``
     - 1
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 8
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 0
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 0
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "throughput_per_gpu": "520.0",
          "tokens_per_gpu": "1100.0",
      }

Multi-node configuration
========================

This is the multi-node ``mi3xx_distributed_megatron_llama.json`` configuration file:

.. dropdown:: ``mi3xx_distributed_megatron_llama.json`` 

  .. code:: json
  
    {
    
        "config":
        {
                "_comments__": "Config file created for 4 nodes, change expected results based on number of nodes",
            "container_image": "rocm/megatron-lm:v25.5_py310",
            "container_name": "megatron_llama3.1_310",
            "distributed_training": "True",
            "_example_nnodes": "4",
            "nnodes": "<changeme>",
            "_example_master_address": "X.X.X.X",
            "master_address": "<changeme>",
            "_example_training_iterations": "30",
            "training_iterations": "<changeme>",
            "_example_nic_type": "ainic|thor2|cx7",
            "nic_type": "<changeme>",
            "_example_nccl_ib_hca_list": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7",
            "nccl_ib_hca_list": "<changeme>",
            "_example_nccl_socket_ifname": "ens51f1np1",
            "nccl_socket_ifname": "<changeme>",
            "_example_gloo_socket_ifname": "ens51f1np1",
            "gloo_socket_ifname": "<changeme>",
            "_example_nccl_ib_gid_index": "3",
            "nccl_ib_gid_index": "<changeme>",
            "nccl_debug": "ERROR",
            "hf_token_file": "/home/{user-id}/.hf_token",
            "shm_size": "128G",
            "_comments_data_cache_dir": "This path should be accessible from all nodes like a common FS like NFS for distributed training",
            "data_cache_dir": "/home/{user-id}/cache",
            "mock_data": "True",
            "log_dir": "/home/{user-id}/LOG_DIR",
            "dataset_source":
            {
            },
            "container_config":
            {
                "device_list": [ "/dev/dri", "/dev/kfd", "/dev/infiniband/rdma_cm" ],
                "volume_dict":
                {
                "/home/{user-id}": "/home/{user-id}",
                "/dev/infiniband": "/dev/infiniband",
                "/usr/local/lib/libbnxt_re-rdmav34.so": "/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host",
                "/lib/libibverbs.d": "/lib/libibverbs.d",
                "/tmp/TRAINING_LOGS": "/workspace/Megatron-LM/output"
                }
            }
        },
        "model_params":
        {
            "multi_node":
            {
                "llama3_1_8b":
                {
                    "mi300x":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-8B",
                        "model_size": "8",
                        "batch_size": "128",
                        "micro_batch_size": "2",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
                            "_example_throughput_per_gpu": "610.0",
                            "_example_tokens_per_gpu": "12000.0",
                            "throughput_per_gpu": "<changeme>",
                            "tokens_per_gpu": "<changeme>"
                        }
                    },
                    "mi325":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-8B",
                        "model_size": "8",
                        "batch_size": "128",
                        "micro_batch_size": "2",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
                            "_example_throughput_per_gpu": "620.0",
                            "_example_tokens_per_gpu": "14000.0",
                            "throughput_per_gpu": "<changeme>",
                            "tokens_per_gpu": "<changeme>"
                        }
                    }
                },
                "llama3_1_70b":
                {
                    "mi300x":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-70B",
                        "model_size": "70",
                        "batch_size": "256",
                        "micro_batch_size": "4",
                        "precision": "TE_FP16",
                        "sequence_length": "8192",
                        "tensor_parallelism": "8",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
                            "_example_throughput_per_gpu": "530.0",
                            "_example_tokens_per_gpu": "1100.0",
                            "throughput_per_gpu": "<changeme>",
                            "tokens_per_gpu": "<changeme>"
                        }
                    },
                    "mi325":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-70B",
                        "model_size": "70",
                        "batch_size": "256",
                        "micro_batch_size": "4",
                        "precision": "TE_FP16",
                        "sequence_length": "8192",
                        "tensor_parallelism": "8",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
                            "_example_throughput_per_gpu": "550.0",
                            "_example_tokens_per_gpu": "1200.0",
                            "throughput_per_gpu": "<changeme>",
                            "tokens_per_gpu": "<changeme>"
                        }
                    }
                }
            }
        }
    
    }

Parameters
----------

.. |br| raw:: html

    <br />

Use the parameters in these tables to configure the training file.

``config``
~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``_comments__``
     - "Config file created for 4 nodes, change expected results based on number of nodes"
     - A generic comment
   * - ``container_image``
     - ``rocm/megatron-lm:v25.5_py310`` 
     - Docker image used to run Megatron-LM
   * - ``container_name``
     - ``megatron_llama3.1_310`` 
     - Name assigned to the container instance
   * - ``distributed_training``
     - True
     - "True"/"False": Ehether to run training across multiple nodes
   * - ``_example_nnodes``
     - 4
     - Example of number of cluster nodes participating in the job 
   * - ``Nnodes``
     - ``<changeme>``
     - Number of cluster nodes participating in the distributed job 
   * - ``_example_master_address``
     - "X.X.X.X"
     - Example IP of the master/coordinator node
   * - ``master_address``
     - ``<changeme>``
     - IP of the master/coordinator node
   * - ``_example_training_iterations``
     - 30
     - Example of number of training iterations/steps to run in this test
   * - ``training_iterations``
     - ``<changeme>`` 
     - Number of training iterations/steps to run in this test
   * - ``_example_nic_type``
     - ``ainic|thor2|cx7``
     - Example of NIC hardware type
   * - ``nic_type``
     - ``<changeme>``
     - NIC hardware type
   * - ``_example_nccl_ib_hca_list``
     - Values:
        - ``bnxt_re0``
        - ``bnxt_re1``
        - ``bnxt_re2``
        - ``bnxt_re3``
        - ``bnxt_re4``
        - ``bnxt_re5``
        - ``bnxt_re6``
        - ``bnxt_re7``
     - Example of a comma-separated list of InfiniBand HCA device names to use for NCCL/communication (multi-rail support)
   * - ``nccl_ib_hca_list``
     - ``<changeme>``
     - Comma-separated list of InfiniBand HCA device names to use for NCCL/communication (multi-rail support)
   * - ``_example_nccl_socket_ifname``
     - ``ens51f1np1``
     - Example of a network interface name used by NCCL Network interface name used by NCCL / control channels
   * - ``nccl_socket_ifname``
     - ``<changeme>``
     - Network interface name used by NCCL Network interface name used by NCCL / control channels
   * - ``_example_gloo_socket_ifname``
     - ``ens51f1np1`` 
     - Example of a network interface name used by Gloo control channels
   * - ``gloo_socket_ifname``
     - ``<changeme>``
     - Network interface name used by Gloo control channels
   * - ``nccl_ib_gid_index``
     - ``<changeme>``
     - GID index used for IB addressing (selects which GID)
   * - ``_example_nccl_ib_gid_index``
     - 3
     - Example of  GID index used for IB addressing (selects which GID entry on the HCA to use)
   * - ``nccl_debug``
     - ERROR
     - NCCL log level
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - Path to a Hugging Face token file used to download tokenized models/datasets requiring authorization
   * - ``shm_size``
     - 128G
     - Docker shared memory size
   * - ``_comments_data_cache_dir``
     - "This path should be accessible from all nodes like a common FS like NFS for distributed training"
     - A comment explaining ``data_cache_dir`` must be accessible from all nodes (NFS/shared FS).
   * - ``data_cache_dir``
     - ``/home/{user-id}/cache``
     - Dataset/cache directory (should be shared across nodes for distributed training unless using per-node copies)
   * - ``mock_data``
     - True
     - "True"/"False": Use synthetic data (True) to avoid real dataset downloads in CI/smoke tests 
   * - ``log_dir``
     - ``/home/{user-id}/LOG_DIR``
     - Path where training logs should be written on the host

``dataset_source/container_config``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``device_list``
     - Values:
        - ``"/dev/dri"``
        - ``"/dev/kfd"``
        - ``"/dev/infiniband/rdma_cm"``
     - Kernel devices exposed into the container
   * - ``volume_dict``
     - N/A
     - Host-to-container mounts: map host paths (home, RDMA libs, ``/lib/libibverbs.d``, log output) into the container so code and drivers are accessible.
   * - ``/home/{user-id}``
     - ``/home/{user-id}``
     - Mount user's home directory into container at the same path
   * - ``/dev/infiniband``
     - ``/dev/infiniband``
     - Expose InfiniBand device nodes into container
   * - ``/usr/local/`` |br| ``lib/libbnxt_`` |br| ``re-rdmav34.so``
     - ``/usr/lib/x86_64-`` |br| ``linux-gnu/libib`` |br| ``verbs/libbnxt_`` |br| ``re-rdmav34.so.host``
     - Mount host's Broadcom NIC driver library into container
   * - ``/lib/libibverbs.d``
     - ``/lib/libibverbs.d``
     - Mount InfiniBand verbs library configuration directory
   * - ``/tmp/TRAINING_LOGS``
     - ``/workspace/Megatr`` |br| ``on-LM/output`` 
     - Map host log directory to Megatron's expected output location inside container

``model_params/multi_node/llama3.1-8b/mi300x``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi300x``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-8B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 8
     - The abbreviated model size
   * - ``batch_size``
     - 128
     - Global batch size
   * - ``micro_batch_size``
     - 2
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 1
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 0
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 0
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "_example_throughput_per_gpu": "610.0",
          "_example_tokens_per_gpu": "12000.0",
          "throughput_per_gpu": "<changeme>",
          "tokens_per_gpu": "<changeme>"
      }

``model_params/multi_node/llama3.1-8b/mi325``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi325``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-8B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 8
     - The abbreviated model size
   * - ``batch_size``
     - 128
     - Global batch size
   * - ``micro_batch_size``
     - 2
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 1
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 0
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 0
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "_example_throughput_per_gpu": "610.0",
          "_example_tokens_per_gpu": "14000.0",
          "throughput_per_gpu": "<changeme>",
          "tokens_per_gpu": "<changeme>"
      }

``model_params/multi_node/llama3_1_70b/mi300x``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi300x``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-70B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 70
     - The abbreviated model size
   * - ``batch_size``
     - 256
     - Global batch size
   * - ``micro_batch_size``
     - 4
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 8
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 0
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 0
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "_example_throughput_per_gpu": "530.0",
          "_example_tokens_per_gpu": "1100.0",
          "throughput_per_gpu": "<changeme>",
          "tokens_per_gpu": "<changeme>"
      }

``model_params/multi_node/llama3_1_70b/mi325``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi325``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-70B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 70
     - The abbreviated model size
   * - ``batch_size``
     - 256
     - Global batch size
   * - ``micro_batch_size``
     - 4
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP16
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 8
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 0
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 0
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "_example_throughput_per_gpu": "550.0",
          "_example_tokens_per_gpu": "1200.0",
          "throughput_per_gpu": "<changeme>",
          "tokens_per_gpu": "<changeme>"
      }

Single node MI35x configuration 
===============================

The ``mi35x_singlenode_megatron_llama.json`` config file is used to run Megatron on a MI35x on a single node.

.. dropdown:: ``mi35x_singlenode_megatron_llama.json`` 

  .. code:: json

    {
 
        "config":

        {
            "container_image": "rocm/megatron-lm:v25.9_gfx950",
            "container_name": "megatron_llama3.1_310",
            "_example_nnodes": "4",
            "nnodes": "<changeme>-no of nodes to run singlenode training",
            "master_address": "localhost",
            "_example_training_iterations": "30",
            "training_iterations": "<changeme>",
            "hf_token_file": "/home/{user-id}/.hf_token",
            "shm_size": "128G",
            "_comments_data_cache_dir": "This path should be accessible from all nodes like a common FS like NFS for distributed training",
            "data_cache_dir": "/home/{user-id}/cache",
            "mock_data": "True",
            "log_dir": "/home/{user-id}/LOG_DIR",
            "dataset_source":
            {
            },
            "container_config":
            {
                "device_list": [ "/dev/dri", "/dev/kfd" ],
                "volume_dict":
                {
                "/home/{user-id}": "/home/{user-id}"
                }
            }
        },
        "model_params":
        {
            "single_node":
            {
                "llama3_1_8b":
                {
                    "mi350":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-8B",
                        "model_size": "8",
                        "batch_size": "128",
                        "micro_batch_size": "4",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                                "fsdp": "0",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "result_dict":
                        {
                            "tokens_per_gpu": "18000.0"
                        }
                    },
                    "mi355":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-8B",
                        "model_size": "8",
                        "batch_size": "128",
                        "micro_batch_size": "4",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                                "fsdp": "1",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "1",
                        "result_dict":
                        {
                            "tokens_per_gpu": "20000.0"
                        }
                    }
                },
                "llama3_1_70b":
                {
                    "mi350":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-70B",
                        "model_size": "70",
                        "batch_size": "24",
                        "micro_batch_size": "3",
                        "precision": "TE_FP16",
                        "sequence_length": "8192",
                                "fsdp": "1",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "1",
                        "result_dict":
                        {
                            "tokens_per_gpu": "2000.0"
                        }
                    },
                    "mi355":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-70B",
                        "model_size": "70",
                        "batch_size": "24",
                        "micro_batch_size": "3",
                        "precision": "TE_FP16",
                        "sequence_length": "8192",
                                "fsdp": "1",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "1",
                        "result_dict":
                        {
                            "tokens_per_gpu": "2100.0"
                        }
                    }
                }
            }
    
        }
  
    }
 
Parameters
----------

Use the parameters in these tables to configure the training file.

``config``
~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``container_image``
     - ``rocm/megatron-lm:v25.9_gfx950`` 
     - Docker image used to run Megatron-LM
   * - ``container_name``
     - ``megatron_llama3.1_310`` 
     - Name assigned to the container instance
   * - ``_example_nnodes``
     - 4
     - Example of number of cluster nodes participating in the job 
   * - ``Nnodes``
     - "``<changeme>``-no of nodes to run singlenode training"
     - Number of nodes in the distributed job 
   * - ``master_address``
     - ``<changeme>``
     - IP of the master/coordinator node
   * - ``_example_training_iterations``
     - 30
     - Example of number of training iterations/steps to run in this test
   * - ``training_iterations``
     - ``<changeme>``
     - Number of training iterations/steps to run in this test
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - Path to a Hugging Face token file used to download tokenized models/datasets requiring authorization
   * - ``shm_size``
     - 128G
     - Docker shared memory size mounted into container
   * - ``_comments_data_cache_dir``
     - "This path should be accessible from all nodes like a common FS like NFS for distributed training"
     - Comment explaining ``data_cache_dir`` must be accessible from all nodes
   * - ``data_cache_dir``
     - ``/home/{user-id}/cache``
     - Dataset/cache directory
   * - ``mock_data``
     - True
     - "True"/"False": Use synthetic data (True) to avoid real dataset downloads in CI/smoke tests 
   * - ``log_dir``
     - ``/home/{user-id}/LOG_DIR``
     - Path where training logs should be written on the host

``dataset_source/container_config``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``device_list``
     - Values:
        - ``"/dev/dri"``
        - ``"/dev/kfd"``
        - ``"/dev/infiniband/rdma_cm"``
     - Kernel devices exposed into the container
   * - ``volume_dict``
     - N/A
     - Host-to-container mounts: map host paths (home, RDMA libs, ``/lib/libibverbs.d``, log output) into the container so code and drivers are accessible
   * - ``/home/<changeme>``
     - ``/home/{user-id}``
     - The user's directory being used

``model_params/single_node/llama3_1_8b/mi350``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi350``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-8B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 8
     - The abbreviated model size
   * - ``batch_size``
     - 128
     - Global batch size
   * - ``micro_batch_size``
     - 2
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP8
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 1
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 0
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 0
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "tokens_per_gpu": "18000.0"
      }

``model_params/single_node/llama3_1_8b/mi355``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi355``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-8B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 8
     - The abbreviated model size
   * - ``batch_size``
     - 128
     - Global batch size
   * - ``micro_batch_size``
     - 4
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP8
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 1
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 1
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 1
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "tokens_per_gpu": "20000.0"
      }

``model_params/single_node/llama3_1_70b/mi350``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi350``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-70B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 70
     - The abbreviated model size
   * - ``batch_size``
     - 24
     - Global batch size
   * - ``micro_batch_size``
     - 3
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP16
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 1
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 1
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 1
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "tokens_per_gpu": "2000.0"
      }

``model_params/single_node/llama3_1_70b/mi355``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``mi355``
     - N/A
     - The GPU being used
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-70B`` 
     - HF model identifier or local path used to initialize tokenizer
   * - ``model_size``
     - 70
     - The abbreviated model size
   * - ``batch_size``
     - 24
     - Global batch size
   * - ``micro_batch_size``
     - 3
     - Per-device micro-batch size
   * - ``precision``
     - TE_FP16
     - Numeric precision mode used
   * - ``sequence_length``
     - 8192
     - Maximum sequence length / context size
   * - ``tensor_parallelism``
     - 1
     - Degree of tensor-model parallelism
   * - ``pipeline_parallelism``
     - 1
     - Pipeline parallel stages count
   * - ``recompute``
     - 1
     - Enable activation recomputation/checkpointing to reduce memory at cost of extra compute
   * - ``fsdp``
     - 1
     - Whether FSDP-style fully-sharded data-parallel is enabled

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
          "tokens_per_gpu": "2100.0"
      }