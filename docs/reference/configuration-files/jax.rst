.. meta::
  :description: Configure the variables in the JAX training configuration files
  :keywords: training, ROCm, install, cvs, JAX,

**********************************
JAX training configuration files
**********************************

The JAX training tests perform end-to-end orchestration and validation of a distributed JAX training run (Llama 3.1) across a multi-node cluster using Docker containers, 
remote execution, and the project's JAX library. 

Change the parameters as needed in the JAX training configuration files: ``mi300x_singlenode_llama3_1_70b.json`` and ``mi35x_singlenode_llama_3_1_70b.json`` for single node configuration,
and ``mi300x_distributed_llama3_1_70b.json`` and ``mi300x_distributed_llama_3_1_405b.json`` for distributed node configuration.

.. note::

  - Parameters with the ``<changeme>`` value must have that value modified to your specifications. 
  - ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username. 

Single node configuration
=========================

``mi300x_singlenode_llama3_1_70b.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the ``mi300x_singlenode_llama3_1_70b.json`` configuration file:

.. dropdown:: ``mi300x_singlenode_llama3_1_70b.json``

  .. code:: json
    
    {
    
        "config":
        {
            "container_image": "rocm/jax-training:maxtext-v25.9",
            "container_name": "rocm-jax-llama3.1-70b",
            "distributed_training": "True",
            "enable_checkpointing": "False",
            "nnodes": "<changeme> = number of nodes used in singlenode training",
            "coordinator_ip": "localhost",
            "training_steps": "30",
            "gpu_max_hw_queues": "2",
            "nvte_ck_bwd_v3": "1",
            "nvte_ck_v3_bf16_cvt": "2",
            "xla_python_client_mem_fraction": "0.975",
                "xla_gpu_executable_warn_stuck_timeout": "90",
            "hf_token_file": "/home/{user-id}/.hf_token",
            "shm_size": "256G",
            "_comments_data_cache_dir": "This path should be accessible from all nodes like a common FS like NFS for distributed training",
            "data_cache_dir": "/home/{user-id}/cache",
            "mock_data": "True",
            "log_dir": "/home/{user-id}/LOGS",
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
                "/tmp/TRAINING_LOGS": "/workspace/maxtext/output"
                },
                "env_dict":
                {
                    "JAX_COORDINATOR_IP": "<changeme>",
                    "JAX_COORDINATOR_PORT": "1234",
                    "NNODES": "1",
                    "JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS": "1800",
                            "JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT_SECONDS": "900"
                }
            }
        },
        "model_params":
        {
            "single_node":
            {
                "llama3.1-70b":
                {
                    "mi300x":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-70B",
                        "model_size": "70",
                        "attention": "cudnn_flash_te",
                        "dcn_data_parallelism": "1",
                        "dcn_fsdp_parallelism": "-1",
                        "dcn_pipeline_parallelism": "1",
                        "dcn_tensor_parallelism": "1",
                        "dcn_sequence_parallelism": "1",
                        "ici_fsdp_parallelism": "8",
                        "ici_data_parallelism": "1",
                        "ici_sequence_parallelism": "1",
                        "ici_tensor_parallelism": "1",
                        "ici_pipeline_parallelism": "1",
                        "remat_policy": "full",
                        "use_iota_embed": "true",
                        "scan_layers": "true",
                        "dataset_type": "synthetic",
                        "hf_path": "parquet",
                        "hf_train_files": "/home/{user-id}/cache/maxtext/data/c4/000*.parquet",
                        "tokenizer_path": "/home/{user-id}/cache/maxtext/Meta-Llama-70-B",
                        "async_checkpointing": "false",
                        "logits_dot_in_fp32": "false",
                        "megablox": "false",
                        "dtype": "bfloat16",
                        "quantization": "",
                        "quantize_kvcache": "false",
                        "kv_quant_axis": "heads_and_dkv",
                        "kv_quant_dtype": "int8",
                        "weight_dtype": "bfloat16",
                        "checkpoint_is_quantized": "false",
                        "per_device_batch_size": "2",
                        "max_target_length": "8192",
                        "skip_first_n_steps_for_profiler": "3",
    
                        "xla_flags":
                        {
                            "xla_gpu_enable_cublaslt": "True",
                                        "xla_gpu_executable_warn_stuck_timeout": "90",
                            "xla_gpu_executable_terminate_timeout": "300",
                            "xla_gpu_first_collective_call_warn_stuck_timeout_seconds": "300",
                            "xla_gpu_first_collective_call_terminate_timeout_seconds": "1200",
                            "xla_gpu_graph_level": "0",
                            "xla_gpu_autotune_level": "4",
                            "xla_gpu_enable_reduce_scatter_combine_by_dim": "false",
                            "xla_gpu_reduce_scatter_combine_threshold_bytes": "8589934592",
                            "xla_gpu_all_reduce_combine_threshold_bytes": "8589934592",
                            "xla_gpu_all_gather_combine_threshold_bytes": "137438953472",
                            "xla_gpu_enable_all_gather_combine_by_dim": "FALSE"
                        },
    
                        "base_emb_dim": "16384",
                        "base_num_query_heads": "128",
                        "base_num_kv_heads": "8",
                        "base_num_decoder_layers": "126",
                        "base_mlp_dim": "53248",
                        "head_dim": "128",
                        "mlp_activations": ["silu","linear"],
                        "vocab_size": "128256",
                        "enable_dropout": "False",
                        "logits_via_embedding": "False",
                        "normalization_layer_epsilon": "1.0e-5",
                        "rope_max_timescale": "500_000",
                        "decoder_block": "llama2",
    
                        "micro_batch_size": "2",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
                                        "tflops_per_sec_per_gpu": "1800.0",
                            "tokens_per_sec_per_gpu": "900.0"
                        }
                    }
                }
            }
    
        }
    
    }


Parameters (MI300X)
^^^^^^^^^^^^^^^^^^^

Use the parameters in these tables to configure the training file.

``config``
""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``container_image``
     - ``rocm/jax-training:maxtext-v25.9`` 
     - Docker image used for training 
   * - ``container_name``
     - ``rocm-jax-llama3.1-70b`` 
     - Name assigned to the container instance
   * - ``distributed_training``
     - True
     - Enable multi-node distributed training   
   * - ``enable_checkpointing``
     - False
     - Whether to write training checkpoints
   * - ``Nnodes``
     - ``<changeme>``
     - Number of nodes in the distributed job 
   * - ``coordinator_ip``
     - ``<changeme>``
     - IP address of the coordinator (master) node used for rendezvous
   * - ``training_steps``
     - 30
     - Number of training steps to run
   * - ``gpu_max_hw_queues``
     - 2 
     - Hardware queue count
   * - ``nccl_proto``
     - Simple
     - NCCL protocol selection 
   * - ``gpu_max_hw_queues``
     - 2
     - Max hardware queues exposed to GPU
   * - ``nvte_ck_bwd_v3``
     - 1 
     - Custom NVTE/JAX options controlling checkpointing / backward passes conversions
   * - ``nvte_ck_v3_bf16_cvt``
     - 2
     - Custom NVTE/JAX options controlling checkpointing / backward passes and bf16 conversions
   * - ``xla_python_client_mem_fraction``
     - 0.975
     - Fraction of host memory XLA client may use
   * - ``xla_gpu_executable_warn_stuck_timeout``
     - 90
     - Warning if GPU kernel runs too long
   * - ``nccl_proto``
     - Simple
     - NCCL protocol selection
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - Path to Hugging Face token file for model access
   * - ``shm_size``
     - 256G
     - Docker shared memory size for container
   * - ``_comments_data_cache_dir``
     - "This path should be accessible from all nodes like a common FS like NFS for distributed training"
     - A comment stating that ``data_cache_dir`` should be shared across nodes
   * - ``data_cache_dir``
     - ``/home/{user-id}/cache``
     - Path where datasets/cache will be stored
   * - ``mock_data``
     - True
     - Use synthetic or mock dataset
   * - ``log_dir``
     - ``/home/{user-id}/LOGS``
     - Directory for logs

``container_config``
""""""""""""""""""""

.. |br| raw:: html

    <br />

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
     - Host-to-container volume mounts
   * - ``/home/{user-id}``
     - ``/home/{user-id}``
     - The user’s home directory 
   * - ``/dev/infiniband``
     - ``/dev/infiniband``
     - Exposes InfiniBand device files for RDMA networking 
   * - ``/usr/local/lib/`` |br| ``libbnxt_re-rdmav34.so``
     - ``/usr/lib/x86_64-linux`` |br| ``-gnu/libibverbs/libbnxt`` |br| ``_re-rdmav34.so.host``
     - Provides the ``bnxt_re`` RDMA provider (Broadcom NIC) inside the container 
   * - ``/lib/libibverbs.d``
     - ``/lib/libibverbs.d``
     - Mounts the RDMA verbs configuration directory — needed by libibverbs to detect RDMA devices
   * - ``/tmp/TRAINING_LOGS``
     - ``/workspace/maxtext/output`` 
     - Stores training logs outside the container 
   * - ``env_dict``
     - N/A
     - Environment variables injected into container, including JAX coordinator IP/port, ``NNODES``, and timeout/heartbeat settings 
   * - ``JAX_COORDINATOR_IP``
     - ``<changeme>``
     - The IP address of the coordinator node in the cluster — all other nodes connect here to initialize 
   * - ``JAX_COORDINATOR_PORT``
     - 12345
     - The port used by JAX for coordination 
   * - ``NNODES``
     - 1
     - Total number of nodes in the distributed job
   * - ``JAX_DISTRIBUTED_INITIAL`` |br| ``IZATION_TIMEOUT_SECONDS``
     - 1800
     - How long JAX waits for all nodes to join the distributed setup — longer is safer for large clusters 
   * - ``JAX_DISTRIBUTED_HEART`` |br| ``BEAT_TIMEOUT_SECONDS``
     - 900
     - Timeout for communication heartbeat


``model_params/single_node/llama3.1-70b/mi300x``
""""""""""""""""""""""""""""""""""""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-70B``
     - HF model identifier used to load tokenizer 
   * - ``model_size``
     - 70
     - Model size label
   * - ``attention``
     - ``cudnn_flash_te``
     - Attention implementation
   * - ``dcn_data_parallelism``
     - 1
     - Parallelism / control flags
   * - ``dcn_fsdp_parallelism``
     - -1
     - Parallelism settings for the training strategy
   * - ``dcn_pipeline_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``dcn_tensor_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``dcn_sequence_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``ici_fsdp_parallelism``
     - 8
     - Interconnect / implementation-specific 
   * - ``ici_data_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_sequence_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_tensor_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_pipeline_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``remat_policy``
     - full
     - Checkpointing/rematerialization policy
   * - ``use_iota_embed``
     - true
     - Use IOTA embeddings
   * - ``scan_layers``
     - true
     - Enable layer scanning optimization 
   * - ``dataset_type``
     - synthetic
     - Source type
   * - ``hf_path``
     - parquet
     - Paths to training data and tokenizer
   * - ``hf_train_files``
     - ``/home/{user-id}/maxtext`` |br| ``/data/c4/000*.parquet``
     - Paths to training data and tokenizer
   * - ``tokenizer_path``
     - ``/home/{user-id}/maxtext`` |br| ``Meta-Llama-70-B``
     - Paths to tokenizer
   * - ``async_checkpointing``
     - false
     - Async checkpointing
   * - ``logits_dot_in_fp32``
     - false
     - Compute logits dot in FP32
   * - ``megablox``
     - false
     - Megablox optimization
   * - ``dtype``
     - bfloat16
     - Base compute dtype 
   * - ``batch_size``
     - 128
     - Global batch size


``quantization``
""""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``quantize_kvcache``
     - false
     - KV cache quantization settings 
   * - ``kv_quant_axis``
     - ``heads_and_dkv`` 
     - KV cache quantization settings
   * - ``kv_quant_dtype``
     - int8
     - KV cache quantization settings
   * - ``weight_dtype``
     - bfloat16
     - Weight numeric type
   * - ``checkpoint_is_quantized``
     - false
     - Whether stored checkpoints are quantized
   * - ``per_device_batch_size``
     - 2
     - Micro-batch assigned per device for data parallelism 
   * - ``max_target_length``
     - 8192
     - Sequence length for generation/training
   * - ``skip_first_n_steps_for_profiler``
     - 3
     - Number of initial steps to skip when profiling

``xla_flags``
""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``xla_gpu_enable_cublaslt``
     - True
     - Use cublasLt optimized routines 
   * - ``xla_gpu_exec`` |br| ``utable_warn`` |br| ``_stuck_timeout``
     - 90
     - Warn if GPU kernel runs too long
   * - ``xla_gpu_exec`` |br| ``utable_term`` |br| ``inate_timeout``
     - 300
     - Sets how long (in seconds) XLA waits before terminating a GPU executable that appears hung during execution
   * - ``xla_gpu_first_`` |br| ``collective_call`` |br| ``_warn_stuck`` |br| ``_timeout_seconds``
     - 300
     - Time before XLA logs a warning if a collective GPU operation hasn't completed yet
   * - ``xla_gpu_first_`` |br| ``collective_call`` |br| ``_terminate_ti`` |br| ``meout_seconds``
     - 1200
     - Time (in seconds) before XLA terminates the first collective GPU call if it's still not finished
   * - ``xla_gpu_graph_level``
     - 0
     - Graph compilation level 
   * - ``xla_gpu_autotune_level``
     - 4
     - Autotune aggressiveness
   * - ``xla_gpu_enable_reduce`` |br| ``_scatter_combine_by_dim``
     - false
     - Combine reduce-scatter by dim 
   * - ``xla_gpu_reduce_scatter`` |br| ``_combine_threshold_bytes``
     - 8589934592 
     - Byte threshold to combine
   * - ``xla_gpu_all_reduce_`` |br| ``combine_threshold_bytes``
     - 8589934592
     - Combine threshold for all-reduce 
   * - ``xla_gpu_all_gather_`` |br| ``combine_threshold_bytes``
     - 137438953472 
     - Threshold for all-gather 
   * - ``xla_gpu_enable_all_`` |br| ``gather_combine_by_dim``
     - FALSE
     - Enable all-gather combine by dimension 
   * - ``base_emb_dim``
     - 16384
     - Model architecture sizes 
   * - ``base_num_query_heads``
     - 128
     - Model architecture sizes 
   * - ``base_num_kv_heads``
     - 8
     - Model architecture sizes  
   * - ``base_num_decoder_layers1``
     - 126
     - Model architecture sizes 
   * - ``base_mlp_dim``
     - 53248
     - Model architecture sizes 
   * - ``head_dim1``
     - 128
     - Model architecture sizes 
   * - ``mlp_activations``
     - Values:
        - ``"silu"``
        - ``"linear"``
     - Activation functions used 
   * - ``vocab_size``
     - 128526
     - Token vocabulary size 
   * - ``enable_dropout``
     - False
     - Whether dropout is enabled for training
   * - ``logits_via_embedding``
     - False
     - Uses the same embedding matrix for output logits
   * - ``normalization_layer_epsilon``
     - 1.0e-5 
     - Numeric epsilon for layernorm stability 
   * - ``rope_max_timescale``
     - 500_000 
     - Rotary positional encoding max timescale 
   * - ``decoder_block``
     - llama2 
     - Decoder block type 
   * - ``micro_batch_size``
     - 2
     - Micro-batch size per step 
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode 
   * - ``sequence_length``
     - 8192
     - Sequence length
   * - ``tensor_parallelism``
     - 1
     - Parallelism degrees
   * - ``pipeline_parallelism``
     - 1
     - Parallelism degrees
   * - ``recompute``
     - 0
     - Recompute flags
   * - ``fsdp``
     - 0
     - FSDP

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
                      "tflops_per_sec_per_gpu": "1800.0",
          "tokens_per_sec_per_gpu": "900.0"
      }



``mi35x_singlenode_llama_3_1_70b.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the ``mi35x_singlenode_llama_3_1_70b.json`` configuration file:

.. dropdown:: ``mi35x_singlenode_llama_3_1_70b.json``

  .. code:: json

    {
 
      "config":
      {
          "container_image": "rocm/jax-training:maxtext-v25.9",
          "container_name": "rocm-jax-llama3.1-70b",
          "distributed_training": "True",
          "enable_checkpointing": "False",
          "nnodes": "<changeme>-number of nodes on which to run single node training",
          "coordinator_ip": "localhost",
          "training_steps": "30",
          "gpu_max_hw_queues": "2",
          "nvte_ck_bwd_v3": "1",
          "nvte_ck_v3_bf16_cvt": "2",
          "xla_python_client_mem_fraction": "0.975",
              "xla_gpu_executable_warn_stuck_timeout": "90",
          "hf_token_file": "/home/{user-id}/.hf_token",
          "shm_size": "256G",
          "_comments_data_cache_dir": "This path should be accessible from all nodes like a common FS like NFS for distributed training",
          "data_cache_dir": "/home/{user-id}/cache",
          "mock_data": "True",
          "log_dir": "/home/{user-id}/LOGS",
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
              "/tmp/TRAINING_LOGS": "/workspace/maxtext/output"
              },
              "env_dict":
              {
                  "JAX_COORDINATOR_IP": "<changeme>",
                  "JAX_COORDINATOR_PORT": "1234",
                  "NNODES": "1",
                  "JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS": "1800",
                          "JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT_SECONDS": "900"
              }
          }
      },
      "model_params":
      {
          "single_node":
          {
              "llama3.1-70b":
              {
                  "mi355":
                  {
                      "tokenizer_model": "NousResearch/Meta-Llama-3-70B",
                      "model_size": "70",
                      "attention": "cudnn_flash_te",
                      "dcn_data_parallelism": "1",
                      "dcn_fsdp_parallelism": "-1",
                      "dcn_pipeline_parallelism": "1",
                      "dcn_tensor_parallelism": "1",
                      "dcn_sequence_parallelism": "1",
                      "ici_fsdp_parallelism": "8",
                      "ici_data_parallelism": "1",
                      "ici_sequence_parallelism": "1",
                      "ici_tensor_parallelism": "1",
                      "ici_pipeline_parallelism": "1",
                      "remat_policy": "full",
                      "use_iota_embed": "true",
                      "scan_layers": "true",
                      "dataset_type": "synthetic",
                      "hf_path": "parquet",
                      "hf_train_files": "/home/{user-id}/cache/maxtext/data/c4/000*.parquet",
                      "tokenizer_path": "/home/{user-id}/cache/maxtext/Meta-Llama-70-B",
                      "async_checkpointing": "false",
                      "logits_dot_in_fp32": "false",
                      "megablox": "false",
                      "dtype": "bfloat16",
                      "quantization": "",
                      "quantize_kvcache": "false",
                      "kv_quant_axis": "heads_and_dkv",
                      "kv_quant_dtype": "int8",
                      "weight_dtype": "bfloat16",
                      "checkpoint_is_quantized": "false",
                      "per_device_batch_size": "2",
                      "max_target_length": "8192",
                      "skip_first_n_steps_for_profiler": "3",
  
                      "xla_flags":
                      {
                          "xla_gpu_enable_cublaslt": "True",
                                      "xla_gpu_executable_warn_stuck_timeout": "90",
                          "xla_gpu_executable_terminate_timeout": "300",
                          "xla_gpu_first_collective_call_warn_stuck_timeout_seconds": "300",
                          "xla_gpu_first_collective_call_terminate_timeout_seconds": "1200",
                          "xla_gpu_graph_level": "0",
                          "xla_gpu_autotune_level": "4",
                          "xla_gpu_enable_reduce_scatter_combine_by_dim": "false",
                          "xla_gpu_reduce_scatter_combine_threshold_bytes": "8589934592",
                          "xla_gpu_all_reduce_combine_threshold_bytes": "8589934592",
                          "xla_gpu_all_gather_combine_threshold_bytes": "137438953472",
                          "xla_gpu_enable_all_gather_combine_by_dim": "FALSE"
                      },
  
                      "base_emb_dim": "16384",
                      "base_num_query_heads": "128",
                      "base_num_kv_heads": "8",
                      "base_num_decoder_layers": "126",
                      "base_mlp_dim": "53248",
                      "head_dim": "128",
                      "mlp_activations": ["silu","linear"],
                      "vocab_size": "128256",
                      "enable_dropout": "False",
                      "logits_via_embedding": "False",
                      "normalization_layer_epsilon": "1.0e-5",
                      "rope_max_timescale": "500_000",
                      "decoder_block": "llama2",
  
                      "micro_batch_size": "2",
                      "precision": "TE_FP8",
                      "sequence_length": "8192",
                      "tensor_parallelism": "1",
                      "pipeline_parallelism": "1",
                      "recompute": "0",
                      "fsdp": "0",
                      "result_dict":
                      {
                                      "tflops_per_sec_per_gpu": "900.0",
                          "tokens_per_sec_per_gpu": "2100.0"
                      }
                  }
              }
          }
  
      }
  
    }
 
Parameters (MI35X)
^^^^^^^^^^^^^^^^^^

Use the parameters in these tables to configure the training file.

``config``
""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``container_image``
     - ``rocm/jax-training:maxtext-v25.9`` 
     - Docker image used for training 
   * - ``container_name``
     - ``rocm-jax-llama3.1-70b`` 
     - Name assigned to the container instance
   * - ``distributed_training``
     - True
     - Enable multi-node distributed training   
   * - ``enable_checkpointing``
     - False
     - Whether to write training checkpoints
   * - ``Nnodes``
     - ``<changeme>``-number of nodes on which to run single node training
     - Number of nodes in the distributed job 
   * - ``coordinator_ip``
     - localhost
     - IP address of the coordinator (master) node used for rendezvous
   * - ``training_steps``
     - 30
     - Number of training steps to run
   * - ``gpu_max_hw_queues``
     - 2 
     - Hardware queue count
   * - ``gpu_max_hw_queues``
     - 2
     - Max hardware queues exposed to GPU
   * - ``nvte_ck_bwd_v3``
     - 1 
     - Custom NVTE/JAX options controlling checkpointing / backward passes conversions
   * - ``nvte_ck_v3_bf16_cvt``
     - 2
     - Custom NVTE/JAX options controlling checkpointing / backward passes and bf16 conversions
   * - ``xla_python_client_mem_fraction``
     - 0.975
     - Fraction of host memory XLA client may use
   * - ``xla_gpu_executable_warn_stuck_timeout``
     - 90
     - Warning if GPU kernel runs too long
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - Path to Hugging Face token file for model access
   * - ``shm_size``
     - 256G
     - Docker shared memory size for container
   * - ``_comments_data_cache_dir``
     - "This path should be accessible from all nodes like a common FS like NFS for distributed training"
     - A comment stating that ``data_cache_dir`` should be shared across nodes
   * - ``data_cache_dir``
     - ``/home/{user-id}/cache``
     - Path where datasets/cache will be stored
   * - ``mock_data``
     - True
     - Use synthetic or mock dataset
   * - ``log_dir``
     - ``/home/{user-id}/LOGS``
     - Directory for logs

``container_config``
""""""""""""""""""""

.. |br| raw:: html

    <br />

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
     - Host-to-container volume mounts
   * - ``/home/{user-id}``
     - ``/home/{user-id}``
     - The user's home directory 
   * - ``/dev/infiniband``
     - ``/dev/infiniband``
     - Exposes InfiniBand device files for RDMA networking 
   * - ``/usr/local/lib/`` |br| ``libbnxt_re-rdmav34.so``
     - ``/usr/lib/x86_64-linux`` |br| ``-gnu/libibverbs/libbnxt`` |br| ``_re-rdmav34.so.host``
     - Provides the ``bnxt_re`` RDMA provider (Broadcom NIC) inside the container 
   * - ``/lib/libibverbs.d``
     - ``/lib/libibverbs.d``
     - Mounts the RDMA verbs configuration directory — needed by libibverbs to detect RDMA devices
   * - ``/tmp/TRAINING_LOGS``
     - ``/workspace/maxtext/output`` 
     - Stores training logs outside the container 
   * - ``env_dict``
     - N/A
     - Environment variables injected into container, including JAX coordinator IP/port, ``NNODES``, and timeout/heartbeat settings 
   * - ``JAX_COORDINATOR_IP``
     - ``<changeme>``
     - The IP address of the coordinator node in the cluster — all other nodes connect here to initialize 
   * - ``JAX_COORDINATOR_PORT``
     - 12345
     - The port used by JAX for coordination 
   * - ``NNODES``
     - 1
     - Total number of nodes in the distributed job
   * - ``JAX_DISTRIBUTED_INITIAL`` |br| ``IZATION_TIMEOUT_SECONDS``
     - 1800
     - How long JAX waits for all nodes to join the distributed setup — longer is safer for large clusters 
   * - ``JAX_DISTRIBUTED_HEART`` |br| ``BEAT_TIMEOUT_SECONDS``
     - 900
     - Timeout for communication heartbeat


``model_params/single_node/llama3.1-70b/mi355``
""""""""""""""""""""""""""""""""""""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-70B``
     - HF model identifier used to load tokenizer 
   * - ``model_size``
     - 70
     - Model size label
   * - ``attention``
     - ``cudnn_flash_te``
     - Attention implementation
   * - ``dcn_data_parallelism``
     - 1
     - Parallelism / control flags
   * - ``dcn_fsdp_parallelism``
     - -1
     - Parallelism settings for the training strategy
   * - ``dcn_pipeline_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``dcn_tensor_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``dcn_sequence_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``ici_fsdp_parallelism``
     - 8
     - Interconnect / implementation-specific 
   * - ``ici_data_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_sequence_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_tensor_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_pipeline_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``remat_policy``
     - full
     - Checkpointing/rematerialization policy
   * - ``use_iota_embed``
     - true
     - Use IOTA embeddings
   * - ``scan_layers``
     - true
     - Enable layer scanning optimization 
   * - ``dataset_type``
     - synthetic
     - Source type
   * - ``hf_path``
     - parquet
     - Paths to training data and tokenizer
   * - ``hf_train_files``
     - ``/home/{user-id}/maxtext`` |br| ``/data/c4/000*.parquet``
     - Paths to training data and tokenizer
   * - ``tokenizer_path``
     - ``/home/{user-id}/maxtext`` |br| ``Meta-Llama-70-B``
     - Paths to tokenizer
   * - ``async_checkpointing``
     - false
     - Async checkpointing
   * - ``logits_dot_in_fp32``
     - false
     - Compute logits dot in FP32
   * - ``megablox``
     - false
     - Megablox optimization
   * - ``dtype``
     - bfloat16
     - Base compute dtype 
   * - ``batch_size``
     - 128
     - Global batch size


``quantization``
""""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``quantize_kvcache``
     - false
     - KV cache quantization settings 
   * - ``kv_quant_axis``
     - ``heads_and_dkv`` 
     - KV cache quantization settings
   * - ``kv_quant_dtype``
     - int8
     - KV cache quantization settings
   * - ``weight_dtype``
     - bfloat16
     - Weight numeric type
   * - ``checkpoint_is_quantized``
     - false
     - Whether stored checkpoints are quantized
   * - ``per_device_batch_size``
     - 2
     - Micro-batch assigned per device for data parallelism 
   * - ``max_target_length``
     - 8192
     - Sequence length for generation/training
   * - ``skip_first_n_steps_for_profiler``
     - 3
     - Number of initial steps to skip when profiling

``xla_flags``
""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``xla_gpu_enable_cublaslt``
     - True
     - Use cublasLt optimized routines 
   * - ``xla_gpu_exec`` |br| ``utable_warn`` |br| ``_stuck_timeout``
     - 90
     - Warn if GPU kernel runs too long
   * - ``xla_gpu_exec`` |br| ``utable_term`` |br| ``inate_timeout``
     - 300
     - Sets how long (in seconds) XLA waits before terminating a GPU executable that appears hung during execution
   * - ``xla_gpu_first_`` |br| ``collective_call`` |br| ``_warn_stuck`` |br| ``_timeout_seconds``
     - 300
     - Time before XLA logs a warning if a collective GPU operation hasn't completed yet
   * - ``xla_gpu_first_`` |br| ``collective_call`` |br| ``_terminate_ti`` |br| ``meout_seconds``
     - 1200
     - Time (in seconds) before XLA terminates the first collective GPU call if it's still not finished
   * - ``xla_gpu_graph_level``
     - 0
     - Graph compilation level 
   * - ``xla_gpu_autotune_level``
     - 4
     - Autotune aggressiveness
   * - ``xla_gpu_enable_reduce`` |br| ``_scatter_combine_by_dim``
     - false
     - Combine reduce-scatter by dim 
   * - ``xla_gpu_reduce_scatter`` |br| ``_combine_threshold_bytes``
     - 8589934592 
     - Byte threshold to combine
   * - ``xla_gpu_all_reduce_`` |br| ``combine_threshold_bytes``
     - 8589934592
     - Combine threshold for all-reduce 
   * - ``xla_gpu_all_gather_`` |br| ``combine_threshold_bytes``
     - 137438953472 
     - Threshold for all-gather 
   * - ``xla_gpu_enable_all_`` |br| ``gather_combine_by_dim``
     - FALSE
     - Enable all-gather combine by dimension 
   * - ``base_emb_dim``
     - 16384
     - Model architecture sizes 
   * - ``base_num_query_heads``
     - 128
     - Model architecture sizes 
   * - ``base_num_kv_heads``
     - 8
     - Model architecture sizes  
   * - ``base_num_decoder_layers1``
     - 126
     - Model architecture sizes 
   * - ``base_mlp_dim``
     - 53248
     - Model architecture sizes 
   * - ``head_dim1``
     - 128
     - Model architecture sizes 
   * - ``mlp_activations``
     - Values:
        - ``"silu"``
        - ``"linear"``
     - Activation functions used 
   * - ``vocab_size``
     - 128526
     - Token vocabulary size 
   * - ``enable_dropout``
     - False
     - Whether dropout is enabled for training
   * - ``logits_via_embedding``
     - False
     - Uses the same embedding matrix for output logits
   * - ``normalization_layer_epsilon``
     - 1.0e-5 
     - Numeric epsilon for layernorm stability 
   * - ``rope_max_timescale``
     - 500_000 
     - Rotary positional encoding max timescale 
   * - ``decoder_block``
     - llama2 
     - Decoder block type 
   * - ``micro_batch_size``
     - 2
     - Micro-batch size per step 
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode 
   * - ``sequence_length``
     - 8192
     - Sequence length
   * - ``tensor_parallelism``
     - 1
     - Parallelism degrees
   * - ``pipeline_parallelism``
     - 1
     - Parallelism degrees
   * - ``recompute``
     - 0
     - Recompute flags
   * - ``fsdp``
     - 0
     - FSDP

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {
                      "tflops_per_sec_per_gpu": "900.0",
          "tokens_per_sec_per_gpu": "2100.0"
      }


Distributed node configuration
==============================

``mi300x_distributed_llama_3_1_405b.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the ``mi300x_distributed_llama_3_1_405b.json`` configuration file:

.. dropdown:: ``mi300x_distributed_llama_3_1_405b.json``

  .. code:: json

    {
    
        "config":
        {
            "container_image": "rocm/jax-training:maxtext-v25.9",
            "container_name": "rocm-jax-llama3.1-405b",
            "distributed_training": "True",
            "enable_checkpointing": "False",
            "nnodes": "<changeme>",
            "coordinator_ip": "<changeme>",
            "training_steps": "30",
            "nic_type": "thor2",
            "_example_nccl_ib_hca_list": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8",
            "nccl_ib_hca_list": "<changeme>",
            "_example_nccl_ib_hca": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8",
            "nccl_ib_hca": "<changeme>",
            "nccl_socket_ifname": "<changeme>",
            "gloo_socket_ifname": "<changeme>",
            "_example_nccl_ib_gid_index": "3",
            "nccl_ib_gid_index": "<changeme>",
            "nccl_debug": "ERROR",
            "nccl_proto": "Simple",
            "gpu_max_hw_queues": "2",
            "nccl_ib_tc": "41",
            "nccl_ib_sl": "0",
            "nccl_checks_disable": "1",
            "nvte_ck_bwd_v3": "1",
            "nvte_ck_v3_bf16_cvt": "2",
            "xla_python_client_mem_fraction": "0.975",
                "xla_gpu_executable_warn_stuck_timeout": "90",
            "hf_token_file": "/home/{user-id}/.hf_token",
            "shm_size": "256G",
            "_comments_data_cache_dir": "This path should be accessible from all nodes like a common FS like NFS for distributed training",
            "data_cache_dir": "/home/{user-id}/cache",
            "mock_data": "True",
            "log_dir": "/home/{user-id}/LOGS",
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
                "/tmp/TRAINING_LOGS": "/workspace/maxtext/output"
                },
                "env_dict":
                {
                    "JAX_COORDINATOR_IP": "<changeme>",
                    "JAX_COORDINATOR_PORT": "1234",
                    "NNODES": "1",
                    "JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS": "1800",
                            "JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT_SECONDS": "900"
                }
            }
        },
        "model_params":
        {
            "multi_node":
            {
                "llama3.1-405b":
                {
                    "mi300x":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-405B",
                        "model_size": "405",
                                "hardware": "gpu",
                        "attention": "cudnn_flash_te",
                        "dcn_data_parallelism": "1",
                        "dcn_fsdp_parallelism": "-1",
                        "dcn_pipeline_parallelism": "1",
                        "dcn_tensor_parallelism": "1",
                        "dcn_sequence_parallelism": "1",
                        "ici_fsdp_parallelism": "8",
                        "ici_data_parallelism": "1",
                        "ici_sequence_parallelism": "1",
                        "ici_tensor_parallelism": "1",
                        "ici_pipeline_parallelism": "1",
                        "remat_policy": "full",
                        "use_iota_embed": "true",
                        "scan_layers": "true",
                        "dataset_type": "synthetic",
                        "hf_path": "parquet",
                        "hf_train_files": "/home/{user-id}/cache/maxtext/data/c4/000*.parquet",
                        "tokenizer_path": "/home/{user-id}/cache/maxtext/Meta-Llama-405-B",
                        "async_checkpointing": "false",
                        "logits_dot_in_fp32": "false",
                        "megablox": "false",
                        "dtype": "bfloat16",
                        "batch_size": "128",
                        "quantization": "",
                        "quantize_kvcache": "false",
                        "kv_quant_axis": "heads_and_dkv",
                        "kv_quant_dtype": "int8",
                        "weight_dtype": "bfloat16",
                        "checkpoint_is_quantized": "false",
                        "per_device_batch_size": "2",
                        "max_target_length": "8192",
                        "skip_first_n_steps_for_profiler": "3",
    
                        "xla_flags":
                        {
                            "xla_gpu_enable_cublaslt": "True",
                                        "xla_gpu_executable_warn_stuck_timeout": "90",
                            "xla_gpu_executable_terminate_timeout": "300",
                            "xla_gpu_first_collective_call_warn_stuck_timeout_seconds": "300",
                            "xla_gpu_first_collective_call_terminate_timeout_seconds": "1200",
                            "xla_gpu_graph_level": "0",
                            "xla_gpu_autotune_level": "4",
                            "xla_gpu_enable_reduce_scatter_combine_by_dim": "false",
                            "xla_gpu_reduce_scatter_combine_threshold_bytes": "8589934592",
                            "xla_gpu_all_reduce_combine_threshold_bytes": "8589934592",
                            "xla_gpu_all_gather_combine_threshold_bytes": "137438953472",
                            "xla_gpu_enable_all_gather_combine_by_dim": "FALSE"
                        },
    
                        "base_emb_dim": "16384",
                        "base_num_query_heads": "128",
                        "base_num_kv_heads": "8",
                        "base_num_decoder_layers": "126",
                        "base_mlp_dim": "53248",
                        "head_dim": "128",
                        "mlp_activations": ["silu","linear"],
                        "vocab_size": "128256",
                        "enable_dropout": "False",
                        "logits_via_embedding": "False",
                        "normalization_layer_epsilon": "1.0e-5",
                        "rope_max_timescale": "500_000",
                        "decoder_block": "llama2",
    
                        "micro_batch_size": "2",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
    
                                        "_example_tflops_per_sec_per_gpu": "380",
                            "_example_tokens_per_sec_per_gpu": "145",
                                        "_comments": "Values change based on number of nodes",
                                        "tflops_per_sec_per_gpu": "<changeme>",
                            "tokens_per_sec_per_gpu": "<changeme>"
                        }
                    }
                }
    
            }
    
    
        }
    
    }


Parameters (``llama_3_1_405b``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the parameters in these tables to configure the training file.

``config``
""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``container_image``
     - ``rocm/jax-training:maxtext-v25.9`` 
     - Docker image used for training 
   * - ``container_name``
     - ``rocm-jax-llama3.1-405b`` 
     - Name assigned to the container instance
   * - ``distributed_training``
     - True
     - Enable multi-node distributed training   
   * - ``enable_checkpointing``
     - False
     - Whether to write training checkpoints
   * - ``Nnodes``
     - ``<changeme>``
     - Number of nodes in the distributed job 
   * - ``coordinator_ip``
     - ``<changeme>``
     - IP address of the coordinator (master) node used for rendezvous
   * - ``training_steps``
     - 30
     - Number of training steps to run
   * - ``nic_type``
     - thor2
     - NIC hardware type
   * - ``_example_nccl_ib_hca``
     - Values:
        - ``bnxt_re0``
        - ``bnxt_re1``
        - ``bnxt_re2``
        - ``bnxt_re3``
        - ``bnxt_re4``
        - ``bnxt_re5``
        - ``bnxt_re6``
        - ``bnxt_re7``
     - Comma-separated list of IB HCAs (host channel adapters) to use for NCCL/RCCL communication
   * - ``nccl_ib_hca``
     - ``<changeme>`` 
     - Comma-separated list of IB HCAs (host channel adapters) to use for NCCL/RCCL communication
   * - ``nccl_socket_ifname``
     - ``<changeme>`` 
     - Network interface name for NCCL
   * - ``gloo_socket_ifname``
     - ``<changeme>``
     - Network interface name for  Gloo fallback control plane
   * - ``_example_nccl_ib_gid_index``
     - 3
     - Example of GID index used for IB addressing (selects which GID entry on the HCA to use)
   * - ``nccl_ib_gid_index``
     - ``<changeme>``
     - GID index for IB addressing
   * - ``nccl_debug``
     - ERROR
     - NCCL debug/log level
   * - ``nccl_proto``
     - Simple
     - NCCL protocol selection 
   * - ``gpu_max_hw_queues``
     - 2
     - Max hardware queues exposed to GPU
   * - ``nccl_ib_tc``
     - 41
     - InfiniBand traffic class used by NCCL
   * - ``nccl_ib_sl``
     - 0
     - InfiniBand service level used by NCCL
   * - ``nccl_checks_disable``
     - 1
     - Disable NCCL runtime checks
   * - ``nvte_ck_bwd_v3``
     - 1
     - Custom NVTE/JAX options controlling checkpointing / backward passes conversions
   * - ``nvte_ck_v3_bf16_cvt``
     - 2
     - Custom NVTE/JAX options controlling checkpointing / backward passes and bf16 conversions
   * - ``xla_python_client_mem_fraction``
     - 0.975
     - Fraction of host memory XLA client may use
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - Path to Hugging Face token file for model access
   * - ``shm_size``
     - 256G
     - Docker shared memory size for container
   * - ``xla_gpu_executable_warn_stuck_timeout``
     - 90
     - Warning if GPU kernel runs too long
   * - ``_comments_data_cache_dir``
     - "This path should be accessible from all nodes like a common FS like NFS for distributed training"
     - A comment stating that ``data_cache_dir`` should be shared across nodes
   * - ``data_cache_dir``
     - ``/home/{user-id}/cache``
     - Path where datasets/cache will be stored
   * - ``mock_data``
     - True
     - Use synthetic or mock dataset
   * - ``log_dir``
     - ``/home/{user-id}/LOGS``
     - Directory for logs

``dataset_source/container_config``
"""""""""""""""""""""""""""""""""""

.. |br| raw:: html

    <br />

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
     - Host-to-container volume mounts
   * - ``/home/{user-id}``
     - ``/home/{user-id}``
     - The user's home directory 
   * - ``/dev/infiniband``
     - ``/dev/infiniband``
     - Exposes InfiniBand device files for RDMA networking 
   * - ``/usr/local/lib/`` |br| ``libbnxt_re-rdmav34.so``
     - ``/usr/lib/x86_64-linux`` |br| ``-gnu/libibverbs/libbnxt`` |br| ``_re-rdmav34.so.host``
     - Provides the ``bnxt_re`` RDMA provider (Broadcom NIC) inside the container 
   * - ``/lib/libibverbs.d``
     - ``/lib/libibverbs.d``
     - Mounts the RDMA verbs configuration directory — needed by libibverbs to detect RDMA devices
   * - ``/tmp/TRAINING_LOGS``
     - ``/workspace/maxtext/output`` 
     - Stores training logs outside the container 
   * - ``env_dict``
     - N/A
     - Environment variables injected into container, including JAX coordinator IP/port, ``NNODES``, and timeout/heartbeat settings 
   * - ``JAX_COORDINATOR_IP``
     - ``<changeme>``
     - The IP address of the coordinator node in the cluster — all other nodes connect here to initialize 
   * - ``JAX_COORDINATOR_PORT``
     - 12345
     - The port used by JAX for coordination 
   * - ``NNODES``
     - 1
     - Total number of nodes in the distributed job
   * - ``JAX_DISTRIBUTED_INITIAL`` |br| ``IZATION_TIMEOUT_SECONDS``
     - 1800
     - How long JAX waits for all nodes to join the distributed setup — longer is safer for large clusters 
   * - ``JAX_DISTRIBUTED_HEART`` |br| ``BEAT_TIMEOUT_SECONDS``
     - 900
     - Timeout for communication heartbeat


``model_params/single_node/llama3.1-405b/mi300x``
"""""""""""""""""""""""""""""""""""""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-405B``
     - HF model identifier used to load tokenizer 
   * - ``model_size``
     - 405
     - Model size label
   * - ``hardware``
     - GPU
     - The specified hardware in use
   * - ``attention``
     - ``cudnn_flash_te``
     - Attention implementation
   * - ``dcn_data_parallelism``
     - 1
     - Parallelism / control flags
   * - ``dcn_fsdp_parallelism``
     - -1
     - Parallelism settings for the training strategy
   * - ``dcn_pipeline_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``dcn_tensor_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``dcn_sequence_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``ici_fsdp_parallelism``
     - 8
     - Interconnect / implementation-specific 
   * - ``ici_data_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_sequence_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_tensor_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_pipeline_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``remat_policy``
     - full
     - Checkpointing/rematerialization policy
   * - ``use_iota_embed``
     - true
     - Use IOTA embeddings
   * - ``scan_layers``
     - true
     - Enable layer scanning optimization 
   * - ``dataset_type``
     - synthetic
     - Source type
   * - ``hf_path``
     - parquet
     - Paths to training data and tokenizer
   * - ``hf_train_files``
     - ``/home/{user-id}/maxtext`` |br| ``/data/c4/000*.parquet``
     - Paths to training data and tokenizer
   * - ``tokenizer_path``
     - ``/home/{user-id}/maxtext`` |br| ``Meta-Llama-405-B``
     - Paths to tokenizer
   * - ``async_checkpointing``
     - false
     - Async checkpointing
   * - ``logits_dot_in_fp32``
     - false
     - Compute logits dot in FP32
   * - ``megablox``
     - false
     - Megablox optimization
   * - ``dtype``
     - bfloat16
     - Base compute dtype 
   * - ``batch_size``
     - 128
     - Global batch size


``quantization``
""""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``quantize_kvcache``
     - false
     - KV cache quantization settings 
   * - ``kv_quant_axis``
     - ``heads_and_dkv`` 
     - KV cache quantization settings
   * - ``kv_quant_dtype``
     - int8
     - KV cache quantization settings
   * - ``weight_dtype``
     - bfloat16
     - Weight numeric type
   * - ``checkpoint_is_quantized``
     - false
     - Whether stored checkpoints are quantized
   * - ``per_device_batch_size``
     - 1
     - Micro-batch assigned per device for data parallelism 
   * - ``max_target_length``
     - 8192
     - Sequence length for generation/training
   * - ``skip_first_n_steps_for_profiler``
     - 3
     - Number of initial steps to skip when profiling

``xla_flags``
""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``xla_gpu_enable_cublaslt``
     - True
     - Use cublasLt optimized routines 
   * - ``xla_gpu_exec`` |br| ``utable_warn`` |br| ``_stuck_timeout``
     - 90
     - Warn if GPU kernel runs too long
   * - ``xla_gpu_exec`` |br| ``utable_term`` |br| ``inate_timeout``
     - 300
     - Sets how long (in seconds) XLA waits before terminating a GPU executable that appears hung during execution  
   * - ``xla_gpu_first_`` |br| ``collective_call`` |br| ``_warn_stuck`` |br| ``_timeout_seconds``
     - 300
     - Time before XLA logs a warning if a collective GPU operation hasn't completed yet
   * - ``xla_gpu_first_`` |br| ``collective_call`` |br| ``_terminate_ti`` |br| ``meout_seconds``
     - 1200
     - Time (in seconds) before XLA terminates the first collective GPU call if it's still not finished
   * - ``xla_gpu_graph_level``
     - 0
     - Graph compilation level 
   * - ``xla_gpu_autotune_level``
     - 4
     - Autotune aggressiveness
   * - ``xla_gpu_enable_reduce`` |br| ``_scatter_combine_by_dim``
     - false
     - Combine reduce-scatter by dim 
   * - ``xla_gpu_reduce_scatter`` |br| ``_combine_threshold_bytes``
     - 8589934592 
     - Byte threshold to combine
   * - ``xla_gpu_all_reduce_`` |br| ``combine_threshold_bytes``
     - 8589934592
     - Combine threshold for all-reduce 
   * - ``xla_gpu_all_gather_`` |br| ``combine_threshold_bytes``
     - 137438953472 
     - Threshold for all-gather 
   * - ``xla_gpu_enable_all_`` |br| ``gather_combine_by_dim``
     - FALSE
     - Enable all-gather combine by dimension 
   * - ``base_emb_dim``
     - 16384
     - Model architecture sizes 
   * - ``base_num_query_heads``
     - 128
     - Model architecture sizes 
   * - ``base_num_kv_heads``
     - 8
     - Model architecture sizes  
   * - ``base_num_decoder_layers1``
     - 126
     - Model architecture sizes 
   * - ``base_mlp_dim``
     - 53248
     - Model architecture sizes 
   * - ``head_dim1``
     - 28
     - Model architecture sizes 
   * - ``mlp_activations``
     - Values:
        - ``"silu"``
        - ``"linear"``
     - Activation functions used 
   * - ``vocab_size``
     - 128526
     - Token vocabulary size 
   * - ``enable_dropout``
     - False
     - Whether dropout is enabled for training
   * - ``logits_via_embedding``
     - False
     - Uses the same embedding matrix for output logits
   * - ``normalization_layer_epsilon``
     - 1.0e-5 
     - Numeric epsilon for layernorm stability 
   * - ``rope_max_timescale``
     - 500_000 
     - Rotary positional encoding max timescale 
   * - ``decoder_block``
     - llama2 
     - Decoder block type 
   * - ``micro_batch_size``
     - 2
     - Micro-batch size per step 
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode 
   * - ``sequence_length``
     - 8192
     - Sequence length
   * - ``tensor_parallelism``
     - 1
     - Parallelism degrees
   * - ``pipeline_parallelism``
     - 1
     - Parallelism degrees
   * - ``recompute``
     - 0
     - Recompute flags
   * - ``fsdp``
     - 0
     - FSDP

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {

                      "_example_tflops_per_sec_per_gpu": "380",
          "_example_tokens_per_sec_per_gpu": "145",
                      "_comments": "Values change based on number of nodes",
                      "tflops_per_sec_per_gpu": "<changeme>",
          "tokens_per_sec_per_gpu": "<changeme>"
      }


``mi300x_distributed_llama3_1_70b.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the ``mi300x_distributed_llama3_1_70b.json`` configuration file:

.. dropdown:: ``mi300x_distributed_llama3_1_70b.json``

  .. code:: json

    {
    
        "config":
        {
            "container_image": "rocm/jax-training:maxtext-v25.9",
            "container_name": "rocm-jax-llama3.1-70b",
            "distributed_training": "True",
            "enable_checkpointing": "False",
            "nnodes": "<changeme>",
            "coordinator_ip": "<changeme>",
            "training_steps": "30",
            "nic_type": "thor2",
            "_example_nccl_ib_hca_list": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8",
            "nccl_ib_hca_list": "<changeme>",
            "_example_nccl_ib_hca": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8",
            "nccl_ib_hca": "<changeme>",
            "nccl_socket_ifname": "<changeme>",
            "gloo_socket_ifname": "<changeme>",
            "_example_nccl_ib_gid_index": "3",
            "nccl_ib_gid_index": "<changeme>",
            "nccl_debug": "ERROR",
            "nccl_proto": "Simple",
            "gpu_max_hw_queues": "2",
            "nccl_ib_tc": "41",
            "nccl_ib_sl": "0",
            "nccl_checks_disable": "1",
            "nvte_ck_bwd_v3": "1",
            "nvte_ck_v3_bf16_cvt": "2",
            "xla_python_client_mem_fraction": "0.975",
                "xla_gpu_executable_warn_stuck_timeout": "90",
            "hf_token_file": "/home/{user-id}/.hf_token",
            "shm_size": "256G",
            "_comments_data_cache_dir": "This path should be accessible from all nodes like a common FS like NFS for distributed training",
            "data_cache_dir": "/home/{user-id}/cache",
            "mock_data": "True",
            "log_dir": "/home/{user-id}/LOGS",
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
                "/tmp/TRAINING_LOGS": "/workspace/maxtext/output"
                },
                "env_dict":
                {
                    "JAX_COORDINATOR_IP": "<changeme>",
                    "JAX_COORDINATOR_PORT": "1234",
                    "NNODES": "1",
                    "JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS": "1800",
                            "JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT_SECONDS": "900"
                }
            }
        },
        "model_params":
        {
            "multi_node":
            {
                "llama3.1-70b":
                {
                    "mi300x":
                    {
                        "tokenizer_model": "NousResearch/Meta-Llama-3-70B",
                        "model_size": "70",
                                "hardware": "gpu",
                        "attention": "cudnn_flash_te",
                        "dcn_data_parallelism": "1",
                        "dcn_fsdp_parallelism": "-1",
                        "dcn_pipeline_parallelism": "1",
                        "dcn_tensor_parallelism": "1",
                        "dcn_sequence_parallelism": "1",
                        "ici_fsdp_parallelism": "8",
                        "ici_data_parallelism": "1",
                        "ici_sequence_parallelism": "1",
                        "ici_tensor_parallelism": "1",
                        "ici_pipeline_parallelism": "1",
                        "remat_policy": "full",
                        "use_iota_embed": "true",
                        "scan_layers": "true",
                        "dataset_type": "synthetic",
                        "hf_path": "parquet",
                        "hf_train_files": "/home/{user-id}/cache/maxtext/data/c4/000*.parquet",
                        "tokenizer_path": "/home/{user-id}/cache/maxtext/Meta-Llama-70-B",
                        "async_checkpointing": "false",
                        "logits_dot_in_fp32": "false",
                        "megablox": "false",
                        "dtype": "bfloat16",
                        "batch_size": "128",
                        "quantization": "",
                        "quantize_kvcache": "false",
                        "kv_quant_axis": "heads_and_dkv",
                        "kv_quant_dtype": "int8",
                        "weight_dtype": "bfloat16",
                        "checkpoint_is_quantized": "false",
                        "per_device_batch_size": "2",
                        "max_target_length": "8192",
                        "skip_first_n_steps_for_profiler": "3",
    
                        "xla_flags":
                        {
                            "xla_gpu_enable_cublaslt": "True",
                                        "xla_gpu_executable_warn_stuck_timeout": "90",
                            "xla_gpu_executable_terminate_timeout": "300",
                            "xla_gpu_first_collective_call_warn_stuck_timeout_seconds": "300",
                            "xla_gpu_first_collective_call_terminate_timeout_seconds": "1200",
                            "xla_gpu_graph_level": "0",
                            "xla_gpu_autotune_level": "4",
                            "xla_gpu_enable_reduce_scatter_combine_by_dim": "false",
                            "xla_gpu_reduce_scatter_combine_threshold_bytes": "8589934592",
                            "xla_gpu_all_reduce_combine_threshold_bytes": "8589934592",
                            "xla_gpu_all_gather_combine_threshold_bytes": "137438953472",
                            "xla_gpu_enable_all_gather_combine_by_dim": "FALSE"
                        },
    
                        "base_emb_dim": "16384",
                        "base_num_query_heads": "128",
                        "base_num_kv_heads": "8",
                        "base_num_decoder_layers": "126",
                        "base_mlp_dim": "53248",
                        "head_dim": "128",
                        "mlp_activations": ["silu","linear"],
                        "vocab_size": "128256",
                        "enable_dropout": "False",
                        "logits_via_embedding": "False",
                        "normalization_layer_epsilon": "1.0e-5",
                        "rope_max_timescale": "500_000",
                        "decoder_block": "llama2",
    
                        "micro_batch_size": "2",
                        "precision": "TE_FP8",
                        "sequence_length": "8192",
                        "tensor_parallelism": "1",
                        "pipeline_parallelism": "1",
                        "recompute": "0",
                        "fsdp": "0",
                        "result_dict":
                        {
    
                                        "_example_tflops_per_sec_per_gpu": "380",
                            "_example_tokens_per_sec_per_gpu": "800",
                                        "_comments": "Values change based on number of nodes",
                                        "tflops_per_sec_per_gpu": "<changeme>",
                            "tokens_per_sec_per_gpu": "<changeme>"
                        }
                    }
                }
    
            }
    
    
        }
    
    }
 
Parameters (``llama3_1_70b``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the parameters in these tables to configure the training file.

``config``
""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``container_image``
     - ``rocm/jax-training:maxtext-v25.9`` 
     - Docker image used for training 
   * - ``container_name``
     - ``rocm-jax-llama3.1-70b`` 
     - Name assigned to the container instance
   * - ``distributed_training``
     - True
     - Enable multi-node distributed training   
   * - ``enable_checkpointing``
     - False
     - Whether to write training checkpoints
   * - ``Nnodes``
     - ``<changeme>``
     - Number of nodes in the distributed job 
   * - ``coordinator_ip``
     - ``<changeme>``
     - IP address of the coordinator (master) node used for rendezvous
   * - ``training_steps``
     - 30
     - Number of training steps to run
   * - ``nic_type``
     - thor2
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
     - Comma-separated list of IB HCAs (host channel adapters) to use for NCCL/RCCL communication
   * - ``nccl_ib_hca_list``
     - ``<changeme>`` 
     - Comma-separated list of InfiniBand HCA device names to use for NCCL/communication (multi-rail support)
   * - ``_example_nccl_ib_hca``
     - Values:
        - ``bnxt_re0``
        - ``bnxt_re1``
        - ``bnxt_re2``
        - ``bnxt_re3``
        - ``bnxt_re4``
        - ``bnxt_re5``
        - ``bnxt_re6``
        - ``bnxt_re7``
     - Comma-separated list of IB HCAs (host channel adapters) to use for NCCL/RCCL communication
   * - ``nccl_ib_hca``
     - ``<changeme>`` 
     - Comma-separated list of IB HCAs (host channel adapters) to use for NCCL/RCCL communication
   * - ``nccl_socket_ifname``
     - ``<changeme>`` 
     - Network interface name for NCCL
   * - ``gloo_socket_ifname``
     - ``<changeme>``
     - Network interface name for  Gloo fallback control plane
   * - ``nccl_ib_gid_index``
     - ``<changeme>``
     - GID index for IB addressing
   * - ``nccl_debug``
     - ERROR
     - NCCL debug/log level
   * - ``nccl_proto``
     - Simple
     - NCCL protocol selection 
   * - ``gpu_max_hw_queues``
     - 2
     - Max hardware queues exposed to GPU
   * - ``nccl_ib_tc``
     - 41
     - InfiniBand traffic class used by NCCL
   * - ``nccl_ib_sl``
     - 0
     - InfiniBand service level used by NCCL
   * - ``nccl_checks_disable``
     - 1
     - Disable NCCL runtime checks
   * - ``nvte_ck_bwd_v3``
     - 1
     - Custom NVTE/JAX options controlling checkpointing / backward passes conversions
   * - ``nvte_ck_v3_bf16_cvt``
     - 2
     - Custom NVTE/JAX options controlling checkpointing / backward passes and bf16 conversions
   * - ``xla_python_client_mem_fraction``
     - 0.975
     - Fraction of host memory XLA client may use
   * - ``xla_gpu_executable_warn_stuck_timeout``
     - 90
     - Warning if GPU kernel runs too long
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - Path to Hugging Face token file for model access
   * - ``shm_size``
     - 256G
     - Docker shared memory size for container
   * - ``_comments_data_cache_dir``
     - "This path should be accessible from all nodes like a common FS like NFS for distributed training"
     - A comment stating that ``data_cache_dir`` should be shared across nodes
   * - ``data_cache_dir``
     - ``/home/{user-id}/cache``
     - Path where datasets/cache will be stored
   * - ``mock_data``
     - True
     - Use synthetic or mock dataset
   * - ``log_dir``
     - ``/home/{user-id}/LOGS``
     - Directory for logs

``container_config``
""""""""""""""""""""

.. |br| raw:: html

    <br />

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
     - Host-to-container volume mounts
   * - ``/home/{user-id}``
     - ``/home/{user-id}``
     - The user's home directory 
   * - ``/dev/infiniband``
     - ``/dev/infiniband``
     - Exposes InfiniBand device files for RDMA networking 
   * - ``/usr/local/lib/`` |br| ``libbnxt_re-rdmav34.so``
     - ``/usr/lib/x86_64-linux`` |br| ``-gnu/libibverbs/libbnxt`` |br| ``_re-rdmav34.so.host``
     - Provides the ``bnxt_re`` RDMA provider (Broadcom NIC) inside the container 
   * - ``/lib/libibverbs.d``
     - ``/lib/libibverbs.d``
     - Mounts the RDMA verbs configuration directory — needed by libibverbs to detect RDMA devices
   * - ``/tmp/TRAINING_LOGS``
     - ``/workspace/maxtext/output`` 
     - Stores training logs outside the container 
   * - ``env_dict``
     - N/A
     - Environment variables injected into container, including JAX coordinator IP/port, ``NNODES``, and timeout/heartbeat settings 
   * - ``JAX_COORDINATOR_IP``
     - ``<changeme>``
     - The IP address of the coordinator node in the cluster — all other nodes connect here to initialize 
   * - ``JAX_COORDINATOR_PORT``
     - 12345
     - The port used by JAX for coordination 
   * - ``NNODES``
     - 1
     - Total number of nodes in the distributed job
   * - ``JAX_DISTRIBUTED_INITIAL`` |br| ``IZATION_TIMEOUT_SECONDS``
     - 1800
     - How long JAX waits for all nodes to join the distributed setup — longer is safer for large clusters 
   * - ``JAX_DISTRIBUTED_HEART`` |br| ``BEAT_TIMEOUT_SECONDS``
     - 900
     - Timeout for communication heartbeat


``model_params/single_node/llama3.1-70b/mi300x``
"""""""""""""""""""""""""""""""""""""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``tokenizer_model``
     - ``NousResearch/Meta-Llama-3-70B``
     - HF model identifier used to load tokenizer 
   * - ``model_size``
     - 70
     - Model size label
   * - ``attention``
     - ``cudnn_flash_te``
     - Attention implementation
   * - ``dcn_data_parallelism``
     - 1
     - Parallelism / control flags
   * - ``dcn_fsdp_parallelism``
     - -1
     - Parallelism settings for the training strategy
   * - ``dcn_pipeline_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``dcn_tensor_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``dcn_sequence_parallelism``
     - 1
     - Parallelism settings for the training strategy 
   * - ``ici_fsdp_parallelism``
     - 8
     - Interconnect / implementation-specific 
   * - ``ici_data_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_sequence_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_tensor_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``ici_pipeline_parallelism``
     - 1
     - Interconnect / implementation-specific 
   * - ``remat_policy``
     - full
     - Checkpointing/rematerialization policy
   * - ``use_iota_embed``
     - true
     - Use IOTA embeddings
   * - ``scan_layers``
     - true
     - Enable layer scanning optimization 
   * - ``dataset_type``
     - synthetic
     - Source type
   * - ``hf_path``
     - parquet
     - Paths to training data and tokenizer
   * - ``hf_train_files``
     - ``/home/{user-id}/maxtext`` |br| ``/data/c4/000*.parquet``
     - Paths to training data and tokenizer
   * - ``tokenizer_path``
     - ``/home/{user-id}/maxtext`` |br| ``Meta-Llama-70-B``
     - Paths to tokenizer
   * - ``async_checkpointing``
     - false
     - Async checkpointing
   * - ``logits_dot_in_fp32``
     - false
     - Compute logits dot in FP32
   * - ``megablox``
     - false
     - Megablox optimization
   * - ``dtype``
     - bfloat16
     - Base compute dtype 
   * - ``batch_size``
     - 128
     - Global batch size


``quantization``
""""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``quantize_kvcache``
     - false
     - KV cache quantization settings 
   * - ``kv_quant_axis``
     - ``heads_and_dkv`` 
     - KV cache quantization settings
   * - ``kv_quant_dtype``
     - int8
     - KV cache quantization settings
   * - ``weight_dtype``
     - bfloat16
     - Weight numeric type
   * - ``checkpoint_is_quantized``
     - false
     - Whether stored checkpoints are quantized
   * - ``per_device_batch_size``
     - 2
     - Micro-batch assigned per device for data parallelism 
   * - ``max_target_length``
     - 8192
     - Sequence length for generation/training
   * - ``skip_first_n_steps_for_profiler``
     - 3
     - Number of initial steps to skip when profiling

``xla_flags``
""""""""""""""

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``xla_gpu_enable_cublaslt``
     - True
     - Use cublasLt optimized routines 
   * - ``xla_gpu_exec`` |br| ``utable_warn`` |br| ``_stuck_timeout``
     - 90
     - Warn if GPU kernel runs too long
   * - ``xla_gpu_graph_level``
     - 0
     - Graph compilation level 
   * - ``xla_gpu_autotune_level``
     - 4
     - Autotune aggressiveness
   * - ``xla_gpu_enable_reduce`` |br| ``_scatter_combine_by_dim``
     - false
     - Combine reduce-scatter by dim 
   * - ``xla_gpu_reduce_scatter`` |br| ``_combine_threshold_bytes``
     - 8589934592 
     - Byte threshold to combine
   * - ``xla_gpu_all_reduce_`` |br| ``combine_threshold_bytes``
     - 8589934592
     - Combine threshold for all-reduce 
   * - ``xla_gpu_all_gather_`` |br| ``combine_threshold_bytes``
     - 137438953472 
     - Threshold for all-gather 
   * - ``xla_gpu_enable_all_`` |br| ``gather_combine_by_dim``
     - FALSE
     - Enable all-gather combine by dimension 
   * - ``base_emb_dim``
     - 16384
     - Model architecture sizes 
   * - ``base_num_query_heads``
     - 128
     - Model architecture sizes 
   * - ``base_num_kv_heads``
     - 8
     - Model architecture sizes  
   * - ``base_num_decoder_layers1``
     - 126
     - Model architecture sizes 
   * - ``base_mlp_dim``
     - 53248
     - Model architecture sizes 
   * - ``head_dim1``
     - 28
     - Model architecture sizes 
   * - ``mlp_activations``
     - Values:
        - ``"silu"``
        - ``"linear"``
     - Activation functions used 
   * - ``vocab_size``
     - 128526
     - Token vocabulary size 
   * - ``enable_dropout``
     - False
     - Whether dropout is enabled for training
   * - ``logits_via_embedding``
     - False
     - Uses the same embedding matrix for output logits
   * - ``normalization_layer_epsilon``
     - 1.0e-5 
     - Numeric epsilon for layernorm stability 
   * - ``rope_max_timescale``
     - 500_000 
     - Rotary positional encoding max timescale 
   * - ``decoder_block``
     - llama2 
     - Decoder block type 
   * - ``micro_batch_size``
     - 2
     - Micro-batch size per step 
   * - ``precision``
     - TE_FP8 
     - Numeric precision mode 
   * - ``sequence_length``
     - 8192
     - Sequence length
   * - ``tensor_parallelism``
     - 1
     - Parallelism degrees
   * - ``pipeline_parallelism``
     - 1
     - Parallelism degrees
   * - ``recompute``
     - 0
     - Recompute flags
   * - ``fsdp``
     - 0
     - FSDP

This section also contains the ``result_dict`` parameter. It describes the expected/target metrics used by tests to verify and run correctness and performance:

.. dropdown:: result_dict

  .. code:: json

      "result_dict":
      {

                      "_example_tflops_per_sec_per_gpu": "380",
          "_example_tokens_per_sec_per_gpu": "800",
                      "_comments": "Values change based on number of nodes",
                      "tflops_per_sec_per_gpu": "<changeme>",
          "tokens_per_sec_per_gpu": "<changeme>"
      }
