"""Config schema for the Megatron-LM training suites.

Defaults mirror the ``tdict.setdefault`` / ``pdict.setdefault`` blocks in
``cvs/lib/megatron_training_lib.py``, which is where the runtime defaults for
this suite actually live. Several of them disagree with every shipped sample.

``model_params`` is deliberately typed as open mappings: the model name is
chosen by the test module and the GPU variant is keyed by live hardware
detection, so neither is a fixed set of field names.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MegatronContainerConfig(BaseModel):
    """Container launch settings for the per-node training container."""

    model_config = ConfigDict(extra="allow")

    device_list: List[str] = Field(
        description="Host device nodes passed through to the container.",
        examples=[["/dev/dri", "/dev/kfd", "/dev/infiniband/rdma_cm"]],
    )
    volume_dict: Dict[str, str] = Field(
        description="Host-to-container bind mounts, keyed by host path.",
        examples=[{"/home/{user-id}": "/home/{user-id}"}],
    )


class MegatronResultThresholds(BaseModel):
    """Per-metric thresholds checked after training completes.

    Every entry is a minimum: the test fails when the measured value is below
    the configured one. Metrics absent from this block are parsed but not
    checked. Note that elapsed_time_per_iteration is a latency, so a floor
    means a faster-than-expected iteration fails.

    Thresholds are node-count dependent; the shipped distributed sample was
    written for 4 nodes and must be retuned for other cluster sizes.
    """

    model_config = ConfigDict(extra="allow")

    throughput_per_gpu: Optional[str] = Field(
        default=None,
        description="Minimum acceptable throughput per GPU in TFLOP/s.",
        examples=["610.0"],
    )
    tokens_per_gpu: Optional[str] = Field(
        default=None,
        description="Minimum acceptable tokens per GPU per second.",
        examples=["12000.0"],
    )
    elapsed_time_per_iteration: Optional[str] = Field(
        default=None,
        description="Minimum acceptable elapsed time per iteration in milliseconds.",
        examples=["12000.0"],
    )
    mem_usage: Optional[str] = Field(
        default=None,
        description="Minimum acceptable reported memory usage. Parsed from the training log but unused by any sample.",
        examples=["100.0"],
    )


class MegatronModelVariant(BaseModel):
    """Training parameters for one model on one GPU generation."""

    model_config = ConfigDict(extra="allow")

    tokenizer_model: str = Field(
        default="meta-llama/Llama-3.1-70B",
        description=(
            "Hugging Face model id for the tokenizer. Also selects the training script, by case-insensitive "
            "regex match against the keys of config.training_scripts."
        ),
        examples=["NousResearch/Meta-Llama-3-8B"],
    )
    model_size: str = Field(
        default="70",
        description="Parameter count in billions, exported as MODEL_SIZE.",
        examples=["8"],
    )
    sequence_length: str = Field(
        default="8192",
        description="Context length, exported as SEQ_LENGTH.",
    )
    batch_size: str = Field(
        default="128",
        description="Global batch size, exported as BS.",
    )
    micro_batch_size: str = Field(
        default="2",
        description="Per-device micro-batch size, exported as MBS.",
    )
    fsdp: str = Field(
        default="0",
        description="Enable fully-sharded data parallel, exported as FSDP.",
        examples=["0", "1"],
    )
    tensor_parallelism: str = Field(
        default="1",
        description="Tensor-parallel degree, exported as TP.",
    )
    pipeline_parallelism: str = Field(
        default="1",
        description="Pipeline-parallel stage count, exported as PP.",
    )
    recompute: str = Field(
        default="0",
        description="Enable activation recomputation, exported as RECOMPUTE.",
        examples=["0", "1"],
    )
    precision: str = Field(
        default="TE_FP8",
        description=(
            "Transformer-Engine precision mode. Only 'TE_BF16' and 'TE_F16' are recognised; every other value, "
            "including the misspelling 'TE_FP16', silently selects FP8."
        ),
        examples=["TE_FP8", "TE_BF16", "TE_F16"],
    )
    result_dict: MegatronResultThresholds = Field(
        description="Minimum acceptable performance results for this model and GPU combination.",
    )


class MegatronConfigFile(BaseModel):
    """Schema for the config section of a Megatron training config file."""

    model_config = ConfigDict(extra="allow")

    container_image: str = Field(
        default="rocm/megatron-lm:v25.5_py312",
        description="Docker image the Megatron-LM job runs in, also exported into the job environment as IMAGE.",
        examples=["rocm/megatron-lm:v25.5_py310"],
    )
    container_name: str = Field(
        default="megatron_llama3.1_8b",
        description="Name of the per-node container. Every docker exec targets it and cleanup kills it by name.",
        examples=["megatron_llama3.1_310"],
    )
    container_config: MegatronContainerConfig = Field(description="Container launch settings.")
    training_iterations: str = Field(
        description="Number of training steps, exported as TOTAL_ITERS. Required; there is no usable default.",
        examples=["30"],
    )
    hf_token_file: str = Field(
        description="Path to a file holding the Hugging Face token, whose contents are exported as HF_TOKEN.",
        examples=["/home/{user-id}/.hf_token"],
    )
    nnodes: str = Field(
        default="2",
        description="Node count, exported as NNODES and used to size the per-rank wrapper-script loop.",
        examples=["4"],
    )
    master_address: str = Field(
        default="127.0.0.1",
        description="Rendezvous node address, exported as MASTER_ADDR for distributed runs.",
        examples=["X.X.X.X", "localhost"],
    )
    data_cache_dir: str = Field(
        default="~/cache",
        description=(
            "Megatron DATA_CACHE_PATH. For distributed training this must be on a filesystem shared by every "
            "node, such as NFS."
        ),
        examples=["/home/{user-id}/cache"],
    )
    log_dir: str = Field(
        default="~/LOGS",
        description="Host log root; per-node training logs are written beneath it and it is exported as LOG_DIR.",
        examples=["/home/{user-id}/LOG_DIR"],
    )
    scripts_dir: str = Field(
        default="~/SCRIPTS",
        description="Per-node directory for the generated wrapper scripts. Removed and recreated on every run.",
        examples=["/home/{user-id}/SCRIPTS"],
    )
    megatron_root: str = Field(
        default="/workspace/Megatron-LM",
        description="Megatron-LM checkout root inside the container, used as the cd target and as the prefix for training_scripts.",
    )
    training_scripts: Dict[str, str] = Field(
        default_factory=lambda: {
            "llama-3": "examples/llama/train_llama3.sh",
            "llama-2": "examples/llama/train_llama2.sh",
        },
        description=(
            "Map of tokenizer-family pattern to script path relative to megatron_root. Matched case-insensitively "
            "against tokenizer_model, first hit wins."
        ),
        examples=[{"llama-3": "examples/llama/train_llama3.sh"}],
    )
    nic_type: str = Field(
        default="thor2",
        description="Backend NIC family. Matching broadcom or thor triggers the in-container libbnxt_re workaround.",
        examples=["ainic", "thor2", "cx7"],
    )
    hca_id_pattern: str = Field(
        default="bnxt_|rocep",
        description=(
            "Pipe-separated literal NIC name prefixes matched against the hca_id lines of ibv_devinfo to confirm "
            "the RDMA driver copy succeeded. Each segment is treated as a literal, not a regex."
        ),
        examples=["bnxt_|rocep|mlx5_"],
    )
    nccl_ib_hca_list: str = Field(
        default="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7",
        description="Comma-separated IB HCA device list, exported as both NCCL_IB_HCA_LIST and NCCL_IB_HCA.",
    )
    nccl_socket_ifname: str = Field(
        default="ensf1np1",
        description="Host interface used by NCCL control channels, exported as NCCL_SOCKET_IFNAME.",
        examples=["ens51f1np1"],
    )
    gloo_socket_ifname: str = Field(
        default="ensf1np1",
        description="Host interface used by Gloo control channels, exported as GLOO_SOCKET_IFNAME.",
        examples=["ens51f1np1"],
    )
    nccl_ib_gid_index: str = Field(
        default="3",
        description="RoCE GID index, exported as NCCL_IB_GID_INDEX. Forced to 3 for Broadcom and Thor NICs.",
    )
    nccl_debug: str = Field(
        default="ERROR",
        description="NCCL log verbosity, exported as NCCL_DEBUG for distributed runs.",
        examples=["ERROR", "INFO"],
    )
    nccl_ib_hca: str = Field(
        default="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7",
        description="Read but never used; NCCL_IB_HCA is exported from nccl_ib_hca_list instead.",
        examples=["bnxt_re0,bnxt_re1"],
    )
    rocm_dir: str = Field(
        default="",
        description="Explicit ROCm install path. Leave empty to auto-detect from /opt/rocm/core-* then /opt/rocm.",
        examples=["/opt/rocm"],
    )
    verify_network_errors: str = Field(
        default="False",
        description="Gate for the post-run RDMA and ethtool error-counter comparison. Set to the string 'True' to enable.",
        examples=["True"],
    )
    shm_size: Optional[str] = Field(
        default=None,
        description="Not read by any code path; the tests pass a hardcoded 128G shared-memory size.",
        examples=["128G"],
    )
    mock_data: Optional[str] = Field(
        default=None,
        description="Not read by any code path; MOCK_DATA=1 is exported unconditionally.",
        examples=["True"],
    )
    dataset_source: Optional[dict] = Field(
        default=None,
        description="Not read by any code path.",
        examples=[{}],
    )
    distributed_training: Optional[str] = Field(
        default=None,
        description=(
            "Not read by any code path. Whether a run is distributed is fixed by the test module, not by this key."
        ),
        examples=["True"],
    )


class MegatronSingleNodeModelParams(BaseModel):
    """Schema for the model_params section of a single-node Megatron config file.

    Maps model name to GPU variant to training parameters. Both levels are
    open: the model name is a string literal chosen by the test module, and
    the GPU variant is keyed by the GPU detected at runtime.
    """

    model_config = ConfigDict(extra="allow")

    single_node: Optional[Dict[str, Dict[str, MegatronModelVariant]]] = Field(
        default=None,
        description="Variants used for single-node runs, keyed by model name then detected GPU.",
    )


class MegatronMultiNodeModelParams(BaseModel):
    """Schema for the model_params section of a distributed Megatron config file.

    Maps model name to GPU variant to training parameters. Both levels are
    open: the model name is a string literal chosen by the test module, and
    the GPU variant is keyed by the GPU detected at runtime.
    """

    model_config = ConfigDict(extra="allow")

    multi_node: Optional[Dict[str, Dict[str, MegatronModelVariant]]] = Field(
        default=None,
        description="Variants used for distributed runs, keyed by model name then detected GPU.",
    )
