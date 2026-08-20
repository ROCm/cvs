"""Config schema for the RCCL suites (rccl_perf, rccl_regression, rccl_pairwise).

Defaults mirror the ``.get(key, default)`` call sites in ``cvs/lib/rccl_lib.py``
and ``cvs/tests/rccl/``, which are what actually takes effect when a key is
absent. Several differ from the values in the shipped sample config.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RcclMpiParams(BaseModel):
    """MPI launch settings for the mpirun that drives rccl-tests."""

    model_config = ConfigDict(extra="allow")

    no_of_nodes: str = Field(
        default="2",
        description="Number of cluster nodes to launch MPI ranks on. Multiplied by no_of_local_ranks to size the job.",
    )
    no_of_local_ranks: str = Field(
        default="8",
        description="MPI ranks per node, normally one per GPU.",
    )
    mpi_pml: str = Field(
        default="auto",
        description=(
            "MPI Point-to-Point Messaging Layer: 'auto' (auto-detect UCX support), 'ucx' (force UCX), "
            "or 'ob1' (force OpenIB/TCP fallback). This is an MPI launch option rather than an env-script setting, "
            "which is why it lives in the config rather than the environment script."
        ),
        examples=["auto", "ucx", "ob1"],
    )
    mpi_dir: str = Field(
        default="/usr/local/bin",
        description="Directory holding the mpirun binary on every node.",
        examples=["/home/{user-id}/openmpi/bin"],
    )
    mpi_oob_port: str = Field(
        default="eth0",
        description="Interface used for MPI out-of-band communication, passed as the oob_tcp_if_include hint.",
    )
    net_dev_list: str = Field(
        default="",
        description=(
            "Optional UCX network device list, consulted only when the PML resolves to UCX. "
            "Absent from the sample config; leave empty to let UCX choose."
        ),
        examples=["mlx5_0:1"],
    )
    ucx_tls: str = Field(
        default="tcp",
        description=(
            "Optional UCX transport list, consulted only when the PML resolves to UCX. Absent from the sample config."
        ),
        examples=["tcp", "rc,sm"],
    )


class RcclTestParams(BaseModel):
    """Arguments passed through to the rccl-tests binaries."""

    model_config = ConfigDict(extra="allow")

    rccl_collective: List[str] = Field(
        default_factory=list,
        description="rccl-tests binaries to run, one test per collective.",
        examples=[["all_reduce_perf", "all_gather_perf"]],
    )
    rccl_tests_dir: str = Field(
        default="/usr/local/rccl-tests/build",
        description="Directory containing the built rccl-tests binaries on every node.",
        examples=["/home/{user-id}/rccl-tests/build"],
    )
    start_msg_size: str = Field(
        default="1024",
        description="Smallest message size to sweep, passed as -b.",
    )
    end_msg_size: str = Field(
        default="16g",
        description="Largest message size to sweep, passed as -e.",
    )
    step_function: str = Field(
        default="2",
        description="Multiplication factor between successive message sizes, passed as -f.",
    )
    threads_per_gpu: str = Field(
        default="1",
        description="Threads per GPU, passed as -t.",
    )
    warmup_iterations: str = Field(
        default="10",
        description="Warm-up iterations run before timing begins, passed as -w.",
    )
    no_of_iterations: str = Field(
        default="20",
        description="Timed iterations per message size, passed as -n.",
    )
    no_of_cycles: str = Field(
        default="1",
        description="Number of times the whole sweep is repeated, passed as -N.",
    )
    check_iteration_count: str = Field(
        default="1",
        description="Correctness-check iteration count, passed as -c. Set to 0 to skip result validation.",
    )
    data_types: List[str] = Field(
        default_factory=lambda: ["float"],
        description=(
            "Data types to sweep. The configured rccl_result_file name is suffixed with each type, so 'float' "
            "produces rccl_result_file_float.json on the head node."
        ),
        examples=[["float"]],
    )
    rccl_timeout: Optional[str] = Field(
        default=None,
        description="rccl-tests internal timer in seconds, passed as -T. Omit to leave the flag unset.",
        examples=["1800"],
    )
    output_algo_proto_channels: bool = Field(
        default=False,
        description="Enable rccl-tests' -A 1 algorithm/protocol/channels diagnostic output.",
    )


class RcclCvsParams(BaseModel):
    """CVS-side verification and bookkeeping, applied around the rccl-tests run."""

    model_config = ConfigDict(extra="allow")

    nic_model: str = Field(
        default="ainic",
        description="NIC family used to select model-specific validations.",
        examples=["thor", "ainic", "connectx"],
    )
    cluster_snapshot_debug: str = Field(
        default="False",
        description=(
            "Capture and diff a cluster metrics snapshot either side of each test. Matched case-insensitively "
            "against 'True', so this is a string rather than a JSON boolean."
        ),
        examples=["False", "True"],
    )
    verify_bus_bw: str = Field(
        default="False",
        description=(
            "Compare measured bus bandwidth against the thresholds in results. Matched case-insensitively "
            "against 'True'. Has no effect unless results is populated."
        ),
        examples=["False", "True"],
    )
    verify_bw_dip: str = Field(
        default="True",
        description=(
            "Flag bandwidth dips across the message-size sweep. Runs with or without expected results, since "
            "the dip analysis is relative to the rest of the run."
        ),
        examples=["True", "False"],
    )
    verify_lat_dip: str = Field(
        default="True",
        description="Flag latency spikes across the message-size sweep, using the same relative analysis as verify_bw_dip.",
        examples=["True", "False"],
    )
    rccl_result_file: str = Field(
        default="/tmp/rccl_result_output.json",
        description="Path on the head node where rccl-tests writes its JSON results, suffixed with each entry of data_types.",
        examples=["/home/{user-id}/rccl_result_file.json"],
    )
    cvs_exec_timeout: str = Field(
        default="2400",
        description="CVS-side outer cap, in seconds, on the SSH-exec call wrapping mpirun.",
    )
    pairwise_min_bw: str = Field(
        default="0",
        description=(
            "Minimum bus bandwidth in GB/s required for rccl_pairwise Phase 2 incremental admission. Phase 1 "
            "pairwise never gates on bandwidth. Set to 0 to skip the check."
        ),
        examples=["300"],
    )
    pairwise_results_file: str = Field(
        default="/tmp/rccl_pairwise_results.json",
        description=(
            "Local JSON file where rccl_pairwise.py persists the Phase 1 pass/fail lists and the Phase 2 final "
            "valid/excluded host lists."
        ),
    )
    results: Dict[str, Dict[str, Dict[str, str]]] = Field(
        default_factory=dict,
        description=(
            "Expected bus bandwidth per collective, keyed by collective name then metric then message size in "
            "bytes. Only consulted when verify_bus_bw is True. Thresholds are cluster-size dependent, so the "
            "values in the sample config apply to a 2-node cluster and must be retuned for larger ones. Note "
            "that this is read from cvs_params, while the sample config places a results block at the rccl "
            "root where nothing reads it."
        ),
        examples=[{"all_reduce_perf": {"bus_bw": {"8589934592": "330.00"}}}],
    )


class RcclConfigFile(BaseModel):
    """Schema for the rccl section of the RCCL test config file."""

    model_config = ConfigDict(extra="allow")

    env_source_script: str = Field(
        default="/dev/null",
        description=(
            "Shell script sourced on every node before the run, carrying RCCL/NCCL/UCX tuning environment "
            "variables. Path settings live in mpi_params and rccl_test_params instead."
        ),
        examples=["/home/{user-id}/thor2_env_script.sh"],
    )
    mpi_params: RcclMpiParams = Field(default_factory=RcclMpiParams, description="MPI launch settings.")
    rccl_test_params: RcclTestParams = Field(
        default_factory=RcclTestParams, description="Arguments passed to the rccl-tests binaries."
    )
    cvs_params: RcclCvsParams = Field(
        default_factory=RcclCvsParams, description="CVS-side verification and bookkeeping."
    )
    regression: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Regression sweep keyed by NCCL_* environment variable name, each mapping to the values to try. "
            "CVS runs the Cartesian product of all combinations. Channel counts use a 'min-max' string such as "
            "'16-16', or 'default'. The Tree algorithm is automatically filtered to run only with all_reduce_perf. "
            "Used by rccl_regression only."
        ),
        examples=[{"NCCL_ALGO": ["Ring", "Tree"], "NCCL_PROTO": ["Simple"]}],
    )
