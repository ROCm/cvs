"""
Preflight check configuration file schema.

Mirrors ``cvs/input/config_file/preflight/``.
"""

import warnings
from copy import deepcopy
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Preflight Check Configuration Schema
# =============================================================================


LEGACY_PREFLIGHT_RDMA_PATHS = {
    "gid_index": "gid_index",
    "rdma_interfaces": "interfaces",
}

PREFLIGHT_METADATA_PREFIXES = ("_comment", "_example")


def strip_preflight_metadata(value):
    """Remove documentation-only pseudo-fields before schema validation.

    Preflight JSON files conventionally carry ``_comment*`` and ``_example*``
    keys so that the files remain self-documenting. They are not runtime
    options. Strip only those reserved prefixes recursively, preserving strict
    rejection of every other unknown customer-facing option.
    """
    if isinstance(value, dict):
        return {
            key: strip_preflight_metadata(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key.startswith(PREFLIGHT_METADATA_PREFIXES))
        }
    if isinstance(value, list):
        return [strip_preflight_metadata(item) for item in value]
    return value


def normalize_legacy_preflight_rdma_config(value):
    """Move the two deprecated node-check RDMA keys to their canonical block.

    Returns a deep-copied configuration and one consolidated warning message,
    or the original value and ``None`` when no legacy keys are present.
    Conflicting legacy and canonical values fail rather than silently choosing
    which RDMA inventory should be tested.
    """
    if not isinstance(value, dict):
        return value, None

    node_check = value.get("node_check")
    if not isinstance(node_check, dict):
        return value, None

    legacy_keys = [key for key in LEGACY_PREFLIGHT_RDMA_PATHS if key in node_check]
    if not legacy_keys:
        return value, None

    normalized = deepcopy(value)
    normalized_node_check = normalized["node_check"]
    connectivity_check = normalized.setdefault("connectivity_check", {})
    if not isinstance(connectivity_check, dict):
        raise ValueError(
            "preflight.connectivity_check must be an object when deprecated node_check RDMA options are used"
        )
    rdma = connectivity_check.setdefault("rdma", {})
    if not isinstance(rdma, dict):
        raise ValueError(
            "preflight.connectivity_check.rdma must be an object when deprecated node_check RDMA options are used"
        )

    migrations = []
    for legacy_key in legacy_keys:
        canonical_key = LEGACY_PREFLIGHT_RDMA_PATHS[legacy_key]
        legacy_value = normalized_node_check.pop(legacy_key)
        if canonical_key in rdma and rdma[canonical_key] != legacy_value:
            raise ValueError(
                f"Conflicting preflight RDMA options: preflight.node_check.{legacy_key} and "
                f"preflight.connectivity_check.rdma.{canonical_key} must have the same value when both are provided"
            )
        rdma.setdefault(canonical_key, legacy_value)
        migrations.append(f"preflight.node_check.{legacy_key} -> preflight.connectivity_check.rdma.{canonical_key}")

    warning_message = (
        "Deprecated preflight RDMA configuration detected: "
        + ", ".join(migrations)
        + ". Use the preflight.connectivity_check.rdma paths; legacy paths will be removed in a future release."
    )
    return normalized, warning_message


LEGACY_PREFLIGHT_NODE_SMOKE_SECTIONS = {
    "node_smoke": "node_smoke_tier1",
    "tier3_info": "node_smoke_tier3",
}


def _preflight_section_has_values(section: dict) -> bool:
    if not isinstance(section, dict):
        return False
    return any(value not in (None, "") for value in section.values())


def normalize_legacy_preflight_node_smoke_sections(value):
    """Copy legacy Node Smoke section names to their canonical tier keys.

    Returns a deep-copied configuration and one consolidated warning message,
    or the original value and ``None`` when no legacy keys need migration.
    Canonical sections win when both legacy and canonical blocks are populated.
    """
    if not isinstance(value, dict):
        return value, None

    legacy_keys = [key for key in LEGACY_PREFLIGHT_NODE_SMOKE_SECTIONS if key in value]
    if not legacy_keys:
        return value, None

    normalized = deepcopy(value)
    migrations = []
    for legacy_key in legacy_keys:
        canonical_key = LEGACY_PREFLIGHT_NODE_SMOKE_SECTIONS[legacy_key]
        legacy_block = normalized.get(legacy_key)
        if not isinstance(legacy_block, dict):
            continue
        canonical_block = normalized.get(canonical_key)
        if isinstance(canonical_block, dict) and _preflight_section_has_values(canonical_block):
            continue
        normalized[canonical_key] = deepcopy(legacy_block)
        migrations.append(f"preflight.{legacy_key} -> preflight.{canonical_key}")

    if not migrations:
        return value, None

    warning_message = (
        "Deprecated preflight Node Smoke section name(s) detected: "
        + ", ".join(migrations)
        + ". Prefer node_smoke_tier1 and node_smoke_tier3 in new configs."
    )
    return normalized, warning_message


class PreflightParallelismConfig(BaseModel):
    """Legacy parallelism settings for preflight checks."""

    model_config = ConfigDict(extra="allow")

    parallel_group_size: int = Field(
        default=128,
        ge=2,
        le=512,
        description=("Legacy alias for RDMA grouping. Prefer connectivity_check.rdma.nodes_per_full_mesh_group."),
    )


class PreflightDebugConfig(BaseModel):
    """Debug and troubleshooting settings for preflight checks."""

    model_config = ConfigDict(extra="allow")

    scriptlet: bool = Field(
        default=False,
        description=(
            "Enable ScriptLet debug: preserve generated scripts/logs on remote nodes. "
            "For RDMA connectivity, also wraps each ibv_rc_pingpong server in strace with "
            "per-test traces under /tmp/preflight/strace_server_<iface>_<port>.log (expensive at scale)."
        ),
    )


class PreflightNodeCheckConfig(BaseModel):
    """Individual node validation settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable generic GPU node health and ROCm validation")
    gpus_per_node: int = Field(default=4, ge=1, description="Expected AMD GPU count on each node")
    expected_rocm_version: str = Field(default="6.2.0", description="Expected ROCm version across all cluster nodes")


class PreflightRdmaConfig(BaseModel):
    """RDMA connectivity testing settings."""

    model_config = ConfigDict(extra="allow")

    connectivity_mode: str = Field(default="basic", description="RDMA connectivity testing: basic, full_mesh, or skip")
    gid_index: str = Field(default="3", description="GID index to check on all RDMA interfaces (typically 3 for RoCE)")
    interfaces: List[str] = Field(
        default_factory=lambda: ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
        min_length=1,
        description="RDMA device names checked for presence, GID consistency, and connectivity",
    )
    nodes_per_full_mesh_group: int = Field(
        default=128,
        ge=2,
        le=512,
        description=(
            "Number of nodes in each full-mesh partition group (2-512). "
            "Smaller groups use fewer resources per node but require more rounds."
        ),
    )
    parallel_group_size: int = Field(
        default=128,
        ge=2,
        le=512,
        description="Legacy alias for nodes_per_full_mesh_group.",
    )
    ibv_test_timeout: int = Field(
        default=90,
        ge=1,
        description="Timeout in seconds for RDMA connectivity tests using ibv_rc_pingpong",
    )
    ibv_test_port_range: str = Field(
        default="10000-50000", description="Port range for RDMA connectivity tests (format: start-end)"
    )
    inter_full_mesh_group_pairs_per_wave: str = Field(
        default="auto", description="Max ordered group-pairs per wave during inter-group testing ('auto' or integer)"
    )
    inter_group_wave_pairs: str = Field(
        default="auto",
        description="Legacy alias for inter_full_mesh_group_pairs_per_wave.",
    )
    prune_failure_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Round 1 (intra) prune before inter-group: prune nodes whose fraction of peers with ≥1 FAIL "
            "intra test is ≥ this value (default 0.5). Peers counted per distinct other node in the same partition group."
        ),
    )
    port_retry_max: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "After each ScriptLet wave (intra/inter), rerun only pairs whose logs show PORT_LISTEN_FAILED, "
            "up to this many extra batches with new TCP ports (default 3)."
        ),
    )
    port_retry_gap: int = Field(
        default=1000,
        ge=1,
        le=65535,
        description=(
            "When remapping ports for PORT_LISTEN_FAILED retries, start at (max port in batch) + this gap "
            "to reduce overlap with ephemeral ports."
        ),
    )
    exclude_failed_interface_nodes: str = Field(
        default="true",
        description=(
            "Legacy hint for reporting: preflight now prunes interface- and GID-failed nodes from the SSH "
            "host list before RDMA; interface failures are not run in the mesh regardless of this flag."
        ),
    )

    @field_validator('connectivity_mode')
    @classmethod
    def validate_connectivity_check(cls, v: str) -> str:
        """Validate RDMA connectivity check setting."""
        valid_modes = ['basic', 'full_mesh', 'skip']
        if v not in valid_modes:
            raise ValueError(f"connectivity_mode must be one of: {', '.join(valid_modes)}")
        return v

    @field_validator('ibv_test_port_range')
    @classmethod
    def validate_port_range(cls, v: str) -> str:
        """Validate port range format."""
        try:
            start, end = map(int, v.split('-'))
            if start >= end or start < 1024 or end > 65535:
                raise ValueError("Invalid port range")
        except (ValueError, AttributeError):
            raise ValueError("ibv_test_port_range must be in format 'start-end' with valid port numbers")
        return v


class PreflightL2PingConfig(BaseModel):
    """Small customer-facing IFoE L2 ping policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable the mandatory IFoE L2 connectivity gate")
    pings_per_port: int = Field(default=3, ge=1, description="Ping samples per selected IFoE port pair")


class PreflightTransferBenchConfig(BaseModel):
    """Small customer-facing TransferBench preflight policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable the mandatory TransferBench preflight gate")
    scope: str = Field(default="node", description="node for independent runs or cluster for one multi-rank run")
    profile: str = Field(default="smoketest", description="CVS-supported TransferBench validation profile")
    message_sizes: List[str] = Field(
        default_factory=lambda: ["1K", "16M"],
        min_length=1,
        description="Message sizes exercised by the selected profile",
    )
    iterations: int = Field(default=2, ge=1, description="Validated iterations per test and message size")
    warmup_iterations: int = Field(default=0, ge=0, description="Warmup iterations before validation")

    @field_validator('scope')
    @classmethod
    def validate_transferbench_scope(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ('node', 'cluster'):
            raise ValueError("TransferBench scope must be one of: node, cluster")
        return normalized

    @field_validator('profile')
    @classmethod
    def validate_transferbench_profile(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized != 'smoketest':
            raise ValueError("TransferBench profile must be a CVS-supported profile: smoketest")
        return normalized

    @field_validator('message_sizes')
    @classmethod
    def validate_transferbench_message_sizes(cls, value: List[str]) -> List[str]:
        normalized = [str(size).strip() for size in value]
        if any(not size for size in normalized):
            raise ValueError("TransferBench message_sizes entries must not be empty")
        return normalized


class PreflightIfoeConfig(BaseModel):
    """MI4XX IFoE admission and data-path checks."""

    model_config = ConfigDict(extra="forbid")

    fabric_checks: bool = Field(
        default=False,
        description="Enable MI4XX AIFM, AFM, vPOD, station-mask, and IFoE port admission",
    )
    l2ping: PreflightL2PingConfig = Field(
        default_factory=PreflightL2PingConfig,
        description="Strict IFoE L2 connectivity admission",
    )
    transferbench: PreflightTransferBenchConfig = Field(
        default_factory=PreflightTransferBenchConfig,
        description="TransferBench IFoE data-path validation",
    )


class PreflightConnectivityCheckConfig(BaseModel):
    """Connectivity check settings by protocol."""

    model_config = ConfigDict(extra="allow")

    rdma: PreflightRdmaConfig = Field(default_factory=PreflightRdmaConfig, description="RDMA connectivity settings")
    ifoe: PreflightIfoeConfig = Field(default_factory=PreflightIfoeConfig, description="IFoE connectivity settings")


class PreflightNodeSmokeConfig(BaseModel):
    """Primus node_smoke settings (primus-cli direct -- node_smoke)."""

    model_config = ConfigDict(extra="allow")

    connectivity_mode: str = Field(
        default="skip",
        description="Primus node_smoke mode: 'run' (host/GPU/RDMA roll-call) or 'skip' (default)",
    )
    auto_setup: bool = Field(
        default=True,
        description="Clone/update Primus and prepare venv on each node before node_smoke",
    )
    setup_timeout: int = Field(default=600, ge=60, description="SSH timeout in seconds for Primus auto_setup")
    force_reclone: bool = Field(
        default=False,
        description="Remove primus_dir and clone fresh on every run (destructive)",
    )
    shared_install: bool = Field(
        default=True,
        description=(
            "When true (default), clone and venv setup run only on the first reachable node; "
            "other nodes wait for the shared NFS home install. Set false only if each node has "
            "a local primus_dir/venv_activate path."
        ),
    )
    pip_install_mode: str = Field(
        default="minimal",
        description="Venv deps: minimal (torch only), requirements, or skip",
    )
    torch_pip_index_url: str = Field(
        default="https://download.pytorch.org/whl/rocm6.2",
        description="PyTorch ROCm wheel index URL for minimal pip_install_mode",
    )
    primus_git_url: str = Field(
        default="https://github.com/AMD-AIG-AIMA/Primus.git",
        description="Primus repository URL for auto_setup clone",
    )
    primus_git_branch: str = Field(
        default="dev/preflight-direct-test",
        description="Git branch to checkout during auto_setup",
    )
    primus_git_recurse_submodules: bool = Field(
        default=False,
        description="Clone git submodules during auto_setup (not required for node_smoke)",
    )
    primus_dir: str = Field(
        default="/home/{user-id}/INSTALL/Primus",
        description="Path to cloned Primus repo under the user's home directory (required when connectivity_mode is 'run')",
    )
    venv_activate: str = Field(
        default="/home/{user-id}/envs/preflight/.venv/bin/activate",
        description="Path to Python venv activate script on each node (required when connectivity_mode is 'run')",
    )
    gpus_per_node: int = Field(default=8, ge=1, description="GPUs per node for node_smoke")
    master_port: int = Field(default=1234, ge=1024, le=65535, description="Distributed master port for node_smoke")
    dump_path: str = Field(
        default="",
        description="Per-node dump directory for smoke JSON (default: <artifacts_root_dir>/node_smoke)",
    )
    expected_rdma_nics: Optional[int] = Field(
        default=None,
        ge=1,
        description="Hard-fail when training RDMA NIC count differs (default: len(connectivity_check.rdma.interfaces))",
    )
    ulimit_l_min_gb: float = Field(default=32.0, ge=0, description="Minimum RLIMIT_MEMLOCK in GiB (0 disables)")
    shm_min_gb: float = Field(default=8.0, ge=0, description="Minimum /dev/shm size in GiB (0 disables)")
    skip_dmesg: bool = Field(default=False, description="Skip dmesg error scan (e.g. unprivileged containers)")
    allow_foreign_procs: bool = Field(
        default=False,
        description="Do not FAIL nodes with foreign GPU processes (still reported)",
    )
    allowed_procs: str = Field(
        default="gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter",
        description="Comma-separated process names allowed to hold GPUs",
    )
    require_tools: str = Field(
        default="",
        description="Comma-separated CLI tools that must exist in PATH (empty = warn only)",
    )
    nccl_socket_ifname: str = Field(default="", description="NCCL_SOCKET_IFNAME override for node_smoke")
    gloo_socket_ifname: str = Field(
        default="", description="GLOO_SOCKET_IFNAME override (defaults to nccl_socket_ifname)"
    )
    nccl_ib_hca: str = Field(
        default="",
        description="NCCL_IB_HCA override (defaults to comma-joined connectivity_check.rdma.interfaces)",
    )
    nccl_ib_gid_index: Optional[int] = Field(
        default=None,
        description="NCCL_IB_GID_INDEX override (defaults to connectivity_check.rdma.gid_index)",
    )
    rdma_nic_allowlist: str = Field(
        default="",
        description="Training NIC allowlist for node_smoke (defaults to connectivity_check.rdma.interfaces)",
    )
    ssh_timeout: int = Field(default=300, ge=30, description="SSH timeout in seconds for each node_smoke run")
    tier2_perf: bool = Field(
        default=False,
        description=(
            "Enable Primus node_smoke Tier 2 perf sanity (--tier2-perf): "
            "8192³ GEMM TFLOPS floor, HBM D2D bandwidth, local multi-GPU RCCL all-reduce"
        ),
    )
    gemm_tflops_min: float = Field(
        default=600.0,
        ge=0,
        description="Tier 2 large GEMM TFLOPS floor (--gemm-tflops-min); used when tier2_perf is true",
    )
    hbm_gbs_min: float = Field(
        default=2000.0,
        ge=0,
        description="Tier 2 HBM device-to-device bandwidth floor in GB/s (--hbm-gbs-min)",
    )
    rccl_gbs_min: float = Field(
        default=100.0,
        ge=0,
        description="Tier 2 local multi-GPU RCCL all-reduce bandwidth floor in GB/s (--rccl-gbs-min)",
    )
    rccl_size_mb: int = Field(
        default=64,
        ge=1,
        description="Tier 2 local RCCL all-reduce message size in MB (--rccl-size-mb)",
    )
    rccl_timeout_sec: int = Field(
        default=120,
        ge=30,
        description="Tier 2 local RCCL all-reduce hard timeout in seconds (--rccl-timeout-sec)",
    )
    extra_args: List[str] = Field(
        default_factory=list,
        description="Additional node_smoke CLI flags forwarded to primus-cli",
    )

    @field_validator("connectivity_mode")
    @classmethod
    def validate_node_smoke_mode(cls, v: str) -> str:
        valid_modes = ["run", "skip"]
        if v not in valid_modes:
            raise ValueError(f"node_smoke.connectivity_mode must be one of: {', '.join(valid_modes)}")
        return v


class PreflightTier3InfoConfig(BaseModel):
    """Primus preflight Tier 3 Host/GPU/Network info (primus-cli direct -- preflight)."""

    model_config = ConfigDict(extra="allow")

    connectivity_mode: str = Field(
        default="skip",
        description="Tier 3 info mode: 'run' (preflight --host --gpu --network) or 'skip' (default)",
    )
    auto_setup: bool = Field(
        default=True,
        description="Clone/update Primus and prepare venv before Tier 3 (uses node_smoke git/pip settings via PrimusSetup fallback)",
    )
    primus_dir: str = Field(
        default="",
        description="Primus checkout path; empty uses tier3_info then node_smoke.primus_dir",
    )
    venv_activate: str = Field(
        default="",
        description="Venv activate script; empty uses tier3_info then node_smoke.venv_activate",
    )
    gpus_per_node: int = Field(default=8, ge=1, description="GPUs per node for torchrun")
    master_port: int = Field(default=1234, ge=1024, le=65535, description="Distributed master port")
    dump_path: str = Field(
        default="",
        description="Tier 3 report directory (default: <artifacts_root_dir>/tier3_info)",
    )
    report_file_name: str = Field(default="tier3_info", description="Base name for Primus markdown/PDF reports")
    dist_timeout_sec: int = Field(
        default=120, ge=30, description="Timeout for torch.distributed init during aggregated report"
    )
    save_pdf: bool = Field(default=False, description="Generate PDF report via Primus")
    nccl_socket_ifname: str = Field(default="", description="NCCL_SOCKET_IFNAME override")
    gloo_socket_ifname: str = Field(default="", description="GLOO_SOCKET_IFNAME override")
    nccl_ib_hca: str = Field(
        default="",
        description="NCCL_IB_HCA override (defaults to comma-joined connectivity_check.rdma.interfaces when empty)",
    )
    nccl_ib_gid_index: Optional[int] = Field(
        default=None,
        description="NCCL_IB_GID_INDEX override (defaults to connectivity_check.rdma.gid_index when null)",
    )
    ssh_timeout: int = Field(default=600, ge=30, description="SSH timeout in seconds for the Tier 3 cluster run")
    extra_args: List[str] = Field(default_factory=list, description="Additional preflight CLI flags")

    @field_validator("connectivity_mode")
    @classmethod
    def validate_tier3_info_mode(cls, v: str) -> str:
        valid_modes = ["run", "skip"]
        if v not in valid_modes:
            raise ValueError(f"tier3_info.connectivity_mode must be one of: {', '.join(valid_modes)}")
        return v


class PreflightReportingConfig(BaseModel):
    """Report generation and output settings."""

    model_config = ConfigDict(extra="allow")

    generate_html_report: bool = Field(default=True, description="Whether to generate HTML report")
    artifacts_root_dir: str = Field(
        default="/tmp/preflight",
        description=(
            "Root directory for preflight artifacts. HTML report output and RDMA full_mesh ScriptLet logs use "
            "<artifacts_root_dir>/rdma_connectivity_workspace/<session>/<round>/ on each node (NFS-friendly). "
            "Sample configs use /home/{user-id}/preflight; that placeholder is resolved only when present in JSON."
        ),
    )
    generate_rdma_pairs_csv: bool = Field(
        default=True,
        description="If true, write preflight_report_*_rdma_pairs.csv beside the HTML report (failed pairs only)",
    )


class PreflightConfigFile(BaseModel):
    """
    Schema for preflight check configuration file.

    Uses nested structure organized by execution phase for better organization.
    """

    model_config = ConfigDict(extra="allow")  # Allow comment fields

    parallelism: PreflightParallelismConfig = Field(
        default_factory=PreflightParallelismConfig, description="Parallel execution settings"
    )
    debug: PreflightDebugConfig = Field(
        default_factory=PreflightDebugConfig, description="Debug and troubleshooting options"
    )
    node_check: PreflightNodeCheckConfig = Field(
        default_factory=PreflightNodeCheckConfig, description="Individual node validation settings"
    )
    connectivity_check: PreflightConnectivityCheckConfig = Field(
        default_factory=PreflightConnectivityCheckConfig, description="Inter-node connectivity tests"
    )
    node_smoke: PreflightNodeSmokeConfig = Field(
        default_factory=PreflightNodeSmokeConfig, description="Primus node_smoke checks"
    )
    tier3_info: PreflightTier3InfoConfig = Field(
        default_factory=PreflightTier3InfoConfig,
        description="Primus Tier 3 preflight Host/GPU/Network info checks",
    )
    reporting: PreflightReportingConfig = Field(
        default_factory=PreflightReportingConfig, description="Report generation and output settings"
    )

    @model_validator(mode="before")
    @classmethod
    def reject_flat_preflight_checks(cls, value):
        if not isinstance(value, dict):
            return value
        cleaned = strip_preflight_metadata(value)
        removed = sorted(set(cleaned) & {"node_health", "l2ping", "transferbench"})
        if removed:
            raise ValueError(
                "Unsupported flat preflight block(s): "
                + ", ".join(removed)
                + "; use node_check and connectivity_check.ifoe"
            )
        normalized, rdma_warning = normalize_legacy_preflight_rdma_config(cleaned)
        if rdma_warning:
            warnings.warn(rdma_warning, FutureWarning, stacklevel=2)
        normalized, smoke_warning = normalize_legacy_preflight_node_smoke_sections(normalized)
        if smoke_warning:
            warnings.warn(smoke_warning, FutureWarning, stacklevel=2)
        return normalized

    @model_validator(mode="after")
    def validate_fabric_prerequisites(self):
        if self.connectivity_check.ifoe.fabric_checks and not self.node_check.enabled:
            raise ValueError("connectivity_check.ifoe.fabric_checks requires node_check.enabled=true")
        return self
