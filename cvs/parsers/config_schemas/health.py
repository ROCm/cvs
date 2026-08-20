"""Config schemas for the health suite sections: agfhc, transferbench, rvs.

One config file serves several tests, each reading a different top-level
section, so the three models here are independent rather than nested under a
common root.

Numeric thresholds are typed ``str`` because they ship as JSON strings and are
only coerced with ``float()`` at the comparison site; typing them as numbers
would change the on-disk contract.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AgfhcConfigFile(BaseModel):
    """Schema for the agfhc section, read by agfhc_cvs, csp_qual_agfhc and install_agfhc.

    Requirements differ per test: agfhc_cvs needs hbm_test_duration and never
    reads log_dir, while csp_qual_agfhc is the reverse.
    """

    model_config = ConfigDict(extra="allow")

    path: str = Field(
        description="Directory containing the installed agfhc binary. Every testcase runs 'sudo {path}/agfhc'.",
        examples=["/opt/amd/agfhc"],
    )
    package_tar_ball: str = Field(
        description="Path to the AGFHC tarball that install_agfhc copies to each node and extracts.",
        examples=["/home/{user-id}/PACKAGES/agfhc-mi300x_1.22.0_ub2204.tar.bz2"],
    )
    install_dir: str = Field(
        description="Staging directory the tarball is copied into and extracted in. Created if absent.",
        examples=["/home/{user-id}/INSTALL/agfhc/"],
    )
    log_dir: str = Field(
        description=(
            "Destination for AGFHC run logs, passed as -o. Must be a non-NFS local filesystem. Wiped and "
            "recreated at the start of each run. Read by csp_qual_agfhc only."
        ),
        examples=["/root/agfhc_logs"],
    )
    hbm_test_duration: str = Field(
        description=(
            "Duration of the HBM test as HH:MM:SS. Drives both the agfhc -t hbm:d= argument and the exec "
            "timeout, which is this duration plus 120 seconds. Read by agfhc_cvs only."
        ),
        examples=["00:01:30"],
    )


class TransferBenchResults(BaseModel):
    """Minimum acceptable bandwidths, in GB/s, compared against parsed TransferBench output."""

    model_config = ConfigDict(extra="allow")

    gpu_to_gpu_a2a_rtotal: str = Field(
        description="Floor for every per-column RTotal bandwidth in the all-to-all run.",
        examples=["320.0"],
    )
    avg_gpu_to_gpu_p2p_unidir_bw: str = Field(
        description="Floor for the 'Averages (During UniDir)' row of the peer-to-peer run.",
        examples=["33.9"],
    )
    avg_gpu_to_gpu_p2p_bidir_bw: str = Field(
        description="Floor for the 'Averages (During BiDir)' row of the peer-to-peer run.",
        examples=["43.9"],
    )
    best_gpu0_bw: str = Field(
        description="Floor for the GPU00 column of the 'Best' summary row in the scaling run.",
        examples=["480.0"],
    )
    local_read_32_cu: str = Field(
        alias="32_cu_local_read",
        description="Floor for local read in the 32-CU schmoo row.",
        examples=["1650"],
    )
    local_write_32_cu: str = Field(
        alias="32_cu_local_write",
        description="Floor for local write in the 32-CU schmoo row.",
        examples=["1250.0"],
    )
    local_copy_32_cu: str = Field(
        alias="32_cu_local_copy",
        description="Floor for local copy in the 32-CU schmoo row.",
        examples=["1250.0"],
    )
    rem_read_32_cu: str = Field(
        alias="32_cu_rem_read",
        description="Floor for remote read in the 32-CU schmoo row.",
        examples=["48.0"],
    )
    rem_write_32_cu: str = Field(
        alias="32_cu_rem_write",
        description="Floor for remote write in the 32-CU schmoo row.",
        examples=["48.0"],
    )
    rem_copy_32_cu: str = Field(
        alias="32_cu_rem_copy",
        description="Floor for remote copy in the 32-CU schmoo row.",
        examples=["48.0"],
    )
    bytes_to_transfer: Optional[str] = Field(
        default=None,
        description="Not read by any code path. TransferBench payload sizes are set by the mode, not this key.",
        examples=["268435456"],
    )
    path: Optional[str] = Field(
        default=None,
        description="Not read by any code path; the tests use the section-level transferbench.path instead.",
        examples=["/home/{user-id}/INSTALL/TransferBench"],
    )


class TransferBenchConfigFile(BaseModel):
    """Schema for the transferbench section, read by transferbench_cvs and install_transferbench."""

    model_config = ConfigDict(extra="allow")

    path: str = Field(
        description="Directory containing the built TransferBench binary, invoked as '{path}/TransferBench <mode>'.",
        examples=["/home/{user-id}/INSTALL/TransferBench"],
    )
    git_install_path: str = Field(
        description="Parent directory the repository is cloned into, as '{git_install_path}/TransferBench'.",
        examples=["/home/{user-id}/INSTALL/"],
    )
    git_url: str = Field(
        description="Git remote cloned by install_transferbench.",
        examples=["https://github.com/ROCm/TransferBench.git"],
    )
    git_tag: str = Field(
        description=(
            "Git tag checked out after clone, then verified with 'git describe --tags --exact-match'. "
            "Omitting it fails the install with a explicit error."
        ),
        examples=["v1.67.00"],
    )
    rocm_path: str = Field(
        default="",
        description=(
            "ROCm installation path. Leave empty to auto-detect from /opt/rocm or /opt/rocm/core-*. Note that "
            "the literal string '<changeme>' cannot be used as the auto-detect sentinel: placeholder resolution "
            "aborts the run before the sentinel is ever examined."
        ),
        examples=["/opt/rocm"],
    )
    results: TransferBenchResults = Field(description="Minimum acceptable bandwidths per TransferBench mode.")


class RvsTestConfig(BaseModel):
    """One entry of the rvs.tests list, selected by name."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        description="Selector matched against the hardcoded test function names.",
        examples=["level_config", "mem_test", "gst_single", "iet_stress", "pebb_single", "pbqt_single", "babel_stream"],
    )
    config_file: Optional[str] = Field(
        default=None,
        description=(
            "RVS .conf filename resolved under config_path_default, preferring a device-specific subdirectory. "
            "Omitted for level_config, which uses 'rvs -r <level>' instead of '-c <conf>'."
        ),
        examples=["mem.conf", "gst_single.conf"],
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable label used in log lines. Falls back to the entry's own name.",
        examples=["Memory Test"],
    )
    timeout: int = Field(
        default=9000,
        description=(
            "Per-test exec timeout in seconds. The default is 9000 for individual modules but 7200 for the "
            "level_config entry."
        ),
    )
    fail_regex_pattern: str = Field(
        default=r"\[ERROR\s*\]",
        description=(
            "Single alternation-joined regex searched case-insensitively in the output of an individual module; "
            "a match on any node fails the test. The default is a bare-minimum fallback that is far weaker than "
            "every shipped value, so omitting this key silently downgrades failure detection."
        ),
        examples=[r"FAIL|\[ERROR\s*\]|RVS-ERROR"],
    )
    fail_regex_patterns: List[str] = Field(
        default_factory=list,
        description=(
            "Plural form used only by the level_config run, where each pattern is searched separately. An empty "
            "list makes the LEVEL test pass unconditionally, so populate it whenever level_config is enabled."
        ),
        examples=[["peqt false", "RVS-ERROR"]],
    )
    expected_pass: Optional[bool] = Field(
        default=None,
        description="Not read by any code path. Pass and fail are decided entirely by the regex fields.",
        examples=[True],
    )


class RvsConfigFile(BaseModel):
    """Schema for the rvs section, read by rvs_cvs and install_rvs."""

    model_config = ConfigDict(extra="allow")

    path: str = Field(
        description=(
            "Directory containing the rvs binary. install_rvs rewrites this value in place once it has detected "
            "the real ROCm path."
        ),
        examples=["/opt/rocm/bin"],
    )
    git_install_path: str = Field(
        description="Staging directory for the pre-built RVS tarball download. No git clone is performed.",
        examples=["/home/{user-id}/INSTALL/rvs"],
    )
    rocm_path: str = Field(
        default="",
        description=(
            "ROCm installation path used by the installer only. Leave empty to auto-detect from /opt/rocm or "
            "/opt/rocm/core-*."
        ),
        examples=["/opt/rocm"],
    )
    rocm_runtime_lib_path: str = Field(
        default="",
        description=(
            "Optional colon-separated directories prepended to LD_LIBRARY_PATH for every rvs invocation and for "
            "the install-time ldd check. Use this when the rvs binary was built against a newer ROCm than "
            "/opt/rocm points at and needs to load amd_smi or rocm_smi from a side-by-side install. Leave empty "
            "for default loader behaviour."
        ),
        examples=["/home/{user-id}/install/lib:/home/{user-id}/install/lib/rocm_sysdeps"],
    )
    config_path_default: str = Field(
        description=(
            "Base directory for RVS .conf files. A device-specific subdirectory is preferred when one matching "
            "the detected GPU exists, otherwise the base directory is used."
        ),
        examples=["/opt/rocm/share/rocm-validation-suite/conf"],
    )
    config_path_mi300x: str = Field(
        description=(
            "First-choice directory probed for gst_single.conf during install verification. Rewritten in place "
            "with the detected ROCm path. Not read at test time, where device directories are discovered "
            "dynamically."
        ),
        examples=["/opt/rocm/share/rocm-validation-suite/conf/MI300X"],
    )
    rvs_test_level: int = Field(
        default=4,
        ge=0,
        le=5,
        description=(
            "RVS test level, passed as 'rvs -r <level>'. 0 runs the individual module tests and skips the LEVEL "
            "test; 1 to 5 run the LEVEL config test when RVS is at least 1.3.0, falling back to the individual "
            "tests otherwise. Values outside 0-5 fall back to 4."
        ),
    )
    tests: List[RvsTestConfig] = Field(
        description=(
            "Per-module test definitions, selected by name. If the level_config entry is absent, a hardcoded "
            "fallback is substituted."
        ),
    )
    git_url: Optional[str] = Field(
        default=None,
        description=(
            "Not read by any code path. The installer uses the ROCm apt package and falls back to a hardcoded "
            "tarball URL."
        ),
        examples=["https://github.com/ROCm/ROCmValidationSuite.git"],
    )
    nfs_install: Optional[str] = Field(
        default=None,
        description="Not read by any code path in the RVS tests.",
        examples=["True"],
    )
