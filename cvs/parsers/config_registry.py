"""Maps each test to the config sections whose parameters ``cvs man`` documents.

The relationship is many-to-many: one config file can serve several tests, each
reading a different top-level section, and one test can read more than one
section. This registry is the only machine-readable statement of that mapping.

Tests absent from here simply have no man page yet; ``cvs man`` reports them as
undocumented rather than failing.
"""

import os
from collections import namedtuple

from cvs.lib.utils_lib import cvs_package_root
from cvs.parsers.config_schemas import (
    AgfhcConfigFile,
    MegatronConfigFile,
    MegatronMultiNodeModelParams,
    MegatronSingleNodeModelParams,
    RcclConfigFile,
    RvsConfigFile,
    TransferBenchConfigFile,
)
from cvs.parsers.schemas import PreflightConfigFile

# key: top-level key in the config file; model: schema documenting that key.
ConfigSection = namedtuple("ConfigSection", ["key", "model"])

# sections: ordered ConfigSection entries; samples: shipped configs, relative to the cvs package
# directory (the same root as cvs/input/, cvs/tests/, ...). Use resolve_sample_path() to turn
# these into real filesystem paths — the cvs package directory is not the repo/install root.
ConfigDoc = namedtuple("ConfigDoc", ["summary", "sections", "samples"])


def resolve_sample_path(relative):
    """Return the real filesystem path to a shipped sample config."""
    return os.path.join(cvs_package_root(), relative)


_RCCL_SAMPLES = ("input/config_file/rccl/rccl_config.json",)
_HEALTH_SAMPLES = ("input/config_file/health/mi300_health_config.json",)
_PREFLIGHT_SAMPLES = ("input/config_file/preflight/preflight_config.json",)
_MEGATRON_SINGLE_SAMPLES = (
    "input/config_file/training/megatron/mi3xx_megatron_llama_single.json",
    "input/config_file/training/megatron/mi35x_megatron_llama_single.json",
)
_MEGATRON_DISTRIBUTED_SAMPLES = ("input/config_file/training/megatron/mi3xx_megatron_llama_distributed.json",)

_RCCL_SECTIONS = (ConfigSection("rccl", RcclConfigFile),)
_AGFHC_SECTIONS = (ConfigSection("agfhc", AgfhcConfigFile),)
_RVS_SECTIONS = (ConfigSection("rvs", RvsConfigFile),)
_TRANSFERBENCH_SECTIONS = (ConfigSection("transferbench", TransferBenchConfigFile),)
# The megatron config's model_params section is split by cluster topology at
# runtime (single_node vs. multi_node); a single/distributed test only ever
# reads its own half, so each variant gets its own section here rather than
# sharing one that would document (and sample) the other topology too.
_MEGATRON_SINGLE_SECTIONS = (
    ConfigSection("config", MegatronConfigFile),
    ConfigSection("model_params", MegatronSingleNodeModelParams),
)
_MEGATRON_DISTRIBUTED_SECTIONS = (
    ConfigSection("config", MegatronConfigFile),
    ConfigSection("model_params", MegatronMultiNodeModelParams),
)

TEST_CONFIG_DOCS = {
    "rccl_perf": ConfigDoc(
        summary="Sweeps RCCL collectives across the cluster and checks bandwidth and latency.",
        sections=_RCCL_SECTIONS,
        samples=_RCCL_SAMPLES,
    ),
    "rccl_regression": ConfigDoc(
        summary="Runs the Cartesian product of the regression NCCL_* settings against the collectives.",
        sections=_RCCL_SECTIONS,
        samples=_RCCL_SAMPLES,
    ),
    "rccl_pairwise": ConfigDoc(
        summary="Isolates bad links by testing node pairs, then admitting nodes incrementally.",
        sections=_RCCL_SECTIONS,
        samples=_RCCL_SAMPLES,
    ),
    "agfhc_cvs": ConfigDoc(
        summary="Runs AMD GPU Field Health Check testcases on every node.",
        sections=_AGFHC_SECTIONS,
        samples=_HEALTH_SAMPLES,
    ),
    "csp_qual_agfhc": ConfigDoc(
        summary="Runs the CSP qualification AGFHC testcase set, collecting logs per node.",
        sections=_AGFHC_SECTIONS,
        samples=_HEALTH_SAMPLES,
    ),
    "install_agfhc": ConfigDoc(
        summary="Installs AGFHC on every node from the configured tarball.",
        sections=_AGFHC_SECTIONS,
        samples=_HEALTH_SAMPLES,
    ),
    "rvs_cvs": ConfigDoc(
        summary="Runs ROCm Validation Suite modules, or the combined LEVEL test.",
        sections=_RVS_SECTIONS,
        samples=_HEALTH_SAMPLES,
    ),
    "install_rvs": ConfigDoc(
        summary="Installs ROCm Validation Suite on every node.",
        sections=_RVS_SECTIONS,
        samples=_HEALTH_SAMPLES,
    ),
    "transferbench_cvs": ConfigDoc(
        summary="Measures GPU-to-GPU copy bandwidth with TransferBench and checks it against floors.",
        sections=_TRANSFERBENCH_SECTIONS,
        samples=_HEALTH_SAMPLES,
    ),
    "install_transferbench": ConfigDoc(
        summary="Clones and builds TransferBench on every node.",
        sections=_TRANSFERBENCH_SECTIONS,
        samples=_HEALTH_SAMPLES,
    ),
    "preflight_checks": ConfigDoc(
        summary="Validates node health and inter-node connectivity before longer suites are run.",
        sections=(ConfigSection("preflight", PreflightConfigFile),),
        samples=_PREFLIGHT_SAMPLES,
    ),
    "megatron_llama3_1_8b_single": ConfigDoc(
        summary="Trains Llama 3.1 8B with Megatron-LM on a single node.",
        sections=_MEGATRON_SINGLE_SECTIONS,
        samples=_MEGATRON_SINGLE_SAMPLES,
    ),
    "megatron_llama3_1_8b_distributed": ConfigDoc(
        summary="Trains Llama 3.1 8B with Megatron-LM across the cluster.",
        sections=_MEGATRON_DISTRIBUTED_SECTIONS,
        samples=_MEGATRON_DISTRIBUTED_SAMPLES,
    ),
    "megatron_llama3_1_70b_single": ConfigDoc(
        summary="Trains Llama 3.1 70B with Megatron-LM on a single node.",
        sections=_MEGATRON_SINGLE_SECTIONS,
        samples=_MEGATRON_SINGLE_SAMPLES,
    ),
    "megatron_llama3_1_70b_distributed": ConfigDoc(
        summary="Trains Llama 3.1 70B with Megatron-LM across the cluster.",
        sections=_MEGATRON_DISTRIBUTED_SECTIONS,
        samples=_MEGATRON_DISTRIBUTED_SAMPLES,
    ),
}


def get_config_doc(test_name):
    """Return the ConfigDoc for a test, or None when it has no man page yet."""
    return TEST_CONFIG_DOCS.get(test_name)


def documented_tests():
    return sorted(TEST_CONFIG_DOCS)
