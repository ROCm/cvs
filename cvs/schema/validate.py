"""
Load and validate CVS configuration files against Pydantic schemas.
"""

from pathlib import Path
from typing import Any, Type, Union

from cvs.schema.cluster_file.cluster import ClusterConfigFile
from cvs.schema.config_file.aorta.benchmark import AortaBenchmarkConfigFile
from cvs.schema.config_file.inference.pytorch_xdit.config import (
    PytorchXditFluxConfigFile,
    PytorchXditWanConfigFile,
)
from cvs.schema.config_file.preflight.config import PreflightConfigFile
from cvs.schema.config_file.training.jaxmaxtext.variant import TrainingVariantConfig
from cvs.schema.config_file.training.megatron.variant import MegatronVariantConfig
from cvs.schema.config_file.training.torchtitan.variant import TorchTitanVariantConfig
from cvs.schema.config_file.inference.atom.variant import AtomVariantConfig
from cvs.schema.config_file.inference.sglang.variant import SglangSingleVariantConfig
from cvs.schema.config_file.inference.vllm.variant import VariantConfig as VllmVariantConfig

# Populated as additional variant schemas land.
_VARIANT_FRAMEWORK_MAP: dict[str, tuple[str, Type[Any]]] = {
    "megatron_single": ("megatron", MegatronVariantConfig),
    "megatron_distributed": ("megatron", MegatronVariantConfig),
    "torchtitan_single": ("torchtitan", TorchTitanVariantConfig),
    "torchtitan_distributed": ("torchtitan", TorchTitanVariantConfig),
    "jaxmaxtext": ("jaxmaxtext", TrainingVariantConfig),
    "vllm": ("vllm", VllmVariantConfig),
    "atom": ("atom", AtomVariantConfig),
    "sglang_single": ("sglang", SglangSingleVariantConfig),
}


def _validate_variant_config(raw_config: dict, model_cls: Type[Any]):
    """Structural validation for variant JSON (no cluster substitution).

    Threshold files are loaded separately at runtime; when ``thresholds`` is absent
    or empty, disable enforcement so shape checks still pass on committed samples.
    """
    known = {k: v for k, v in raw_config.items() if k in model_cls.model_fields}
    known.setdefault("thresholds", {})
    if not known["thresholds"]:
        known["enforce_thresholds"] = False
    return model_cls.model_validate(known)


def validate_config_file(
    config_path: Union[str, Path], config_type: str = "auto"
) -> Union[
    AortaBenchmarkConfigFile,
    ClusterConfigFile,
    PytorchXditWanConfigFile,
    PytorchXditFluxConfigFile,
    PreflightConfigFile,
    MegatronVariantConfig,
    TorchTitanVariantConfig,
    TrainingVariantConfig,
]:
    """
    Load and validate a configuration file.

    Args:
        config_path: Path to configuration file (YAML or JSON)
        config_type: Type of config - "aorta", "cluster", "pytorch_xdit_wan",
            "pytorch_xdit_flux", "preflight", "megatron", "torchtitan", or
            "auto" (detect from content)

    Returns:
        Validated Pydantic model

    Raises:
        ValueError: If config is invalid with detailed error message
        FileNotFoundError: If config file doesn't exist
    """
    import json
    import yaml

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix in ('.yaml', '.yml'):
            raw_config = yaml.safe_load(f)
        else:
            raw_config = json.load(f)

    if raw_config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    if config_type == "auto":
        if "node_dict" in raw_config:
            config_type = "cluster"
        elif "preflight" in raw_config:
            config_type = "preflight"
        elif "aorta_path" in raw_config:
            config_type = "aorta"
        elif raw_config.get("framework") in _VARIANT_FRAMEWORK_MAP:
            config_type = _VARIANT_FRAMEWORK_MAP[raw_config["framework"]][0]
        elif "config" in raw_config and "benchmark_params" in raw_config:
            config_section = raw_config.get("config", {})
            benchmark_section = raw_config.get("benchmark_params", {})

            if "flux1_dev_t2i" in benchmark_section or "FLUX" in config_section.get("model_repo", ""):
                config_type = "pytorch_xdit_flux"
            elif "wan22_i2v_a14b" in benchmark_section or "Wan" in config_section.get("model_repo", ""):
                config_type = "pytorch_xdit_wan"
            else:
                config_type = "pytorch_xdit_wan"
        else:
            raise ValueError(
                f"Cannot auto-detect config type for {config_path}. "
                f"Specify config_type='aorta', config_type='cluster', "
                f"config_type='pytorch_xdit_wan', config_type='pytorch_xdit_flux', "
                "config_type='preflight', config_type='megatron', config_type='torchtitan', "
                f"config_type='jaxmaxtext', config_type='vllm', config_type='atom', or "
                f"config_type='sglang'"
            )

    try:
        if config_type == "cluster":
            return ClusterConfigFile.model_validate(raw_config)
        if config_type == "preflight":
            if "preflight" in raw_config:
                return PreflightConfigFile.model_validate(raw_config["preflight"])
            raise ValueError("Preflight config must contain 'preflight' section")
        if config_type == "aorta":
            return AortaBenchmarkConfigFile.model_validate(raw_config)
        if config_type == "pytorch_xdit_wan":
            return PytorchXditWanConfigFile.model_validate(raw_config)
        if config_type == "pytorch_xdit_flux":
            return PytorchXditFluxConfigFile.model_validate(raw_config)
        if config_type in ("megatron", "torchtitan", "jaxmaxtext", "vllm", "atom", "sglang"):
            model_cls = {
                "megatron": MegatronVariantConfig,
                "torchtitan": TorchTitanVariantConfig,
                "jaxmaxtext": TrainingVariantConfig,
                "vllm": VllmVariantConfig,
                "atom": AtomVariantConfig,
                "sglang": SglangSingleVariantConfig,
            }[config_type]
            return _validate_variant_config(raw_config, model_cls)
        raise ValueError(f"Unknown config_type: {config_type}")
    except Exception as e:
        raise ValueError(f"Invalid configuration in {config_path}:\n{e}") from e
