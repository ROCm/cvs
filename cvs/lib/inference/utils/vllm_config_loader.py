'''Load a CVS vLLM config. The schema lives in ``vllm_schema``.'''

from cvs.lib.inference.utils.vllm_schema import VariantConfig, serialize_cli_options  # noqa: F401 - re-exported
from cvs.lib.utils.config_loader import load_config


def load_variant(config_path, cluster):
    """Load a vLLM config and its threshold file against a cluster file path or dict.

    Raises ``ConfigError`` listing every invalid field, including each
    ``<changeme>`` placeholder still in the config.
    """
    return load_config(VariantConfig, config_path, cluster).config
