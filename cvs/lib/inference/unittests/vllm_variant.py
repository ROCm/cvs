'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Builds a VllmConfig for unit tests from an old-format vLLM config dict.
'''

from cvs.lib.inference.utils.vllm_config_loader import VllmConfig, resolve_old_vllm_keys


def old_vllm_config(**raw):
    """A VllmConfig converted from an old-format config, as load_variant converts one."""
    converted, _message = resolve_old_vllm_keys(raw)
    return VllmConfig.model_validate(converted)
