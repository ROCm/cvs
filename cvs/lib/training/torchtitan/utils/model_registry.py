'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Model registry for TorchTitan training — all model-specific lookup tables live here.

Adding a new model family: add one entry to each of MODEL_FLAVORS and
PRECISION_FLAGS. No changes to torchtitan_lib.py needed.

TorchTitan uses TOML config files instead of shell scripts, so no training
script lookup is needed (unlike Megatron).
'''

# Model flavor mappings: maps model_name to TorchTitan model flavor
# TorchTitan format: model.name = "llama3", model.flavor = "8B"
MODEL_FLAVORS = {
    'llama3_1_8b': {
        'name': 'llama3',
        'flavor': '8B',
        'module': 'llama3',
        'model_size': '8B',
        'tokenizer_path': 'meta-llama/Llama-3.1-8B',
        'hf_assets_subdir': 'llama3',
    },
    'llama3_1_70b': {
        'name': 'llama3',
        'flavor': '70B',
        'module': 'llama3',
        'model_size': '70B',
        'tokenizer_path': 'meta-llama/Llama-3.1-70B',
        'hf_assets_subdir': 'llama3',
    },
    'llama3_1_405b': {
        'name': 'llama3',
        'flavor': '405B',
        'module': 'llama3',
        'model_size': '405B',
        'tokenizer_path': 'meta-llama/Llama-3.1-405B',
        'hf_assets_subdir': 'llama3',
    },
    'llama3_3_70b': {
        'name': 'llama3',
        'flavor': '70B',
        'module': 'llama3',
        'model_size': '70B',
        'tokenizer_path': 'meta-llama/Llama-3.3-70B-Instruct',
        'hf_assets_subdir': 'llama3',
    },
    'deepseek_v2_lite': {
        'name': 'deepseek',
        'flavor': 'lite',
        'module': 'deepseek',
        'model_size': 'lite',
        'tokenizer_path': 'deepseek-ai/DeepSeek-V2-Lite',
        'hf_assets_subdir': 'deepseek',
    },
    'deepseek_v3_16b': {
        'name': 'deepseek_v3',
        'flavor': '16b',
        'module': 'deepseek_v3',
        'model_size': '16b',
        'tokenizer_path': 'deepseek-ai/DeepSeek-V3',
        'hf_assets_subdir': 'deepseek',
    },
    'qwen3_32b': {
        'name': 'qwen3',
        'flavor': '32B',
        'module': 'qwen3',
        'model_size': '32B',
        'tokenizer_path': 'Qwen/Qwen3-32B',
        'hf_assets_subdir': 'qwen',
    },
    'mixtral_8x22b': {
        'name': 'mixtral',
        'flavor': '8x22B',
        'module': 'mixtral',
        'model_size': '8x22B',
        'tokenizer_path': 'mistralai/Mixtral-8x22B-v0.1',
        'hf_assets_subdir': 'mixtral',
    },
}

# Precision/dtype mappings per precision type
# TorchTitan format: keyed by precision name, returns dtype config
PRECISION_FLAGS = {
    'bf16': {
        'dtype': 'bfloat16',
        'enable_float8': False,
        'converters': {},
    },
    'fp8': {
        'dtype': 'bfloat16',
        'enable_float8': True,
        'converters': {'enable_fsdp_float8_all_gather': True, 'precompute_float8_dynamic_scale_for_fsdp': True},
    },
    'BF16': {
        'dtype': 'bfloat16',
        'enable_float8': False,
        'converters': {},
    },
    'FP8': {
        'dtype': 'bfloat16',
        'enable_float8': True,
        'converters': {'enable_fsdp_float8_all_gather': True, 'precompute_float8_dynamic_scale_for_fsdp': True},
    },
}

# Float8 config flags per precision
# TorchTitan enables float8 via [quantize.linear.float8] section
FLOAT8_CONFIG = {
    'fp8': {
        'enable_fsdp_float8_all_gather': True,
        'precompute_float8_dynamic_scale_for_fsdp': True,
    },
    'bf16': {
        'enable_fsdp_float8_all_gather': False,
        'precompute_float8_dynamic_scale_for_fsdp': False,
    },
}


# TorchTitan model configurations - maps model_name to complete model config
# This is a compatibility layer for torchtitan_lib.py
TORCHTITAN_MODELS = {
    'llama3_1_8b': MODEL_FLAVORS['llama3_1_8b'],
    'llama3_1_70b': MODEL_FLAVORS['llama3_1_70b'],
    'llama3_1_405b': MODEL_FLAVORS['llama3_1_405b'],
    'llama3_3_70b': MODEL_FLAVORS['llama3_3_70b'],
    'deepseek_v2_lite': MODEL_FLAVORS['deepseek_v2_lite'],
    'deepseek_v3_16b': MODEL_FLAVORS['deepseek_v3_16b'],
    'qwen3_32b': MODEL_FLAVORS['qwen3_32b'],
    'mixtral_8x22b': MODEL_FLAVORS['mixtral_8x22b'],
}

# Default training parameters for TorchTitan TOML config generation
DEFAULT_TRAINING_PARAMS = {
    'training_iterations': '10',
    'warmup_steps': '200',
    'lr': '3e-4',
    'activation_checkpointing': 'selective',
    'compile': 'false',
    'dataset': 'c4',
}
