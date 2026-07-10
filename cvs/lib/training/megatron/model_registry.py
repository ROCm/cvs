'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Model registry for Megatron training — all model-specific lookup tables live here.

Adding a new model family: add one entry to each of TRAINING_SCRIPTS,
PRECISION_FLAGS, and BATCH_SIZE_FLAGS. No changes to megatron_lib.py needed.

Key ordering note: qwen3 must appear before qwen2 in every dict so that
tokenizer names containing "Qwen3" do not match the "qwen2" pattern first.
'''

# Training script paths relative to megatron_root, keyed by tokenizer family regex.
TRAINING_SCRIPTS = {
    'llama-3':  'examples/llama/train_llama3.sh',
    'llama-2':  'examples/llama/train_llama2.sh',
    'deepseek': 'examples/deepseek_v2/train_deepseekv2.sh',
    'mixtral':  'examples/mixtral/train_mixtral_moe.sh',
    'qwen3':    'examples/qwen3/train_qwen3.sh',
    'qwen2':    'examples/qwen2/train_qwen2.sh',
}

# Precision env var strings per model family and precision name.
PRECISION_FLAGS = {
    'llama': {
        'TE_FP8':  'TE_FP8=1',
        'TE_BF16': 'TE_FP8=0 TE_BF16=1 TE_FP4=0',
        'MXFP4':   'TE_FP4=1 TE_FP4_RECIPE=mxfp4',
        'MXFP8':   'TE_FP8=1 TE_FP8_RECIPE=mxfp8',
    },
    'deepseek': {
        'FP8':  'PR=fp8',
        'FP16': 'PR=fp16',
        'BF16': 'PR=bf16',
    },
    'qwen3': {
        'FP8':   'PR=fp8',
        'BF16':  'PR=bf16',
        'MXFP8': 'PR=fp8 FP8_RECIPE=mxfp8',
    },
    'qwen2': {
        'FP8':  'TE_FP8=1',
        'BF16': 'TE_FP8=0',
    },
    'mixtral': {
        'FP8':  'PR=fp8',
        'FP16': 'PR=fp16',
        'BF16': 'PR=bf16',
    },
}

# Batch size env var names per model family.
# mbs → micro batch size env var name, gbs → global batch size env var name.
BATCH_SIZE_FLAGS = {
    'llama':    {'mbs': 'MBS',              'gbs': 'BS'},
    'deepseek': {'mbs': 'MBS',              'gbs': 'GBS'},
    'qwen3':    {'mbs': 'MICRO_BATCH_SIZE', 'gbs': 'GLOBAL_BATCH_SIZE'},
    'qwen2':    {'mbs': 'MBS',              'gbs': 'BS'},
    'mixtral':  {'mbs': 'MBS',              'gbs': 'GBS'},
}
