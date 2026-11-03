'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Megatron run-card hook for JSON deck profiles.
'''

from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row


def _image_name(variant):
    container = getattr(variant, "container", None)
    image = getattr(container, "image", None) if container is not None else None
    if isinstance(container, dict):
        image = container.get("image")
    return str(image or "—")


def _framework_label(variant):
    image = _image_name(variant).lower()
    if "primus" in image:
        return "Primus"
    return "Megatron-LM"


def _train_param(variant, key, default="—"):
    params = getattr(variant, "train_params", None) or {}
    if isinstance(params, dict):
        value = params.get(key)
    else:
        value = getattr(params, key, None)
    if value in (None, ""):
        return default
    return str(value)


def megatron_run_card_display(variant, provenance):
    rows = [
        ("Model", _train_param(variant, "tokenizer_model", _train_param(variant, "model_name")), False),
        ("GPU", str(getattr(variant, "gpu_arch", None) or getattr(variant, "gpu_name", None) or "—"), False),
        ("Framework", _framework_label(variant), False),
        ("Image", _image_name(variant), False),
        ("TP", _train_param(variant, "tensor_parallelism", "1"), False),
        ("PP", _train_param(variant, "pipeline_parallelism", "1"), False),
        thresholds_run_card_row(variant),
    ]
    rows.extend(provenance_link_rows(provenance or {}))
    return rows
