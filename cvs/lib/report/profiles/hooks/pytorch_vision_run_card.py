'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

PyTorch Vision run-card hook for the training Run Deck.
'''

from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row


def _image_name(variant):
    container = getattr(variant, "container", None)
    image = getattr(container, "image", None) if container is not None else None
    if isinstance(container, dict):
        image = container.get("image")
    return str(image or "\u2014")


def _model_id(variant):
    model = getattr(variant, "model", None)
    model_id = getattr(model, "id", None) if model is not None else None
    return str(model_id or "\u2014")


def _mode(variant):
    training = getattr(variant, "training", None)
    if training is not None and getattr(training, "distributed", False):
        return "distributed"
    return "single-node"


def _workload(variant):
    training = getattr(variant, "training", None)
    return str(getattr(training, "workload", None) or "\u2014")


def pytorch_vision_run_card_display(variant, provenance):
    rows = [
        ("Workload", _workload(variant), False),
        ("Model", _model_id(variant), False),
        ("GPU", str(getattr(variant, "gpu_arch", None) or getattr(variant, "gpu_name", None) or "\u2014"), False),
        ("Framework", "PyTorch Vision", False),
        ("Mode", _mode(variant), False),
        ("Image", _image_name(variant), False),
        thresholds_run_card_row(variant),
    ]
    rows.extend(provenance_link_rows(provenance or {}))
    return rows
