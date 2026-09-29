'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

JAX MaxText run-card hook for JSON deck profiles.
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
    return "single"


def _node_count(variant):
    cluster = getattr(variant, "cluster", None) or {}
    node_dict = cluster.get("node_dict") if isinstance(cluster, dict) else None
    if isinstance(node_dict, dict) and node_dict:
        return str(len(node_dict))
    return None


def jaxmaxtext_run_card_display(variant, provenance):
    rows = [
        ("Model", _model_id(variant), False),
        ("GPU", str(getattr(variant, "gpu_arch", None) or getattr(variant, "gpu_name", None) or "\u2014"), False),
        ("Framework", "JAX MaxText", False),
        ("Mode", _mode(variant), False),
        ("Image", _image_name(variant), False),
    ]
    nodes = _node_count(variant)
    if nodes:
        rows.append(("Nodes", nodes, False))
    rows.append(thresholds_run_card_row(variant))
    rows.extend(provenance_link_rows(provenance or {}))
    return rows
