'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

JAX MaxText run-card hook shared by single-node and distributed suites.
'''


def jaxmaxtext_run_card_display(variant, _provenance):
    return [
        ("Model", getattr(getattr(variant, "model", None), "id", "\u2014"), False),
        ("GPU arch", getattr(variant, "gpu_arch", "\u2014"), False),
        ("nnodes", str(getattr(variant, "nnodes", "\u2014")), False),
    ]


__all__ = ["jaxmaxtext_run_card_display"]
