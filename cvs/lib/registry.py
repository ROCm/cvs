"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

# framework literal -> adapter class. Driver dispatch is the ONLY place that
# picks an adapter by name; everything else is mode-blind.
INFERENCE_REGISTRY: Dict[str, Type] = {}
TRAINING_REGISTRY: Dict[str, Type] = {}

_VALID_KINDS = ("inference", "training")


def register_adapter(framework: str, kind: str = "inference") -> Callable[[Type], Type]:
    """Decorator registering an adapter class under ``framework``.

    ``kind`` selects the inference vs training registry. Registration also
    stamps the class ``framework`` attribute so it matches the config dispatch
    key. Re-registering the same ``framework`` (in either registry) raises
    ``ValueError``: silent overwrites would let an adapter import shadow a
    sibling and we lose the only signal that there is a name collision.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    registry = INFERENCE_REGISTRY if kind == "inference" else TRAINING_REGISTRY

    def _wrap(cls: Type) -> Type:
        if framework in INFERENCE_REGISTRY or framework in TRAINING_REGISTRY:
            raise ValueError(f"adapter already registered for framework {framework!r}")
        cls.framework = framework
        registry[framework] = cls
        return cls

    return _wrap


def get_adapter(framework: str) -> Type:
    """Return the adapter class for ``framework`` (searches both registries).

    Raises ``ValueError`` when no adapter is registered: silent fallback to a
    default would hide a config typo until a real workload run.
    """
    if framework in INFERENCE_REGISTRY:
        return INFERENCE_REGISTRY[framework]
    if framework in TRAINING_REGISTRY:
        return TRAINING_REGISTRY[framework]
    known = ", ".join(sorted(set(INFERENCE_REGISTRY) | set(TRAINING_REGISTRY))) or "<none>"
    raise ValueError(f"no adapter registered for framework {framework!r}; registered: {known}")
