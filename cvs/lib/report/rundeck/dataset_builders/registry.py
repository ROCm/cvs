'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Registry for Run Deck dataset builders (``sweep``, ``series``, ``matrix``).

Builders are the Python extension point for new data shapes. Card types consume
builder output; they are not builders themselves.
'''

from __future__ import annotations

from typing import Any, Callable

_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {}


def register_dataset_builder(builder_id: str):
    """Decorator to register a dataset builder by id."""

    def decorator(fn: Callable[..., dict[str, Any]]):
        _BUILDERS[builder_id] = fn
        return fn

    return decorator


def get_dataset_builder(builder_id: str) -> Callable[..., dict[str, Any]] | None:
    return _BUILDERS.get(builder_id)


def build_datasets(
    builder_id: str,
    sources: dict[str, Any],
    profile: Any,
) -> dict[str, Any]:
    """Run the named builder; returns empty dict when builder is not registered."""
    builder = get_dataset_builder(builder_id)
    if builder is None:
        return {}
    return builder(sources, profile)
