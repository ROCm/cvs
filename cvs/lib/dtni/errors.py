"""DTNI v1 error model.

Single exception type with a `failed_phase` tag. Replaces dtni's 5-category
taxonomy (per spec: only `failed_phase` + message).
"""

from __future__ import annotations


class WorkloadError(Exception):
    """Raised on any failure inside Job.run().

    `phase` is set by Job at the raise site (one of:
    prepare/launch/await/parse/verify/teardown). `message` is human-readable.
    """

    def __init__(self, message: str, *, phase: str | None = None) -> None:
        super().__init__(message)
        self.phase = phase
        self.message = message


class ConfigError(WorkloadError):
    """Config-file validation/loader error. Raised before Job starts; phase=None."""


class CatalogError(ConfigError):
    """Unknown catalog id (model/dataset/benchmark). Includes 'did you mean'."""
