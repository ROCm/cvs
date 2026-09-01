"""Shared Pydantic base classes for CVS config schemas."""

from pydantic import BaseModel, ConfigDict


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Allow(BaseModel):
    model_config = ConfigDict(extra="allow")
