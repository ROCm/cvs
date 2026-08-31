"""Framework-agnostic variant config blocks shared by every CVS suite."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import Field, model_validator
from typing_extensions import Literal

from cvs.schema.base import _Allow, _Forbid


class Paths(_Forbid):
    shared_fs: str
    models_dir: str
    log_dir: str
    hf_token_file: str
    # Host-user-namespaced scratch (jaxmaxtext launchers/yml). Optional so
    # inference configs that omit it still load; jaxmaxtext configs set it to
    # /tmp/{user-id}/jaxmaxtext so container-root /tmp/root is never used.
    temp_dir: str = ""


class ModelSpec(_Forbid):
    id: str
    remote: Literal[0, 1]
    precision: str = ""


class RuntimeSpec(_Allow):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ContainerSpec(_Forbid):
    lifetime: Literal["no_launch", "per_run", "persistent"] = "per_run"
    name: str
    image: str
    runtime: RuntimeSpec


class BaseVariantConfig(_Forbid):
    """The framework-agnostic skeleton of a variant config.

    Carries the fields every suite shares (schema/paths/model/image/container/
    thresholds + the enforce gate) and the remote-not-implemented guard.
    Per-framework subclasses add their own ``framework``/``Params``/``Sweep`` and the
    ``cell_key``/coverage-check pair that depend on them.
    """

    schema_version: Literal[1]
    # When false, the threshold-coverage gate warns instead of raising and the
    # test records metrics without asserting pass/fail (record-only). Use for
    # un-calibrated shapes (e.g. a throughput characterization whose published
    # numbers are curves, not tabulated values). Default true keeps the gate
    # strict for calibrated configs -- no regression to the remediation work.
    enforce_thresholds: bool = True
    threshold_json: str = ""
    paths: Paths
    model: ModelSpec
    # The container image is declared once, on container.image (ContainerSpec).
    # There is no separate top-level image block.
    container: ContainerSpec
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # pydantic runs @model_validator(mode="after") hooks in definition order,
    # parent-class hooks before subclass hooks. This remote check is intentionally
    # the first to run: an unimplemented remote config fails fast
    # (NotImplementedError) before any subclass's threshold-coverage check runs,
    # which is meaningless for a config we are going to reject anyway.
    @model_validator(mode="after")
    def _check_remote_not_implemented(self):
        if self.model.remote == 1:
            raise NotImplementedError(
                "model.remote=1 (remote model download) is not implemented in the PoC. "
                "Port from cvs-dtni-v1/resource_resolver.py before enabling."
            )
        return self
