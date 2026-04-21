"""
Optional `runtime` block parser for cluster.json (CVS docker-mode P3).

This module reads the optional `runtime` key inside a parsed `cluster.json`
dict and returns a typed `RuntimeConfig` describing how CVS should execute
test commands on the cluster.

By design this is *additive*: cluster.json files that do NOT have a `runtime`
block produce a RuntimeConfig with `mode == "host"`, which is the historical
default behavior. P4 onward consumes this object to decide whether to wrap
commands via docker exec; in P3 itself nothing else in CVS reads this yet.

Schema (v1, single-arch cluster assumption):

    "runtime": {
        "mode": "host" | "docker",
        "image": "<tag>",                    # required when mode == "docker"
        "container_name": "cvs-runner",      # default
        "container_envs": {},                # default
        "ensure_image": "pull"               # default; or "load:<tarball>"
        "agfhc_tarball": null,               # default
        "expected_gfx_arch": null,           # default
        "installs": null                     # default; null resolves to a
                                             # standard install list at use
                                             # time (see DEFAULT_INSTALLS)
    }

Validation errors are raised as `RuntimeConfigError` with a single clear
sentence; callers (P4 fixtures, P6 prepare_runtime) should let them
propagate so the user sees the message immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# Components that runtime.installs == null resolves to. Order matters:
# install_rvs first (longest), install_transferbench second, install_agfhc
# only added if a tarball is supplied. install_ibperf_tools is intentionally
# excluded from v1 defaults because the upstream test discovery has a
# pre-existing bug; users can still opt in by listing it explicitly.
DEFAULT_INSTALLS_NO_AGFHC: Sequence[str] = (
    "install_rvs",
    "install_transferbench",
)
DEFAULT_INSTALLS_WITH_AGFHC: Sequence[str] = (
    "install_rvs",
    "install_transferbench",
    "install_agfhc",
)

VALID_MODES = ("host", "docker")
VALID_EXCLUSIVITY = ("warn", "strict")


class RuntimeConfigError(ValueError):
    """Raised when the `runtime` block in cluster.json is malformed."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Parsed `runtime` block from cluster.json.

    `mode == "host"` is the historical default and is what every existing
    cluster.json file (no `runtime` block) maps to. All other fields are
    only meaningful when `mode == "docker"`.
    """

    mode: str = "host"
    image: Optional[str] = None
    container_name: str = "cvs-runner"
    container_envs: Dict[str, str] = field(default_factory=dict)
    ensure_image: str = "pull"
    agfhc_tarball: Optional[str] = None
    expected_gfx_arch: Optional[str] = None
    installs: Optional[List[str]] = None  # None => resolve via resolved_installs()
    exclusivity: str = "warn"  # "warn" (default) or "strict"
    allowed_containers: List[str] = field(default_factory=list)
    sanitize_host: bool = False  # P11: opt-in host tuning (governor, drop_caches, perflevel)

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------
    def is_docker(self) -> bool:
        """True iff this cluster is configured to run tests inside containers."""
        return self.mode == "docker"

    def resolved_installs(self) -> List[str]:
        """Return the list of install_* scripts to run at prepare_runtime.

        If `installs` is None (the unset default), resolve to the standard
        list (with install_agfhc added when an AGFHC tarball is supplied).
        If `installs` is an empty list, return an empty list (the user has
        opted out of runtime installs, e.g. they pre-bake their own image).
        Otherwise return the explicit list as-is.
        """
        if self.installs is None:
            base = (
                DEFAULT_INSTALLS_WITH_AGFHC
                if self.agfhc_tarball
                else DEFAULT_INSTALLS_NO_AGFHC
            )
            return list(base)
        return list(self.installs)


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------
def parse_runtime(cluster_dict: Dict[str, Any]) -> RuntimeConfig:
    """Parse the optional `runtime` block from a cluster_dict.

    Args:
        cluster_dict: parsed cluster.json (the dict with `node_dict`,
            `username`, etc.).

    Returns:
        RuntimeConfig with defaults filled in. Always returns a value;
        absent block ⇒ host-mode RuntimeConfig.

    Raises:
        RuntimeConfigError if the `runtime` block is present but invalid.
    """
    raw = cluster_dict.get("runtime") if isinstance(cluster_dict, dict) else None

    # Absent block => host mode, all defaults.
    if raw is None:
        return RuntimeConfig()

    if not isinstance(raw, dict):
        raise RuntimeConfigError(
            f"cluster.json `runtime` must be an object, got {type(raw).__name__}"
        )

    mode = raw.get("mode", "host")
    if mode not in VALID_MODES:
        raise RuntimeConfigError(
            f"cluster.json `runtime.mode` must be one of {VALID_MODES}, got {mode!r}"
        )

    # Host mode: ignore other docker-only fields with permissive defaults so
    # users can leave a stub `runtime: {mode: "host"}` block without surprise.
    if mode == "host":
        return RuntimeConfig(mode="host")

    # Docker-mode validation.
    image = raw.get("image")
    if not isinstance(image, str) or not image.strip():
        raise RuntimeConfigError(
            "cluster.json `runtime.image` is required and must be a non-empty string when mode == 'docker'"
        )

    container_name = raw.get("container_name", "cvs-runner")
    if not isinstance(container_name, str) or not container_name.strip():
        raise RuntimeConfigError(
            "cluster.json `runtime.container_name` must be a non-empty string"
        )

    container_envs = raw.get("container_envs", {})
    if not isinstance(container_envs, dict):
        raise RuntimeConfigError(
            f"cluster.json `runtime.container_envs` must be an object, got {type(container_envs).__name__}"
        )
    for k, v in container_envs.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise RuntimeConfigError(
                "cluster.json `runtime.container_envs` must map string keys to string values"
            )

    ensure_image = raw.get("ensure_image", "pull")
    if not isinstance(ensure_image, str) or not (
        ensure_image == "pull" or ensure_image.startswith("load:")
    ):
        raise RuntimeConfigError(
            "cluster.json `runtime.ensure_image` must be 'pull' or 'load:<path-to-tarball>'"
        )

    agfhc_tarball = raw.get("agfhc_tarball")
    if agfhc_tarball is not None and (
        not isinstance(agfhc_tarball, str) or not agfhc_tarball.strip()
    ):
        raise RuntimeConfigError(
            "cluster.json `runtime.agfhc_tarball` must be a non-empty string or null"
        )

    expected_gfx_arch = raw.get("expected_gfx_arch")
    if expected_gfx_arch is not None and (
        not isinstance(expected_gfx_arch, str) or not expected_gfx_arch.strip()
    ):
        raise RuntimeConfigError(
            "cluster.json `runtime.expected_gfx_arch` must be a non-empty string (e.g. 'gfx942') or null"
        )

    installs = raw.get("installs")
    if installs is not None:
        if not isinstance(installs, list):
            raise RuntimeConfigError(
                "cluster.json `runtime.installs` must be a list of install-script names or null"
            )
        for entry in installs:
            if not isinstance(entry, str) or not entry.startswith("install_"):
                raise RuntimeConfigError(
                    "cluster.json `runtime.installs` entries must be strings starting with 'install_'"
                )

    exclusivity = raw.get("exclusivity", "warn")
    if exclusivity not in VALID_EXCLUSIVITY:
        raise RuntimeConfigError(
            f"cluster.json `runtime.exclusivity` must be one of {VALID_EXCLUSIVITY}, got {exclusivity!r}"
        )

    allowed_containers = raw.get("allowed_containers", [])
    if not isinstance(allowed_containers, list) or not all(
        isinstance(x, str) for x in allowed_containers
    ):
        raise RuntimeConfigError(
            "cluster.json `runtime.allowed_containers` must be a list of strings"
        )

    sanitize_host = raw.get("sanitize_host", False)
    if not isinstance(sanitize_host, bool):
        raise RuntimeConfigError(
            "cluster.json `runtime.sanitize_host` must be a boolean"
        )

    return RuntimeConfig(
        mode="docker",
        image=image,
        container_name=container_name,
        container_envs=dict(container_envs),
        ensure_image=ensure_image,
        agfhc_tarball=agfhc_tarball,
        expected_gfx_arch=expected_gfx_arch,
        installs=list(installs) if installs is not None else None,
        exclusivity=exclusivity,
        allowed_containers=list(allowed_containers),
        sanitize_host=sanitize_host,
    )
