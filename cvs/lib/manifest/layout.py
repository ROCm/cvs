"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

_PathLike = Union[str, "Path"]


class RunLayout:
    """Content-addressable per-run directory layout.

    ``<artifact_dir>/<test_id>/<cell_id>/<short_hash>/<run_id>/``

    The ``short_hash`` component (derived from the workload hash) makes the path
    content-addressable, which is what lets v2.A reuse-manifests find a prior
    identical run without any historical migration.

    A1 (staging seam): the harness may run commands on a *remote* node while
    artifacts are read/written on a *local* layout. ``remote_artifact_dir``
    gives the same content-addressable tree a mirror root on the remote side so
    G5's ``RunContext.stage_in``/``fetch`` has a deterministic place to put and
    retrieve files. It is purely a path-builder here -- no I/O, no executor.
    ``remote_artifact_dir`` is supplied by G5's run setup; until then it is
    ``None`` and ``remote_root`` is ``None`` (single-host runs never stage).
    """

    MANIFEST = "manifest.json"
    SAMPLES = "samples.parquet"
    TRAJECTORY = "trajectory.parquet"
    EVENTS = "events.jsonl"
    RESOLVED_CONFIG = "config.resolved.yaml"
    LOGS = "logs"

    def __init__(
        self,
        artifact_dir: _PathLike,
        test_id: str,
        cell_id: str,
        short_hash: str,
        run_id: str,
        remote_artifact_dir: Optional[_PathLike] = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.test_id = test_id
        self.cell_id = cell_id
        self.short_hash = short_hash
        self.run_id = run_id
        self.remote_artifact_dir = Path(remote_artifact_dir) if remote_artifact_dir is not None else None

    @property
    def _relative(self) -> Path:
        """The content-addressable path relative to either artifact root."""
        return Path(self.test_id) / self.cell_id / self.short_hash / self.run_id

    @property
    def root(self) -> Path:
        return self.artifact_dir / self._relative

    @property
    def manifest_path(self) -> Path:
        return self.root / self.MANIFEST

    @property
    def samples_path(self) -> Path:
        return self.root / self.SAMPLES

    @property
    def trajectory_path(self) -> Path:
        return self.root / self.TRAJECTORY

    @property
    def events_path(self) -> Path:
        return self.root / self.EVENTS

    @property
    def resolved_config_path(self) -> Path:
        return self.root / self.RESOLVED_CONFIG

    @property
    def logs_dir(self) -> Path:
        return self.root / self.LOGS

    @property
    def remote_root(self) -> Optional[Path]:
        """Mirror of ``root`` on the remote node, or ``None`` if not staged.

        Uses the identical content-addressable suffix so a file at
        ``root / x`` corresponds to ``remote_root / x`` (see :meth:`to_remote`).
        """
        if self.remote_artifact_dir is None:
            return None
        return self.remote_artifact_dir / self._relative

    def to_remote(self, local_path: _PathLike) -> Path:
        """Re-base a path under ``root`` to the matching path under ``remote_root``.

        The A1 seam is keyed on the *local* path in both directions: ``stage_in``
        uploads ``local -> to_remote(local)`` and ``fetch`` downloads
        ``to_remote(local) -> local``. The mapping is bijective (the remote tree
        reuses the identical content-addressable suffix), so no separate
        remote->local inverse is needed.

        Fail-closed with two *distinct* ``ValueError`` messages so a caller can
        tell them apart: no remote root configured (staging not set up) vs
        ``local_path`` is not under ``root`` (a caller bug).
        """
        remote = self.remote_root
        if remote is None:
            raise ValueError("RunLayout has no remote_artifact_dir; cannot map a path to the remote root")
        try:
            relative = Path(local_path).relative_to(self.root)
        except ValueError:
            raise ValueError(
                f"path {str(local_path)!r} is not under RunLayout.root {str(self.root)!r}; cannot map to remote"
            ) from None
        return remote / relative

    def ensure(self) -> "RunLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self
