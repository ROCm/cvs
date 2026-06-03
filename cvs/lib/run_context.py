"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cvs.lib.config.thresholds import ResultView
from cvs.lib.manifest.events import EventWriter
from cvs.lib.manifest.layout import RunLayout


class RunContext:
    """Mutable state threaded through one workload run's lifecycle.

    The ``Job`` driver constructs this once per (config, sweep cell) and
    passes it to every adapter method. Adapters read inputs (config, cell
    params, bindings, executor) and write outputs (``result``, ``logs``,
    registered container handles) onto it -- the on-disk/in-memory
    replacement for the v1 module-level globals.

    A1 staging seam DEFERRED. Real CVS clusters use either a shared
    filesystem (Weka/NFS at the same path on devbox + nodes) or an
    end-of-run rsync, so per-file ``stage_in``/``fetch`` is not needed
    for the integration milestone. Add the SFTP wrappers (and a
    ``RunLayout.to_remote``-keyed re-base) when the first cluster
    without a shared FS lands. The G3 ``remote_artifact_dir`` /
    ``remote_root`` / ``to_remote`` plumbing is already in place, so
    this is a pure addition when needed.

    A2 per-role executors also deferred (addendum §3). The single
    ``executor`` slot is present for the (eventual) multi-host
    case; adapters that need to issue commands today receive it via
    constructor injection.
    """

    def __init__(
        self,
        config: Any,
        cell: Any,
        bindings: Dict[str, List[str]],
        layout: RunLayout,
        events: EventWriter,
        run_id: str,
        executor: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.cell = cell
        self.bindings = bindings
        self.layout = layout
        self.events = events
        self.run_id = run_id
        self.executor = executor
        self.result = ResultView()
        self.logs: Dict[str, str] = {}
        self.containers: List[Any] = []
        self.scratch: Dict[str, Any] = {}

    def param(self, name: str, default: Any = None) -> Any:
        """Resolve a parameter: sweep-cell override first, then static params."""
        cell_params = getattr(self.cell, "params", None)
        if isinstance(cell_params, dict) and name in cell_params:
            return cell_params[name]
        params = getattr(self.config, "params", None)
        if params is not None and hasattr(params, name):
            return getattr(params, name)
        return default
