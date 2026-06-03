"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cvs.lib.config import expand_sweep, parse_config
from cvs.lib.manifest.events import EventWriter
from cvs.lib.manifest.layout import RunLayout
from cvs.lib.run_context import RunContext

BASE = {
    "schema_version": "2",
    "framework": "vllm",
    "target_gpu": "mi300",
    "model": "m",
    "topology": {"roles": {"server": {"count": 1, "gpus_per_node": 8}}},
    "params": {"server_script": "s.sh", "base_url": "http://localhost"},
    "sweep": {
        "concurrency": [16],
        "sequence_combinations": [{"isl": 1024, "osl": 1024, "name": "balanced"}],
    },
}


def _ctx(tmp):
    cfg = parse_config(BASE)
    cell = expand_sweep(cfg.sweep)[0]
    layout = RunLayout(tmp, "t", cell.id, "h", "r")
    layout.ensure()
    return cfg, RunContext(
        config=cfg,
        cell=cell,
        bindings={"server": ["node-a"]},
        layout=layout,
        events=EventWriter(None),
        run_id="r",
    )


class TestRunContextState(unittest.TestCase):
    """RunContext is a pure state struct: construction populates inputs and
    initializes outputs; ``param`` resolves cell-override-then-static."""

    def test_construction_initializes_outputs(self):
        tmp = Path(tempfile.mkdtemp())
        _, ctx = _ctx(tmp)
        # Inputs preserved.
        self.assertEqual(ctx.bindings, {"server": ["node-a"]})
        self.assertEqual(ctx.run_id, "r")
        # Outputs initialized empty (so adapters don't blow up on first write).
        self.assertEqual(ctx.scratch, {})
        self.assertEqual(ctx.logs, {})
        self.assertEqual(ctx.containers, [])
        # ResultView has empty samples/scalars by default.
        self.assertEqual(ctx.result.samples, [])
        self.assertEqual(ctx.result.scalars, {})

    def test_param_resolves_cell_then_static(self):
        tmp = Path(tempfile.mkdtemp())
        _, ctx = _ctx(tmp)
        # Static param (from config.params) wins when not in cell.
        self.assertEqual(ctx.param("server_script"), "s.sh")
        # Cell param wins over static when both present.
        ctx.cell.params["server_script"] = "override.sh"
        self.assertEqual(ctx.param("server_script"), "override.sh")
        # Default when neither has it.
        self.assertEqual(ctx.param("missing", default="d"), "d")


if __name__ == "__main__":
    unittest.main()
