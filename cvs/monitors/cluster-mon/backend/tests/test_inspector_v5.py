"""
Tests for Inspector v5.0 field awareness (P2.5):
- graphCaptured field parsing
- inspector_format_version field parsing
- v4.0 records produce graph_captured=None (backward compatibility)
- Secondary oracle: inspector_v5 set on NodeRCCLCapability
- Cross-check warning: inspector_v5=True + json_ras=False
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock


from app.collectors.inspector_parser import InspectorParser
from app.collectors.inspector_collector import InspectorCollector
from app.models.rccl_models import NodeRCCLCapability


V5_FIXTURE = Path(__file__).parent / "fixtures" / "inspector_v5_sample.jsonl"
V4_FIXTURE = Path(__file__).parent / "fixtures" / "inspector_sample.jsonl"


def _make_v5_line(
    rank: int = 0,
    graph_captured: bool = True,
    hostname: str = "gpu-node-01",
    pid: int = 12345,
    fmt_version: str = "v5.0",
) -> str:
    return json.dumps(
        {
            "header": {"id": "0xabc123", "rank": rank, "n_ranks": 8, "nnodes": 1},
            "metadata": {
                "inspector_output_format_version": fmt_version,
                "git_rev": "def456",
                "rec_mechanism": "nccl_profiler_interface",
                "dump_timestamp_us": 1_716_700_000_000_000,
                "hostname": hostname,
                "pid": pid,
            },
            "coll_perf": {
                "coll": "AllReduce",
                "coll_sn": 100,
                "coll_msg_size_bytes": 2097152,
                "coll_exec_time_us": 400,
                "coll_timing_source": "kernel_gpu",
                "coll_algobw_gbs": 200.0,
                "coll_busbw_gbs": 400.0,
                "graphCaptured": graph_captured,
            },
        }
    )


def _make_v4_line(rank: int = 0, hostname: str = "gpu-node-01") -> str:
    return json.dumps(
        {
            "header": {"id": "0xabc123", "rank": rank, "n_ranks": 8, "nnodes": 1},
            "metadata": {
                "inspector_output_format_version": "v4.0",
                "git_rev": "abc123",
                "rec_mechanism": "nccl_profiler_interface",
                "dump_timestamp_us": 1_711_800_000_000_000,
                "hostname": hostname,
                "pid": 9999,
            },
            "coll_perf": {
                "coll": "AllReduce",
                "coll_sn": 50,
                "coll_msg_size_bytes": 1048576,
                "coll_exec_time_us": 200,
                "coll_timing_source": "kernel_gpu",
                "coll_algobw_gbs": 100.0,
                "coll_busbw_gbs": 200.0,
            },
        }
    )


# ---------------------------------------------------------------------------
# Parser field extraction
# ---------------------------------------------------------------------------


class TestV5FieldParsing:
    def setup_method(self):
        self.parser = InspectorParser()

    def test_v5_graph_captured_true(self):
        records = self.parser.parse_lines(_make_v5_line(graph_captured=True))
        assert len(records) == 1
        assert records[0].graph_captured is True

    def test_v5_graph_captured_false(self):
        records = self.parser.parse_lines(_make_v5_line(graph_captured=False))
        assert len(records) == 1
        assert records[0].graph_captured is False

    def test_v5_format_version_field(self):
        records = self.parser.parse_lines(_make_v5_line())
        assert records[0].inspector_format_version == "v5.0"

    def test_v4_graph_captured_is_none(self):
        """v4.0 records have no graphCaptured — must return None, not False."""
        records = self.parser.parse_lines(_make_v4_line())
        assert len(records) == 1
        assert records[0].graph_captured is None

    def test_v4_format_version_field(self):
        records = self.parser.parse_lines(_make_v4_line())
        assert records[0].inspector_format_version == "v4.0"

    def test_missing_format_version_defaults_to_v4(self):
        """metadata without inspector_output_format_version should default to v4.0."""
        raw = json.loads(_make_v4_line())
        del raw["metadata"]["inspector_output_format_version"]
        records = self.parser.parse_lines(json.dumps(raw))
        assert records[0].inspector_format_version == "v4.0"

    def test_v5_fixture_file(self):
        """All records in the v5 fixture should have graph_captured set."""
        records = self.parser.parse_lines(V5_FIXTURE.read_text())
        assert len(records) == 4
        for r in records:
            assert r.graph_captured is not None
            assert r.inspector_format_version == "v5.0"

    def test_v4_fixture_file_graph_captured_none(self):
        """v4 fixture records must all have graph_captured=None."""
        records = self.parser.parse_lines(V4_FIXTURE.read_text())
        v4_records = [r for r in records if r.inspector_format_version == "v4.0"]
        assert len(v4_records) > 0
        for r in v4_records:
            assert r.graph_captured is None


# ---------------------------------------------------------------------------
# Secondary oracle: _apply_inspector_v5_oracle
# ---------------------------------------------------------------------------


def _cap(json_ras: bool = True, inspector_v5: bool = False) -> NodeRCCLCapability:
    return NodeRCCLCapability(
        json_ras=json_ras,
        detected_rccl_version=None,
        detection_method="probe",
        inspector_v5=inspector_v5,
    )


class TestInspectorV5Oracle:
    def setup_method(self):
        self.collector = InspectorCollector()

    def _parse(self, text: str):
        parser = InspectorParser()
        return parser.parse_lines(text)

    def test_oracle_sets_inspector_v5_on_capability(self):
        cap = _cap(json_ras=True, inspector_v5=False)
        app_state = MagicMock()
        app_state.node_capabilities = {"gpu-node-01": cap}

        records = self._parse(_make_v5_line(hostname="gpu-node-01"))
        self.collector._apply_inspector_v5_oracle(records, app_state)

        assert cap.inspector_v5 is True

    def test_oracle_no_change_for_v4_records(self):
        cap = _cap(json_ras=True, inspector_v5=False)
        app_state = MagicMock()
        app_state.node_capabilities = {"gpu-node-01": cap}

        records = self._parse(_make_v4_line(hostname="gpu-node-01"))
        self.collector._apply_inspector_v5_oracle(records, app_state)

        assert cap.inspector_v5 is False

    def test_oracle_already_v5_no_duplicate_log(self, caplog):
        cap = _cap(json_ras=True, inspector_v5=True)
        app_state = MagicMock()
        app_state.node_capabilities = {"gpu-node-01": cap}

        records = self._parse(_make_v5_line(hostname="gpu-node-01"))
        with caplog.at_level(logging.INFO, logger="app.collectors.inspector_collector"):
            self.collector._apply_inspector_v5_oracle(records, app_state)

        # Already True — no new INFO log should fire
        info_logs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 0

    def test_oracle_unknown_hostname_no_crash(self):
        """Nodes not yet in node_capabilities must be silently skipped."""
        app_state = MagicMock()
        app_state.node_capabilities = {}

        records = self._parse(_make_v5_line(hostname="unknown-host"))
        # Should not raise
        self.collector._apply_inspector_v5_oracle(records, app_state)

    def test_oracle_emits_warning_when_json_ras_false(self, caplog):
        """inspector_v5=True + json_ras=False → warning about mismatch."""
        cap = _cap(json_ras=False, inspector_v5=False)
        app_state = MagicMock()
        app_state.node_capabilities = {"gpu-node-01": cap}

        records = self._parse(_make_v5_line(hostname="gpu-node-01"))
        with caplog.at_level(logging.WARNING, logger="app.collectors.inspector_collector"):
            self.collector._apply_inspector_v5_oracle(records, app_state)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "mid-upgrade" in warnings[0].message or "json_ras=False" in warnings[0].message

    def test_oracle_no_warning_when_json_ras_true(self, caplog):
        cap = _cap(json_ras=True, inspector_v5=False)
        app_state = MagicMock()
        app_state.node_capabilities = {"gpu-node-01": cap}

        records = self._parse(_make_v5_line(hostname="gpu-node-01"))
        with caplog.at_level(logging.WARNING, logger="app.collectors.inspector_collector"):
            self.collector._apply_inspector_v5_oracle(records, app_state)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 0

    def test_oracle_multiple_nodes(self):
        cap1 = _cap(json_ras=True, inspector_v5=False)
        cap2 = _cap(json_ras=True, inspector_v5=False)
        app_state = MagicMock()
        app_state.node_capabilities = {"gpu-node-01": cap1, "gpu-node-02": cap2}

        lines = _make_v5_line(hostname="gpu-node-01") + "\n" + _make_v5_line(hostname="gpu-node-02")
        records = self._parse(lines)
        self.collector._apply_inspector_v5_oracle(records, app_state)

        assert cap1.inspector_v5 is True
        assert cap2.inspector_v5 is True

    def test_oracle_node_capabilities_missing_no_crash(self):
        """app_state without node_capabilities attribute must not crash."""
        app_state = MagicMock(spec=[])  # no attributes

        records = self._parse(_make_v5_line(hostname="gpu-node-01"))
        self.collector._apply_inspector_v5_oracle(records, app_state)
