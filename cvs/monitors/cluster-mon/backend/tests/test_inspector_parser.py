"""
Unit tests for InspectorParser and aggregate_snapshot.
"""

import json
from pathlib import Path

import pytest

from app.collectors.inspector_parser import InspectorParser, aggregate_snapshot
from app.models.rccl_models import InspectorCollPerf, InspectorEventTrace


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "inspector_sample.jsonl"


# ---------------------------------------------------------------------------
# parse_lines
# ---------------------------------------------------------------------------

class TestParseLines:
    def setup_method(self):
        self.parser = InspectorParser()

    def test_parses_valid_allreduce_record(self):
        line = json.dumps({
            "header": {"id": "0xabc", "rank": 0, "n_ranks": 8, "nnodes": 1},
            "metadata": {
                "inspector_output_format_version": "v4.0",
                "git_rev": "deadbeef",
                "rec_mechanism": "nccl_profiler_interface",
                "dump_timestamp_us": 1_711_800_000_000_000,
                "hostname": "gpu-node-01",
                "pid": 9999,
            },
            "coll_perf": {
                "coll": "AllReduce",
                "coll_sn": 1,
                "coll_msg_size_bytes": 1048576,
                "coll_exec_time_us": 200,
                "coll_timing_source": "kernel_gpu",
                "coll_algobw_gbs": 5.0,
                "coll_busbw_gbs": 9.375,
            },
        })
        records = self.parser.parse_lines(line)
        assert len(records) == 1
        r = records[0]
        assert r.comm_hash == "0xabc"
        assert r.rank == 0
        assert r.nranks == 8
        assert r.nnodes == 1
        assert r.hostname == "gpu-node-01"
        assert r.pid == 9999
        assert r.collective == "AllReduce"
        assert r.sequence_num == 1
        assert r.msg_size_bytes == 1048576
        assert r.exec_time_us == 200
        assert r.timing_source == "kernel_gpu"
        assert r.algo_bw_gbps == pytest.approx(5.0)
        assert r.bus_bw_gbps == pytest.approx(9.375)
        assert r.timestamp == pytest.approx(1_711_800_000.0)

    def test_skips_malformed_json_silently(self):
        text = "not json at all\n" + json.dumps({
            "header": {"id": "0x1", "rank": 0, "n_ranks": 2, "nnodes": 1},
            "metadata": {"dump_timestamp_us": 1000000, "hostname": "h", "pid": 1},
            "coll_perf": {"coll": "AllReduce", "coll_sn": 1, "coll_msg_size_bytes": 64,
                          "coll_exec_time_us": 10, "coll_timing_source": "kernel_gpu",
                          "coll_algobw_gbs": 1.0, "coll_busbw_gbs": 1.875},
        })
        records = self.parser.parse_lines(text)
        assert len(records) == 1

    def test_skips_missing_field_silently(self):
        # Missing coll_perf entirely
        line = json.dumps({"header": {"id": "0x1", "rank": 0}, "metadata": {}})
        records = self.parser.parse_lines(line)
        assert records == []

    def test_empty_text_returns_empty(self):
        assert self.parser.parse_lines("") == []

    def test_blank_lines_skipped(self):
        assert self.parser.parse_lines("\n\n   \n") == []

    def test_timestamp_conversion(self):
        """dump_timestamp_us is microseconds; should convert to Unix seconds."""
        line = json.dumps({
            "header": {"id": "0x1", "rank": 0, "n_ranks": 1, "nnodes": 1},
            "metadata": {"dump_timestamp_us": 2_000_000_000_000, "hostname": "h", "pid": 1},
            "coll_perf": {"coll": "AllReduce", "coll_sn": 1, "coll_msg_size_bytes": 64,
                          "coll_exec_time_us": 10, "coll_timing_source": "kernel_gpu",
                          "coll_algobw_gbs": 1.0, "coll_busbw_gbs": 1.875},
        })
        records = self.parser.parse_lines(line)
        assert records[0].timestamp == pytest.approx(2_000_000.0)

    def test_zero_exec_time_record_parsed(self):
        """Zero exec_time (timing fallback) should parse, not be skipped."""
        line = json.dumps({
            "header": {"id": "0x1", "rank": 0, "n_ranks": 1, "nnodes": 1},
            "metadata": {"dump_timestamp_us": 1000000, "hostname": "h", "pid": 1},
            "coll_perf": {"coll": "AllGather", "coll_sn": 5, "coll_msg_size_bytes": 128,
                          "coll_exec_time_us": 0, "coll_timing_source": "collective_cpu",
                          "coll_algobw_gbs": 0.0, "coll_busbw_gbs": 0.0},
        })
        records = self.parser.parse_lines(line)
        assert len(records) == 1
        assert records[0].exec_time_us == 0
        assert records[0].bus_bw_gbps == 0.0

    def test_multiple_valid_lines(self):
        lines = []
        for rank in range(4):
            lines.append(json.dumps({
                "header": {"id": "0xfeed", "rank": rank, "n_ranks": 4, "nnodes": 1},
                "metadata": {"dump_timestamp_us": 5_000_000_000_000, "hostname": f"node{rank}", "pid": rank + 100},
                "coll_perf": {"coll": "ReduceScatter", "coll_sn": 10, "coll_msg_size_bytes": 256,
                              "coll_exec_time_us": 50 + rank * 5, "coll_timing_source": "kernel_gpu",
                              "coll_algobw_gbs": 2.0, "coll_busbw_gbs": 1.875},
            }))
        records = self.parser.parse_lines("\n".join(lines))
        assert len(records) == 4
        assert {r.rank for r in records} == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# parse_file
# ---------------------------------------------------------------------------

class TestParseFile:
    def setup_method(self):
        self.parser = InspectorParser()

    def test_parses_fixture_file(self):
        records = self.parser.parse_file(FIXTURE_PATH)
        # fixture has 5 valid lines + 1 malformed + 1 zero-bw
        # All valid JSON lines should parse (including zero-bw)
        assert len(records) >= 4

    def test_fixture_contains_allreduce_records(self):
        records = self.parser.parse_file(FIXTURE_PATH)
        collectives = {r.collective for r in records}
        assert "AllReduce" in collectives

    def test_fixture_contains_multiple_hosts(self):
        records = self.parser.parse_file(FIXTURE_PATH)
        hosts = {r.hostname for r in records}
        assert len(hosts) >= 2  # gpu-node-01 and gpu-node-02

    def test_missing_file_returns_empty(self):
        records = self.parser.parse_file(Path("/nonexistent/path/inspector.log"))
        assert records == []

    def test_tail_limits_records(self, tmp_path):
        """tail=2 should only parse the last 2 lines."""
        lines = []
        for i in range(10):
            lines.append(json.dumps({
                "header": {"id": "0x1", "rank": i, "n_ranks": 10, "nnodes": 1},
                "metadata": {"dump_timestamp_us": 1000000, "hostname": "h", "pid": i + 1},
                "coll_perf": {"coll": "AllReduce", "coll_sn": i, "coll_msg_size_bytes": 128,
                              "coll_exec_time_us": 10, "coll_timing_source": "kernel_gpu",
                              "coll_algobw_gbs": 1.0, "coll_busbw_gbs": 1.875},
            }))
        log_file = tmp_path / "test.log"
        log_file.write_text("\n".join(lines))
        records = self.parser.parse_file(log_file, tail=2)
        assert len(records) == 2
        assert records[0].rank == 8
        assert records[1].rank == 9


# ---------------------------------------------------------------------------
# aggregate_snapshot
# ---------------------------------------------------------------------------

def _make_record(rank: int, bus_bw: float, collective: str = "AllReduce") -> InspectorCollPerf:
    return InspectorCollPerf(
        timestamp=1711800000.0,
        comm_hash="0xtest",
        rank=rank,
        nranks=4,
        nnodes=1,
        hostname=f"node{rank}",
        pid=1000 + rank,
        collective=collective,
        sequence_num=1,
        msg_size_bytes=1048576,
        exec_time_us=100 if bus_bw > 0 else 0,
        timing_source="kernel_gpu",
        algo_bw_gbps=bus_bw / 1.875,
        bus_bw_gbps=bus_bw,
    )


class TestAggregateSnapshot:
    def test_avg_min_max_computed(self):
        records = [_make_record(i, bw) for i, bw in enumerate([300.0, 350.0, 400.0, 200.0])]
        snap = aggregate_snapshot(records)
        assert snap.avg_bus_bw_gbps == pytest.approx(312.5)
        assert snap.min_bus_bw_gbps == pytest.approx(200.0)
        assert snap.max_bus_bw_gbps == pytest.approx(400.0)

    def test_slowest_rank_identified(self):
        records = [_make_record(i, bw) for i, bw in enumerate([300.0, 200.0, 350.0, 400.0])]
        snap = aggregate_snapshot(records)
        assert snap.slowest_rank == 1  # rank 1 has 200 GB/s

    def test_zero_bw_excluded_from_stats(self):
        """Zero-bw records (exec_time=0) should not affect bandwidth statistics."""
        records = [
            _make_record(0, 300.0),
            _make_record(1, 0.0),   # timing fallback — excluded
            _make_record(2, 400.0),
        ]
        snap = aggregate_snapshot(records)
        assert snap.avg_bus_bw_gbps == pytest.approx(350.0)  # (300+400)/2, not /3

    def test_collective_breakdown_counts(self):
        records = [
            _make_record(0, 300.0, "AllReduce"),
            _make_record(1, 300.0, "AllReduce"),
            _make_record(2, 200.0, "ReduceScatter"),
        ]
        snap = aggregate_snapshot(records)
        assert snap.collective_breakdown["AllReduce"] == 2
        assert snap.collective_breakdown["ReduceScatter"] == 1

    def test_empty_records_returns_valid_snapshot(self):
        snap = aggregate_snapshot([])
        assert snap.avg_bus_bw_gbps is None
        assert snap.min_bus_bw_gbps is None
        assert snap.max_bus_bw_gbps is None
        assert snap.slowest_rank is None
        assert snap.collective_breakdown == {}
        assert snap.records == []

    def test_all_zero_bw_returns_none_stats(self):
        records = [_make_record(i, 0.0) for i in range(3)]
        snap = aggregate_snapshot(records)
        assert snap.avg_bus_bw_gbps is None
        assert snap.slowest_rank is None

    def test_records_deduplicated_to_latest_per_rank(self):
        """records field should contain only the highest sequence_num per (rank, comm_hash)."""
        def _make_seq(rank: int, seq: int, bw: float) -> InspectorCollPerf:
            r = _make_record(rank, bw)
            return r.model_copy(update={"sequence_num": seq})

        records = [
            _make_seq(0, 10, 300.0),
            _make_seq(0, 11, 320.0),  # newer — should win
            _make_seq(1, 10, 280.0),
            _make_seq(1,  9, 260.0),  # older — should lose
        ]
        snap = aggregate_snapshot(records)
        # Only 2 display records (one per rank)
        assert len(snap.records) == 2
        by_rank = {r.rank: r for r in snap.records}
        assert by_rank[0].sequence_num == 11
        assert by_rank[1].sequence_num == 10
        # Stats still computed from all 4 records
        assert snap.avg_bus_bw_gbps == pytest.approx((300 + 320 + 280 + 260) / 4)

    # ------------------------------------------------------------------
    # Verbose / event_trace tests
    # ------------------------------------------------------------------

class TestVerboseParsing:
    def setup_method(self):
        self.parser = InspectorParser()

    def _verbose_line(self, rank=0, n_channels=2) -> str:
        kernel_sn = [
            {"channel_id": ch, "kernel_start_sn": 100 + ch, "kernel_stop_sn": 110 + ch, "kernel_record_sn": 111 + ch}
            for ch in range(n_channels)
        ]
        kernel_ts = [
            {"channel_id": ch, "kernel_start_ts": 1000 + ch * 100, "kernel_stop_ts": 1050 + ch * 100, "kernel_record_ts": 1051 + ch * 100}
            for ch in range(n_channels)
        ]
        return json.dumps({
            "header": {"id": "0xverb", "rank": rank, "n_ranks": 4, "nnodes": 1},
            "metadata": {"dump_timestamp_us": 2_000_000_000_000, "hostname": f"node{rank}", "pid": rank + 1},
            "coll_perf": {
                "coll": "AllReduce", "coll_sn": 1, "coll_msg_size_bytes": 1048576,
                "coll_exec_time_us": 200, "coll_timing_source": "kernel_gpu",
                "coll_algobw_gbs": 5.0, "coll_busbw_gbs": 9.375,
                "event_trace_sn": {"coll_start_sn": 99, "coll_stop_sn": 120, "kernel_events": kernel_sn},
                "event_trace_ts": {"coll_start_ts": 900, "coll_stop_ts": 1200, "kernel_events": kernel_ts},
            },
        })

    def test_non_verbose_record_has_no_event_trace(self):
        line = json.dumps({
            "header": {"id": "0x1", "rank": 0, "n_ranks": 1, "nnodes": 1},
            "metadata": {"dump_timestamp_us": 1000000, "hostname": "h", "pid": 1},
            "coll_perf": {"coll": "AllReduce", "coll_sn": 1, "coll_msg_size_bytes": 64,
                          "coll_exec_time_us": 10, "coll_timing_source": "kernel_gpu",
                          "coll_algobw_gbs": 1.0, "coll_busbw_gbs": 1.875},
        })
        records = self.parser.parse_lines(line)
        assert records[0].event_trace is None

    def test_verbose_record_has_event_trace(self):
        records = self.parser.parse_lines(self._verbose_line())
        assert records[0].event_trace is not None

    def test_verbose_coll_level_sn_and_ts(self):
        records = self.parser.parse_lines(self._verbose_line())
        et = records[0].event_trace
        assert et.coll_start_sn == 99
        assert et.coll_stop_sn == 120
        assert et.coll_start_ts == 900
        assert et.coll_stop_ts == 1200

    def test_verbose_channel_count(self):
        records = self.parser.parse_lines(self._verbose_line(n_channels=3))
        assert len(records[0].event_trace.channels) == 3

    def test_verbose_channel_fields(self):
        records = self.parser.parse_lines(self._verbose_line(n_channels=2))
        ch0 = records[0].event_trace.channels[0]
        assert ch0.channel_id == 0
        assert ch0.kernel_start_sn == 100
        assert ch0.kernel_stop_sn == 110
        assert ch0.kernel_record_sn == 111
        assert ch0.kernel_start_ts == 1000
        assert ch0.kernel_stop_ts == 1050
        assert ch0.kernel_record_ts == 1051

    def test_verbose_channels_sorted_by_channel_id(self):
        records = self.parser.parse_lines(self._verbose_line(n_channels=4))
        ids = [ch.channel_id for ch in records[0].event_trace.channels]
        assert ids == sorted(ids)

    def test_fixture_contains_verbose_record(self):
        records = InspectorParser().parse_file(FIXTURE_PATH)
        verbose = [r for r in records if r.event_trace is not None]
        assert len(verbose) >= 1
        assert verbose[0].event_trace.channels[0].channel_id == 0
        assert verbose[0].event_trace.channels[1].channel_id == 1

    def test_malformed_event_trace_falls_back_to_none(self):
        """Malformed event_trace content should not fail the whole record."""
        line = json.dumps({
            "header": {"id": "0x1", "rank": 0, "n_ranks": 1, "nnodes": 1},
            "metadata": {"dump_timestamp_us": 1000000, "hostname": "h", "pid": 1},
            "coll_perf": {
                "coll": "AllReduce", "coll_sn": 1, "coll_msg_size_bytes": 64,
                "coll_exec_time_us": 10, "coll_timing_source": "kernel_gpu",
                "coll_algobw_gbs": 1.0, "coll_busbw_gbs": 1.875,
                "event_trace_sn": "not_an_object",   # malformed
            },
        })
        records = self.parser.parse_lines(line)
        assert len(records) == 1
        assert records[0].event_trace is None


    def test_collective_breakdown_uses_all_records(self):
        """collective_breakdown counts ALL records, not just deduplicated ones."""
        def _make_seq(rank: int, seq: int, coll: str) -> InspectorCollPerf:
            r = _make_record(rank, 300.0, coll)
            return r.model_copy(update={"sequence_num": seq})

        records = [
            _make_seq(0, 1, "AllReduce"),
            _make_seq(0, 2, "AllReduce"),  # same rank, newer seq
            _make_seq(1, 1, "ReduceScatter"),
        ]
        snap = aggregate_snapshot(records)
        assert snap.collective_breakdown["AllReduce"] == 2
        assert snap.collective_breakdown["ReduceScatter"] == 1
        assert len(snap.records) == 2  # deduplicated
