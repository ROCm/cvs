"""
Tests for RCCL RAS JSON output parser (RCCL v2.28.7+ with -f json).
Fixtures are synthetic but schema-faithful to client_support.cc jsonWrite* output.
"""

import pytest
from pathlib import Path

from app.collectors.rccl_json_parser import RCCLJsonParser
from app.models.rccl_models import RCCLJobState, NCCLFunction

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def parser():
    return RCCLJsonParser()


@pytest.fixture
def healthy_json():
    return (FIXTURES_DIR / "rccl_v2289_json_healthy.json").read_text()


@pytest.fixture
def degraded_json():
    return (FIXTURES_DIR / "rccl_v2289_json_degraded.json").read_text()


# -- Basic parse ---------------------------------------------------------------


def test_parse_healthy_state(parser, healthy_json):
    snap = parser.parse(healthy_json)
    assert snap.state == RCCLJobState.HEALTHY


def test_parse_healthy_rccl_version(parser, healthy_json):
    snap = parser.parse(healthy_json)
    assert snap.job_summary is not None
    assert snap.job_summary.rccl_version == "2.28.9"


def test_parse_healthy_hip_runtime_version(parser, healthy_json):
    snap = parser.parse(healthy_json)
    assert snap.job_summary.hip_runtime_version == 70226015


def test_parse_healthy_driver_version(parser, healthy_json):
    snap = parser.parse(healthy_json)
    assert snap.job_summary.amdgpu_driver_version == 70226015


def test_parse_healthy_total_gpus(parser, healthy_json):
    snap = parser.parse(healthy_json)
    assert snap.job_summary.total_gpus == 8


def test_parse_healthy_total_nodes(parser, healthy_json):
    snap = parser.parse(healthy_json)
    assert snap.job_summary.total_nodes == 1


def test_parse_healthy_total_processes(parser, healthy_json):
    snap = parser.parse(healthy_json)
    # 8 ranks on node1, each with a distinct pid
    assert snap.job_summary.total_processes == 8


# -- Communicator fields -------------------------------------------------------


def test_parse_healthy_communicator_count(parser, healthy_json):
    snap = parser.parse(healthy_json)
    assert len(snap.communicators) == 1


def test_parse_healthy_communicator_hash(parser, healthy_json):
    snap = parser.parse(healthy_json)
    assert snap.communicators[0].comm_hash == "0xabc123def456"


def test_parse_healthy_communicator_ranks(parser, healthy_json):
    snap = parser.parse(healthy_json)
    comm = snap.communicators[0]
    assert comm.total_ranks == 8
    assert comm.responding_ranks == 8
    assert comm.missing_ranks == 0


def test_parse_healthy_communicator_health(parser, healthy_json):
    snap = parser.parse(healthy_json)
    assert snap.communicators[0].health == RCCLJobState.HEALTHY


# -- Per-rank details (only available from JSON parser) ------------------------


def test_parse_healthy_ranks_populated(parser, healthy_json):
    snap = parser.parse(healthy_json)
    comm = snap.communicators[0]
    assert len(comm.ranks) == 8


def test_parse_healthy_rank_fields(parser, healthy_json):
    snap = parser.parse(healthy_json)
    rank0 = snap.communicators[0].ranks[0]
    assert rank0.comm_rank == 0
    assert rank0.node_addr == "node1"
    assert rank0.cuda_dev == 0
    assert rank0.nvml_dev == 0


def test_parse_healthy_rank_status_flags(parser, healthy_json):
    snap = parser.parse(healthy_json)
    status = snap.communicators[0].ranks[0].status
    assert status.init_state == 0
    assert status.async_error == 0
    assert status.abort_flag is False
    assert status.finalize_called is False
    assert status.destroy_flag is False


def test_parse_healthy_rank_collective_counts(parser, healthy_json):
    snap = parser.parse(healthy_json)
    counts = snap.communicators[0].ranks[0].coll_op_counts
    assert counts[NCCLFunction.ALL_REDUCE] == 100
    assert counts[NCCLFunction.ALL_GATHER] == 20
    assert counts[NCCLFunction.BROADCAST] == 10


# -- Degraded fixture ----------------------------------------------------------


def test_parse_degraded_state(parser, degraded_json):
    snap = parser.parse(degraded_json)
    assert snap.state == RCCLJobState.DEGRADED


def test_parse_degraded_missing_ranks(parser, degraded_json):
    snap = parser.parse(degraded_json)
    comm = snap.communicators[0]
    assert comm.total_ranks == 8
    assert comm.missing_ranks == 1
    assert comm.responding_ranks == 7


def test_parse_degraded_communicator_health(parser, degraded_json):
    snap = parser.parse(degraded_json)
    assert snap.communicators[0].health == RCCLJobState.DEGRADED


def test_parse_degraded_aborted_rank_detected(parser, degraded_json):
    """Rank 6 has abort_flag=true and async_error=5 — parser must surface these."""
    snap = parser.parse(degraded_json)
    rank6 = snap.communicators[0].ranks[6]
    assert rank6.status.abort_flag is True
    assert rank6.status.async_error == 5


def test_parse_degraded_comm_hash(parser, degraded_json):
    snap = parser.parse(degraded_json)
    assert snap.communicators[0].comm_hash == "0x3b2fe521bf43bc04"


def test_parse_degraded_no_dead_peers(parser, degraded_json):
    snap = parser.parse(degraded_json)
    assert snap.dead_peers == []


def test_parse_degraded_no_raw_errors(parser, degraded_json):
    """JSON path has no raw error lines section (unlike text path)."""
    snap = parser.parse(degraded_json)
    assert snap.errors == []


# -- Edge / error cases --------------------------------------------------------


def test_parse_empty_string_returns_no_job(parser):
    snap = parser.parse("")
    assert snap.state == RCCLJobState.NO_JOB


def test_parse_invalid_json_returns_error(parser):
    snap = parser.parse("{not valid json")
    assert snap.state == RCCLJobState.ERROR


def test_parse_json_array_returns_error(parser):
    """Top-level array is not a valid RAS JSON document."""
    snap = parser.parse("[1, 2, 3]")
    assert snap.state == RCCLJobState.ERROR


def test_parse_empty_communicators_list(parser):
    """Job starting up — communicators list is empty."""
    snap = parser.parse(
        '{"nccl_version": "2.28.9", "cuda_runtime_version": 0, '
        '"cuda_driver_version": 0, "communicators_count": 0, "communicators": []}'
    )
    assert snap.state == RCCLJobState.HEALTHY
    assert len(snap.communicators) == 0


def test_parse_text_input_returns_error(parser):
    """Text-format VERBOSE STATUS must not be silently accepted by JSON parser."""
    text_input = (
        "RCCL version 2.28.3 compiled with ROCm \"7.2.0.0\"\n"
        "HIP runtime version 70226015, amdgpu driver version 70226015\n"
    )
    snap = parser.parse(text_input)
    assert snap.state == RCCLJobState.ERROR


def test_parse_missing_ranks_count_defaults_gracefully(parser):
    """missing_ranks_count absent → defaults to 0, state stays HEALTHY."""
    doc = (
        '{"nccl_version": "2.28.9", "cuda_runtime_version": 0, "cuda_driver_version": 0,'
        ' "communicators_count": 1, "communicators": [{"hash": "0xaaa", "size": 2,'
        ' "ranks_count": 2, "ranks": ['
        '{"rank": 0, "host": "n1", "pid": 1, "cuda_dev": 0, "nvml_dev": 0,'
        ' "status": {"init_state": 0, "async_error": 0, "finalize_called": false,'
        ' "destroy_flag": false, "abort_flag": false}, "collective_counts": {}},'
        '{"rank": 1, "host": "n1", "pid": 2, "cuda_dev": 1, "nvml_dev": 1,'
        ' "status": {"init_state": 0, "async_error": 0, "finalize_called": false,'
        ' "destroy_flag": false, "abort_flag": false}, "collective_counts": {}}'
        ']}]}'
    )
    snap = parser.parse(doc)
    assert snap.state == RCCLJobState.HEALTHY
    assert snap.communicators[0].missing_ranks == 0
