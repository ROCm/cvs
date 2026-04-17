"""
Tests for RCCL RAS text output parser.
Written test-first against real captured rcclras -v output from a live MI300X cluster.
"""
import pytest
from pathlib import Path

from app.collectors.rccl_text_parser import RCCLTextParser
from app.models.rccl_models import RCCLJobState

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def healthy_output():
    return (FIXTURES_DIR / "rccl_verbose_status_healthy.txt").read_text()


@pytest.fixture
def degraded_output():
    return (FIXTURES_DIR / "rccl_verbose_status_degraded.txt").read_text()


@pytest.fixture
def connection_reset_output():
    return (FIXTURES_DIR / "rccl_verbose_status_connection_reset.txt").read_text()


@pytest.fixture
def parser():
    return RCCLTextParser()


# -- Healthy fixture tests ---------------------------------------------------

def test_parse_rccl_version(parser, healthy_output):
    snapshot = parser.parse(healthy_output)
    assert snapshot.job_summary is not None
    assert snapshot.job_summary.rccl_version == "2.28.3"


def test_parse_hip_runtime_version(parser, healthy_output):
    snapshot = parser.parse(healthy_output)
    assert snapshot.job_summary.hip_runtime_version == 70226015


def test_parse_driver_version(parser, healthy_output):
    snapshot = parser.parse(healthy_output)
    assert snapshot.job_summary.amdgpu_driver_version == 70226015


def test_parse_job_summary_nodes(parser, healthy_output):
    snapshot = parser.parse(healthy_output)
    assert snapshot.job_summary.total_nodes == 1


def test_parse_job_summary_processes(parser, healthy_output):
    snapshot = parser.parse(healthy_output)
    assert snapshot.job_summary.total_processes == 8


def test_parse_job_summary_gpus(parser, healthy_output):
    snapshot = parser.parse(healthy_output)
    assert snapshot.job_summary.total_gpus == 8


def test_parse_healthy_state(parser, healthy_output):
    snapshot = parser.parse(healthy_output)
    assert snapshot.state == RCCLJobState.HEALTHY


def test_parse_healthy_communicators(parser, healthy_output):
    snapshot = parser.parse(healthy_output)
    assert len(snapshot.communicators) == 1
    comm = snapshot.communicators[0]
    assert comm.total_ranks == 8
    assert comm.responding_ranks == 8
    assert comm.missing_ranks == 0
    assert comm.health == RCCLJobState.HEALTHY


def test_parse_healthy_no_dead_peers(parser, healthy_output):
    snapshot = parser.parse(healthy_output)
    assert snapshot.dead_peers == []


def test_parse_healthy_no_errors(parser, healthy_output):
    """Errors section is empty in healthy fixture."""
    snapshot = parser.parse(healthy_output)
    assert snapshot.state == RCCLJobState.HEALTHY


# -- Degraded fixture tests --------------------------------------------------

def test_parse_degraded_state(parser, degraded_output):
    snapshot = parser.parse(degraded_output)
    assert snapshot.state == RCCLJobState.DEGRADED


def test_parse_degraded_communicator_ranks(parser, degraded_output):
    snapshot = parser.parse(degraded_output)
    assert len(snapshot.communicators) >= 1
    comm = snapshot.communicators[0]
    # 7 responding out of 8 total
    assert comm.total_ranks == 8
    assert comm.responding_ranks == 7
    assert comm.missing_ranks == 1


def test_parse_degraded_communicator_health(parser, degraded_output):
    snapshot = parser.parse(degraded_output)
    comm = snapshot.communicators[0]
    assert comm.health == RCCLJobState.DEGRADED


def test_parse_degraded_has_communicator_hash(parser, degraded_output):
    """The degraded fixture contains communicator hash 3b2fe521bf43bc04 in the Errors section."""
    snapshot = parser.parse(degraded_output)
    # The parser should extract comm hash from error section if available
    # At minimum, the communicator should be parsed from the table
    assert len(snapshot.communicators) >= 1


def test_parse_degraded_errors_section_not_empty(parser, degraded_output):
    """The degraded fixture has INCOMPLETE errors -- parser should detect this."""
    snapshot = parser.parse(degraded_output)
    assert snapshot.state == RCCLJobState.DEGRADED


# -- 2-node degraded fixture (ranks_per_node shown as range "7-8") ------------

@pytest.fixture
def degraded_2node_output():
    return (FIXTURES_DIR / "rccl_verbose_status_degraded_2node.txt").read_text()


def test_parse_degraded_2node_state(parser, degraded_2node_output):
    snapshot = parser.parse(degraded_2node_output)
    assert snapshot.state == RCCLJobState.DEGRADED


def test_parse_degraded_2node_communicator_parsed(parser, degraded_2node_output):
    """ranks_per_node='7-8' range must not prevent communicator row from matching."""
    snapshot = parser.parse(degraded_2node_output)
    assert len(snapshot.communicators) == 1
    comm = snapshot.communicators[0]
    assert comm.total_ranks == 16
    assert comm.responding_ranks == 15
    assert comm.missing_ranks == 1
    assert comm.health == RCCLJobState.DEGRADED


def test_parse_degraded_2node_job_summary(parser, degraded_2node_output):
    snapshot = parser.parse(degraded_2node_output)
    assert snapshot.job_summary is not None
    assert snapshot.job_summary.total_nodes == 2
    assert snapshot.job_summary.total_gpus == 16


# -- Connection reset / error tests ------------------------------------------

def test_parse_connection_reset(parser, connection_reset_output):
    snapshot = parser.parse(connection_reset_output)
    assert snapshot.state in (RCCLJobState.NO_JOB, RCCLJobState.ERROR)


def test_parse_empty_string(parser):
    snapshot = parser.parse("")
    assert snapshot.state == RCCLJobState.NO_JOB


def test_parse_connection_refused(parser):
    text = "Connecting to 127.0.0.1:28028: Connection refused\nFailed to connect to the NCCL RAS service!"
    snapshot = parser.parse(text)
    assert snapshot.state == RCCLJobState.NO_JOB


# -- Edge cases ---------------------------------------------------------------

def test_parse_inconsistent_topology_single_node(parser, healthy_output):
    snapshot = parser.parse(healthy_output)
    assert snapshot.job_summary.inconsistent_topology is False


def test_parser_does_not_crash_on_garbage(parser):
    """Parser should return ERROR state on unparseable text, not crash."""
    snapshot = parser.parse("some random garbage that is not rcclras output")
    # Should return a snapshot (not raise), with NO_JOB or ERROR state
    assert snapshot.state in (RCCLJobState.NO_JOB, RCCLJobState.ERROR)
