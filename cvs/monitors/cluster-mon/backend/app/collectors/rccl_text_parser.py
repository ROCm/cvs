"""
Parser for rcclras VERBOSE STATUS text output (RCCL v2.28.3+).

Driven by regex against real captured output. The C source (client_support.cc)
confirms field semantics but this parser is built from actual rcclras output.
"""

import re
import time
import logging
from typing import Optional

from app.models.rccl_models import (
    RCCLSnapshot,
    RCCLJobState,
    RCCLJobSummary,
    RCCLCommunicator,
)

logger = logging.getLogger(__name__)


class RCCLTextParser:
    """
    Parses rcclras -v (VERBOSE STATUS) text output into RCCLSnapshot.

    Format (RCCL v2.28.3):
    - Line 1: RCCL version X.Y.Z compiled with ROCm "..."
    - Line 2: HIP runtime version N, amdgpu driver version N
    - Job summary table: Nodes/Processes/GPUs counts
    - Communicators table: Group-based with Status and Errors columns
    - Errors section (if any)
    - Warnings section (if any)
    """

    # Regex patterns
    _VERSION_RE = re.compile(
        r"RCCL version (\S+)\s+compiled with ROCm"
    )
    _HIP_DRIVER_RE = re.compile(
        r"HIP runtime version (\d+),\s*amdgpu driver version (\d+)"
    )
    # Anchored to the "Job summary" section to avoid matching spurious 5-integer lines.
    # Lookahead stops at the next section header (word chars followed by a === underline)
    # or end-of-string.  We deliberately do NOT use (?=^\S|\Z) because the column header
    # line "(total)   per node ..." starts at column 0 and would prematurely end the match.
    _JOB_SUMMARY_SECTION_RE = re.compile(
        r"^Job summary\s*\n=+\s*\n(.*?)(?=^\w[^\n]*\n=+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    _JOB_SUMMARY_ROW_RE = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
        re.MULTILINE,
    )
    # Anchored to the "Communicators" section (same lookahead rationale as above)
    _COMM_SECTION_RE = re.compile(
        r"^Communicators.*?\n=+\s*\n(.*?)(?=^\w[^\n]*\n=+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    _COMM_ROW_RE = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+(?:-\d+)?)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s*$",
        re.MULTILINE,
    )
    _CONNECTION_REFUSED_RE = re.compile(
        r"Connection refused|Failed to connect|Connection reset by peer",
        re.IGNORECASE,
    )

    def parse(self, raw_text: str) -> RCCLSnapshot:
        """Parse rcclras -v output into an RCCLSnapshot."""
        if not raw_text or not raw_text.strip():
            return RCCLSnapshot.empty(state=RCCLJobState.NO_JOB)

        # Check for connection refused (no job running)
        if self._CONNECTION_REFUSED_RE.search(raw_text):
            return RCCLSnapshot.empty(state=RCCLJobState.NO_JOB)

        try:
            job_summary = self._parse_job_summary(raw_text)
            communicators = self._parse_communicators(raw_text)
            dead_peers = self._parse_dead_peers(raw_text)
            errors = self._parse_errors_section(raw_text)

            # Determine overall state
            state = self._determine_state(communicators, dead_peers, errors, job_summary)

            return RCCLSnapshot(
                timestamp=time.time(),
                state=state,
                job_summary=job_summary,
                communicators=communicators,
                peers=[],       # Not in v2.28.3 text output
                dead_peers=dead_peers,
                errors=errors,
            )
        except Exception as e:
            logger.error(f"Failed to parse rcclras output: {e}", exc_info=True)
            return RCCLSnapshot.empty(state=RCCLJobState.ERROR)

    def _parse_job_summary(self, text: str) -> Optional[RCCLJobSummary]:
        """Extract job summary from the table after 'Job summary'."""
        # RCCL version
        version_match = self._VERSION_RE.search(text)
        rccl_version = version_match.group(1) if version_match else "unknown"

        # HIP/driver versions
        hip_match = self._HIP_DRIVER_RE.search(text)
        hip_version = int(hip_match.group(1)) if hip_match else 0
        driver_version = int(hip_match.group(2)) if hip_match else 0

        # Job summary numbers table — search within the "Job summary" section only
        # to avoid false-matching any other line that happens to have 5 integers.
        section_match = self._JOB_SUMMARY_SECTION_RE.search(text)
        section = section_match.group(1) if section_match else ""
        summary_match = self._JOB_SUMMARY_ROW_RE.search(section)
        if not summary_match:
            return None

        total_nodes = int(summary_match.group(1))
        procs_per_node = int(summary_match.group(2))
        gpus_per_proc = int(summary_match.group(3))
        total_procs = int(summary_match.group(4))
        total_gpus = int(summary_match.group(5))

        # Check for inconsistent topology (when processes/node varies)
        # In v2.28.3, the table always shows uniform values. Non-uniform
        # topologies would show different format. For now, assume consistent
        # if we get a single row.
        inconsistent = False

        return RCCLJobSummary(
            total_nodes=total_nodes,
            total_processes=total_procs,
            total_gpus=total_gpus,
            rccl_version=rccl_version,
            hip_runtime_version=hip_version,
            amdgpu_driver_version=driver_version,
            inconsistent_topology=inconsistent,
        )

    def _parse_communicators(self, text: str) -> list[RCCLCommunicator]:
        """
        Extract communicator groups from the table after 'Communicators'.

        Format:
        Group  Comms  Nodes  Ranks    Ranks    Ranks   Status  Errors
            #  in grp per c  per node per comm in group
            0      1      2    7-8       16       15   RUNNING  INCOMPLETE
        """
        # Search within the Communicators section to avoid matching other tables
        section_match = self._COMM_SECTION_RE.search(text)
        section = section_match.group(1) if section_match else text

        comms = []
        for match in self._COMM_ROW_RE.finditer(section):
            group_num = int(match.group(1))
            comms_in_group = int(match.group(2))
            # ranks_per_node (group 4) may be a range like "7-8" on heterogeneous topologies
            ranks_per_comm = int(match.group(5))   # expected ranks in ONE communicator
            ranks_in_group = int(match.group(6))   # actual respondents across ALL comms in group
            status = match.group(7)
            errors = match.group(8)

            # Determine health from status and errors columns
            if errors == "OK" and status == "RUNNING":
                health = RCCLJobState.HEALTHY
            elif errors != "OK":
                health = RCCLJobState.DEGRADED
            else:
                health = RCCLJobState.HEALTHY

            # "Ranks in group" spans ALL communicators in the group, so total expected
            # = ranks_per_comm * comms_in_group. Missing = expected - responding.
            total_expected = ranks_per_comm * comms_in_group
            missing = max(0, total_expected - ranks_in_group)

            # In v2.28.3, we don't get per-communicator hashes from the table,
            # so use group number as placeholder
            comms.append(RCCLCommunicator(
                comm_hash=f"group_{group_num}",
                total_ranks=total_expected,
                responding_ranks=ranks_in_group,
                missing_ranks=missing,
                ranks=[],         # Per-rank detail only in verbose with outliers
                health=health,
            ))
        return comms

    def _parse_dead_peers(self, text: str) -> list[str]:
        """Extract dead peer addresses if present."""
        # v2.28.3: dead peers appear between job summary and communicators
        # Format: "Dead peers: IP:port, IP:port, ..."
        dead_re = re.compile(r"Dead peers?:\s*(.+)", re.IGNORECASE)
        match = dead_re.search(text)
        if match:
            peers_str = match.group(1).strip()
            return [p.strip() for p in peers_str.split(",") if p.strip()]
        return []

    def _parse_errors_section(self, text: str) -> list[str]:
        """Extract error lines from the Errors section."""
        errors = []
        # Find content between "Errors" header and "Warnings" header (or end)
        # Stop at any section header (word followed by === underline) or end of string.
        # Using \Z (end-of-string) instead of $ (end-of-line) so the lazy .*?
        # doesn't stop at the first line ending when re.MULTILINE is active.
        errors_section = re.search(
            r"^Errors\s*\n=+\s*\n(.*?)(?=^\w[^\n]*\n=+|\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        if errors_section:
            content = errors_section.group(1).strip()
            if content:
                errors = [line.strip() for line in content.splitlines() if line.strip()]
        return errors

    def _determine_state(
        self,
        communicators: list[RCCLCommunicator],
        dead_peers: list[str],
        errors: list[str],
        job_summary: Optional[RCCLJobSummary] = None,
    ) -> RCCLJobState:
        """Determine overall job state from parsed data."""
        # If we couldn't parse a job summary, the text isn't valid rcclras output
        if job_summary is None and not communicators:
            return RCCLJobState.NO_JOB
        if dead_peers:
            return RCCLJobState.DEGRADED
        if errors:
            return RCCLJobState.DEGRADED
        for comm in communicators:
            if comm.health == RCCLJobState.DEGRADED:
                return RCCLJobState.DEGRADED
            if comm.missing_ranks > 0:
                return RCCLJobState.DEGRADED
        if not communicators:
            return RCCLJobState.HEALTHY  # No comms but no errors = job starting up
        return RCCLJobState.HEALTHY
