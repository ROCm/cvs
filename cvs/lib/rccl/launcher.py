from __future__ import annotations

import re
import shlex
import time
from typing import Any

from cvs.lib import globals

from .config import RcclConfig

log = globals.log

RCCL_ERROR_SUBSTRINGS = (
    "orte does not know how to route",
    "orte was unable to reliably start",
    "nccl error",
    "test failure",
    "no such file or directory",
)

# Bash / launcher failures we surface before the generic "missing bandwidth line" check.
_RCCL_LAUNCH_FAILURE_RES = (
    (
        re.compile(r"bash:\s*(?:line\s+\d+:\s*)?(.+?):\s*Is a directory"),
        "Shell tried to run a path that is a directory (common mistake: MPI_HOME must be the install prefix "
        "so ${{MPI_HOME}}/bin/mpirun is the executable, not a directory). Path: {0}",
    ),
    (
        re.compile(r"bash:\s*(?:line\s+\d+:\s*)?(.+?):\s*command not found"),
        "Shell reported command not found: {0}. Check PATH, env_script, MPI_HOME, and ROCm/RCCL paths.",
    ),
    (
        re.compile(r"RCCL launch:\s*(.+)"),
        "{0}",
    ),
)


def _env_setup_lines(config: RcclConfig) -> list[str]:
    """Source the site env_script (PATH, LD_LIBRARY_PATH, RCCL_TESTS_BUILD_DIR, MPI_HOME, etc.)."""
    return [f"source {shlex.quote(config.env_script)}"]


def _shell_runtime_guards(config: RcclConfig, collective: str) -> list[str]:
    """Bash checks after env_script; fails fast with RCCL launch: ... messages."""
    lines: list[str] = []
    lines.append(
        ': "${RCCL_TESTS_BUILD_DIR:?export RCCL_TESTS_BUILD_DIR in env_script '
        '(directory containing rccl-tests *_perf binaries).}"'
    )
    if not config.is_single_node:
        lines.append(
            ': "${MPI_HOME:?multi-node launch requires MPI_HOME in env_script '
            '(install prefix — the directory that contains bin/mpirun).}"'
        )
        lines.append(
            'if [[ ! -x "${MPI_HOME}/bin/mpirun" ]]; then '
            'echo "RCCL launch: ${MPI_HOME}/bin/mpirun is missing or not executable; '
            'MPI_HOME must be the MPI install prefix (parent of bin), not the bin directory itself."; '
            "exit 127; fi"
        )
    bin_path = f"\"${{RCCL_TESTS_BUILD_DIR%/}}/{collective}\""
    lines.append(
        f"if [[ ! -x {bin_path} ]]; then "
        f'echo "RCCL launch: rccl-tests binary not executable for collective {collective} '
        '(set RCCL_TESTS_BUILD_DIR in env_script to the rccl-tests build directory)."; '
        "exit 127; fi"
    )
    return lines


def _build_shell_payload(config: RcclConfig, collective: str, remote_result_file: str) -> str:
    binary_word = f'"${{RCCL_TESTS_BUILD_DIR%/}}/{collective}"'
    gpus_per_rank = config.ranks_per_node if config.is_single_node else 1
    benchmark_cmd = (
        f"{binary_word} "
        f"-b {shlex.quote(config.start_size)} "
        f"-e {shlex.quote(config.end_size)} "
        f"-f {shlex.quote(config.step_factor)} "
        f"-g {gpus_per_rank} "
        f"-c 1 "
        f"-w {shlex.quote(config.warmups)} "
        f"-d {shlex.quote(config.datatype)} "
        f"-n {shlex.quote(config.iterations)} "
        f"-N {shlex.quote(config.cycles)} "
        f"-Z json -x {shlex.quote(remote_result_file)}"
    )
    return "; ".join(_env_setup_lines(config) + _shell_runtime_guards(config, collective) + [benchmark_cmd])


def _prepare_hostfile(shdl: Any, launch_hosts: list[str], ranks_per_node: int, remote_work_dir: str) -> str:
    base = remote_work_dir.rstrip("/")
    hostfile = f"{base}/rccl_hostfile_{int(time.time() * 1000000)}.txt"
    shdl.exec(f"mkdir -p {shlex.quote(base)}", print_console=False)
    shdl.exec(f"rm -f {shlex.quote(hostfile)}", print_console=False)
    slot_lines = " ".join(shlex.quote(f"{host} slots={ranks_per_node}") for host in launch_hosts)
    shdl.exec(f"printf '%s\\n' {slot_lines} > {shlex.quote(hostfile)}", print_console=False)
    return hostfile


def build_collective_command(
    config: RcclConfig, collective: str, remote_result_file: str, launch_hosts: list[str], shdl: Any
) -> str:
    shell_payload = _build_shell_payload(config, collective, remote_result_file)
    shell_cmd = f"bash -lc {shlex.quote(shell_payload)}"

    if config.is_single_node:
        return shell_cmd

    hostfile = _prepare_hostfile(shdl, launch_hosts, config.ranks_per_node, config.artifacts_remote_work_dir)
    mpirun_args = (
        f"--np {config.num_ranks} "
        "--allow-run-as-root "
        "--bind-to numa "
        f"--hostfile {shlex.quote(hostfile)} "
        f"{shell_cmd}"
    )
    # MPI_HOME and RCCL_TESTS_BUILD_DIR come from env_script; source before mpirun (outer wrapper).
    wrapped = f"source {shlex.quote(config.env_script)} && exec \"${{MPI_HOME}}/bin/mpirun\" {mpirun_args}"
    return f"bash -lc {shlex.quote(wrapped)}"


def _scan_rccl_launch_failures(line: str) -> None:
    if "cannot execute binary file" in line.lower():
        raise RuntimeError(
            "RCCL launch failed: cannot execute binary (wrong architecture or bad interpreter): "
            f"{line.strip()}"
        )
    for cre, tmpl in _RCCL_LAUNCH_FAILURE_RES:
        match = cre.search(line)
        if match:
            arg = next((g for g in match.groups() if g), line.strip())
            raise RuntimeError(f"RCCL launch failed: {tmpl.format(arg)}")


def _scan_rccl_stdout(output: str) -> None:
    warnings = []
    for line in output.splitlines():
        _scan_rccl_launch_failures(line)
        lower_line = line.lower()
        if any(needle in lower_line for needle in RCCL_ERROR_SUBSTRINGS):
            raise RuntimeError(f"RCCL execution failed: {line}")
        if "NCCL WARN" in line:
            warnings.append(line)

    if warnings:
        log.warning(
            "NCCL warnings observed in RCCL test output:\n%s",
            "\n".join(warnings),
        )

    if "# Avg bus bandwidth" not in output:
        raise RuntimeError(
            "RCCL output did not contain '# Avg bus bandwidth' (benchmark may not have started; "
            "check earlier lines for RCCL launch / mpirun / rccl-tests errors, env_script, MPI_HOME, "
            "and RCCL_TESTS_BUILD_DIR)"
        )
