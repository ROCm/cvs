'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import base64
import os
import re
import shlex
import time
from pathlib import Path
from typing import Optional

from cvs.lib import globals
from cvs.lib.inference.base import InferenceBaseJob
from cvs.lib.inference.inferencemax_host_scripts import bundled_script_body
from cvs.lib.verify_lib import fail_test

log = globals.log

_BENCH_FAILED_REQUESTS_RE = re.compile(r"Failed\s+requests:\s*(\d+)", re.I)
_BENCH_CLIENT_TRACEBACK_RE = re.compile(r"Traceback\s*\(most recent call last\):", re.I)

_GIT_CLONE_FAIL_RE = re.compile(
    r"(?:^|\n)(?:fatal:\s|error:\s)|Could not resolve host|Repository not found|"
    r"Permission denied \(publickey\)|SSL certificate problem",
    re.I,
)

_SERVER_LOG_ERROR_SNIPPET_CHARS = 2500


def _clone_dirname_from_git_url(repo_url: str) -> str:
    part = (repo_url or "").rstrip("/").rsplit("/", 1)[-1]
    return part.removesuffix(".git") if part.endswith(".git") else part or "InferenceX"


def _wrapper_dir_ok(d: Path, entry_basename: str) -> bool:
    return d.is_dir() and (d / entry_basename).is_file()


def _discover_host_benchmark_scripts_dir(relpath: Path, entry_basename: str) -> Optional[Path]:
    """
    Return the host directory containing ``entry_basename`` under the ``cvs`` package tree.

    ``relpath`` is relative to the ``cvs`` package root (e.g.
    ``lib/dtni/vllm_benchmark_scripts`` or legacy ``input/.../benchmark_server_scripts``).
    The shared :mod:`cvs.lib.dtni.vllm_benchmark_scripts` directory is tried first when it
    contains ``entry_basename``. When checkout paths are missing, bundled bytes are used
    when deploying to cluster nodes.
    """
    import cvs as _cvs_module

    root = Path(_cvs_module.__file__).resolve().parent
    candidates: list[Path] = []
    try:
        from cvs.lib.dtni.vllm_benchmark_scripts import bundled_scripts_dir as _vbs_scripts_root

        _shared = _vbs_scripts_root()
        if _wrapper_dir_ok(_shared, entry_basename):
            candidates.append(_shared)
    except (ImportError, OSError, TypeError, ValueError):
        pass
    candidates.extend(
        [
            root / relpath,
            Path(__file__).resolve().parents[2] / relpath,
        ]
    )
    try:
        from importlib.resources import files

        t = files("cvs").joinpath(*relpath.parts)
        if t.is_dir():
            candidates.append(Path(str(t)))
    except (ModuleNotFoundError, FileNotFoundError, NotImplementedError, OSError, TypeError, ValueError):
        pass

    seen: set[str] = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except OSError:
            continue
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        if _wrapper_dir_ok(rp, entry_basename):
            return rp

    try:
        for sh in root.rglob(entry_basename):
            parent = sh.parent
            if parent.name == "benchmark_server_scripts" and _wrapper_dir_ok(parent, entry_basename):
                return parent.resolve()
    except OSError:
        pass
    return None


def _stage_bundle_subdir(relpath: Path) -> str:
    """Stable subdirectory name under host_scripts_staged/ (per variant)."""
    parts = relpath.parts
    if len(parts) >= 2 and parts[-1] == "benchmark_server_scripts":
        return parts[-2]
    return "default"


def _benchmark_server_script_path_is_auto(if_dict: dict) -> bool:
    raw = if_dict.get("benchmark_server_script_path")
    if raw is None:
        return True
    s = str(raw).strip()
    return not s or s.lower() == "auto"


def _compute_staged_host_script_dir(if_dict: dict, relpath: Path) -> str:
    """Return ``{log_dir}/inference-max/host_scripts_staged/<variant>/`` (POSIX string, no I/O)."""
    log_dir = str(if_dict.get("log_dir") or "").strip().replace("\\", "/")
    if not log_dir:
        msg = "benchmark_server_script_path is auto but log_dir is empty; cannot stage host server scripts."
        fail_test(msg)
        raise FileNotFoundError(msg)
    dest = Path(log_dir) / "inference-max" / "host_scripts_staged" / _stage_bundle_subdir(relpath)
    return str(dest).replace("\\", "/")


def _read_host_script_payload(if_dict: dict, relpath: Path, entry_basename: str) -> bytes:
    """Read server script bytes from local checkout discovery or bundled registry."""
    found = _discover_host_benchmark_scripts_dir(relpath, entry_basename)
    if found is not None and _wrapper_dir_ok(found, entry_basename):
        try:
            return (found / entry_basename).read_bytes()
        except OSError as exc:
            fail_test(f"Cannot read {found / entry_basename}: {exc}")
            raise
    body = bundled_script_body(entry_basename)
    if not body:
        msg = (
            f"No checkout or bundled host server script for {entry_basename!r}. "
            "Add it under ``cvs.lib.dtni.vllm_benchmark_scripts`` (see that package's README), "
            "or set benchmark_server_script_path to a host directory that contains this file."
        )
        fail_test(msg)
        raise FileNotFoundError(msg)
    return body.replace("\r\n", "\n").encode("utf-8")


def _deploy_host_script_to_cluster_nodes(
    s_phdl,
    dest_dir: str,
    entry_basename: str,
    content: bytes,
    *,
    container_name: str,
) -> None:
    """
    Write ``content`` to ``{dest_dir}/{entry_basename}`` on every node ``s_phdl`` reaches.

    Uses ``docker exec`` so directories are created as root inside the container, matching
    how log/output paths under ``log_dir`` are typically prepared. That avoids
    ``PermissionError`` when ``{log_dir}/inference-max`` was created earlier by ``docker exec``
    and is not writable over SSH as the cluster login user.
    """
    dest_file = f"{dest_dir.rstrip('/')}/{entry_basename}"
    b64 = base64.standard_b64encode(content).decode("ascii")
    py = (
        "import base64,os,pathlib;"
        f"p=pathlib.Path({dest_file!r});"
        "p.parent.mkdir(parents=True,exist_ok=True);"
        f"p.write_bytes(base64.standard_b64decode({b64!r}));"
        "os.chmod(p,0o755)"
    )
    inner = f"python3 -c {shlex.quote(py)}"
    cmd = f"docker exec {shlex.quote(container_name)} /bin/bash -c {shlex.quote(inner)}"
    out_dict = s_phdl.exec(cmd)
    for node, out in (out_dict or {}).items():
        text = out or ""
        if re.search(r"traceback|syntaxerror|filenotfounderror|permissionerror|nosuchfile", text, re.I):
            log.error("FAIL - staging host script on node %s: %s", node, text[-2000:])
            fail_test(f"Staging host server script failed on node {node}")
            raise RuntimeError(f"Staging host server script failed on node {node}: {text[-1500:]}")
    log.info(
        "Deployed host server script %s on cluster nodes under %s (via docker exec %s)",
        entry_basename,
        dest_dir,
        container_name,
    )


def _resolved_host_benchmark_scripts_dir(if_dict: dict, relpath: Path, entry_basename: str) -> str:
    """
    Host directory containing ``entry_basename`` for ``docker exec`` ``cd``.

    If ``benchmark_server_script_path`` is set to a non-empty string other than
    ``auto``, that path is used. Otherwise return the staged directory under ``log_dir``; the script
    file is written on each GPU node (via ``docker exec`` into the inference container) in :meth:`InferenceMaxJob.launch_server`.
    """
    raw = if_dict.get("benchmark_server_script_path")
    if raw is not None:
        s = str(raw).strip()
        if s and s.lower() != "auto":
            return s.rstrip("/")

    return _compute_staged_host_script_dir(if_dict, relpath)


class InferenceMaxJob(InferenceBaseJob):
    uses_local_server_scripts = True

    def __init__(self, *args, orch=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.orch = orch
        self.if_dict.setdefault(
            'inferencemax_repo',
            'https://github.com/SemiAnalysisAI/InferenceX.git',
        )
        self.if_dict.setdefault(
            'host_benchmark_scripts_relpath',
            'lib/dtni/vllm_benchmark_scripts',
        )
        self.default_server_precheck_error_pattern = re.compile(
            'no such file or directory|command not found|cannot access|permission denied|error:|exception:|traceback|failed to start|'
            'hiperrorlaunchoutofresources|too many resources requested for launch|distbackender',
            re.I,
        )
        self.default_server_error_pattern_poll = re.compile(
            'failed to start|no such file or directory|command not found|cannot access|'
            'engine core initialization failed|server died before becoming healthy|failed core proc|'
            'hiperrorlaunchoutofresources|too many resources requested for launch|distbackender|'
            'worker proc .+ died unexpectedly',
            re.I,
        )

    def _using_host_server_scripts(self) -> bool:
        return bool(self.if_dict.get('use_host_mounted_server_script'))

    def _host_mount_relative_script(self) -> str:
        ss = self.server_script.replace('\\', '/')
        base = 'single_node' if int(self.nnodes) == 1 else 'multi_node'
        for prefix in (f'benchmarks/{base}/', 'benchmarks/single_node/', 'benchmarks/multi_node/'):
            if ss.startswith(prefix):
                ss = ss[len(prefix) :]
                break
        if ss.startswith('fixed_seq_len/'):
            ss = ss[len('fixed_seq_len/') :]
        return ss

    def get_server_script_directory(self):
        if self._using_host_server_scripts():
            rel = Path(str(self.if_dict.get("host_benchmark_scripts_relpath", "")).strip().replace("\\", "/"))
            if not rel.parts:
                fail_test("host_benchmark_scripts_relpath is empty")
                raise ValueError("host_benchmark_scripts_relpath is empty")
            entry = os.path.basename(self._host_mount_relative_script())
            return _resolved_host_benchmark_scripts_dir(self.if_dict, rel, entry).rstrip('/')
        name = _clone_dirname_from_git_url(self.if_dict.get('inferencemax_repo', ''))
        return f'/app/{name}'

    def get_server_script_path(self):
        if self._using_host_server_scripts():
            return self._host_mount_relative_script()
        base = "single_node" if int(self.nnodes) == 1 else "multi_node"
        return f'benchmarks/{base}/{self.server_script}'

    def get_result_filename(self):
        return 'inferencemax_test_result.json'

    def get_completion_pattern(self):
        return re.compile('Serving Benchmark Result', re.I)

    def get_log_subdir(self):
        return 'inference-max'

    def exec_nic_setup_scripts(self):
        if self.distributed_inference is True:
            if re.search('broadcom|thor', self.nic_type, re.I):
                self.nccl_ib_gid_index = 3
                out_dict = self.s_phdl.exec(
                    f'docker exec {self.container_name} /bin/bash -c "sudo \
                    cp /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host \
                    /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so; \
                    sleep 2;ibv_devinfo;sleep 2;"'
                )
                for node in out_dict.keys():
                    if not re.search(r'hca_id:\s+bnxt_', out_dict[node], re.I):
                        log.info("%s", out_dict[node])
                        fail_test(f'Broadcom libbnxt rdma driver is not properly copied on node {node}')

    def launch_server(self):
        script_dir = self.get_server_script_directory()
        log_file = f'{self.server_script}_server.log'
        script_path = self.get_server_script_path()

        if self._using_host_server_scripts() and _benchmark_server_script_path_is_auto(self.if_dict):
            rel = Path(str(self.if_dict.get("host_benchmark_scripts_relpath", "")).strip().replace("\\", "/"))
            if not rel.parts:
                fail_test("host_benchmark_scripts_relpath is empty")
                raise ValueError("host_benchmark_scripts_relpath is empty")
            entry = os.path.basename(self._host_mount_relative_script())
            payload = _read_host_script_payload(self.if_dict, rel, entry)
            _deploy_host_script_to_cluster_nodes(
                self.s_phdl,
                script_dir,
                entry,
                payload,
                container_name=self.container_name,
            )

        cmd_list = []
        workspace_host = f'{self.log_dir}/{self.get_log_subdir()}/workspace'.replace('\\', '/')
        for i in range(0, int(self.nnodes)):
            log_base = f'{self.log_dir}/{self.get_log_subdir()}/out-node{i}'
            log_path = f'{log_base}/{log_file}'
            log_parent = os.path.dirname(log_path).replace('\\', '/')
            cmd = f'''docker exec {self.container_name} /bin/bash -c "mkdir -p {log_parent} {workspace_host}; cd {script_dir}; source /tmp/server_env_script.sh; nohup /bin/bash {script_path} > {log_path} 2>&1 &" '''
            cmd_list.append(cmd)
        out_dict = self.s_phdl.exec_cmd_list(cmd_list)

        for node in out_dict.keys():
            if re.search(
                'No such file or directory|command not found|cannot access|Permission denied', out_dict[node], re.I
            ):
                log.error(f'FAIL - Failed to start server on node {node}: {out_dict[node]}')
                raise Exception(f'Failed to start server on node {node}: {out_dict[node]}')

    def check_server_status(self, log_file, log_subdir, error_pattern):
        cmd_list = []
        for i in range(0, int(self.nnodes)):
            cmd = f'tail -30 {self.log_dir}/{log_subdir}/out-node{i}/{log_file}'
            cmd_list.append(cmd)
        out_dict = self.s_phdl.exec_cmd_list(cmd_list)

        for node, output in out_dict.items():
            if error_pattern.search(output or ''):
                error_msg = f'Failed to start server on node {node}: {output[-_SERVER_LOG_ERROR_SNIPPET_CHARS:]}'
                fail_test(error_msg)
                raise Exception(error_msg)

        return out_dict

    def poll_server_startup(self):
        log_file = f'{self.server_script}_server.log'

        log.info(f'Waiting {self.default_server_precheck_wait_time} secs for server to start writing logs...')
        time.sleep(self.default_server_precheck_wait_time)

        cmd_list = []
        for i in range(0, int(self.nnodes)):
            cmd = f'tail -30 {self.log_dir}/{self.get_log_subdir()}/out-node{i}/{log_file}'
            cmd_list.append(cmd)
        out_dict = self.s_phdl.exec_cmd_list(cmd_list)
        for node in out_dict.keys():
            log_content = out_dict[node].lower()
            if self.default_server_precheck_error_pattern.search(log_content):
                error_msg = f'Failed to start server on node {node}: {out_dict[node][-_SERVER_LOG_ERROR_SNIPPET_CHARS:]}'
                fail_test(error_msg)
                raise Exception(error_msg)

        log.info(
            f'No immediate errors detected. Waiting {self.default_server_wait_time} more secs for server to fully launch...'
        )
        time.sleep(self.default_server_wait_time)

        for j in range(0, self.default_server_poll_count):
            log.info(f'Polling for application startup complete on all nodes, iteration {j}')
            tail_cmds = []
            for i in range(0, int(self.nnodes)):
                tail_cmds.append(f'tail -30 {self.log_dir}/{self.get_log_subdir()}/out-node{i}/{log_file}')
            out_dict = self.s_phdl.exec_cmd_list(tail_cmds)

            for node in out_dict.keys():
                if self.default_server_error_pattern_poll.search(out_dict[node] or ''):
                    error_msg = f'Failed to start server on node {node}: {out_dict[node][-_SERVER_LOG_ERROR_SNIPPET_CHARS:]}'
                    fail_test(error_msg)
                    raise Exception(error_msg)

            grep_out = self.s_phdl.exec_cmd_list(self._readiness_grep_cmd_list(log_file))
            if self._grep_readiness_outputs_ok(grep_out):
                log.info('Server startup confirmed on all nodes')
                return

            log.info(f'Waiting {self.default_server_poll_wait_time} secs for next poll')
            time.sleep(self.default_server_poll_wait_time)

        error_msg = 'Server did not report readiness before timeout; aborting startup'
        fail_test(error_msg)
        raise Exception(error_msg)

    def _tail_client_bench_logs(self) -> dict:
        """Return host -> last lines of bench_serv_script.log (inside the inference container).

        Uses :meth:`ContainerOrchestrator.exec` whenever ``self.orch`` is set (routes through
        core container runtime). For ``nnodes == 1``, a single ``exec`` on all hosts matches
        the vLLM-style pattern. For ``nnodes > 1``, rank *i* is assumed to run on ``orch.hosts[i]``
        (same ordering as ``Pssh.exec_cmd_list`` used to launch clients), so we issue one
        ``exec(..., hosts=[host])`` per rank with that rank's log path — one global ``exec``
        cannot apply different paths per host.
        """
        log_sub = self.get_log_subdir()
        orch = getattr(self, "orch", None)
        if orch is None:
            cmd_list = []
            for i in range(0, int(self.nnodes)):
                path = f"{self.log_dir}/{log_sub}/out-node{i}/bench_serv_script.log".replace("\\", "/")
                inner = f"tail -30 {shlex.quote(path)}"
                cmd_list.append(f"docker exec {shlex.quote(self.container_name)} bash -c {shlex.quote(inner)}")
            return self.s_phdl.exec_cmd_list(cmd_list)

        n = int(self.nnodes)
        hosts = list(getattr(orch, "hosts", ()) or ())
        if not hosts:
            msg = "orch.hosts is empty; cannot poll client logs via orchestrator"
            fail_test(msg)
            raise RuntimeError(msg)
        if n > len(hosts):
            msg = f"nnodes={n} exceeds orchestrator host count {len(hosts)}; cannot map ranks to hosts"
            fail_test(msg)
            raise RuntimeError(msg)

        if n == 1:
            path = f"{self.log_dir}/{log_sub}/out-node0/bench_serv_script.log".replace("\\", "/")
            return orch.exec(f"tail -30 {shlex.quote(path)}")

        out: dict = {}
        for rank in range(n):
            host = hosts[rank]
            path = f"{self.log_dir}/{log_sub}/out-node{rank}/bench_serv_script.log".replace("\\", "/")
            cmd = f"tail -30 {shlex.quote(path)}"
            out.update(orch.exec(cmd, hosts=[host]))
        return out

    def poll_client_completion(self):
        log.info(f'Waiting for {self.default_client_wait_time} secs for benchmark scripts to start')
        time.sleep(self.default_client_wait_time)
        for j in range(0, self.default_client_poll_count):
            log.info(f'Polling for Benchmark script to complete on all nodes, iteration {j}')
            out_dict = self._tail_client_bench_logs()
            done = []
            for node in out_dict.keys():
                log_tail = out_dict[node] or ""
                if _BENCH_CLIENT_TRACEBACK_RE.search(log_tail):
                    msg = (
                        f"Benchmark client traceback on node {node} "
                        f"(see bench_serv_script.log); aborting before completion."
                    )
                    fail_test(msg)
                    raise Exception(msg)
                m_fail = _BENCH_FAILED_REQUESTS_RE.search(log_tail)
                if m_fail and int(m_fail.group(1)) > 0:
                    msg = (
                        f"Benchmark client on node {node} reports "
                        f"Failed requests: {m_fail.group(1)} (non-zero); see bench_serv_script.log."
                    )
                    fail_test(msg)
                    raise Exception(msg)
                done.append(
                    bool(
                        re.search(
                            r"Serving Benchmark Result|End-to-end Latency",
                            log_tail,
                            re.I,
                        )
                    )
                )
            if done and all(done):
                log.info("Benchmark client complete on all nodes (iter=%d)", j)
                return
            log.info(f"Waiting {self.default_client_poll_wait_time} secs for next poll")
            time.sleep(self.default_client_poll_wait_time)
        msg = "client did not complete before poll cap"
        fail_test(msg)
        raise Exception(msg)

    def get_inference_results_dict(self, out_dict):
        log.info('Get the inference results dict using get_inference_results_dict')
        self.inference_results_dict = {}
        for node in out_dict.keys():
            self.inference_results_dict[node] = {}
            if re.search('Successful requests:', out_dict[node], re.I):
                match = re.search(r'Successful requests:\s+([0-9]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['successful_requests'] = match.group(1)
            if re.search(r'Benchmark duration\s+\(s\):\s+([0-9]+)', out_dict[node], re.I):
                match = re.search(r'Benchmark duration\s+\(s\):\s+([0-9]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['benchmark_duration'] = match.group(1)
            if re.search('Total input tokens:', out_dict[node], re.I):
                match = re.search(r'Total input tokens:\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['total_input_tokens'] = match.group(1)
            if re.search('Total generated tokens:', out_dict[node], re.I):
                match = re.search(r'Total generated tokens:\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['total_generated_tokens'] = match.group(1)
            if re.search(r'Request throughput \(req/s\):', out_dict[node], re.I):
                match = re.search(r'Request throughput \(req/s\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['request_throughput_per_sec'] = match.group(1)
            if re.search(r'Output token throughput \(tok/s\):', out_dict[node], re.I):
                match = re.search(r'Output token throughput \(tok/s\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['output_throughput_per_sec'] = match.group(1)
            if re.search(r'Total Token throughput \(tok/s\):', out_dict[node], re.I):
                match = re.search(r'Total Token throughput \(tok/s\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['total_throughput_per_sec'] = match.group(1)
            if re.search(r'Mean TTFT \(ms\):', out_dict[node], re.I):
                match = re.search(r'Mean TTFT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['mean_ttft_ms'] = match.group(1)
            if re.search(r'Median TTFT \(ms\):', out_dict[node], re.I):
                match = re.search(r'Median TTFT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['median_ttft_ms'] = match.group(1)
            if re.search(r'P99 TTFT \(ms\):', out_dict[node], re.I):
                match = re.search(r'P99 TTFT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['p99_ttft_ms'] = match.group(1)
            if re.search(r'Mean TPOT \(ms\)', out_dict[node], re.I):
                match = re.search(r'Mean TPOT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['mean_tpot_ms'] = match.group(1)
            if re.search(r'Median TPOT \(ms\):', out_dict[node], re.I):
                match = re.search(r'Median TPOT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['median_tpot_ms'] = match.group(1)
            if re.search(r'P99 TPOT \(ms\):', out_dict[node], re.I):
                match = re.search(r'P99 TPOT \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['p99_tpot_ms'] = match.group(1)
            if re.search(r'Mean ITL \(ms\):', out_dict[node], re.I):
                match = re.search(r'Mean ITL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['mean_itl_ms'] = match.group(1)
            if re.search(r'Median ITL \(ms\):', out_dict[node], re.I):
                match = re.search(r'Median ITL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['median_itl_ms'] = match.group(1)
            if re.search(r'P99 ITL \(ms\):', out_dict[node], re.I):
                match = re.search(r'P99 ITL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['p99_itl_ms'] = match.group(1)
            if re.search(r'Mean E2EL \(ms\):', out_dict[node], re.I):
                match = re.search(r'Mean E2EL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['mean_e2el_ms'] = match.group(1)
            if re.search(r'Median E2EL \(ms\):', out_dict[node], re.I):
                match = re.search(r'Median E2EL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['median_e2el_ms'] = match.group(1)
            if re.search(r'P99 E2EL \(ms\):', out_dict[node], re.I):
                match = re.search(r'P99 E2EL \(ms\):\s+([0-9\.]+)', out_dict[node], re.I)
                self.inference_results_dict[node]['p99_e2el_ms'] = match.group(1)

        log.info("%s", self.inference_results_dict)
        return self.inference_results_dict

    def clone_inferencemax_repo(self):
        if self._using_host_server_scripts():
            self.s_phdl.exec(
                f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote('mkdir -p /workspace')}"
            )
            return
        repo = self.if_dict["inferencemax_repo"]
        clone_name = _clone_dirname_from_git_url(repo)
        inner = (
            "set -e; mkdir -p /app; cd /app; "
            "rm -rf InferenceMAX; "
            f"if [ ! -d {shlex.quote(clone_name)} ]; then git clone {shlex.quote(repo)}; fi"
        )
        cmd = f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote(inner)}"
        out_dict = self.s_phdl.exec(cmd)
        for node in out_dict.keys():
            out = out_dict[node] or ""
            if re.search("already exists", out, re.I):
                continue
            if _GIT_CLONE_FAIL_RE.search(out):
                fail_test("Errors or failures seen in pulling InferenceMAX repo from Github, pls check")
        time.sleep(3)
        ls_inner = f"ls -ld /app/{clone_name}"
        self.s_phdl.exec(
            f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote(ls_inner)}"
        )
        self.s_phdl.exec(
            f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote('mkdir -p /workspace')}"
        )

    def start_inference_server_job(self):
        self.clone_inferencemax_repo()
        super().start_inference_server_job()
