'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import re
import shlex
import time
from pathlib import Path
from typing import Optional

from cvs.lib import globals
from cvs.lib.inference.base import InferenceBaseJob
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

# Staged under {log_dir}/inference-max/host_scripts_bundled/ when ``benchmark_server_script_path``
# is auto and checkout ``cvs/input/.../benchmark_server_scripts`` is absent. Keep aligned with
# ``cvs/input/config_file/inference/inferencemax_single/mi300x_gpt_oss_120b_single/benchmark_server_scripts/gptoss_fp4_mi300x.sh``.
_BUNDLED_GPTOSS_FP4_MI300X_SH = r'''#!/usr/bin/env bash
# Server-only entrypoint for CVS InferenceMax + vLLM on MI300-class GPUs.
# Single entrypoint for this variant (legacy gptoss_fp4_mi300.sh alias removed).
# CVS runs: source /tmp/server_env_script.sh; nohup bash this_script ...
# The benchmark client is started separately by CVS (bench_serving).
#
# Default: --enforce-eager to reduce HIP "too many resources requested for launch"
# failures during CUDA graph capture on some ROCm + vLLM + AITER stacks.
# Disable with container env: VLLM_ENFORCE_EAGER=0

set -euo pipefail

: "${MODEL:?}" "${PORT:?}" "${TP:?}" "${MAX_MODEL_LEN:?}"

# Bash: ${VAR:-1} does not apply default when VAR is set-but-empty (e.g. from env_dict).
# Normalize whitespace-only to default, then drop the var so vLLM does not log
# "Unknown vLLM environment variable: VLLM_ENFORCE_EAGER" (only the CLI flag is official).
_raw_eager="${VLLM_ENFORCE_EAGER-}"
[[ -z "${_raw_eager//[[:space:]]/}" ]] && _raw_eager=1
ENFORCE_EAGER="${_raw_eager}"
unset VLLM_ENFORCE_EAGER || true

EAGER_FLAG=()
case "${ENFORCE_EAGER}" in
  1|true|TRUE|yes|YES) EAGER_FLAG=(--enforce-eager) ;;
  *) ;;
esac

GPU_MEM="${VLLM_GPU_MEMORY_UTIL:-0.92}"

echo "$(date -Is) [cvs gptoss_fp4_mi300x] vllm serve model=${MODEL} tp=${TP} max_model_len=${MAX_MODEL_LEN} port=${PORT} enforce_eager=${ENFORCE_EAGER} eager_flag_count=${#EAGER_FLAG[@]}"

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --block-size 64 \
  --no-enable-prefix-caching \
  "${EAGER_FLAG[@]}" \
  "$@"
'''


def _clone_dirname_from_git_url(repo_url: str) -> str:
    part = (repo_url or "").rstrip("/").rsplit("/", 1)[-1]
    return part.removesuffix(".git") if part.endswith(".git") else part or "InferenceX"


_REL_BENCH_SCRIPTS = Path(
    "input/config_file/inference/inferencemax_single/mi300x_gpt_oss_120b_single/benchmark_server_scripts"
)


def _gptoss_wrapper_dir_ok(d: Path) -> bool:
    return d.is_dir() and (d / "gptoss_fp4_mi300x.sh").is_file()


def _discover_host_benchmark_scripts_dir() -> Optional[Path]:
    """
    Return the directory containing gptoss_fp4_mi300x.sh inside the installed ``cvs`` tree.

    Checks the usual ``cvs/input/.../benchmark_server_scripts`` layout (editable installs and
    checkouts) and rglob under the package root. Plain ``pip install`` wheels often omit
    ``cvs/input/**`` because that tree is not a discovered subpackage; callers fall back to
    :func:`_stage_bundled_gptoss_wrapper`.
    """
    import cvs as _cvs_module

    root = Path(_cvs_module.__file__).resolve().parent
    candidates: list[Path] = [
        root / _REL_BENCH_SCRIPTS,
        Path(__file__).resolve().parents[2] / _REL_BENCH_SCRIPTS,
    ]
    try:
        from importlib.resources import files

        t = files("cvs").joinpath(*_REL_BENCH_SCRIPTS.parts)
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
        if _gptoss_wrapper_dir_ok(rp):
            return rp

    try:
        for sh in root.rglob("gptoss_fp4_mi300x.sh"):
            parent = sh.parent
            if parent.name == "benchmark_server_scripts" and _gptoss_wrapper_dir_ok(parent):
                return parent.resolve()
    except OSError:
        pass
    return None


def _stage_bundled_gptoss_wrapper(if_dict: dict) -> str:
    """Write bundled ``gptoss_fp4_mi300x.sh`` under ``{log_dir}/inference-max/host_scripts_bundled``."""
    log_dir = str(if_dict.get("log_dir") or "").strip().replace("\\", "/")
    if not log_dir:
        msg = (
            "benchmark_server_script_path is auto but host scripts were not found under the cvs "
            "install and log_dir is empty; cannot stage bundled gptoss_fp4_mi300x.sh."
        )
        fail_test(msg)
        raise FileNotFoundError(msg)
    dest = Path(log_dir) / "inference-max" / "host_scripts_bundled"
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fail_test(f"Cannot create {dest}: {exc}")
        raise
    script_path = dest / "gptoss_fp4_mi300x.sh"
    script_path.write_text(_BUNDLED_GPTOSS_FP4_MI300X_SH, encoding="utf-8", newline="\n")
    try:
        script_path.chmod(0o755)
    except OSError:
        pass
    log.info(
        "Using bundled gptoss_fp4_mi300x.sh staged at %s (checkout cvs/input/... not on this install)",
        dest,
    )
    return str(dest.resolve())


def _resolved_host_benchmark_scripts_dir(if_dict: dict) -> str:
    """
    Host directory containing gptoss_fp4_mi300x.sh (bind-mounted into the container).

    If ``benchmark_server_script_path`` is set to a non-empty string other than
    ``auto``, that path is used. Otherwise discover under the installed ``cvs`` package.
    """
    raw = if_dict.get("benchmark_server_script_path")
    if raw is not None:
        s = str(raw).strip()
        if s and s.lower() != "auto":
            return s.rstrip("/")

    found = _discover_host_benchmark_scripts_dir()
    if found is not None:
        return str(found)
    return _stage_bundled_gptoss_wrapper(if_dict)


class InferenceMaxJob(InferenceBaseJob):
    uses_local_server_scripts = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_dict.setdefault(
            'inferencemax_repo',
            'https://github.com/SemiAnalysisAI/InferenceX.git',
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
            return _resolved_host_benchmark_scripts_dir(self.if_dict).rstrip('/')
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

    def clone_bench_serving_repo(self, clone_dir):
        inner = f"cd {shlex.quote(clone_dir)} && git clone {shlex.quote(self.if_dict['benchmark_script_repo'])}"
        cmd = f"docker exec {shlex.quote(self.container_name)} /bin/bash -c {shlex.quote(inner)}"
        out_dict = self.c_phdl.exec(cmd)
        for node in out_dict.keys():
            out = out_dict[node] or ""
            if re.search("already exists", out, re.I):
                continue
            if _GIT_CLONE_FAIL_RE.search(out):
                fail_test("Errors or failures seen in pulling bench_serving repo from Github, pls check")
        time.sleep(3)

    def launch_server(self):
        script_dir = self.get_server_script_directory()
        log_file = f'{self.server_script}_server.log'
        script_path = self.get_server_script_path()

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
        readiness_pattern = self.readiness_pattern

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
            cmd_list = []
            for i in range(0, int(self.nnodes)):
                cmd = f'tail -30 {self.log_dir}/{self.get_log_subdir()}/out-node{i}/{log_file}'
                cmd_list.append(cmd)
            out_dict = self.s_phdl.exec_cmd_list(cmd_list)

            for node in out_dict.keys():
                if self.default_server_error_pattern_poll.search(out_dict[node] or ''):
                    error_msg = f'Failed to start server on node {node}: {out_dict[node][-_SERVER_LOG_ERROR_SNIPPET_CHARS:]}'
                    fail_test(error_msg)
                    raise Exception(error_msg)

            if self.is_server_ready(out_dict, readiness_pattern):
                log.info('Server startup confirmed on all nodes')
                return

            log.info(f'Waiting {self.default_server_poll_wait_time} secs for next poll')
            time.sleep(self.default_server_poll_wait_time)

        error_msg = 'Server did not report readiness before timeout; aborting startup'
        fail_test(error_msg)
        raise Exception(error_msg)

    def poll_client_completion(self):
        log.info(f'Waiting for {self.default_client_wait_time} secs for benchmark scripts to start')
        time.sleep(self.default_client_wait_time)
        for j in range(0, self.default_client_poll_count):
            log.info(f'Polling for Benchmark script to complete on all nodes, iteration {j}')
            cmd_list = []
            for i in range(0, int(self.nnodes)):
                cmd = f'tail -30 {self.log_dir}/{self.get_log_subdir()}/out-node{i}/bench_serv_script.log'
                cmd_list.append(cmd)
            out_dict = self.c_phdl.exec_cmd_list(cmd_list)
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
                if not re.search("End-to-end Latency", log_tail, re.I):
                    log.info(f"Waiting {self.default_client_poll_wait_time} secs for next poll")
                    time.sleep(self.default_client_poll_wait_time)

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
                self.inference_results_dict[node]['Total generated tokens:'] = match.group(1)
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
                match = re.search(r'Median TPOT \(ms\):\s+([0-9]+)', out_dict[node], re.I)
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
