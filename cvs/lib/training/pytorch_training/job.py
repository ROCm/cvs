"""Container-orchestrated PyTorch Training (LLM) job."""

from __future__ import annotations

import base64
import gzip
import json
import os
import re
import shlex
import sys
import time
import uuid
from pathlib import Path

from cvs.lib import globals
from cvs.lib.training.pytorch_training.utils.metrics import (
    BENCHMARK_TASKS,
    benchmark_scores,
    compute_scaling_efficiency,
    parse_codecarbon_metrics,
    to_training_metrics,
)
from cvs.lib.utils.gpu import agg_readings, start_gpu_poller, stop_and_collect_gpu_poller


log = globals.log

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
WARMUP_OK = "PYTORCH_TRAINING_WARMUP_OK"
RUN_OK = "PYTORCH_TRAINING_OK "
DATA_OK = "PYTORCH_TRAINING_DATA "
ENV_OK = "PYTORCH_TRAINING_ENV "
EVAL_OK = "PYTORCH_TRAINING_EVAL "


def _normalized_hardware_name(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _output_text(result):
    if isinstance(result, dict):
        return result.get("output") or result.get("stdout") or ""
    return str(result or "")


def _marker(text, prefix):
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    return None


class _BoundedHostHandle:
    def __init__(self, handle, expected_hosts):
        self._handle = handle
        self._expected_hosts = set(expected_hosts)

    def exec(self, command, print_console=False):
        # Kernel-log scans must stay out of the run log, so print_console is
        # accepted for the shared verify_lib caller and always forced off.
        del print_console
        if "dmesg --time-format iso -x" in command:
            command = (
                "set -o pipefail; sudo -n dmesg --time-format iso -x | "
                "awk '$0 !~ /ALLOWED|DENIED|IPVS: .*no destination available/'"
            )
        else:
            command = command.replace("sudo dmesg", "sudo -n dmesg").replace("| sudo tee ", "| sudo -n tee ")
        result = self._handle.exec(command, timeout=60, detailed=True, print_console=False)
        missing = self._expected_hosts - set(result or {})
        failures = {
            host: _output_text(host_result)[-1000:]
            for host, host_result in (result or {}).items()
            if not isinstance(host_result, dict) or host_result.get("exit_code") != 0
        }
        if missing or failures:
            raise RuntimeError(f"bounded host command failed: missing={sorted(missing)}, failures={failures}")
        return {host: _output_text(host_result) for host, host_result in result.items()}


class PyTorchTrainingJob:
    """Stage, launch, and collect one LLM training sweep cell."""

    def __init__(self, orch, variant, sweep_name):
        self.orch = orch
        self.variant = variant
        self.sweep_name = sweep_name
        self.sweep = variant.sweep(sweep_name)
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.run_id = run_id
        self.output_dir = f"{variant.paths.log_dir}/pytorch_training/{self.sweep.label}/{run_id}"
        self.result_path = f"{self.output_dir}/results.json"
        self.loss_series = []
        # Workload-scoped so two workloads staging onto the same host cannot
        # overwrite each other's script or kill each other's ranks.
        stem = f"/tmp/cvs_pytorch_training_{variant.training.workload.lower()}"
        self.train_path = f"{stem}.py"
        self.prepare_path = f"{stem}_prepare.py"
        self.log_glob = f"{shlex.quote(self.output_dir)}/training.node*.log"
        self.checkpoint_dir = f"{variant.training.checkpoint.directory}/{self.sweep.label}/{run_id}"
        self.export_dir = f"{self.checkpoint_dir}/hf_export"
        self.startup_retries_used = 0
        self.training_start_time = None
        self._dmesg_window = None
        self.monitor_metrics = {}
        self.train_tokens = ""
        self.eval_tokens = ""

    @property
    def workload(self):
        return self.variant.training.workload

    @property
    def num_nodes(self):
        return len(self.orch.hosts)

    @property
    def world_size(self):
        return self.variant.training.gpus_per_node * self.num_nodes

    @property
    def benchmarks_enabled(self):
        return self.variant.training.benchmarks.enabled

    def stage_scripts(self):
        # One exec per file: both payloads in one command exceed the SSH request limit.
        for name, target in (("train.py", self.train_path), ("prepare_data.py", self.prepare_path)):
            source = Path(__file__).with_name(name).read_bytes()
            encoded = base64.b64encode(gzip.compress(source, compresslevel=9)).decode("ascii")
            command = (
                f"mkdir -p {shlex.quote(self.output_dir)} && "
                f"printf %s {shlex.quote(encoded)} | base64 -d | gzip -d > {shlex.quote(target)}"
            )
            result = self.orch.exec(command, timeout=30, detailed=True, print_console=False)
            self._require_success(result, f"stage PyTorch Training {name}")

    def verify_environment(self):
        hosts = self.orch.hosts
        if self.variant.training.distributed:
            if len(hosts) < 2:
                raise RuntimeError(f"{self.workload} distributed requires 2+ nodes, received {len(hosts)}")
            self.prepare_host_verbs()
        elif len(hosts) != 1:
            raise RuntimeError(f"{self.workload} single-node requires exactly one node, received {len(hosts)}")
        probe = "python3 -c " + shlex.quote(
            "import json, torch, transformers, torchao; "
            "p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; "
            f"print({ENV_OK!r} + json.dumps({{"
            "'torch': torch.__version__, 'transformers': transformers.__version__, "
            "'torchao': torchao.__version__, 'hip': torch.version.hip, 'cuda': torch.cuda.is_available(), "
            "'gpus': torch.cuda.device_count(), 'devices': [x.name for x in p], "
            "'architectures': [getattr(x, 'gcnArchName', '') for x in p]}))"
        )
        result = self.orch.exec(probe, timeout=120, detailed=True, print_console=False)
        self._require_success(result, "verify PyTorch Training environment")
        summaries = []
        for host, host_result in result.items():
            marker = _marker(_output_text(host_result), ENV_OK)
            if marker is None:
                raise RuntimeError(f"environment probe returned no result marker on {host}")
            info = json.loads(marker)
            if not info["cuda"]:
                raise RuntimeError(f"ROCm GPU support is unavailable in the container on {host}")
            expected = self.variant.training.gpus_per_node
            if int(info["gpus"]) < expected:
                raise RuntimeError(f"{self.workload} requires {expected} visible GPUs on {host}, found {info['gpus']}")
            expected_arch = _normalized_hardware_name(self.variant.gpu_arch)
            mismatches = [name for name in info["devices"] if expected_arch not in _normalized_hardware_name(name)]
            if mismatches:
                raise RuntimeError(
                    f"{self.workload} config requires {self.variant.gpu_arch} on {host}, found devices {info['devices']}"
                )
            summaries.append(
                f"{host}: torch={info['torch']} transformers={info['transformers']} torchao={info['torchao']} "
                f"hip={info['hip']} gpus={info['gpus']} arch={info['architectures'][0]}"
            )
        return "; ".join(summaries)

    def prepare_host_verbs(self):
        """Install the host Thor2 verbs stack and open every configured rail.

        The container's libibverbs is older than the host kernel driver, and
        bind-mounting only the Broadcom provider then segfaults inside that
        library, so the host libibverbs (mounted at
        ``/opt/cvs-host/libibverbs.so.1``) replaces the container copy before
        any collective starts.
        """
        hca = str(self.variant.training.env_vars.get("NCCL_IB_HCA", "")).strip()
        names = [part.strip() for part in hca.split(",") if part.strip()]
        if not names or any(not re.fullmatch(r"(?:rdma|bnxt_re)\d+", name) for name in names):
            raise RuntimeError(f"distributed PyTorch Training requires NCCL_IB_HCA to name the RDMA rails; got {hca!r}")
        script = r"""
import ctypes, os, shutil
src = "/opt/cvs-host/libibverbs.so.1"
if not os.path.isfile(src) or os.path.getsize(src) == 0:
    raise SystemExit("host libibverbs is not mounted at /opt/cvs-host/libibverbs.so.1")
seen = set()
src_real = os.path.realpath(src)
for link in ("/usr/lib/x86_64-linux-gnu/libibverbs.so.1", "/lib/x86_64-linux-gnu/libibverbs.so.1"):
    if not os.path.lexists(link):
        continue
    target = os.path.realpath(link)
    if target in seen or target == src_real:
        continue
    seen.add(target)
    shutil.copyfile(src, target)
expected = [name for name in os.environ["NCCL_IB_HCA"].split(",") if name]
lib = ctypes.CDLL("libibverbs.so.1")
lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
lib.ibv_get_device_name.restype = ctypes.c_char_p
lib.ibv_open_device.argtypes = [ctypes.c_void_p]
lib.ibv_open_device.restype = ctypes.c_void_p
lib.ibv_close_device.argtypes = [ctypes.c_void_p]
count = ctypes.c_int()
devices = lib.ibv_get_device_list(ctypes.byref(count))
if not devices:
    raise SystemExit("ibv_get_device_list returned no devices")
opened = []
for index in range(count.value):
    name = lib.ibv_get_device_name(devices[index]).decode()
    context = lib.ibv_open_device(devices[index])
    if not context:
        raise SystemExit(f"ibv_open_device failed for {name}")
    lib.ibv_close_device(context)
    opened.append(name)
missing = [name for name in expected if name not in opened]
if missing:
    raise SystemExit(f"RDMA rails not open: {missing}; opened {sorted(opened)}")
print("RDMA_RAILS " + " ".join(sorted(opened)))
"""
        command = "export NCCL_IB_HCA=" + shlex.quote(",".join(names)) + "; python3 -c " + shlex.quote(script)
        result = self.orch.exec(command, timeout=60, detailed=True, print_console=False)
        self._require_success(result, "prepare host verbs for the RDMA rails")

    def prepare_data(self):
        """Tokenize the corpus with this sweep's tokenizer, or reuse the cached copy."""
        if self.sweep.data_mode != "tokens":
            return "synthetic tokens"
        data = self.variant.training.data
        splits = [data.train_split]
        if self.variant.training.eval_enabled:
            splits.append(data.eval_split)
        command = " ".join(
            shlex.quote(part)
            for part in [
                "python3",
                self.prepare_path,
                "--source",
                data.source,
                "--tokenizer",
                self.sweep.model_path,
                "--cache-dir",
                data.cache_dir,
                "--splits",
                ",".join(splits),
                "--text-column",
                data.text_column,
                *(["--append-eos"] if data.append_eos else []),
            ]
        )
        hosts = list(self.orch.hosts)
        if len(hosts) > 1:
            # The cache is shared: one node tokenizes and the rest reuse it,
            # rather than every node writing its own full temporary copy.
            first = self.orch.exec(
                command, hosts=hosts[:1], timeout=data.prepare_timeout_s, detailed=True, print_console=False
            )
            self._require_success(first, f"prepare token data for {self.sweep_name} on {hosts[0]}")
        result = self.orch.exec(command, timeout=data.prepare_timeout_s, detailed=True, print_console=False)
        self._require_success(result, f"prepare token data for {self.sweep_name}")
        summaries = []
        for host, host_result in result.items():
            marker = _marker(_output_text(host_result), DATA_OK)
            if marker is None:
                raise RuntimeError(f"data preparation returned no result marker on {host}")
            info = json.loads(marker)
            self.train_tokens = info["splits"][data.train_split]["path"]
            if self.variant.training.eval_enabled:
                self.eval_tokens = info["splits"][data.eval_split]["path"]
            counts = ", ".join(f"{k}={v['tokens']}" for k, v in info["splits"].items())
            summaries.append(f"{host}: tokenizer {info['tokenizer_digest']} {counts}")
        return "; ".join(summaries)

    def _rendezvous_args(self, node_rank):
        training = self.variant.training
        if not training.distributed:
            return ["--standalone", "--nnodes=1"]
        return [
            f"--nnodes={self.num_nodes}",
            f"--node-rank={node_rank}",
            f"--master-addr={self.orch.hosts[0]}",
            f"--master-port={training.master_port}",
        ]

    def build_command(self, node_rank=0):
        training = self.variant.training
        sweep = self.sweep
        args = [
            "torchrun",
            *self._rendezvous_args(node_rank),
            f"--nproc-per-node={training.gpus_per_node}",
            self.train_path,
            "--workload",
            self.workload,
            "--model",
            sweep.model,
            "--model-path",
            sweep.model_path,
            "--init",
            sweep.init,
            "--precision",
            sweep.precision,
            "--seq-len",
            str(sweep.seq_len),
            "--micro-batch-size",
            str(sweep.micro_batch_size),
            "--gradient-accumulation-steps",
            str(sweep.accumulation_steps(self.world_size)),
            "--activation-checkpointing-every",
            str(sweep.activation_checkpointing_every),
            "--master-weights",
            sweep.master_weights,
            "--attn-implementation",
            sweep.attn_implementation,
            "--loss-chunk-tokens",
            str(sweep.loss_chunk_tokens),
            "--data-mode",
            sweep.data_mode,
            "--warmup-steps",
            str(training.warmup_steps),
            "--steps",
            str(training.steps),
            "--seed",
            str(training.seed),
            "--run-mode",
            training.run_mode,
            "--phase",
            training.phase,
            "--eval-every-steps",
            str(training.eval_every_steps),
            "--eval-batches",
            str(training.eval_batches),
            "--milestone-steps",
            ",".join(str(step) for step in training.milestone_steps),
            "--learning-rate",
            str(training.learning_rate),
            "--min-learning-rate-ratio",
            str(training.min_learning_rate_ratio),
            "--lr-schedule",
            training.lr_schedule,
            "--lr-warmup-steps",
            str(training.lr_warmup_steps),
            "--weight-decay",
            str(training.weight_decay),
            "--adam-beta1",
            str(training.adam_beta1),
            "--adam-beta2",
            str(training.adam_beta2),
            "--grad-clip-norm",
            str(training.grad_clip_norm),
            "--fsdp-strategy",
            training.fsdp_strategy,
            "--gpus-per-node",
            str(training.gpus_per_node),
            "--peak-tflops-per-gpu",
            str(training.peak_tflops_per_gpu[sweep.precision]),
            "--checkpoint-dir",
            self.checkpoint_dir,
            "--resume-steps",
            str(training.checkpoint.resume_steps),
            "--loss-curve-sample-every",
            str(training.loss_curve.sample_every_steps),
            "--loss-curve-minimum-points",
            str(training.loss_curve.minimum_points),
            "--output",
            self.result_path,
            "--collective-timeout-seconds",
            str(min(600, training.timeout_s)),
            "--startup-timeout-seconds",
            str(training.startup_timeout_s),
            "--setup-timeout-seconds",
            str(training.setup_timeout_s),
        ]
        if sweep.compile:
            args.append("--compile")
        if sweep.data_mode == "tokens":
            args.extend(["--train-tokens", self.train_tokens])
        if training.eval_enabled:
            args.extend(["--eval-enabled", "--eval-tokens", self.eval_tokens])
        if training.max_duration_seconds is not None:
            args.extend(["--max-duration-seconds", str(training.max_duration_seconds)])
        if sweep.training_flops_per_token is not None:
            args.extend(["--training-flops-per-token", str(sweep.training_flops_per_token)])
        if training.convergence.target_train_loss is not None:
            args.extend(["--convergence-train-loss", str(training.convergence.target_train_loss)])
        if training.convergence.target_eval_loss is not None:
            args.extend(["--convergence-eval-loss", str(training.convergence.target_eval_loss)])
        if training.checkpoint.keep:
            args.append("--keep-checkpoint")
        if self.benchmarks_enabled:
            args.extend(["--export-hf-dir", self.export_dir])
        if training.codecarbon.enabled:
            args.extend(
                ["--codecarbon-enabled", "--codecarbon-measure-power-secs", str(training.codecarbon.measure_power_secs)]
            )
            if training.codecarbon.required:
                args.append("--codecarbon-required")
            if training.codecarbon.country_iso_code:
                args.extend(["--codecarbon-country-iso-code", training.codecarbon.country_iso_code])

        exports = {
            "OMP_NUM_THREADS": str(training.omp_num_threads),
            "PYTHONUNBUFFERED": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            **training.env_vars,
            # Written last so a multi-node launch cannot inherit a stale
            # single-node NNODES/NODE_RANK from training.env_vars.
            "NNODES": str(self.num_nodes),
            "NODE_RANK": str(node_rank),
        }
        export_lines = []
        for name, value in exports.items():
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
            export_lines.append(f"export {name}={shlex.quote(str(value))}")
        launch = " ".join(shlex.quote(str(arg)) for arg in args)
        return (
            "set -o pipefail; "
            + "; ".join(export_lines)
            + f"; rm -f {shlex.quote(self.result_path)}; "
            + f"timeout --signal=TERM --kill-after=30s {training.timeout_s}s {launch} "
            + f"2>&1 | tee {shlex.quote(self.log_path(node_rank))}"
        )

    def log_path(self, node_rank=0):
        return f"{self.output_dir}/training.node{node_rank}.log"

    def run_benchmark(self):
        if not self.variant.training.enabled:
            raise RuntimeError("this PyTorch Training profile is disabled")
        retries = self.variant.training.startup_retries
        poller = start_gpu_poller(
            self.orch,
            self.sweep.label,
            poll_interval_s=2.0,
            hard_cap_s=(self.variant.training.timeout_s + 120) * (retries + 1),
        )
        result = None
        try:
            for attempt in range(retries + 1):
                if self.variant.training.verify_dmesg:
                    self._open_dmesg_window(attempt + 1)
                result = self._launch()
                if attempt == retries or not self._failed_before_measuring(result):
                    break
                self._keep_failed_start(attempt + 1)
                log.warning(
                    "%s: launch ended before warmup finished on attempt %d; relaunching", self.sweep_name, attempt + 1
                )
            self.startup_retries_used = attempt
        finally:
            readings = stop_and_collect_gpu_poller(self.orch, poller)
            try:
                self._set_monitor_metrics(readings)
            except Exception:
                # Never let a telemetry failure mask the training error that
                # very likely caused it.
                if sys.exc_info()[0] is None:
                    raise
                log.warning("AMD-SMI telemetry unusable for %s; preserving the primary error", self.sweep_name)
        self._require_success(result, f"run PyTorch Training sweep {self.sweep_name}")

    def _launch(self):
        timeout = self.variant.training.timeout_s + 60
        if self.variant.training.distributed:
            # cmd_list[i] runs on hosts[i], which is the only way each node
            # receives its own --node-rank. This path has no per-host exit
            # codes; a failed rank is caught by the log scan and by
            # parse_results rejecting a missing artifact.
            return self.orch.exec_cmd_list(
                [self.build_command(node_rank=i) for i in range(self.num_nodes)], timeout=timeout
            )
        return self.orch.exec(self.build_command(), timeout=timeout, detailed=True)

    @staticmethod
    def _failed_before_measuring(result):
        """True when the launch ended before the measured window began.

        Rank 0 prints WARMUP_OK as the measured window opens, so a failure
        inside that window is never relaunched.
        """
        outputs = [_output_text(host_result) for host_result in (result or {}).values()]
        return bool(outputs) and not any(WARMUP_OK in text or RUN_OK in text for text in outputs)

    def _keep_failed_start(self, attempt):
        self.stop_training_processes()
        for node_rank in range(self.num_nodes):
            source = self.log_path(node_rank)
            target = f"{self.output_dir}/startup_failure{attempt}.node{node_rank}.log"
            self.orch.exec(
                f"if [ -f {shlex.quote(source)} ]; then mv -f {shlex.quote(source)} {shlex.quote(target)}; fi",
                timeout=30,
                detailed=True,
                print_console=False,
            )

    def _set_monitor_metrics(self, readings):
        if not readings:
            raise RuntimeError("AMD-SMI memory tracking produced no valid samples")
        peak_used = agg_readings(readings).get("peak_gpu_memory_per_device_mb")
        if peak_used is None:
            raise RuntimeError("AMD-SMI memory tracking did not report per-device used memory")
        self.monitor_metrics = {"training.peak_memory_used_mb": peak_used}

    def _open_dmesg_window(self, attempt):
        from cvs.lib.verify_lib import DmesgWindow

        label = f"PyTorch Training {self.sweep_name} {self.run_id} attempt {attempt}"
        handle = getattr(self.orch, "all", None)
        if handle is None or not hasattr(handle, "exec"):
            raise RuntimeError("container orchestrator does not expose a host execution handle")
        self._dmesg_window = DmesgWindow(_BoundedHostHandle(handle, self.orch.hosts), label, use_markers=True)
        self._dmesg_window.open()
        self.training_start_time = self._dmesg_window.start_time

    def scan_dmesg_for_errors(self):
        if not self.variant.training.verify_dmesg:
            return
        if self._dmesg_window is None:
            raise RuntimeError("cannot verify dmesg without a training start marker")
        self._dmesg_window.close_and_scan()

    def verify_training_log(self):
        patterns = self.variant.training.error_patterns
        if not patterns:
            return
        result = self.orch.exec(f"cat {self.log_glob}", timeout=30, detailed=True, print_console=False)
        self._require_success(result, f"read training log for {self.sweep_name}")
        matches = {}
        for host, host_result in result.items():
            text = _output_text(host_result)
            found = [name for name, pattern in patterns.items() if re.search(pattern, text, re.IGNORECASE)]
            if found:
                matches[host] = found
        if matches:
            raise RuntimeError(f"training log matched configured failure patterns: {matches}")

    def run_checkpoint_benchmarks(self):
        """Score the exported checkpoint with lm-evaluation-harness on the first node.

        Data parallel across the node's GPUs when one GPU holds the model;
        otherwise one process spreads the model over every GPU.
        """
        if not self.benchmarks_enabled:
            return {}
        config = self.variant.training.benchmarks
        tasks = sorted({BENCHMARK_TASKS[metric][0] for metric in config.metrics})
        out_dir = f"{self.output_dir}/lm_eval"
        model_args = f"pretrained={self.export_dir},dtype=bfloat16"
        if config.data_parallel:
            launcher = ["accelerate", "launch", "--multi_gpu", f"--num_processes={self.variant.training.gpus_per_node}"]
            launcher += ["-m", "lm_eval"]
        else:
            launcher = ["lm_eval"]
            model_args += ",parallelize=True"
        args = [
            *launcher,
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            ",".join(tasks),
            "--batch_size",
            config.batch_size,
            "--output_path",
            out_dir,
        ]
        if config.limit is not None:
            args.extend(["--limit", str(config.limit)])
        exports = " ".join(
            f"{name}={shlex.quote(str(value))}"
            for name, value in self.variant.training.env_vars.items()
            if name.startswith(("HF_", "TRANSFORMERS_"))
        )
        report = f"import json,sys; print({EVAL_OK!r} + json.dumps(json.load(open(sys.argv[1]))['results']))"
        command = (
            f"set -o pipefail; mkdir -p {shlex.quote(out_dir)}; "
            f"env {exports} timeout {config.timeout_s}s {' '.join(shlex.quote(a) for a in args)} "
            f"2>&1 | tee {shlex.quote(self.output_dir)}/lm_eval.log; "
            f"f=$(ls -t $(find {shlex.quote(out_dir)} -name 'results_*.json') | head -1); "
            f"python3 -c {shlex.quote(report)} \"$f\""
        )
        first = self.orch.hosts[0]
        timeout = config.timeout_s + 300
        if self.num_nodes > 1:
            # Only rank 0's node holds the export; the others run a no-op.
            result = self.orch.exec_cmd_list([command] + ["true"] * (self.num_nodes - 1), timeout=timeout)
        else:
            result = self.orch.exec(command, timeout=timeout, detailed=True, print_console=False)
            host_result = (result or {}).get(first)
            if not isinstance(host_result, dict) or host_result.get("exit_code") != 0:
                raise RuntimeError(f"lm-eval failed on {first}: {_output_text(host_result)[-1500:]}")
        marker = _marker(_output_text((result or {}).get(first)), EVAL_OK)
        if marker is None:
            raise RuntimeError(f"lm-eval returned no results on {first}")
        scores = benchmark_scores(json.loads(marker))
        missing = sorted(set(config.metrics) - set(scores))
        if missing:
            raise RuntimeError(f"lm-eval did not report {missing}")
        return {f"training.{metric}": scores[metric] for metric in config.metrics}

    def parse_results(self):
        result = self.orch.exec(f"cat {shlex.quote(self.result_path)}", timeout=30, detailed=True, print_console=False)
        self._require_success(result, f"read result artifact for {self.sweep_name}")
        parsed = {}
        for host, host_result in result.items():
            text = _output_text(host_result).strip()
            if not text:
                raise RuntimeError(f"empty result artifact on {host}: {self.result_path}")
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid result artifact on {host}: {self.result_path}") from exc
            ga = self.sweep.accumulation_steps(self.world_size)
            expected = {
                "workload": self.workload,
                "model": self.sweep.model,
                "precision": self.sweep.precision,
                "init": self.sweep.init,
                "seq_len": self.sweep.seq_len,
                "micro_batch_size": self.sweep.micro_batch_size,
                "gradient_accumulation_steps": ga,
                "global_batch_size": self.sweep.micro_batch_size * ga * self.world_size,
                "world_size": self.world_size,
                "num_nodes": self.num_nodes,
                "phase": self.variant.training.phase,
                "run_mode": self.variant.training.run_mode,
                "data_mode": self.sweep.data_mode,
            }
            mismatches = {
                key: {"expected": value, "actual": raw.get(key)}
                for key, value in expected.items()
                if raw.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"unexpected {self.workload} result metadata on {host}: {mismatches}")
            if self.variant.training.run_mode != "soak" and raw.get("completed_steps") != self.variant.training.steps:
                if self.variant.training.max_duration_seconds is None:
                    raise RuntimeError(
                        f"incomplete {self.workload} run on {host}: completed {raw.get('completed_steps')} of "
                        f"{self.variant.training.steps} steps"
                    )
            metrics = to_training_metrics(raw)
            if self.variant.training.codecarbon.enabled:
                metrics.update(
                    parse_codecarbon_metrics(raw.get("codecarbon") or {}, self.variant.training.gpus_per_node)
                )
            metrics.update(self.monitor_metrics)
            energy_kwh = metrics.get("training.energy_kwh")
            processed = int(raw.get("processed_tokens") or 0)
            if energy_kwh is not None and energy_kwh > 0:
                if processed <= 0:
                    raise RuntimeError(f"{self.workload} artifact has no positive processed_tokens count on {host}")
                # CodeCarbon measures rank 0's node only, so tokens are scaled
                # to that node's share to keep tokens/kWh a per-node figure.
                metrics["training.tokens_per_kwh"] = processed / self.num_nodes / energy_kwh
            baseline = self.variant.scaling_baseline_for(self.sweep_name)
            efficiency = compute_scaling_efficiency(
                metrics.get("training.tokens_per_sec"),
                self.num_nodes,
                baseline.tokens_per_sec_total,
                baseline.num_nodes,
            )
            if efficiency is not None:
                metrics["training.scaling_efficiency_pct"] = efficiency
            if baseline.eval_loss is not None and "training.eval_loss" in metrics:
                metrics["training.scale_parity_eval_loss_delta"] = metrics["training.eval_loss"] - baseline.eval_loss
            parsed[host] = metrics
            self.loss_series = [
                (int(p["step"]), float(p["loss"]))
                for p in (raw.get("loss_time_series") or [])
                if p.get("step") is not None and p.get("loss") is not None
            ]
        return parsed

    def cleanup_checkpoints(self):
        if self.variant.training.checkpoint.keep:
            return
        parent = shlex.quote(os.path.dirname(self.checkpoint_dir))
        self.orch.exec(
            f"rm -rf {shlex.quote(self.checkpoint_dir)}; rmdir {parent} 2>/dev/null; true",
            timeout=600,
            detailed=True,
            print_console=False,
        )

    def stop_training_processes(self):
        # Bracket the first character so pgrep/pkill cannot match themselves.
        name = os.path.basename(self.train_path)
        pattern = f"[{name[0]}]{name[1:]}"
        command = (
            f"pkill -TERM -f {shlex.quote(pattern)} 2>/dev/null || true; "
            "sleep 2; "
            f"pkill -KILL -f {shlex.quote(pattern)} 2>/dev/null || true; "
            "sleep 1; "
            f"! pgrep -f {shlex.quote(pattern)} >/dev/null"
        )
        result = self.orch.exec(command, timeout=15, detailed=True, print_console=False)
        self._require_success(result, f"stop PyTorch Training processes for {self.sweep_name}")

    @staticmethod
    def _require_success(result, action):
        if not result:
            raise RuntimeError(f"{action} returned no host results")
        failures = {}
        for host, host_result in result.items():
            exit_code = host_result.get("exit_code", 1) if isinstance(host_result, dict) else 0
            if exit_code != 0:
                failures[host] = _output_text(host_result)[-1500:]
        if failures:
            raise RuntimeError(f"{action} failed: {failures}")
