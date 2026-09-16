"""Container-orchestrated PyTorch Vision training job."""

from __future__ import annotations

import base64
import gzip
import json
import math
import re
import shlex
import time
import uuid
from pathlib import Path

from cvs.lib.training.pytorch_vision.utils.metrics import parse_codecarbon_metrics, to_training_metrics
from cvs.lib.utils.gpu import agg_readings, start_gpu_poller, stop_and_collect_gpu_poller


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalized_hardware_name(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _output_text(result):
    if isinstance(result, dict):
        return result.get("output") or result.get("stdout") or ""
    return str(result or "")


class _BoundedHostHandle:
    def __init__(self, handle, expected_hosts):
        self._handle = handle
        self._expected_hosts = set(expected_hosts)

    def exec(self, command):
        if "dmesg --time-format iso -x" in command:
            command = (
                "set -o pipefail; sudo -n dmesg --time-format iso -x | "
                "awk '$0 !~ /ALLOWED|DENIED|IPVS: .*no destination available/'"
            )
        else:
            command = command.replace("sudo dmesg", "sudo -n dmesg")
        result = self._handle.exec(
            command,
            timeout=60,
            detailed=True,
            print_console=False,
        )
        missing = self._expected_hosts - set(result or {})
        failures = {
            host: _output_text(host_result)[-1000:]
            for host, host_result in (result or {}).items()
            if not isinstance(host_result, dict) or host_result.get("exit_code") != 0
        }
        if missing or failures:
            raise RuntimeError(f"bounded host command failed: missing={sorted(missing)}, failures={failures}")
        return {host: _output_text(host_result) for host, host_result in result.items()}


class PyTorchVisionJob:
    """Stage, launch, and collect one vision-training sweep cell."""

    BENCHMARK_PATH = "/tmp/cvs_pytorch_vision_w1.py"

    def __init__(self, orch, variant, sweep_name):
        self.orch = orch
        self.variant = variant
        self.sweep_name = sweep_name
        self.sweep = variant.sweep(sweep_name)
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.output_dir = f"{variant.paths.log_dir}/pytorch_vision/{self.sweep.label}/{run_id}"
        self.result_path = f"{self.output_dir}/results.json"
        self.loss_series = []
        self.log_path = f"{self.output_dir}/training.log"
        self.checkpoint_path = f"{self.output_dir}/checkpoint.pt"
        self.training_start_time = None
        self.monitor_metrics = {}
        self.run_elapsed_seconds = None

    def stage_benchmark(self):
        source = Path(__file__).with_name("benchmark.py").read_bytes()
        encoded = base64.b64encode(gzip.compress(source, compresslevel=9)).decode("ascii")
        command = (
            f"mkdir -p {shlex.quote(self.output_dir)} && "
            f"printf %s {shlex.quote(encoded)} | base64 -d | gzip -d > {shlex.quote(self.BENCHMARK_PATH)}"
        )
        result = self.orch.exec(command, timeout=30, detailed=True, print_console=False)
        self._require_success(result, "stage PyTorch Vision benchmark")

    def verify_environment(self):
        if len(self.orch.hosts) != 1:
            raise RuntimeError(f"W1 requires exactly one node, received {len(self.orch.hosts)}")

        probe = "python3 -c " + shlex.quote(
            "import json, torch, torchvision; "
            "p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; "
            "print('PYTORCH_VISION_ENV ' + json.dumps({"
            "'torch': torch.__version__, 'torchvision': torchvision.__version__, "
            "'hip': torch.version.hip, 'cuda': torch.cuda.is_available(), "
            "'gpus': torch.cuda.device_count(), 'devices': [x.name for x in p], "
            "'architectures': [getattr(x, 'gcnArchName', '') for x in p]}))"
        )
        result = self.orch.exec(probe, timeout=60, detailed=True, print_console=False)
        self._require_success(result, "verify PyTorch Vision environment")

        summaries = []
        for host, host_result in result.items():
            marker = next(
                (
                    line.removeprefix("PYTORCH_VISION_ENV ")
                    for line in _output_text(host_result).splitlines()
                    if line.startswith("PYTORCH_VISION_ENV ")
                ),
                None,
            )
            if marker is None:
                raise RuntimeError(f"environment probe returned no result marker on {host}")
            info = json.loads(marker)
            if not info["cuda"]:
                raise RuntimeError(f"ROCm GPU support is unavailable in the container on {host}")
            expected = self.variant.training.gpus_per_node
            if int(info["gpus"]) != expected:
                raise RuntimeError(f"W1 requires {expected} visible GPUs on {host}, found {info['gpus']}")
            expected_arch = _normalized_hardware_name(self.variant.gpu_arch)
            mismatches = [name for name in info["devices"] if expected_arch not in _normalized_hardware_name(name)]
            if mismatches:
                raise RuntimeError(
                    f"W1 config requires {self.variant.gpu_arch} on {host}, found devices {info['devices']}"
                )
            summaries.append(
                f"{host}: torch={info['torch']} torchvision={info['torchvision']} "
                f"hip={info['hip']} gpus={info['gpus']} arch={info['architectures'][0]}"
            )
        return "; ".join(summaries)

    def build_command(self):
        training = self.variant.training
        sweep = self.sweep
        args = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={training.gpus_per_node}",
            self.BENCHMARK_PATH,
            "--model",
            sweep.model,
            "--backend",
            sweep.backend,
            "--precision",
            sweep.precision,
            "--batch-size",
            str(sweep.batch_size),
            "--image-size",
            str(sweep.image_size),
            "--num-classes",
            str(training.num_classes),
            "--warmup-steps",
            str(training.warmup_steps),
            "--measure-steps",
            str(training.steps),
            "--run-mode",
            training.run_mode,
            "--phase",
            training.phase,
            "--eval-every-epochs",
            str(training.eval_every_epochs),
            "--eval-sample-count",
            str(training.eval_sample_count),
            "--milestone-steps",
            ",".join(str(step) for step in training.milestone_steps),
            "--learning-rate",
            str(training.learning_rate),
            "--momentum",
            str(training.momentum),
            "--weight-decay",
            str(training.weight_decay),
            "--lr-schedule",
            training.lr_schedule.name,
            "--lr-warmup-epochs",
            str(training.lr_schedule.warmup_epochs),
            "--lr-milestones-epochs",
            ",".join(str(epoch) for epoch in training.lr_schedule.milestones_epochs),
            "--lr-gamma",
            str(training.lr_schedule.gamma),
            "--gradient-accumulation-steps",
            str(sweep.gradient_accumulation_steps),
            "--training-flops-per-image",
            str(sweep.training_flops_per_image),
            "--data-mode",
            sweep.data_mode,
            "--rocal-device",
            sweep.rocal_device,
            "--augmentation",
            sweep.augmentation,
            "--rocal-num-threads",
            str(sweep.rocal_num_threads),
            "--loader-warmup-steps",
            str(sweep.loader_warmup_steps),
            "--loader-benchmark-steps",
            str(sweep.loader_benchmark_steps),
            "--peak-tflops-per-gpu",
            str(training.peak_tflops_per_gpu),
            "--output",
            self.result_path,
            "--loss-curve-sample-every",
            str(training.loss_curve.sample_every_steps),
            "--loss-curve-minimum-points",
            str(training.loss_curve.minimum_points),
            "--collective-timeout-seconds",
            str(min(120, training.timeout_s)),
        ]
        if training.max_epochs is not None:
            args.extend(["--max-epochs", str(training.max_epochs)])
        if training.max_duration_seconds is not None:
            args.extend(["--max-duration-seconds", str(training.max_duration_seconds)])
        if training.eval_enabled:
            args.append("--eval-enabled")
        if training.eval_steps is not None:
            args.extend(["--eval-steps", str(training.eval_steps)])
        if training.convergence.target_top1_pct is not None:
            args.extend(["--convergence-top1", str(training.convergence.target_top1_pct)])
        if training.convergence.target_eval_loss is not None:
            args.extend(["--convergence-eval-loss", str(training.convergence.target_eval_loss)])
        if training.accuracy.target_top5_pct is not None:
            args.extend(["--target-top5", str(training.accuracy.target_top5_pct)])
        if training.convergence.stop_when_reached:
            args.append("--stop-on-convergence")
        if training.codecarbon.enabled:
            args.extend(
                [
                    "--codecarbon-enabled",
                    "--codecarbon-measure-power-secs",
                    str(training.codecarbon.measure_power_secs),
                ]
            )
            if training.codecarbon.required:
                args.append("--codecarbon-required")
            if training.codecarbon.country_iso_code:
                args.extend(["--codecarbon-country-iso-code", training.codecarbon.country_iso_code])
        if training.checkpoint_enabled:
            args.extend(
                [
                    "--checkpoint-path",
                    self.checkpoint_path,
                    "--checkpoint-loss-tolerance",
                    str(training.checkpoint_loss_tolerance),
                ]
            )
            if training.checkpoint_keep_file:
                args.append("--keep-checkpoint")
        if training.channels_last:
            args.append("--channels-last")
        if sweep.dataset_path:
            args.extend(["--dataset-path", sweep.dataset_path])

        exports = {
            "OMP_NUM_THREADS": str(training.omp_num_threads),
            "PYTHONUNBUFFERED": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            **training.env_vars,
        }
        export_lines = []
        for name, value in exports.items():
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
            export_lines.append(f"export {name}={shlex.quote(str(value))}")

        launch = " ".join(shlex.quote(str(arg)) for arg in args)
        timeout = training.timeout_s
        return (
            "set -o pipefail; "
            + "; ".join(export_lines)
            + f"; rm -f {shlex.quote(self.result_path)}; "
            + f"timeout --signal=TERM --kill-after=30s {timeout}s {launch} "
            + f"2>&1 | tee {shlex.quote(self.log_path)}"
        )

    def run_benchmark(self):
        if not self.variant.training.enabled:
            raise RuntimeError("this PyTorch Vision profile is disabled")
        if self.variant.training.verify_dmesg:
            self.training_start_time = self._host_date()
        poller = start_gpu_poller(
            self.orch,
            self.sweep.label,
            poll_interval_s=1.0,
            hard_cap_s=self.variant.training.timeout_s + 120,
        )
        started = time.monotonic()
        try:
            result = self.orch.exec(
                self.build_command(),
                timeout=self.variant.training.timeout_s + 60,
                detailed=True,
            )
        finally:
            self.run_elapsed_seconds = time.monotonic() - started
            readings = stop_and_collect_gpu_poller(self.orch, poller)
            self._set_monitor_metrics(readings)
        self._require_success(result, f"run PyTorch Vision sweep {self.sweep_name}")

    def _set_monitor_metrics(self, readings):
        if not readings:
            raise RuntimeError("AMD-SMI memory tracking produced no valid samples")
        aggregated = agg_readings(readings)
        peak_used = aggregated.get("peak_gpu_memory_per_device_mb")
        if peak_used is None:
            raise RuntimeError("AMD-SMI memory tracking did not report per-device used memory")
        self.monitor_metrics = {"training.peak_memory_used_mb": peak_used}

    def scan_dmesg_for_errors(self):
        if not self.variant.training.verify_dmesg:
            return
        if not self.training_start_time:
            raise RuntimeError("cannot verify dmesg without a training start time")
        from cvs.lib.verify_lib import verify_dmesg_for_errors

        verify_dmesg_for_errors(
            _BoundedHostHandle(self.orch.all, self.orch.hosts),
            self.training_start_time,
            self._host_date(),
            till_end_flag=False,
        )

    def verify_training_log(self):
        patterns = self.variant.training.error_patterns
        if not patterns:
            return
        result = self.orch.exec(
            f"cat {shlex.quote(self.log_path)}",
            timeout=30,
            detailed=True,
            print_console=False,
        )
        self._require_success(result, f"read training log for {self.sweep_name}")
        matches = {}
        for host, host_result in result.items():
            text = _output_text(host_result)
            found = [name for name, pattern in patterns.items() if re.search(pattern, text, re.IGNORECASE)]
            if found:
                matches[host] = found
        if matches:
            raise RuntimeError(f"training log matched configured failure patterns: {matches}")

    def parse_results(self):
        result = self.orch.exec(
            f"cat {shlex.quote(self.result_path)}",
            timeout=30,
            detailed=True,
            print_console=False,
        )
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
            expected_metadata = {
                "workload": "W1",
                "model": self.sweep.model,
                "backend": self.sweep.backend,
                "precision": self.sweep.precision,
                "image_size": self.sweep.image_size,
                "batch_size_per_gpu": self.sweep.batch_size,
                "gradient_accumulation_steps": self.sweep.gradient_accumulation_steps,
                "effective_global_batch_size": (
                    self.sweep.batch_size * self.sweep.gradient_accumulation_steps * self.variant.training.gpus_per_node
                ),
                "world_size": self.variant.training.gpus_per_node,
                "phase": self.variant.training.phase,
                "run_mode": self.variant.training.run_mode,
                "synthetic_data": self.sweep.data_mode == "synthetic",
                "data_mode": self.sweep.data_mode,
                "rocal_device": self.sweep.rocal_device,
                "augmentation": self.sweep.augmentation,
                "lr_schedule": {
                    "name": self.variant.training.lr_schedule.name,
                    "warmup_epochs": self.variant.training.lr_schedule.warmup_epochs,
                    "milestones_epochs": self.variant.training.lr_schedule.milestones_epochs,
                    "gamma": self.variant.training.lr_schedule.gamma,
                },
            }
            mismatches = {
                key: {"expected": expected, "actual": raw.get(key)}
                for key, expected in expected_metadata.items()
                if raw.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(f"unexpected W1 result metadata on {host}: {mismatches}")
            metrics = to_training_metrics(raw)
            expected_epochs = self.variant.training.max_epochs
            completed_epochs_raw = raw.get("completed_epochs")
            if expected_epochs is not None:
                if self.variant.training.run_mode == "train_to_accuracy":
                    if not isinstance(completed_epochs_raw, int) or not 0 < completed_epochs_raw <= expected_epochs:
                        raise RuntimeError(
                            f"invalid target-accuracy W1 run length on {host}: "
                            f"cap {expected_epochs} epochs, completed {completed_epochs_raw}"
                        )
                elif completed_epochs_raw != expected_epochs:
                    raise RuntimeError(
                        f"incomplete epoch-based W1 run on {host}: "
                        f"expected {expected_epochs} epochs, completed {completed_epochs_raw}"
                    )
            if expected_epochs is not None:
                samples_per_rank = int(raw.get("samples_per_rank_per_epoch") or 0)
                batches_per_epoch = int(raw.get("batches_per_epoch") or 0)
                steps_per_epoch = int(raw.get("steps_per_epoch") or 0)
                expected_batches = math.ceil(samples_per_rank / self.sweep.batch_size)
                expected_steps = math.ceil(expected_batches / self.sweep.gradient_accumulation_steps)
                if samples_per_rank <= 0 or batches_per_epoch != expected_batches or steps_per_epoch != expected_steps:
                    raise RuntimeError(
                        f"invalid W1 epoch accounting on {host}: samples_per_rank={samples_per_rank}, "
                        f"batches={batches_per_epoch}/{expected_batches}, "
                        f"steps={steps_per_epoch}/{expected_steps}"
                    )
            schedule = self.variant.training.lr_schedule
            expected_initial_lr = self.variant.training.learning_rate
            if schedule.warmup_epochs:
                steps_per_epoch = int(raw.get("steps_per_epoch") or 0)
                if steps_per_epoch <= 0:
                    raise RuntimeError(f"scheduled W1 artifact has no valid steps_per_epoch on {host}")
                expected_initial_lr /= schedule.warmup_epochs * steps_per_epoch
            completed_epochs = int(raw.get("completed_epochs") or 0)
            expected_final_lr = self.variant.training.learning_rate * schedule.gamma ** sum(
                completed_epochs >= milestone for milestone in schedule.milestones_epochs
            )
            for metric_name, expected_lr in (
                ("training.learning_rate_initial", expected_initial_lr),
                ("training.learning_rate_final", expected_final_lr),
            ):
                actual_lr = metrics.get(metric_name)
                if actual_lr is None or not math.isclose(actual_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12):
                    raise RuntimeError(
                        f"unexpected W1 learning-rate trajectory on {host}: "
                        f"{metric_name} expected {expected_lr}, got {actual_lr}"
                    )
            if self.variant.training.codecarbon.enabled:
                metrics.update(
                    parse_codecarbon_metrics(
                        raw.get("codecarbon") or {},
                        self.variant.training.gpus_per_node,
                    )
                )
            metrics.update(self.monitor_metrics)
            energy_kwh = metrics.get("training.energy_kwh")
            completed_steps = int(raw.get("completed_steps") or 0)
            if energy_kwh is not None and energy_kwh > 0 and completed_steps > 0:
                total_images = int(raw.get("processed_images") or 0)
                if total_images <= 0:
                    raise RuntimeError(f"W1 artifact has no positive processed_images count on {host}")
                metrics["training.images_per_kwh"] = total_images / energy_kwh
            parsed[host] = metrics
            # Ordered (step, loss) pairs for the loss-curve PNG. Kept off the
            # metric dict because it is an artifact input, not a gated metric.
            self.loss_series = [
                (int(p["step"]), float(p["loss"]))
                for p in (raw.get("loss_time_series") or [])
                if p.get("step") is not None and p.get("loss") is not None
            ]
        return parsed

    def stop_training_processes(self):
        pattern = "[c]vs_pytorch_vision_w1.py"
        command = (
            f"pkill -TERM -f {shlex.quote(pattern)} 2>/dev/null || true; "
            "sleep 2; "
            f"pkill -KILL -f {shlex.quote(pattern)} 2>/dev/null || true; "
            f"! pgrep -f {shlex.quote(pattern)} >/dev/null"
        )
        result = self.orch.exec(command, timeout=10, detailed=True, print_console=False)
        self._require_success(result, f"stop PyTorch Vision processes for {self.sweep_name}")

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

    def _host_date(self):
        handle = getattr(self.orch, "all", None)
        if handle is None or not hasattr(handle, "exec"):
            raise RuntimeError("container orchestrator does not expose a host execution handle")
        result = handle.exec("date", timeout=30, print_console=False)
        if not result:
            raise RuntimeError("host timestamp probe returned no results")
        return result
