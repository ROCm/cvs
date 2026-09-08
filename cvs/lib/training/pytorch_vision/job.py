"""Container-orchestrated PyTorch Vision training job."""

from __future__ import annotations

import base64
import json
import re
import shlex
import time
import uuid
from pathlib import Path

from cvs.lib.training.pytorch_vision.utils.metrics import to_training_metrics


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

    def __init__(self, orch, variant, combo_key):
        self.orch = orch
        self.variant = variant
        self.combo_key = combo_key
        self.combo = variant.sweep.combinations[combo_key]
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.output_dir = f"{variant.paths.log_dir}/pytorch_vision/{combo_key}/{run_id}"
        self.result_path = f"{self.output_dir}/results.json"
        self.log_path = f"{self.output_dir}/training.log"
        self.training_start_time = None

    def stage_benchmark(self):
        source = Path(__file__).with_name("benchmark.py").read_bytes()
        encoded = base64.b64encode(source).decode("ascii")
        command = (
            f"mkdir -p {shlex.quote(self.output_dir)} && "
            f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(self.BENCHMARK_PATH)}"
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
            expected = self.variant.params.nproc_per_node
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
        params = self.variant.params
        combo = self.combo
        args = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={params.nproc_per_node}",
            self.BENCHMARK_PATH,
            "--model",
            combo.model,
            "--backend",
            combo.backend,
            "--precision",
            combo.precision,
            "--batch-size",
            str(combo.batch_size),
            "--image-size",
            str(combo.image_size),
            "--num-classes",
            str(params.num_classes),
            "--warmup-steps",
            str(params.warmup_steps),
            "--measure-steps",
            str(params.measure_steps),
            "--learning-rate",
            str(params.learning_rate),
            "--momentum",
            str(params.momentum),
            "--weight-decay",
            str(params.weight_decay),
            "--output",
            self.result_path,
        ]
        if params.channels_last:
            args.append("--channels-last")

        exports = {
            "OMP_NUM_THREADS": str(params.omp_num_threads),
            "PYTHONUNBUFFERED": "1",
            **self.variant.env,
        }
        export_lines = []
        for name, value in exports.items():
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
            export_lines.append(f"export {name}={shlex.quote(str(value))}")

        launch = " ".join(shlex.quote(str(arg)) for arg in args)
        timeout = params.timeout_s
        return (
            "set -o pipefail; "
            + "; ".join(export_lines)
            + f"; rm -f {shlex.quote(self.result_path)}; "
            + f"timeout --signal=TERM --kill-after=30s {timeout}s {launch} "
            + f"2>&1 | tee {shlex.quote(self.log_path)}"
        )

    def run_benchmark(self):
        if self.variant.params.verify_dmesg:
            self.training_start_time = self._host_date()
        result = self.orch.exec(
            self.build_command(),
            timeout=self.variant.params.timeout_s + 60,
            detailed=True,
        )
        self._require_success(result, f"run PyTorch Vision cell {self.combo_key}")

    def scan_dmesg_for_errors(self):
        if not self.variant.params.verify_dmesg:
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

    def parse_results(self):
        result = self.orch.exec(
            f"cat {shlex.quote(self.result_path)}",
            timeout=30,
            detailed=True,
            print_console=False,
        )
        self._require_success(result, f"read result artifact for {self.combo_key}")

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
                "model": self.combo.model,
                "backend": self.combo.backend,
                "precision": self.combo.precision,
                "image_size": self.combo.image_size,
                "batch_size_per_gpu": self.combo.batch_size,
                "world_size": self.variant.params.nproc_per_node,
                "synthetic_data": True,
            }
            mismatches = {
                key: {"expected": expected, "actual": raw.get(key)}
                for key, expected in expected_metadata.items()
                if raw.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(f"unexpected W1 result metadata on {host}: {mismatches}")
            parsed[host] = to_training_metrics(raw)
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
        self._require_success(result, f"stop PyTorch Vision processes for {self.combo_key}")

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
