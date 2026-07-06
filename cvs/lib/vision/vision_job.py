'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Standalone pytorch_vision single-node job driven by a ContainerOrchestrator.

Mirrors `cvs.lib.inference.vllm_single.VllmJob`: it talks only to `orch.exec`
(which already routes into the running container) and a typed `VariantConfig`.
The container image is pulled/launched by the orchestrator (per the container
block in the config), so this class never touches docker directly.

The benchmark is self-contained -- the timing script is BUILT IN PYTHON here and
written into the container at /tmp/vision_bench.py, exactly the way VllmJob builds
its serve/bench commands rather than cloning an external `.sh`. It instantiates a
torchvision model with random weights (weights=None -> fully offline, measures
compute not accuracy), warms up, times forward passes, and writes a JSON artifact
that `parse_results` fetches and hands to the pure `to_vision_metrics`.
'''

from __future__ import annotations

import json
import shlex

from cvs.lib import globals
from cvs.lib.vision.utils.vision_parsing import to_vision_metrics

log = globals.log


# The in-container benchmark. A plain (non-f) string: all inputs arrive as argv so
# there is nothing to interpolate here and no brace-escaping to get wrong. Kept
# dependency-light -- torch + torchvision only, both present in a pytorch image.
_BENCH_SCRIPT = r'''
import argparse, json, statistics, sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--channels-last", type=int, default=1)
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() is False -- no GPU visible in container", file=sys.stderr)
        sys.exit(2)

    try:
        import torchvision
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: torchvision import failed ({e}); use an image that ships torchvision", file=sys.stderr)
        sys.exit(3)

    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtypes[args.precision]
    device = "cuda"

    model = torchvision.models.get_model(args.model, weights=None)
    model = model.to(device=device, dtype=dtype).eval()
    mem_fmt = torch.channels_last if args.channels_last else torch.contiguous_format
    model = model.to(memory_format=mem_fmt)
    if args.compile:
        model = torch.compile(model)

    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device, dtype=dtype)
    x = x.to(memory_format=mem_fmt)

    latencies_ms = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        torch.cuda.synchronize()
        start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters):
            start_evt.record()
            model(x)
            end_evt.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_evt.elapsed_time(end_evt))

    latencies_ms.sort()

    def pct(p):
        if not latencies_ms:
            return None
        idx = min(len(latencies_ms) - 1, int(round((p / 100.0) * (len(latencies_ms) - 1))))
        return latencies_ms[idx]

    mean_ms = statistics.fmean(latencies_ms) if latencies_ms else None
    throughput = (args.batch_size * 1000.0 / mean_ms) if mean_ms else None

    result = {
        "model": args.model,
        "precision": args.precision,
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "device": torch.cuda.get_device_name(0),
        "iters": args.iters,
        "images": args.batch_size * args.iters,
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": pct(50),
        "latency_ms_p90": pct(90),
        "latency_ms_p99": pct(99),
        "throughput_img_s": throughput,
    }
    with open(args.output, "w") as fp:
        json.dump(result, fp)
    print("VISION_BENCH_OK " + json.dumps(result))


if __name__ == "__main__":
    main()
'''


class VisionJob:
    """Single-node torchvision benchmark job driven by an injected ContainerOrchestrator.

    All container/SSH plumbing belongs to `orch`. This class writes the env
    script + benchmark script into the container, runs a torch/GPU sanity probe,
    executes the benchmark for one cell, and parses the JSON artifact.

    The `orch` instance is expected to already have `setup_containers()` (and, for
    multinode, `setup_sshd()`) called against it by the test fixture; lifecycle is
    explicitly NOT owned here.
    """

    ENV_SCRIPT = "/tmp/vision_env_script.sh"
    BENCH_SCRIPT_PATH = "/tmp/vision_bench.py"

    def __init__(
        self,
        orch,
        variant,
        arch,
        precision,
        input_size,
        batch_size,
        log_subdir="pytorch_vision",
    ):
        self.orch = orch
        self.variant = variant
        self.arch = str(arch)
        self.precision = str(precision)
        self.input_size = str(input_size)
        self.batch_size = str(batch_size)
        self.log_subdir = log_subdir

        p = variant.params
        self.warmup_iters = p.warmup_iters
        self.bench_iters = p.bench_iters
        self.channels_last = "1" if str(p.channels_last) not in ("0", "false", "False", "") else "0"
        self.use_torch_compile = "1" if str(p.use_torch_compile) not in ("0", "false", "False", "") else "0"
        self.bench_timeout_s = int(p.bench_timeout_s)
        self.env = dict(variant.env)

        self.log_dir = variant.paths.log_dir
        # Per-cell output directory keyed by the cell (arch/prec/res/bs) so a
        # multi-cell sweep never overwrites an earlier cell's artifact -- and so
        # parse_results can never read a stale `results.json` from a prior cell.
        self.out_dir = (
            f"{self.log_dir}/{self.log_subdir}/"
            f"{self.arch}_prec{self.precision}_res{self.input_size}_bs{self.batch_size}"
        )
        self.result_json = f"{self.out_dir}/results.json"
        self.bench_log = f"{self.out_dir}/bench.log"

    # ---------- setup ----------

    def stage_scripts(self):
        """Write the env script + benchmark script into the container, mkdir out-dir.

        Each value is shlex.quoted so an env value or the (multiline) script body
        containing a space/$/quote cannot break the outer bash layer. Mirrors
        VllmJob.build_server_cmd.
        """
        env_lines = [f"export {k}={shlex.quote(str(v))}" for k, v in self.env.items()]
        env_script = ("\n".join(env_lines) + "\n") if env_lines else "\n"
        self.orch.exec("bash -c " + shlex.quote(f"printf '%s' {shlex.quote(env_script)} > {self.ENV_SCRIPT}"))
        self.orch.exec(
            "bash -c " + shlex.quote(f"printf '%s' {shlex.quote(_BENCH_SCRIPT)} > {self.BENCH_SCRIPT_PATH}")
        )
        self.orch.exec(f"mkdir -p {shlex.quote(self.out_dir)}")

    def verify_env(self):
        """Probe torch + GPU + torchvision inside the container; raise on failure.

        Proves the pulled image is usable before any benchmarking. Returns a
        short version/GPU-count string for the report row.
        """
        probe = (
            "python -c "
            + shlex.quote(
                "import torch, json; "
                "d={'torch': torch.__version__, 'cuda': torch.cuda.is_available(), "
                "'gpus': torch.cuda.device_count()}; "
                "import importlib.util as u; d['torchvision'] = u.find_spec('torchvision') is not None; "
                "print('VISION_ENV ' + json.dumps(d))"
            )
        )
        out = self.orch.exec("bash -c " + shlex.quote(probe), detailed=True)
        summaries = []
        for host, res in (out or {}).items():
            text = res.get("output", "") if isinstance(res, dict) else str(res)
            exit_code = res.get("exit_code", 1) if isinstance(res, dict) else 1
            marker = None
            for line in text.splitlines():
                if line.startswith("VISION_ENV "):
                    marker = line[len("VISION_ENV "):]
                    break
            if exit_code != 0 or marker is None:
                raise RuntimeError(f"torch env probe failed on {host}: {text[-500:]}")
            info = json.loads(marker)
            if not info.get("cuda") or int(info.get("gpus", 0)) < 1:
                raise RuntimeError(f"no GPU visible in container on {host}: {info}")
            if not info.get("torchvision"):
                raise RuntimeError(
                    f"torchvision not importable in image on {host}: {info}; "
                    "use an image that ships torchvision"
                )
            summaries.append(f"{host}: torch {info['torch']}, {info['gpus']} GPU(s)")
        return "; ".join(summaries)

    # ---------- benchmark ----------

    def _bench_argv(self):
        return [
            "python",
            self.BENCH_SCRIPT_PATH,
            "--model", self.arch,
            "--precision", self.precision,
            "--batch-size", self.batch_size,
            "--input-size", self.input_size,
            "--warmup", self.warmup_iters,
            "--iters", self.bench_iters,
            "--channels-last", self.channels_last,
            "--compile", self.use_torch_compile,
            "--output", self.result_json,
        ]

    def run_benchmark(self):
        """Run the benchmark once for this cell (foreground) and fail on error.

        Sources the env script, then runs the staged benchmark script. A nonzero
        exit or a Python traceback in the log is a hard failure for the cell (the
        test wraps this in try/except ... raise) rather than a silently-green
        empty row.
        """
        bench_cmd = " ".join(shlex.quote(str(a)) for a in self._bench_argv())
        inner = f"source {self.ENV_SCRIPT} && {bench_cmd} > {shlex.quote(self.bench_log)} 2>&1"
        out = self.orch.exec("bash -c " + shlex.quote(inner), timeout=self.bench_timeout_s, detailed=True)
        for host, res in (out or {}).items():
            exit_code = res.get("exit_code", 1) if isinstance(res, dict) else 1
            if exit_code != 0:
                tail = self.orch.exec(f"tail -50 {shlex.quote(self.bench_log)}")
                text = next(iter((tail or {}).values()), "") if tail else ""
                raise RuntimeError(f"vision benchmark failed on {host} (exit {exit_code}): {text[-800:]}")

    def parse_results(self):
        """Return {host: {vision.METRIC: value}} parsed from the results.json artifact.

        Raises if the artifact is missing/empty/unparseable -- the test wraps the
        job in try/except ... raise, so this hard-fails the cell rather than
        recording an empty (silently-green) row. The fetch lives here (artifact
        layout is job-specific); the transform lives in vision_parsing so future
        vision variants can reuse it.
        """
        out = self.orch.exec(f"cat {shlex.quote(self.result_json)}")
        results = {}
        for host, text in (out or {}).items():
            text = (text or "").strip()
            if not text:
                raise RuntimeError(f"empty/missing results artifact on {host}: {self.result_json}")
            try:
                raw = json.loads(text)
            except (json.JSONDecodeError, ValueError) as e:
                raise RuntimeError(f"unparseable results artifact on {host}: {self.result_json}: {e}") from e
            results[host] = to_vision_metrics(raw)
        return results
