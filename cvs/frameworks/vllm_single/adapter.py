"""VllmAdapter — single role, single host (in v1 smoke).

launch: pull image, mount models, start vllm serve, wait /health
await: vllm runs forever; we let benchmark/parse drive the lifetime
parse: in v1 smoke we just record TTFB via a single completion request

Step 4 will add benchmark harness; for now the smoke proves end-to-end.
"""

from __future__ import annotations

import json
import shlex
import time

from cvs.lib.base_adapter import BaseWorkloadAdapter
from cvs.lib.errors import WorkloadError
from cvs.lib.substitution import substitute


class VllmAdapter(BaseWorkloadAdapter):
    framework = "vllm_single"
    required_roles = ("server",)
    launch_timeout_s: float = 1800.0  # 30 min for 80B model load

    def launch(self, ctx) -> None:
        wl = ctx.workload
        role_spec = wl["roles"]["server"]
        # Resolve {model.path} etc inside command/env using ctx.scratch
        sub_ctx = ctx.scratch["sub_ctx"]
        command = substitute(role_spec["command"], sub_ctx)
        env = {k: substitute(v, sub_ctx) for k, v in role_spec.get("env", {}).items()}
        volumes = {substitute(k, sub_ctx): substitute(v, sub_ctx)
                   for k, v in role_spec.get("volumes", {}).items()}
        image_tag = wl["image"]["tag"]
        port = role_spec["port"]

        extra_args = list(role_spec.get("extra_args", []))
        self._launch_role(
            ctx,
            "server",
            image=image_tag,
            env=env,
            command=command,
            volumes=volumes,
            devices=role_spec.get("devices", []),
            shm_size=role_spec.get("shm_size"),
            ports={str(port): str(port)},
            network="host",
            ipc="host",
            seccomp_unconfined=bool(role_spec.get("seccomp_unconfined", False)),
            extra_args=extra_args,
        )
        self._wait_http_pool(
            "server",
            path=role_spec.get("health_path", "/health"),
            port=port,
            timeout_s=self.launch_timeout_s,
        )

    def parse(self, ctx) -> None:
        """Smoke: issue one completion request, record TTFB + tokens/sec."""
        role_spec = ctx.workload["roles"]["server"]
        port = role_spec["port"]
        host = ctx.bindings["server"][0]
        executor = ctx.executor.executor_for(host) if hasattr(ctx.executor, "executor_for") else ctx.executor

        prompt = "Once upon a time"
        body = json.dumps({
            "model": ctx.workload["model"]["id"],
            "prompt": prompt,
            "max_tokens": int(ctx.workload.get("params", {}).get("output_len", 32)),
            "temperature": 0.0,
        })
        cmd = (
            f"t0=$(date +%s.%N); "
            f"curl -s -X POST http://localhost:{port}/v1/completions "
            f"-H 'Content-Type: application/json' "
            f"-d {shlex.quote(body)}; "
            f"t1=$(date +%s.%N); "
            f"echo \"ELAPSED=$(echo $t1 - $t0 | bc)\""
        )
        out = executor.exec(cmd, timeout=300)
        elapsed_s = None
        for line in out.splitlines():
            if line.startswith("ELAPSED="):
                try:
                    elapsed_s = float(line.split("=", 1)[1])
                except ValueError:
                    pass
        # Try to count completion tokens (rough — vllm response JSON)
        n_tokens = 0
        try:
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("{") and '"choices"' in line:
                    payload = json.loads(line)
                    if "usage" in payload and "completion_tokens" in payload["usage"]:
                        n_tokens = int(payload["usage"]["completion_tokens"])
                        break
        except Exception:
            pass

        if elapsed_s is None or elapsed_s <= 0:
            raise WorkloadError("vllm parse: could not measure elapsed time from completion request")
        tok_per_s = (n_tokens / elapsed_s) if n_tokens > 0 else 0.0
        ctx.result.scalars["smoke_request_latency_ms"] = elapsed_s * 1000.0
        ctx.result.scalars["smoke_completion_tokens"] = float(n_tokens)
        ctx.result.scalars["smoke_throughput_tok_s"] = tok_per_s
        ctx.logs["smoke_request_raw.txt"] = out
