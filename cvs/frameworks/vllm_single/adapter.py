"""VllmAdapter — single role, single host (in v1 smoke + accuracy).

launch: pull image, mount models, start vllm serve, wait /health
await: vllm runs forever; parse phase drives the lifetime
parse: smoke (single completion). If workload.benchmarks is non-empty, also
       runs lm-eval against the live server and projects scores to scalars.
"""

from __future__ import annotations

import json
import shlex

from cvs.lib.base_adapter import BaseWorkloadAdapter
from cvs.lib.benchmarks.harness_invokers import OUTPUT_DIR_IN_CONTAINER
from cvs.lib.benchmarks.runner import run_benchmarks
from cvs.lib.errors import WorkloadError
from cvs.lib.substitution import substitute


class VllmAdapter(BaseWorkloadAdapter):
    framework = "vllm_single"
    required_roles = ("server",)
    launch_timeout_s: float = 1800.0  # 30 min for 80B model load

    def launch(self, ctx) -> None:
        wl = ctx.workload
        role_spec = wl["roles"]["server"]
        sub_ctx = ctx.scratch["sub_ctx"]
        command = substitute(role_spec["command"], sub_ctx)
        env = {k: substitute(v, sub_ctx) for k, v in role_spec.get("env", {}).items()}
        volumes = {substitute(k, sub_ctx): substitute(v, sub_ctx)
                   for k, v in role_spec.get("volumes", {}).items()}
        # Bind-mount the run's host artifacts dir into the container so the
        # benchmark harness (run via `docker exec`) can drop result JSONs
        # that the parse phase reads back on the host. The in-container path
        # is fixed so harness_invokers / runner can agree without ctx.
        volumes[str(ctx.artifacts_dir)] = OUTPUT_DIR_IN_CONTAINER

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
        """Smoke + (optionally) accuracy benchmarks.

        Smoke is always done — one completion request through the server,
        records latency / token count. When ``workload.benchmarks`` is
        non-empty, also runs lm-eval-harness against the live server and
        projects each benchmark's score into ``ctx.result.scalars``.
        """

        self._run_smoke(ctx)

        benchmarks = list(ctx.workload.get("benchmarks") or [])
        if not benchmarks:
            return

        role_spec = ctx.workload["roles"]["server"]
        port = role_spec["port"]
        server_handle = self._server_handle(ctx)

        # The harness exec'd into the server container shares network=host,
        # so localhost reaches the server. The output dir is the well-known
        # in-container mount established at launch time.
        scalars = run_benchmarks(
            benchmarks=benchmarks,
            server_handle=server_handle,
            base_url=f"http://localhost:{port}",
            model_id=ctx.workload["model"]["id"],
            model_path=ctx.scratch["sub_ctx"]["model.path"],
            output_dir_in_container=OUTPUT_DIR_IN_CONTAINER,
            output_dir_on_host=ctx.artifacts_dir,
        )
        ctx.result.scalars.update(scalars)

    def _run_smoke(self, ctx) -> None:
        role_spec = ctx.workload["roles"]["server"]
        port = role_spec["port"]
        host = ctx.bindings["server"][0]
        executor = (
            ctx.executor.executor_for(host)
            if hasattr(ctx.executor, "executor_for") else ctx.executor
        )

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
            f"echo; t1=$(date +%s.%N); "
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

    def _server_handle(self, ctx):
        """Return the ContainerHandle for the server role on this host.

        Match the full name to avoid collisions if a future role/hostname
        happens to contain ``_server_`` as a substring.
        """
        host = ctx.bindings["server"][0]
        expected = f"{self.framework}_server_{host}_{ctx.run_id}"
        for handle in ctx.containers:
            if handle.name == expected:
                return handle
        raise WorkloadError(
            f"vllm parse: no server container handle named {expected!r} "
            f"(have: {[h.name for h in ctx.containers]})"
        )
