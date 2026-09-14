'''Lifecycle controller for an llm-d gateway backed by vLLM replicas.'''

import base64
import json
import shlex
import time

from cvs.lib.utils.model_query_lib import OpenAIProbe

from cvs.lib.orchestrator.llm_d.llm_d_common import (
    envoy_run_command,
    epp_run_command,
    render_endpoints,
    render_envoy_config,
    render_epp_config,
)

from .llm_d_vllm_lib import worker_run_command


class LlmdVllmStack:
    def __init__(self, orch, config, topology, log, hf_token=""):
        self.orch = orch
        self.config = config
        self.topology = topology
        self.log = log
        self.has_hf_token = bool(hf_token)
        self.start_time = orch.exec('date +"%a %b %e %H:%M:%S"')
        self.end_time = None

    def _head(self, command, timeout=None):
        return self.orch.exec_on_head(command, timeout=timeout)

    def _write_root_file(self, path, content):
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        sudo = self.orch.sudo_prefix()
        command = f"echo {shlex.quote(encoded)} | base64 -d | {sudo}tee {shlex.quote(path)} >/dev/null"
        self._head(command)

    def stage_gateway_config(self):
        gateway = self.config.gateway
        sudo = self.orch.sudo_prefix()
        self._head(f"{sudo}mkdir -p {shlex.quote(gateway.config_dir)} {shlex.quote(gateway.envoy_config_dir)}")
        self._write_root_file(f"{gateway.config_dir}/config.yaml", render_epp_config(self.config))
        self._write_root_file(f"{gateway.config_dir}/endpoints.yaml", render_endpoints(self.config, self.topology))
        self._write_root_file(
            f"{gateway.envoy_config_dir}/envoy.yaml",
            render_envoy_config(self.config),
        )

    def _per_host_commands(self, builder):
        commands = []
        for host in self.orch.hosts:
            host_commands = [builder(worker) for worker in self.topology.workers_on(host)]
            commands.append(" && ".join(host_commands) if host_commands else "true")
        return commands

    def launch_workers(self):
        commands = self._per_host_commands(
            lambda worker: worker_run_command(self.config, worker, include_hf_token=self.has_hf_token)
        )
        self.orch.all.exec_cmd_list(commands, timeout=120, print_console=True)
        checks = self._per_host_commands(
            lambda worker: f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(worker['name'])}"
        )
        results = self.orch.all.exec_cmd_list(checks, timeout=30, print_console=True)
        for host in self.orch.hosts:
            expected = len(self.topology.workers_on(host))
            if not expected:
                continue
            running = [line for line in str(results.get(host, "")).splitlines() if line.strip() == "true"]
            if len(running) != expected:
                raise RuntimeError(f"vLLM containers are not running on {host}: {results.get(host)!r}")

    def wait_workers_ready(self):
        iterations = int(self.config.server_params.get("server_poll_iterations", 60))
        wait_s = float(self.config.server_params.get("server_poll_wait_s", 30))
        pending = {worker["name"]: worker for worker in self.topology.workers}
        for _ in range(iterations):
            for name, worker in list(pending.items()):
                url = self.topology.worker_url(worker)
                output = self._head(f"curl -sf --max-time 10 {shlex.quote(url + '/v1/models')}", timeout=20)
                text = next(iter(output.values()), "") if output else ""
                if text and '"data"' in text:
                    pending.pop(name)
            if not pending:
                return
            time.sleep(wait_s)
        self.dump_logs()
        raise RuntimeError(f"vLLM workers not ready: {sorted(pending)}")

    def _require_head_container(self, name):
        output = self._head(f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(name)}", timeout=30)
        text = next(iter(output.values()), "") if output else ""
        if text.strip() != "true":
            raise RuntimeError(f"container {name!r} is not running on gateway: {text!r}")

    def launch_epp(self):
        self._head(epp_run_command(self.config), timeout=120)
        self._require_head_container(self.config.gateway.get("epp_container_name", "epp"))

    def launch_envoy(self):
        self._head(envoy_run_command(self.config), timeout=120)
        self._require_head_container(self.config.gateway.get("envoy_container_name", "envoy"))

    def wait_gateway_ready(self):
        gateway = self.config.gateway
        iterations = int(gateway.get("poll_iterations", 30))
        wait_s = float(gateway.get("poll_wait_s", 2))
        ready_url = f"http://127.0.0.1:{gateway.admin_port}/ready"
        metrics_url = f"http://127.0.0.1:{gateway.epp_metrics_port}/metrics"
        for _ in range(iterations):
            ready = self._head(f"curl -sf --max-time 5 {shlex.quote(ready_url)}", timeout=10)
            metrics = self._head(f"curl -sf --max-time 5 {shlex.quote(metrics_url)}", timeout=10)
            ready_text = next(iter(ready.values()), "") if ready else ""
            metrics_text = next(iter(metrics.values()), "") if metrics else ""
            if "LIVE" in ready_text.upper() and metrics_text.strip():
                return
            time.sleep(wait_s)
        self.dump_logs()
        raise RuntimeError("llm-d gateway did not become ready")

    def probe_openai_endpoints(self):
        port = int(self.config.gateway.listen_port)
        model = self.config.server_params.served_model_name
        probe = OpenAIProbe.probe_script(port, model)
        encoded = base64.b64encode(probe.encode("utf-8")).decode("ascii")
        path = f"{self.config.paths.log_dir}/openai_probe.py"
        command = (
            f"mkdir -p {shlex.quote(self.config.paths.log_dir)} && "
            f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)} && "
            f"python3 {shlex.quote(path)}"
        )
        output = self._head(command, timeout=int(self.config.smoke.get("timeout_s", 120)))
        raw = next(iter(output.values()), "") if output else ""
        if not raw.strip():
            raise RuntimeError("llm-d OpenAI-compatible probe produced no output")
        try:
            parsed = json.loads(raw.strip().splitlines()[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"llm-d OpenAI-compatible probe returned invalid JSON: {raw!r}") from exc
        results = {name: (int(value[0]), value[1]) for name, value in parsed.items()}
        OpenAIProbe.log_results(results, self.log)
        ok, error = OpenAIProbe.check_results(results, port=port, logger=self.log)
        summary = OpenAIProbe.summarize_results(results, ok, error)
        if not ok:
            raise RuntimeError(error)
        return summary

    def dump_logs(self):
        gateway = self.config.gateway
        names = (gateway.get("epp_container_name", "epp"), gateway.get("envoy_container_name", "envoy"))
        self._head(" ; ".join(f"docker logs --tail 100 {shlex.quote(name)} || true" for name in names))
        commands = self._per_host_commands(
            lambda worker: f"docker logs --tail 100 {shlex.quote(worker['name'])} || true"
        )
        self.orch.all.exec_cmd_list(commands, timeout=30, print_console=True)

    def cleanup(self):
        gateway = self.config.gateway
        commands = []
        for host in self.orch.hosts:
            names = [worker["name"] for worker in self.topology.workers_on(host)]
            if host == self.topology.gateway_node:
                names.extend(
                    (
                        gateway.get("envoy_container_name", "envoy"),
                        gateway.get("epp_container_name", "epp"),
                    )
                )
            commands.append(
                "docker rm -f " + " ".join(shlex.quote(name) for name in names) + " >/dev/null 2>&1 || true"
                if names
                else "true"
            )
        self.orch.all.exec_cmd_list(commands, timeout=60, print_console=True)
        self.end_time = self.orch.exec('date +"%a %b %e %H:%M:%S"')
