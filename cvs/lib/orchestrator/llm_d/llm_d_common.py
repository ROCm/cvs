'''Shared llm-d gateway templates and container commands.

YAML under ``cvs/lib/orchestrator/llm_d/scripts`` is engine-agnostic so vLLM,
SGLang, and later backends can mount or copy the same files onto the gateway.
'''

import json
import shlex
from importlib import resources


SCRIPTS_PACKAGE = "cvs.lib.orchestrator.llm_d.scripts"


def scripts_dir():
    return resources.files(SCRIPTS_PACKAGE)


def load_script(name):
    return scripts_dir().joinpath(name).read_text(encoding="utf-8")


def _fill(text, values):
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def gateway_template_values(config):
    gateway = config.gateway
    return {
        "config_dir": gateway.config_dir,
        "admin_port": gateway.admin_port,
        "listen_port": gateway.listen_port,
        "epp_grpc_port": gateway.epp_grpc_port,
        "listener_name": config.server_params.get("backend", "llm-d"),
    }


def render_epp_config(config):
    return _fill(load_script("epp_config.yaml"), gateway_template_values(config))


def render_envoy_config(config):
    return _fill(load_script("envoy.yaml"), gateway_template_values(config))


def render_endpoints(config, topology):
    entry_template = load_script("endpoint_entry.yaml").rstrip("\n")
    model = json.dumps(config.server_params.model)
    entries = []
    for worker in topology.workers:
        entries.append(
            _fill(
                entry_template,
                {
                    "name": worker["name"],
                    "address": worker["address"],
                    "port": worker["port"],
                    "model": model,
                },
            )
        )
    return _fill(load_script("endpoints.yaml"), {"endpoint_entries": "\n".join(entries)})


def join_argv(argv):
    return shlex.join([str(value) for value in argv])


def epp_run_command(config):
    gateway = config.gateway
    image = f"{gateway.epp_image}:{gateway.epp_version}"
    name = gateway.get("epp_container_name", "epp")
    argv = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--network",
        "host",
        "-v",
        f"{gateway.config_dir}:{gateway.config_dir}:ro",
        image,
        f"--config-file={gateway.config_dir}/config.yaml",
        "--pool-name=file-discovery",
        "--pool-namespace=default",
        f"--grpc-port={gateway.epp_grpc_port}",
        f"--grpc-health-port={gateway.epp_grpc_health_port}",
        f"--metrics-port={gateway.epp_metrics_port}",
        "--secure-serving=false",
        "--v=2",
    ]
    return f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true; " + join_argv(argv)


def envoy_run_command(config):
    gateway = config.gateway
    name = gateway.get("envoy_container_name", "envoy")
    envoy_file = f"{gateway.envoy_config_dir}/envoy.yaml"
    argv = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--network",
        "host",
        "-v",
        f"{envoy_file}:{envoy_file}:ro",
        gateway.envoy_image,
        "--service-node",
        "envoy-proxy",
        "--log-level",
        gateway.get("log_level", "warn"),
        "--concurrency",
        gateway.get("concurrency", 8),
        "--drain-strategy",
        "immediate",
        "--drain-time-s",
        "60",
        "-c",
        envoy_file,
    ]
    return f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true; " + join_argv(argv)
