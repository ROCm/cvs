'''vLLM replica launch commands for the llm-d gateway suite.'''

import shlex

from cvs.lib.orchestrator.llm_d.llm_d_common import join_argv


def _append_many(argv, flag, values):
    for value in values or []:
        argv.extend((flag, value))


def worker_run_command(config, worker, include_hf_token=False):
    runtime = config.container.runtime["args"]
    server = config.server_params
    name = worker["name"]
    argv = ["docker", "run", "-d", "--name", name]
    if runtime.get("network"):
        argv.extend(("--network", runtime["network"]))
    if runtime.get("ipc"):
        argv.extend(("--ipc", runtime["ipc"]))
    if runtime.get("privileged"):
        argv.append("--privileged")
    if runtime.get("shm_size"):
        argv.extend(("--shm-size", runtime["shm_size"]))
    _append_many(argv, "--device", runtime.get("devices"))
    _append_many(argv, "--group-add", runtime.get("group_add"))
    _append_many(argv, "--cap-add", runtime.get("cap_add"))
    _append_many(argv, "--security-opt", runtime.get("security_opt"))
    _append_many(argv, "-v", runtime.get("volumes"))

    visible_devices = worker.get("hip_visible_devices", server.hip_visible_devices)
    argv.extend(("-e", f"HIP_VISIBLE_DEVICES={visible_devices}"))
    if include_hf_token:
        argv.extend(("-e", "HF_TOKEN"))
    for key, value in (config.container.get("env", {}) or {}).items():
        argv.extend(("-e", f"{key}={value}"))

    argv.extend(
        (
            config.container.image,
            "--model",
            worker.get("model", server.model),
            "--served-model-name",
            server.served_model_name,
            "--port",
            worker["port"],
            "--host",
            server.get("host", "0.0.0.0"),
            "--tensor-parallel-size",
            worker.get("tensor_parallel_size", server.tensor_parallel_size),
        )
    )
    for raw_flag in server.get("add_flags", []):
        argv.extend(shlex.split(str(raw_flag)))

    prefix = ""
    if include_hf_token:
        token_file = shlex.quote(config.paths.hf_token_file)
        prefix = f'export HF_TOKEN="$(cat {token_file})"; '
    return prefix + f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true; " + join_argv(argv)
