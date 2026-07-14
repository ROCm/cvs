'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Minimal Infera + vLLM 1P1D (prefill/decode) controller for smoke-testing
OpenAI-compatible /v1/models and /v1/completions via infera.server.
'''

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any, Optional

from cvs.lib import docker_lib, globals
from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import fail_test

log = globals.log


def _as_node_list(value):
    if isinstance(value, str):
        return [value]
    return list(value)


def textwrap_for_yml(msg_string: str) -> str:
    return "\n".join([m.lstrip() for m in msg_string.split("\n")])


class InferaVllmDisaggOrchestrator:
    """Container lifecycle for Infera disagg vLLM (role-aware PSSH + docker_lib)."""

    def __init__(self, inference: dict[str, Any], p_phdl, d_phdl, r_phdl, b_phdl):
        self.inference = inference
        self.handles = {
            "p_phdl": p_phdl,
            "d_phdl": d_phdl,
            "r_phdl": r_phdl,
            "b_phdl": b_phdl,
        }
        self.container_name = inference["container_name"]
        self._launch_timeout = 60 * 20

    def launch_handle_keys(self) -> list[str]:
        hdl_list = ["p_phdl", "d_phdl"]
        proxy = self.inference["proxy_router_node"]
        bench = self.inference["benchmark_serv_node"]
        prefill = self.inference["prefill_node_list"]
        decode = self.inference["decode_node_list"]
        if proxy not in prefill and proxy not in decode:
            hdl_list.append("r_phdl")
        if bench not in prefill and bench not in decode and bench != proxy:
            hdl_list.append("b_phdl")
        return hdl_list

    def all_handles(self):
        return list(self.handles.values())

    def cleanup_log_dir(self) -> None:
        log.info("Cleaning up log directory")
        self.handles["r_phdl"].exec(f"sudo rm -rf {self.inference['log_dir']}")

    def setup_containers(self) -> bool:
        inf = self.inference
        cc = inf["container_config"]
        shm = inf.get("shm_size", "128G")
        for key in self.launch_handle_keys():
            docker_lib.launch_docker_container(
                self.handles[key],
                self.container_name,
                inf["container_image"],
                cc["device_list"],
                cc["volume_dict"],
                cc["env_dict"],
                shm_size=shm,
                timeout=self._launch_timeout,
            )
        time.sleep(30)
        return self.verify_containers_running()

    def verify_containers_running(self) -> bool:
        ok = True
        for key in self.launch_handle_keys():
            out_dict = self.handles[key].exec("docker ps")
            for node, out in out_dict.items():
                if not re.search(re.escape(self.container_name), out or "", re.I):
                    fail_test(f"Failed to launch container on node {node}")
                    ok = False
        return ok and not globals.error_list

    def teardown_containers(self) -> bool:
        for a_phdl in self.all_handles():
            docker_lib.kill_docker_container(a_phdl, self.container_name)
            docker_lib.delete_all_containers_and_volumes(a_phdl)
        return True


class InferaVllmDisaggPD:
    """1P1D Infera deployment: etcd + infera.server + prefill/decode workers."""

    def __init__(
        self,
        model_name: str,
        inference_config_dict: dict[str, Any],
        benchmark_params_dict: dict[str, Any],
        hf_token: str,
        p_phdl=None,
        d_phdl=None,
        r_phdl=None,
        b_phdl=None,
        gpu_type: str = "mi300",
        user_name: Optional[str] = None,
        priv_key_file: Optional[str] = None,
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.gpu_type = gpu_type
        self.inf_dict = inference_config_dict
        self.bp_dict = benchmark_params_dict

        self.prefill_node_list = _as_node_list(self.inf_dict["prefill_node_list"])
        self.decode_node_list = _as_node_list(self.inf_dict["decode_node_list"])
        self.proxy_node = _as_node_list(self.inf_dict["proxy_router_node"])
        self.benchmark_serv_node = _as_node_list(self.inf_dict["benchmark_serv_node"])

        self.p_phdl = p_phdl or Pssh(
            log, self.prefill_node_list, user=user_name, pkey=priv_key_file
        )
        self.d_phdl = d_phdl or Pssh(
            log, self.decode_node_list, user=user_name, pkey=priv_key_file
        )
        self.r_phdl = r_phdl or Pssh(log, self.proxy_node, user=user_name, pkey=priv_key_file)
        self.b_phdl = b_phdl or Pssh(
            log, self.benchmark_serv_node, user=user_name, pkey=priv_key_file
        )

        self.home_dir = os.path.expanduser("~")
        self.inf_dict.setdefault("container_name", "infera_container")
        self.inf_dict.setdefault("log_dir", f"{self.home_dir}/LOGS/infera")
        self.inf_dict.setdefault("etcd_port", "2379")
        self.inf_dict.setdefault("frontend_port", "8000")
        self.inf_dict.setdefault("prefill_serv_port", "30001")
        self.inf_dict.setdefault("decode_serv_port", "30002")
        self.inf_dict.setdefault("proxy_router_serv_port", "8000")
        self.inf_dict.setdefault(
            "etcd_host",
            self.prefill_node_list[0],
        )

        self.container_name = self.inf_dict["container_name"]
        self.log_dir = self.inf_dict["log_dir"]
        self.tp = str(self.bp_dict.get("tensor_parallelism", "8"))
        self.served_model_name = self.bp_dict.get("served_model_name") or self.model_name

    def _common_env_lines(self) -> str:
        return f"""
export NCCL_DEBUG={self.inf_dict.get('nccl_debug', 'ERROR')}
export NCCL_IB_HCA={self.inf_dict['nccl_ib_hca']}
export NCCL_IB_GID_INDEX={self.inf_dict.get('nccl_ib_gid_index', '3')}
export NCCL_SOCKET_IFNAME={self.inf_dict['nccl_socket_ifname']}
export GLOO_SOCKET_IFNAME={self.inf_dict['gloo_socket_ifname']}
export GLOO_TCP_IFNAME={self.inf_dict.get('gloo_tcp_ifname', self.inf_dict['gloo_socket_ifname'])}
export VLLM_ROCM_USE_AITER=1
export QUARK_MXFP4_IMPL=triton
export PYTORCH_ROCM_ARCH=gfx950
export HSA_ENABLE_SDMA=1
export HF_TOKEN={self.hf_token}
export MODEL={self.model_name}
export TP={self.tp}
"""

    def _write_env_script(self, hdl, path: str) -> None:
        cmd = f'''docker exec {self.container_name} /bin/bash -c "echo '
{self._common_env_lines()}
' > {path}"'''
        hdl.exec(textwrap_for_yml(cmd))
        hdl.exec(
            f'docker exec {self.container_name} /bin/bash -c "chmod 755 {path}"'
        )

    def _mooncake_json(self, prefill_ip: str, decode_ip: str) -> str:
        etcd = f"{self.inf_dict['etcd_host']}:{self.inf_dict['etcd_port']}"
        hca_prefix = self.inf_dict.get("hca_id_prefix", "rdma")
        device = f"{hca_prefix}0"
        payload = {
            "prefill_url": f"{prefill_ip}:13003",
            "decode_url": f"{decode_ip}:13003",
            "metadata_server": etcd,
            "metadata_backend": "etcd",
            "protocol": "rdma",
            "device_name": device,
        }
        return json.dumps(payload)

    def start_etcd(self) -> None:
        """Start etcd on the prefill node (host, not container)."""
        etcd_host = self.inf_dict["etcd_host"]
        etcd_port = self.inf_dict["etcd_port"]
        log.info("Starting etcd on %s:%s", etcd_host, etcd_port)
        cmd = (
            f"pkill -f 'etcd --listen-client-urls' || true; "
            f"nohup etcd --listen-client-urls http://0.0.0.0:{etcd_port} "
            f"--advertise-client-urls http://{etcd_host}:{etcd_port} "
            f"> {self.log_dir}/etcd.log 2>&1 &"
        )
        self.p_phdl.exec(f"mkdir -p {self.log_dir}")
        self.p_phdl.exec(cmd)
        time.sleep(5)
        out = self.p_phdl.exec(f"curl -s http://{etcd_host}:{etcd_port}/health")
        health = out.get(etcd_host, "")
        if "true" not in str(health).lower():
            fail_test(f"etcd health check failed on {etcd_host}: {health!r}")

    def write_mooncake_config(self) -> None:
        prefill_ip = self.prefill_node_list[0]
        decode_ip = self.decode_node_list[0]
        mooncake = self._mooncake_json(prefill_ip, decode_ip)
        b64 = base64.b64encode(mooncake.encode("utf-8")).decode("ascii")
        for hdl in (self.p_phdl, self.d_phdl, self.r_phdl, self.b_phdl):
            cmd = f'''docker exec {self.container_name} /bin/bash -c "
mkdir -p /tmp;
echo '{b64}' | base64 -d > /tmp/mooncake.json"'''
            hdl.exec(textwrap_for_yml(cmd))
        self.inf_dict["mooncake_config_path"] = "/tmp/mooncake.json"

    def setup_worker_env(self) -> None:
        self._write_env_script(self.p_phdl, "/tmp/prefill_env_script.sh")
        self._write_env_script(self.d_phdl, "/tmp/decode_env_script.sh")
        self._write_env_script(self.r_phdl, "/tmp/frontend_env_script.sh")

    def launch_frontend(self) -> None:
        etcd_ep = f"{self.inf_dict['etcd_host']}:{self.inf_dict['etcd_port']}"
        port = self.inf_dict["frontend_port"]
        script = f"""#!/bin/bash
set -e
source /tmp/frontend_env_script.sh
export MOONCAKE_CONFIG_PATH={self.inf_dict.get('mooncake_config_path', '/tmp/mooncake.json')}
python3 -m infera.server \\
  --host 0.0.0.0 \\
  --port {port} \\
  --router-policy kv-aware \\
  --router-tokenizer-path {self.model_name} \\
  --etcd-endpoint {etcd_ep}
"""
        self._stage_and_run(
            self.r_phdl,
            script,
            "/tmp/frontend_launch.sh",
            f"{self.log_dir}/frontend_node/frontend.log",
            env_script="/tmp/frontend_env_script.sh",
        )

    def launch_prefill_servers(self) -> None:
        node_ip = self.prefill_node_list[0]
        port = self.inf_dict["prefill_serv_port"]
        etcd_ep = f"{self.inf_dict['etcd_host']}:{self.inf_dict['etcd_port']}"
        kv_cfg = json.dumps(
            {"kv_connector": "MooncakeConnector", "kv_role": "kv_producer"}
        )
        devices = ",".join(str(i) for i in range(int(self.tp)))
        script = f"""#!/bin/bash
set -e
source /tmp/prefill_env_script.sh
export HIP_VISIBLE_DEVICES={devices}
export VLLM_HOST_IP={node_ip}
export MOONCAKE_CONFIG_PATH={self.inf_dict.get('mooncake_config_path', '/tmp/mooncake.json')}
python3 -m infera.engine.vllm \\
  --model {self.model_name} \\
  --port {port} \\
  --host 0.0.0.0 \\
  --advertise-host {node_ip} \\
  --etcd-endpoint {etcd_ep} \\
  --tp-size {self.tp} \\
  --trust-remote-code \\
  --mem-fraction-static {self.bp_dict.get('memory_fraction', '0.90')} \\
  --no-enable-prefix-caching \\
  --block-size 1 \\
  --kv-transfer-config '{kv_cfg}'
"""
        self._stage_and_run(
            self.p_phdl,
            script,
            "/tmp/prefill_launch.sh",
            f"{self.log_dir}/prefill_node0/prefill.log",
            env_script="/tmp/prefill_env_script.sh",
        )

    def launch_decode_servers(self) -> None:
        node_ip = self.decode_node_list[0]
        port = self.inf_dict["decode_serv_port"]
        etcd_ep = f"{self.inf_dict['etcd_host']}:{self.inf_dict['etcd_port']}"
        kv_cfg = json.dumps(
            {"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer"}
        )
        devices = ",".join(str(i) for i in range(int(self.tp)))
        script = f"""#!/bin/bash
set -e
source /tmp/decode_env_script.sh
export HIP_VISIBLE_DEVICES={devices}
export VLLM_HOST_IP={node_ip}
export MOONCAKE_CONFIG_PATH={self.inf_dict.get('mooncake_config_path', '/tmp/mooncake.json')}
python3 -m infera.engine.vllm \\
  --model {self.model_name} \\
  --port {port} \\
  --host 0.0.0.0 \\
  --advertise-host {node_ip} \\
  --etcd-endpoint {etcd_ep} \\
  --tp-size {self.tp} \\
  --trust-remote-code \\
  --mem-fraction-static {self.bp_dict.get('memory_fraction', '0.90')} \\
  --no-enable-prefix-caching \\
  --block-size 1 \\
  --kv-transfer-config '{kv_cfg}'
"""
        self._stage_and_run(
            self.d_phdl,
            script,
            "/tmp/decode_launch.sh",
            f"{self.log_dir}/decode_node0/decode.log",
            env_script="/tmp/decode_env_script.sh",
        )

    def _stage_and_run(
        self,
        hdl,
        script_body: str,
        script_path: str,
        log_path: str,
        env_script: Optional[str] = None,
    ) -> None:
        b64 = base64.b64encode(script_body.encode("utf-8")).decode("ascii")
        log_dir = os.path.dirname(log_path)
        cmd = f'''docker exec {self.container_name} /bin/bash -c "
mkdir -p {log_dir};
echo '{b64}' | base64 -d > {script_path};
chmod 755 {script_path};
nohup bash {script_path} > {log_path} 2>&1 &"'''
        hdl.exec(textwrap_for_yml(cmd))
        time.sleep(3)

    def poll_and_check_server_ready(self, wait_s: int = 180) -> None:
        log.info("Waiting %s s for Infera workers to start", wait_s)
        time.sleep(wait_s)
        port = int(self.inf_dict["proxy_router_serv_port"])
        probe = self._minimal_probe_script(port, self.served_model_name)
        b64 = base64.b64encode(probe.encode("utf-8")).decode("ascii")
        cmd = f'''docker exec {self.container_name} /bin/bash -c "
echo '{b64}' | base64 -d > /tmp/infera_ready_probe.py;
python3 /tmp/infera_ready_probe.py"'''
        out = self.b_phdl.exec(textwrap_for_yml(cmd), timeout=300)
        host = self.benchmark_serv_node[0]
        raw = out.get(host) or next(iter(out.values()), "")
        if "READY" not in str(raw):
            fail_test(f"Infera frontend not ready on port {port}: {raw!r}")

    @staticmethod
    def _minimal_probe_script(port: int, model: str) -> str:
        return f"""import json, sys, time, urllib.request, urllib.error
PORT = {int(port)}
MODEL = {json.dumps(model)}
BASE = "http://0.0.0.0:%d" % PORT
deadline = time.time() + 900
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as resp:
            body = json.loads(resp.read().decode())
            if resp.status == 200 and body.get("data"):
                print("READY")
                sys.exit(0)
    except Exception as exc:
        last = repr(exc)
    time.sleep(15)
print("NOT_READY", last)
sys.exit(1)
"""

    def verify_openai_compatible_endpoints(self) -> list[str]:
        """Smoke-test GET /v1/models and POST /v1/completions on infera.server."""
        port = int(self.inf_dict["proxy_router_serv_port"])
        model = self.served_model_name
        probe = self._smoke_probe_script(port, model)
        b64 = base64.b64encode(probe.encode("utf-8")).decode("ascii")
        cmd = f'''docker exec {self.container_name} /bin/bash -c "
mkdir -p {self.log_dir}/benchmark_node;
echo '{b64}' | base64 -d > /tmp/infera_smoke_probe.py;
python3 /tmp/infera_smoke_probe.py"'''
        out = self.b_phdl.exec(textwrap_for_yml(cmd), timeout=300)
        host = self.benchmark_serv_node[0]
        raw = out.get(host) or next(iter(out.values()), "")
        lines = [ln.strip() for ln in str(raw).strip().splitlines() if ln.strip()]
        if not lines:
            fail_test(f"Infera smoke probe produced no output: {out!r}")
            return []
        try:
            parsed = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            fail_test(f"Infera smoke probe invalid JSON: {exc!r} raw={raw!r}")
            return []

        summary = []
        for step, val in parsed.items():
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                fail_test(f"Bad probe shape at {step!r}: {val!r}")
                continue
            code, _body = int(val[0]), val[1]
            ok = code == 200
            summary.append(f"{step} -> {'Pass' if ok else 'Fail'} ({code})")
            if not ok:
                fail_test(f"{step} failed with HTTP {code}")

        for line in summary:
            log.info("%s", line)
        return summary

    @staticmethod
    def _smoke_probe_script(port: int, model: str) -> str:
        prompt = "The capital of France is"
        return f"""import json, urllib.request, urllib.error
PORT = {int(port)}
MODEL = {json.dumps(model)}
BASE = "http://0.0.0.0:%d" % PORT
TIMEOUT = 120

def req(method, path, body=None):
    url = BASE + path
    data = None if body is None else json.dumps(body).encode("utf-8")
    hdrs = {{}}
    if body is not None:
        hdrs["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(r, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = int(getattr(resp, "status", 200))
            try:
                return code, json.loads(raw) if raw else {{}}
            except json.JSONDecodeError:
                return code, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return int(e.code), json.loads(raw) if raw else {{}}
        except json.JSONDecodeError:
            return int(e.code), raw

out = {{}}
out["model_endpoint"] = req("GET", "/v1/models")
out["completion_endpoint"] = req("POST", "/v1/completions", {{
    "model": MODEL,
    "prompt": {json.dumps(prompt)},
    "max_tokens": 16,
    "temperature": 0.0,
}})
print(json.dumps(out, default=str))
"""
