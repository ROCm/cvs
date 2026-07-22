'''Container lifecycle for SGLang inference (single-node and disaggregated).'''

from __future__ import annotations

import re
import time
from typing import Any

from cvs.lib import docker_lib, globals
from cvs.lib.utils_lib import fail_test

log = globals.log


class SglangSingleOrchestrator:
    """Container lifecycle for unified single-node SGLang on ``benchmark_serv_node``."""

    def __init__(self, inference: dict[str, Any], b_phdl):
        self.inference = inference
        self.handles = {"b_phdl": b_phdl}
        self.container_name = inference["container_name"]
        self._launch_timeout = 60 * 20

    def launch_handle_keys(self) -> list[str]:
        return ["b_phdl"]

    def all_handles(self):
        return [self.handles["b_phdl"]]

    def cleanup_log_dir(self) -> None:
        log.info("Cleaning up log directory")
        self.handles["b_phdl"].exec(f"sudo rm -rf {self.inference['log_dir']}")

    def setup_containers(self) -> bool:
        inf = self.inference
        cc = inf["container_config"]
        shm = inf.get("shm_size", "48G")

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
        log.info("Verify if the container has been launched properly")
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


class SglangDisaggOrchestrator:
    """Container lifecycle for disagg SGLang. Not ContainerOrchestrator — role-aware."""

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

        if proxy == bench:
            if proxy not in prefill and proxy not in decode:
                hdl_list.append("r_phdl")
        else:
            if proxy not in prefill and proxy not in decode:
                hdl_list.append("r_phdl")
            if bench not in prefill and bench not in decode:
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
        shm = inf.get("shm_size", "48G")

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
        log.info("Verify if the containers have been launched properly")
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