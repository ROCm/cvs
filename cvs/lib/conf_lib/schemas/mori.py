'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Config schema for the Mori suites, install_mori and mori_benchmark_test, which share one config file.
'''

from typing import Any, Dict, Literal, Optional

from cvs.lib.conf_lib.config_blocks import ConfigStr, ContainerSpec, _Forbid


class MoriConfig(_Forbid):
    gpu_name: Optional[str] = None
    # Not read; kept so configs from earlier releases still load.
    no_of_nodes: Any = None
    orchestrator: Optional[Literal["baremetal", "container"]] = None
    container: Optional[ContainerSpec] = None
    env: Dict[str, ConfigStr]
    # Required, unlike the install keys: a model default would not get its {user-id} expanded.
    mori_dir: ConfigStr
    git_url: ConfigStr = "https://github.com/ROCm/mori"
    git_ref: ConfigStr = "main"
    install_system_packages: bool = True
    master_addr: ConfigStr = ""
    master_port: int
    log_dir: ConfigStr
    expected_results: Dict[str, Dict[str, Any]]
