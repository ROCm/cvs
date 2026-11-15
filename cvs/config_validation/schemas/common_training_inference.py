'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Paths and container blocks shared by every training and inference config, and
``WorkloadConfig``, the base those configs build on.

Field names, order and defaults match ``cvs.lib.conf_lib.config_blocks``, so a
block keeps its values when its suite ports. ``RuntimeSpec.args`` is typed here, so a wrong type fails and ``_`` keys in it
are dropped. The cluster file's container block is not validated.
'''

from typing import Dict, List, Literal, Optional, Union

from pydantic import Field, model_serializer

from cvs.config_validation.schemas.base import ConfigStr, OpenBlock, StrictBlock, SuiteConfig


class Paths(StrictBlock):
    shared_fs: ConfigStr
    models_dir: ConfigStr
    log_dir: ConfigStr
    hf_token_file: ConfigStr
    temp_dir: ConfigStr = ""  # JAX MaxText scratch; other suites omit it


class RuntimeArgs(OpenBlock):
    """The ``runtime.args`` keys the Docker runtime reads.

    An unset key is left out of the dump, so the runtime falls back to its own
    default exactly as if the config had not named it.
    """

    volumes: Optional[List[ConfigStr]] = None
    devices: Optional[List[str]] = None
    env: Optional[Dict[str, ConfigStr]] = None
    cap_add: Optional[List[str]] = None
    security_opt: Optional[List[str]] = None
    group_add: Optional[List[Union[str, int]]] = None
    ulimit: Optional[List[str]] = None
    network: Optional[str] = None
    ipc: Optional[str] = None
    user: Optional[Union[str, int]] = None
    shm_size: Optional[Union[str, int]] = None
    privileged: Optional[bool] = None
    registry: Optional[Dict[str, str]] = None

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler):
        return {key: value for key, value in handler(self).items() if value is not None}


class RuntimeSpec(OpenBlock):
    name: str
    args: RuntimeArgs = Field(default_factory=RuntimeArgs)


class ContainerSpec(StrictBlock):
    lifetime: Literal["no_launch", "per_run", "persistent"] = "per_run"
    name: ConfigStr
    image: ConfigStr
    runtime: RuntimeSpec
    env: Dict[str, ConfigStr] = Field(default_factory=dict)


class WorkloadConfig(SuiteConfig):
    paths: Paths
    container: Optional[ContainerSpec] = None
