'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Fixtures for Infera vLLM 1P1D disaggregated smoke tests.
'''

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from cvs.lib import globals
from cvs.lib.inference.infera.vllm_disagg_lib import InferaVllmDisaggOrchestrator, InferaVllmDisaggPD
from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import resolve_cluster_config_placeholders, resolve_test_config_placeholders, update_test_result
from cvs.tests.inference.infera._shared import INFERA_DISAGG_TEST_ORDER, resolve_benchmark_variant_key

log = globals.log


@dataclass(frozen=True)
class InferaVariantConfig:
    config_path: str
    variant_key: str
    inference: dict[str, Any]
    benchmark_params: dict[str, Any]

    @property
    def hf_token_file(self) -> str:
        return self.inference["hf_token_file"]

    @property
    def model(self) -> str:
        return self.benchmark_params["model"]


def load_infera_variant(config_path: str, cluster_dict: Mapping[str, Any]) -> InferaVariantConfig:
    with open(config_path, encoding="utf-8") as fp:
        root = json.load(fp)
    variant_key = resolve_benchmark_variant_key(root, config_path)
    cfg = root["config"] if "config" in root else root
    inference = resolve_test_config_placeholders(cfg, cluster_dict)
    bp_all = resolve_test_config_placeholders(root["benchmark_params"], cluster_dict)
    return InferaVariantConfig(
        config_path=config_path,
        variant_key=variant_key,
        inference=inference,
        benchmark_params=dict(bp_all[variant_key]),
    )


class _Lifecycle:
    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report: dict[str, list[tuple[str, float, str]]] = {}
        self.smoke_results: list | None = None

    def record(self, nodeid: str, label: str, value: float, unit: str = "s") -> None:
        self.report.setdefault(nodeid, []).append((label, value, unit))

    def complete_stage(self, request, label: str, t0: float) -> None:
        self.record(request.node.nodeid, label, time.monotonic() - t0)
        if globals.error_list:
            self.failed = True
        update_test_result()


@pytest.fixture(scope="module")
def cluster_dict(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    if not cluster_file:
        pytest.fail("--cluster_file is required")
    with open(cluster_file, encoding="utf-8") as fp:
        return resolve_cluster_config_placeholders(json.load(fp))


@pytest.fixture(scope="module")
def variant_config(pytestconfig, cluster_dict) -> InferaVariantConfig:
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    return load_infera_variant(config_file, cluster_dict)


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


@pytest.fixture(scope="module")
def hf_token(variant_config):
    path = variant_config.hf_token_file
    if not os.path.isfile(path):
        pytest.skip(f"hf_token file missing: {path}")
    with open(path, encoding="utf-8") as fp:
        return fp.read().strip()


@pytest.fixture(scope="module")
def p_phdl(cluster_dict, variant_config):
    return Pssh(
        log,
        variant_config.inference["prefill_node_list"],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=cluster_dict.get("env_vars"),
    )


@pytest.fixture(scope="module")
def d_phdl(cluster_dict, variant_config):
    return Pssh(
        log,
        variant_config.inference["decode_node_list"],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=cluster_dict.get("env_vars"),
    )


@pytest.fixture(scope="module")
def r_phdl(cluster_dict, variant_config):
    return Pssh(
        log,
        [variant_config.inference["proxy_router_node"]],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=cluster_dict.get("env_vars"),
    )


@pytest.fixture(scope="module")
def b_phdl(cluster_dict, variant_config):
    return Pssh(
        log,
        [variant_config.inference["benchmark_serv_node"]],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=cluster_dict.get("env_vars"),
    )


@pytest.fixture(scope="module")
def im_obj(p_phdl, d_phdl, r_phdl, b_phdl, variant_config, hf_token, cluster_dict):
    return InferaVllmDisaggPD(
        variant_config.model,
        variant_config.inference,
        variant_config.benchmark_params,
        hf_token,
        p_phdl,
        d_phdl,
        r_phdl,
        b_phdl,
        user_name=cluster_dict["username"],
        priv_key_file=cluster_dict["priv_key_file"],
    )


@pytest.fixture(scope="module")
def orch(p_phdl, d_phdl, r_phdl, b_phdl, variant_config, lifecycle):
    o = InferaVllmDisaggOrchestrator(
        variant_config.inference, p_phdl, d_phdl, r_phdl, b_phdl
    )
    yield o
    if not lifecycle.torn_down:
        o.teardown_containers()
        o.cleanup_log_dir()


def pytest_collection_modifyitems(items):
    items.sort(
        key=lambda it: INFERA_DISAGG_TEST_ORDER.get(
            it.originalname or it.name.split("[")[0], 99
        )
    )
