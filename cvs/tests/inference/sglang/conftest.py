'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Fixtures and hooks for ``sglang_disagg_distributed`` (multi-node PSSH + Docker).
'''

import json

import pytest

from cvs.lib import globals
from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import (
    get_model_from_rocm_smi_output,
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)

from ._shared import SGLANG_DISAGG_TEST_ORDER, resolve_benchmark_variant_key

log = globals.log


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    path = pytestconfig.getoption("cluster_file")
    if not path:
        pytest.fail("--cluster_file is required")
    return path


@pytest.fixture(scope="module")
def inference_config_file(pytestconfig):
    path = pytestconfig.getoption("config_file")
    if not path:
        pytest.fail("--config_file is required")
    return path


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as fp:
        d = json.load(fp)
    return resolve_cluster_config_placeholders(d)


@pytest.fixture(scope="module")
def inference_config_root(inference_config_file):
    with open(inference_config_file) as fp:
        return json.load(fp)


@pytest.fixture(scope="module")
def inference_dict(inference_config_root, cluster_dict):
    cfg = inference_config_root["config"]
    return resolve_test_config_placeholders(cfg, cluster_dict)


@pytest.fixture(scope="module")
def benchmark_params_dict(inference_config_root, cluster_dict):
    bp = inference_config_root["benchmark_params"]
    return resolve_test_config_placeholders(bp, cluster_dict)


@pytest.fixture(scope="module")
def benchmark_variant(inference_config_root, inference_config_file):
    return resolve_benchmark_variant_key(inference_config_root, inference_config_file)


@pytest.fixture(scope="module")
def hf_token(inference_dict):
    hf_token_file = inference_dict["hf_token_file"]
    try:
        with open(hf_token_file, "r") as fp:
            return fp.read().rstrip("\n")
    except FileNotFoundError:
        pytest.fail(f"hf_token file not found: {hf_token_file}")
    except OSError as e:
        pytest.fail(f"cannot read hf_token file {hf_token_file}: {e}")


@pytest.fixture(scope="module")
def p_phdl(cluster_dict, inference_dict):
    env_vars = cluster_dict.get("env_vars")
    return Pssh(
        log,
        inference_dict["prefill_node_list"],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=env_vars,
    )


@pytest.fixture(scope="module")
def d_phdl(cluster_dict, inference_dict):
    env_vars = cluster_dict.get("env_vars")
    return Pssh(
        log,
        inference_dict["decode_node_list"],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=env_vars,
    )


@pytest.fixture(scope="module")
def r_phdl(cluster_dict, inference_dict):
    env_vars = cluster_dict.get("env_vars")
    return Pssh(
        log,
        [inference_dict["proxy_router_node"]],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=env_vars,
    )


@pytest.fixture(scope="module")
def b_phdl(cluster_dict, inference_dict):
    env_vars = cluster_dict.get("env_vars")
    return Pssh(
        log,
        [inference_dict["benchmark_serv_node"]],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=env_vars,
    )


@pytest.fixture(scope="module")
def gpu_type(p_phdl, cluster_dict):
    head_node = p_phdl.host_list[0]
    smi_out_dict = p_phdl.exec("rocm-smi -a | head -30")
    smi_out = smi_out_dict[head_node]
    return get_model_from_rocm_smi_output(smi_out)


@pytest.fixture(scope="module")
def im_obj(
    p_phdl,
    d_phdl,
    r_phdl,
    b_phdl,
    gpu_type,
    inference_dict,
    benchmark_params_dict,
    benchmark_variant,
    hf_token,
):
    from cvs.lib import sglang_disagg_lib

    bp_dict = benchmark_params_dict[benchmark_variant]
    return sglang_disagg_lib.SglangDisaggPD(
        bp_dict["model"],
        inference_dict,
        bp_dict,
        hf_token,
        p_phdl,
        d_phdl,
        r_phdl,
        b_phdl,
        gpu_type,
    )


def pytest_collection_modifyitems(items):
    rank = SGLANG_DISAGG_TEST_ORDER
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))