'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Fixtures and hooks for ``sglang_disagg_distributed`` (multi-node PSSH + Docker).

``--config_file`` supplies cluster/runtime settings: either a monolithic JSON with
top-level ``"config"`` and ``"benchmark_params"``, or (when ``--model_config_file``
is also passed) the main config only—either flat key/value or under ``"config"``.

When ``--model_config_file`` is set, it must be JSON with top-level
``"benchmark_params"`` (e.g. ``llama-70b-config.json``). When omitted, benchmark
parameters are read from ``--config_file`` as before.
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

from cvs.tests.inference.sglang._shared import SGLANG_DISAGG_TEST_ORDER, resolve_benchmark_variant_key

log = globals.log


def _model_config_path(pytestconfig):
    try:
        p = pytestconfig.getoption("model_config_file")
    except ValueError:
        return None
    if p is None:
        return None
    s = str(p).strip()
    return s or None


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
def model_config_file(pytestconfig):
    return _model_config_path(pytestconfig)


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file, encoding="utf-8") as fp:
        d = json.load(fp)
    return resolve_cluster_config_placeholders(d)


@pytest.fixture(scope="module")
def inference_config_root(inference_config_file):
    with open(inference_config_file, encoding="utf-8") as fp:
        return json.load(fp)


@pytest.fixture(scope="module")
def benchmark_config_root(model_config_file):
    if not model_config_file:
        return None
    with open(model_config_file, encoding="utf-8") as fp:
        return json.load(fp)


@pytest.fixture(scope="module")
def inference_dict(inference_config_root, cluster_dict):
    if isinstance(inference_config_root, dict) and "config" in inference_config_root:
        cfg = inference_config_root["config"]
    else:
        cfg = inference_config_root
    return resolve_test_config_placeholders(cfg, cluster_dict)


@pytest.fixture(scope="module")
def benchmark_params_dict(inference_config_root, benchmark_config_root, cluster_dict):
    if benchmark_config_root is not None:
        bp = benchmark_config_root["benchmark_params"]
    else:
        bp = inference_config_root["benchmark_params"]
    return resolve_test_config_placeholders(bp, cluster_dict)


@pytest.fixture(scope="module")
def benchmark_variant(
    inference_config_root,
    benchmark_config_root,
    inference_config_file,
    model_config_file,
):
    root = benchmark_config_root if benchmark_config_root is not None else inference_config_root
    label = model_config_file if benchmark_config_root is not None else inference_config_file
    return resolve_benchmark_variant_key(root, label)


@pytest.fixture(scope="module")
def hf_token(inference_dict):
    hf_token_file = inference_dict["hf_token_file"]
    try:
        with open(hf_token_file, encoding="utf-8") as fp:
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
def inf_res_dict():
    return {}

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