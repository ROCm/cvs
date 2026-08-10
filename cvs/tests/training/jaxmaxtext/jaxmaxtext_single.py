'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

JAX MaxText SINGLE-NODE training suite.

Run with a single-node config (training.distributed = false), e.g.:
  cvs run jaxmaxtext_single --cluster_file <cluster>.json \
      --config_file .../mi300x_jaxmaxtext_llama-3.3-70b_single.json --html <out>.html

This is the single-node variant: no RDMA / NIC setup stages. The lifecycle,
per-sweep training, metric gating, loss curve, and reporting are the shared
implementations in _common.py. pytest_generate_tests (sweep parametrization) and
all fixtures/hooks live in conftest.py. The "single" mode is reflected in the
test module name (this file), the metric-results HTML title, and loss-curve
titles/artifacts.
'''

import importlib.util as _ilu
import pathlib as _pl

_spec = _ilu.spec_from_file_location("_jaxmaxtext_common", _pl.Path(__file__).with_name("_common.py"))
_c = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_c)

# Bind the shared lifecycle stages as this module's tests (pytest collects these).
# Single-node: intentionally NO test_setup_rdma (distributed-only stage).
test_launch_container = _c.test_launch_container
test_setup_tokenizer = _c.test_setup_tokenizer
test_training_run = _c.test_training_run
test_metric = _c.test_metric
test_loss_curve = _c.test_loss_curve
test_print_results_table = _c.test_print_results_table
test_teardown = _c.test_teardown
