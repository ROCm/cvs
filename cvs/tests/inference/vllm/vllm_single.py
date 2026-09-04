'''First-host single-node vLLM benchmark suite.'''
# ruff: noqa: F401

from cvs.tests.inference.vllm._common import (
    pytest_generate_tests,
    test_accuracy_eval,
    test_discover_topology,
    test_gpu_metric,
    test_launch_container,
    test_metric,
    test_model_fetch,
    test_openai_compatible_smoke,
    test_prom_metric,
    test_setup_sshd,
    test_teardown,
    test_vllm_inference,
)  # noqa: F401
from cvs.tests.inference.vllm._shared import test_print_results_table  # noqa: F401
