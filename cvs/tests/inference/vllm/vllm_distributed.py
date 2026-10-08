'''Distributed vLLM benchmark suite; a one-host cluster runs the vllm_single path.

Tests performed (in order):
1. test_launch_container          - Launch the vLLM container on every node
2. test_setup_sshd                - Always skipped; vLLM needs no in-container sshd
3. test_discover_topology         - Resolve and validate IB HCAs across nodes
4. test_model_fetch               - Verify every node's model cache has weights
5. test_openai_compatible_smoke   - Short-lived server answers the OpenAI-compatible API
6. test_vllm_inference[cell]      - Benchmark one sweep cell and verify its metrics
7. test_accuracy_eval[task]       - lm-eval accuracy tasks, if configured
8. test_print_results_table       - Console summary table
9. test_teardown                  - Tear the containers down

Cells are parametrized from the config's `runs`; the shared stage logic lives
in _common.py.
'''

from cvs.lib.inference.utils import inference_suite_lifecycle
from cvs.tests.inference.vllm import _common, _shared
from cvs.tests.inference.vllm._common import pytest_generate_tests  # noqa: F401


def test_launch_container(orch, vllm_targets, lifecycle, request):
    """Launch the vLLM container on every node and verify it is running."""
    return _common.launch_container(orch, vllm_targets, lifecycle, request)


def test_setup_sshd():
    """Skipped: vLLM uses host-network NCCL and gloo rather than in-container sshd."""
    return _common.setup_sshd()


def test_discover_topology(orch, variant_config, vllm_targets, lifecycle, request):
    """Discover IB HCAs on every node and validate the requested set; a no-op when the effective topology is one node."""
    return _common.discover_topology(orch, variant_config, vllm_targets, lifecycle, request)


def test_model_fetch(orch, variant_config, lifecycle, request):
    """Verify the mounted model cache holds weights on every node."""
    return _common.model_fetch(orch, variant_config, lifecycle, request)


def test_openai_compatible_smoke(orch, variant_config, hf_token, vllm_targets, lifecycle, request):
    """Start a short-lived server and check the OpenAI-compatible endpoints answer."""
    return _common.openai_compatible_smoke(orch, variant_config, hf_token, vllm_targets, lifecycle, request)


def test_vllm_inference(orch, variant_config, hf_token, vllm_targets, run, inf_res_dict, lifecycle, request, subtests):
    """Benchmark one sweep cell, then report its metrics and verify enforced thresholds as subtests."""
    return _common.vllm_inference(
        orch, variant_config, hf_token, vllm_targets, run, inf_res_dict, lifecycle, request, subtests
    )


def test_accuracy_eval(orch, variant_config, accuracy_task, lifecycle, request):
    """Run one configured lm-eval accuracy task against the live server."""
    return inference_suite_lifecycle.test_accuracy_eval(orch, variant_config, accuracy_task, lifecycle, request)


def test_print_results_table(inf_res_dict):
    """Log the per-cell results table."""
    return _shared.print_results_table(inf_res_dict)


def test_teardown(orch, lifecycle, request):
    """Tear the containers down and verify they are gone."""
    return _common.teardown(orch, lifecycle, request)
