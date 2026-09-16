"""Single-node FLUX xDiT lifecycle suite."""

from cvs.tests.inference.xdit._shared import (
    launch_container_stage,
    parse_thresholds_stage,
    print_results_stage,
    run_benchmark_stage,
    teardown_stage,
    verify_model_stage,
    verify_parallelism_stage,
    verify_prerequisites_stage,
)


def test_launch_container(orch, lifecycle, request):
    launch_container_stage(orch, lifecycle, request)


def test_verify_prerequisites(orch, lifecycle, request):
    verify_prerequisites_stage(orch, lifecycle, request)


def test_verify_model(orch, variant_config, xdit_spec, lifecycle, request):
    verify_model_stage(orch, variant_config, xdit_spec, lifecycle, request)


def test_verify_parallelism(variant_config, cluster_dict, xdit_spec, lifecycle, request):
    verify_parallelism_stage(variant_config, cluster_dict, xdit_spec, lifecycle, request)


def test_run_benchmark(orch, variant_config, hf_token, cluster_dict, xdit_spec, lifecycle, request):
    run_benchmark_stage(orch, variant_config, hf_token, cluster_dict, xdit_spec, lifecycle, request)


def test_parse_thresholds(variant_config, gpu_type, xdit_spec, lifecycle, request):
    parse_thresholds_stage(variant_config, gpu_type, xdit_spec, lifecycle, request)


def test_print_results(lifecycle):
    print_results_stage(lifecycle)


def test_teardown(orch, lifecycle, request):
    teardown_stage(orch, lifecycle, request)
