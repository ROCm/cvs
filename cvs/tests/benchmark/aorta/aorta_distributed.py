"""Aorta distributed benchmark with one test per lifecycle stage.

Usage:
    cvs run aorta_distributed --cluster_file cluster.json \
        --config_file mi3xx_aorta_profile_overlap_2gpu_distributed.json

Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

from cvs.tests.benchmark.aorta import _common


def test_launch_container(aorta_job, lifecycle):
    return _common.launch_container(aorta_job, lifecycle)


def test_clone_aorta(aorta_job, lifecycle):
    return _common.clone_aorta(aorta_job, lifecycle)


def test_setup_rdma(aorta_job, lifecycle):
    return _common.setup_rdma(aorta_job, lifecycle)


def test_build_rccl(aorta_job, lifecycle):
    return _common.build_rccl(aorta_job, lifecycle)


def test_run_benchmark(aorta_job, lifecycle):
    return _common.run_benchmark(aorta_job, lifecycle)


def test_collect_traces(aorta_job, lifecycle):
    return _common.collect_traces(aorta_job, lifecycle)


def test_analyze(aorta_job, lifecycle):
    return _common.analyze(aorta_job, lifecycle)


def test_parse_results(aorta_job, lifecycle):
    return _common.parse_results(aorta_job, lifecycle)


def test_validate_thresholds(aorta_job, lifecycle):
    return _common.validate_thresholds(aorta_job, lifecycle)


def test_generate_report(aorta_job, lifecycle):
    return _common.generate_report(aorta_job, lifecycle)


def test_teardown(aorta_job, lifecycle):
    return _common.teardown(aorta_job, lifecycle)
