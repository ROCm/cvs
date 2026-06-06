"""Unit tests for cvs.lib.benchmarks.harness_invokers."""

from __future__ import annotations

import pytest

from cvs.lib.benchmarks.harness_invokers import (
    HARNESS_INVOKERS,
    OUTPUT_DIR_IN_CONTAINER,
    HarnessCtx,
    build_command,
)
from cvs.lib.benchmarks.registry import lookup


def _hctx() -> HarnessCtx:
    return HarnessCtx(
        base_url="http://localhost:8000",
        model_id="llama-3.1-8b",
        output_dir="/output",
    )


def test_invoker_registry_has_lm_eval():
    assert "lm-eval-harness" in HARNESS_INVOKERS


def test_output_dir_in_container_is_absolute():
    assert OUTPUT_DIR_IN_CONTAINER.startswith("/")


def test_build_command_mmlu_includes_required_flags():
    cmd = build_command(lookup("mmlu"), _hctx())
    assert "lm_eval" in cmd
    assert "--model local-completions" in cmd
    assert "base_url=http://localhost:8000/v1/completions" in cmd
    assert "model=llama-3.1-8b" in cmd
    assert "--tasks mmlu" in cmd
    assert "--num_fewshot 5" in cmd
    # --output_path is a directory per lm-eval >=0.4 convention.
    assert "--output_path /output/mmlu" in cmd
    assert "/output/mmlu.json" not in cmd
    assert "--log_samples" in cmd


def test_build_command_gsm8k_uses_task_arg_from_extra():
    cmd = build_command(lookup("gsm8k"), _hctx())
    assert "--tasks gsm8k" in cmd
    assert "--num_fewshot 5" in cmd
    assert "--output_path /output/gsm8k" in cmd


def test_build_command_unknown_harness_raises():
    from cvs.lib.benchmarks.registry import BenchmarkSpec
    spec = BenchmarkSpec("fake", "no-such-harness", "mmlu", "acc")
    with pytest.raises(KeyError) as exc_info:
        build_command(spec, _hctx())
    assert "no-such-harness" in str(exc_info.value)
    assert "lm-eval-harness" in str(exc_info.value)
