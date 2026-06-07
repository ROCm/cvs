"""Unit tests for cvs.lib.benchmarks.harness_invokers."""

from __future__ import annotations

import pytest

from cvs.lib.benchmarks.harness_invokers import (
    HARNESS_INVOKERS,
    OUTPUT_DIR_IN_CONTAINER,
    HarnessCtx,
    build_command,
)
from cvs.lib.benchmarks.registry import BenchmarkSpec, lookup
from cvs.lib.errors import ConfigError


def _hctx() -> HarnessCtx:
    return HarnessCtx(
        base_url="http://localhost:8000",
        model_id="llama-3.1-8b",
        model_path="/models/llama-3.1-8b",
        output_dir="/output",
    )


# ---- registry-level -------------------------------------------------------


def test_invoker_registry_has_both_harnesses():
    assert "lm-eval-harness" in HARNESS_INVOKERS
    assert "vllm-bench-serve" in HARNESS_INVOKERS


def test_output_dir_in_container_is_absolute():
    assert OUTPUT_DIR_IN_CONTAINER.startswith("/")


# ---- lm-eval --------------------------------------------------------------


def test_lm_eval_command_includes_required_flags():
    cmd = build_command(lookup("mmlu"), _hctx())
    assert "lm_eval" in cmd
    assert "--model local-completions" in cmd
    assert "base_url=http://localhost:8000/v1/completions" in cmd
    assert "model=llama-3.1-8b" in cmd
    # tokenizer must be the on-disk path so HF does not try to fetch
    # the served-model-name as a repo id.
    assert "tokenizer=/models/llama-3.1-8b" in cmd
    assert "tokenizer_backend=huggingface" in cmd
    assert "--tasks mmlu" in cmd
    assert "--num_fewshot 5" in cmd
    assert "--output_path /output/mmlu" in cmd
    assert "/output/mmlu.json" not in cmd
    assert "--log_samples" in cmd


def test_lm_eval_gsm8k_uses_task_arg_from_extra():
    cmd = build_command(lookup("gsm8k"), _hctx())
    assert "--tasks gsm8k" in cmd
    assert "--output_path /output/gsm8k" in cmd


def test_unknown_harness_raises():
    spec = BenchmarkSpec("fake", "no-such-harness")
    with pytest.raises(KeyError) as exc_info:
        build_command(spec, _hctx())
    assert "no-such-harness" in str(exc_info.value)


# ---- vllm-bench-serve -----------------------------------------------------


def test_serve_random_includes_required_flags():
    cmd = build_command(lookup("serve_synth_short"), _hctx())
    assert "vllm bench serve" in cmd
    assert "--backend vllm" in cmd
    assert "--base-url http://localhost:8000" in cmd
    assert "--model llama-3.1-8b" in cmd
    assert "--tokenizer /models/llama-3.1-8b" in cmd
    assert "--dataset-name random" in cmd
    assert "--num-prompts 200" in cmd
    assert "--random-input-len 128" in cmd
    assert "--random-output-len 256" in cmd
    assert "--max-concurrency 32" in cmd
    assert "--request-rate inf" in cmd
    assert "--metric-percentiles 50,90,95,99" in cmd
    # Goodput SLOs are split into separate tokens after --goodput.
    assert "--goodput ttft:2000 tpot:200" in cmd
    # Result lands at <output_dir>/<id>/result.json — runner globs it.
    assert "--save-result" in cmd
    assert "--result-dir /output/serve_synth_short" in cmd
    assert "--result-filename result.json" in cmd
    # The mkdir -p prefix is left as a shell operator (not shlex-quoted away).
    assert cmd.startswith("mkdir -p /output/serve_synth_short && vllm bench serve")


def test_serve_random_requires_random_lengths():
    spec = BenchmarkSpec(
        id="bad_random",
        harness="vllm-bench-serve",
        extra={"dataset_name": "random", "num_prompts": 10},
    )
    with pytest.raises(ConfigError) as exc_info:
        build_command(spec, _hctx())
    assert "random_input_len" in str(exc_info.value)


def test_serve_sharegpt_requires_dataset_path():
    spec = BenchmarkSpec(
        id="bad_sharegpt",
        harness="vllm-bench-serve",
        extra={"dataset_name": "sharegpt", "num_prompts": 10},
    )
    with pytest.raises(ConfigError) as exc_info:
        build_command(spec, _hctx())
    assert "dataset_path" in str(exc_info.value)


def test_serve_missing_required_extras_raise():
    spec = BenchmarkSpec(id="empty", harness="vllm-bench-serve", extra={})
    with pytest.raises(ConfigError):
        build_command(spec, _hctx())


def test_serve_optional_flags_only_when_present():
    spec = BenchmarkSpec(
        id="bare",
        harness="vllm-bench-serve",
        extra={
            "dataset_name": "random",
            "num_prompts": 10,
            "random_input_len": 64,
            "random_output_len": 64,
        },
    )
    cmd = build_command(spec, _hctx())
    assert "--max-concurrency" not in cmd
    assert "--request-rate" not in cmd
    assert "--metric-percentiles" not in cmd
    assert "--goodput" not in cmd


def test_serve_sharegpt_path_threaded_through():
    spec = BenchmarkSpec(
        id="sharegpt_local",
        harness="vllm-bench-serve",
        extra={
            "dataset_name": "sharegpt",
            "num_prompts": 50,
            "dataset_path": "/data/sharegpt.json",
        },
    )
    cmd = build_command(spec, _hctx())
    assert "--dataset-name sharegpt" in cmd
    assert "--dataset-path /data/sharegpt.json" in cmd
