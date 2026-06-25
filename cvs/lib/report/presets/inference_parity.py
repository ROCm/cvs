'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

IX / W1 inference framework parity preset (not generic).

Uses fixed framework ids (``atom``, ``vllm``, ``sglang``) and automation-plan
``compare.*`` ratio keys. For arbitrary frameworks, use
``cvs.lib.report.parity.inference.build_inference_parity_config`` instead.
'''

from __future__ import annotations

from cvs.lib.report.types import InferenceParityConfig, InferenceParityMetric, InferenceParitySource

_REFERENCE_ID = "atom"

# Section 12.6 compare.* keys (W1 automation plan).
W1_INFERENCE_PARITY_METRICS = (
    InferenceParityMetric(
        "client.output_throughput",
        "vllm",
        "compare.vllm.output_throughput_ratio",
        reference_framework_id=_REFERENCE_ID,
        title="Output throughput",
    ),
    InferenceParityMetric(
        "client.output_throughput",
        "sglang",
        "compare.sglang.output_throughput_ratio",
        reference_framework_id=_REFERENCE_ID,
        title="Output throughput",
    ),
    InferenceParityMetric(
        "client.mean_ttft_ms",
        "vllm",
        "compare.vllm.mean_ttft_ms_ratio",
        reference_framework_id=_REFERENCE_ID,
        title="Mean TTFT",
    ),
    InferenceParityMetric(
        "client.mean_ttft_ms",
        "sglang",
        "compare.sglang.mean_ttft_ms_ratio",
        reference_framework_id=_REFERENCE_ID,
        title="Mean TTFT",
    ),
    InferenceParityMetric(
        "client.mean_tpot_ms",
        "vllm",
        "compare.vllm.mean_tpot_ms_ratio",
        reference_framework_id=_REFERENCE_ID,
        title="Mean TPOT",
    ),
    InferenceParityMetric(
        "client.mean_tpot_ms",
        "sglang",
        "compare.sglang.mean_tpot_ms_ratio",
        reference_framework_id=_REFERENCE_ID,
        title="Mean TPOT",
    ),
)


def w1_triple_parity_config(
    *,
    atom_json: str,
    vllm_json: str = "",
    sglang_json: str = "",
) -> InferenceParityConfig:
    """W1 convenience wrapper: ATOM + atom-vLLM + atom-SGLang report JSON paths."""
    sources = [
        InferenceParitySource(_REFERENCE_ID, "ATOM", atom_json),
    ]
    if vllm_json:
        sources.append(InferenceParitySource("vllm", "atom-vLLM", vllm_json))
    if sglang_json:
        sources.append(InferenceParitySource("sglang", "atom-SGLang", sglang_json))
    return InferenceParityConfig(
        reference_framework_id=_REFERENCE_ID,
        sources=tuple(sources),
        metrics=W1_INFERENCE_PARITY_METRICS,
    )
