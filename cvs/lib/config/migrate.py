"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

One-shot JSON (v1) -> v2 YAML config migration. Splits a vLLM mega-config (many
models in one file) into one typed config per (framework, model), converts the
stringified ``result_dict`` lookups into typed Threshold predicates, stamps
``schema_version: "2"`` and forbids ``<changeme>`` sentinels.
"""

from __future__ import annotations

import re
from typing import Dict, List

from cvs.lib.config.loader import ConfigError, validate_config

_SENTINEL = "<changeme>"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _csv_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [tok.strip() for tok in value.split(",") if tok.strip()]
    return []


def _thresholds_from_result_dict(result_dict: Dict) -> List[Dict]:
    """Best-effort: floor throughput at the observed min, ceil latencies at max.

    The per-cell v1 ``result_dict`` is lossy to map onto v2's config-global
    predicates; the migrated thresholds are a safe envelope and should be
    reviewed/tightened against a fresh baseline run.

    C5: the v1 metrics are *means* (``mean_ttft_ms`` / ``mean_tpot_ms``). They
    are emitted as honest scalar (``rate``-type) predicates on the same
    mean-named metric -- never relabeled as a P99 percentile, which would claim
    a tail guarantee the data does not support.

    Contract for the adapter (vertical, out of G2 scope): a ``rate`` threshold
    reads ``ResultView.scalar(metric)``, so the vLLM ``parse`` step must surface
    scalars literally named ``mean_ttft_ms`` / ``mean_tpot_ms`` or these
    migrated thresholds evaluate to ``actual=None -> passed=False``.
    """
    throughputs, ttfts, tpots = [], [], []
    for entry in (result_dict or {}).values():
        if not isinstance(entry, dict):
            continue
        if "total_throughput_per_sec" in entry:
            throughputs.append(float(entry["total_throughput_per_sec"]))
        if "mean_ttft_ms" in entry:
            ttfts.append(float(entry["mean_ttft_ms"]))
        if "mean_tpot_ms" in entry:
            tpots.append(float(entry["mean_tpot_ms"]))
    thresholds: List[Dict] = []
    if throughputs:
        thresholds.append(
            {"type": "rate", "metric": "total_throughput", "op": ">=", "value": min(throughputs), "unit": "tok/s"}
        )
    if ttfts:
        thresholds.append({"type": "rate", "metric": "mean_ttft_ms", "op": "<=", "value": max(ttfts), "unit": "ms"})
    if tpots:
        thresholds.append({"type": "rate", "metric": "mean_tpot_ms", "op": "<=", "value": max(tpots), "unit": "ms"})
    return thresholds


def _assert_no_sentinel(obj) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            _assert_no_sentinel(value)
    elif isinstance(obj, list):
        for value in obj:
            _assert_no_sentinel(value)
    elif isinstance(obj, str) and _SENTINEL in obj:
        raise ValueError(f"refusing to migrate config containing '{_SENTINEL}' sentinel")


def migrate_vllm_megaconfig(raw: Dict, target_gpu: str) -> Dict[str, Dict]:
    """Return ``{model_slug: v2_config_dict}`` for a vLLM v1 mega-config.

    The HF token is emitted as a deferred ``${env:HF_TOKEN}`` reference under
    ``container.env`` (no secret/redaction layer; security removed). It is
    resolved by the loader at parse time and rides inline as ``-e HF_TOKEN=...``
    on closed/internal clusters.
    """
    _assert_no_sentinel(raw)
    top = raw.get("config", {})
    bench_params = raw.get("benchmark_params", {})
    nnodes = _int(top.get("nnodes"), 1) or 1

    out: Dict[str, Dict] = {}
    for model_key, entry in bench_params.items():
        tp = _int(entry.get("tensor_parallelism"), 1) or 1
        seqs = []
        for combo in entry.get("sequence_combinations", []):
            seqs.append(
                {
                    "isl": _int(combo.get("isl")),
                    "osl": _int(combo.get("osl")),
                    "name": combo.get("name", f"isl{combo.get('isl')}_osl{combo.get('osl')}"),
                }
            )
        v2 = {
            "schema_version": "2",
            "framework": "vllm",
            "target_gpu": target_gpu,
            "model": entry.get("model", model_key),
            "seed": _int(entry.get("seed"), 0) or 0,
            "knobs": {"backend": entry.get("backend", "vllm")},
            "container": {"env": {"HF_TOKEN": "${env:HF_TOKEN}"}},
            "params": {
                "server_script": entry.get("server_script", f"{model_key}.sh"),
                "bench_serv_script": entry.get("bench_serv_script", "benchmark_serving.py"),
                "backend": entry.get("backend", "vllm"),
                "container_image": entry.get("container_image", top.get("container_image")),
                "dataset_name": entry.get("dataset_name", "random"),
                "num_prompts": _int(entry.get("num_prompts"), 3200) or 3200,
                "max_model_length": _int(entry.get("max_model_length"), 9216) or 9216,
                "request_rate": str(entry.get("request_rate", "inf")),
                "tokenizer_mode": entry.get("tokenizer_mode", "auto"),
                "percentile_metrics": _csv_list(entry.get("percentile_metrics")) or ["ttft", "tpot", "itl", "e2el"],
                "metric_percentiles": _int(entry.get("metric_percentiles"), 99) or 99,
                "port_no": _int(entry.get("port_no"), 8888) or 8888,
            },
            # TP is represented once, as the server role's gpus_per_node. It is
            # deliberately NOT also emitted as a sweep axis: the topology is
            # fixed at bind time, so a swept TP would silently diverge from the
            # GPUs actually allocated (and a single-value axis only adds a
            # redundant '-tensor_parallelism<n>' token to every cell ID).
            "topology": {"roles": {"server": {"count": nnodes, "gpus_per_node": tp, "selector": target_gpu}}},
            "sweep": {
                "concurrency": list(entry.get("concurrency_levels", [16, 32, 64])),
                "sequence_combinations": seqs,
            },
            "thresholds": _thresholds_from_result_dict(entry.get("result_dict", {})),
            "benchmarks": ["throughput", "ttft_mean"],
        }
        slug = _slug(model_key)
        if slug in out:
            raise ValueError(
                f"model key {model_key!r} slugifies to {slug!r}, which already maps to "
                f"another model; rename one of the source model keys so no workload is "
                f"silently dropped"
            )
        # Fail fast at conversion time on a structurally invalid output (e.g. an
        # empty sequence_combinations / concurrency axis) instead of surfacing it
        # only when the file is loaded for a run days later. Schema-only: deferred
        # ${env:...} values are not resolved here.
        try:
            validate_config(v2)
        except ConfigError as exc:
            raise ValueError(f"migrated config for model {model_key!r} is invalid: {exc}") from exc
        out[slug] = v2
    return out
