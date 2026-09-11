"""
Run Deck result normalization for PyTorch xDiT diffusion suites.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from pathlib import Path


def _resolution(params):
    height = params.get("height")
    width = params.get("width")
    if height and width:
        return f"{height}×{width}", float(height) * float(width) / 1_000_000

    size = str(params.get("size") or "")
    parts = size.lower().replace("×", "*").split("*")
    if len(parts) == 2:
        try:
            height, width = (int(part.strip()) for part in parts)
            return f"{height}×{width}", height * width / 1_000_000
        except ValueError:
            pass
    return size or "—", None


def build_xdit_result_record(
    *,
    workload,
    label,
    inference_config,
    benchmark_params,
    gpu,
    nnodes,
    sample_times,
    average_time,
    sample_kind,
    passed,
    sample_count=None,
):
    """Normalize one parsed diffusion run for the generic series builder."""
    resolution, resolution_mp = _resolution(benchmark_params)
    average_time = float(average_time)
    samples = [float(value) for value in sample_times]
    model = str(inference_config.get("model_repo") or "—")

    return {
        "workload": workload,
        "label": str(label),
        "model": model,
        "model_name": Path(model.rstrip("/")).name or model,
        "gpu": str(gpu or "—"),
        "nnodes": int(nnodes),
        "resolution": resolution,
        "resolution_mp": resolution_mp,
        "frame_count": benchmark_params.get("frame_num"),
        "inference_steps": benchmark_params.get("num_inference_steps"),
        "sample_kind": sample_kind,
        "sample_times_s": samples,
        "sample_count": len(samples) if sample_count is None else int(sample_count),
        "average_time_s": average_time,
        "output_throughput_per_s": 1.0 / average_time if average_time > 0 else None,
        "status": "PASS" if passed else "FAIL",
    }


def xdit_run_card_display(records, provenance):
    """Build diffusion-specific model, GPU, and node metadata."""
    record = records[0] if isinstance(records, list) and records else {}
    rows = [
        ("Model", record.get("model", "—"), False),
        ("GPU", record.get("gpu", "—"), False),
        ("Framework", "PyTorch xDiT", False),
        ("Nodes", str(record.get("nnodes", "—")), False),
        ("Resolution", record.get("resolution", "—"), False),
    ]
    pytest_href = provenance.get("pytest_html_href") or provenance.get("pytest_html_path")
    if pytest_href:
        rows.append(("Pytest report", str(pytest_href), True))
    log_href = provenance.get("log_file_href") or provenance.get("log_file_path")
    if log_href:
        rows.append(("Run log", str(log_href), True))
    return rows
