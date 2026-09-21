'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Peak matrix TFLOP/s lookup for Model FLOPs Utilization (MFU).

MFU% = achieved TFLOP/s/GPU / peak TFLOP/s/GPU * 100. The peak depends on the GPU
and the compute precision, so it must not be assumed from a fixed constant. The
authoritative source is the config (per GPU/precision); this module only provides
a small reference table as a convenience fallback and returns ``None`` when the
peak is unknown -- MFU is then recorded but not computed (never a wrong number).

Reference values are published dense matrix peaks (TFLOP/s per GPU); override per
cluster/config when a datasheet or measured roofline differs.
'''

from __future__ import annotations

# gpu-arch substring -> precision -> peak TFLOP/s per GPU (dense).
_DEFAULT_PEAK_TFLOPS = {
    "MI300X": {"BF16": 1307.4, "FP16": 1307.4, "FP8": 2614.9},
    "MI325X": {"BF16": 1307.4, "FP16": 1307.4, "FP8": 2614.9},
}


def _normalize_precision(precision):
    if not precision:
        return ""
    p = str(precision).strip().upper()
    aliases = {"BFLOAT16": "BF16", "BF16": "BF16", "FLOAT16": "FP16", "FP16": "FP16", "FP8": "FP8", "E4M3": "FP8"}
    return aliases.get(p, p)


def peak_tflops(gpu_arch, precision, overrides=None):
    """Peak TFLOP/s/GPU for ``gpu_arch`` at ``precision``, or ``None`` if unknown.

    ``overrides`` (from config) takes precedence and may be either
    ``{precision: peak}`` or ``{gpu_substr: {precision: peak}}``. Matching on the
    built-in table is by case-insensitive substring (e.g. "AMD Instinct MI325X"
    matches "MI325X"), so a full ``device_kind`` string works.
    """
    prec = _normalize_precision(precision)
    if not prec:
        return None

    if isinstance(overrides, dict) and overrides:
        # Flat {precision: peak} override.
        flat = {_normalize_precision(k): v for k, v in overrides.items() if not isinstance(v, dict)}
        if prec in flat:
            return float(flat[prec])
        # Nested {gpu_substr: {precision: peak}} override.
        for substr, table in overrides.items():
            if isinstance(table, dict) and gpu_arch and substr.upper() in str(gpu_arch).upper():
                by_prec = {_normalize_precision(k): v for k, v in table.items()}
                if prec in by_prec:
                    return float(by_prec[prec])

    if not gpu_arch:
        return None
    arch = str(gpu_arch).upper()
    for substr, by_prec in _DEFAULT_PEAK_TFLOPS.items():
        if substr in arch and prec in by_prec:
            return float(by_prec[prec])
    return None


def compute_mfu(achieved_tflops_per_sec, peak_tflops_per_gpu):
    """MFU as a percentage, or ``None`` when either input is missing/non-positive."""
    if not achieved_tflops_per_sec or not peak_tflops_per_gpu:
        return None
    if peak_tflops_per_gpu <= 0:
        return None
    return float(achieved_tflops_per_sec) / float(peak_tflops_per_gpu) * 100.0
