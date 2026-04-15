from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from cvs.schema.rccl import RCCL_WRONG_NONZERO_ERROR_TYPE, RcclTests, RcclTestsMultinodeRaw

from .config import RcclConfig, _normalize_threshold_map


def _normalize_collective_name(collective: str) -> str:
    normalized = collective.strip().lower()
    if normalized.endswith("_perf"):
        normalized = normalized[:-5]
    return normalized.replace("_", "").replace("-", "")


def _validation_has_wrong_nonzero(err: ValidationError) -> bool:
    """True if failure is due to rccl-tests wrong>0 (schema RcclTests.validate_wrong_is_zero)."""
    return any(item.get("type") == RCCL_WRONG_NONZERO_ERROR_TYPE for item in err.errors())


def _matching_rows(collective: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if _normalize_collective_name(collective) in {"alltoall", "alltoallv"}:
        return [row for row in rows if row["inPlace"] == 0]
    return [row for row in rows if row["inPlace"] == 1]


def _check_bus_bw(collective: str, rows: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> None:
    for row in _matching_rows(collective, rows):
        message_size = str(row["size"])
        if message_size not in thresholds:
            continue
        expected = float(thresholds[message_size]["bus_bw"])
        actual = float(row["busBw"])
        if actual < expected * 0.95:
            raise RuntimeError(
                f"{collective} bus BW {actual} for message size {message_size} is below expected {expected}"
            )


def _check_series_dip(
    collective: str,
    rows: list[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
    *,
    field: str,
    label: str,
) -> None:
    """Flag a >5% drop vs the previous in-threshold size (monotonic expectation for bus_bw or time)."""
    if not thresholds:
        return

    previous = 0.0
    previous_size = None
    valid_sizes = set(thresholds.keys())
    for row in _matching_rows(collective, rows):
        if str(row["size"]) not in valid_sizes:
            continue
        current = float(row[field])
        if previous and current < previous * 0.95:
            raise RuntimeError(
                f"{collective} {label} dip detected: {current} at size {row['size']} "
                f"after {previous} at size {previous_size}"
            )
        previous = current
        previous_size = row["size"]


def parse_and_validate_results(
    config: RcclConfig,
    collective: str,
    raw_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = config.validation_profile
    validation_summary: dict[str, Any] = {
        "parse": "passed",
        "schema": "skipped",
        "thresholds_bus_bw": "skipped",
        "bw_dip": "skipped",
        "lat_dip": "skipped",
    }

    if profile == "none":
        normalized_rows = list(raw_results)
        return normalized_rows, validation_summary

    validator = RcclTests if config.is_single_node else RcclTestsMultinodeRaw
    try:
        validated = [validator.model_validate(result) for result in raw_results]
    except ValidationError as err:
        if _validation_has_wrong_nonzero(err):
            raise RuntimeError(f"RCCL results for {collective} are corrupted (#wrong > 0)") from err
        raise RuntimeError(f"RCCL schema validation failed for {collective}: {err}") from err

    normalized_rows = [row.model_dump() for row in validated]
    validation_summary["schema"] = "passed"

    if profile == "smoke":
        return normalized_rows, validation_summary

    norm_thresholds = _normalize_threshold_map(config.thresholds.get(collective))
    if norm_thresholds:
        _check_bus_bw(collective, normalized_rows, norm_thresholds)
        validation_summary["thresholds_bus_bw"] = "passed"
        if profile == "strict":
            _check_series_dip(collective, normalized_rows, norm_thresholds, field="busBw", label="bus BW")
            validation_summary["bw_dip"] = "passed"
            _check_series_dip(collective, normalized_rows, norm_thresholds, field="time", label="latency")
            validation_summary["lat_dip"] = "passed"

    if profile in {"thresholds", "strict"}:
        return normalized_rows, validation_summary

    raise RuntimeError(f"Unexpected validation profile {profile!r}")
