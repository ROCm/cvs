from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .case_ids import _slug
from .config import RcclConfig


def _ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _format_run_id_utc(now: datetime | None = None) -> str:
    dt = now if now is not None else datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H-%M-%SZ")


def _resolved_case_payload(config: RcclConfig, collective: str) -> dict[str, Any]:
    return {
        "collective": collective,
        "datatype": config.datatype,
        "start_size": config.start_size,
        "end_size": config.end_size,
        "step_factor": config.step_factor,
        "warmups": config.warmups,
        "iterations": config.iterations,
        "cycles": config.cycles,
        "env": {},
    }


def _build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    passed = sum(1 for c in cases if c.get("status") == "passed")
    failed = sum(1 for c in cases if c.get("status") == "failed")
    skipped = sum(1 for c in cases if c.get("status") == "skipped")
    if total > 0 and failed == 0:
        overall = "passed"
    else:
        overall = "failed"
    return {
        "cases_total": total,
        "cases_passed": passed,
        "cases_failed": failed,
        "cases_skipped": skipped,
        "overall_status": overall,
    }


def _write_optional_raw(run_dir: str, case_id: str, raw_results: list[dict[str, Any]], export_raw: bool) -> None:
    if not export_raw:
        return
    raw_dir = os.path.join(run_dir, "raw")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(raw_dir, f"{case_id}.json")
    with open(path, "w") as handle:
        json.dump(raw_results, handle, indent=2)


def _write_text_artifact(path: str, text: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_host_outputs(base_dir: str, outputs: dict[str, str]) -> None:
    for host, text in outputs.items():
        path = os.path.join(base_dir, f"{_slug(host)}.txt")
        _write_text_artifact(path, text)
