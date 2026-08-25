'''Framework parity panel — compare perf cells against a reference ATOM run (M4).'''

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from cvs.lib.report.compare import build_framework_parity_row
from cvs.lib.report.json_io import cell_id_host_key, index_cells_by_id_host, load_report_json
from cvs.lib.report.metrics import HEADLINE_THROUGHPUT_METRIC

PARITY_REF_ENV = "CVS_ATOM_PARITY_REF_JSON"


def resolve_parity_ref_json_path(config_path: str = "") -> str:
    explicit = (config_path or os.environ.get(PARITY_REF_ENV, "")).strip()
    return explicit


def parity_metric_prefix(driver: str) -> str:
    if driver == "vllm_atom":
        return "compare.vllm"
    if driver == "sglang":
        return "compare.sglang"
    return "compare.framework"


def build_framework_parity_panel(
    cells: List[dict],
    reference_json_path: Path,
    *,
    driver: str = "atom",
    headline_metric: str = HEADLINE_THROUGHPUT_METRIC,
) -> Optional[dict]:
    if not reference_json_path.is_file():
        return None
    reference = index_cells_by_id_host(load_report_json(reference_json_path) or {})
    if not reference:
        return None

    prefix = parity_metric_prefix(driver)
    ratio_key = f"{prefix}.output_throughput_ratio"
    ttft_key = f"{prefix}.mean_ttft_ms_ratio"

    rows = []
    for cell in cells:
        ref_cell = reference.get(cell_id_host_key(cell))
        rows.append(
            build_framework_parity_row(
                cell,
                ref_cell,
                headline_metric=headline_metric,
                ratio_metric_key=ratio_key,
                ttft_metric_key=ttft_key,
            )
        )

    return {
        "reference_json": str(reference_json_path),
        "driver": driver,
        "headline_metric": headline_metric,
        "rows": rows,
    }
