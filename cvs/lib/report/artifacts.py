'''Shared HTML + JSON artifact writers for suite reports.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping


def export_payload(payload: Mapping[str, Any]) -> dict:
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def write_html_json_artifacts(
    html_path: Path,
    *,
    payload: Mapping[str, Any],
    render_html: Callable[[Mapping[str, Any]], str],
) -> tuple[Path, Path]:
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(payload), encoding="utf-8")
    json_path = html_path.with_suffix(".json")
    json_path.write_text(json.dumps(export_payload(payload), indent=2, default=str), encoding="utf-8")
    return html_path, json_path
