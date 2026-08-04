'''
Generate a sample InferenceX ATOM Run Deck from unit-test fixtures.

Use this to demonstrate Run Deck output on ``main`` before the full
``inferencex_atom`` suite merges from ``dev/dtni``::

    python -m cvs.lib.report.demo.generate_inferencex_atom_rundeck --out sample_reports
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvs.lib.report.rundeck.payload import apply_summary_meta, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.unittests._fixtures import generic_variant, two_cell_inf_res


def generate(out_dir: Path) -> dict[str, Path]:
    profile = load_json_profile("inferencex_atom_single")
    if profile is None:
        raise SystemExit("profiles/inferencex_atom_single.json not found")

    store = {
        "inf_res_dict": two_cell_inf_res(),
        "variant_config": generic_variant(),
        "lifecycle_report": {
            "test/setup": [
                ("container_launch", "12.0", "s"),
                ("model_fetch", "45.0", "s"),
                ("server_ready", "90.0", "s"),
            ],
        },
    }
    payload = build_rundeck_payload(
        profile=profile,
        store=store,
        cvs_version="demo",
        provenance={"cvs_version": "demo", "pytest_html_href": "run.html"},
    )
    config = resolve_report_config(profile)
    payload = apply_summary_meta(payload, config)

    out_dir.mkdir(parents=True, exist_ok=True)
    basename = profile["report_basename"]
    html_path = out_dir / f"{basename}.html"
    json_path = out_dir / f"{basename}.json"
    html_path.write_text(render_rundeck_html(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {"html": html_path, "json": json_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sample_reports"),
        help="Output directory for HTML and JSON artifacts",
    )
    args = parser.parse_args(argv)
    paths = generate(args.out)
    print(f"Wrote {paths['html']}")
    print(f"Wrote {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
