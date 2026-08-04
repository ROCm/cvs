#!/usr/bin/env python3
'''
Generate sample Run Deck HTML/JSON artifacts for review (no cluster run required).

Writes into sample_reports/ by default::

    python scripts/generate_sample_rundecks.py
    python scripts/generate_sample_rundecks.py --out sample_reports
'''

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cvs.lib.report.profile import load_json_profile, profile_json_path
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.payload import apply_summary_meta, build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.unittests._fixtures import generic_variant, two_cell_inf_res
from cvs.lib.report.viewer.scaffold import viewer_basename_for, write_interactive_viewer

# Inference sweep demos (shared cell shape).
INFERENCE_STORE = {
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

VLLM_STORE = {
    **INFERENCE_STORE,
    "lifecycle_report": {
        "test/setup": [
            ("container_launch", "10.0", "s"),
            ("topology_discovery", "5.0", "s"),
            ("model_fetch", "40.0", "s"),
            ("server_ready", "85.0", "s"),
        ],
    },
}

RCCL_GRAPH = {
    "all_reduce": {
        "8": {"bus_bw": 12.5, "alg_bw": 11.0, "time": 100},
        "64": {"bus_bw": 45.0, "alg_bw": 40.0, "time": 200},
        "512": {"bus_bw": 98.0, "alg_bw": 88.0, "time": 350},
    },
    "all_gather": {
        "8": {"bus_bw": 10.2, "alg_bw": 9.1, "time": 110},
        "64": {"bus_bw": 38.0, "alg_bw": 34.0, "time": 220},
    },
}

RCCL_STORE = {
    "cvs_results_dict": RCCL_GRAPH,
    "variant_config": generic_variant(),
}


def _write_profile_sample(stem: str, store: dict, out_dir: Path) -> dict[str, Path] | None:
    profile = load_json_profile(stem)
    if profile is None:
        print(f"  skip {stem}: no profile at {profile_json_path(stem)}", file=sys.stderr)
        return None

    payload = build_rundeck_payload(
        profile=profile,
        store=store,
        cvs_version="demo",
        provenance={"cvs_version": "demo", "pytest_html_href": "run.html", "image_display": "demo"},
    )
    config = resolve_report_config(profile)
    payload = apply_summary_meta(payload, config)

    basename = profile["report_basename"]
    html_path = out_dir / f"{basename}.html"
    json_path = out_dir / f"{basename}.json"
    html_path.write_text(render_rundeck_html(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    paths: dict[str, Path] = {"html": html_path, "json": json_path}

    builder = profile.get("dataset_builder", "sweep")
    if config.interactive_viewer and builder == "sweep":
        viewer_path = out_dir / viewer_basename_for(basename)
        write_interactive_viewer(
            viewer_path,
            json_basename=f"{basename}.json",
            title=config.title,
            subtitle=config.subtitle,
            tier_order=config.metric_tier_order,
            embed_payload=payload,
        )
        paths["viewer"] = viewer_path

    return paths


def generate_all(out_dir: Path) -> list[dict[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("inferencex_atom_single", INFERENCE_STORE),
        ("vllm", VLLM_STORE),
        ("rccl_perf", RCCL_STORE),
    ]
    written = []
    for stem, store in jobs:
        print(f"Generating {stem}...")
        paths = _write_profile_sample(stem, store, out_dir)
        if paths:
            written.append(paths)
            print(f"  {paths['html']}")
            print(f"  {paths['json']}")
            if paths.get("viewer"):
                print(f"  {paths['viewer']}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "sample_reports")
    args = parser.parse_args(argv)
    written = generate_all(args.out)
    if not written:
        return 1
    print(f"\nWrote {len(written)} Run Deck sample(s) to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
