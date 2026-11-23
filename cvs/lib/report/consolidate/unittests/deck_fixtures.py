'''Decks and run layouts for consolidated report tests, shaped like what ``cvs run`` writes.'''

import json
import zipfile
from pathlib import Path

from cvs.lib.report.consolidate.discovery import FoundDeck


def deck(suite_id, overall_status="pass", builder="status_matrix", dataset=None, **fields):
    '''A valid deck JSON object. ``fields`` override or add top-level keys.

    A status-matrix deck without a ``dataset`` gets one node whose check matches ``overall_status``.
    '''
    if dataset is None and builder == "status_matrix":
        dataset = status_matrix({"node-01": {"check": overall_status if overall_status in ("fail", "na") else "pass"}})
    data = {
        "schema_version": 1,
        "suite_id": suite_id,
        "generated_at": "2026-11-10 19:32 UTC",
        "cvs_version": "0.4.0",
        "overall_status": overall_status,
        "report": {"title": "%s Run Deck" % suite_id},
        "run_card_display": [["Cluster", "c1", False]],
        "provenance": {},
        "datasets": {builder: dataset if dataset is not None else {}},
        "deck_profile": {"cards": []},
    }
    data.update(fields)
    return data


def status_matrix(rows):
    '''A status_matrix dataset from ``{node: {group: status}}``.'''
    groups = sorted({group for cells in rows.values() for group in cells})
    grid = {
        node: {group: {"status": status, "items": [], "metrics": []} for group, status in cells.items()}
        for node, cells in rows.items()
    }
    return {"nodes": list(rows), "groups": groups, "grid": grid}


def run_files(stem, data, basename=None):
    '''Files of one run, by path relative to the run folder, as both the folder and its bundle hold them.'''
    suite_id = data.get("suite_id") if isinstance(data, dict) else stem
    base = basename or "%s_run_deck" % suite_id
    return {
        "%s.html" % stem: "<html><body>pytest report</body></html>",
        "%s_html/%s.json" % (stem, base): data if isinstance(data, str) else json.dumps(data, indent=2),
        "%s_html/%s.html" % (stem, base): "<html><head></head><body><h1>%s</h1></body></html>" % base,
        "%s_html/%s.log" % (stem, stem): "run log",
    }


def write_run(root, stem, data, run_id="local-1", folder=True, bundle=None, basename=None):
    '''Write one run under ``root/run_id``: unzipped when ``folder``, and as ``<stem>_<bundle>.zip`` when given.'''
    run = Path(root) / run_id
    run.mkdir(parents=True, exist_ok=True)
    files = run_files(stem, data, basename)
    if folder:
        for rel, content in files.items():
            path = run / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
        (run / ("%s.log" % stem)).write_bytes(b"run log")
    if bundle:
        with zipfile.ZipFile(run / ("%s_%s.zip" % (stem, bundle)), "w") as out:
            for rel, content in files.items():
                out.writestr(rel, content)
    return run


def found(root, data, json_rel="x_html/x_run_deck.json", source=None, bundle=False):
    '''A FoundDeck without files on disk, for code that only reads its fields.'''
    root = Path(root)
    raw = json.dumps(data).encode("utf-8") if data is not None else b"{"
    if bundle:
        return FoundDeck(root, source or root / "x_2026-11-10T193205.zip", json_rel, raw, data)
    source = source or root / "run" / json_rel
    return FoundDeck(root, source, json_rel, raw, data, run_root=source.parent.parent)
