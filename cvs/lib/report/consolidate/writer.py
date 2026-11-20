'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Assemble the consolidated report folder and write it as one zip.
'''

import html
import json
import os
import re
import shutil
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from cvs.lib import globals
from cvs.lib.report.consolidate.discovery import ATTACHMENTS_SUFFIX
from cvs.lib.report.consolidate.warning_log import UNREADABLE_FILE
from cvs.lib.report.rundeck.runtime.theme import OVERVIEW_LINK_MARKER, THEME_BOOT_SCRIPT, tokens_css

log = globals.log

PAGE_DIR = Path(__file__).with_name("page")
BACK_LINK_ID = "cvs-consolidated-report-link"
LARGE_ZIP_BYTES = 500 * 1024 * 1024
_HEADER_LINK = '<a id="%s" class="cvs-back" href="%s">&larr; Consolidated report</a>'
# Decks written before the CVS header have no slot for the link, so it gets a bar of its own.
_BACK_LINK = (
    '<nav id="%s" style="font:13px/1.5 system-ui,sans-serif;padding:8px 16px;'
    'border-bottom:1px solid rgba(128,128,128,.35)"><a href="%s">&larr; Consolidated report</a></nav>'
)
_BODY_TAG = re.compile(r"<body\b[^>]*>", re.IGNORECASE)
_ZIP_READ_ERRORS = (zipfile.BadZipFile, EOFError, RuntimeError, NotImplementedError, zlib.error)


def place_run(deck, variant, report_root, warnings):
    '''Copy one run's files to ``runs/<variant>/<run>/`` and return links to them, relative to ``report_root``.'''
    runs_dir = report_root / "runs" / variant
    if deck.in_bundle:
        target = _new_dir(runs_dir, deck.source.stem)
        try:
            extract_bundle(deck.source, target)
        except _ZIP_READ_ERRORS as exc:
            warnings.add(UNREADABLE_FILE, "Could not extract every file from %s: %s" % (deck.source, exc), deck.source)
    else:
        target = _new_dir(runs_dir, deck.run_root.name)
        _copy_run_folder(deck, target)
    return run_links(deck, target, report_root)


def _new_dir(parent, name):
    candidate, n = parent / name, 1
    while candidate.exists():
        candidate = parent / ("%s-%d" % (name, n))
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def extract_bundle(zip_path, target):
    '''Extract a run bundle into ``target``, skipping any member that would land outside it.'''
    base = target.resolve()
    with zipfile.ZipFile(zip_path) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            out = (base / info.filename).resolve()
            if base not in out.parents:
                log.debug("Skipped %s in %s: outside the bundle folder", info.filename, zip_path)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _report_and_log_names(deck):
    '''File names of the run's pytest report and log: from the deck's provenance, else named after its stem.'''
    provenance = deck.data.get("provenance") if isinstance(deck.data, dict) else None
    provenance = provenance if isinstance(provenance, dict) else {}
    names = []
    for key, suffix in (("pytest_html_path", ".html"), ("log_file_path", ".log")):
        value = provenance.get(key)
        name = PurePosixPath(value.strip().replace("\\", "/")).name if isinstance(value, str) else ""
        if not name and deck.stem:
            name = deck.stem + suffix
        names.append(name if name not in ("", ".", "..") else None)
    return names


def _copy_tree(source, target):
    # Contents only: copied read-only attributes or permissions would stop the temporary folder being deleted.
    for dirpath, _, filenames in os.walk(source):
        folder = target / Path(dirpath).relative_to(source)
        folder.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            shutil.copyfile(Path(dirpath) / name, folder / name)


def _copy_run_folder(deck, target):
    folder = deck.source.parent
    if folder.name.endswith(ATTACHMENTS_SUFFIX):
        _copy_tree(folder, target / folder.name)
    else:
        for source in sorted(folder.iterdir()):
            if source.is_file() and source.name.startswith(deck.source.stem):
                shutil.copyfile(source, target / source.name)
    for name in _report_and_log_names(deck):
        source = deck.run_root / name if name else None
        if source is not None and source.is_file() and not (target / name).exists():
            shutil.copyfile(source, target / name)
    assets = deck.run_root / "assets"
    if assets.is_dir():
        _copy_tree(assets, target / "assets")


def run_links(deck, target, report_root):
    '''Links to a placed run's deck, viewer, deck JSON, pytest report, and log; None for files the run lacks.'''

    def link(path):
        return path.relative_to(report_root).as_posix() if path is not None and path.is_file() else None

    json_path = target.joinpath(*PurePosixPath(deck.json_rel).parts)
    summary = deck.data.get("summary") if isinstance(deck.data, dict) else None
    viewer = summary.get("viewer_html") if isinstance(summary, dict) else None
    viewer_name = PurePosixPath(viewer).name if isinstance(viewer, str) and viewer else json_path.stem + "_viewer.html"
    report_name, log_name = _report_and_log_names(deck)
    log_link = None
    if log_name:
        log_link = link(target / log_name) or link(json_path.parent / log_name)
    return {
        "jsonUrl": link(json_path),
        "reportUrl": link(json_path.with_suffix(".html")),
        "viewerUrl": link(json_path.parent / viewer_name),
        "pytestUrl": link(target / report_name) if report_name else None,
        "logUrl": log_link,
    }


def add_back_link(page, report_root):
    '''Put a link back to the overview at the top of a copied deck or viewer page. Returns True when added.'''
    try:
        document = page.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.debug("No back-link added to %s: %s", page, exc)
        return False
    if BACK_LINK_ID in document:
        return False
    href = html.escape(os.path.relpath(report_root / "index.html", page.parent).replace(os.sep, "/"))
    if OVERVIEW_LINK_MARKER in document:
        updated = document.replace(OVERVIEW_LINK_MARKER, _HEADER_LINK % (BACK_LINK_ID, href), 1)
    else:
        nav = _BACK_LINK % (BACK_LINK_ID, href)
        updated, count = _BODY_TAG.subn(lambda match: match.group(0) + nav, document, count=1)
        if not count:
            return False
    page.write_text(updated, encoding="utf-8")
    return True


def _render_page(template, css_files, js_files, page_data, title=None):
    page = (PAGE_DIR / template).read_text(encoding="utf-8")
    css = tokens_css() + "".join((PAGE_DIR / name).read_text(encoding="utf-8") for name in css_files)
    script = "\n".join((PAGE_DIR / name).read_text(encoding="utf-8") for name in js_files)
    payload = json.dumps(page_data, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    page = page.replace("__THEME_BOOT__", THEME_BOOT_SCRIPT).replace("__CSS__", css).replace("__JS__", script)
    if title is not None:
        page = page.replace("__TITLE__", html.escape(title))
    return page.replace("__DATA__", payload)


def render_index(page_data):
    '''The overview page, with its data embedded so it opens from disk without a server.'''
    return _render_page("overview.html", ("overview.css",), ("shared.js", "overview.js"), page_data)


def render_suite_page(page_data):
    '''One suite's page in the AI Lab deck layout, built from its Run Deck JSON, with its data embedded.'''
    title = "%s \u00b7 CVS consolidated report" % page_data["suite"]["name"]
    return _render_page("suite.html", ("overview.css", "suite.css"), ("shared.js", "suite.js"), page_data, title)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_zip(report_root, output_dir):
    '''Zip ``report_root`` under a hidden name in ``output_dir``, then rename it into place.

    The zip is named after ``report_root``, with a numeric suffix when that name is taken.
    Returns the path of the zip.
    '''
    name = report_root.name
    partial = output_dir / (".%s.zip.tmp-%d" % (name, os.getpid()))
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as out:
            for path in sorted(report_root.rglob("*")):
                if path.is_file():
                    out.write(path, "%s/%s" % (name, path.relative_to(report_root).as_posix()))
        final, n = output_dir / (name + ".zip"), 1
        while final.exists():
            final = output_dir / ("%s-%d.zip" % (name, n))
            n += 1
        os.replace(partial, final)
    finally:
        if partial.exists():
            partial.unlink()
    size = final.stat().st_size
    if size > LARGE_ZIP_BYTES:
        log.warning("The consolidated report is %d MB; sharing it may need a large-file service.", size // 2**20)
    return final
