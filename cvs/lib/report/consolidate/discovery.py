'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Find Run Deck JSON under one or more directories, both in run folders and inside
the zip bundles ``cvs run`` writes next to them.
'''

import hashlib
import json
import os
import re
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from cvs.lib import globals
from cvs.lib.report.consolidate.warning_log import UNREADABLE_FILE

log = globals.log

SCHEMA_VERSION = 1
OUTPUT_PREFIX = "consolidated_report_"
WORK_PREFIX = ".consolidated_report"
ATTACHMENTS_SUFFIX = "_html"
BUNDLE_NAME = re.compile(r"^(?P<stem>.+)_\d{4}-\d{2}-\d{2}T\d{6}\.zip$")
CANDIDATE_KEYS = ("schema_version", "suite_id", "deck_profile")
_ZIP_ERRORS = (
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    OSError,
    EOFError,
    RuntimeError,
    NotImplementedError,
    zlib.error,
)


class FoundDeck:
    '''One deck JSON found in a run folder or a zip bundle.

    ``data`` is the parsed deck, or None when a file in a ``<stem>_html/`` folder
    cannot be parsed, for example because it was still being written.
    ``source`` is the JSON file for a run folder and the zip file for a bundle.
    ``json_rel`` is the JSON path relative to ``run_root``, or the zip member name.
    ``run_root`` is the run folder that holds ``<stem>_html/``; None for a bundle.
    '''

    def __init__(self, root, source, json_rel, raw, data, run_root=None):
        self.root = root
        self.source = source
        self.json_rel = json_rel
        self.data = data
        self.digest = hashlib.sha256(raw).hexdigest()
        self.run_root = run_root
        self.stem = resolve_stem(data, json_rel, source if run_root is None else None)

    @property
    def in_bundle(self):
        return self.run_root is None

    @property
    def relative_path(self):
        rel = self.source.relative_to(self.root).as_posix()
        return "%s/%s" % (rel, self.json_rel) if self.in_bundle else rel

    @property
    def display_path(self):
        return "%s/%s" % (self.source, self.json_rel) if self.in_bundle else str(self.source)


def is_candidate(data):
    return isinstance(data, dict) and all(key in data for key in CANDIDATE_KEYS)


def is_valid(data):
    return (
        is_candidate(data)
        and data.get("schema_version") == SCHEMA_VERSION
        and isinstance(data.get("overall_status"), str)
        and bool(data.get("overall_status"))
        and isinstance(data.get("datasets"), dict)
        and isinstance(data.get("report"), dict)
        and bool(data.get("report"))
    )


def resolve_stem(data, json_rel, bundle=None):
    '''Return the suite variant key: the stem of the test file that wrote the deck.

    In order: the deck's ``suite_stem``, the ``<stem>_html/`` folder holding it, the
    pytest report named in its provenance, the bundle name, then its ``suite_id``.
    The folder comes before the pytest report because ``--html`` can rename the report.
    '''
    if isinstance(data, dict):
        declared = data.get("suite_stem")
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
    folder = PurePosixPath(json_rel).parent.name
    if folder.endswith(ATTACHMENTS_SUFFIX) and len(folder) > len(ATTACHMENTS_SUFFIX):
        return folder[: -len(ATTACHMENTS_SUFFIX)]
    if isinstance(data, dict):
        provenance = data.get("provenance")
        pytest_html = provenance.get("pytest_html_path") if isinstance(provenance, dict) else None
        if isinstance(pytest_html, str) and pytest_html.strip():
            return PurePosixPath(pytest_html.strip().replace("\\", "/")).stem
    if bundle is not None:
        match = BUNDLE_NAME.match(bundle.name)
        if match:
            return match.group("stem")
    if isinstance(data, dict) and isinstance(data.get("suite_id"), str) and data["suite_id"]:
        return data["suite_id"]
    return None


def is_own_output(name):
    '''True for this command's zips, extracted reports, and temporary folders, which are never read as input.'''
    return name.startswith((OUTPUT_PREFIX, WORK_PREFIX))


def discover(roots, warnings):
    '''Return ``(decks, ignored)``: every deck under ``roots`` and the number of other JSON and zip files skipped.'''
    decks, ignored = [], 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(name for name in dirnames if not is_own_output(name))
            for name in sorted(filenames):
                path = Path(dirpath) / name
                lowered = name.lower()
                if lowered.endswith(".json"):
                    found = _read_folder_deck(root, path, warnings)
                elif lowered.endswith(".zip") and not is_own_output(name):
                    found = _read_bundle_decks(root, path, warnings)
                else:
                    continue
                if found is None:
                    ignored += 1
                else:
                    decks.extend(found)
    return decks, ignored


def _parse(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def _in_attachments(rel_path):
    return PurePosixPath(rel_path).parent.name.endswith(ATTACHMENTS_SUFFIX)


def _read_folder_deck(root, path, warnings):
    '''Return a list with the deck at ``path``, an empty list when it cannot be read, or None when it is not a deck.'''
    try:
        raw = path.read_bytes()
    except OSError as exc:
        warnings.add(UNREADABLE_FILE, "Could not read %s: %s" % (path, exc), path)
        return []
    data = _parse(raw)
    in_attachments = path.parent.name.endswith(ATTACHMENTS_SUFFIX)
    if data is None and not in_attachments:
        log.debug("Ignored %s: not valid JSON", path)
        return None
    if data is not None and not is_candidate(data):
        log.debug("Ignored %s: not a Run Deck", path)
        return None
    run_root = path.parent.parent if in_attachments else path.parent
    return [FoundDeck(root, path, path.relative_to(run_root).as_posix(), raw, data, run_root=run_root)]


def _read_bundle_decks(root, path, warnings):
    '''Return the decks inside the zip at ``path``, an empty list when it cannot be opened, or None when it has none.'''
    found = []
    try:
        with zipfile.ZipFile(path) as bundle:
            for info in bundle.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".json"):
                    continue
                raw = bundle.read(info)
                data = _parse(raw)
                if (data is None and _in_attachments(info.filename)) or is_candidate(data):
                    found.append(FoundDeck(root, path, info.filename, raw, data))
    except _ZIP_ERRORS as exc:
        warnings.add(UNREADABLE_FILE, "Could not open %s: %s" % (path, exc), path)
        return []
    if not found:
        log.debug("Ignored %s: no Run Deck inside", path)
        return None
    return found


def drop_duplicates(decks):
    '''Return ``(unique, duplicates)``.

    A run folder holds each deck twice, unzipped in ``<stem>_html/`` and inside its
    bundle. Decks with byte-identical JSON count once, and the unzipped copy is kept.
    '''
    ordered = sorted(decks, key=lambda deck: (deck.in_bundle, str(deck.source), deck.json_rel))
    seen, unique, duplicates = set(), [], []
    for deck in ordered:
        if deck.digest in seen:
            log.debug("Duplicate dropped: %s", deck.display_path)
            duplicates.append(deck)
        else:
            seen.add(deck.digest)
            unique.append(deck)
    return unique, duplicates
