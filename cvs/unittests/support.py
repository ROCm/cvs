'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared helpers for CVS unit tests.

Unit tests run generated shell locally, parse XML they wrote themselves, and
assert the ``/tmp`` paths CVS uses on remote nodes. The static-analysis
suppressions for those patterns live here, once, instead of on each test line.
Tests also use ``FAKE_CRED`` instead of credential-looking literals.
'''

import json
import re
import subprocess  # nosec B404 - test helper; callers pass argv lists of commands the test itself builds
import xml.etree.ElementTree as ElementTree  # nosec B405 - parses only files the calling test generated
from pathlib import Path

TMP = "/tmp"  # nosec B108 - CVS uses /tmp as its shared remote scratch dir; tests assert the real paths
FAKE_CRED = "fake-cred"


def run_trusted(argv, **kwargs):
    """Run ``argv`` (a list, never a shell string) with :func:`subprocess.run`.

    ``kwargs`` are passed through unchanged; ``shell=True`` is rejected.
    """
    if isinstance(argv, str):
        raise TypeError("run_trusted takes an argv list, not a command string")
    if kwargs.get("shell"):
        raise ValueError("run_trusted does not run commands through a shell")
    return subprocess.run(argv, **kwargs)  # nosec B603 - argv list built by the calling test, shell disabled


def parse_trusted_xml(path):
    """Parse an XML file the calling test generated and return the ElementTree."""
    return ElementTree.parse(path)  # nosec B314 - input is a file the calling test generated


_PLACEHOLDER_RE = re.compile(r"<changeme(?:-[^<>]*)?>", re.IGNORECASE)


def write_filled_config(template_path, out_dir):
    """Write a copy of a shipped config with every ``<changeme>`` value filled in.

    Comment keys are copied unchanged. ``threshold_json`` is made absolute so
    the copy still reads the shipped threshold file. Returns the copy's path.
    """
    template_path = Path(template_path)

    def fill(node):
        if isinstance(node, dict):
            return {key: item if key.startswith("_") else fill(item) for key, item in node.items()}
        if isinstance(node, list):
            return [fill(item) for item in node]
        if isinstance(node, str):
            return _PLACEHOLDER_RE.sub("filled", node)
        return node

    config = fill(json.loads(template_path.read_text()))
    config["threshold_json"] = str(template_path.parent / config["threshold_json"])
    out_path = Path(out_dir) / template_path.name
    out_path.write_text(json.dumps(config))
    return out_path
