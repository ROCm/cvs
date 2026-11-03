'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared helpers for CVS unit tests.

Unit tests run generated shell locally, parse XML they wrote themselves, and
assert the ``/tmp`` paths CVS uses on remote nodes. The static-analysis
suppressions for those patterns live here, once, instead of on each test line.
Tests also use ``FAKE_CRED`` instead of credential-looking literals.
'''

import subprocess  # nosec B404 - test helper; callers pass argv lists of commands the test itself builds
import xml.etree.ElementTree as ElementTree  # nosec B405 - parses only files the calling test generated

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
