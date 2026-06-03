"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import re
from typing import List, Optional

from cvs.lib.cluster.pool import Node

_SPLIT_RE = re.compile(r"[\s,]+")


def _required_labels(selector: Optional[str]) -> List[str]:
    if not selector:
        return []
    return [tok for tok in _SPLIT_RE.split(selector.strip()) if tok]


def node_matches(node: Node, selector: Optional[str]) -> bool:
    """Return True if ``node`` satisfies the role ``selector``.

    Selector grammar (intentionally simple and deterministic): a
    whitespace/comma-separated list of required labels, all of which must be
    present in ``node.labels`` (logical AND). Empty/None matches any node.
    """
    required = _required_labels(selector)
    if not required:
        return True
    return all(label in node.labels for label in required)
