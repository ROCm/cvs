'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re

# The pass gate predates this module; keep it byte-for-byte so verdicts do not change.
TOTAL_FAILED_ZERO_RE = re.compile(r'"total_failed":\s+0,', re.I)
TOTAL_FAILED_RE = re.compile(r'"total_failed"\s*:\s*\d+', re.I)
_DETAIL_LIMIT = 200


def results_text(output):
    """Return the text from a results.json read."""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return str(output.get('output') or '')
    if output is None:
        return ''
    return str(output)


def describe_read(text):
    """Return a short, single-line description of a failed read."""
    detail = ' '.join(results_text(text).split())
    if not detail:
        return 'no output'
    if len(detail) > _DETAIL_LIMIT:
        return detail[:_DETAIL_LIMIT] + '...'
    return detail


def classify_results_json(text):
    """Classify results.json, preserving the historical zero pass gate unchanged."""
    body = results_text(text)
    if TOTAL_FAILED_ZERO_RE.search(body):
        return 'pass', None
    found = TOTAL_FAILED_RE.search(body)
    if found:
        return 'nonzero', found.group(0)
    return 'unread', describe_read(body)
