'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Warnings recorded while building a consolidated report.
'''

from cvs.lib import globals

log = globals.log

INVALID_DECK = "invalid_deck"
UNREADABLE_FILE = "unreadable_file"
MISSING_FIELD = "missing_field"
TIED_TIMESTAMP = "tied_timestamp"
UNKNOWN_SUITE_ID = "unknown_suite_id"


class WarningLog:
    '''The ``warnings`` array of ``summary.json``. Every entry is also logged.'''

    def __init__(self):
        self.entries = []

    def add(self, code, message, file=None):
        log.warning("%s", message)
        self.entries.append({"code": code, "message": message, "file": str(file) if file else None})
