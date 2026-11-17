'''Tests for the consolidated report's warning log.'''

import unittest

from cvs.lib.report.consolidate.warning_log import INVALID_DECK, TIED_TIMESTAMP, WarningLog


class TestWarningLog(unittest.TestCase):
    def test_add_records_and_logs_the_warning(self):
        warnings = WarningLog()
        with self.assertLogs(level="WARNING") as logs:
            warnings.add(INVALID_DECK, "deck.json is malformed.", "/runs/deck.json")
        self.assertEqual(
            warnings.entries,
            [{"code": "invalid_deck", "message": "deck.json is malformed.", "file": "/runs/deck.json"}],
        )
        self.assertIn("deck.json is malformed.", logs.output[0])

    def test_file_is_optional(self):
        warnings = WarningLog()
        with self.assertLogs(level="WARNING"):
            warnings.add(TIED_TIMESTAMP, "Two runs tie.")
        self.assertIsNone(warnings.entries[0]["file"])

    def test_has(self):
        warnings = WarningLog()
        self.assertFalse(warnings.has(INVALID_DECK))
        with self.assertLogs(level="WARNING"):
            warnings.add(INVALID_DECK, "bad")
        self.assertTrue(warnings.has(INVALID_DECK))
        self.assertFalse(warnings.has(TIED_TIMESTAMP))


if __name__ == "__main__":
    unittest.main()
