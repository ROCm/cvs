'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/utils/polling.py::poll_until.
'''

import unittest
from unittest.mock import MagicMock, patch

from cvs.lib.utils import polling


class TestPollUntil(unittest.TestCase):
    @patch.object(polling.time, 'sleep')
    def test_returns_once_done(self, mock_sleep):
        probe = MagicMock(side_effect=[1, 2, 3])

        self.assertEqual(polling.poll_until(probe, lambda r: r == 2, 60, 5), (2, True))
        self.assertEqual(probe.call_count, 2)
        mock_sleep.assert_called_once_with(5)

    @patch.object(polling.time, 'sleep')
    def test_timeout_returns_last_result_after_one_probe(self, mock_sleep):
        probe = MagicMock(return_value='pending')

        self.assertEqual(polling.poll_until(probe, lambda r: False, 0, 5), ('pending', False))
        probe.assert_called_once_with()
        mock_sleep.assert_not_called()

    @patch.object(polling.time, 'sleep')
    @patch.object(polling.time, 'monotonic', side_effect=[0, 7])
    def test_sleep_never_passes_deadline(self, _monotonic, mock_sleep):
        probe = MagicMock(side_effect=[None, 'ok'])

        polling.poll_until(probe, bool, 10, 5)

        mock_sleep.assert_called_once_with(3)


if __name__ == '__main__':
    unittest.main()
