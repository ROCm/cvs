"""
test_globals.py

Unit tests for the pssh host_logger suppression applied at cvs.lib.globals
import time.
"""

import logging
import unittest
from unittest import mock

import cvs.lib.globals
from cvs.lib.globals import get_verbosity, set_verbosity, verbose_log

HOST_LOGGER = 'pssh.host_logger'


class _CollectingHandler(logging.Handler):
    """Records every LogRecord it is handed."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _emit_host_line():
    """Emit a line the way pssh/clients/base/single.py does."""
    logging.getLogger(HOST_LOGGER).info("[%s]%s\t%s", '10.0.0.1', '', 'remote stdout line')


class TestHostLoggerSuppression(unittest.TestCase):
    def test_handler_attached_directly_to_host_logger_receives_nothing(self):
        # A handler bolted straight onto pssh.host_logger bypasses propagation
        # entirely. This is what pytest's catching_logs does, so suppressing by
        # detaching the logger from root is not enough -- the suppression has to
        # stop the record before any handler is consulted.
        handler = _CollectingHandler()
        host_logger = logging.getLogger(HOST_LOGGER)
        root = logging.getLogger()
        orig_level = root.level
        host_logger.addHandler(handler)
        # host_logger sets no level of its own, so without this the INFO record
        # is never created and the test would pass without exercising anything.
        root.setLevel(logging.INFO)
        try:
            _emit_host_line()
        finally:
            root.setLevel(orig_level)
            host_logger.removeHandler(handler)

        self.assertEqual(handler.records, [])

    def test_no_duplicate_captured_under_pytest_catching_logs(self):
        # The real mechanism: _pytest.logging.catching_logs attaches its handler
        # to root AND to every non-propagating logger.
        try:
            from _pytest.logging import catching_logs
        except ImportError:  # pragma: no cover - pytest is a declared dependency
            self.skipTest("pytest not installed")

        handler = _CollectingHandler()
        with catching_logs(handler, level=logging.INFO):
            _emit_host_line()

        host_lines = [r for r in handler.records if r.name == HOST_LOGGER]
        self.assertEqual(host_lines, [])

    def test_other_loggers_are_still_captured(self):
        # Guards against suppressing more than the one third-party logger.
        try:
            from _pytest.logging import catching_logs
        except ImportError:  # pragma: no cover - pytest is a declared dependency
            self.skipTest("pytest not installed")

        handler = _CollectingHandler()
        with catching_logs(handler, level=logging.INFO):
            logging.getLogger('cvs.some.module').info("kept")

        self.assertEqual([r.getMessage() for r in handler.records], ["kept"])

    def test_filter_is_identifiable_by_name(self):
        # The filter has to be recognisable in
        # logging.getLogger('pssh.host_logger').filters when someone is
        # debugging log routing on a live node. An anonymous lambda shows up
        # there as a bare <function <lambda>> with no hint of what installed it
        # or why, so the suppression is pinned to a named callable.
        installed = logging.getLogger(HOST_LOGGER).filters
        names = [getattr(f, '__name__', '') for f in installed]
        self.assertIn('_suppress_pssh_host_logger', names, f"no named suppression filter among {installed}")


class TestVerboseLog(unittest.TestCase):
    def setUp(self):
        self._saved = cvs.lib.globals.verbosity
        set_verbosity(0)

    def tearDown(self):
        set_verbosity(self._saved)

    def test_silent_when_verbosity_is_zero(self):
        agent = logging.getLogger('cvs.agent')
        with mock.patch.object(agent, 'debug') as debug:
            verbose_log(logging.getLogger('cvs.test.verbose_log'), "hidden", 1)

        debug.assert_not_called()

    def test_logs_when_current_verbosity_meets_level(self):
        set_verbosity(1)
        agent = logging.getLogger('cvs.agent')
        with mock.patch.object(agent, 'debug') as debug:
            verbose_log(logging.getLogger('cvs.test.verbose_log'), "shown", 1)

        debug.assert_called_once_with("shown", stacklevel=2)

    def test_logs_requested_levels_up_to_current_count(self):
        set_verbosity(3)
        agent = logging.getLogger('cvs.agent')
        with mock.patch.object(agent, 'debug') as debug:
            verbose_log(logging.getLogger('cvs.test.verbose_log'), "v1", 1)
            verbose_log(logging.getLogger('cvs.test.verbose_log'), "v2", 2)
            verbose_log(logging.getLogger('cvs.test.verbose_log'), "v3", 3)
            verbose_log(logging.getLogger('cvs.test.verbose_log'), "v4", 4)

        self.assertEqual([call.args[0] for call in debug.call_args_list], ["v1", "v2", "v3"])

    def test_set_verbosity_floors_negative_to_zero(self):
        set_verbosity(-2)
        self.assertEqual(get_verbosity(), 0)

    def test_httpx_request_logs_hidden_until_vvv(self):
        handler = _CollectingHandler()
        httpx_log = logging.getLogger("httpx")
        httpx_log.addHandler(handler)
        try:
            set_verbosity(0)
            httpx_log.info('HTTP Request: POST http://h:1/v1/exec "HTTP/1.1 200 OK"')
            self.assertEqual(handler.records, [])

            set_verbosity(3)
            httpx_log.info('HTTP Request: POST http://h:1/v1/exec "HTTP/1.1 200 OK"')
            self.assertEqual(len(handler.records), 1)
        finally:
            httpx_log.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()
