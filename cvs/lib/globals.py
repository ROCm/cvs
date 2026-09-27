'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import logging

log = logging.getLogger()


# pssh's own host_logger emits every remote stdout/stderr line, tagged with the
# host (pssh/clients/base/single.py). Upstream keeps it quiet behind a
# NullHandler unless enable_host_logger() is called -- which CVS never does --
# but `log` above is the ROOT logger, so propagation delivers those lines to
# CVS's handlers anyway. The result is that every line is logged twice: once by
# pssh, once by Pssh._process_output (cvs/lib/parallel/pssh.py). Dropping the
# pssh copy keeps the _process_output one, which is the one that honors
# print_console and can therefore be suppressed for bulk-data commands.
#
# A filter, not propagate=False: pytest's catching_logs attaches its capture
# handler to root AND to every non-propagating logger (_pytest/logging.py), so
# clearing propagate makes pytest attach directly and the duplicate survives.
# A filter drops the record before any handler is consulted, however attached.
#
# Named rather than a lambda so it is identifiable in
# logging.getLogger('pssh.host_logger').filters when someone is debugging log
# routing on a live node.
def _suppress_pssh_host_logger(_record):
    return False


logging.getLogger('pssh.host_logger').addFilter(_suppress_pssh_host_logger)

error_list = []


def _sanitize_log_content(content):
    """Return log-safe single-line text to prevent log injection via CR/LF."""
    return str(content).replace('\r', '').replace('\n', '')


# CLI -v count: 0 default, 1 = -v, 2 = -vv, 3 = -vvv, ...
# verbose_log() emits only when this value is at least the caller's requested level.
verbosity = 0


def _apply_httpx_log_level():
    # httpx logs every request at INFO:
    #   HTTP Request: POST http://host:port/v1/exec "HTTP/1.1 200 OK"
    # That is one line per host per command and floods managed-compute pytest
    # logs. Agent verbose_log() already covers that path; keep httpx (and
    # httpcore) at WARNING unless the operator asked for -vvv.
    level = logging.INFO if verbosity >= 3 else logging.WARNING
    logging.getLogger("httpx").setLevel(level)
    logging.getLogger("httpcore").setLevel(level)


def set_verbosity(level):
    """
    Set the current CLI verbosity count.

    0 is the default (no -v). Each additional -v increments the count
    (1 = -v, 2 = -vv, 3 = -vvv, ...). verbose_log() emits only when
    this value is at least the caller's requested level.

    Args:
        level: Non-negative integer verbosity count.
    """
    global verbosity
    verbosity = max(0, int(level))
    _apply_httpx_log_level()


def get_verbosity():
    """Return the current CLI verbosity count."""
    return verbosity


def verbose_log(logger, content, verbosity_level):
    """
    Log content at DEBUG when the current CLI verbosity is at least verbosity_level.

    Records go to the cvs.agent logger (cvs.core.agent.logger). On a managed run
    with -v / -vv / -vvv they are written to {run_dir}/agent/rankN.log, including
    startup and worker traces. The same records propagate to the root logger, so
    pytest shows them when it is run with --log-level=DEBUG. The logger argument
    is kept for call sites; emission does not use that logger.

    Args:
        logger: Call-site logger. Retained so existing callers stay unchanged.
        content: Message to log.
        verbosity_level: Minimum -v count required to emit (1, 2, 3, ...).

    Example:
        from cvs.lib.globals import verbose_log

        verbose_log(log, "connecting to node", 1)   # shown at -v or higher
        verbose_log(log, "SSH handshake details", 2)  # shown at -vv or higher
        verbose_log(log, "full packet dump", 3)     # shown at -vvv or higher
    """
    if verbosity >= int(verbosity_level):
        # Imported here so lib.globals does not load cvs.core at import time.
        from cvs.core.agent.logger import agent_logger

        agent_logger().debug(_sanitize_log_content(content), stacklevel=2)


def set_log_level(level):
    """
    Set the global CVS log level.

    Args:
        level: A logging level constant (e.g. logging.ERROR, logging.WARNING).

    Example:
        from cvs.lib.globals import set_log_level
        set_log_level(logging.ERROR)   # suppress SSH/pssh WARNING noise
        set_log_level(logging.DEBUG)   # enable full debug output
    """
    log.setLevel(level)
