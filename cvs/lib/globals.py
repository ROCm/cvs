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
logging.getLogger('pssh.host_logger').addFilter(lambda _record: False)

error_list = []


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
