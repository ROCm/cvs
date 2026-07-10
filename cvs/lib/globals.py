'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import logging

log = logging.getLogger()

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
