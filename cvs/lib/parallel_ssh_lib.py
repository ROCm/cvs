'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

DEPRECATED: This module is provided for backward compatibility only.

For new code, please import directly from the parallel package:

    # Recommended (multi-process with auto-sharding):
    from cvs.lib.parallel.multiprocess_pssh import MultiProcessPssh

    # For single-process SSH only:
    from cvs.lib.parallel.pssh import Pssh

    # Configuration:
    from cvs.lib.parallel.config import ParallelConfig

This module re-exports parallel SSH types and helpers from cvs.lib.parallel
so existing ``from cvs.lib.parallel_ssh_lib import Pssh`` imports continue working.

Implementation lives in:
  - cvs.lib.parallel.pssh — Pssh (basic single-process)
  - cvs.lib.parallel.multiprocess_pssh — MultiProcessPssh (multi-process sharded)
  - cvs.lib.parallel.scp — standalone scp helper
'''

from pssh.clients import ParallelSSHClient
from pssh.exceptions import ConnectionError, SessionError, Timeout

# Direct imports from parallel package modules
from cvs.lib.parallel.multiprocess_pssh import MultiProcessPssh as Pssh  # Main interface (auto-sharding)
from cvs.lib.parallel.config import ParallelConfig
from cvs.lib.parallel.scp import scp

__all__ = [
    'Pssh',
    'scp',
    'ParallelConfig',
    'ParallelSSHClient',
    'Timeout',
    'ConnectionError',
    'SessionError',
]
