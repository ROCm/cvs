'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Backward-compatible module: re-exports parallel SSH types and helpers from cvs.lib.parallel
so existing ``from cvs.lib.parallel_ssh_lib import Pssh`` (and star imports) keep working.

Implementation lives in:
  - cvs.lib.parallel.base — PsshShard
  - cvs.lib.parallel.ssh — Pssh
  - cvs.lib.parallel.scp — standalone scp helper
'''

from pssh.clients import ParallelSSHClient
from pssh.exceptions import ConnectionError, SessionError, Timeout

from cvs.lib.parallel import Pssh, PsshShard, scp

__all__ = [
    'Pssh',
    'PsshShard',
    'scp',
    'ParallelSSHClient',
    'Timeout',
    'ConnectionError',
    'SessionError',
]
