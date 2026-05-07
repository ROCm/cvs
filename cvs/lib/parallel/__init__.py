'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Main exports for the parallel module
from cvs.lib.parallel.multiprocess_pssh import MultiProcessPssh
from cvs.lib.parallel.pssh import Pssh
from cvs.lib.parallel.scp import scp

# Backward compatibility aliases
PsshSharded = MultiProcessPssh  # Old name for multiprocess SSH

__all__ = [
    'Pssh',
    'MultiProcessPssh',
    'scp',
    # Backward compatibility
    'PsshSharded',
]
