'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.core.schedulers.slurm import SlurmBackend


class SpurBackend(SlurmBackend):
    """A SPUR job step. SPUR mirrors each SPUR_* variable it sets to a SLURM_* twin,
    so every value reads the same way as on Slurm."""

    name = 'spur'
