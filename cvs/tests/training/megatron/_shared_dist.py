'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Shared test steps for Megatron distributed training suites only.
Import via `from ._shared_dist import *` in each distributed test file.
'''

import re

from cvs.lib.utils_lib import *
from cvs.lib import globals

log = globals.log

__all__ = ['test_disable_firewall']


def test_disable_firewall(phdl):
    globals.error_list = []
    out_dict = phdl.exec('sudo service ufw status')
    for node in out_dict.keys():
        if not re.search('inactive', out_dict[node], re.I):
            phdl.exec('sudo service ufw stop')
            continue
    out_dict = phdl.exec('sudo ufw status')
    for node in out_dict.keys():
        if not re.search('inactive|disabled', out_dict[node], re.I):
            fail_test(f'Failed to disable firewall on node {node}')
    update_test_result()
