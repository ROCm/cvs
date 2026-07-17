'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

'''
ANC "run all CPU groups" suite.

Installs/verifies ANC and fixes ROCm ldconfig ONCE as pre-tasks, then runs
every CPU group (anc_lib.CPU_GROUPS) sequentially, each as its own parametrized
test so its logs are captured separately under
``{runner_log_folder}/anc_logs/<node>/test_<group>/<timestamp>``.

For a single group, run its standalone suite instead, e.g.
``cvs run anc_test_cpu_all``.

Fixtures come from cvs/tests/anc/conftest.py; all logic lives in
cvs/lib/anc_lib.py.
'''

import pytest

from cvs.lib import globals
from cvs.lib import anc_lib

log = globals.log


class TestAncExecAllCpuPreTasks:
    '''Pre-tasks: install ANC and ensure ROCm libs, once for the whole run.'''

    def test_install_anc(self, phdl, config_dict):
        '''Install/verify ANC once before running all CPU groups.'''
        log.info("ANC exec-all CPU Pre-Task: install/verify ANC")
        anc_lib.install_anc(phdl, config_dict)

    def test_rocm_ldconfig(self, phdl):
        '''Ensure ROCm libs resolvable once before running all CPU groups.'''
        log.info("ANC exec-all CPU Pre-Task: ensure ROCm ldconfig")
        anc_lib.ensure_rocm_ldconfig(phdl)


class TestAncExecAllCpuCoreTasks:
    '''Run every CPU group sequentially, one test (and log dir) per group.'''

    @pytest.mark.parametrize("group", anc_lib.CPU_GROUPS)
    def test_cpu_group(self, phdl, cluster_dict, config_dict, group, request):
        '''Run one CPU group and collect/judge its logs independently.'''
        log.info("ANC exec-all CPU: running group '%s'", group)
        anc_lib.run_anc_groups(
            phdl, cluster_dict, config_dict, [group], f"test_{group}",
            request=request,
        )
