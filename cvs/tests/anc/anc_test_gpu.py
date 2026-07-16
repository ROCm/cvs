'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

'''
ANC GPU validation suite.

Pre-task: install ANC (dispatched by release-archive flavour) at the target
anc_version. Core task: run the full GPU group set in a single
``anc.py -g <groups...>`` invocation on all nodes and collect artifacts.

Fixtures (cluster_dict / config_dict / phdl) come from the dir-local
conftest.py; all logic lives in cvs.lib.anc_lib.

Per-node artifacts are downloaded to:
    <runner_log_folder>/anc/<ip>_<hostname>/test_gpu/
(runner_log_folder defaults to /tmp/cvs_results).
'''

from cvs.lib import globals
from cvs.lib import anc_lib

log = globals.log


class TestAncGpuPreTasks:
    '''Pre-tasks: ensure ANC is installed at the target version.'''

    def test_install_anc(self, phdl, config_dict):
        '''
        Install ANC on all nodes before running the GPU groups.

        Skips the install when every node already reports config anc_version;
        otherwise installs from the release archive and verifies the version.
        '''
        log.info("ANC GPU Pre-Task: install/verify ANC")
        anc_lib.install_anc(phdl, config_dict)

    def test_rocm_ldconfig(self, phdl):
        '''
        Ensure ROCm shared libraries are resolvable before running ANC.

        Some ANC tools fail to load librocblas even when the .so is on disk
        because the ROCm lib dirs are not in the dynamic-linker cache. This
        registers those dirs and runs ldconfig, or fails if the lib is
        genuinely missing.
        '''
        log.info("ANC GPU Pre-Task: ensure ROCm ldconfig")
        anc_lib.ensure_rocm_ldconfig(phdl)


class TestAncGpuCoreTasks:
    '''Core ANC GPU validation.'''

    def test_gpu(self, phdl, cluster_dict, config_dict):
        '''
        Run the ANC GPU validation groups on all nodes.

        Executes ``sudo ./anc.py -g <GPU_GROUPS>`` under the ANC install dir
        and collects journal.log/console.log (mandatory) plus summary.json/
        errors.json/system_monitor.json (when present). PASS only when every
        node reports a final ANC_SUCCESS [0] return code.
        '''
        log.info("ANC GPU Core Task: groups=%s", anc_lib.GPU_GROUPS)
        anc_lib.run_anc_groups(
            phdl, cluster_dict, config_dict, anc_lib.GPU_GROUPS, "test_gpu"
        )


class TestAncGpuPostTasks:
    '''Post-tasks: cleanup and result collection (placeholder).'''
    pass
