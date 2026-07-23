'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

ANC installation suite: download + install ANC on every node.

Fixtures (cluster_dict / config_dict / phdl) are provided by the dir-local
conftest.py. All install logic lives in cvs.lib.anc_lib so it can be
reused by the anc_test_cpu / anc_test_gpu suites (which install ANC as a
pre-task before running their group sets).
'''

from cvs.lib.utils_lib import fail_test, update_test_result, print_test_output
from cvs.lib import globals
from cvs.lib import anc_lib

log = globals.log

# Re-exported for callers/tests that import the ANC entrypoint from here.
ANC_BIN = anc_lib.ANC_BIN
detect_package_type = anc_lib.detect_package_type


class TestAncInstallPreTasks:
    '''Pre-tasks: validate connectivity and gather host information.'''

    def test_get_hostname(self, phdl, cluster_dict):
        '''
        Retrieve hostname from all remote nodes; fail on any empty response.
        '''
        globals.error_list = []
        log.info("ANC Pre-Task: Getting hostname from remote nodes")

        out_dict = phdl.exec("hostname", timeout=30)
        print_test_output(log, out_dict)

        for host, output in out_dict.items():
            hostname = output.strip()
            if not hostname:
                fail_test(f"Empty hostname returned from node {host}")
            else:
                log.info("Node %s reports hostname: %s", host, hostname)

        update_test_result()

    def test_download_install_anc_in_node_cvs_home(self, phdl, cluster_dict, config_dict):
        '''
        Install ANC on all nodes, dispatching by release-archive flavour
        (deb/rpm/tar) with an optional anc_version precheck / post-verify.

        See cvs.lib.anc_lib.install_anc for the full behaviour.
        '''
        anc_lib.install_anc(phdl, cluster_dict, config_dict)


class TestAncInstallCoreTasks:
    '''Core ANC execution tasks (placeholder for ANC workload).'''

    pass


class TestAncInstallPostTasks:
    '''Post-tasks: cleanup and result collection (placeholder).'''

    pass
