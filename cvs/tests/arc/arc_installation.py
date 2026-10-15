'''Install and validate ARC on the configured rack controllers.'''

from cvs.lib import arc_lib, globals
from cvs.lib.utils_lib import fail_test, print_test_output, update_test_result

log = globals.log


class TestArcInstallPreTasks:
    '''Controller connectivity and installation.'''

    def test_get_hostname(self, orch, cluster_dict, config_dict):
        '''Read hostnames from the configured controllers.'''
        globals.error_list = []
        controllers = arc_lib.resolve_controllers(cluster_dict, config_dict)
        out = orch.exec("hostname", hosts=controllers, timeout=30)
        print_test_output(log, out)
        for host in controllers:
            if not (out.get(host) or "").strip():
                fail_test(f"Empty hostname returned from controller {host}")
        update_test_result()

    def test_download_install_arc_on_controllers(self, orch, cluster_dict, config_dict):
        '''Install ARC only where it controls a rack.'''
        arc_lib.install_arc(orch, cluster_dict, config_dict)


class TestArcInstallCoreTasks:
    '''Placeholder for ARC execution tasks.'''

    pass


class TestArcInstallPostTasks:
    '''Validate the rack configuration with installed ARC.'''

    def test_validate_rack_config(self, orch, cluster_dict, config_dict):
        '''Check the rack JSON on every controller.'''
        arc_lib.validate_rack_config(orch, cluster_dict, config_dict)
