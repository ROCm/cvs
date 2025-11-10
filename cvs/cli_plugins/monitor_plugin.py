from .base import SubcommandPlugin
import argparse

class MonitorPlugin(SubcommandPlugin):
    def get_name(self):
        return "monitor"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser('monitor', help='Run cluster monitoring scripts')
        parser.add_argument('monitor', nargs='?', help='Name of the monitor to use')
        parser.add_argument('monitor_args', nargs=argparse.REMAINDER, help='Arguments for the monitor')
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Monitor Commands:
  cvs monitor                                          List all available monitors
  cvs monitor check_cluster_health --help              Show help for check_cluster_health monitor"""

    def run(self, args):
        from cvs.monitors.base import _discover_monitors, _run_monitor
        monitors = _discover_monitors()
        if args.monitor is None:
            if monitors:
                print("Available monitors:")
                for name, plugin in sorted(monitors.items()):
                    print(f"  {name} - {plugin.get_description()}")
            else:
                print("No monitors found in cvs/monitors/ directory.")
        else:
            if getattr(args, 'extra_pytest_args', None):
                args.monitor_args.extend(args.extra_pytest_args)
                args.extra_pytest_args = []
            if args.monitor_args and args.monitor_args[0] in ['-h', '--help']:
                _run_monitor(args.monitor, ['-h'])
            else:
                _run_monitor(args.monitor, args.monitor_args)
