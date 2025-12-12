from .base import SubcommandPlugin
import argparse
from cvs.debuggers.base import _discover_debuggers, _run_debugger


class DebugPlugin(SubcommandPlugin):
    def get_name(self):
        return "debug"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("debug", help="Run cluster debugging tools")
        parser.add_argument("debugger", nargs="?", help="Name of the debugger to use")
        parser.add_argument("debugger_args", nargs=argparse.REMAINDER, help="Arguments for the debugger")
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Debug Commands:
  cvs debug                                            List all available debuggers
  cvs debug gdb_backtrace_collector --help             Show help for gdb_backtrace_collector debugger"""

    def run(self, args):
        debuggers = _discover_debuggers()
        if args.debugger is None:
            if debuggers:
                print("Available debuggers:")
                for name, plugin in sorted(debuggers.items()):
                    print(f"  {name} - {plugin.get_description()}")
            else:
                print("No debuggers found in cvs/debuggers/ directory.")
        else:
            if getattr(args, "extra_pytest_args", None):
                args.debugger_args.extend(args.extra_pytest_args)
                args.extra_pytest_args = []
            if args.debugger_args and args.debugger_args[0] in ["-h", "--help"]:
                _run_debugger(args.debugger, ["-h"])
            else:
                _run_debugger(args.debugger, args.debugger_args)
