'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from tabulate import tabulate

from .base import SubcommandPlugin
from cvs.lib.env_vars import ENV_VARS


def _display_current(ev):
    """Render the current value of an env var, masking secrets."""
    if not ev.is_set():
        return "(unset)"
    if ev.secret:
        raw = ev.raw() or ""
        return f"(set: {len(raw)} chars)"
    return ev.raw()


class EnvPlugin(SubcommandPlugin):
    def get_name(self):
        return "env"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("env", help="List supported CVS environment variables")
        parser.add_argument(
            "--set-only",
            action="store_true",
            help="Show only variables currently set in the environment",
        )
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Env Commands:
  cvs env              List all supported environment variables, defaults, and current values
  cvs env --set-only   Show only the variables currently set"""

    def run(self, args):
        env_vars = [ev for ev in ENV_VARS if ev.is_set() or not args.set_only]

        if not env_vars:
            print("No CVS environment variables are currently set.")
            return

        # Detailed, grouped-by-category view.
        categories = {}
        for ev in env_vars:
            categories.setdefault(ev.category, []).append(ev)

        print("\nSupported CVS environment variables")
        print("=" * 90)
        for category in sorted(categories):
            print(f"\n[{category}]")
            print("-" * 90)
            for ev in categories[category]:
                print(f"  {ev.name}")
                print(f"      default : {ev.default!r}")
                print(f"      current : {_display_current(ev)}")
                print(f"      {ev.description}")

        # Compact table at the end for quick scanning.
        rows = [[ev.name, ev.category, repr(ev.default), _display_current(ev)] for ev in env_vars]
        print("\nQuick view")
        print(tabulate(rows, headers=["Variable", "Category", "Default", "Current"], tablefmt="grid"))
        print()
