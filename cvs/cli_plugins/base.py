import os
import sys
import importlib.resources as resources

class SubcommandPlugin:
    """Base class for CLI subcommand plugins."""
    def get_name(self):
        raise NotImplementedError

    def get_parser(self, subparsers):
        """Register subcommand with argparse subparsers."""
        raise NotImplementedError

    def get_epilog(self):
        """Return examples or help text for this subcommand. Default is empty."""
        return ""

    def run(self, args):
        """Run the subcommand logic."""
        raise NotImplementedError