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

    def get_order(self):
        """Return the display order for this plugin. Lower numbers appear first. Default is 0."""
        return 0

    def run(self, args):
        """Run the subcommand logic."""
        raise NotImplementedError
