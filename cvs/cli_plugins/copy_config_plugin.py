import sys
import warnings

from .base import SubcommandPlugin
from . import config_files

COPY_CONFIG_LIST_DEPRECATION = "copy-config --list is deprecated; use 'cvs config list' or 'cvs config list-dirs'"


class CopyConfigPlugin(SubcommandPlugin):
    def get_name(self):
        return "copy-config"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser(
            "copy-config", help="List or copy config files from CVS package. Lists configs if --output not specified."
        )
        parser.add_argument(
            "path",
            nargs="?",
            help="Path to config file (e.g. configDir1/config1.json)",
        )
        parser.add_argument("--all", action="store_true", help="Copy all config files preserving directory structure")
        parser.add_argument("--output", help="Destination path to copy config file(s)")
        parser.add_argument(
            "--list",
            action="store_true",
            help="List available config files at the given path (lists all if no path specified)",
        )
        parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Copy-Config Commands:
  cvs copy-config                         List all available config files (deprecated: use cvs config list)
  cvs copy-config <configDir1>            List configs in a directory
  cvs copy-config --list                  Same as above (list all)
  cvs copy-config <configDir1> --list     Same as above (list directory)

  Note: prefer cvs config list / cvs config list-dirs for browsing templates

  cvs copy-config --all --output <destDir>              Copy all config files preserving directory structure
  cvs copy-config <configPath> --output <dest>          Copy a specific config file
  cvs copy-config --all --output <destDir> --force        Force overwrite existing files"""

    def _warn_list_deprecated(self):
        warnings.warn(COPY_CONFIG_LIST_DEPRECATION, DeprecationWarning, stacklevel=3)
        print(COPY_CONFIG_LIST_DEPRECATION, file=sys.stderr)

    def run(self, args):
        roots = config_files.find_config_roots()
        path = args.path or ""

        if args.all:
            config_files.copy_all_configs(roots, args.output, force=args.force)
            return

        if args.list or not args.output:
            self._warn_list_deprecated()
            config_files.print_flat_config_list(roots, path)
            return

        config_files.copy_single_config(roots, path, args.output, force=args.force)
