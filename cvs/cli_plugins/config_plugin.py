import argparse
import sys

from cvs.config_validation.migrate import maps

from .base import SubcommandPlugin
from . import config_files

CONFIG_COMMANDS = (
    ("list", "List config files grouped by directory"),
    ("list-dirs", "List config directories grouped by category"),
    ("copy", "Copy bundled config file(s) to --output"),
    ("migrate", "Convert old suite config or threshold files to the new format"),
)

MIGRATE_EXIT_CODES = (
    "exit status, the highest over the input files: 0 when the new file is written, or the input is already in "
    "the new format; 1 when the new file is written with fields to fix; 2 when nothing is written"
)


def _known_suite(suites):
    """An argparse ``type`` that accepts only a name in ``suites`` and lists them otherwise."""

    def check(name):
        if name not in suites:
            raise argparse.ArgumentTypeError(f"unknown suite {name}; available suites: {', '.join(suites) or 'none'}")
        return name

    return check


class ConfigPlugin(SubcommandPlugin):
    def get_name(self):
        return "config"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser(
            "config",
            help="Browse, copy or migrate CVS config files",
        )
        config_subparsers = parser.add_subparsers(dest="config_command")

        config_subparsers.add_parser(
            "list",
            help=CONFIG_COMMANDS[0][1],
        ).add_argument(
            "path",
            nargs="?",
            default="",
            help="Optional path scope (e.g. configDir1 or configDir1/configDir2)",
        )

        config_subparsers.add_parser(
            "list-dirs",
            help=CONFIG_COMMANDS[1][1],
        ).add_argument(
            "path",
            nargs="?",
            default="",
            help="Optional path scope (e.g. configDir1 or rootName/configDir1)",
        )

        copy_parser = config_subparsers.add_parser(
            "copy",
            help=CONFIG_COMMANDS[2][1],
        )
        copy_parser.add_argument(
            "path",
            nargs="?",
            help="Path to config file (required unless --all)",
        )
        copy_parser.add_argument(
            "--all",
            action="store_true",
            help="Copy all config files preserving directory structure",
        )
        copy_parser.add_argument(
            "--output",
            required=True,
            help="Destination path to copy config file(s)",
        )
        copy_parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
        copy_parser.prog = "cvs config copy"

        suites = maps.suite_names()
        migrate_parser = config_subparsers.add_parser(
            "migrate",
            help=CONFIG_COMMANDS[3][1],
            description=f"{CONFIG_COMMANDS[3][1]}. The input files are never changed.",
            epilog=MIGRATE_EXIT_CODES,
        )
        migrate_parser.add_argument(
            "suite",
            type=_known_suite(suites),
            help=f"Suite the file belongs to (available: {', '.join(suites) or 'none yet'})",
        )
        migrate_parser.add_argument("inputs", metavar="input.json", nargs="+", help="Old config or threshold file")
        migrate_parser.add_argument(
            "output",
            help="File to write, or a directory to write it in under the input's file name; "
            "with several input files, an existing directory",
        )
        migrate_parser.add_argument("--force", action="store_true", help="Force overwrite of an existing output file")
        migrate_parser.prog = "cvs config migrate"

        self._copy_parser = copy_parser
        self._migrate_parser = migrate_parser
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Config Commands:
  cvs config                                           List available config commands
  cvs config list-dirs                                 List config directories by category
  cvs config list-dirs <configDir1>                    List directories under a category
  cvs config list                                      List all config files grouped by directory
  cvs config list <configDir1>/<configDir2>            List config files under a directory
  cvs config copy --all --output <destDir>             Copy all bundled configs
  cvs config copy <configPath> --output <dest>         Copy a specific config file
  cvs config copy --all --output <destDir> --force     Copy all bundled configs, overwrite existing
  cvs config migrate <suite> <input.json> <output>     Convert an old config or threshold file
  cvs config migrate <suite> <input.json>... <dir>     Convert several files into one directory"""

    def _print_command_catalog(self):
        print("Available config commands:")
        for name, description in CONFIG_COMMANDS:
            print(f"  {name} - {description}")

    def run(self, args):
        command = getattr(args, "config_command", None)
        if command == "list":
            self._run_list(args.path or "")
        elif command == "list-dirs":
            self._run_list_dirs(args.path or "")
        elif command == "copy":
            self._run_copy(args)
        elif command == "migrate":
            self._run_migrate(args)
        else:
            self._print_command_catalog()

    def _scoped_roots(self, path):
        root_filter, subpath = config_files.parse_scope(path)
        roots = config_files.filter_roots(config_files.find_config_roots(), root_filter)
        return roots, subpath, root_filter

    def _is_scoped(self, path, root_filter):
        return bool(path) and root_filter is None

    def _run_list_dirs(self, path):
        roots, subpath, root_filter = self._scoped_roots(path)
        scoped = self._is_scoped(path, root_filter)
        found = False
        root_results = []

        for root in roots:
            files = config_files.list_config_files(root, subpath)
            if not files:
                continue

            found = True
            label = config_files.root_name(root)
            dir_paths, root_files = config_files.collect_dir_entries(files)
            root_results.append((label, dir_paths, root_files))

        if not found:
            print("No config directories found at the specified path.")
            return

        show_root_headers = not scoped or len(root_results) > 1

        for root_idx, (label, dir_paths, root_files) in enumerate(root_results):
            if root_idx > 0:
                print()

            if show_root_headers:
                print(f"{label}_dirs:")

            if dir_paths:
                if scoped:
                    for dir_path in dir_paths:
                        prefix = "  " if show_root_headers else ""
                        print(f"{prefix}{dir_path}/")
                else:
                    groups = config_files.group_dirs_by_first_segment(dir_paths)
                    for group_idx, (group_name, entries) in enumerate(groups.items()):
                        if group_idx > 0:
                            print()
                        print(f"  {group_name}:")
                        for entry in entries:
                            print(f"    {entry}")

            if root_files:
                prefix = "  " if show_root_headers or not scoped else ""
                for entry in root_files:
                    print(f"{prefix}{entry}")

    def _run_list(self, path):
        roots, subpath, root_filter = self._scoped_roots(path)
        scoped = self._is_scoped(path, root_filter)
        root_results = []

        for root in roots:
            files = config_files.list_config_files(root, subpath)
            if not files:
                continue

            label = config_files.root_name(root)
            grouped = config_files.group_files_by_dir(files)
            root_results.append((label, grouped))

        if not root_results:
            print("No config files found at the specified path.")
            return

        show_root_headers = not scoped or len(root_results) > 1

        for root_idx, (label, grouped) in enumerate(root_results):
            if root_idx > 0:
                print()

            if show_root_headers:
                print(f"{label}:")

            for group_idx, (parent, filepaths) in enumerate(grouped.items()):
                if group_idx > 0:
                    print()

                if not parent:
                    for filepath in filepaths:
                        prefix = "  " if show_root_headers or not scoped else ""
                        print(f"{prefix}{filepath}")
                else:
                    prefix = "  " if show_root_headers or not scoped else ""
                    print(f"{prefix}{parent}:")
                    for filepath in filepaths:
                        inner = "    " if show_root_headers or not scoped else "  "
                        print(f"{inner}{filepath}")

    def _run_copy(self, args):
        path = args.path or ""

        if not args.all and not path:
            self._copy_parser.print_help()
            sys.exit(2)

        roots = config_files.find_config_roots()

        if args.all:
            config_files.copy_all_configs(roots, args.output, force=args.force)
            return

        config_files.copy_single_config(roots, path, args.output, force=args.force)

    def _run_migrate(self, args):
        if args.extra_pytest_args:
            self._migrate_parser.error(f"unrecognized arguments: {' '.join(args.extra_pytest_args)}")
        # Imported here so the engine and the suite's map load only for this command.
        from cvs.config_validation.migrate import engine

        sys.exit(engine.migrate_many(args.suite, args.inputs, args.output, force=args.force))
