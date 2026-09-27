#!/usr/bin/env python3
import argparse
import sys
import os
import importlib
import pkgutil
import importlib.metadata as metadata
from cvs.cli_plugins.base import SubcommandPlugin
from cvs.extension import ExtensionConfig
from cvs.lib.globals import set_verbosity

PLUGIN_DIR = os.path.join(os.path.dirname(__file__), "cli_plugins")


def get_version():
    """Get the version of cvs and any extension packages.

    Returns version information for:
    - cvs package (always)
    - extension package if configured via extension.ini or CVS_EXTENSION_PKG_NAMES

    Supports extension packages that override package name and version file location
    via extension.ini configuration.
    """
    config = ExtensionConfig()
    versions = []

    # Get cvs version
    try:
        cvs_version = metadata.version("cvs")
    except metadata.PackageNotFoundError:
        # Try version.txt in the package directory first (for installed packages)
        version_file = os.path.join(os.path.dirname(__file__), "version.txt")
        if not os.path.exists(version_file):
            # Fall back to parent directory (for development/source installations)
            version_file = os.path.join(os.path.dirname(__file__), "..", "version.txt")
        if os.path.exists(version_file):
            with open(version_file) as f:
                cvs_version = f.read().strip()
        else:
            cvs_version = "unknown"
    versions.append(f"cvs: {cvs_version}")

    # Get extension version if configured
    extension_pkg_name = config.get_package_name()
    if extension_pkg_name != "cvs":
        try:
            ext_version = metadata.version(extension_pkg_name)
        except metadata.PackageNotFoundError:
            ext_version = "unknown"
        versions.append(f"{extension_pkg_name}: {ext_version}")

    return "\n".join(versions)


def discover_plugins():
    """Discover and instantiate all CLI subcommand plugin classes from the cli_plugins directory.

    This function scans the cli_plugins directory for Python modules, imports them,
    and looks for classes that are subclasses of SubcommandPlugin. It instantiates
    each valid plugin class and returns a list of plugin instances.

    Only classes defined directly in the module (not imported) are considered to
    avoid duplicates from relative imports.

    Returns:
        list: A list of instantiated plugin objects, sorted by order then alphabetically by name.
    """
    plugins = []
    for _, name, ispkg in pkgutil.iter_modules([PLUGIN_DIR]):
        if not ispkg:
            try:
                mod = importlib.import_module(f"cvs.cli_plugins.{name}")
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    try:
                        # Check if obj is a plugin class: must be a class, subclass of SubcommandPlugin,
                        # not the base class itself, and defined in this module (not imported)
                        if (
                            isinstance(obj, type)
                            and issubclass(obj, SubcommandPlugin)
                            and obj is not SubcommandPlugin
                            and obj.__module__ == mod.__name__
                        ):
                            plugins.append(obj())
                    except Exception:
                        continue
            except Exception as e:
                print(f"Warning: Failed to load plugin {name}: {e}")

    # Sort plugins by order first, then by name
    return sorted(plugins, key=lambda p: (p.get_order(), p.get_name()))


def build_arg_parser(plugins):
    """Build the main argument parser for the CVS CLI.

    This function creates an ArgumentParser with subparsers for each subcommand plugin.
    It collects epilog text (examples/help) from all plugins and concatenates them
    into the main parser's epilog.

    Args:
        plugins (list): List of instantiated plugin objects.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    # Collect epilogs from all plugins
    epilogs = [plugin.get_epilog() for plugin in plugins if plugin.get_epilog().strip()]
    epilog = "\n".join(epilogs) if epilogs else ""

    parser = argparse.ArgumentParser(
        description="Cluster Validation Suite (CVS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("--version", action="version", version=get_version())
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="cli_verbose",
        help="Increase CVS verbosity (-v, -vv, -vvv).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    for plugin in plugins:
        plugin.get_parser(subparsers)
    return parser


def _count_v_flags(args):
    """Return the -v count in this slice of argv."""
    total = 0
    for arg in args:
        if arg == "--verbose":
            total += 1
        elif arg.startswith("-") and not arg.startswith("--") and set(arg[1:]) == {"v"}:
            total += len(arg) - 1
    return total


def _cli_verbosity(argv=None):
    """Count -v / -vv / -vvv / --verbose for this CVS command.

    ``cvs -v exec -v`` is 1, not 2.
    Repeated flags on one side still add (``-v -v`` and ``-vv`` are both 2).
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    for index, arg in enumerate(tokens):
        if arg == "--":
            return _count_v_flags(tokens[:index])
        if not arg.startswith("-"):
            return max(_count_v_flags(tokens[:index]), _count_v_flags(tokens[index + 1 :]))
    return _count_v_flags(tokens)


def main(plugins=None):
    if plugins is None:
        plugins = discover_plugins()
    parser = build_arg_parser(plugins)
    args, extra_pytest_args = parser.parse_known_args()
    args.extra_pytest_args = extra_pytest_args
    set_verbosity(_cli_verbosity())

    # Dispatch to plugin
    if hasattr(args, "_plugin"):
        args._plugin.run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
