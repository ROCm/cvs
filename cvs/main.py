#!/usr/bin/env python3
import argparse
import sys
import os
import importlib
import pkgutil
import importlib.resources as resources
import importlib.metadata as metadata
import pytest

PLUGIN_DIR = os.path.join(os.path.dirname(__file__), 'cli_plugins')


def get_version():
    """Get the version from importlib.metadata or fallback to version.txt file."""
    try:
        return metadata.version('cvs')
    except metadata.PackageNotFoundError:
        # Fallback for development
        version_file = os.path.join(os.path.dirname(__file__), '..', 'version.txt')
        if os.path.exists(version_file):
            with open(version_file) as f:
                return f.read().strip()
    return 'unknown'


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
    from cvs.cli_plugins.base import SubcommandPlugin
    for _, name, ispkg in pkgutil.iter_modules([PLUGIN_DIR]):
        if not ispkg:
            try:
                mod = importlib.import_module(f'cvs.cli_plugins.{name}')
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    try:
                        # Check if obj is a plugin class: must be a class, subclass of SubcommandPlugin,
                        # not the base class itself, and defined in this module (not imported)
                        if isinstance(obj, type) and issubclass(obj, SubcommandPlugin) and obj is not SubcommandPlugin and obj.__module__ == mod.__name__:
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
        epilog=epilog
    )
    parser.add_argument('--version', action='version', version=get_version())
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    for plugin in plugins:
        plugin.get_parser(subparsers)
    return parser

def main():
    plugins = discover_plugins()
    parser = build_arg_parser(plugins)
    args, extra_pytest_args = parser.parse_known_args()
    args.extra_pytest_args = extra_pytest_args

    # Dispatch to plugin
    if hasattr(args, '_plugin'):
        args._plugin.run(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
