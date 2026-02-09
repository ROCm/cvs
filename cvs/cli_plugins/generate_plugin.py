from .base import SubcommandPlugin
import argparse
import sys
import os
import pkgutil
import importlib
from abc import ABC, abstractmethod


class GeneratorPlugin(ABC):
    """Base class for all generator plugins"""

    @abstractmethod
    def get_name(self):
        """Return the name of this generator"""
        pass

    @abstractmethod
    def get_description(self):
        """Return a description of this generator"""
        pass

    @abstractmethod
    def get_parser(self):
        """Return an argparse parser for this generator's arguments"""
        pass

    @abstractmethod
    def generate(self, args):
        """Generate the output based on parsed arguments"""
        pass


def _discover_generators():
    """
    Dynamically discover all generator plugins from multiple directories.
    Searches in:
      - cvs/input/generate/ (input/config generators)
      - cvs/reports/generate/ (report generators)
    Returns a dict mapping generator names to plugin instances.
    """
    generators = {}

    # Define directories to scan for generators
    # Use tuples of (directory_path, package_name)
    base_dir = os.path.dirname(os.path.dirname(__file__))

    search_paths = [
        (os.path.join(base_dir, "input", "generate"), "cvs.input.generate"),
        (os.path.join(base_dir, "reports", "generate"), "cvs.reports.generate"),
    ]

    for search_dir, package_name in search_paths:
        if not os.path.exists(search_dir):
            continue

        for module_info in pkgutil.iter_modules([search_dir]):
            try:
                # Import the module using the full package path
                module = importlib.import_module(f"{package_name}.{module_info.name}")

                # Find classes that inherit from GeneratorPlugin
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, GeneratorPlugin)
                        and attr != GeneratorPlugin
                        and (not hasattr(attr, '__abstractmethods__') or not attr.__abstractmethods__)
                    ):
                        # Instantiate the plugin
                        plugin_instance = attr()
                        generator_name = plugin_instance.get_name()

                        # Avoid duplicates - first one wins
                        if generator_name not in generators:
                            generators[generator_name] = plugin_instance

            except Exception as e:
                print(f"Warning: Failed to load generator {module_info.name}: {e}")
                continue

    return generators


def _run_generator(generator_name, args):
    """
    Run a generator plugin with the provided arguments.
    """
    generators = _discover_generators()

    if generator_name not in generators:
        print(f"Error: Generator '{generator_name}' not found.")
        sys.exit(1)

    plugin = generators[generator_name]

    # Parse arguments using the plugin's parser
    parser = plugin.get_parser()
    # Set the program name to include the full command context
    parser.prog = f"cvs generate {generator_name}"
    try:
        parsed_args = parser.parse_args(args)
        # Call the plugin's generate method
        plugin.generate(parsed_args)
    except SystemExit as e:
        # argparse exits with SystemExit on help or error
        sys.exit(e.code)


class GeneratePlugin(SubcommandPlugin):
    def get_name(self):
        return "generate"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("generate", help="Generate configuration files or templates")
        parser.add_argument("generator", nargs="?", help="Name of the generator to use")
        parser.add_argument("generator_args", nargs=argparse.REMAINDER, help="Arguments for the generator")
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Generate Commands:
  cvs generate                       List available generators
  cvs generate cluster_json --help   Show help for cluster_json generator
  cvs generate heatmap --help        Show help for heatmap generator"""

    def run(self, args):
        generators = _discover_generators()
        if args.generator is None:
            if generators:
                print("Available generators:")
                for name, plugin in sorted(generators.items()):
                    print(f"  {name} - {plugin.get_description()}")
            else:
                print("No generators found in cvs/generate/ directory.")
        else:
            if getattr(args, "extra_pytest_args", None):
                args.generator_args.extend(args.extra_pytest_args)
                args.extra_pytest_args = []
            if args.generator_args and args.generator_args[0] in ["-h", "--help"]:
                _run_generator(args.generator, ["-h"])
            else:
                _run_generator(args.generator, args.generator_args)
