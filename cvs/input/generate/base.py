#!/usr/bin/env python3
'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

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
    Dynamically discover all generator plugins in the generate/ directory.
    Returns a dict mapping generator names to plugin instances.
    """
    generators = {}

    # Get the directory containing this module (where generator plugins are located)
    generate_dir = os.path.dirname(__file__)

    if not os.path.exists(generate_dir):
        return generators

    for module_info in pkgutil.iter_modules([generate_dir]):
        try:
            # Import the module using the full package path
            module = importlib.import_module(f"cvs.input.generate.{module_info.name}")

            # Find classes that inherit from GeneratorPlugin
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, GeneratorPlugin)
                    and attr != GeneratorPlugin
                    and (not hasattr(attr, '__abstractmethods__') or not attr.__abstractmethods__)
                ):  # Allow if no abstract methods left
                    # Instantiate the plugin
                    plugin_instance = attr()
                    generators[plugin_instance.get_name()] = plugin_instance

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
