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


class DebugPlugin(ABC):
    """Base class for all debug plugins"""

    @abstractmethod
    def get_name(self):
        """Return the name of this debugger"""
        pass

    @abstractmethod
    def get_description(self):
        """Return a description of this debugger"""
        pass

    @abstractmethod
    def get_parser(self):
        """Return an argparse parser for this debugger's arguments"""
        pass

    @abstractmethod
    def debug(self, args):
        """Execute the debugging logic based on parsed arguments"""
        pass


def _discover_debuggers():
    """
    Dynamically discover all debug plugins in the debug/ directory.
    Returns a dict mapping debugger names to plugin instances.
    """
    debuggers = {}

    debuggers_dir = os.path.dirname(__file__)

    if not os.path.exists(debuggers_dir):
        return debuggers

    for module_info in pkgutil.iter_modules([debuggers_dir]):
        try:
            module = importlib.import_module(f"cvs.debuggers.{module_info.name}")

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, DebugPlugin)
                    and attr != DebugPlugin
                    and (not hasattr(attr, "__abstractmethods__") or not attr.__abstractmethods__)
                ):
                    plugin_instance = attr()
                    debuggers[plugin_instance.get_name()] = plugin_instance

        except Exception as e:
            print(f"Warning: Failed to load debugger {module_info.name}: {e}")
            continue

    return debuggers


def _run_debugger(debugger_name, args):
    """
    Run a debugger plugin with the provided arguments.
    """
    debuggers = _discover_debuggers()

    if debugger_name not in debuggers:
        print(f"Error: Debugger '{debugger_name}' not found.")
        sys.exit(1)

    plugin = debuggers[debugger_name]

    parser = plugin.get_parser()
    parser.prog = f"cvs debug {debugger_name}"
    try:
        parsed_args = parser.parse_args(args)
        plugin.debug(parsed_args)
    except SystemExit as e:
        sys.exit(e.code)
