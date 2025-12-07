#!/usr/bin/env python3
"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import sys
import os
import pkgutil
import importlib
from abc import ABC, abstractmethod


class MonitorPlugin(ABC):
    """Base class for all monitor plugins"""

    @abstractmethod
    def get_name(self):
        """Return the name of this monitor"""
        pass

    @abstractmethod
    def get_description(self):
        """Return a description of this monitor"""
        pass

    @abstractmethod
    def get_parser(self):
        """Return an argparse parser for this monitor's arguments"""
        pass

    @abstractmethod
    def monitor(self, args):
        """Execute the monitoring logic based on parsed arguments"""
        pass


def _discover_monitors():
    """
    Dynamically discover all monitor plugins in the monitors/ directory.
    Returns a dict mapping monitor names to plugin instances.
    """
    monitors = {}

    # Get the directory containing this module (where monitor plugins are located)
    monitors_dir = os.path.dirname(__file__)

    if not os.path.exists(monitors_dir):
        return monitors

    for module_info in pkgutil.iter_modules([monitors_dir]):
        try:
            # Import the module using the full package path
            module = importlib.import_module(f"cvs.monitors.{module_info.name}")

            # Find classes that inherit from MonitorPlugin
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, MonitorPlugin)
                    and attr != MonitorPlugin
                    and (not hasattr(attr, "__abstractmethods__") or not attr.__abstractmethods__)
                ):  # Allow if no abstract methods left
                    # Instantiate the plugin
                    plugin_instance = attr()
                    monitors[plugin_instance.get_name()] = plugin_instance

        except Exception as e:
            print(f"Warning: Failed to load monitor {module_info.name}: {e}")
            continue

    return monitors


def _run_monitor(monitor_name, args):
    """
    Run a monitor plugin with the provided arguments.
    """
    monitors = _discover_monitors()

    if monitor_name not in monitors:
        print(f"Error: Monitor '{monitor_name}' not found.")
        sys.exit(1)

    plugin = monitors[monitor_name]

    # Parse arguments using the plugin's parser
    parser = plugin.get_parser()
    # Set the program name to include the full command context
    parser.prog = f"cvs monitor {monitor_name}"
    try:
        parsed_args = parser.parse_args(args)
        # Call the plugin's monitor method
        plugin.monitor(parsed_args)
    except SystemExit as e:
        # argparse exits with SystemExit on help or error
        sys.exit(e.code)
