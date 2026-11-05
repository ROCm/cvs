#!/usr/bin/env python3
'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Validates exe_path entries from ANC content YAML configs.

Usage:
    python3 validate_exe_paths.py <content_dir>

Scans <content_dir>/*/exe/ for .yml/.yaml files, reads the 'exe_path' key
from each, and checks that the directory exists on the local filesystem.
exe_path may be one directory or a list of alternate directories; a list
passes when at least one of those directories exists.

Exit codes:
    0 - All exe_path entries validated successfully
    1 - One or more exe_path directories are missing, or a value is not a path
    2 - No exe directories found or import error
'''

import os
import sys
import glob

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed on this node")
    sys.exit(2)


def _exe_dirs(exe_path):
    '''Return (directories, error). error is set when exe_path is not usable.

    A string is one directory. A list is alternate directories for the same
    tool, so any one existing directory satisfies the check.
    '''
    if isinstance(exe_path, str):
        if not exe_path:
            return None, "exe_path is an empty string"
        return [exe_path], None
    if isinstance(exe_path, list):
        if not exe_path:
            return None, "exe_path is an empty list"
        for index, item in enumerate(exe_path):
            if not isinstance(item, str) or not item:
                return None, (f"exe_path[{index}] is {type(item).__name__}, expected a path string: {item!r}")
        return list(exe_path), None
    return None, (f"exe_path is {type(exe_path).__name__}, expected a path string or list of paths: {exe_path!r}")


def validate(content_dir):
    """Validate all exe_path entries under content_dir/*/exe/*.yml|yaml."""
    exe_dirs = glob.glob(os.path.join(content_dir, "*/exe"))
    if not exe_dirs:
        print(f"ERROR: No exe directories found under {content_dir}")
        return 2

    missing = []
    invalid = []
    checked = 0

    for exe_dir in sorted(exe_dirs):
        yml_files = sorted(glob.glob(os.path.join(exe_dir, "*.yml")) + glob.glob(os.path.join(exe_dir, "*.yaml")))
        for yml_file in yml_files:
            with open(yml_file) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict) or "exe_path" not in data:
                continue
            checked += 1
            basename = os.path.basename(yml_file)
            entries, error = _exe_dirs(data["exe_path"])
            if error:
                invalid.append(f"{error} (from {yml_file})")
                print(f"INVALID: {error} (from {basename})")
                continue
            found = [path for path in entries if os.path.isdir(path)]
            if found:
                print(f"OK: {', '.join(found)} (from {basename})")
            else:
                label = ", ".join(entries)
                missing.append(f"{label} (from {yml_file})")
                print(f"MISSING: {label} (from {basename})")

    summary = f"--- Checked {checked} exe_path entries, {len(missing)} missing"
    if invalid:
        summary += f", {len(invalid)} invalid"
    print(summary + " ---")

    if missing or invalid:
        print("VALIDATION_FAILED")
        for problem in missing + invalid:
            print(f"  - {problem}")
        return 1

    print("VALIDATION_SUCCESS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <content_dir>")
        sys.exit(2)
    sys.exit(validate(sys.argv[1]))
