#!/usr/bin/env python3
'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Validates exe_path entries from ANC content YAML configs.

Usage:
    python3 validate_exe_paths.py <content_dir>

Scans <content_dir>/*/exe/ for .yml/.yaml files, reads the 'exe_path' key
from each, and checks that the directory exists on the local filesystem.

Exit codes:
    0 - All exe_path entries validated successfully
    1 - One or more exe_path directories are missing
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


def validate(content_dir):
    """Validate all exe_path entries under content_dir/*/exe/*.yml|yaml."""
    exe_dirs = glob.glob(os.path.join(content_dir, "*/exe"))
    if not exe_dirs:
        print(f"ERROR: No exe directories found under {content_dir}")
        return 2

    missing = []
    checked = 0

    for exe_dir in sorted(exe_dirs):
        yml_files = sorted(glob.glob(os.path.join(exe_dir, "*.yml")) + glob.glob(os.path.join(exe_dir, "*.yaml")))
        for yml_file in yml_files:
            with open(yml_file) as f:
                data = yaml.safe_load(f)
            if data and "exe_path" in data:
                exe_path = data["exe_path"]
                checked += 1
                basename = os.path.basename(yml_file)
                if os.path.isdir(exe_path):
                    print(f"OK: {exe_path} (from {basename})")
                else:
                    missing.append(f"{exe_path} (from {yml_file})")
                    print(f"MISSING: {exe_path} (from {basename})")

    print(f"--- Checked {checked} exe_path entries, {len(missing)} missing ---")

    if missing:
        print("VALIDATION_FAILED")
        for m in missing:
            print(f"  - {m}")
        return 1

    print("VALIDATION_SUCCESS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <content_dir>")
        sys.exit(2)
    sys.exit(validate(sys.argv[1]))
