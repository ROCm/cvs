"""Portable artifact packing, also executed inside Aorta containers.

Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def pack_traces(root, archive, min_mtime, all_traces=True):
    """Pack fresh profiler files, preserving their paths relative to the repository."""
    root = Path(root)
    trees = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if name not in ("combined_traces", ".cvs-aorta", ".git")]
        relative = Path(directory).relative_to(root)
        if "torch_profiler" not in relative.parts:
            continue
        tree = Path(*relative.parts[: relative.parts.index("torch_profiler") + 1])
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file() or path.stat().st_mtime < min_mtime:
                continue
            trees.setdefault(tree, []).append(path)
    if trees and not all_traces:
        newest = max(trees, key=lambda tree: max(path.stat().st_mtime for path in trees[tree]))
        trees = {newest: trees[newest]}
    with tarfile.open(archive, "w:gz") as bundle:
        for files in trees.values():
            for path in files:
                bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    return [str(tree) for tree in sorted(trees)]


def pack_reports(root, archive, min_mtime):
    """Pack fresh analysis reports without following links outside the output tree."""
    root = Path(root)
    with tarfile.open(archive, "w:gz") as bundle:
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in files:
                path = Path(directory) / name
                if path.is_symlink() or not path.is_file() or path.stat().st_mtime < min_mtime:
                    continue
                bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)


def extract_artifacts(archive, destination):
    """Extract regular files only, rejecting escaping paths and archive links."""
    destination = Path(destination).resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError(f"Invalid artifact archive member: {member.name}")
            target = destination.joinpath(*relative.parts)
            if destination not in target.resolve().parents:
                raise ValueError(f"Artifact path escapes destination: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def main(argv=None):
    """Run the packing helper inside a container."""
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("traces", "reports"))
    parser.add_argument("root")
    parser.add_argument("archive")
    parser.add_argument("--min-mtime", type=float, default=0)
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args(argv)
    if args.kind == "traces":
        trees = pack_traces(args.root, args.archive, args.min_mtime, all_traces=not args.latest)
        print(json.dumps(trees))
    else:
        pack_reports(args.root, args.archive, args.min_mtime)


if __name__ == "__main__":
    main()
