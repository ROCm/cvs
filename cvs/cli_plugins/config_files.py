import os
import shutil

from cvs.extension import ExtensionConfig

CONFIG_EXTENSIONS = (".json", ".yaml", ".sh")
KNOWN_ROOT_NAMES = ("config_file", "cluster_file", "env_file")


def is_config_file(filename):
    return filename.endswith(CONFIG_EXTENSIONS)


def find_config_roots():
    """
    Find config directories from cvs and extension packages.

    Searches core cvs input dirs first, then extension packages via extension.ini.
    """
    plugin_dir = os.path.dirname(__file__)
    cvs_dir = os.path.dirname(plugin_dir)
    roots = []

    for name in KNOWN_ROOT_NAMES:
        root = os.path.join(cvs_dir, "input", name)
        if os.path.exists(root):
            roots.append(root)

    config = ExtensionConfig()
    for input_dir in config.get_input_dirs():
        for name in KNOWN_ROOT_NAMES:
            root = os.path.join(input_dir, name)
            if os.path.exists(root):
                roots.append(root)

    return roots


def root_name(root_path):
    return os.path.basename(root_path)


def parse_scope(path):
    """
    Parse optional root prefix from a user path.

    Returns (root_filter, subpath) where root_filter is None or one of KNOWN_ROOT_NAMES.
    """
    normalized = (path or "").strip("/")
    if not normalized:
        return None, ""

    parts = normalized.split("/", 1)
    if parts[0] in KNOWN_ROOT_NAMES:
        return parts[0], parts[1] if len(parts) > 1 else ""
    return None, normalized


def filter_roots(roots, root_filter):
    if root_filter is None:
        return roots
    return [root for root in roots if root_name(root) == root_filter]


def list_config_files(root, subpath):
    base = os.path.join(root, subpath) if subpath else root
    if not os.path.exists(base):
        return []

    result = []
    for dirpath, _dirs, files in os.walk(base):
        for filename in files:
            if is_config_file(filename):
                rel = os.path.relpath(os.path.join(dirpath, filename), root)
                result.append(rel)
    return sorted(result)


def find_config_file(roots, subpath):
    for root in roots:
        candidate = os.path.join(root, subpath)
        if os.path.isfile(candidate):
            return candidate
    return None


def group_files_by_dir(files):
    grouped = {}
    for filepath in files:
        parent = os.path.dirname(filepath)
        grouped.setdefault(parent, []).append(filepath)
    for parent in grouped:
        grouped[parent] = sorted(grouped[parent])
    return dict(sorted(grouped.items()))


def collect_dir_entries(files):
    """Return (dir_paths, root_level_files) relative to a config root."""
    dirs = set()
    root_files = []
    for filepath in files:
        parent = os.path.dirname(filepath)
        if not parent:
            root_files.append(filepath)
        else:
            dirs.add(parent)
    return sorted(dirs), sorted(root_files)


def group_dirs_by_first_segment(dir_paths):
    groups = {}
    for dir_path in dir_paths:
        first = dir_path.split("/", 1)[0]
        entry = f"{dir_path}/"
        groups.setdefault(first, []).append(entry)
    for first in groups:
        groups[first] = sorted(groups[first])
    return dict(sorted(groups.items()))


def copy_all_configs(roots, output_dir, force=False):
    if not output_dir:
        print("Error: --output required when using --all")
        return False

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        print(f"Error creating output directory {output_dir}: {exc}")
        return False

    copied_count = 0
    for root in roots:
        label = root_name(root)
        configs = list_config_files(root, "")
        for config in configs:
            src = os.path.join(root, config)
            dest = os.path.join(output_dir, label, config)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest) and not force:
                print(f"Error: File {dest} already exists. Use --force to overwrite.")
                return False
            try:
                shutil.copyfile(src, dest)
                copied_count += 1
            except OSError as exc:
                print(f"Error copying {src} to {dest}: {exc}")
                return False

    print(f"Copied {copied_count} config files to {output_dir}")
    return True


def copy_single_config(roots, subpath, output, force=False):
    if not subpath:
        print("Error: path to config file required for copying")
        return False

    config_file = find_config_file(roots, subpath)
    if not config_file:
        print(f"Config file not found: {subpath}")
        return False

    if os.path.isdir(output):
        dest = os.path.join(output, os.path.basename(config_file))
    else:
        dest = output

    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest) and not force:
        print(f"Error: File {dest} already exists. Use --force to overwrite.")
        return False

    try:
        shutil.copyfile(config_file, dest)
        print(f"Copied {config_file} to {dest}")
        return True
    except OSError as exc:
        print(f"Error copying {config_file} to {dest}: {exc}")
        return False


def print_flat_config_list(roots, subpath):
    found = False
    for root in roots:
        configs = list_config_files(root, subpath)
        if configs:
            display_path = os.path.join(root, subpath) if subpath else root
            print(f"Configs under {display_path}:")
            for config in configs:
                print(f"  {config}")
            found = True
    if not found:
        print("No config files found at the specified path.")
    return found
