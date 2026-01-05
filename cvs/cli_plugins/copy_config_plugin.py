from .base import SubcommandPlugin
from cvs.extension import ExtensionConfig
import os
import shutil


class CopyConfigPlugin(SubcommandPlugin):
    def get_name(self):
        return "copy-config"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser(
            "copy-config", help="List or copy config files from CVS package. Lists configs if --output not specified."
        )
        parser.add_argument(
            "path", nargs="?", help="Path to config file (e.g. training/jax/mi300x_distributed_llama_3_1_405b.json)"
        )
        parser.add_argument("--all", action="store_true", help="Copy all config files preserving directory structure")
        parser.add_argument("--output", help="Destination path to copy config file(s)")
        parser.add_argument(
            "--list",
            action="store_true",
            help="List available config files at the given path (lists all if no path specified)",
        )
        parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Copy-Config Commands:
  cvs copy-config                   List all available config files
  cvs copy-config training          List configs in training directory
  cvs copy-config training/jax      List configs in training/jax directory
  cvs copy-config --list            Same as above (list all)
  cvs copy-config training --list   Same as above (list training)

  Note: --list is optional, same behavior without it
  
  cvs copy-config --all --output /tmp/cvs/input/                          Copy all config files preserving directory structure
  cvs copy-config training/jax/mi300x_config.json --output ~/mi300.json   Copy specific config file
  cvs copy-config --all --output /tmp/cvs/input/ --force                  Force overwrite existing files"""

    def _find_config_root(self):
        """
        Find config directories from cvs and extension packages.

        Searches for config directories in:
        1. Core cvs package (cvs/input/config_file, cvs/input/cluster_file)
        2. Extension packages configured via extension.ini (e.g., cvs_extension/input)
        """
        plugin_dir = os.path.dirname(__file__)
        cvs_dir = os.path.dirname(plugin_dir)  # cvs/
        config_root = os.path.join(cvs_dir, "input", "config_file")
        cluster_root = os.path.join(cvs_dir, "input", "cluster_file")
        roots = []

        # Add core cvs config directories
        if os.path.exists(config_root):
            roots.append(config_root)
        if os.path.exists(cluster_root):
            roots.append(cluster_root)

        # Add extension config directories
        config = ExtensionConfig()
        for input_dir in config.get_input_dirs():
            config_file_dir = os.path.join(input_dir, "config_file")
            cluster_file_dir = os.path.join(input_dir, "cluster_file")

            if os.path.exists(config_file_dir):
                roots.append(config_file_dir)
            if os.path.exists(cluster_file_dir):
                roots.append(cluster_file_dir)

        return roots

    def _list_configs(self, root, subpath):
        base = os.path.join(root, subpath) if subpath else root
        if not os.path.exists(base):
            return []
        result = []
        for dirpath, dirs, files in os.walk(base):
            for f in files:
                if f.endswith(".json") or f.endswith(".yaml"):
                    rel = os.path.relpath(os.path.join(dirpath, f), root)
                    result.append(rel)
        return sorted(result)

    def _find_config_file(self, roots, subpath):
        for root in roots:
            candidate = os.path.join(root, subpath)
            if os.path.isfile(candidate):
                return candidate
        return None

    def run(self, args):
        roots = self._find_config_root()
        path = args.path or ""

        if args.all:
            if not args.output:
                print("Error: --output required when using --all")
                return
            # Create output directory if it doesn't exist
            try:
                os.makedirs(args.output, exist_ok=True)
            except Exception as e:
                print(f"Error creating output directory {args.output}: {e}")
                return
            copied_count = 0
            for root in roots:
                root_name = os.path.basename(root)
                configs = self._list_configs(root, "")
                for config in configs:
                    src = os.path.join(root, config)
                    dest = os.path.join(args.output, root_name, config)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    # Check for existing files
                    if os.path.exists(dest) and not args.force:
                        print(f"Error: File {dest} already exists. Use --force to overwrite.")
                        return
                    try:
                        shutil.copyfile(src, dest)
                        copied_count += 1
                    except Exception as e:
                        print(f"Error copying {src} to {dest}: {e}")
            print(f"Copied {copied_count} config files to {args.output}")
            return

        if args.list or not args.output:
            found = False
            for root in roots:
                configs = self._list_configs(root, path)
                if configs:
                    display_path = os.path.join(root, path) if path else root
                    print(f"Configs under {display_path}:")
                    for c in configs:
                        print(f"  {c}")
                    found = True
            if not found:
                print("No config files found at the specified path.")
            return
        else:
            if not path:
                print("Error: path to config file required for copying")
                return

            config_file = self._find_config_file(roots, path)
            if not config_file:
                print(f"Config file not found: {path}")
                return
            if os.path.isdir(args.output):
                dest = os.path.join(args.output, os.path.basename(config_file))
            else:
                dest = args.output
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            # Check for existing files
            if os.path.exists(dest) and not args.force:
                print(f"Error: File {dest} already exists. Use --force to overwrite.")
                return
            try:
                shutil.copyfile(config_file, dest)
                print(f"Copied {config_file} to {dest}")
            except Exception as e:
                print(f"Error copying {config_file} to {dest}: {e}")
