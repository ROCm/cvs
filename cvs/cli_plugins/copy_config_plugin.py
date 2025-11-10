from .base import SubcommandPlugin
import os
import shutil

class CopyConfigPlugin(SubcommandPlugin):
    def get_name(self):
        return "copy-config"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser('copy-config', help='List or copy config files from CVS package. Lists configs if --output not specified.')
        parser.add_argument('path', nargs='*', help='Hierarchical path to config file (e.g. training jax mi300x_distributed_llama_3_1_405b.json)')
        parser.add_argument('--output', help='Destination path to copy config file')
        parser.add_argument('--list', action='store_true', help='List available config files at the given path (lists all if no path specified)')
        parser.set_defaults(_plugin=self)
        return parser

    def _find_config_root(self):
        # Use the directory relative to this plugin file
        plugin_dir = os.path.dirname(__file__)
        cvs_dir = os.path.dirname(plugin_dir)  # cvs/
        config_root = os.path.join(cvs_dir, 'input', 'config_file')
        cluster_root = os.path.join(cvs_dir, 'input', 'cluster_file')
        roots = []
        if os.path.exists(config_root):
            roots.append(config_root)
        if os.path.exists(cluster_root):
            roots.append(cluster_root)
        return roots

    def _list_configs(self, root, subpath):
        base = os.path.join(root, *subpath)
        if not os.path.exists(base):
            return []
        result = []
        for dirpath, dirs, files in os.walk(base):
            for f in files:
                if f.endswith('.json'):
                    rel = os.path.relpath(os.path.join(dirpath, f), root)
                    result.append(rel)
        return sorted(result)

    def _find_config_file(self, roots, subpath):
        for root in roots:
            candidate = os.path.join(root, *subpath)
            if os.path.isfile(candidate):
                return candidate
        return None

    def run(self, args):
        roots = self._find_config_root()
        if args.list or not args.output:
            found = False
            for root in roots:
                configs = self._list_configs(root, args.path)
                if configs:
                    print(f"Configs under {os.path.join(root, *args.path)}:")
                    for c in configs:
                        print(f"  {c}")
                    found = True
            if not found:
                print("No config files found at the specified path.")
            return
        else:
            if not args.path:
                print("Error: path to config file required for copying")
                return
            config_file = self._find_config_file(roots, args.path)
            if not config_file:
                print(f"Config file not found: {'/'.join(args.path)}")
                return
            if os.path.isdir(args.output):
                dest = os.path.join(args.output, os.path.basename(config_file))
            else:
                dest = args.output
            shutil.copyfile(config_file, dest)
            print(f"Copied {config_file} to {dest}")
