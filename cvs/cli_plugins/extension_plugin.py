import os
import sys

from .base import SubcommandPlugin
from cvs.extension import (
    ExtensionConfig,
    CVS_EXTENSION_PKG_NAMES,
    get_plug_file_path,
    read_plug_list,
)


class ExtensionPlugin(SubcommandPlugin):
    """Manage the persistent, per-venv list of plugged CVS extensions.

    The plug-list is a simple text file (one package name per line) stored at
    ``<sys.prefix>/etc/cvs/extensions.txt``. It is a persistent equivalent of the
    ``CVS_EXTENSION_PKG_NAMES`` environment variable: names recorded here are
    folded into extension discovery for every subsequent cvs command in this venv.
    """

    def get_name(self):
        return "extension"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser(
            "extension",
            help="Manage CVS extensions (plug/unplug/list). Lists discovered extensions if no flag given.",
        )
        parser.add_argument(
            "--plug",
            metavar="PKG[,PKG...]",
            help="Add one or more extension package names to the persistent plug-list",
        )
        parser.add_argument(
            "--unplug",
            metavar="PKG[,PKG...]",
            help="Remove one or more extension package names from the plug-list",
        )
        parser.add_argument(
            "--list",
            action="store_true",
            help="List discovered extensions and their versions (default when no flag given)",
        )
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Extension Commands:
  cvs extension                      List discovered extensions and versions
  cvs extension --list               Same as above
  cvs extension --plug ext1          Persistently register extension 'ext1'
  cvs extension --plug ext1,ext2     Register multiple extensions
  cvs extension --unplug ext2        Remove 'ext2' from the plug-list"""

    @staticmethod
    def _parse_names(value):
        return [n.strip() for n in value.split(",") if n.strip()]

    @staticmethod
    def _write_plug_list(names):
        """Write the plug-list atomically. Returns the path on success.

        Raises OSError if the location is not writable.
        """
        path = get_plug_file_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = (
            "# CVS plugged extensions - managed by `cvs extension --plug/--unplug`\n"
            "# One package name per line; lines starting with # are ignored.\n"
        )
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            f.write(header)
            for name in names:
                f.write(f"{name}\n")
        os.replace(tmp, path)
        return path

    def _plug(self, names):
        existing = read_plug_list()
        merged = list(existing)
        added = []
        for name in names:
            if name not in merged:
                merged.append(name)
                added.append(name)
        try:
            path = self._write_plug_list(merged)
        except OSError as e:
            print(f"Error: could not write plug-list at {get_plug_file_path()}: {e}", file=sys.stderr)
            print(
                f"The environment (sys.prefix) appears read-only. "
                f"Use the {CVS_EXTENSION_PKG_NAMES} environment variable instead, e.g.:\n"
                f"  export {CVS_EXTENSION_PKG_NAMES}={','.join(merged)}",
                file=sys.stderr,
            )
            sys.exit(1)
        if added:
            print(f"Plugged: {', '.join(added)}")
        else:
            print("No new extensions added (already plugged).")
        print(f"Plug-list: {path}")

    def _unplug(self, names):
        existing = read_plug_list()
        remove = set(names)
        merged = [n for n in existing if n not in remove]
        removed = [n for n in existing if n in remove]
        not_present = [n for n in names if n not in existing]
        try:
            path = self._write_plug_list(merged)
        except OSError as e:
            print(f"Error: could not write plug-list at {get_plug_file_path()}: {e}", file=sys.stderr)
            sys.exit(1)
        if removed:
            print(f"Unplugged: {', '.join(removed)}")
        if not_present:
            print(f"Not in plug-list (ignored): {', '.join(not_present)}")
        print(f"Plug-list: {path}")

    def _list(self):
        config = ExtensionConfig()
        extensions = config.get_extensions()
        print("\nDiscovered Extensions")
        print("=" * 80)
        if not extensions:
            print("  (none)")
        else:
            for ext in extensions:
                status = "" if ext.found else "  [not found]"
                sources = ", ".join(ext.sources)
                print(f"  {ext.name}: {ext.version}  (sources: {sources}){status}")
        print("-" * 80)
        print(f"Plug-list file: {get_plug_file_path()}")
        env = os.environ.get(CVS_EXTENSION_PKG_NAMES)
        if env:
            print(f"{CVS_EXTENSION_PKG_NAMES}={env}")
        print()

    def run(self, args):
        if args.plug:
            self._plug(self._parse_names(args.plug))
        if args.unplug:
            self._unplug(self._parse_names(args.unplug))
        if args.list or (not args.plug and not args.unplug):
            self._list()
