"""
Extension configuration loader for cvs.

This module discovers CVS extension packages and loads their configuration
(test directories, input/config directories, version). Names are collected from
several sources, then simply unioned and de-duplicated by package name. There is
no precedence or override between sources; a de-duped extension records the set
of all sources that referenced it, for information only.

Primary sources:
1. pip metadata / entry points (group ``cvs.extensions``) - auto-discovered for
   any extension installed in the same environment as cvs.
2. The per-venv plug-list managed by ``cvs extension --plug/--unplug``
   (``<sys.prefix>/etc/cvs/extensions.txt``).

Backward-compatible sources:
3. The ``CVS_EXTENSION_PKG_NAMES`` environment variable (comma-separated).
4. A legacy ``extension.ini`` copied into the cvs package directory by an
   extension's ``setup.py`` (Pattern 1 in extension.ini.sample).

For each discovered package, per-extension configuration is resolved from pip
metadata first (an entry-point object exposing ``tests_dirs``/``input_dirs``),
falling back to an ``extension.ini`` shipped inside the package, and finally to
the legacy bundled ``extension.ini``. Version is resolved via
``importlib.metadata.version`` first, then a ``version.txt`` in the package
directory, then ``"unknown"``.

Discovery is bounded by the running cvs interpreter's ``sys.path``: extensions
installed in a *different* venv are not discoverable (a Python import limitation).
"""

import os
import sys
import configparser
import importlib.util
import importlib.metadata as metadata


CORE_PKG_NAME = "cvs"
CORE_TESTS_DIR = "tests"

# Entry-point group that pip-installed extensions register under to opt in to
# auto-discovery, e.g. in pyproject.toml:
#   [project.entry-points."cvs.extensions"]
#   ext1 = "ext1"
ENTRY_POINT_GROUP = "cvs.extensions"

# Environment variable holding a comma-separated list of extension package names.
CVS_EXTENSION_PKG_NAMES = "CVS_EXTENSION_PKG_NAMES"


def get_plug_file_path():
    """Return the path to the per-venv plug-list file.

    The file may not exist. It lives outside the cvs package (under sys.prefix)
    so it survives a cvs reinstall/upgrade.
    """
    return os.path.join(sys.prefix, "etc", "cvs", "extensions.txt")


def read_plug_list(path=None):
    """Read extension package names from the plug-list file.

    One package name per line; blank lines and lines starting with '#' are
    ignored. Returns an empty list if the file is missing or unreadable (reads
    never raise, so normal cvs usage is unaffected by a read-only prefix).
    """
    path = path or get_plug_file_path()
    names = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    names.append(line)
    except OSError:
        pass
    return names


class Extension:
    """A single resolved CVS extension."""

    def __init__(self, name, version, tests_dirs, input_dirs, sources, found=True):
        self.name = name
        self.version = version
        # tests_dirs: list of (module_path, absolute_path)
        self.tests_dirs = tests_dirs
        # input_dirs: list of absolute paths
        self.input_dirs = input_dirs
        # sources: sorted list of every source that referenced this extension,
        # each one of 'metadata' | 'plugged' | 'env' | 'bundled-ini'. Informational
        # only - there is no precedence between sources.
        self.sources = sources
        # found: whether the package could actually be located/resolved
        self.found = found


class ExtensionConfig:
    """Discover CVS extensions and expose their aggregated configuration."""

    def __init__(self):
        """Discover all extensions from every supported source."""
        self.extensions = []
        # (package_name, parser) for a legacy bundled extension.ini, or None.
        self._bundled = None
        self._discover()

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    def _discover(self):
        """Populate ``self.extensions`` from all sources, unioned and de-duplicated.

        All sources are merged with no precedence between them. Each distinct
        package name yields one :class:`Extension` that records the set of every
        source that referenced it. Config/version resolution is source-independent.
        """
        # name -> list of sources (order-preserving, de-duplicated)
        sources_by_name = {}
        order = []

        def add(name, source):
            name = (name or "").strip()
            if not name or name == CORE_PKG_NAME:
                return
            if name not in sources_by_name:
                sources_by_name[name] = []
                order.append(name)
            if source not in sources_by_name[name]:
                sources_by_name[name].append(source)

        # Primary: pip metadata / entry points (auto-discovered).
        for name in self._entry_point_names():
            add(name, "metadata")

        # Primary: persistent plug-list (cvs extension --plug).
        for name in read_plug_list():
            add(name, "plugged")

        # Backward compatible: CVS_EXTENSION_PKG_NAMES environment variable.
        env = os.environ.get(CVS_EXTENSION_PKG_NAMES)
        if env:
            for name in env.split(","):
                add(name, "env")

        # Backward compatible: legacy extension.ini bundled into the cvs package.
        self._bundled = self._read_bundled_ini()
        if self._bundled:
            add(self._bundled[0], "bundled-ini")

        for name in order:
            self.extensions.append(self._resolve_extension(name, sorted(sources_by_name[name])))

        # Deterministic display/order.
        self.extensions.sort(key=lambda e: e.name)

    @staticmethod
    def _entry_point_names():
        """Return names of packages registered under the cvs.extensions group."""
        names = []
        try:
            eps = metadata.entry_points()
            # Python 3.10+ selectable API vs. 3.9 dict-like API.
            if hasattr(eps, "select"):
                selected = eps.select(group=ENTRY_POINT_GROUP)
            else:
                selected = eps.get(ENTRY_POINT_GROUP, [])
            for ep in selected:
                names.append(ep.name)
        except Exception:
            pass
        return names

    def _resolve_extension(self, name, sources):
        """Resolve a package name into an :class:`Extension` record.

        Config resolution order (all source-independent): pip metadata, then an
        extension.ini inside the package, then the legacy bundled extension.ini.
        """
        pkg_dir = self._find_package_dir(name)

        tests_dirs, input_dirs = self._resolve_config_from_metadata(name, pkg_dir)
        if tests_dirs is None and input_dirs is None:
            tests_dirs, input_dirs = self._resolve_config_from_ini(pkg_dir)
        if tests_dirs is None and input_dirs is None:
            tests_dirs, input_dirs = self._resolve_config_from_bundled(name)
        tests_dirs = tests_dirs or []
        input_dirs = input_dirs or []

        version = self._resolve_version(name, pkg_dir)
        found = pkg_dir is not None or version != "unknown" or bool(tests_dirs) or bool(input_dirs)
        return Extension(name, version, tests_dirs, input_dirs, sources, found)

    @staticmethod
    def _read_bundled_ini():
        """Read the legacy extension.ini bundled in the cvs package directory.

        This is "Pattern 1" in extension.ini.sample. Returns (package_name, parser)
        or None if there is no bundled ini or it names no extension package.
        """
        ini_path = os.path.join(os.path.dirname(__file__), "extension.ini")
        parser = ExtensionConfig._read_ini(ini_path)
        if parser is None or not parser.has_section("extensions"):
            return None
        name = parser.get("extensions", "package_name", fallback=None)
        if not name or name == CORE_PKG_NAME:
            return None
        return name, parser

    def _resolve_config_from_bundled(self, name):
        """Resolve tests/input dirs from the legacy bundled extension.ini."""
        if not self._bundled or self._bundled[0] != name:
            return None, None
        parser = self._bundled[1]
        base = os.path.dirname(os.path.dirname(__file__))  # site-packages
        tests_dirs = self._build_tests_dirs(self._ini_list(parser, "tests_dirs"), base)
        input_dirs = self._build_input_dirs(self._ini_list(parser, "input_dirs"), base)
        return tests_dirs, input_dirs

    # ------------------------------------------------------------------ #
    # Per-extension config resolution helpers
    # ------------------------------------------------------------------ #
    def _resolve_config_from_metadata(self, name, pkg_dir):
        """Resolve tests/input dirs from an entry-point object, if it exposes them.

        Returns (tests_dirs, input_dirs) or (None, None) to signal a fall back to
        extension.ini. Paths from metadata are resolved the same way as ini paths
        (relative to the package's site-packages directory).
        """
        try:
            eps = metadata.entry_points()
            if hasattr(eps, "select"):
                selected = list(eps.select(group=ENTRY_POINT_GROUP, name=name))
            else:
                selected = [ep for ep in eps.get(ENTRY_POINT_GROUP, []) if ep.name == name]
            if not selected:
                return None, None
            obj = selected[0].load()
        except Exception:
            return None, None

        tests_raw = getattr(obj, "tests_dirs", None)
        input_raw = getattr(obj, "input_dirs", None)
        if not tests_raw and not input_raw:
            # Metadata does not carry the required fields; fall back to ini.
            return None, None

        base = os.path.dirname(pkg_dir) if pkg_dir else None
        tests_dirs = self._build_tests_dirs(list(tests_raw or []), base)
        input_dirs = self._build_input_dirs(list(input_raw or []), base)
        return tests_dirs, input_dirs

    def _resolve_config_from_ini(self, pkg_dir):
        """Resolve tests/input dirs from an extension.ini inside the package dir."""
        if not pkg_dir:
            return None, None
        ini_path = os.path.join(pkg_dir, "extension.ini")
        parser = self._read_ini(ini_path)
        if parser is None:
            return None, None
        base = os.path.dirname(pkg_dir)  # site-packages
        tests_dirs = self._build_tests_dirs(self._ini_list(parser, "tests_dirs"), base)
        input_dirs = self._build_input_dirs(self._ini_list(parser, "input_dirs"), base)
        return tests_dirs, input_dirs

    @staticmethod
    def _read_ini(ini_path):
        """Read and parse an ini file, or return None if missing/unparseable."""
        if not os.path.exists(ini_path):
            return None
        parser = configparser.ConfigParser()
        try:
            parser.read(ini_path)
        except Exception as e:
            print(f"Warning: Could not parse extension config {ini_path}: {e}", file=sys.stderr)
            return None
        return parser

    @staticmethod
    def _ini_list(parser, option):
        """Return a comma-separated ini option under [extensions] as a list."""
        if parser.has_option("extensions", option):
            return [d.strip() for d in parser.get("extensions", option).split(",") if d.strip()]
        return []

    @staticmethod
    def _build_tests_dirs(paths, base):
        """Build (module_path, absolute_path) tuples from relative test paths."""
        result = []
        for p in paths:
            module_path = p.replace(os.sep, ".").replace("/", ".")
            abs_path = p if os.path.isabs(p) else os.path.join(base or "", p)
            result.append((module_path, abs_path))
        return result

    @staticmethod
    def _build_input_dirs(paths, base):
        """Build absolute input directory paths from relative/absolute input paths."""
        result = []
        for p in paths:
            if os.path.isabs(p):
                result.append(p)
            else:
                result.append(os.path.join(base or "", p))
        return result

    @staticmethod
    def _resolve_version(name, pkg_dir):
        """Resolve a package version: pip metadata -> version.txt -> 'unknown'."""
        try:
            return metadata.version(name)
        except Exception:
            pass
        if pkg_dir:
            version_file = os.path.join(pkg_dir, "version.txt")
            if os.path.exists(version_file):
                try:
                    with open(version_file) as f:
                        return f.read().strip()
                except OSError:
                    pass
        return "unknown"

    @staticmethod
    def _find_package_dir(name):
        """Return the directory of an importable package, or None."""
        try:
            spec = importlib.util.find_spec(name)
            if spec and spec.origin:
                return os.path.dirname(spec.origin)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_extensions(self):
        """Return the list of discovered :class:`Extension` records."""
        return self.extensions

    def get_package_name(self):
        """Back-compat: return the first extension name, or 'cvs' if none.

        Prefer :meth:`get_extensions` for multi-extension aware code.
        """
        if self.extensions:
            return self.extensions[0].name
        return CORE_PKG_NAME

    def get_tests_dirs(self):
        """Return aggregated test directories across all extensions.

        Returns:
            list: List of tuples (pkg_name, module_path, absolute_path).
        """
        result = []
        for ext in self.extensions:
            for module_path, abs_path in ext.tests_dirs:
                result.append((ext.name, module_path, abs_path))
        return result

    def get_input_dirs(self):
        """Return aggregated input/config directories across all extensions."""
        result = []
        for ext in self.extensions:
            result.extend(ext.input_dirs)
        return result
