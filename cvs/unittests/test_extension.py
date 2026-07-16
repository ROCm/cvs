import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import configparser

import importlib.metadata as metadata

from cvs.extension import ExtensionConfig, read_plug_list


# --------------------------------------------------------------------------- #
# Test helpers: fakes for entry points and package specs
# --------------------------------------------------------------------------- #
class _FakeEP:
    """Minimal stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name, obj=None):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


class _FakeEntryPoints:
    """Mimics the Python 3.10+ selectable entry_points() API."""

    def __init__(self, mapping):
        # mapping: group -> list[_FakeEP]
        self._mapping = mapping

    def select(self, group=None, name=None):
        eps = self._mapping.get(group, [])
        if name is not None:
            eps = [e for e in eps if e.name == name]
        return eps


def _make_pkg(tmpdir, name, tests_dirs=None, input_dirs=None, package_name=None, version=None):
    """Create a fake installed package directory with an optional extension.ini."""
    pkg_dir = os.path.join(tmpdir, name)
    os.makedirs(pkg_dir, exist_ok=True)
    init_file = os.path.join(pkg_dir, "__init__.py")
    with open(init_file, "w") as f:
        f.write("")
    if tests_dirs is not None or input_dirs is not None or package_name is not None:
        parser = configparser.ConfigParser()
        parser.add_section("extensions")
        parser.set("extensions", "package_name", package_name or name)
        if tests_dirs is not None:
            parser.set("extensions", "tests_dirs", tests_dirs)
        if input_dirs is not None:
            parser.set("extensions", "input_dirs", input_dirs)
        with open(os.path.join(pkg_dir, "extension.ini"), "w") as f:
            parser.write(f)
    if version is not None:
        with open(os.path.join(pkg_dir, "version.txt"), "w") as f:
            f.write(version)
    return pkg_dir, init_file


def _find_spec_factory(specs_by_name):
    """Return a find_spec replacement resolving names to fake specs."""

    def _find_spec(name):
        origin = specs_by_name.get(name)
        if origin is None:
            return None
        spec = MagicMock()
        spec.origin = origin
        return spec

    return _find_spec


class TestExtensionConfig(unittest.TestCase):
    """Tests for the multi-source extension discovery in ExtensionConfig."""

    def setUp(self):
        # Neutralize ambient sources unless a test opts in.
        self._patchers = [
            patch("cvs.extension.read_plug_list", return_value=[]),
            patch("cvs.extension.metadata.entry_points", return_value=_FakeEntryPoints({})),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()

    def test_no_extensions(self):
        """With nothing configured, discovery yields no extensions."""
        with patch.dict(os.environ, {}, clear=True):
            config = ExtensionConfig()
            self.assertEqual(config.get_extensions(), [])
            self.assertEqual(config.get_package_name(), "cvs")
            self.assertEqual(config.get_tests_dirs(), [])
            self.assertEqual(config.get_input_dirs(), [])

    def test_env_var_discovery_via_ini(self):
        """An extension named via CVS_EXTENSION_PKG_NAMES is resolved from its ini."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir, init_file = _make_pkg(
                tmp, "ext1", tests_dirs="ext1/tests", input_dirs="ext1/input", version="1.2.3"
            )
            with (
                patch.dict(os.environ, {"CVS_EXTENSION_PKG_NAMES": "ext1"}, clear=True),
                patch("importlib.util.find_spec", side_effect=_find_spec_factory({"ext1": init_file})),
                patch("cvs.extension.metadata.version", side_effect=metadata.PackageNotFoundError()),
            ):
                config = ExtensionConfig()
                exts = config.get_extensions()
                self.assertEqual(len(exts), 1)
                self.assertEqual(exts[0].name, "ext1")
                self.assertEqual(exts[0].sources, ["env"])
                # Version falls back to version.txt.
                self.assertEqual(exts[0].version, "1.2.3")
                # tests_dirs aggregated as (pkg_name, module_path, abs_path) triples.
                triples = config.get_tests_dirs()
                self.assertEqual(len(triples), 1)
                pkg_name, module_path, abs_path = triples[0]
                self.assertEqual(pkg_name, "ext1")
                self.assertEqual(module_path, "ext1.tests")
                self.assertEqual(abs_path, os.path.join(tmp, "ext1", "tests"))
                self.assertIn(os.path.join(tmp, "ext1", "input"), config.get_input_dirs())

    def test_entry_point_discovery(self):
        """A pip-installed extension is auto-discovered via entry points."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir, init_file = _make_pkg(tmp, "ext1", tests_dirs="ext1/tests", input_dirs="ext1/input")
            eps = _FakeEntryPoints({"cvs.extensions": [_FakeEP("ext1", obj=None)]})
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("cvs.extension.metadata.entry_points", return_value=eps),
                patch("importlib.util.find_spec", side_effect=_find_spec_factory({"ext1": init_file})),
                patch("cvs.extension.metadata.version", return_value="2.0.0"),
            ):
                config = ExtensionConfig()
                exts = config.get_extensions()
                self.assertEqual(len(exts), 1)
                self.assertEqual(exts[0].name, "ext1")
                self.assertEqual(exts[0].sources, ["metadata"])
                self.assertEqual(exts[0].version, "2.0.0")

    def test_metadata_config_overrides_ini(self):
        """Entry-point object exposing tests_dirs/input_dirs takes precedence over ini."""
        with tempfile.TemporaryDirectory() as tmp:
            # ini says ext1/tests, but the metadata object says ext1/other_tests.
            pkg_dir, init_file = _make_pkg(tmp, "ext1", tests_dirs="ext1/tests", input_dirs="ext1/input")
            meta_obj = MagicMock()
            meta_obj.tests_dirs = ["ext1/other_tests"]
            meta_obj.input_dirs = ["ext1/other_input"]
            eps = _FakeEntryPoints({"cvs.extensions": [_FakeEP("ext1", obj=meta_obj)]})
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("cvs.extension.metadata.entry_points", return_value=eps),
                patch("importlib.util.find_spec", side_effect=_find_spec_factory({"ext1": init_file})),
                patch("cvs.extension.metadata.version", return_value="2.0.0"),
            ):
                config = ExtensionConfig()
                triples = config.get_tests_dirs()
                self.assertEqual(triples[0][2], os.path.join(tmp, "ext1", "other_tests"))

    def test_multiple_extensions_aggregate(self):
        """Multiple extensions from different sources aggregate and sort by name."""
        with tempfile.TemporaryDirectory() as tmp:
            _, init1 = _make_pkg(tmp, "ext1", tests_dirs="ext1/tests", input_dirs="ext1/input", version="1.0.0")
            _, init2 = _make_pkg(tmp, "ext2", tests_dirs="ext2/tests", input_dirs="ext2/input", version="2.0.0")
            eps = _FakeEntryPoints({"cvs.extensions": [_FakeEP("ext2", obj=None)]})
            with (
                patch.dict(os.environ, {"CVS_EXTENSION_PKG_NAMES": "ext1"}, clear=True),
                patch("cvs.extension.metadata.entry_points", return_value=eps),
                patch(
                    "importlib.util.find_spec",
                    side_effect=_find_spec_factory({"ext1": init1, "ext2": init2}),
                ),
                patch("cvs.extension.metadata.version", side_effect=metadata.PackageNotFoundError()),
            ):
                config = ExtensionConfig()
                names = [e.name for e in config.get_extensions()]
                self.assertEqual(names, ["ext1", "ext2"])
                pkg_names = {t[0] for t in config.get_tests_dirs()}
                self.assertEqual(pkg_names, {"ext1", "ext2"})
                self.assertEqual(len(config.get_input_dirs()), 2)

    def test_env_var_dedupes_with_entry_point(self):
        """A name from both env var and entry points appears once, recording both sources."""
        with tempfile.TemporaryDirectory() as tmp:
            _, init1 = _make_pkg(tmp, "ext1", tests_dirs="ext1/tests")
            eps = _FakeEntryPoints({"cvs.extensions": [_FakeEP("ext1", obj=None)]})
            with (
                patch.dict(os.environ, {"CVS_EXTENSION_PKG_NAMES": "ext1"}, clear=True),
                patch("cvs.extension.metadata.entry_points", return_value=eps),
                patch("importlib.util.find_spec", side_effect=_find_spec_factory({"ext1": init1})),
                patch("cvs.extension.metadata.version", return_value="1.0.0"),
            ):
                config = ExtensionConfig()
                exts = config.get_extensions()
                self.assertEqual(len(exts), 1)
                # No precedence: both sources are recorded (sorted).
                self.assertEqual(exts[0].sources, ["env", "metadata"])

    def test_plugged_but_missing_package(self):
        """A plugged name whose package is absent is reported not found, never crashes."""
        with patch.dict(os.environ, {}, clear=True), patch("cvs.extension.read_plug_list", return_value=["ghost"]):
            with (
                patch("importlib.util.find_spec", side_effect=_find_spec_factory({})),
                patch("cvs.extension.metadata.version", side_effect=metadata.PackageNotFoundError()),
            ):
                config = ExtensionConfig()
                exts = config.get_extensions()
                self.assertEqual(len(exts), 1)
                self.assertEqual(exts[0].name, "ghost")
                self.assertFalse(exts[0].found)
                self.assertEqual(exts[0].version, "unknown")

    def test_absolute_tests_dir(self):
        """Absolute tests_dirs paths are preserved."""
        with tempfile.TemporaryDirectory() as tmp:
            abs_tests = os.path.join(tmp, "abs_tests")
            _, init1 = _make_pkg(tmp, "ext1", tests_dirs=abs_tests)
            with (
                patch.dict(os.environ, {"CVS_EXTENSION_PKG_NAMES": "ext1"}, clear=True),
                patch("importlib.util.find_spec", side_effect=_find_spec_factory({"ext1": init1})),
                patch("cvs.extension.metadata.version", return_value="1.0.0"),
            ):
                config = ExtensionConfig()
                abs_paths = [t[2] for t in config.get_tests_dirs()]
                self.assertIn(abs_tests, abs_paths)


class TestPlugList(unittest.TestCase):
    """Tests for the plug-list read helper."""

    def test_read_plug_list_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_plug_list(os.path.join(tmp, "nope.txt")), [])

    def test_read_plug_list_parses_and_ignores_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "extensions.txt")
            with open(path, "w") as f:
                f.write("# comment\n\next1\n  ext2  \n# another\n")
            self.assertEqual(read_plug_list(path), ["ext1", "ext2"])


class TestExtensionPlugin(unittest.TestCase):
    """Tests for the `cvs extension` plug/unplug command."""

    def _plugin(self):
        from cvs.cli_plugins.extension_plugin import ExtensionPlugin

        return ExtensionPlugin()

    def test_plug_and_unplug_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            plug_file = os.path.join(tmp, "etc", "cvs", "extensions.txt")
            with (
                patch("cvs.cli_plugins.extension_plugin.get_plug_file_path", return_value=plug_file),
                patch(
                    "cvs.cli_plugins.extension_plugin.read_plug_list",
                    side_effect=lambda: read_plug_list(plug_file),
                ),
            ):
                plugin = self._plugin()
                plugin._plug(["ext1", "ext2"])
                self.assertEqual(read_plug_list(plug_file), ["ext1", "ext2"])
                # Re-plug is idempotent.
                plugin._plug(["ext2"])
                self.assertEqual(read_plug_list(plug_file), ["ext1", "ext2"])
                # Unplug removes one.
                plugin._unplug(["ext1"])
                self.assertEqual(read_plug_list(plug_file), ["ext2"])

    def test_parse_names(self):
        plugin = self._plugin()
        self.assertEqual(plugin._parse_names("a, b ,,c"), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
