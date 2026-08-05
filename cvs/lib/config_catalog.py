"""Load and validate the bundled CVS test-suite configuration catalog."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from cvs.extension import ExtensionConfig


CATALOG_FILENAME = "config_catalog.json"
SUPPORTED_PLATFORMS = frozenset({"all", "mi300", "mi300x", "mi350", "mi355"})
CONFIG_SUFFIXES = frozenset({".json", ".yaml", ".yml"})


class ConfigCatalogError(ValueError):
    """Raised when a configuration catalog is malformed or incomplete."""


@dataclass(frozen=True)
class Configuration:
    """A bundled config file and the suites/platforms it supports."""

    path: str
    platforms: Tuple[str, ...]
    suites: Tuple[str, ...]


class ConfigCatalog:
    """The explicit compatibility mapping used by CVS and consuming applications."""

    def __init__(self, configurations: Sequence[Configuration], unavailable_suites: dict):
        self.configurations = tuple(configurations)
        self.unavailable_suites = dict(unavailable_suites)

    @classmethod
    def from_input_roots(cls, input_roots: Iterable[Path]):
        """Load catalogs from core and optional extension input roots."""
        roots = tuple(Path(root) for root in input_roots)
        configurations = []
        unavailable_suites = {}
        known_paths = set()

        for root in roots:
            catalog_path = root / CATALOG_FILENAME
            if not catalog_path.is_file():
                continue
            try:
                document = json.loads(catalog_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ConfigCatalogError(f"Unable to read {catalog_path}: {error}") from error

            cls._validate_document(document, catalog_path)
            for raw_configuration in document["configurations"]:
                configuration = cls._parse_configuration(raw_configuration, catalog_path)
                if configuration.path in known_paths:
                    raise ConfigCatalogError(f"Duplicate config path in catalogs: {configuration.path}")
                known_paths.add(configuration.path)
                configurations.append(configuration)

            for raw_unavailable in document["unavailable_suites"]:
                suite, reason = cls._parse_unavailable_suite(raw_unavailable, catalog_path)
                if suite in unavailable_suites:
                    raise ConfigCatalogError(f"Duplicate unavailable suite in catalogs: {suite}")
                unavailable_suites[suite] = reason

        catalog = cls(configurations, unavailable_suites)
        catalog._validate_config_paths(roots)
        catalog.validate_config_files(roots)
        catalog._validate_no_conflicts()
        return catalog

    @staticmethod
    def _validate_document(document, catalog_path):
        if not isinstance(document, dict):
            raise ConfigCatalogError(f"{catalog_path} must contain a JSON object")
        if document.get("schema_version") != 1:
            raise ConfigCatalogError(f"{catalog_path} must use schema_version 1")
        if not isinstance(document.get("configurations"), list):
            raise ConfigCatalogError(f"{catalog_path} field 'configurations' must be a list")
        if not isinstance(document.get("unavailable_suites"), list):
            raise ConfigCatalogError(f"{catalog_path} field 'unavailable_suites' must be a list")

    @staticmethod
    def _parse_configuration(raw_configuration, catalog_path):
        if not isinstance(raw_configuration, dict):
            raise ConfigCatalogError(f"Invalid configuration entry in {catalog_path}")
        path = raw_configuration.get("path")
        platforms = raw_configuration.get("platforms")
        suites = raw_configuration.get("suites")
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ConfigCatalogError(f"Invalid config path in {catalog_path}: {path!r}")
        if not isinstance(platforms, list) or not platforms or not all(isinstance(item, str) for item in platforms):
            raise ConfigCatalogError(f"Config {path} must declare one or more platforms")
        if not isinstance(suites, list) or not suites or not all(isinstance(item, str) and item for item in suites):
            raise ConfigCatalogError(f"Config {path} must declare one or more suites")

        normalized_platforms = tuple(platform.lower() for platform in platforms)
        invalid_platforms = set(normalized_platforms) - SUPPORTED_PLATFORMS
        if invalid_platforms:
            raise ConfigCatalogError(f"Config {path} uses unsupported platforms: {sorted(invalid_platforms)}")
        if len(set(normalized_platforms)) != len(normalized_platforms):
            raise ConfigCatalogError(f"Config {path} repeats a platform")
        if "all" in normalized_platforms and len(normalized_platforms) != 1:
            raise ConfigCatalogError(f"Config {path} cannot combine 'all' with specific platforms")
        if len(set(suites)) != len(suites):
            raise ConfigCatalogError(f"Config {path} repeats a suite")
        return Configuration(path=path, platforms=normalized_platforms, suites=tuple(suites))

    @staticmethod
    def _parse_unavailable_suite(raw_unavailable, catalog_path):
        if not isinstance(raw_unavailable, dict):
            raise ConfigCatalogError(f"Invalid unavailable suite entry in {catalog_path}")
        suite = raw_unavailable.get("suite")
        reason = raw_unavailable.get("reason")
        if not isinstance(suite, str) or not suite or not isinstance(reason, str) or not reason:
            raise ConfigCatalogError(f"Unavailable suite entries in {catalog_path} require suite and reason")
        return suite, reason

    def _validate_config_paths(self, input_roots: Iterable[Path]):
        config_roots = [root / "config_file" for root in input_roots]
        for configuration in self.configurations:
            if not any((config_root / configuration.path).is_file() for config_root in config_roots):
                raise ConfigCatalogError(f"Catalog config file does not exist: {configuration.path}")

    def _validate_no_conflicts(self):
        configured_suites = {suite for configuration in self.configurations for suite in configuration.suites}
        conflicting_suites = configured_suites & self.unavailable_suites.keys()
        if conflicting_suites:
            raise ConfigCatalogError(f"Suites cannot be both configured and unavailable: {sorted(conflicting_suites)}")

    def validate_config_files(self, input_roots: Iterable[Path]):
        """Require every bundled JSON/YAML test config to be present in the catalog."""
        bundled_paths = set()
        for input_root in input_roots:
            config_root = Path(input_root) / "config_file"
            if not config_root.is_dir():
                continue
            for config_path in config_root.rglob("*"):
                if config_path.is_file() and config_path.suffix.lower() in CONFIG_SUFFIXES:
                    relative_path = config_path.relative_to(config_root).as_posix()
                    if relative_path in bundled_paths:
                        raise ConfigCatalogError(f"Duplicate bundled config path: {relative_path}")
                    bundled_paths.add(relative_path)

        catalog_paths = {configuration.path for configuration in self.configurations}
        missing_paths = bundled_paths - catalog_paths
        if missing_paths:
            raise ConfigCatalogError(f"Catalog is missing config files: {sorted(missing_paths)}")

    def validate_suites(self, suites: Iterable[str]):
        """Require every runnable suite to be configured or explicitly unavailable."""
        suite_names = set(suites)
        catalog_suites = {suite for configuration in self.configurations for suite in configuration.suites}
        catalog_suites.update(self.unavailable_suites)
        missing_suites = suite_names - catalog_suites
        unknown_suites = catalog_suites - suite_names
        if missing_suites or unknown_suites:
            messages = []
            if missing_suites:
                messages.append(f"missing suites: {sorted(missing_suites)}")
            if unknown_suites:
                messages.append(f"unknown suites: {sorted(unknown_suites)}")
            raise ConfigCatalogError("Catalog suite coverage is invalid: " + "; ".join(messages))

    def configurations_for(self, suite: str, platform: Optional[str] = None) -> List[Configuration]:
        """Return compatible configurations for a suite, optionally filtered by platform."""
        normalized_platform = platform.lower() if platform else None
        if normalized_platform and normalized_platform not in SUPPORTED_PLATFORMS - {"all"}:
            raise ConfigCatalogError(
                f"Unknown platform '{platform}'. Supported platforms: {', '.join(self.platforms())}"
            )
        return [
            configuration
            for configuration in self.configurations
            if suite in configuration.suites
            and (
                not normalized_platform
                or "all" in configuration.platforms
                or normalized_platform in configuration.platforms
            )
        ]

    def unavailable_reason(self, suite: str) -> Optional[str]:
        """Return the catalogued reason when no bundled config exists for a suite."""
        return self.unavailable_suites.get(suite)

    @staticmethod
    def platforms() -> List[str]:
        """Return the public platform names accepted by the CLI."""
        return sorted(SUPPORTED_PLATFORMS - {"all"})


def catalog_input_roots() -> Tuple[Path, ...]:
    """Return the core and configured extension input roots in lookup order."""
    core_input_root = Path(__file__).resolve().parents[1] / "input"
    extension_roots = [Path(path) for path in ExtensionConfig().get_input_dirs()]
    return tuple([core_input_root, *extension_roots])


def load_config_catalog() -> ConfigCatalog:
    """Load the core catalog and any optional extension catalogs."""
    return ConfigCatalog.from_input_roots(catalog_input_roots())
