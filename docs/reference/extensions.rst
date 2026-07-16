.. meta::
  :description: Reference specification for CVS extension discovery and configuration
  :keywords: CVS, extension, extension.ini, entry points, plug-list, cvs.extensions

**********
Extensions
**********

This page is the formal specification for how CVS discovers and configures
extension packages. For a task-oriented walkthrough, see
:doc:`Extend CVS with custom tests </how-to/extend-cvs>`.

An extension resolves to a record ``(name, version, tests_dirs, input_dirs,
sources, found)``. CVS discovers extensions, merges their tests into ``cvs list``,
their input/config files into ``cvs copy-config``, and their versions into
``cvs --version``.

Discovery sources
=================

Extension package names come from four sources. They are **unioned** (there is no
precedence or override between them) and **de-duplicated** by package name. A
single package may legitimately appear in more than one source; the resulting
extension records the set of all sources that referenced it, shown for
information only by ``cvs extension --list``.

.. list-table::
  :header-rows: 1
  :widths: 22 18 60

  * - Source
    - ``--list`` label
    - Description
  * - pip entry points
    - ``metadata``
    - Distributions declaring an entry point in the ``cvs.extensions`` group
      (see below). The common, pip-installed case; no env var or extra file
      needed.
  * - Plug-list
    - ``plugged``
    - Package names registered with ``cvs extension --plug``, stored per-venv at
      ``<sys.prefix>/etc/cvs/extensions.txt``.
  * - ``CVS_EXTENSION_PKG_NAMES``
    - ``env``
    - Comma-separated package names in an environment variable; ephemeral,
      per-invocation registration.
  * - Bundled ``extension.ini``
    - ``bundled-ini``
    - A legacy ``extension.ini`` copied into the ``cvs`` package directory by an
      extension's ``setup.py`` (Pattern 1).

Entry-point group
=================

A pip-installable extension opts in by declaring an entry point in the
``cvs.extensions`` group. In ``pyproject.toml``:

.. code:: toml

  [project.entry-points."cvs.extensions"]
  my_cvs_ext = "my_cvs_ext"

The ``setup.py`` equivalent:

.. code:: python

  entry_points={"cvs.extensions": ["my_cvs_ext = my_cvs_ext"]}

The entry-point name is the extension's package name. The entry-point value may
resolve to an object exposing ``tests_dirs`` / ``input_dirs`` attributes; if it
does, those are used, otherwise CVS falls back to the package's ``extension.ini``.

``extension.ini`` keys
=====================

The ``[extensions]`` section supports the following keys. Directory paths are
comma-separated and resolved relative to the site-packages root (the parent of
the extension's package directory); absolute paths are used as-is.

.. list-table::
  :header-rows: 1
  :widths: 20 15 65

  * - Key
    - Type
    - Description
  * - ``package_name``
    - string
    - Name of the extension package, shown alongside ``cvs`` in
      ``cvs --version``.
  * - ``tests_dirs``
    - comma-separated paths
    - Test directories added to ``cvs list`` / ``cvs run`` discovery, bucketed
      under the extension's package name.
  * - ``input_dirs``
    - comma-separated paths
    - Input/config directories searched by ``cvs copy-config`` (each may contain
      ``config_file/``, ``cluster_file/``, and ``env_file/`` subdirectories).

A fully commented ``extension.ini.sample`` lives at the repository root.

Per-package resolution (metadata-then-ini)
==========================================

For each discovered package, CVS resolves fields in this order:

- ``tests_dirs`` / ``input_dirs``: an entry-point object's attributes first, then
  an ``extension.ini`` shipped inside the package, then the legacy bundled
  ``extension.ini``.
- **version**: ``importlib.metadata.version(name)`` first, then ``version.txt`` in
  the package directory, then ``"unknown"``.

This "support both" rule lets the same package work whether it was pip-installed
(has ``.dist-info`` metadata) or tar-extracted (ships only ``extension.ini`` and
``version.txt``).

Plug-list file
==============

``cvs extension --plug`` / ``--unplug`` maintain a per-environment plug-list:

- **Location**: ``<sys.prefix>/etc/cvs/extensions.txt``.
- **Format**: one package name per line; blank lines and lines starting with
  ``#`` are ignored. Names are stored de-duplicated.

.. code:: text

  # CVS plugged extensions - managed by `cvs extension --plug/--unplug`
  # One package name per line; lines starting with # are ignored.
  ext1
  ext2

- **Reads never fail**: a missing or unreadable plug-list is treated as "no
  plugged extensions", so a read-only ``sys.prefix`` never breaks normal usage.
- **Writes** (``--plug`` / ``--unplug``) require a writable ``sys.prefix`` (the
  common user-owned venv case). On a read-only prefix the command prints a clear
  error and points at ``CVS_EXTENSION_PKG_NAMES`` as an escape hatch.

Discovery scope
==============

Discovery is bounded by the running ``cvs`` interpreter's ``sys.path``:

- **Same venv**: fully supported by all sources.
- **Different location, same interpreter** (on ``PYTHONPATH``): supported via the
  plug-list or ``CVS_EXTENSION_PKG_NAMES``; the version falls back to
  ``version.txt`` / ``extension.ini`` when no ``.dist-info`` is present.
- **Different venv**: not supported by design -- this is a Python import
  limitation, not a CVS one.

Commands
========

.. list-table::
  :header-rows: 1
  :widths: 32 68

  * - Command
    - Effect
  * - ``cvs extension`` / ``cvs extension --list``
    - List discovered extensions, their sources, and versions.
  * - ``cvs extension --plug PKGS``
    - Add comma-separated package name(s) to the plug-list.
  * - ``cvs extension --unplug PKGS``
    - Remove comma-separated package name(s) from the plug-list.
