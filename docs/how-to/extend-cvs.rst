.. meta::
  :description: Extend CVS with custom tests and configuration through extension packages
  :keywords: CVS, extension, plugin, custom tests, entry points, cvs.extensions

***************************
Extend CVS with custom tests
***************************

CVS can be extended with additional test suites and input/config files that live
outside the core ``cvs`` package. An *extension* is a Python package that ships
its own ``tests`` and ``input`` directories; CVS discovers it at runtime and
merges its tests into ``cvs list``, its configs into ``cvs copy-config``, and its
version into ``cvs --version`` -- without forking core CVS.

For the field-by-field specification, see :doc:`Extensions </reference/extensions>`.

Two ways to package an extension
================================

There are two usage patterns, described in the repo-root ``extension.ini.sample``:

- **Pattern 1 -- bundled**: the extension copies its ``extension.ini`` into the
  installed ``cvs`` package directory during its own ``setup.py`` install. CVS
  reads ``site-packages/cvs/extension.ini`` at runtime.
- **Pattern 2 -- standalone package**: the extension is its own importable package
  (pip-installed or extracted into ``site-packages``) that ships an
  ``extension.ini``. You register it with CVS in any of three ways (originally
  only the environment variable; the entry point and plug-list were added):

  a) a pip **entry point** in the ``cvs.extensions`` group (recommended,
     auto-discovered);
  b) the persistent **plug-list** via ``cvs extension --plug``;
  c) the ``CVS_EXTENSION_PKG_NAMES`` environment variable (ephemeral).

All discovered extensions are combined and de-duplicated by package name; the
same package registered more than once is listed once.

.. note::

  Discovery is bounded by the ``cvs`` interpreter's ``sys.path``. Extensions
  installed into the *same* environment (or importable via ``PYTHONPATH``) are
  found; an extension living in a *different* virtual environment is not.

Author a pip-installable extension (Pattern 2a)
===============================================

This is the recommended approach. A minimal extension package looks like:

.. code:: text

  my_cvs_ext/
    my_cvs_ext/
      __init__.py
      extension.ini
      version.txt
      tests/
        test_my_check.py
      input/
        config_file/
          my_config.json
    pyproject.toml

Declare the entry point in ``pyproject.toml`` so CVS auto-discovers the package:

.. code:: toml

  [project.entry-points."cvs.extensions"]
  my_cvs_ext = "my_cvs_ext"

The setuptools/``setup.py`` equivalent is:

.. code:: python

  entry_points={"cvs.extensions": ["my_cvs_ext = my_cvs_ext"]}

Ship an ``extension.ini`` in the package that points at your directories (paths
are relative to the site-packages root):

.. code:: ini

  [extensions]
  package_name = my_cvs_ext
  tests_dirs = my_cvs_ext/tests
  input_dirs = my_cvs_ext/input

CVS resolves ``tests_dirs`` / ``input_dirs`` from pip metadata first (an
entry-point object exposing those attributes) and falls back to ``extension.ini``.
The version comes from ``importlib.metadata.version()``, falling back to
``version.txt`` in the package directory, then ``"unknown"``.

After ``pip install my_cvs_ext`` into the same environment as CVS, it is
discovered automatically -- no further steps.

Register a non-pip (tar-extracted) extension (Pattern 2b/2c)
============================================================

If you cannot pip-install the extension (for example, a tarball extracted into
``site-packages``), register it by name with the plug-list:

.. code:: bash

  cvs extension --plug my_cvs_ext

Or set the environment variable for a single invocation (handy in CI):

.. code:: bash

  CVS_EXTENSION_PKG_NAMES=my_cvs_ext cvs list

The plug-list is stored per-environment at ``<sys.prefix>/etc/cvs/extensions.txt``
(one package name per line). The package must still be importable by the CVS
interpreter and ship its own ``extension.ini`` (and ``version.txt``).

Manage and verify extensions
============================

List discovered extensions, the sources that referenced each one, and versions:

.. code:: bash

  cvs extension --list

.. code:: text

  Discovered Extensions
  ================================================================================
    my_cvs_ext: 1.0.0  (sources: metadata, plugged)
  --------------------------------------------------------------------------------
  Plug-list file: /path/to/venv/etc/cvs/extensions.txt

Register or unregister package names in the persistent plug-list:

.. code:: bash

  cvs extension --plug ext1,ext2
  cvs extension --unplug ext2

Once discovered, an extension appears everywhere core content does:

.. code:: bash

  cvs --version      # prints core cvs plus each extension's version
  cvs list           # extension tests are bucketed under their package name
  cvs copy-config    # extension input/config files are listed and copyable

.. note::

  If ``<sys.prefix>`` is read-only (for example, a system Python or a managed
  container image), ``cvs extension --plug`` prints an error and points you at
  ``CVS_EXTENSION_PKG_NAMES`` as an escape hatch. Everyday read-only commands
  (``--version``, ``list``, ``run``, ``copy-config``) are unaffected.

See also
========

- :doc:`Extensions </reference/extensions>` -- formal reference for every key,
  source, and file.
- :doc:`Run tests </how-to/run-cvs-tests>` -- running the tests once discovered.
