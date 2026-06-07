# Updating end-user Sphinx docs for a new DTNI suite

End-user docs for DTNI suites live in `docs/reference/configuration-files/` and
ship to `rocm.docs.amd.com` via the ROCm docs CI. Update them whenever a new
DTNI framework/variant becomes user-visible.

## Scope

- **Update when:** a new suite, framework, or variant is exposed to end users
  for the first time (new config file in `cvs/input/dtni/<framework>/`, new
  framework adapter, new platform/topology combination users can select).
- **Skip when:** internal-only changes (refactors, new private helpers,
  threshold tweaks that don't change the public config schema, new tests).

## Files to touch

| File | Action |
|---|---|
| `docs/reference/configuration-files/<framework>_<topology>_<platform>.rst` | Create new |
| `docs/sphinx/_toc.yml.in` | Add entry under the existing Reference subtree |

That's it. No `conf.py` change, no new images, no extension to register.

## Naming convention

Observed pattern in `docs/reference/configuration-files/`:

```
<framework>_<topology>_<platform>.rst   # e.g. vllm_singlenode_mi355x.rst
<framework>_<variant>.rst               # e.g. sglang_disagg_pd.rst, flux1_t2i.rst
<framework>.rst                         # e.g. megatron.rst, jax.rst (generic)
```

Use the longest-form name your suite needs to disambiguate. If the suite is
platform- or topology-specific, include both. Filename is lowercase,
underscore-separated, no version numbers.

## RST template

Copy the structure of `vllm_singlenode_mi355x.rst`. Replace the placeholders
marked `<...>`:

```rst
.. meta::
  :description: Configure the variables in the <Suite Name> configuration file
  :keywords: inference, ROCm, install, cvs, <Framework>, <Platform>, <Topology>

*********************************************************
<Platform> <topology> <framework> <workload> configuration file
*********************************************************

<One-paragraph description of what the suite validates and why.>

The <Suite Name> tests check:

- **Container orchestration**: <what the harness brings up>
- **Model serving** / **Training**: <core workload>
- **Performance metrics**: <metric names, e.g. TTFT, TPOT, throughput>
- **Workload scenarios**: <variant names if any>
- **Result verification**: <which thresholds gate the verdict>

Change the parameters as needed in the configuration file:
``<config_filename>.json``.

.. note::

  - ``{user-id}`` will be resolved to the current username in the runtime.
  - Parameters with the ``<changeme>`` value must have that value modified
    to your specifications.

``<config_filename>.json``
==========================

Here's a code snippet of the ``<config_filename>.json`` file for reference:

.. dropdown:: ``<config_filename>.json``

  .. code:: json

    {
      "config": { ... },
      "benchmark_params": { ... }
    }

Parameters
==========

Use the parameters in this table to configure the <Suite Name> file.

.. |br| raw:: html

    <br />

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``<param>``
     - <default>
     - <what it does>
```

## Sphinx conventions used in this repo

- **Theme:** `rocm_docs_theme` (configured via the `rocm_docs` extension in
  `docs/conf.py`; the only extension loaded).
- **Code blocks:** use `.. code:: json` / `.. code:: bash` (the existing rst
  files use the short `code` form, not `code-block`).
- **Dropdowns:** `.. dropdown::` is available (provided by `rocm_docs`).
- **Tables:** `.. list-table::` with `|br|` substitution for in-cell line
  breaks (defined inline as shown in the template).
- **Cross-doc links:** use `:doc:` for other rst pages, `:ref:` for labelled
  anchors. Do not invent new label conventions.
- **No autodoc**, **no sphinx-tabs**, **no mermaid** — none of these are
  enabled. Don't add them.

### Section underline characters

Verified from `vllm_singlenode_mi355x.rst` and peers:

- **H1 (page title):** `*` overline AND underline, length >= title
- **H2 (section):** `=` underline only
- **H3:** not used in any existing config-files rst; if you need one, use `-`
  to stay consistent with Sphinx convention

## `_toc.yml.in` entry

The toc file is `docs/sphinx/_toc.yml.in`. Add your file under the existing
`Reference > Test configuration files` subtree, as a sibling of the other
configuration-files entries:

```yaml
- caption: Reference
  entries:
  - file: reference/configuration-files/configure-config
    title: Test configuration files
    subtrees:
    - entries:
      ...
      - file: reference/configuration-files/<your_new_file>
        title: <Short Title Shown in Sidebar>
```

**Ordering:** entries are NOT alphabetized — they are grouped loosely by
category (cluster/platform first, then health/network, then framework
suites). Place your new entry next to the closest sibling (e.g. a new vLLM
variant goes adjacent to `vllm_singlenode_mi355x`).

Omit the `.rst` extension in the `file:` value.

## Local verification

Local Sphinx build is best-effort. There is no `make docs` target in this
repo; the canonical build happens in ROCm CI. If you want to preview locally:

```bash
pip install rocm_docs_theme sphinx
sphinx-build -M html docs docs/_build
```

It may fail with missing extensions or theme assets (the `rocm_docs`
extension pulls a stack of theme- and CI-specific helpers). **Trust the CI
preview build on the PR** as the source of truth for rendering.

## Verification checklist

- [ ] New `.rst` file renders (best-effort local build OR CI preview job
      on the PR).
- [ ] `docs/sphinx/_toc.yml.in` entry is placed next to its closest
      category sibling (not blindly appended).
- [ ] Title appears in the sidebar under `Reference > Test configuration
      files` in the CI preview.
- [ ] All `:doc:` / `:ref:` links resolve in the CI preview (no warnings
      in the build log).
- [ ] Filename matches the `<framework>_<topology>_<platform>.rst` pattern
      (or the closest legitimate variant).
- [ ] `.. meta::` block has accurate `:description:` and `:keywords:`.
