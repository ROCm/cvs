'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

One map module per suite, named for the suite: ``maps/<suite>.py``. The suites
``cvs config migrate`` converts are the modules present here.

A map module defines:

- ``SCHEMA``: the suite's ``SuiteConfig`` subclass, which a converted config is
  validated against. Its ``suite`` is the module's name.
- ``CONFIG``: old dotted path to new dotted path, or to ``None`` to leave the
  key out. A ``*`` in an old path matches any key except a ``_`` comment, or any
  list item, and each ``*`` in the new path takes what the matching ``*`` in the
  old path matched. A list item keeps its index when it stays in the same list,
  and is appended, in order, to any other list, so ``sweep.runs.*.combo`` to
  ``cells.*`` leaves no gap for a run without a ``combo``. A path matches a
  whole value, so a key's children are not listed beside it. Each key is an old
  key: an input holding none is already in the new format, or another suite's
  file when that suite's map matches, and is not converted. The suite's
  ``old_keys`` hook can return them through ``engine.old_paths``.
- ``THRESHOLDS``: old threshold kind to new kind, renamed inside each metric
  spec of a threshold file. A spec with an old kind that is not listed, such as
  ``within`` or ``min_ratio``, goes to ``_unmapped``.
- ``FIELDS`` (optional): old dotted path, written as in ``CONFIG``, to a function
  for a field whose value decides where it goes. It is called with the field's
  value and the whole input, and returns a dict of new dotted paths, with ``*``
  filled as in ``CONFIG``, to values; an empty dict leaves the field out, and
  ``engine.TO_UNMAPPED`` moves it to ``_unmapped``. A ``FIELDS`` path is tried
  before ``CONFIG`` and is not an old key. A suite whose new schema takes
  ``cells`` in place of ``sweeps`` maps ``sweeps.*`` to a function that returns
  ``{}`` for an empty per-cell entry and ``engine.TO_UNMAPPED`` for a real
  override, since ``sweeps`` has no place in the new format.
- ``FILL_THRESHOLD_FROM_SIBLING`` (optional, default ``False``): when true, a
  converted config whose ``threshold_json`` is empty names the one
  ``*threshold.json`` beside the input, if there is exactly one. Leave it off
  for a suite that records metrics without gating when ``threshold_json`` is
  empty, since filling it would start gating.

When several paths match one key, the first written applies.
'''

import pkgutil


def suite_names():
    """The suites ``cvs config migrate`` converts, one per map module here, sorted; imports none of them."""
    return sorted(
        info.name for info in pkgutil.iter_modules(__path__) if not info.ispkg and not info.name.startswith("_")
    )
