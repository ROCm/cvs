'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Config blocks every CVS suite schema is built from, re-exported from their
home in ``cvs.parsers.schemas`` so existing imports keep working.
'''

from cvs.parsers.schemas import (  # noqa: F401 - re-exported for suite schemas
    PLACEHOLDER_MESSAGE,
    _PLACEHOLDER_RE,
    ConfigStr,
    ContainerSpec,
    FiniteNumber,
    MinRatioThreshold,
    ModelSpec,
    Paths,
    RuntimeArgs,
    RuntimeSpec,
    ThresholdCell,
    ThresholdFile,
    ThresholdSpec,
    ValueThreshold,
    WithinThreshold,
    _Allow,
    _drop_comment_keys,
    _Forbid,
    _reject_placeholder,
    has_placeholder,
    placeholders_rejected,
)
