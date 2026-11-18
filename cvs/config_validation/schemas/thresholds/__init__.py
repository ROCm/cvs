'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Threshold specs, the kind-based threshold file, and the ``Thresholds`` mixin.

A spec is a kind (``min``, ``max`` or ``info``) and a value; any other kind,
such as ``max_ms``, ``min_tok_s``, ``within`` or ``min_ratio``, fails.

``accuracy`` mirrors a config's ``accuracy`` block: one section of specs per
lm-eval task id and per custom eval id, and one each for quant parity and MTP
quality.
'''

from typing import Annotated, Dict, Literal

from pydantic import BeforeValidator, Field

from cvs.config_validation.schemas.base import OpenBlock, StrictBlock, drop_comment_keys

# Strict float still takes a JSON int, which shipped files use; a quoted number or a boolean fails.
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class Threshold(StrictBlock):
    kind: Literal["min", "max", "info"]
    value: Number
    optional: bool = False


ThresholdCell = Annotated[Dict[str, Threshold], BeforeValidator(drop_comment_keys)]


ThresholdSections = Annotated[Dict[str, ThresholdCell], BeforeValidator(drop_comment_keys)]

ACCURACY_SECTIONS = ("lm_eval", "custom_eval", "quant_parity", "mtp_quality")


class ThresholdAccuracy(StrictBlock):
    """A threshold file's ``accuracy``: specs by lm-eval task id, by custom eval id, and for quant parity and MTP."""

    lm_eval: ThresholdSections = Field(default_factory=dict)
    custom_eval: ThresholdSections = Field(default_factory=dict)
    quant_parity: ThresholdCell = Field(default_factory=dict)
    mtp_quality: ThresholdCell = Field(default_factory=dict)


class ThresholdFile(OpenBlock):
    """A threshold file: one cell of metric specs per cell key, plus ``accuracy``."""

    accuracy: ThresholdAccuracy = Field(default_factory=ThresholdAccuracy)
    __pydantic_extra__: Dict[str, ThresholdCell]


class Thresholds(StrictBlock):
    """``threshold_json`` names the suite's threshold file; an empty name records metrics without gating."""

    threshold_json: str = ""
    enforce_thresholds: bool = True
