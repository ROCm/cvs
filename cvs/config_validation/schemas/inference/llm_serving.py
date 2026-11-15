'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

``LlmServingConfig``, the base that vLLM, SGLang and ATOM configs extend, with
its server and benchmark blocks.

Only fields every serving engine reads are typed. ``engine_args`` and
``client_args`` hold the server's and the benchmark client's own flags, kept as
written for the engine to render; ``_`` keys inside them are not dropped.
'''

from typing import Any, Dict

from pydantic import Field

from cvs.config_validation.schemas.base import ConfigStr, StrictBlock
from cvs.config_validation.schemas.common_training_inference import WorkloadConfig
from cvs.config_validation.schemas.inference.accuracy import Accuracy
from cvs.config_validation.schemas.sweeps import Cells
from cvs.config_validation.schemas.thresholds import ThresholdFile, Thresholds


class ServingServerParams(StrictBlock):
    model: ConfigStr
    tp: int
    pp: int = 1
    dp: int = 1
    ep: int = 1
    engine_args: Dict[str, Any] = Field(default_factory=dict)


class ServingBenchmarkParams(StrictBlock):
    num_prompts: int
    client_args: Dict[str, Any] = Field(default_factory=dict)


class LlmServingConfig(Accuracy, Cells, Thresholds, WorkloadConfig):
    """An LLM serving suite's config. An engine subclass sets ``axes`` and adds its own fields.

    The mixins come before ``WorkloadConfig``, so a loader hook one of them
    defines overrides ``SuiteConfig``'s default. The loader fills ``thresholds``
    from the file ``threshold_json`` names.
    """

    thresholds: ThresholdFile = Field(default_factory=ThresholdFile)
    server_params: ServingServerParams
    benchmark_params: ServingBenchmarkParams
