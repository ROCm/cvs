'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Accuracy-evaluation config schema, shared across inference suites.

`AccuracyTask` describes an lm-evaluation-harness 0.4.12 invocation against a
live OpenAI-compatible server. CVS owns the live-server endpoint, model path,
output path, and sample logging. Every other supported ``lm_eval run`` setting
has an explicit field and its 0.4.12 default.
'''

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from cvs.lib.utils.config_loader import _Forbid


class AccuracyTask(_Forbid):
    id: str
    # ``task`` remains accepted so existing Atom and vLLM configs need no
    # migration. ``tasks`` is the lm-eval 0.4.12 spelling and supports a
    # comma-separated string or a list of task names.
    tasks: Optional[Union[str, List[str]]] = None
    task: Optional[str] = None

    # CVS/OpenAI adapter selection. These are deliberately narrower than
    # lm-eval's generic --model field because CVS always targets a live local
    # OpenAI-compatible server.
    backend: str = ""
    lm_eval_model: Optional[Literal["local-completions", "local-chat-completions"]] = None
    extra_model_args: str = ""
    num_concurrent: int = 8

    # lm-eval 0.4.12 evaluation settings.
    num_fewshot: Optional[int] = None
    batch_size: str = "1"
    max_batch_size: Optional[int] = None
    device: Optional[str] = "cuda:0"
    gen_kwargs: Dict[str, Any] = Field(default_factory=dict)
    limit: Optional[float] = None
    samples: Optional[Union[str, Dict[str, Any]]] = None
    use_cache: Optional[str] = None
    cache_requests: Optional[Literal["true", "refresh", "delete"]] = None
    check_integrity: bool = False
    system_instruction: Optional[str] = None
    apply_chat_template: Union[bool, str] = False
    fewshot_as_multiturn: Optional[bool] = None
    include_path: Optional[str] = None
    predict_only: bool = False
    seed: List[Optional[int]] = Field(default_factory=lambda: [0, 1234, 1234, 1234])
    trust_remote_code: bool = False
    confirm_run_unsafe_code: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # CVS controls execution duration, not lm-eval.
    exec_timeout_sec: int = 4 * 60 * 60

    @field_validator("apply_chat_template", mode="before")
    @classmethod
    def _coerce_boolean_template_values(cls, value):
        if isinstance(value, str) and value.lower() in {"true", "false", "0", "1"}:
            return value.lower() in {"true", "1"}
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def _normalize_seed(cls, value):
        defaults = [0, 1234, 1234, 1234]
        if isinstance(value, int):
            return [value] * 4
        if isinstance(value, str):
            value = value.split(",")
        if isinstance(value, list):
            normalized = [None if item is None or item == "None" else int(item) for item in value]
            if len(normalized) == 1:
                return normalized * 4
            if len(normalized) == 3:
                return [*normalized, defaults[3]]
            return normalized
        return value

    @field_validator("tasks", mode="before")
    @classmethod
    def _reject_empty_task_lists(cls, value):
        if value == []:
            raise ValueError("tasks must not be empty")
        return value

    @field_validator("seed")
    @classmethod
    def _validate_seed_count(cls, value):
        if len(value) != 4:
            raise ValueError("seed must contain exactly four values: python,numpy,torch,fewshot")
        return value

    @model_validator(mode="after")
    def _resolve_legacy_task_alias(self):
        if self.tasks is None and self.task is None:
            raise ValueError("one of tasks or task is required")
        if self.tasks is None:
            self.tasks = self.task
        elif self.task is not None and self.task_names() != self._split_task_string(self.task):
            raise ValueError("task and tasks disagree")
        # Preserve the historical public attribute for Atom and callers that
        # consume a single comma-separated task selector.
        task_names = self.task_names()
        if not task_names:
            raise ValueError("tasks must contain at least one task name")
        self.task = ",".join(task_names)
        return self

    @staticmethod
    def _split_task_string(value: str) -> List[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    def task_names(self) -> List[str]:
        if isinstance(self.tasks, list):
            return self.tasks
        return self._split_task_string(self.tasks or "")

    @property
    def resolved_lm_eval_model(self) -> str:
        if self.lm_eval_model:
            return self.lm_eval_model
        return "local-chat-completions" if self.apply_chat_template else "local-completions"


class AccuracyConfig(_Forbid):
    tasks: List[AccuracyTask] = []

    @model_validator(mode="after")
    def _check_unique_task_ids(self):
        from collections import Counter

        counts = Counter(t.id for t in self.tasks)
        dupes = sorted(i for i, n in counts.items() if n > 1)
        if dupes:
            rendered = ", ".join(repr(d) for d in dupes)
            raise ValueError(f"duplicate task id(s): {rendered}")
        return self
