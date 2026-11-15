'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

lm-eval accuracy tasks and long-context accuracy cells for LLM serving configs.

``AccuracyTask`` describes an lm-evaluation-harness 0.4.12 run against the live
server. CVS sets the endpoint, model, output path and sample logging; every
other ``lm_eval run`` setting has a field at its 0.4.12 default.
'''

from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from cvs.config_validation.schemas.base import StrictBlock

_DEFAULT_SEED = (0, 1234, 1234, 1234)


def _duplicate_ids(ids):
    return ", ".join(repr(item) for item, count in sorted(Counter(ids).items()) if count > 1)


class AccuracyTask(StrictBlock):
    id: str
    # One of the two is required. ``task`` is the older spelling shipped configs use;
    # ``tasks`` is lm-eval's, a comma-separated string or a list of task names.
    tasks: Optional[Union[str, List[str]]] = None
    task: Optional[str] = None

    # CVS always targets the live OpenAI-compatible server, so these are narrower than lm-eval's --model.
    backend: str = ""
    lm_eval_model: Optional[Literal["local-completions", "local-chat-completions"]] = None
    extra_model_args: str = ""
    num_concurrent: int = 8

    num_fewshot: Optional[int] = None
    batch_size: str = "1"
    max_batch_size: Optional[int] = None
    device: Optional[str] = None
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
    seed: List[Optional[int]] = Field(default_factory=lambda: list(_DEFAULT_SEED))
    trust_remote_code: bool = False
    confirm_run_unsafe_code: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    exec_timeout_sec: int = 4 * 60 * 60  # CVS's limit on the lm-eval exec, not an lm-eval option

    @field_validator("apply_chat_template", mode="before")
    @classmethod
    def _coerce_boolean_template_values(cls, value):
        if isinstance(value, str) and value.lower() in {"true", "false", "0", "1"}:
            return value.lower() in {"true", "1"}
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def _normalize_seed(cls, value):
        if isinstance(value, int):
            return [value] * 4
        if isinstance(value, str):
            value = value.split(",")
        if isinstance(value, list):
            normalized = [None if item is None or item == "None" else int(item) for item in value]
            if len(normalized) == 1:
                return normalized * 4
            if len(normalized) == 3:
                return [*normalized, _DEFAULT_SEED[3]]
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
        task_names = self.task_names()
        if not task_names:
            raise ValueError("tasks must contain at least one task name")
        # ``task`` stays readable as the comma-separated form of ``tasks``.
        self.task = ",".join(task_names)
        return self

    @staticmethod
    def _split_task_string(value):
        return [item.strip() for item in value.split(",") if item.strip()]

    def task_names(self):
        if isinstance(self.tasks, list):
            return self.tasks
        return self._split_task_string(self.tasks or "")

    @property
    def resolved_lm_eval_model(self):
        if self.lm_eval_model:
            return self.lm_eval_model
        return "local-chat-completions" if self.apply_chat_template else "local-completions"


class AccuracyConfig(StrictBlock):
    tasks: List[AccuracyTask] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_task_ids(self):
        duplicates = _duplicate_ids(task.id for task in self.tasks)
        if duplicates:
            raise ValueError(f"duplicate task id(s): {duplicates}")
        return self


class Accuracy(StrictBlock):
    """Mixin: the ``accuracy`` block. No tasks means no lm-eval run."""

    accuracy: AccuracyConfig = Field(default_factory=AccuracyConfig)


class LongContextCell(StrictBlock):
    id: str
    isl: int
    osl: int = 32
    num_prompts: int = 8
    seed: int = 42


class LongContextAccuracy(StrictBlock):
    cells: List[LongContextCell] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_cell_ids(self):
        duplicates = _duplicate_ids(cell.id for cell in self.cells)
        if duplicates:
            raise ValueError(f"duplicate long-context cell id(s): {duplicates}")
        return self
