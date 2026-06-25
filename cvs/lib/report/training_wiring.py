'''Generic pytest wiring for training suite reports (Phase F).'''

from __future__ import annotations

from cvs.lib.report.registry import bind_session_results, register_suite_report
from cvs.lib.report.types import TrainingReportConfig


def configure_training_suite_report(pytest_config, preset: TrainingReportConfig) -> None:
    register_suite_report(pytest_config, preset)


def bind_training_suite_report_session(
    *,
    training_res_dict,
    variant_config=None,
) -> None:
    bind_session_results(
        training_res_dict=training_res_dict,
        variant_config=variant_config,
    )
