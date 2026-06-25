'''Megatron Llama 3.1 8B single-node training report preset.'''

from __future__ import annotations

from cvs.lib.report.types import TrainingReportConfig

MEGATRON_LLAMA3_8B_SINGLE_REPORT_CONFIG = TrainingReportConfig(
    suite_id="megatron_training",
    report_basename="megatron_training_report",
    title="Megatron training report",
    subtitle="Llama 3.1 8B single-node · CVS training summary",
    footer="CVS megatron_llama3_1_8b_single · render-only",
    link_name="Megatron training report",
)
