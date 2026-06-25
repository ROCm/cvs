'''

Copyright 2025 Advanced Micro Devices, Inc.

All rights reserved.



CVS suite reports: HTML/JSON dashboards bundled with pytest-html output.



Generic engines live in ``inference.py``, ``training.py``, ``parity/``, and

``panels/``. Per-suite bindings belong in ``presets/`` and are registered from

suite ``conftest.py`` (not re-exported here).

'''



from cvs.lib.report.inference import (

    build_inference_report_payload,

    make_report_test,

    publish_inference_suite_report,

    render_report_embed_html,

    render_report_html,

    write_report,

)

from cvs.lib.report.parity.inference import (

    build_inference_parity_config,

    build_inference_parity_payload,

    build_session_parity_config,

    default_parity_metrics,

    publish_inference_parity_report,

    write_inference_parity_report,

)

from cvs.lib.report.registry import (

    bind_session_results,

    get_session_results,

    get_suite_report_config,

    register_suite_report,

)

from cvs.lib.report.training import publish_training_suite_report, write_training_report

from cvs.lib.report.types import (

    InferenceParityConfig,

    InferenceParityMetric,

    InferenceParitySource,

    InferenceReportConfig,

    ReportChartSeries,

    TrainingReportConfig,

)



__all__ = [

    "InferenceParityConfig",

    "InferenceParityMetric",

    "InferenceParitySource",

    "InferenceReportConfig",

    "TrainingReportConfig",

    "ReportChartSeries",
    "bind_session_results",

    "build_inference_parity_config",

    "build_inference_parity_payload",

    "build_inference_report_payload",

    "build_session_parity_config",

    "default_parity_metrics",

    "get_session_results",

    "get_suite_report_config",

    "make_report_test",

    "publish_inference_parity_report",

    "publish_inference_suite_report",

    "register_suite_report",

    "render_report_embed_html",

    "render_report_html",

    "write_inference_parity_report",

    "write_report",

    "write_training_report",

    "publish_training_suite_report",

]


