'''Inference framework parity reports.'''

from cvs.lib.report.parity.inference import (
    STANDARD_CLIENT_PARITY_METRICS,
    build_inference_parity_config,
    build_inference_parity_payload,
    build_session_parity_config,
    default_parity_metrics,
    load_framework_cell_indexes,
    main,
    publish_inference_parity_report,
    render_inference_parity_html,
    write_inference_parity_report,
)
from cvs.lib.report.parity.session import (
    publish_session_inference_parity,
    resolve_parity_compare_jsons,
)

__all__ = [
    "STANDARD_CLIENT_PARITY_METRICS",
    "build_inference_parity_config",
    "build_inference_parity_payload",
    "build_session_parity_config",
    "default_parity_metrics",
    "load_framework_cell_indexes",
    "main",
    "publish_inference_parity_report",
    "publish_session_inference_parity",
    "render_inference_parity_html",
    "resolve_parity_compare_jsons",
    "write_inference_parity_report",
]
