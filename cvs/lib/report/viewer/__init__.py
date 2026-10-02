'''Interactive report viewer (JSON sidecar consumer).'''

from cvs.lib.report.viewer.scaffold import viewer_basename_for, write_interactive_viewer
from cvs.lib.report.viewer.status_matrix import write_status_matrix_viewer

__all__ = ["viewer_basename_for", "write_interactive_viewer", "write_status_matrix_viewer"]
