'''Normalize session data into ``datasets.*`` structures (no HTML).'''

from cvs.lib.report.rundeck.dataset_builders import (  # noqa: F401
    matrix,
    series,
    status_matrix,
    sweep,
    training_sweep,
)
