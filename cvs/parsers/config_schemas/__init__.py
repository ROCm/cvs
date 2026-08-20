"""Per-suite pydantic schemas documenting user-editable test config files.

These models are the source of truth for ``cvs man``. Every field carries a
``description`` and, where the code supplies one, the real runtime default
taken from the ``dict.get(key, default)`` call site rather than the value that
happens to ship in the sample config -- the two frequently disagree.

Models are declared ``extra="allow"`` and are not yet wired into runtime
validation, so an existing customer config is never rejected on account of them.
"""

from .health import AgfhcConfigFile, RvsConfigFile, RvsTestConfig, TransferBenchConfigFile
from .rccl import RcclConfigFile

__all__ = [
    "AgfhcConfigFile",
    "RcclConfigFile",
    "RvsConfigFile",
    "RvsTestConfig",
    "TransferBenchConfigFile",
]
