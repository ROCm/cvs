'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import tempfile
from enum import Enum
from pathlib import Path


STATE_FILENAME = "rank0.state"


class Rank0State(str, Enum):
    STARTING = "STARTING"
    WAITING_FOR_RANKS = "WAITING_FOR_RANKS"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class Rank0StateStore:
    '''Shared-filesystem query and update API for rank-0 state.'''

    def __init__(self, agent_dir: Path):
        self.path = agent_dir / STATE_FILENAME

    def get_state(self) -> Rank0State:
        try:
            return Rank0State(self.path.read_text(encoding="utf-8").strip())
        except OSError as exc:
            raise RuntimeError(f"cannot read rank-0 state: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"invalid rank-0 state in {self.path}") from exc

    def update_state(self, state: Rank0State) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(f"{state.value}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
        except OSError:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise
