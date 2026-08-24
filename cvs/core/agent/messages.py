'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from enum import Enum
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError, model_validator

# Auth constants
AUTH_HEADER = "Authorization"
AUTH_SCHEME = "Bearer"
# Bootstrap token file under RunLayout's agent_dir (mode 0600). The token is static for the
# job's lifetime, so callers should read it once at process startup and cache it in memory
# rather than re-reading it per request.
AUTH_TOKEN_FILENAME = "secret"

# Path constants
REGISTER_PATH = "/v1/register"
EXEC_PATH = "/v1/exec"
SHUTDOWN_PATH = "/v1/shutdown"
HEALTH_PATH = "/v1/health"

# Limits constants
MAX_INLINE_RESPONSE_BYTES = 4 * 1024 * 1024  # INLINE output beyond this is truncated


class RegisterRequest(BaseModel):
    '''Data model for a agent to register itself with the server'''

    rank: int = Field(ge=0)
    hostname: str
    port: int = Field(gt=0, le=65535)


class RegisterResponse(BaseModel):
    '''Response regarding agents registration request'''

    ok: bool


class ExecOutputMode(str, Enum):
    '''Selects how ExecResponse reports captured output'''

    INLINE = "inline"
    FILE = "file"
    EXIT_CODE_ONLY = "exit_code_only"


class ExecRequest(BaseModel):
    cmd: str
    env: dict[str, str]
    cwd: Path
    timeout: int | None  # some tests / benchmarks are long running require None timeout
    inactivity_timeout: int | None  # None disables the inactivity bound
    cmd_id: str
    out_path: Path | None  # directory for cmd_id.stdout/cmd_id.stderr; required when output_mode is FILE
    output_mode: ExecOutputMode

    @model_validator(mode="after")
    def _out_path_required_for_file_mode(self):
        if self.output_mode == ExecOutputMode.FILE and self.out_path is None:
            raise ValueError("out_path is required when output_mode is FILE")
        return self


class ExecResponse(BaseModel):
    '''
    Successful remote cmd execution (regardless of the cmd result) returns ExecResponse.
    Field applicability depends on the request's output_mode:
      INLINE: stdout/stderr hold the full captured output; truncated reflects MAX_INLINE_RESPONSE_BYTES
      FILE: stdout/stderr hold a trailing preview only; stdout_path/stderr_path hold the full output on shared FS
      EXIT_CODE_ONLY: only exit_code is set, every other field is None
    '''

    exit_code: int
    stdout: list[str] | None
    stderr: list[str] | None
    stdout_path: Path | None
    stderr_path: Path | None
    truncated: bool | None  # set for INLINE mode only; None when not applicable


class ErrorResponse(BaseModel):
    '''Agent / Transport failure returns error response'''

    errors: list[str]


class HealthResponse(BaseModel):
    '''Check status of current agent'''

    ok: bool


class ShutdownRequest(BaseModel):
    '''Request agent shutdown'''


class ShutdownResponse(BaseModel):
    '''Respond to shutdown requests'''

    ok: bool


# Upload/Download models are not implemented as the ranks can directly read and write data into the shared FS
# Serialization and Deserialization would be done by pydantic - .model_dump_json()/.model_validate_json()

T = TypeVar("T", bound=BaseModel)


class MessageParseError(Exception):
    '''Raised when an inbound request/response body fails to validate against its schema'''


def parse_message(model_cls: type[T], raw: str) -> T:
    '''Deserialize raw JSON into model_cls, wrapping pydantic's ValidationError uniformly'''
    try:
        return model_cls.model_validate_json(raw)
    except ValidationError as exc:
        raise MessageParseError(str(exc)) from exc
