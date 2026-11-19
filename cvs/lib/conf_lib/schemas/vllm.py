'''vLLM config schema, re-exported from cvs.parsers.schemas for existing callers.'''

from cvs.parsers.schemas import (  # noqa: F401 - re-exported for callers
    ContainerEnvTokenError,
    ResolvedRun,
    RunCell,
    VllmBlock,
    VllmConfig,
    VllmContainerSpec,
    VllmRuntime,
    VllmRuntimeArgs,
    serialize_cli_options,
)

VariantConfig = VllmConfig
