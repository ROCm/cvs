"""
Base runner interface and common data structures.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import logging

log = logging.getLogger(__name__)


class RunStatus(Enum):
    """Status of a benchmark run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class RunConfig:
    """
    Base configuration for all runners.

    Specific runners should extend this with their own config dataclasses.
    """

    # Cluster configuration
    nodes: List[str]
    username: str
    pkey: Optional[str] = None
    password: Optional[str] = None

    # Execution settings
    timeout_seconds: int = 3600  # 1 hour default
    work_dir: Path = field(default_factory=lambda: Path("/tmp"))
    output_dir: Path = field(default_factory=lambda: Path("/tmp/benchmark_output"))

    # Additional environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)

    # Metadata for tracking
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """
    Result from a benchmark runner.

    Contains raw outputs and artifact paths - no parsed/validated data.
    Parsing is the responsibility of the parsers layer.
    """

    status: RunStatus
    start_time: float
    end_time: float

    # Raw outputs from execution
    stdout: Dict[str, str] = field(default_factory=dict)  # node -> stdout
    stderr: Dict[str, str] = field(default_factory=dict)  # node -> stderr

    # Paths to output artifacts
    artifacts: Dict[str, Path] = field(default_factory=dict)  # name -> path

    # Error information if failed
    error_message: Optional[str] = None
    exit_codes: Dict[str, int] = field(default_factory=dict)  # node -> exit code

    # Metadata collected during run
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Total execution time in seconds."""
        return self.end_time - self.start_time

    @property
    def succeeded(self) -> bool:
        """Whether the run completed successfully."""
        return self.status == RunStatus.COMPLETED

    def get_artifact(self, name: str) -> Optional[Path]:
        """Get artifact path by name, returns None if not found."""
        return self.artifacts.get(name)


class BaseRunner(ABC):
    """
    Abstract base class for benchmark runners.

    Runners follow a lifecycle:
    1. setup() - Prepare environment (deploy containers, install deps)
    2. run() - Execute the benchmark
    3. teardown() - Cleanup resources

    The execute() method orchestrates this lifecycle with proper error handling.
    """

    def __init__(self, config: RunConfig):
        """
        Initialize runner with configuration.

        Args:
            config: Runner configuration
        """
        self.config = config
        self._setup_complete = False

    @property
    def head_node(self) -> str:
        """First node in the cluster, typically used for orchestration."""
        if not self.config.nodes:
            raise ValueError("No nodes configured")
        return self.config.nodes[0]

    @abstractmethod
    def setup(self) -> bool:
        """
        Prepare the environment before benchmark execution.

        This may include:
        - Deploying containers
        - Installing dependencies
        - Copying files
        - Setting up networking

        Returns:
            True if setup succeeded, False otherwise
        """
        pass

    @abstractmethod
    def run(self, **kwargs) -> RunResult:
        """
        Execute the benchmark.

        Args:
            **kwargs: Benchmark-specific parameters

        Returns:
            RunResult with raw outputs and artifact paths
        """
        pass

    @abstractmethod
    def teardown(self) -> bool:
        """
        Cleanup resources after benchmark execution.

        This may include:
        - Stopping containers
        - Removing temporary files
        - Releasing resources

        Returns:
            True if teardown succeeded, False otherwise
        """
        pass

    def execute(self, **kwargs) -> RunResult:
        """
        Full execution lifecycle: setup -> run -> teardown.

        Handles errors at each stage and ensures teardown is always called.

        Args:
            **kwargs: Passed to run()

        Returns:
            RunResult from the run, or a failed result if setup fails
        """
        start_time = time.time()

        try:
            # Setup phase
            log.info(f"Setting up {self.__class__.__name__}...")
            if not self.setup():
                return RunResult(
                    status=RunStatus.FAILED, start_time=start_time, end_time=time.time(), error_message="Setup failed"
                )
            self._setup_complete = True

            # Run phase
            log.info(f"Running {self.__class__.__name__}...")
            result = self.run(**kwargs)
            return result

        except Exception as e:
            log.exception(f"Error during {self.__class__.__name__} execution")
            return RunResult(status=RunStatus.FAILED, start_time=start_time, end_time=time.time(), error_message=str(e))

        finally:
            # Always attempt teardown
            if self._setup_complete:
                log.info(f"Tearing down {self.__class__.__name__}...")
                try:
                    self.teardown()
                except Exception as e:
                    log.warning(f"Teardown error (non-fatal): {e}")

    def validate_config(self) -> List[str]:
        """
        Validate runner configuration.

        Override in subclasses to add specific validation.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.config.nodes:
            errors.append("No nodes configured")

        if not self.config.username:
            errors.append("No username configured")

        if not self.config.pkey and not self.config.password:
            errors.append("No authentication method configured (pkey or password)")

        return errors
