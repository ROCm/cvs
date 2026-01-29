"""
PyTorch XDit WAN output parser.

Locates and parses benchmark JSONs (rank0_step*.json) from WAN inference runs.
Computes average total_time and validates against thresholds.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cvs.lib import globals

log = globals.log


@dataclass
class WanBenchmarkResult:
    """Parsed WAN benchmark results."""

    avg_total_time_s: float
    step_count: int
    step_times: List[float]
    json_files: List[str]
    artifact_path: Optional[str] = None


class WanOutputParser:
    """
    Parser for PyTorch XDit WAN benchmark outputs.

    Handles:
    - Locating rank0_step*.json files (including nested outputs/outputs/outputs/ layouts)
    - Parsing numeric total_time fields
    - Computing averages
    - Locating generated artifacts (video.mp4)
    - Validating against GPU-specific thresholds
    """

    def __init__(self, output_dir: str, expected_artifact: str = "video.mp4"):
        """
        Initialize parser.

        Args:
            output_dir: Base output directory containing benchmark results
            expected_artifact: Artifact filename to locate (default: video.mp4)
        """
        self.output_dir = Path(output_dir)
        self.expected_artifact = expected_artifact

    def find_benchmark_jsons(self) -> List[Path]:
        """
        Locate all rank0_step*.json files under output_dir.

        Handles nested outputs/outputs/outputs/ layouts that can occur
        when output directories are mounted recursively.

        Returns:
            List of Path objects pointing to benchmark JSON files
        """
        json_files = []

        # Search recursively for rank0_step*.json files
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file.startswith("rank0_step") and file.endswith(".json"):
                    json_path = Path(root) / file
                    json_files.append(json_path)

        # Sort by filename for consistent ordering
        json_files.sort()

        log.info(f"Found {len(json_files)} benchmark JSON files under {self.output_dir}")
        for json_file in json_files:
            log.debug(f"  - {json_file}")

        return json_files

    def parse_benchmark_jsons(self, json_files: List[Path]) -> Tuple[List[float], List[str]]:
        """
        Parse total_time from benchmark JSON files.

        Args:
            json_files: List of JSON file paths to parse

        Returns:
            Tuple of (step_times, error_messages)
            - step_times: List of numeric total_time values
            - error_messages: List of parse errors (empty if all successful)
        """
        step_times = []
        errors = []

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Extract total_time field
                if "total_time" not in data:
                    errors.append(f"{json_file.name}: missing 'total_time' field")
                    continue

                total_time = data["total_time"]

                # Ensure it's numeric
                if not isinstance(total_time, (int, float)):
                    errors.append(f"{json_file.name}: total_time is not numeric (got {type(total_time).__name__})")
                    continue

                step_times.append(float(total_time))
                log.debug(f"{json_file.name}: total_time = {total_time:.2f}s")

            except json.JSONDecodeError as e:
                errors.append(f"{json_file.name}: JSON parse error - {e}")
            except Exception as e:
                errors.append(f"{json_file.name}: unexpected error - {e}")

        return step_times, errors

    def find_artifact(self) -> Optional[Path]:
        """
        Locate the expected artifact file under output_dir.

        Returns:
            Path to artifact if found, None otherwise
        """
        # Search recursively for the artifact
        for root, dirs, files in os.walk(self.output_dir):
            if self.expected_artifact in files:
                artifact_path = Path(root) / self.expected_artifact
                log.info(f"Found artifact: {artifact_path}")
                return artifact_path

        log.warning(f"Artifact '{self.expected_artifact}' not found under {self.output_dir}")
        return None

    def parse(self) -> Tuple[Optional[WanBenchmarkResult], List[str]]:
        """
        Parse WAN benchmark output directory.

        Returns:
            Tuple of (result, errors)
            - result: WanBenchmarkResult if parsing succeeded, None otherwise
            - errors: List of error messages (empty if all successful)
        """
        all_errors = []

        # Find benchmark JSONs
        json_files = self.find_benchmark_jsons()
        if not json_files:
            all_errors.append(f"No rank0_step*.json files found under {self.output_dir}")
            return None, all_errors

        # Parse JSONs
        step_times, parse_errors = self.parse_benchmark_jsons(json_files)
        all_errors.extend(parse_errors)

        if not step_times:
            all_errors.append("No valid total_time values extracted from JSON files")
            return None, all_errors

        # Compute average
        avg_total_time_s = sum(step_times) / len(step_times)
        log.info(f"Average total_time: {avg_total_time_s:.2f}s (from {len(step_times)} steps)")

        # Find artifact
        artifact_path = self.find_artifact()

        result = WanBenchmarkResult(
            avg_total_time_s=avg_total_time_s,
            step_count=len(step_times),
            step_times=step_times,
            json_files=[str(f) for f in json_files],
            artifact_path=str(artifact_path) if artifact_path else None,
        )

        return result, all_errors

    def validate_threshold(
        self, result: WanBenchmarkResult, expected_results: Dict[str, Dict[str, float]], gpu_type: str = "auto"
    ) -> Tuple[bool, str]:
        """
        Validate benchmark result against expected threshold.

        Args:
            result: Parsed benchmark result
            expected_results: Dict mapping GPU types to thresholds
                             Example: {"mi300x": {"max_avg_total_time_s": 10.5}, "auto": {...}}
            gpu_type: GPU type to use for threshold lookup (default: "auto")

        Returns:
            Tuple of (passed, message)
            - passed: True if benchmark met threshold, False otherwise
            - message: Human-readable pass/fail message
        """
        # Select threshold based on GPU type (fallback to 'auto')
        if gpu_type in expected_results:
            threshold_dict = expected_results[gpu_type]
            log.info(f"Using GPU-specific threshold for '{gpu_type}'")
        elif "auto" in expected_results:
            threshold_dict = expected_results["auto"]
            log.info(f"Using 'auto' threshold (no specific threshold for '{gpu_type}')")
        else:
            return False, f"No threshold found for GPU type '{gpu_type}' and no 'auto' fallback"

        max_avg_time = threshold_dict.get("max_avg_total_time_s")
        if max_avg_time is None:
            return False, f"Threshold missing 'max_avg_total_time_s' for GPU type '{gpu_type}'"

        # Compare average time to threshold
        if result.avg_total_time_s <= max_avg_time:
            message = (
                f"PASS: Average total_time {result.avg_total_time_s:.2f}s <= "
                f"threshold {max_avg_time:.2f}s (GPU: {gpu_type})"
            )
            log.info(message)
            return True, message
        else:
            message = (
                f"FAIL: Average total_time {result.avg_total_time_s:.2f}s > "
                f"threshold {max_avg_time:.2f}s (GPU: {gpu_type})"
            )
            log.error(message)
            return False, message
