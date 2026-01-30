"""
PyTorch XDit Flux output parser.

Locates and parses timing.json from Flux inference runs.
Computes average pipe_time and validates against thresholds.

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
class FluxBenchmarkResult:
    """Parsed Flux benchmark results."""

    avg_pipe_time_s: float
    repetition_count: int
    pipe_times: List[float]
    timing_json_path: str
    image_paths: List[str]


class FluxOutputParser:
    """
    Parser for PyTorch XDit Flux benchmark outputs.

    Handles:
    - Locating results/timing.json file
    - Parsing numeric pipe_time fields from JSON list
    - Computing averages
    - Locating generated images (flux_*.png)
    - Validating against GPU-specific thresholds
    """

    def __init__(self, output_dir: str, expected_image_pattern: str = "flux_*.png"):
        """
        Initialize parser.

        Args:
            output_dir: Base output directory containing benchmark results
            expected_image_pattern: Glob pattern for generated images (default: flux_*.png)
        """
        self.output_dir = Path(output_dir)
        self.expected_image_pattern = expected_image_pattern

    def find_timing_json(self) -> Optional[Path]:
        """
        Locate timing.json file under output_dir/results/.

        Returns:
            Path to timing.json if found, None otherwise
        """
        # Expected location: output_dir/results/timing.json
        timing_json = self.output_dir / "results" / "timing.json"

        if timing_json.exists():
            log.info(f"Found timing.json: {timing_json}")
            return timing_json

        # Fallback: search recursively
        for root, dirs, files in os.walk(self.output_dir):
            if "timing.json" in files:
                timing_path = Path(root) / "timing.json"
                log.info(f"Found timing.json (fallback search): {timing_path}")
                return timing_path

        log.warning(f"timing.json not found under {self.output_dir}")
        return None

    def parse_timing_json(self, timing_json: Path) -> Tuple[List[float], List[str]]:
        """
        Parse pipe_time from timing.json.

        The timing.json is expected to be a JSON list where each entry has a 'pipe_time' field.

        Args:
            timing_json: Path to timing.json file

        Returns:
            Tuple of (pipe_times, error_messages)
            - pipe_times: List of numeric pipe_time values
            - error_messages: List of parse errors (empty if all successful)
        """
        pipe_times = []
        errors = []

        try:
            with open(timing_json, 'r') as f:
                data = json.load(f)

            # Expect a list of timing entries
            if not isinstance(data, list):
                errors.append(f"timing.json: expected a JSON list, got {type(data).__name__}")
                return pipe_times, errors

            for i, entry in enumerate(data):
                if not isinstance(entry, dict):
                    errors.append(f"timing.json[{i}]: expected dict, got {type(entry).__name__}")
                    continue

                # Extract pipe_time field
                if "pipe_time" not in entry:
                    errors.append(f"timing.json[{i}]: missing 'pipe_time' field")
                    continue

                pipe_time = entry["pipe_time"]

                # Ensure it's numeric
                if not isinstance(pipe_time, (int, float)):
                    errors.append(f"timing.json[{i}]: pipe_time is not numeric (got {type(pipe_time).__name__})")
                    continue

                pipe_times.append(float(pipe_time))
                log.debug(f"timing.json[{i}]: pipe_time = {pipe_time:.2f}s")

        except json.JSONDecodeError as e:
            errors.append(f"timing.json: JSON parse error - {e}")
        except Exception as e:
            errors.append(f"timing.json: unexpected error - {e}")

        return pipe_times, errors

    def find_images(self) -> List[Path]:
        """
        Locate generated image files under output_dir.

        Returns:
            List of Path objects pointing to generated images
        """
        image_paths = []

        # Expected location: output_dir/results/flux_*.png
        results_dir = self.output_dir / "results"
        if results_dir.exists():
            image_paths.extend(results_dir.glob(self.expected_image_pattern))

        # Fallback: search recursively
        if not image_paths:
            for root, dirs, files in os.walk(self.output_dir):
                root_path = Path(root)
                for pattern_match in root_path.glob(self.expected_image_pattern):
                    if pattern_match.is_file():
                        image_paths.append(pattern_match)

        # Sort for consistent ordering
        image_paths.sort()

        log.info(f"Found {len(image_paths)} generated images under {self.output_dir}")
        for img_path in image_paths:
            log.debug(f"  - {img_path}")

        return image_paths

    def parse(self) -> Tuple[Optional[FluxBenchmarkResult], List[str]]:
        """
        Parse Flux benchmark output directory.

        Returns:
            Tuple of (result, errors)
            - result: FluxBenchmarkResult if parsing succeeded, None otherwise
            - errors: List of error messages (empty if all successful)
        """
        all_errors = []

        # Find timing.json
        timing_json = self.find_timing_json()
        if not timing_json:
            all_errors.append(f"timing.json not found under {self.output_dir}")
            return None, all_errors

        # Parse timing.json
        pipe_times, parse_errors = self.parse_timing_json(timing_json)
        all_errors.extend(parse_errors)

        if not pipe_times:
            all_errors.append("No valid pipe_time values extracted from timing.json")
            return None, all_errors

        # Compute average
        avg_pipe_time_s = sum(pipe_times) / len(pipe_times)
        log.info(f"Average pipe_time: {avg_pipe_time_s:.2f}s (from {len(pipe_times)} repetitions)")

        # Find generated images
        image_paths = self.find_images()
        if not image_paths:
            all_errors.append(f"No images matching '{self.expected_image_pattern}' found under {self.output_dir}")

        result = FluxBenchmarkResult(
            avg_pipe_time_s=avg_pipe_time_s,
            repetition_count=len(pipe_times),
            pipe_times=pipe_times,
            timing_json_path=str(timing_json),
            image_paths=[str(p) for p in image_paths],
        )

        return result, all_errors

    def validate_threshold(
        self, result: FluxBenchmarkResult, expected_results: Dict[str, Dict[str, float]], gpu_type: str = "auto"
    ) -> Tuple[bool, str]:
        """
        Validate benchmark result against expected threshold.

        Args:
            result: Parsed benchmark result
            expected_results: Dict mapping GPU types to thresholds
                             Example: {"mi300x": {"max_avg_pipe_time_s": 5.0}, "auto": {...}}
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

        max_avg_time = threshold_dict.get("max_avg_pipe_time_s")
        if max_avg_time is None:
            return False, f"Threshold missing 'max_avg_pipe_time_s' for GPU type '{gpu_type}'"

        # Compare average time to threshold
        if result.avg_pipe_time_s <= max_avg_time:
            message = (
                f"PASS: Average pipe_time {result.avg_pipe_time_s:.2f}s <= "
                f"threshold {max_avg_time:.2f}s (GPU: {gpu_type})"
            )
            log.info(message)
            return True, message
        else:
            message = (
                f"FAIL: Average pipe_time {result.avg_pipe_time_s:.2f}s > "
                f"threshold {max_avg_time:.2f}s (GPU: {gpu_type})"
            )
            log.error(message)
            return False, message
