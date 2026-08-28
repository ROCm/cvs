"""
PyTorch XDit Wan I2V (xFuser) output parser.

Locates and parses ``results/timing.json`` from Wan xFuser I2V runs, validates
``video_i2v.mp4``, and checks thresholds using Flux-style ``pipe_time`` metrics.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cvs.lib import globals
from cvs.lib.inference.pytorch_xdit.pytorch_xdit_flux import log_results_summary

log = globals.log

WAN_I2V_TIMING_JSON_REL = Path("results") / "timing.json"
WAN_I2V_VIDEO_ARTIFACT = "video_i2v.mp4"


@dataclass
class WanI2vBenchmarkResult:
    """Parsed Wan xFuser I2V benchmark results."""

    avg_pipe_time_s: float
    repetition_count: int
    pipe_times: List[float]
    timing_json_path: str
    video_path: Optional[str]


class WanI2vOutputParser:
    """
    Parser for Wan xFuser I2V benchmark outputs (Flux-style layout).

    Handles:
    - Locating results/timing.json
    - Parsing pipe_time values
    - Locating video_i2v.mp4
    - Validating against GPU-specific thresholds (max_avg_pipe_time_s)
    """

    def __init__(
        self,
        output_dir: str,
        *,
        expected_video_name: str = WAN_I2V_VIDEO_ARTIFACT,
        require_video_artifact: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.expected_video_name = expected_video_name
        self.require_video_artifact = require_video_artifact

    def find_timing_json(self) -> Optional[Path]:
        timing_json = self.output_dir / WAN_I2V_TIMING_JSON_REL
        if timing_json.exists():
            log.info("Found timing.json: %s", timing_json)
            return timing_json

        for root, _, files in os.walk(self.output_dir):
            if "timing.json" in files:
                timing_path = Path(root) / "timing.json"
                log.info("Found timing.json (fallback search): %s", timing_path)
                return timing_path

        log.warning("timing.json not found under %s", self.output_dir)
        return None

    def find_video_artifact(self) -> Optional[Path]:
        direct = self.output_dir / "results" / self.expected_video_name
        if direct.exists():
            log.info("Found video artifact: %s", direct)
            return direct

        for root, _, files in os.walk(self.output_dir):
            if self.expected_video_name in files:
                video_path = Path(root) / self.expected_video_name
                log.info("Found video artifact (fallback search): %s", video_path)
                return video_path

        if self.require_video_artifact:
            log.warning("%s not found under %s", self.expected_video_name, self.output_dir)
        return None

    def parse_timing_json(self, timing_json: Path) -> Tuple[List[float], List[str]]:
        pipe_times: List[float] = []
        errors: List[str] = []

        try:
            with open(timing_json, encoding="utf-8") as handle:
                data = json.load(handle)

            if not isinstance(data, list):
                errors.append(f"timing.json: expected a JSON list, got {type(data).__name__}")
                return pipe_times, errors

            for index, entry in enumerate(data):
                if not isinstance(entry, dict):
                    errors.append(f"timing.json[{index}]: expected dict, got {type(entry).__name__}")
                    continue
                if "pipe_time" not in entry:
                    errors.append(f"timing.json[{index}]: missing 'pipe_time' field")
                    continue
                pipe_time = entry["pipe_time"]
                if not isinstance(pipe_time, (int, float)):
                    errors.append(f"timing.json[{index}]: pipe_time is not numeric (got {type(pipe_time).__name__})")
                    continue
                pipe_times.append(float(pipe_time))
        except json.JSONDecodeError as exc:
            errors.append(f"timing.json: JSON parse error - {exc}")
        except OSError as exc:
            errors.append(f"timing.json: read error - {exc}")

        return pipe_times, errors

    def parse(self) -> Tuple[Optional[WanI2vBenchmarkResult], List[str]]:
        all_errors: List[str] = []

        timing_json = self.find_timing_json()
        if not timing_json:
            all_errors.append(f"timing.json not found under {self.output_dir}")
            return None, all_errors

        pipe_times, parse_errors = self.parse_timing_json(timing_json)
        all_errors.extend(parse_errors)
        if not pipe_times:
            all_errors.append("No valid pipe_time values extracted from timing.json")
            return None, all_errors

        video_path = self.find_video_artifact()
        if self.require_video_artifact and video_path is None:
            all_errors.append(f"Artifact '{self.expected_video_name}' not found under {self.output_dir}")
            return None, all_errors

        avg_pipe_time_s = sum(pipe_times) / len(pipe_times)
        log.info(
            "Average pipe_time: %.2fs (from %d repetitions)",
            avg_pipe_time_s,
            len(pipe_times),
        )

        return (
            WanI2vBenchmarkResult(
                avg_pipe_time_s=avg_pipe_time_s,
                repetition_count=len(pipe_times),
                pipe_times=pipe_times,
                timing_json_path=str(timing_json),
                video_path=str(video_path) if video_path else None,
            ),
            all_errors,
        )

    def validate_threshold(
        self,
        result: WanI2vBenchmarkResult,
        expected_results: Dict[str, Dict[str, float]],
        gpu_type: str = "auto",
    ) -> Tuple[bool, str]:
        if gpu_type in expected_results:
            threshold_dict = expected_results[gpu_type]
            log.info("Using GPU-specific threshold for '%s'", gpu_type)
        elif "auto" in expected_results:
            threshold_dict = expected_results["auto"]
            log.info("Using 'auto' threshold (no specific threshold for '%s')", gpu_type)
        else:
            return False, f"No threshold found for GPU type '{gpu_type}' and no 'auto' fallback"

        max_avg_time = threshold_dict.get("max_avg_pipe_time_s")
        if max_avg_time is None:
            return False, f"Threshold missing 'max_avg_pipe_time_s' for GPU type '{gpu_type}'"

        if result.avg_pipe_time_s <= max_avg_time:
            message = (
                f"PASS: Average pipe_time {result.avg_pipe_time_s:.2f}s <= "
                f"threshold {max_avg_time:.2f}s (GPU: {gpu_type})"
            )
            log.info("%s", message)
            return True, message

        message = (
            f"FAIL: Average pipe_time {result.avg_pipe_time_s:.2f}s > threshold {max_avg_time:.2f}s (GPU: {gpu_type})"
        )
        log.error("%s", message)
        return False, message


__all__ = [
    "WAN_I2V_TIMING_JSON_REL",
    "WAN_I2V_VIDEO_ARTIFACT",
    "WanI2vBenchmarkResult",
    "WanI2vOutputParser",
    "log_results_summary",
]
