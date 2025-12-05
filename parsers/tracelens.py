"""
TraceLens parser for PyTorch profiler traces.

Parses PyTorch profiler JSON traces and extracts performance metrics
using TraceLens analysis.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from parsers.schemas import (
    AortaTraceMetrics,
    AortaBenchmarkResult,
    ParseResult,
    ParseStatus,
)

# Import runners for type hints only
from runners._base_runner import RunResult

log = logging.getLogger(__name__)

# Try to import TraceLens - it's optional for basic parsing
try:
    from TraceLens import generate_perf_report_pytorch
    TRACELENS_AVAILABLE = True
except ImportError:
    TRACELENS_AVAILABLE = False
    generate_perf_report_pytorch = None  # type: ignore


class TraceLensParser:
    """
    Parser for PyTorch profiler traces.
    
    Can use TraceLens for detailed analysis or fall back to basic JSON parsing.
    """
    
    def __init__(self, use_tracelens: bool = True):
        """
        Initialize parser.
        
        Args:
            use_tracelens: Whether to use TraceLens for analysis (if available)
        """
        self.use_tracelens = use_tracelens and TRACELENS_AVAILABLE
        
        if use_tracelens and not TRACELENS_AVAILABLE:
            log.warning("TraceLens not available, falling back to basic parsing")
    
    def parse(self, run_result: RunResult) -> ParseResult[AortaTraceMetrics]:
        """
        Parse benchmark results into validated metrics.
        
        Args:
            run_result: Result from AortaRunner
            
        Returns:
            ParseResult containing validated AortaTraceMetrics
        """
        if not run_result.succeeded:
            return ParseResult(
                status=ParseStatus.FAILED,
                errors=[f"Run did not succeed: {run_result.error_message}"]
            )
        
        trace_dir = run_result.get_artifact("torch_traces")
        if not trace_dir:
            return ParseResult(
                status=ParseStatus.FAILED,
                errors=["No torch_traces artifact found in run result"]
            )
        
        if not trace_dir.exists():
            return ParseResult(
                status=ParseStatus.FAILED,
                errors=[f"Trace directory does not exist: {trace_dir}"]
            )
        
        return self.parse_trace_directory(trace_dir)
    
    def parse_trace_directory(self, trace_dir: Path) -> ParseResult[AortaTraceMetrics]:
        """
        Parse all trace files in a directory.
        
        Looks for PyTorch profiler JSON traces and extracts metrics.
        
        Args:
            trace_dir: Directory containing trace files
            
        Returns:
            ParseResult with metrics for each rank
        """
        results: List[AortaTraceMetrics] = []
        warnings: List[str] = []
        errors: List[str] = []
        
        # Find all trace JSON files
        # PyTorch profiler typically outputs: rank0/trace.json, rank1/trace.json, etc.
        # Or: worker0_trace.json, worker1_trace.json
        trace_files = list(trace_dir.glob("**/trace*.json"))
        
        if not trace_files:
            # Try alternative patterns
            trace_files = list(trace_dir.glob("**/*.json"))
        
        if not trace_files:
            return ParseResult(
                status=ParseStatus.FAILED,
                errors=[f"No trace files found in {trace_dir}"]
            )
        
        log.info(f"Found {len(trace_files)} trace files to parse")
        
        for trace_file in trace_files:
            try:
                rank = self._extract_rank_from_path(trace_file)
                
                if self.use_tracelens:
                    metrics = self._parse_with_tracelens(trace_file, rank)
                else:
                    metrics = self._parse_basic(trace_file, rank)
                
                results.append(metrics)
                log.debug(f"Parsed rank {rank}: {metrics.total_time_us:.2f}us total")
                
            except Exception as e:
                warnings.append(f"Failed to parse {trace_file}: {e}")
                log.warning(f"Failed to parse {trace_file}: {e}")
        
        # Determine status
        if not results:
            status = ParseStatus.FAILED
            errors.append("No traces could be parsed")
        elif warnings:
            status = ParseStatus.PARTIAL
        else:
            status = ParseStatus.SUCCESS
        
        return ParseResult(
            status=status,
            results=results,
            warnings=warnings,
            errors=errors,
            metadata={
                "trace_dir": str(trace_dir),
                "files_found": len(trace_files),
                "files_parsed": len(results),
            }
        )
    
    def _extract_rank_from_path(self, trace_file: Path) -> int:
        """
        Extract rank ID from trace file path.
        
        Handles patterns like:
        - rank0/trace.json -> 0
        - rank_0/trace.json -> 0
        - worker0_trace.json -> 0
        - trace_rank0.json -> 0
        """
        path_str = str(trace_file)
        
        # Try common patterns
        import re
        
        patterns = [
            r'rank[_]?(\d+)',
            r'worker[_]?(\d+)',
            r'gpu[_]?(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, path_str, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Default to 0 if no pattern matched
        log.warning(f"Could not extract rank from {trace_file}, defaulting to 0")
        return 0
    
    def _parse_with_tracelens(self, trace_file: Path, rank: int) -> AortaTraceMetrics:
        """
        Parse trace using TraceLens for detailed analysis.
        
        Args:
            trace_file: Path to trace JSON file
            rank: Rank ID
            
        Returns:
            Parsed metrics
        """
        log.info(f"Parsing with TraceLens: {trace_file}")
        
        try:
            import tempfile
            import os
            
            # TraceLens needs a writable output path for its xlsx report
            # Use a temp file to avoid permission issues
            with tempfile.TemporaryDirectory() as tmpdir:
                output_xlsx = os.path.join(tmpdir, f"trace_rank{rank}_report.xlsx")
                
                # Run TraceLens - it returns Dict[str, DataFrame] directly
                result_dfs = generate_perf_report_pytorch(
                    profile_json_path=str(trace_file),
                    output_xlsx_path=output_xlsx,
                )
            
            log.debug(f"TraceLens returned dataframes: {list(result_dfs.keys())}")
            
            # Extract metrics from TraceLens output
            total_time_us = 0.0
            compute_time_us = 0.0
            comm_time_us = 0.0
            
            # TraceLens returns gpu_timeline with key metrics
            if 'gpu_timeline' in result_dfs:
                df = result_dfs['gpu_timeline']
                log.debug(f"gpu_timeline data:\n{df.to_string()}")
                
                # Create a lookup dict from type -> time ms
                timeline = dict(zip(df['type'], df['time ms']))
                
                # Extract metrics (TraceLens returns in ms, convert to us)
                # total_time is the full trace duration
                total_time_ms = timeline.get('total_time', 0)
                compute_time_ms = timeline.get('computation_time', 0)
                # exposed_comm_time is communication NOT overlapped with compute
                exposed_comm_ms = timeline.get('exposed_comm_time', 0)
                # total_comm_time includes overlapped communication
                total_comm_ms = timeline.get('total_comm_time', 0)
                
                # Convert to microseconds
                total_time_us = total_time_ms * 1000
                compute_time_us = compute_time_ms * 1000
                # Use exposed comm time for non-overlapped communication cost
                comm_time_us = exposed_comm_ms * 1000
                
                log.info(f"TraceLens gpu_timeline for rank {rank}:")
                log.info(f"  Total time: {total_time_ms:.2f}ms")
                log.info(f"  Compute time: {compute_time_ms:.2f}ms ({100*compute_time_ms/total_time_ms:.1f}%)")
                log.info(f"  Exposed comm: {exposed_comm_ms:.2f}ms ({100*exposed_comm_ms/total_time_ms:.1f}%)")
                log.info(f"  Total comm: {total_comm_ms:.2f}ms (overlap: {100*(total_comm_ms-exposed_comm_ms)/total_comm_ms:.1f}%)")
            
            if total_time_us > 0:
                log.info(f"TraceLens metrics for rank {rank}: total={total_time_us:.2f}us, compute={compute_time_us:.2f}us, comm={comm_time_us:.2f}us")
                return AortaTraceMetrics(
                    rank=rank,
                    total_time_us=float(total_time_us),
                    compute_time_us=float(compute_time_us),
                    communication_time_us=float(comm_time_us),
                )
            else:
                log.warning(f"TraceLens returned no usable metrics, falling back to basic parsing")
                return self._parse_basic(trace_file, rank)
                
        except Exception as e:
            log.warning(f"TraceLens parsing failed: {e}, falling back to basic parsing")
            import traceback
            log.debug(traceback.format_exc())
            return self._parse_basic(trace_file, rank)
    
    def _parse_basic(self, trace_file: Path, rank: int) -> AortaTraceMetrics:
        """
        Basic parsing of PyTorch profiler JSON without TraceLens.
        
        Extracts timing information directly from the trace events.
        
        Args:
            trace_file: Path to trace JSON file
            rank: Rank ID
            
        Returns:
            Parsed metrics
        """
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        
        # PyTorch profiler output structure varies by version
        # Common structure: {"traceEvents": [...], ...}
        events = trace_data.get("traceEvents", [])
        
        if not events:
            # Try alternative structure
            events = trace_data if isinstance(trace_data, list) else []
        
        # Categorize events by type
        compute_time_us = 0.0
        comm_time_us = 0.0
        total_time_us = 0.0
        memory_time_us = 0.0
        
        compute_kernels = 0
        comm_kernels = 0
        
        peak_memory = 0.0
        
        for event in events:
            if not isinstance(event, dict):
                continue
            
            name = event.get("name", "")
            cat = event.get("cat", "")
            dur = event.get("dur", 0)  # Duration in microseconds
            
            # Skip instant events (dur = 0 or missing)
            if dur <= 0:
                continue
            
            # Categorize by event name/category
            name_lower = name.lower()
            cat_lower = cat.lower()
            
            # NCCL/communication events
            if any(k in name_lower for k in ["nccl", "allreduce", "allgather", "broadcast", "reduce_scatter"]):
                comm_time_us += dur
                comm_kernels += 1
            
            # Memory events
            elif any(k in name_lower for k in ["memcpy", "memset", "cudamemcpy"]):
                memory_time_us += dur
            
            # Compute kernels (HIP/CUDA kernels)
            elif cat_lower == "kernel" or "kernel" in name_lower:
                compute_time_us += dur
                compute_kernels += 1
            
            # GPU activity
            elif cat_lower == "gpu":
                compute_time_us += dur
            
            # Track total
            total_time_us += dur
            
            # Memory tracking
            if "args" in event:
                args = event["args"]
                if "Total Allocated" in args:
                    try:
                        mem_bytes = float(args["Total Allocated"])
                        peak_memory = max(peak_memory, mem_bytes / (1024**3))  # Convert to GB
                    except (ValueError, TypeError):
                        pass
        
        # If we couldn't categorize, estimate from total
        if total_time_us == 0:
            # Fallback: use trace span
            if events:
                start_times = [e.get("ts", 0) for e in events if isinstance(e, dict) and "ts" in e]
                end_times = [e.get("ts", 0) + e.get("dur", 0) for e in events if isinstance(e, dict)]
                if start_times and end_times:
                    total_time_us = max(end_times) - min(start_times)
        
        return AortaTraceMetrics(
            rank=rank,
            total_time_us=total_time_us,
            compute_time_us=compute_time_us,
            communication_time_us=comm_time_us,
            memory_time_us=memory_time_us if memory_time_us > 0 else None,
            peak_memory_gb=peak_memory if peak_memory > 0 else None,
            compute_kernel_count=compute_kernels if compute_kernels > 0 else None,
            comm_kernel_count=comm_kernels if comm_kernels > 0 else None,
        )
    
    def aggregate(
        self, 
        parse_result: ParseResult[AortaTraceMetrics],
        num_nodes: int,
        gpus_per_node: int,
        **metadata
    ) -> Optional[AortaBenchmarkResult]:
        """
        Aggregate per-rank metrics into a benchmark result.
        
        Args:
            parse_result: Parsed per-rank metrics
            num_nodes: Number of nodes in cluster
            gpus_per_node: GPUs per node
            **metadata: Additional metadata fields
            
        Returns:
            Aggregated benchmark result, or None if aggregation failed
        """
        if not parse_result.has_results:
            log.error("Cannot aggregate: no results to aggregate")
            return None
        
        try:
            return AortaBenchmarkResult.from_rank_metrics(
                metrics=parse_result.results,
                num_nodes=num_nodes,
                gpus_per_node=gpus_per_node,
                **metadata
            )
        except Exception as e:
            log.exception(f"Aggregation failed: {e}")
            return None
    
    def validate_thresholds(
        self,
        result: AortaBenchmarkResult,
        expected: Dict[str, Any]
    ) -> List[str]:
        """
        Validate benchmark results against expected thresholds.
        
        Args:
            result: Aggregated benchmark result
            expected: Dictionary of threshold configurations
            
        Returns:
            List of validation failure messages (empty if all pass)
        """
        failures = []
        
        # Check iteration time
        max_iteration_ms = expected.get("max_avg_iteration_ms")
        if max_iteration_ms is not None:
            if result.avg_iteration_time_ms > max_iteration_ms:
                failures.append(
                    f"Average iteration time {result.avg_iteration_time_ms:.2f}ms "
                    f"exceeds threshold {max_iteration_ms}ms"
                )
        
        # Check compute ratio
        min_compute_ratio = expected.get("min_compute_ratio")
        if min_compute_ratio is not None:
            if result.avg_compute_ratio < min_compute_ratio:
                failures.append(
                    f"Average compute ratio {result.avg_compute_ratio:.2%} "
                    f"below threshold {min_compute_ratio:.2%}"
                )
        
        # Check overlap ratio
        min_overlap_ratio = expected.get("min_overlap_ratio")
        if min_overlap_ratio is not None:
            if result.avg_overlap_ratio < min_overlap_ratio:
                failures.append(
                    f"Average overlap ratio {result.avg_overlap_ratio:.2%} "
                    f"below threshold {min_overlap_ratio:.2%}"
                )
        
        # Check per-rank variance (iteration time skew)
        max_time_variance = expected.get("max_time_variance_ratio")
        if max_time_variance is not None:
            if result.avg_iteration_time_us > 0:
                variance_ratio = result.std_iteration_time_us / result.avg_iteration_time_us
                if variance_ratio > max_time_variance:
                    failures.append(
                        f"Iteration time variance {variance_ratio:.2%} "
                        f"exceeds threshold {max_time_variance:.2%}"
                    )
        
        return failures

