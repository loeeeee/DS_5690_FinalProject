"""Profiling utilities for latency and memory."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Tuple

import torch

try:
    import psutil
except ImportError:
    psutil = None


def _supports_cuda() -> bool:
    return torch.cuda.is_available()


def _cuda_memory_reset() -> None:
    if _supports_cuda():
        torch.cuda.reset_peak_memory_stats(0)


def _cuda_peak_memory() -> int:
    if _supports_cuda():
        return torch.cuda.max_memory_reserved(0)
    return 0


def _cpu_peak_memory_tracked(run_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, int]:
    """Track peak CPU memory during function execution and return result.
    
    Uses psutil if available, otherwise falls back to resource module.
    Returns (result, peak_memory_bytes).
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("_cpu_peak_memory_tracked: Starting memory tracking")
    peak_memory = 0
    result = None
    
    if psutil is not None:
        logger.info("_cpu_peak_memory_tracked: Using psutil for memory tracking")
        # Use psutil for cross-platform memory tracking
        process = psutil.Process(os.getpid())
        memory_samples: list[int] = []
        stop_sampling = threading.Event()
        
        def sample_memory() -> None:
            """Sample memory usage in a background thread."""
            while not stop_sampling.is_set():
                try:
                    mem_info = process.memory_info()
                    memory_samples.append(mem_info.rss)
                except Exception:
                    pass
                time.sleep(0.01)  # Sample every 10ms
        
        sampler_thread = threading.Thread(target=sample_memory, daemon=True)
        sampler_thread.start()
        logger.info("_cpu_peak_memory_tracked: Memory sampler thread started")
        
        try:
            logger.info("_cpu_peak_memory_tracked: About to call run_fn (psutil path)")
            result = run_fn(*args, **kwargs)
            logger.info("_cpu_peak_memory_tracked: run_fn completed (psutil path)")
        finally:
            logger.info("_cpu_peak_memory_tracked: Stopping memory sampler")
            stop_sampling.set()
            sampler_thread.join(timeout=1.0)
            if memory_samples:
                peak_memory = max(memory_samples)
                logger.info(f"_cpu_peak_memory_tracked: Peak memory: {peak_memory} bytes ({peak_memory / (1024**3):.2f} GB)")
            else:
                logger.warning("_cpu_peak_memory_tracked: No memory samples collected")
    else:
        logger.info("_cpu_peak_memory_tracked: psutil not available, trying resource module")
        # Fallback: try to use resource module with sampling (Unix/Linux only)
        # Note: resource.ru_maxrss is cumulative, so we sample during execution
        try:
            import resource
            logger.info("_cpu_peak_memory_tracked: Using resource module for memory tracking")
            
            memory_samples: list[int] = []
            stop_sampling = threading.Event()
            
            def sample_memory_resource() -> None:
                """Sample memory usage using resource module in a background thread."""
                while not stop_sampling.is_set():
                    try:
                        # ru_maxrss is in KB, convert to bytes
                        mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        memory_samples.append(mem_kb * 1024)
                    except Exception:
                        pass
                    time.sleep(0.01)  # Sample every 10ms
            
            sampler_thread = threading.Thread(target=sample_memory_resource, daemon=True)
            sampler_thread.start()
            logger.info("_cpu_peak_memory_tracked: Resource module sampler thread started")
            
            try:
                logger.info("_cpu_peak_memory_tracked: About to call run_fn (resource path)")
                result = run_fn(*args, **kwargs)
                logger.info("_cpu_peak_memory_tracked: run_fn completed (resource path)")
            finally:
                logger.info("_cpu_peak_memory_tracked: Stopping resource sampler")
                stop_sampling.set()
                sampler_thread.join(timeout=1.0)
                if memory_samples:
                    peak_memory = max(memory_samples)
                    logger.info(f"_cpu_peak_memory_tracked: Peak memory: {peak_memory} bytes ({peak_memory / (1024**3):.2f} GB)")
                else:
                    peak_memory = 0
                    logger.warning("_cpu_peak_memory_tracked: No memory samples collected")
        except (ImportError, AttributeError):
            logger.warning("_cpu_peak_memory_tracked: resource module not available, running without memory tracking")
            # resource module not available or doesn't support RUSAGE_SELF
            logger.info("_cpu_peak_memory_tracked: About to call run_fn (no tracking path)")
            result = run_fn(*args, **kwargs)
            logger.info("_cpu_peak_memory_tracked: run_fn completed (no tracking path)")
            peak_memory = 0
    
    logger.info("_cpu_peak_memory_tracked: Returning result and peak_memory")
    return result, peak_memory


def profile_generation(run_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, float]]:
    """Run generation under profiling; returns (result, metrics)."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("profile_generation: Starting profiling")
    logger.info(f"profile_generation: CUDA available: {_supports_cuda()}")
    logger.info(f"profile_generation: Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
    
    metrics: Dict[str, float] = {}
    if _supports_cuda():
        # ROCm compatibility: ensure synchronization before timing
        logger.info("profile_generation: Using CUDA profiling path")
        torch.cuda.synchronize()
        _cuda_memory_reset()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        logger.info("profile_generation: About to call run_fn (CUDA path)")
        result = run_fn(*args, **kwargs)
        logger.info("profile_generation: run_fn completed (CUDA path)")
        end_evt.record()
        # Critical: synchronize before reading elapsed time
        torch.cuda.synchronize()
        latency_ms = start_evt.elapsed_time(end_evt)
        metrics["latency_ms"] = float(latency_ms)
        metrics["peak_memory_bytes"] = float(_cuda_peak_memory())
        logger.info(f"profile_generation: CUDA profiling complete - latency: {latency_ms:.2f}ms")
    else:
        logger.info("profile_generation: Using CPU profiling path")
        start = time.monotonic()
        logger.info("profile_generation: About to call run_fn (CPU path)")
        result, peak_memory_bytes = _cpu_peak_memory_tracked(run_fn, *args, **kwargs)
        logger.info("profile_generation: run_fn completed (CPU path)")
        latency_ms = (time.monotonic() - start) * 1000.0
        metrics["latency_ms"] = float(latency_ms)
        metrics["peak_memory_bytes"] = float(peak_memory_bytes)
        logger.info(f"profile_generation: CPU profiling complete - latency: {latency_ms:.2f}ms, peak_memory: {peak_memory_bytes} bytes")

    logger.info("profile_generation: Returning results and metrics")
    return result, metrics

