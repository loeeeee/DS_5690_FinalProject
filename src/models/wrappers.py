"""Adapters to present a unified generate() API for benchmarking."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol, Sequence

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when a function call times out."""
    pass


def _run_with_timeout(func: Any, timeout_seconds: int, *args: Any, **kwargs: Any) -> Any:
    """Run a function with a timeout using threading (works on all platforms)."""
    result_container: list[Any] = [None]
    exception_container: list[Exception | None] = [None]
    
    def target() -> None:
        try:
            result_container[0] = func(*args, **kwargs)
        except Exception as e:
            exception_container[0] = e
    
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        logger.error(f"Function call timed out after {timeout_seconds} seconds")
        raise TimeoutError(f"Function call timed out after {timeout_seconds} seconds")
    
    if exception_container[0] is not None:
        raise exception_container[0]
    
    return result_container[0]


def _run_with_progress(
    func: Callable[[], Any],
    timeout_seconds: int,
    max_tokens: int,
    device_type: str,
    desc: str = "Generating",
) -> Any:
    """Run a function with a progress bar showing elapsed time.
    
    Args:
        func: Function to execute (no arguments)
        timeout_seconds: Maximum time to wait
        max_tokens: Maximum tokens to generate (for display purposes)
        device_type: Device type ("cpu" or "cuda") for progress bar positioning
        desc: Description for progress bar
    """
    result_container: list[Any] = [None]
    exception_container: list[Exception | None] = [None]
    progress_stop = threading.Event()
    start_time = time.time()
    
    # Create progress bar - use position 2 for generation progress (below batch progress)
    # Use a time-based progress bar since we can't track actual token generation progress
    pbar = tqdm(
        total=100,  # Use 100 as total for percentage display
        desc=desc,
        unit="%",
        position=2,
        leave=False,
        bar_format="{desc}: {elapsed} elapsed | {percentage:3.0f}%",
        ncols=80,
    )
    
    def update_progress() -> None:
        """Update progress bar with elapsed time."""
        while not progress_stop.is_set():
            elapsed = time.time() - start_time
            # Show progress as percentage of timeout (capped at 100%)
            progress_pct = min(int((elapsed / timeout_seconds) * 100), 100)
            pbar.n = progress_pct
            pbar.refresh()
            time.sleep(1.0)  # Update every second
    
    def target() -> None:
        try:
            result_container[0] = func()
        except Exception as e:
            exception_container[0] = e
    
    # Start generation thread
    gen_thread = threading.Thread(target=target, daemon=True)
    gen_thread.start()
    
    # Start progress update thread
    progress_thread = threading.Thread(target=update_progress, daemon=True)
    progress_thread.start()
    
    # Wait for completion or timeout
    gen_thread.join(timeout=timeout_seconds)
    
    # Stop progress updates
    progress_stop.set()
    progress_thread.join(timeout=1.0)
    
    # Check if generation completed successfully
    if gen_thread.is_alive():
        pbar.close()
        logger.error(f"Function call timed out after {timeout_seconds} seconds")
        raise TimeoutError(f"Function call timed out after {timeout_seconds} seconds")
    
    # Generation completed - show 100%
    pbar.n = 100
    pbar.refresh()
    pbar.close()
    
    if exception_container[0] is not None:
        raise exception_container[0]
    
    return result_container[0]


class GenerationBatchResult(Protocol):
    texts: List[str]
    token_counts: List[int]
    timings: Dict[str, float]


@dataclass
class ModelWrapper:
    model: Any
    tokenizer: Any

    def generate_batch(
        self,
        prompts: Sequence[str],
        steps: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationBatchResult:
        raise NotImplementedError("Wrap model-specific generate logic here.")


class AutoRegressiveWrapper(ModelWrapper):
    """Wrapper for standard autoregressive CausalLM models."""

    def generate_batch(
        self,
        prompts: Sequence[str],
        steps: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationBatchResult:
        logger.debug(f"AutoRegressiveWrapper.generate_batch: Starting - prompts: {len(prompts)}, max_new_tokens: {max_new_tokens}")
        device = next(self.model.parameters()).device
        logger.debug(f"AutoRegressiveWrapper.generate_batch: Model device: {device}")
        
        logger.debug("AutoRegressiveWrapper.generate_batch: Tokenizing prompts")
        tokenizer_outputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        logger.debug(f"AutoRegressiveWrapper.generate_batch: Tokenization complete - input_ids shape: {tokenizer_outputs['input_ids'].shape}")
        
        # Move all tensors in tokenizer_outputs to the model's device
        logger.debug(f"AutoRegressiveWrapper.generate_batch: Moving tensors to device: {device}")
        tokenizer_outputs = {k: v.to(device) for k, v in tokenizer_outputs.items()}
        logger.debug("AutoRegressiveWrapper.generate_batch: Tensors moved to device")
        
        max_tokens = max_new_tokens or 128
        logger.debug(f"AutoRegressiveWrapper.generate_batch: Starting model.generate() with max_new_tokens={max_tokens}")
        
        # On CPU, generation can be extremely slow. Add timeout and progress bar.
        # For CPU, allow up to 30 minutes per generation (very conservative)
        timeout_seconds = 1800 if device.type == "cpu" else 300
        
        def _generate() -> Any:
            with torch.no_grad():
                # Standard generation - works on CUDA, may have issues on ROCm
                return self.model.generate(
                    **tokenizer_outputs,
                    max_new_tokens=max_tokens,
                    use_cache=True,  # LLaMA supports KV cache
                    do_sample=False,  # Greedy decoding for reproducibility
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        
        start_time = time.time()
        logger.debug(f"AutoRegressiveWrapper.generate_batch: Starting generation (timeout: {timeout_seconds}s)")
        
        try:
            generated = _run_with_progress(
                _generate,
                timeout_seconds,
                max_tokens,
                device.type,
                desc="Generating tokens",
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(f"AutoRegressiveWrapper.generate_batch: model.generate() completed in {elapsed_time:.2f}s - output shape: {generated.shape}")
        except TimeoutError as e:
            elapsed_time = time.time() - start_time
            logger.error(f"AutoRegressiveWrapper.generate_batch: Generation timed out after {elapsed_time:.2f}s: {e}")
            raise
        
        logger.debug("AutoRegressiveWrapper.generate_batch: Decoding generated tokens")
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        logger.debug(f"AutoRegressiveWrapper.generate_batch: Decoding complete - {len(texts)} texts")
        
        logger.debug("AutoRegressiveWrapper.generate_batch: Computing token counts")
        token_counts: List[int] = []
        for out_ids, in_ids in zip(generated, tokenizer_outputs["input_ids"]):
            token_counts.append(len(out_ids) - len(in_ids))
        logger.debug(f"AutoRegressiveWrapper.generate_batch: Token counts: {token_counts}")
        
        logger.debug("AutoRegressiveWrapper.generate_batch: Returning result")
        return type(
            "GenerationBatchResult",
            (),
            {"texts": texts, "token_counts": token_counts, "timings": {}},
        )()


class DiffusionLikeWrapper(ModelWrapper):
    """Wrapper placeholder for LLaDA diffusion-style generation."""

    def generate_batch(
        self,
        prompts: Sequence[str],
        steps: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationBatchResult:
        logger.debug(f"DiffusionLikeWrapper.generate_batch: Starting - prompts: {len(prompts)}, steps: {steps}, max_new_tokens: {max_new_tokens}")
        device = next(self.model.parameters()).device
        logger.debug(f"DiffusionLikeWrapper.generate_batch: Model device: {device}")
        
        logger.debug("DiffusionLikeWrapper.generate_batch: Tokenizing prompts")
        tokenizer_outputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        logger.debug(f"DiffusionLikeWrapper.generate_batch: Tokenization complete - input_ids shape: {tokenizer_outputs['input_ids'].shape}")
        
        # Move all tensors in tokenizer_outputs to the model's device
        logger.debug(f"DiffusionLikeWrapper.generate_batch: Moving tensors to device: {device}")
        tokenizer_outputs = {k: v.to(device) for k, v in tokenizer_outputs.items()}
        logger.debug("DiffusionLikeWrapper.generate_batch: Tensors moved to device")
        
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or 128,
            "use_cache": False,  # LLaDA doesn't support KV cache
        }
        logger.debug(f"DiffusionLikeWrapper.generate_batch: Generation kwargs: {generation_kwargs}")
        
        # TODO: LLaDA step control - num_inference_steps is not accepted by this model's generate()
        # Steps may need to be controlled via generation_config or model config
        # For now, ignore steps parameter to allow benchmark to run
        max_tokens = max_new_tokens or 128
        logger.debug("DiffusionLikeWrapper.generate_batch: Starting model.generate()")
        
        # On CPU, generation can be extremely slow. Add timeout and progress bar.
        timeout_seconds = 1800 if device.type == "cpu" else 300
        
        def _generate() -> Any:
            with torch.no_grad():
                return self.model.generate(
                    **tokenizer_outputs,
                    **generation_kwargs,
                )
        
        start_time = time.time()
        logger.debug(f"DiffusionLikeWrapper.generate_batch: Starting generation (timeout: {timeout_seconds}s)")
        
        try:
            generated = _run_with_progress(
                _generate,
                timeout_seconds,
                max_tokens,
                device.type,
                desc="Generating tokens",
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(f"DiffusionLikeWrapper.generate_batch: model.generate() completed in {elapsed_time:.2f}s - output shape: {generated.shape}")
        except TimeoutError as e:
            elapsed_time = time.time() - start_time
            logger.error(f"DiffusionLikeWrapper.generate_batch: Generation timed out after {elapsed_time:.2f}s: {e}")
            raise
        
        logger.debug("DiffusionLikeWrapper.generate_batch: Decoding generated tokens")
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        logger.debug(f"DiffusionLikeWrapper.generate_batch: Decoding complete - {len(texts)} texts")
        
        logger.debug("DiffusionLikeWrapper.generate_batch: Computing token counts")
        token_counts: List[int] = []
        for out_ids, in_ids in zip(generated, tokenizer_outputs["input_ids"]):
            token_counts.append(len(out_ids) - len(in_ids))
        logger.debug(f"DiffusionLikeWrapper.generate_batch: Token counts: {token_counts}")
        
        logger.debug("DiffusionLikeWrapper.generate_batch: Creating result object")
        result = type(
            "GenerationBatchResult",
            (),
            {"texts": texts, "token_counts": token_counts, "timings": {}},
        )()
        logger.debug("DiffusionLikeWrapper.generate_batch: Returning result")
        return result

