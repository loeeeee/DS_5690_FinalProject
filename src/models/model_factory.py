"""Factory functions to load baseline and target models.

Expected responsibilities:
- Read model identifiers/config from CLI and experiments.yaml.
- Initialize tokenizers and models with correct precision/device mapping.
- Provide fail-fast checks for GPU availability and memory footprint.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def load_models(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Placeholder loader returning (baseline_model, target_model)."""
    raise NotImplementedError("Model loading not implemented; integrate HF/vLLM as needed.")

