"""Quality metrics helpers.

Expected responsibilities:
- Compute perplexity and BERTScore (or alternative quality proxy) headlessly.
- Accept batched generations and references; return scalar summaries.
- Consume metric definitions from ``src.config.structures.MetricSuite`` to
  keep reporting aligned with experiment design.
"""

from __future__ import annotations

from typing import Any, Dict

from config.structures import MetricSuite


def compute_metrics(predictions: Any, references: Any) -> Dict[str, float]:
    """Compute quality metrics based on the configured metric suite.

    The concrete implementation should rely on ``MetricSuite`` to decide which
    scores to surface (e.g., perplexity, BERTScore) and return scalar summaries.
    """
    raise NotImplementedError(
        "Implement perplexity/BERTScore once models and data are defined."
    )

