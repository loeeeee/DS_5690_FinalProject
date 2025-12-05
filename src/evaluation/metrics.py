"""Quality metrics helpers.

Expected responsibilities:
- Compute perplexity and BERTScore (or alternative quality proxy) headlessly.
- Accept batched generations and references; return scalar summaries.
"""

from __future__ import annotations

from typing import Any, Dict


def compute_metrics(predictions: Any, references: Any) -> Dict[str, float]:
    raise NotImplementedError("Implement perplexity/BERTScore once models and data are defined.")

