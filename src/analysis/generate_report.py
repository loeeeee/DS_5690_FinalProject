"""Headless report generator.

Expected responsibilities:
- Read raw metrics CSV/JSON from results/raw_data.
- Plot latency/throughput/memory and quality-vs-steps curves.
- Save figures to results/figures using matplotlib Agg backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")


def main() -> None:
    raise NotImplementedError("Parse inputs, aggregate metrics, and render plots.")


if __name__ == "__main__":
    main()


