"""Download helper for models and datasets used in benchmarks."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import datasets
import yaml
from huggingface_hub import snapshot_download
from tqdm import tqdm


@dataclass
class BenchmarkTargets:
    baseline_id: str
    target_id: str
    dataset_name: str
    dataset_config: str
    dataset_split: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-download benchmark assets (models + dataset)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments.yaml"),
        help="Path to experiments.yaml for default model IDs.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=None,
        help="Custom cache dir for Hugging Face models (defaults to HF_HOME/TRANSFORMERS_CACHE).",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=None,
        help="Cache dir for datasets (defaults to HF_DATASETS_CACHE).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token (otherwise uses HF_TOKEN/HUGGINGFACE_TOKEN env vars).",
    )
    parser.add_argument(
        "--extra-model-id",
        action="append",
        default=None,
        help="Additional model IDs to fetch (can be repeated).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download; only check existing cache.",
    )
    return parser


def _load_targets(config_path: Path) -> BenchmarkTargets:
    with config_path.open("r") as handle:
        raw = yaml.safe_load(handle)
    experiment = raw.get("experiment", {})
    dataset = raw.get("dataset", {})
    return BenchmarkTargets(
        baseline_id=str(experiment.get("baseline_id")),
        target_id=str(experiment.get("target_id")),
        dataset_name=str(dataset.get("name")),
        dataset_config=str(dataset.get("config")),
        dataset_split=str(dataset.get("split", "validation")),
    )


def _resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    return env_token


def _require_token_if_gated(model_ids: Sequence[str], token: str | None) -> None:
    if any("meta-llama" in mid.lower() for mid in model_ids) and not token:
        message = (
            "Hugging Face token required for gated model(s) (e.g., Meta LLaMA 3). "
            "Pass --hf-token or set HF_TOKEN/HUGGINGFACE_TOKEN."
        )
        raise RuntimeError(message)


def _download_models(
    model_ids: Sequence[str],
    cache_dir: Path | None,
    token: str | None,
    local_files_only: bool,
) -> List[Path]:
    cached_paths: List[Path] = []
    for model_id in tqdm(model_ids, desc="Models", unit="model"):
        logging.info("Fetching model %s", model_id)
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=token,
            resume_download=True,
            local_files_only=local_files_only,
        )
        cached_paths.append(Path(local_path))
    return cached_paths


def _download_dataset(
    name: str,
    config: str,
    split: str,
    cache_dir: Path | None,
    token: str | None,
) -> Path:
    logging.info("Fetching dataset %s/%s [%s]", name, config, split)
    dataset = datasets.load_dataset(
        name,
        config,
        split=split,
        cache_dir=cache_dir,
        token=token,
        download_mode="reuse_dataset_if_exists",
    )
    cache_location = Path(dataset.cache_files[0]["filename"]).parent if dataset.cache_files else cache_dir
    return cache_location or Path(datasets.config.HF_DATASETS_CACHE)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    targets = _load_targets(args.config)
    token = _resolve_token(args.hf_token)

    model_ids: List[str] = [targets.baseline_id, targets.target_id]
    if args.extra_model_id:
        model_ids.extend(args.extra_model_id)
    _require_token_if_gated(model_ids, token)

    model_cache_dir = args.model_cache_dir
    dataset_cache_dir = args.dataset_cache_dir

    downloaded_models = _download_models(
        model_ids=model_ids,
        cache_dir=model_cache_dir,
        token=token,
        local_files_only=args.local_files_only,
    )
    dataset_path = _download_dataset(
        name=targets.dataset_name,
        config=targets.dataset_config,
        split=targets.dataset_split,
        cache_dir=dataset_cache_dir,
        token=token,
    )

    logging.info("Models cached at: %s", downloaded_models)
    logging.info("Dataset cached at: %s", dataset_path)
    logging.info("Download prep complete.")


if __name__ == "__main__":
    main()

