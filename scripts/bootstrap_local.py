#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.dataset_layout import (
    CANONICAL_CUSTOMER_REGION_LOOKUP_PATH,
    CANONICAL_FASTTEXT_CORPUS_PATH,
    CANONICAL_FASTTEXT_MODEL_PATH,
    CANONICAL_PREPROCESSED_DB_PATH,
    CANONICAL_SEARCH_DB_PATH,
    PROCESSED_DIR,
    ensure_dataset_dirs,
    resolve_raw_contracts_path,
    resolve_raw_ste_path,
)


def run_step(title: str, command: list[str]) -> None:
    print(f"[bootstrap] {title}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {description}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap local TenderHack assets from raw source files.")
    parser.add_argument("--ste-path", default=None, help="Path to the raw STE CSV.")
    parser.add_argument("--contracts-path", default=None, help="Path to the raw contracts CSV.")
    parser.add_argument("--skip-fasttext", action="store_true", help="Skip fastText model training.")
    parser.add_argument(
        "--train-personalization",
        action="store_true",
        help="Run offline personalization training after asset build.",
    )
    parser.add_argument("--semantic-min-frequency", type=int, default=10)
    parser.add_argument("--semantic-neighbors-per-token", type=int, default=8)
    args = parser.parse_args()

    ste_path = Path(args.ste_path) if args.ste_path else resolve_raw_ste_path(PROJECT_ROOT)
    contracts_path = Path(args.contracts_path) if args.contracts_path else resolve_raw_contracts_path(PROJECT_ROOT)
    require_file(ste_path, "raw STE catalog")
    require_file(contracts_path, "raw contracts dataset")

    ensure_dataset_dirs()
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)

    python = sys.executable
    run_step("normalize dataset layout", [python, "scripts/normalize_datasets.py", "--copy"])
    run_step(
        "preprocess raw datasets",
        [
            python,
            "scripts/preprocess_data.py",
            "--ste-path",
            str(ste_path),
            "--contracts-path",
            str(contracts_path),
            "--output-dir",
            str(PROCESSED_DIR),
        ],
    )
    run_step(
        "build search database and region lookup",
        [
            python,
            "scripts/build_search_assets.py",
            "--catalog-path",
            str(PROCESSED_DIR / "ste_catalog_search_ready.csv"),
            "--contracts-path",
            str(contracts_path),
            "--search-db-path",
            str(CANONICAL_SEARCH_DB_PATH),
            "--preprocessed-db-path",
            str(CANONICAL_PREPROCESSED_DB_PATH),
            "--customer-region-output",
            str(CANONICAL_CUSTOMER_REGION_LOOKUP_PATH),
            "--semantic-min-frequency",
            str(args.semantic_min_frequency),
            "--semantic-neighbors-per-token",
            str(args.semantic_neighbors_per_token),
        ],
    )
    run_step(
        "build supplier offer assets from contracts",
        [
            python,
            "scripts/build_offer_assets.py",
            "--contracts-path",
            str(contracts_path),
            "--preprocessed-db-path",
            str(CANONICAL_PREPROCESSED_DB_PATH),
        ],
    )

    if not args.skip_fasttext:
        run_step(
            "train local fastText model",
            [
                python,
                "scripts/train_fasttext.py",
                "--catalog-path",
                str(PROCESSED_DIR / "ste_catalog_search_ready.csv"),
                "--corpus-path",
                str(CANONICAL_FASTTEXT_CORPUS_PATH),
                "--model-path",
                str(CANONICAL_FASTTEXT_MODEL_PATH),
            ],
        )

    if args.train_personalization:
        run_step(
            "train offline personalization ranker",
            [python, "scripts/run_personalization_pipeline.py"],
        )

    print("[bootstrap] complete", flush=True)


if __name__ == "__main__":
    main()
