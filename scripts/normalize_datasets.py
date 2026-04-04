#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
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
    CANONICAL_RAW_CONTRACTS_PATH,
    CANONICAL_RAW_STE_PATH,
    CANONICAL_SEARCH_DB_PATH,
    CANONICAL_SYNONYMS_PATH,
    CANONICAL_TEST_QUERIES_PATH,
    KNOWN_PROCESSED_ARTIFACTS,
    LEGACY_DATA_DIR,
    LEGACY_PROCESSED_DIR,
    ensure_dataset_dirs,
    raw_contract_candidates,
    raw_ste_candidates,
)


def transfer(source: Path, target: Path, *, move: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(source), str(target))
        return "moved"
    shutil.copy2(source, target)
    return "copied"


def normalize_single(source_candidates: list[Path], target: Path, *, move: bool) -> list[str]:
    if target.exists():
        return []
    for source in source_candidates:
        if not source.exists():
            continue
        if source.resolve() == target.resolve():
            return []
        action = transfer(source, target, move=move)
        return [f"{action}: {source} -> {target}"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize local datasets into canonical datasets/ layout.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them.")
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Do not migrate legacy processed artifacts from data/processed.",
    )
    args = parser.parse_args()

    move = bool(args.move and not args.copy)
    ensure_dataset_dirs()

    operations: list[str] = []
    operations.extend(normalize_single(raw_ste_candidates(PROJECT_ROOT), CANONICAL_RAW_STE_PATH, move=move))
    operations.extend(normalize_single(raw_contract_candidates(PROJECT_ROOT), CANONICAL_RAW_CONTRACTS_PATH, move=move))
    operations.extend(
        normalize_single([LEGACY_DATA_DIR / "reference" / "search_synonyms.json"], CANONICAL_SYNONYMS_PATH, move=move)
    )
    operations.extend(
        normalize_single(
            [LEGACY_DATA_DIR / "reference" / "search_test_queries.json"],
            CANONICAL_TEST_QUERIES_PATH,
            move=move,
        )
    )

    if not args.skip_processed:
        target_by_name = {
            "tenderhack_search.sqlite": CANONICAL_SEARCH_DB_PATH,
            "tenderhack_preprocessed.sqlite": CANONICAL_PREPROCESSED_DB_PATH,
            "tenderhack_fasttext_corpus.txt": CANONICAL_FASTTEXT_CORPUS_PATH,
            "tenderhack_fasttext.bin": CANONICAL_FASTTEXT_MODEL_PATH,
            "customer_region_lookup.csv": CANONICAL_CUSTOMER_REGION_LOOKUP_PATH,
        }
        for artifact_name in KNOWN_PROCESSED_ARTIFACTS:
            source = LEGACY_PROCESSED_DIR / artifact_name
            target = target_by_name.get(artifact_name, PROJECT_ROOT / "datasets" / "processed" / artifact_name)
            operations.extend(normalize_single([source], target, move=move))

    if not operations:
        print("[datasets] layout already normalized", flush=True)
        return

    for line in operations:
        print(f"[datasets] {line}", flush=True)


if __name__ == "__main__":
    main()
