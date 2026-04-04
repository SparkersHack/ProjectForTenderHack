from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
REFERENCE_DIR = DATASETS_DIR / "reference"
PROCESSED_DIR = DATASETS_DIR / "processed"

LEGACY_DATA_DIR = PROJECT_ROOT / "data"
LEGACY_REFERENCE_DIR = LEGACY_DATA_DIR / "reference"
LEGACY_PROCESSED_DIR = LEGACY_DATA_DIR / "processed"

CANONICAL_RAW_STE_PATH = RAW_DIR / "ste_catalog.csv"
CANONICAL_RAW_CONTRACTS_PATH = RAW_DIR / "contracts.csv"
CANONICAL_SYNONYMS_PATH = REFERENCE_DIR / "search_synonyms.json"
CANONICAL_TEST_QUERIES_PATH = REFERENCE_DIR / "search_test_queries.json"
CANONICAL_SEARCH_DB_PATH = PROCESSED_DIR / "tenderhack_search.sqlite"
CANONICAL_PREPROCESSED_DB_PATH = PROCESSED_DIR / "tenderhack_preprocessed.sqlite"
CANONICAL_FASTTEXT_CORPUS_PATH = PROCESSED_DIR / "tenderhack_fasttext_corpus.txt"
CANONICAL_FASTTEXT_MODEL_PATH = PROCESSED_DIR / "tenderhack_fasttext.bin"
CANONICAL_CUSTOMER_REGION_LOOKUP_PATH = PROCESSED_DIR / "customer_region_lookup.csv"

KNOWN_PROCESSED_ARTIFACTS = [
    "ste_catalog_clean.csv",
    "ste_catalog_search_ready.csv",
    "category_lookup.csv",
    "customer_ste_stats.csv",
    "customer_category_stats.csv",
    "region_category_stats.csv",
    "contracts_clean.csv",
    "contracts_flat.csv",
    "contracts.csv",
    "customer_region_lookup.csv",
    "tenderhack_preprocessed.sqlite",
    "tenderhack_search.sqlite",
    "tenderhack_fasttext_corpus.txt",
    "tenderhack_fasttext.bin",
]


def ensure_dataset_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _first_existing(candidates: Iterable[Path], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback


def ste_catalog_candidates(project_root: Path | None = None) -> List[Path]:
    root = project_root or PROJECT_ROOT
    return [
        root / "datasets" / "processed" / "ste_catalog_clean.csv",
        root / "datasets" / "processed" / "ste_catalog_search_ready.csv",
        root / "data" / "processed" / "ste_catalog_clean.csv",
        root / "data" / "processed" / "ste_catalog_search_ready.csv",
    ]


def contracts_candidates(project_root: Path | None = None) -> List[Path]:
    root = project_root or PROJECT_ROOT
    return [
        root / "datasets" / "processed" / "contracts_clean.csv",
        root / "datasets" / "processed" / "contracts_flat.csv",
        root / "datasets" / "processed" / "contracts.csv",
        root / "data" / "processed" / "contracts_clean.csv",
        root / "data" / "processed" / "contracts_flat.csv",
        root / "data" / "processed" / "contracts.csv",
    ]


def raw_ste_candidates(project_root: Path | None = None) -> List[Path]:
    root = project_root or PROJECT_ROOT
    candidates = [root / "datasets" / "raw" / "ste_catalog.csv"]
    candidates.extend(sorted((root / "datasets" / "raw").glob("СТЕ_*.csv")))
    candidates.extend(sorted(root.glob("СТЕ_*.csv")))
    return candidates


def raw_contract_candidates(project_root: Path | None = None) -> List[Path]:
    root = project_root or PROJECT_ROOT
    candidates = [root / "datasets" / "raw" / "contracts.csv"]
    candidates.extend(sorted((root / "datasets" / "raw").glob("Контракты_*.csv")))
    candidates.extend(sorted(root.glob("Контракты_*.csv")))
    return candidates


def resolve_raw_ste_path(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    return _first_existing(raw_ste_candidates(root), root / "datasets" / "raw" / "ste_catalog.csv")


def resolve_raw_contracts_path(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    return _first_existing(raw_contract_candidates(root), root / "datasets" / "raw" / "contracts.csv")


def resolve_synonyms_path(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    return _first_existing(
        [
            root / "datasets" / "reference" / "search_synonyms.json",
            root / "data" / "reference" / "search_synonyms.json",
        ],
        root / "datasets" / "reference" / "search_synonyms.json",
    )


def resolve_search_test_queries_path(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    return _first_existing(
        [
            root / "datasets" / "reference" / "search_test_queries.json",
            root / "data" / "reference" / "search_test_queries.json",
        ],
        root / "datasets" / "reference" / "search_test_queries.json",
    )


def resolve_search_db_path(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    return _first_existing(
        [
            root / "datasets" / "processed" / "tenderhack_search.sqlite",
            root / "data" / "processed" / "tenderhack_search.sqlite",
        ],
        root / "datasets" / "processed" / "tenderhack_search.sqlite",
    )


def resolve_preprocessed_db_path(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    return _first_existing(
        [
            root / "datasets" / "processed" / "tenderhack_preprocessed.sqlite",
            root / "data" / "processed" / "tenderhack_preprocessed.sqlite",
        ],
        root / "datasets" / "processed" / "tenderhack_preprocessed.sqlite",
    )


def resolve_fasttext_model_path(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    return _first_existing(
        [
            root / "datasets" / "processed" / "tenderhack_fasttext.bin",
            root / "data" / "processed" / "tenderhack_fasttext.bin",
        ],
        root / "datasets" / "processed" / "tenderhack_fasttext.bin",
    )
