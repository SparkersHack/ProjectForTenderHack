#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.rerank_dataset import build_rerank_row
from tenderhack.search import SearchService
from tenderhack.text import clean_text, normalize_text


def discover_contracts_path() -> Path:
    data_dir = PROJECT_ROOT / "data"
    preferred_names = (
        "Контракты_20260403.csv",
        "contracts.csv",
    )
    for name in preferred_names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    for pattern in ("*Контракт*.csv", "*контракт*.csv", "*contracts*.csv", "*.csv"):
        matches = sorted(path for path in data_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return data_dir / "Контракты_20260403.csv"


DEFAULT_CONTRACTS_PATH = discover_contracts_path()
DEFAULT_SEARCH_DB_PATH = PROJECT_ROOT / "data" / "processed" / "tenderhack_search.sqlite"
DEFAULT_SYNONYMS_PATH = PROJECT_ROOT / "data" / "reference" / "search_synonyms.json"
DEFAULT_FASTTEXT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "tenderhack_fasttext.bin"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rerank_train.csv"


def iter_unique_contract_examples(
    contracts_path: Path,
    max_groups: int,
) -> Iterator[Dict[str, str]]:
    seen = set()
    yielded = 0
    with contracts_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";", quotechar='"')
        for row in reader:
            if len(row) != 11:
                continue
            query = clean_text(row[0])
            contract_id = clean_text(row[1])
            ste_id = clean_text(row[2])
            customer_inn = clean_text(row[5])
            customer_region = clean_text(row[7])
            if not query or not ste_id:
                continue
            normalized_query = normalize_text(query)
            if not normalized_query:
                continue
            dedupe_key = (normalized_query, ste_id, customer_inn)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            yielded += 1
            yield {
                "query": query,
                "normalized_query": normalized_query,
                "contract_id": contract_id,
                "positive_ste_id": ste_id,
                "customer_inn": customer_inn,
                "customer_region": customer_region,
            }
            if max_groups and yielded >= max_groups:
                return


def write_rerank_dataset(
    *,
    contracts_path: Path,
    output_path: Path,
    report_path: Path,
    search_db_path: Path,
    synonyms_path: Path,
    semantic_backend: str,
    fasttext_model_path: Path,
    top_k: int,
    candidate_limit: int,
    max_groups: int,
    progress_every: int,
) -> Dict[str, object]:
    service = SearchService(
        search_db_path=search_db_path,
        synonyms_path=synonyms_path,
        semantic_backend=semantic_backend,
        fasttext_model_path=fasttext_model_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "groups_seen": 0,
        "groups_written": 0,
        "groups_without_positive": 0,
        "rows_written": 0,
        "avg_candidates_per_group": 0.0,
        "avg_positive_rank": 0.0,
    }
    total_candidates = 0
    positive_ranks: List[int] = []
    writer = None
    handle = None
    temp_output_path = output_path.with_name(f"{output_path.name}.tmp")
    if temp_output_path.exists():
        temp_output_path.unlink()

    try:
        handle = temp_output_path.open("w", encoding="utf-8", newline="")
        for group_index, example in enumerate(
            iter_unique_contract_examples(contracts_path=contracts_path, max_groups=max_groups),
            start=1,
        ):
            stats["groups_seen"] += 1
            payload = service.search(
                query=example["query"],
                top_k=top_k,
                candidate_limit=max(candidate_limit, top_k),
            )
            results = list(payload["results"])
            positive_ste_id = example["positive_ste_id"]
            positive_rank = None
            for rank, candidate in enumerate(results, start=1):
                if str(candidate.get("ste_id") or "") == positive_ste_id:
                    positive_rank = rank
                    break
            if positive_rank is None:
                stats["groups_without_positive"] += 1
                if progress_every and stats["groups_seen"] % progress_every == 0:
                    _print_progress(stats)
                continue

            rows = [
                build_rerank_row(
                    group_id=f"group-{group_index}",
                    query=example["query"],
                    query_meta=payload["query"],
                    contract_id=example["contract_id"],
                    customer_inn=example["customer_inn"],
                    customer_region=example["customer_region"],
                    positive_ste_id=positive_ste_id,
                    candidate=candidate,
                    candidate_rank=rank,
                )
                for rank, candidate in enumerate(results, start=1)
            ]
            if writer is None:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerows(rows)

            stats["groups_written"] += 1
            stats["rows_written"] += len(rows)
            total_candidates += len(rows)
            positive_ranks.append(positive_rank)

            if progress_every and stats["groups_seen"] % progress_every == 0:
                _print_progress(stats)

        if stats["groups_written"]:
            stats["avg_candidates_per_group"] = round(total_candidates / stats["groups_written"], 4)
            stats["avg_positive_rank"] = round(sum(positive_ranks) / len(positive_ranks), 4)
    finally:
        service.close()
        if handle is not None:
            handle.close()

    os.replace(temp_output_path, output_path)
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _print_progress(stats: Dict[str, object]) -> None:
    groups_seen = int(stats["groups_seen"])
    groups_written = int(stats["groups_written"])
    rows_written = int(stats["rows_written"])
    hit_rate = groups_written / max(groups_seen, 1)
    print(
        "[progress]",
        f"seen={groups_seen}",
        f"written={groups_written}",
        f"hit_rate={hit_rate:.3f}",
        f"rows={rows_written}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grouped ranking dataset for CatBoost YetiRank.")
    parser.add_argument("--contracts-path", default=str(DEFAULT_CONTRACTS_PATH))
    parser.add_argument("--search-db-path", default=str(DEFAULT_SEARCH_DB_PATH))
    parser.add_argument("--synonyms-path", default=str(DEFAULT_SYNONYMS_PATH))
    parser.add_argument("--fasttext-model-path", default=str(DEFAULT_FASTTEXT_MODEL_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report-path", help="Optional JSON report path.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--max-groups", type=int, default=3000)
    parser.add_argument("--semantic-backend", choices=["auto", "fasttext", "sqlite"], default="auto")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    report_path = Path(args.report_path) if args.report_path else output_path.with_suffix(".report.json")
    stats = write_rerank_dataset(
        contracts_path=Path(args.contracts_path),
        output_path=output_path,
        report_path=report_path,
        search_db_path=Path(args.search_db_path),
        synonyms_path=Path(args.synonyms_path),
        semantic_backend=args.semantic_backend,
        fasttext_model_path=Path(args.fasttext_model_path),
        top_k=args.top_k,
        candidate_limit=args.candidate_limit,
        max_groups=args.max_groups,
        progress_every=args.progress_every,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nSaved dataset to: {output_path}")
    print(f"Saved report to:  {report_path}")


if __name__ == "__main__":
    main()
