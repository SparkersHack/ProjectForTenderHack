#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.search import SearchService
from tenderhack.text import normalize_text


DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "reference" / "search_relevance_benchmark.json"


def _matches_expectation(item: Dict[str, object], benchmark_item: Dict[str, object]) -> bool:
    positive_ste_ids = {str(value) for value in benchmark_item.get("positive_ste_ids", []) if str(value)}
    if positive_ste_ids:
        return str(item.get("ste_id") or "") in positive_ste_ids

    haystack = " ".join(
        normalize_text(str(value or ""))
        for value in (
            item.get("clean_name"),
            item.get("normalized_name"),
            item.get("category"),
            item.get("normalized_category"),
            item.get("key_tokens"),
        )
    )
    match_all = [normalize_text(value) for value in benchmark_item.get("match_all", []) if normalize_text(value)]
    match_any = [normalize_text(value) for value in benchmark_item.get("match_any", []) if normalize_text(value)]
    if match_all and not all(value in haystack for value in match_all):
        return False
    if match_any and not any(value in haystack for value in match_any):
        return False
    return bool(match_all or match_any)


def evaluate_benchmark(
    *,
    benchmark_path: Path,
    search_db_path: Path,
    synonyms_path: Path,
    semantic_backend: str,
    top_k: int,
    max_items: int | None = None,
    progress_every: int = 0,
) -> Dict[str, object]:
    benchmark_items = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if max_items is not None and max_items > 0:
        benchmark_items = benchmark_items[:max_items]
    service = SearchService(
        search_db_path=search_db_path,
        synonyms_path=synonyms_path,
        semantic_backend=semantic_backend,
    )
    try:
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_10 = 0
        reciprocal_rank_sum = 0.0
        evaluated_items: List[Dict[str, object]] = []

        for index, benchmark_item in enumerate(benchmark_items, start=1):
            query = str(benchmark_item["query"])
            payload = service.search(query, top_k=top_k, candidate_limit=max(100, top_k * 4))
            positive_rank = None
            for index, item in enumerate(payload["results"], start=1):
                if _matches_expectation(item, benchmark_item):
                    positive_rank = index
                    break
            if positive_rank == 1:
                hit_at_1 += 1
            if positive_rank is not None and positive_rank <= 3:
                hit_at_3 += 1
            if positive_rank is not None and positive_rank <= 10:
                hit_at_10 += 1
                reciprocal_rank_sum += 1.0 / positive_rank
            evaluated_items.append(
                {
                    "query": query,
                    "type": benchmark_item.get("type", "unknown"),
                    "positive_rank": positive_rank,
                    "positive_ste_ids": benchmark_item.get("positive_ste_ids"),
                    "dense_backend": payload["query"].get("dense_retrieval_backend"),
                    "semantic_backend": payload["query"].get("semantic_backend"),
                    "top_result": payload["results"][0]["clean_name"] if payload["results"] else None,
                }
            )
            if progress_every > 0 and index % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "evaluated": index,
                            "hit_rate_at_10": round(hit_at_10 / index, 4),
                            "mrr_at_10": round(reciprocal_rank_sum / index, 4),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        total = max(1, len(benchmark_items))
        return {
            "benchmark_items": len(benchmark_items),
            "hit_rate_at_1": round(hit_at_1 / total, 4),
            "hit_rate_at_3": round(hit_at_3 / total, 4),
            "hit_rate_at_10": round(hit_at_10 / total, 4),
            "mrr_at_10": round(reciprocal_rank_sum / total, 4),
            "items": evaluated_items,
        }
    finally:
        service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search relevance benchmark.")
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--search-db-path", default="data/processed/tenderhack_search.sqlite")
    parser.add_argument("--synonyms-path", default="data/reference/search_synonyms.json")
    parser.add_argument("--semantic-backend", choices=["auto", "fasttext", "sqlite"], default="sqlite")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    report = evaluate_benchmark(
        benchmark_path=Path(args.benchmark_path),
        search_db_path=Path(args.search_db_path),
        synonyms_path=Path(args.synonyms_path),
        semantic_backend=str(args.semantic_backend),
        top_k=int(args.top_k),
        max_items=int(args.max_items) if args.max_items else None,
        progress_every=int(args.progress_every),
    )
    if args.report_path:
        Path(args.report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.summary_only:
        summary = dict(report)
        summary.pop("items", None)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
