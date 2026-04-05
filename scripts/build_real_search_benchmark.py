#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.contract_queries import choose_benchmark_query, iter_contract_rows, resolve_contracts_path


def build_real_search_benchmark(
    *,
    contracts_path: Path,
    search_db_path: Path,
    output_path: Path,
    min_query_support: int,
    min_positive_support: int,
    min_dominant_share: float,
    max_positive_ste_ids: int,
    max_queries: int,
    progress_every: int,
) -> Dict[str, object]:
    conn = sqlite3.connect(search_db_path)
    try:
        known_ste_ids = {
            str(row[0])
            for row in conn.execute("SELECT ste_id FROM ste_catalog").fetchall()
        }
    finally:
        conn.close()

    query_ste_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    query_customers: Dict[str, set[str]] = defaultdict(set)
    rows_seen = 0
    skipped_unknown_ste = 0
    skipped_empty_query = 0

    for contract in iter_contract_rows(contracts_path):
        rows_seen += 1
        ste_id = str(contract["ste_id"] or "")
        if not ste_id or ste_id not in known_ste_ids:
            skipped_unknown_ste += 1
            continue
        selected_query = choose_benchmark_query(contract["contract_item_name"])
        if selected_query is None:
            skipped_empty_query += 1
            continue
        _query_variant, query = selected_query
        query_ste_counts[query][ste_id] += 1
        customer_inn = str(contract.get("customer_inn") or "")
        if customer_inn:
            query_customers[query].add(customer_inn)
        if progress_every > 0 and rows_seen % progress_every == 0:
            print(
                json.dumps(
                    {
                        "rows_seen": rows_seen,
                        "unique_queries": len(query_ste_counts),
                        "skipped_unknown_ste": skipped_unknown_ste,
                        "skipped_empty_query": skipped_empty_query,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    benchmark_items: List[Dict[str, object]] = []
    for query, ste_counter in query_ste_counts.items():
        total_support = sum(ste_counter.values())
        if total_support < min_query_support:
            continue
        ranked = ste_counter.most_common(max_positive_ste_ids)
        top_ste_id, top_support = ranked[0]
        dominant_share = top_support / total_support
        if top_support < min_positive_support:
            continue
        if dominant_share < min_dominant_share:
            continue

        support_floor = max(min_positive_support, int(round(top_support * 0.25)))
        positive_ste_ids = [ste_id for ste_id, support in ranked if support >= support_floor][:max_positive_ste_ids]
        if not positive_ste_ids:
            continue
        benchmark_items.append(
            {
                "query": query,
                "positive_ste_ids": positive_ste_ids,
                "query_support": total_support,
                "positive_support": top_support,
                "dominant_share": round(dominant_share, 4),
                "unique_customers": len(query_customers.get(query, set())),
                "ste_support": {ste_id: support for ste_id, support in ranked},
            }
        )

    benchmark_items.sort(
        key=lambda item: (
            int(item["query_support"]),
            float(item["dominant_share"]),
            int(item["unique_customers"]),
            str(item["query"]),
        ),
        reverse=True,
    )
    if max_queries > 0:
        benchmark_items = benchmark_items[:max_queries]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "rows_seen": rows_seen,
        "benchmark_items": len(benchmark_items),
        "unique_queries_seen": len(query_ste_counts),
        "skipped_unknown_ste": skipped_unknown_ste,
        "skipped_empty_query": skipped_empty_query,
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real search benchmark from contracts CSV.")
    parser.add_argument("--contracts-path", default=None)
    parser.add_argument("--search-db-path", default="data/processed/tenderhack_search.sqlite")
    parser.add_argument("--output-path", default="data/reference/search_real_benchmark.json")
    parser.add_argument("--min-query-support", type=int, default=3)
    parser.add_argument("--min-positive-support", type=int, default=2)
    parser.add_argument("--min-dominant-share", type=float, default=0.6)
    parser.add_argument("--max-positive-ste-ids", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=2500)
    parser.add_argument("--progress-every", type=int, default=250000)
    args = parser.parse_args()

    summary = build_real_search_benchmark(
        contracts_path=resolve_contracts_path(PROJECT_ROOT, args.contracts_path),
        search_db_path=Path(args.search_db_path),
        output_path=Path(args.output_path),
        min_query_support=int(args.min_query_support),
        min_positive_support=int(args.min_positive_support),
        min_dominant_share=float(args.min_dominant_share),
        max_positive_ste_ids=int(args.max_positive_ste_ids),
        max_queries=int(args.max_queries),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
