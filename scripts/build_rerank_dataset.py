#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.rerank_dataset import build_rerank_row
from tenderhack.search import SearchService
from tenderhack.text import normalize_text, tokenize, unique_preserve_order


SERVICE_WORDS = {
    "выполнение",
    "закупка",
    "оказание",
    "оказания",
    "приобретение",
    "поставка",
    "поставку",
    "работ",
    "работы",
    "товара",
    "товаров",
    "услуг",
    "услуги",
}

CONTRACT_DATASET_CANDIDATES = [
    Path("data/processed/contracts_clean.csv"),
    Path("data/processed/contracts_flat.csv"),
    Path("data/processed/contracts.csv"),
]


def _resolve_contracts_path(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Contracts CSV was not found: {path}")
        return path

    candidates: List[Path] = [PROJECT_ROOT / path for path in CONTRACT_DATASET_CANDIDATES]
    candidates.extend(sorted(PROJECT_ROOT.glob("Контракты_*.csv")))
    candidates.extend(sorted((PROJECT_ROOT / "data").glob("Контракты_*.csv")))
    candidates.extend(sorted((PROJECT_ROOT / "data").glob("*.csv"), key=lambda path: (path.stat().st_size, str(path))))

    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is None:
        looked_up = "\n".join(f"- {path}" for path in candidates[:8])
        raise FileNotFoundError(
            "Contracts CSV was not found. Looked in these locations:\n"
            f"{looked_up}\n"
            "Pass --contracts-path explicitly if your file lives elsewhere."
        )
    return resolved


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    return ";" if sample.count(";") >= sample.count(",") else ","


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\ufeff", " ").replace("\t", " ").split())


def _normalize_header(value: object) -> str:
    return _clean_text(value).lower().replace(" ", "_")


def _iter_contract_rows(path: Path) -> Iterator[Dict[str, str]]:
    delimiter = _detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter, quotechar='"')
        first_row = next(reader, [])
        normalized_headers = [_normalize_header(value) for value in first_row]
        expected_headers = {
            "contract_item_name",
            "contract_id",
            "ste_id",
            "contract_datetime",
            "contract_amount",
            "customer_inn",
            "customer_name",
            "customer_region",
            "supplier_inn",
            "supplier_name",
            "supplier_region",
        }

        if set(normalized_headers) >= expected_headers:
            header_mapping = {normalized: original for normalized, original in zip(normalized_headers, first_row)}
            dict_reader = csv.DictReader(handle, fieldnames=first_row, delimiter=delimiter, quotechar='"')
            for row in dict_reader:
                yield {
                    "contract_item_name": _clean_text(row.get(header_mapping["contract_item_name"], "")),
                    "contract_id": _clean_text(row.get(header_mapping["contract_id"], "")),
                    "ste_id": _clean_text(row.get(header_mapping["ste_id"], "")),
                    "customer_inn": _clean_text(row.get(header_mapping["customer_inn"], "")),
                    "customer_region": _clean_text(row.get(header_mapping["customer_region"], "")),
                }
            return

        first_cleaned = [_clean_text(value) for value in first_row]
        if len(first_cleaned) >= 8:
            yield {
                "contract_item_name": first_cleaned[0],
                "contract_id": first_cleaned[1],
                "ste_id": first_cleaned[2],
                "customer_inn": first_cleaned[5],
                "customer_region": first_cleaned[7],
            }

        for row in reader:
            cleaned = [_clean_text(value) for value in row]
            if len(cleaned) < 8:
                continue
            yield {
                "contract_item_name": cleaned[0],
                "contract_id": cleaned[1],
                "ste_id": cleaned[2],
                "customer_inn": cleaned[5],
                "customer_region": cleaned[7],
            }


def _query_variants(text: str) -> List[Tuple[str, str]]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    first_clause = re.split(r"[;,]|\s+-\s+|\s{2,}", normalized, maxsplit=1)[0].strip()
    service_stripped = " ".join(token for token in tokenize(normalized) if token not in SERVICE_WORDS).strip()
    compact = " ".join(tokenize(normalized)[:8]).strip()

    variants: List[Tuple[str, str]] = []
    for name, value in [
        ("contract_full", normalized),
        ("contract_first_clause", first_clause),
        ("contract_service_stripped", service_stripped),
        ("contract_compact", compact),
    ]:
        if value and value not in [existing for _variant_name, existing in variants]:
            variants.append((name, value))
    return variants


def write_rerank_dataset(
    *,
    contracts_path: Path,
    search_db_path: Path,
    synonyms_path: Path,
    output_path: Path,
    report_path: Path,
    top_k: int,
    candidate_limit: int,
    max_groups: int,
    semantic_backend: str,
    progress_every: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "groups_seen": 0,
        "groups_written": 0,
        "groups_without_positive": 0,
        "rows_written": 0,
        "avg_candidates_per_group": 0.0,
        "avg_positive_rank": 0.0,
        "query_variant_hit_counts": {},
    }

    total_candidates = 0
    total_positive_rank = 0
    query_variant_hit_counts: Dict[str, int] = {}

    service = SearchService(
        search_db_path=search_db_path,
        synonyms_path=synonyms_path,
        semantic_backend=semantic_backend,
    )
    fieldnames: List[str] | None = None

    try:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer: csv.DictWriter | None = None

            for contract in _iter_contract_rows(contracts_path):
                if stats["groups_seen"] >= max_groups:
                    break
                stats["groups_seen"] += 1

                positive_ste_id = str(contract["ste_id"])
                variants = _query_variants(contract["contract_item_name"])
                best_variant_name = ""
                best_variant_query = ""
                best_payload = None
                best_positive_rank = None

                for variant_name, variant_query in variants:
                    payload = service.search(
                        query=variant_query,
                        top_k=top_k,
                        candidate_limit=candidate_limit,
                    )
                    positive_rank = next(
                        (
                            index
                            for index, item in enumerate(payload["results"], start=1)
                            if str(item.get("ste_id") or "") == positive_ste_id
                        ),
                        None,
                    )
                    if positive_rank is None:
                        continue
                    if best_positive_rank is None or positive_rank < best_positive_rank:
                        best_variant_name = variant_name
                        best_variant_query = variant_query
                        best_payload = payload
                        best_positive_rank = positive_rank

                if best_payload is None or best_positive_rank is None:
                    stats["groups_without_positive"] += 1
                    continue

                query_variant_hit_counts[best_variant_name] = query_variant_hit_counts.get(best_variant_name, 0) + 1
                group_id = f"group-{stats['groups_seen']}"
                rows: List[Dict[str, object]] = []
                for rank, candidate in enumerate(best_payload["results"], start=1):
                    rows.append(
                        build_rerank_row(
                            group_id=group_id,
                            query=best_variant_query,
                            query_meta=dict(best_payload["query"]),
                            contract_id=str(contract["contract_id"]),
                            customer_inn=str(contract["customer_inn"]),
                            customer_region=str(contract["customer_region"]),
                            positive_ste_id=positive_ste_id,
                            candidate=dict(candidate),
                            candidate_rank=rank,
                        )
                    )

                if not rows:
                    stats["groups_without_positive"] += 1
                    continue

                if writer is None:
                    fieldnames = list(rows[0].keys())
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerows(rows)

                stats["groups_written"] += 1
                stats["rows_written"] += len(rows)
                total_candidates += len(rows)
                total_positive_rank += best_positive_rank

                if progress_every > 0 and stats["groups_seen"] % progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "groups_seen": stats["groups_seen"],
                                "groups_written": stats["groups_written"],
                                "groups_without_positive": stats["groups_without_positive"],
                            },
                            ensure_ascii=False,
                        )
                    )
    finally:
        service.close()

    if stats["groups_written"] > 0:
        stats["avg_candidates_per_group"] = round(total_candidates / stats["groups_written"], 4)
        stats["avg_positive_rank"] = round(total_positive_rank / stats["groups_written"], 4)
    stats["query_variant_hit_counts"] = query_variant_hit_counts
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rerank dataset from current search results.")
    parser.add_argument("--contracts-path", default=None, help="Path to contracts CSV. Defaults to the smallest CSV under data/.")
    parser.add_argument("--search-db-path", default="data/processed/tenderhack_search.sqlite")
    parser.add_argument("--synonyms-path", default="data/reference/search_synonyms.json")
    parser.add_argument("--output-path", default="data/processed/rerank_train_current.csv")
    parser.add_argument("--report-path", default="data/processed/rerank_train_current.report.json")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--candidate-limit", type=int, default=150)
    parser.add_argument("--max-groups", type=int, default=3000)
    parser.add_argument("--semantic-backend", choices=["auto", "fasttext", "sqlite"], default="sqlite")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    stats = write_rerank_dataset(
        contracts_path=_resolve_contracts_path(args.contracts_path),
        search_db_path=Path(args.search_db_path),
        synonyms_path=Path(args.synonyms_path),
        output_path=Path(args.output_path),
        report_path=Path(args.report_path),
        top_k=int(args.top_k),
        candidate_limit=int(args.candidate_limit),
        max_groups=int(args.max_groups),
        semantic_backend=str(args.semantic_backend),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
