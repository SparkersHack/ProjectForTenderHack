from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from .text import normalize_text, tokenize


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


def resolve_contracts_path(project_root: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    candidates = sorted(project_root.glob("Контракты_*.csv"))
    if candidates:
        return candidates[0]
    data_candidates = sorted((project_root / "data").glob("*.csv"), key=lambda path: path.stat().st_size)
    if data_candidates:
        return data_candidates[0]
    raise FileNotFoundError("Contracts CSV was not found.")


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    return ";" if sample.count(";") >= sample.count(",") else ","


def clean_contract_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\ufeff", " ").replace("\t", " ").split())


def normalize_contract_header(value: object) -> str:
    return clean_contract_text(value).lower().replace(" ", "_")


def iter_contract_rows(path: Path) -> Iterator[Dict[str, str]]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter, quotechar='"')
        first_row = next(reader, [])
        normalized_headers = [normalize_contract_header(value) for value in first_row]
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
                    "contract_item_name": clean_contract_text(row.get(header_mapping["contract_item_name"], "")),
                    "contract_id": clean_contract_text(row.get(header_mapping["contract_id"], "")),
                    "ste_id": clean_contract_text(row.get(header_mapping["ste_id"], "")),
                    "customer_inn": clean_contract_text(row.get(header_mapping["customer_inn"], "")),
                    "customer_region": clean_contract_text(row.get(header_mapping["customer_region"], "")),
                }
            return

        first_cleaned = [clean_contract_text(value) for value in first_row]
        if len(first_cleaned) >= 8:
            yield {
                "contract_item_name": first_cleaned[0],
                "contract_id": first_cleaned[1],
                "ste_id": first_cleaned[2],
                "customer_inn": first_cleaned[5],
                "customer_region": first_cleaned[7],
            }

        for row in reader:
            cleaned = [clean_contract_text(value) for value in row]
            if len(cleaned) < 8:
                continue
            yield {
                "contract_item_name": cleaned[0],
                "contract_id": cleaned[1],
                "ste_id": cleaned[2],
                "customer_inn": cleaned[5],
                "customer_region": cleaned[7],
            }


def query_variants(text: str) -> List[Tuple[str, str]]:
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


def choose_benchmark_query(text: str) -> Tuple[str, str] | None:
    variants = query_variants(text)
    if not variants:
        return None

    ranked: List[Tuple[Tuple[int, int, int, str], Tuple[str, str]]] = []
    priority = {
        "contract_service_stripped": 4,
        "contract_first_clause": 3,
        "contract_compact": 2,
        "contract_full": 1,
    }
    for variant_name, variant_query in variants:
        tokens = tokenize(variant_query)
        if not tokens:
            continue
        informative_tokens = [token for token in tokens if token not in SERVICE_WORDS]
        informative_count = len(informative_tokens)
        if informative_count == 0:
            continue
        token_count = len(tokens)
        if token_count > 12:
            continue
        score = (
            priority.get(variant_name, 0),
            informative_count,
            -token_count,
            variant_query,
        )
        ranked.append((score, (variant_name, variant_query)))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1]
