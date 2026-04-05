#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
WHITESPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "по",
    "под",
    "над",
    "для",
    "с",
    "со",
    "к",
    "ко",
    "из",
    "от",
    "до",
    "без",
    "при",
    "или",
    "а",
    "но",
    "не",
    "ни",
    "о",
    "об",
    "у",
    "же",
    "ли",
    "бы",
    "the",
    "and",
    "for",
    "of",
    "to",
    "n",
}

STE_COLUMNS = ["ste_id", "raw_name", "raw_category", "raw_attributes"]
CONTRACT_COLUMNS = [
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
]


@dataclass
class AggregateStats:
    count: int = 0
    total_amount: float = 0.0
    first_date: str = ""
    last_date: str = ""
    category_id: int = 0

    def update(self, amount: float, dt: str, category_id: int) -> None:
        self.count += 1
        self.total_amount += amount
        self.category_id = category_id
        if dt:
            if not self.first_date or dt < self.first_date:
                self.first_date = dt
            if not self.last_date or dt > self.last_date:
                self.last_date = dt


def clean_text(value: str) -> str:
    if value is None:
        return ""
    value = value.replace("\ufeff", " ").replace("\t", " ").replace("\n", " ").replace("\r", " ")
    value = value.strip().strip('"').strip()
    return WHITESPACE_RE.sub(" ", value)


def normalize_for_search(value: str) -> str:
    value = clean_text(value).lower().replace("ё", "е")
    return " ".join(TOKEN_RE.findall(value))


def parse_attributes(raw_attributes: str) -> Tuple[List[str], List[str]]:
    keys: List[str] = []
    values: List[str] = []
    raw_attributes = clean_text(raw_attributes)
    if not raw_attributes:
        return keys, values
    for chunk in raw_attributes.split(";"):
        chunk = clean_text(chunk)
        if not chunk:
            continue
        if ":" in chunk:
            key, value = chunk.split(":", 1)
        else:
            key, value = chunk, ""
        key = clean_text(key)
        value = clean_text(value)
        if key:
            keys.append(key)
        if value:
            values.append(value)
    return keys, values


def extract_keywords(*parts: str, limit: int = 24) -> str:
    seen = set()
    result: List[str] = []
    merged = " ".join(part for part in parts if part)
    for token in TOKEN_RE.findall(merged.lower().replace("ё", "е")):
        if token in STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return " ".join(result)


def parse_float(value: str) -> float:
    value = clean_text(value).replace(",", ".")
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def build_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;

        CREATE TABLE IF NOT EXISTS category_lookup (
            category_id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            normalized_category TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS customer_ste_stats (
            customer_inn TEXT NOT NULL,
            ste_id TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            purchase_count INTEGER NOT NULL,
            total_amount REAL NOT NULL,
            first_purchase_dt TEXT,
            last_purchase_dt TEXT,
            PRIMARY KEY (customer_inn, ste_id)
        );

        CREATE TABLE IF NOT EXISTS customer_category_stats (
            customer_inn TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            purchase_count INTEGER NOT NULL,
            total_amount REAL NOT NULL,
            first_purchase_dt TEXT,
            last_purchase_dt TEXT,
            PRIMARY KEY (customer_inn, category_id)
        );

        CREATE TABLE IF NOT EXISTS supplier_ste_stats (
            supplier_inn TEXT NOT NULL,
            ste_id TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            purchase_count INTEGER NOT NULL,
            total_amount REAL NOT NULL,
            first_purchase_dt TEXT,
            last_purchase_dt TEXT,
            PRIMARY KEY (supplier_inn, ste_id)
        );

        CREATE TABLE IF NOT EXISTS supplier_category_stats (
            supplier_inn TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            purchase_count INTEGER NOT NULL,
            total_amount REAL NOT NULL,
            first_purchase_dt TEXT,
            last_purchase_dt TEXT,
            PRIMARY KEY (supplier_inn, category_id)
        );

        CREATE TABLE IF NOT EXISTS supplier_region_lookup (
            supplier_inn TEXT PRIMARY KEY,
            supplier_region TEXT NOT NULL,
            frequency INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS customer_name_lookup (
            customer_inn TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS supplier_name_lookup (
            supplier_inn TEXT PRIMARY KEY,
            supplier_name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS region_category_stats (
            customer_region TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            purchase_count INTEGER NOT NULL,
            total_amount REAL NOT NULL,
            first_purchase_dt TEXT,
            last_purchase_dt TEXT,
            PRIMARY KEY (customer_region, category_id)
        );

        CREATE TABLE IF NOT EXISTS contract_key_stats (
            contract_id TEXT NOT NULL,
            ste_id TEXT NOT NULL,
            customer_inn TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            PRIMARY KEY (contract_id, ste_id, customer_inn)
        );
        """
    )
    conn.commit()


def flush_customer_ste(
    conn: sqlite3.Connection,
    payload: Dict[Tuple[str, str], AggregateStats],
) -> None:
    if not payload:
        return
    rows = [
        (
            customer_inn,
            ste_id,
            agg.category_id,
            agg.count,
            agg.total_amount,
            agg.first_date,
            agg.last_date,
        )
        for (customer_inn, ste_id), agg in payload.items()
    ]
    conn.executemany(
        """
        INSERT INTO customer_ste_stats (
            customer_inn, ste_id, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(customer_inn, ste_id) DO UPDATE SET
            purchase_count = customer_ste_stats.purchase_count + excluded.purchase_count,
            total_amount = customer_ste_stats.total_amount + excluded.total_amount,
            first_purchase_dt = CASE
                WHEN customer_ste_stats.first_purchase_dt IS NULL OR customer_ste_stats.first_purchase_dt = '' THEN excluded.first_purchase_dt
                WHEN excluded.first_purchase_dt IS NULL OR excluded.first_purchase_dt = '' THEN customer_ste_stats.first_purchase_dt
                WHEN excluded.first_purchase_dt < customer_ste_stats.first_purchase_dt THEN excluded.first_purchase_dt
                ELSE customer_ste_stats.first_purchase_dt
            END,
            last_purchase_dt = CASE
                WHEN customer_ste_stats.last_purchase_dt IS NULL OR customer_ste_stats.last_purchase_dt = '' THEN excluded.last_purchase_dt
                WHEN excluded.last_purchase_dt IS NULL OR excluded.last_purchase_dt = '' THEN customer_ste_stats.last_purchase_dt
                WHEN excluded.last_purchase_dt > customer_ste_stats.last_purchase_dt THEN excluded.last_purchase_dt
                ELSE customer_ste_stats.last_purchase_dt
            END
        """,
        rows,
    )
    conn.commit()
    payload.clear()


def flush_customer_category(
    conn: sqlite3.Connection,
    payload: Dict[Tuple[str, int], AggregateStats],
) -> None:
    if not payload:
        return
    rows = [
        (
            customer_inn,
            category_id,
            agg.count,
            agg.total_amount,
            agg.first_date,
            agg.last_date,
        )
        for (customer_inn, category_id), agg in payload.items()
    ]
    conn.executemany(
        """
        INSERT INTO customer_category_stats (
            customer_inn, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(customer_inn, category_id) DO UPDATE SET
            purchase_count = customer_category_stats.purchase_count + excluded.purchase_count,
            total_amount = customer_category_stats.total_amount + excluded.total_amount,
            first_purchase_dt = CASE
                WHEN customer_category_stats.first_purchase_dt IS NULL OR customer_category_stats.first_purchase_dt = '' THEN excluded.first_purchase_dt
                WHEN excluded.first_purchase_dt IS NULL OR excluded.first_purchase_dt = '' THEN customer_category_stats.first_purchase_dt
                WHEN excluded.first_purchase_dt < customer_category_stats.first_purchase_dt THEN excluded.first_purchase_dt
                ELSE customer_category_stats.first_purchase_dt
            END,
            last_purchase_dt = CASE
                WHEN customer_category_stats.last_purchase_dt IS NULL OR customer_category_stats.last_purchase_dt = '' THEN excluded.last_purchase_dt
                WHEN excluded.last_purchase_dt IS NULL OR excluded.last_purchase_dt = '' THEN customer_category_stats.last_purchase_dt
                WHEN excluded.last_purchase_dt > customer_category_stats.last_purchase_dt THEN excluded.last_purchase_dt
                ELSE customer_category_stats.last_purchase_dt
            END
        """,
        rows,
    )
    conn.commit()
    payload.clear()


def flush_supplier_ste(
    conn: sqlite3.Connection,
    payload: Dict[Tuple[str, str], AggregateStats],
) -> None:
    if not payload:
        return
    rows = [
        (
            supplier_inn,
            ste_id,
            agg.category_id,
            agg.count,
            agg.total_amount,
            agg.first_date,
            agg.last_date,
        )
        for (supplier_inn, ste_id), agg in payload.items()
    ]
    conn.executemany(
        """
        INSERT INTO supplier_ste_stats (
            supplier_inn, ste_id, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(supplier_inn, ste_id) DO UPDATE SET
            purchase_count = supplier_ste_stats.purchase_count + excluded.purchase_count,
            total_amount = supplier_ste_stats.total_amount + excluded.total_amount,
            first_purchase_dt = CASE
                WHEN supplier_ste_stats.first_purchase_dt IS NULL OR supplier_ste_stats.first_purchase_dt = '' THEN excluded.first_purchase_dt
                WHEN excluded.first_purchase_dt IS NULL OR excluded.first_purchase_dt = '' THEN supplier_ste_stats.first_purchase_dt
                WHEN excluded.first_purchase_dt < supplier_ste_stats.first_purchase_dt THEN excluded.first_purchase_dt
                ELSE supplier_ste_stats.first_purchase_dt
            END,
            last_purchase_dt = CASE
                WHEN supplier_ste_stats.last_purchase_dt IS NULL OR supplier_ste_stats.last_purchase_dt = '' THEN excluded.last_purchase_dt
                WHEN excluded.last_purchase_dt IS NULL OR excluded.last_purchase_dt = '' THEN supplier_ste_stats.last_purchase_dt
                WHEN excluded.last_purchase_dt > supplier_ste_stats.last_purchase_dt THEN excluded.last_purchase_dt
                ELSE supplier_ste_stats.last_purchase_dt
            END
        """,
        rows,
    )
    conn.commit()
    payload.clear()


def flush_supplier_category(
    conn: sqlite3.Connection,
    payload: Dict[Tuple[str, int], AggregateStats],
) -> None:
    if not payload:
        return
    rows = [
        (
            supplier_inn,
            category_id,
            agg.count,
            agg.total_amount,
            agg.first_date,
            agg.last_date,
        )
        for (supplier_inn, category_id), agg in payload.items()
    ]
    conn.executemany(
        """
        INSERT INTO supplier_category_stats (
            supplier_inn, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(supplier_inn, category_id) DO UPDATE SET
            purchase_count = supplier_category_stats.purchase_count + excluded.purchase_count,
            total_amount = supplier_category_stats.total_amount + excluded.total_amount,
            first_purchase_dt = CASE
                WHEN supplier_category_stats.first_purchase_dt IS NULL OR supplier_category_stats.first_purchase_dt = '' THEN excluded.first_purchase_dt
                WHEN excluded.first_purchase_dt IS NULL OR excluded.first_purchase_dt = '' THEN supplier_category_stats.first_purchase_dt
                WHEN excluded.first_purchase_dt < supplier_category_stats.first_purchase_dt THEN excluded.first_purchase_dt
                ELSE supplier_category_stats.first_purchase_dt
            END,
            last_purchase_dt = CASE
                WHEN supplier_category_stats.last_purchase_dt IS NULL OR supplier_category_stats.last_purchase_dt = '' THEN excluded.last_purchase_dt
                WHEN excluded.last_purchase_dt IS NULL OR excluded.last_purchase_dt = '' THEN supplier_category_stats.last_purchase_dt
                WHEN excluded.last_purchase_dt > supplier_category_stats.last_purchase_dt THEN excluded.last_purchase_dt
                ELSE supplier_category_stats.last_purchase_dt
            END
        """,
        rows,
    )
    conn.commit()
    payload.clear()


def flush_region_category(
    conn: sqlite3.Connection,
    payload: Dict[Tuple[str, int], AggregateStats],
) -> None:
    if not payload:
        return
    rows = [
        (
            customer_region,
            category_id,
            agg.count,
            agg.total_amount,
            agg.first_date,
            agg.last_date,
        )
        for (customer_region, category_id), agg in payload.items()
    ]
    conn.executemany(
        """
        INSERT INTO region_category_stats (
            customer_region, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(customer_region, category_id) DO UPDATE SET
            purchase_count = region_category_stats.purchase_count + excluded.purchase_count,
            total_amount = region_category_stats.total_amount + excluded.total_amount,
            first_purchase_dt = CASE
                WHEN region_category_stats.first_purchase_dt IS NULL OR region_category_stats.first_purchase_dt = '' THEN excluded.first_purchase_dt
                WHEN excluded.first_purchase_dt IS NULL OR excluded.first_purchase_dt = '' THEN region_category_stats.first_purchase_dt
                WHEN excluded.first_purchase_dt < region_category_stats.first_purchase_dt THEN excluded.first_purchase_dt
                ELSE region_category_stats.first_purchase_dt
            END,
            last_purchase_dt = CASE
                WHEN region_category_stats.last_purchase_dt IS NULL OR region_category_stats.last_purchase_dt = '' THEN excluded.last_purchase_dt
                WHEN excluded.last_purchase_dt IS NULL OR excluded.last_purchase_dt = '' THEN region_category_stats.last_purchase_dt
                WHEN excluded.last_purchase_dt > region_category_stats.last_purchase_dt THEN excluded.last_purchase_dt
                ELSE region_category_stats.last_purchase_dt
            END
        """,
        rows,
    )
    conn.commit()
    payload.clear()


def flush_contract_key_stats(conn: sqlite3.Connection, payload: Dict[Tuple[str, str, str], int]) -> None:
    if not payload:
        return
    rows = [(contract_id, ste_id, customer_inn, row_count) for (contract_id, ste_id, customer_inn), row_count in payload.items()]
    conn.executemany(
        """
        INSERT INTO contract_key_stats (contract_id, ste_id, customer_inn, row_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(contract_id, ste_id, customer_inn) DO UPDATE SET
            row_count = contract_key_stats.row_count + excluded.row_count
        """,
        rows,
    )
    conn.commit()
    payload.clear()


def export_query_to_csv(conn: sqlite3.Connection, query: str, output_path: Path) -> None:
    cursor = conn.execute(query)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([column[0] for column in cursor.description])
        writer.writerows(cursor)


def process_ste_catalog(
    ste_path: Path,
    output_dir: Path,
    conn: sqlite3.Connection,
) -> Tuple[dict, Dict[str, int], Dict[int, str]]:
    stats = {
        "encoding": "utf-8-sig",
        "delimiter": ";",
        "expected_columns": len(STE_COLUMNS),
        "rows_total": 0,
        "rows_valid": 0,
        "rows_invalid": 0,
        "rows_with_empty_name": 0,
        "rows_with_empty_category": 0,
        "rows_with_empty_attributes": 0,
        "duplicate_ste_id_rows": 0,
        "unique_ste_ids": 0,
        "avg_name_length": 0.0,
        "avg_attribute_count": 0.0,
        "top_categories": [],
        "sample_rows": [],
    }
    category_to_id: Dict[str, int] = {}
    category_id_to_name: Dict[int, str] = {}
    ste_id_to_category_id: Dict[str, int] = {}
    category_counter: Counter[str] = Counter()
    seen_ste_ids = set()
    total_name_length = 0
    total_attribute_count = 0

    ste_output_path = output_dir / "ste_catalog_clean.csv"
    with ste_path.open("r", encoding="utf-8-sig", newline="") as source, ste_output_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.reader(source, delimiter=";", quotechar='"')
        writer = csv.writer(target)
        writer.writerow(
            [
                "ste_id",
                "clean_name",
                "normalized_name",
                "category",
                "normalized_category",
                "attribute_keys",
                "attribute_count",
                "key_tokens",
            ]
        )

        for row_index, row in enumerate(reader, start=1):
            stats["rows_total"] += 1
            if len(row) != len(STE_COLUMNS):
                stats["rows_invalid"] += 1
                continue

            stats["rows_valid"] += 1
            ste_id, raw_name, raw_category, raw_attributes = row
            ste_id = clean_text(ste_id)
            raw_name = clean_text(raw_name)
            raw_category = clean_text(raw_category)
            raw_attributes = clean_text(raw_attributes)

            if not raw_name:
                stats["rows_with_empty_name"] += 1
            if not raw_category:
                stats["rows_with_empty_category"] += 1
            if not raw_attributes:
                stats["rows_with_empty_attributes"] += 1

            if ste_id in seen_ste_ids:
                stats["duplicate_ste_id_rows"] += 1
            else:
                seen_ste_ids.add(ste_id)

            normalized_name = normalize_for_search(raw_name)
            normalized_category = normalize_for_search(raw_category)
            attribute_keys, attribute_values = parse_attributes(raw_attributes)
            attribute_count = len(attribute_keys)
            attribute_keys_joined = " | ".join(attribute_keys)
            attributes_text = " ".join(attribute_values)
            key_tokens = extract_keywords(raw_name, raw_category, attribute_keys_joined, attributes_text)

            if raw_category not in category_to_id:
                category_id = len(category_to_id) + 1
                category_to_id[raw_category] = category_id
                category_id_to_name[category_id] = raw_category
            else:
                category_id = category_to_id[raw_category]

            ste_id_to_category_id[ste_id] = category_id
            category_counter[raw_category] += 1
            total_name_length += len(raw_name)
            total_attribute_count += attribute_count

            writer.writerow(
                [
                    ste_id,
                    raw_name,
                    normalized_name,
                    raw_category,
                    normalized_category,
                    attribute_keys_joined,
                    attribute_count,
                    key_tokens,
                ]
            )

            if len(stats["sample_rows"]) < 5:
                stats["sample_rows"].append(
                    {
                        "ste_id": ste_id,
                        "clean_name": raw_name,
                        "category": raw_category,
                        "attribute_count": attribute_count,
                        "key_tokens": key_tokens,
                    }
                )

            if row_index % 100_000 == 0:
                print(f"[STE] processed {row_index:,} rows", flush=True)

    stats["unique_ste_ids"] = len(seen_ste_ids)
    if stats["rows_valid"]:
        stats["avg_name_length"] = round(total_name_length / stats["rows_valid"], 2)
        stats["avg_attribute_count"] = round(total_attribute_count / stats["rows_valid"], 2)
    stats["top_categories"] = [{"category": name, "count": count} for name, count in category_counter.most_common(15)]

    category_lookup_rows = [
        (category_id, category_name, normalize_for_search(category_name))
        for category_name, category_id in category_to_id.items()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO category_lookup (category_id, category, normalized_category) VALUES (?, ?, ?)",
        category_lookup_rows,
    )
    conn.commit()

    category_lookup_path = output_dir / "category_lookup.csv"
    with category_lookup_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category_id", "category", "normalized_category"])
        for category_name, category_id in sorted(category_to_id.items(), key=lambda item: item[1]):
            writer.writerow([category_id, category_name, normalize_for_search(category_name)])

    return stats, ste_id_to_category_id, category_id_to_name


def process_contracts(
    contracts_path: Path,
    conn: sqlite3.Connection,
    ste_id_to_category_id: Dict[str, int],
    category_id_to_name: Dict[int, str],
    flush_threshold: int = 200_000,
) -> dict:
    unknown_category_id = 0
    if unknown_category_id not in category_id_to_name:
        category_id_to_name[unknown_category_id] = "UNKNOWN"
        conn.execute(
            "INSERT OR REPLACE INTO category_lookup (category_id, category, normalized_category) VALUES (?, ?, ?)",
            (unknown_category_id, "UNKNOWN", "unknown"),
        )
        conn.commit()

    stats = {
        "encoding": "utf-8-sig",
        "delimiter": ";",
        "expected_columns": len(CONTRACT_COLUMNS),
        "rows_total": 0,
        "rows_valid": 0,
        "rows_invalid": 0,
        "rows_with_missing_ste_reference": 0,
        "rows_with_empty_customer_inn": 0,
        "rows_with_empty_customer_region": 0,
        "rows_with_empty_contract_item_name": 0,
        "rows_with_empty_amount": 0,
        "unique_customers": 0,
        "unique_regions": 0,
        "avg_contract_amount": 0.0,
        "top_customer_regions": [],
        "top_contract_years": [],
        "top_categories": [],
        "duplicate_contract_keys": 0,
        "sample_rows": [],
    }

    customer_ste_buffer: Dict[Tuple[str, str], AggregateStats] = {}
    customer_category_buffer: Dict[Tuple[str, int], AggregateStats] = {}
    supplier_ste_buffer: Dict[Tuple[str, str], AggregateStats] = {}
    supplier_category_buffer: Dict[Tuple[str, int], AggregateStats] = {}
    region_category_buffer: Dict[Tuple[str, int], AggregateStats] = {}
    contract_key_buffer: Dict[Tuple[str, str, str], int] = defaultdict(int)
    supplier_region_counter: dict[str, Counter[str]] = defaultdict(Counter)
    customer_name_lookup: Dict[str, str] = {}
    supplier_name_lookup: Dict[str, str] = {}

    unique_customers = set()
    unique_regions = set()
    region_counter: Counter[str] = Counter()
    year_counter: Counter[str] = Counter()
    category_counter: Counter[int] = Counter()
    total_amount = 0.0

    with contracts_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, delimiter=";", quotechar='"')
        for row_index, row in enumerate(reader, start=1):
            stats["rows_total"] += 1
            if len(row) != len(CONTRACT_COLUMNS):
                stats["rows_invalid"] += 1
                continue

            stats["rows_valid"] += 1
            (
                contract_item_name,
                contract_id,
                ste_id,
                contract_datetime,
                contract_amount,
                customer_inn,
                customer_name,
                customer_region,
                supplier_inn,
                supplier_name,
                supplier_region,
            ) = [clean_text(value) for value in row]

            if not contract_item_name:
                stats["rows_with_empty_contract_item_name"] += 1
            if not customer_inn:
                stats["rows_with_empty_customer_inn"] += 1
            if not customer_region:
                stats["rows_with_empty_customer_region"] += 1
            if not contract_amount:
                stats["rows_with_empty_amount"] += 1

            category_id = ste_id_to_category_id.get(ste_id, unknown_category_id)
            if category_id == unknown_category_id:
                stats["rows_with_missing_ste_reference"] += 1

            amount = parse_float(contract_amount)
            total_amount += amount
            contract_date = contract_datetime[:10] if contract_datetime else ""
            contract_year = contract_datetime[:4] if contract_datetime else "UNKNOWN"

            customer_inn = customer_inn or "UNKNOWN"
            customer_region = customer_region or "UNKNOWN"
            supplier_inn = supplier_inn or "UNKNOWN"
            supplier_region = supplier_region or "UNKNOWN"

            unique_customers.add(customer_inn)
            unique_regions.add(customer_region)
            region_counter[customer_region] += 1
            supplier_region_counter[supplier_inn][supplier_region] += 1
            if customer_name and len(normalize_for_search(customer_name)) >= len(
                normalize_for_search(customer_name_lookup.get(customer_inn, ""))
            ):
                customer_name_lookup[customer_inn] = customer_name
            if supplier_name and len(normalize_for_search(supplier_name)) >= len(
                normalize_for_search(supplier_name_lookup.get(supplier_inn, ""))
            ):
                supplier_name_lookup[supplier_inn] = supplier_name
            year_counter[contract_year] += 1
            category_counter[category_id] += 1

            customer_ste_key = (customer_inn, ste_id)
            if customer_ste_key not in customer_ste_buffer:
                customer_ste_buffer[customer_ste_key] = AggregateStats(category_id=category_id)
            customer_ste_buffer[customer_ste_key].update(amount, contract_date, category_id)

            customer_category_key = (customer_inn, category_id)
            if customer_category_key not in customer_category_buffer:
                customer_category_buffer[customer_category_key] = AggregateStats(category_id=category_id)
            customer_category_buffer[customer_category_key].update(amount, contract_date, category_id)

            supplier_ste_key = (supplier_inn, ste_id)
            if supplier_ste_key not in supplier_ste_buffer:
                supplier_ste_buffer[supplier_ste_key] = AggregateStats(category_id=category_id)
            supplier_ste_buffer[supplier_ste_key].update(amount, contract_date, category_id)

            supplier_category_key = (supplier_inn, category_id)
            if supplier_category_key not in supplier_category_buffer:
                supplier_category_buffer[supplier_category_key] = AggregateStats(category_id=category_id)
            supplier_category_buffer[supplier_category_key].update(amount, contract_date, category_id)

            region_category_key = (customer_region, category_id)
            if region_category_key not in region_category_buffer:
                region_category_buffer[region_category_key] = AggregateStats(category_id=category_id)
            region_category_buffer[region_category_key].update(amount, contract_date, category_id)

            contract_key_buffer[(contract_id, ste_id, customer_inn)] += 1

            if len(stats["sample_rows"]) < 5:
                stats["sample_rows"].append(
                    {
                        "contract_id": contract_id,
                        "ste_id": ste_id,
                        "customer_inn": customer_inn,
                        "customer_region": customer_region,
                        "category": category_id_to_name.get(category_id, "UNKNOWN"),
                        "contract_amount": round(amount, 2),
                    }
                )

            if len(customer_ste_buffer) >= flush_threshold:
                flush_customer_ste(conn, customer_ste_buffer)
            if len(customer_category_buffer) >= flush_threshold:
                flush_customer_category(conn, customer_category_buffer)
            if len(supplier_ste_buffer) >= flush_threshold:
                flush_supplier_ste(conn, supplier_ste_buffer)
            if len(supplier_category_buffer) >= flush_threshold:
                flush_supplier_category(conn, supplier_category_buffer)
            if len(region_category_buffer) >= flush_threshold:
                flush_region_category(conn, region_category_buffer)
            if len(contract_key_buffer) >= flush_threshold:
                flush_contract_key_stats(conn, contract_key_buffer)

            if row_index % 250_000 == 0:
                print(f"[Contracts] processed {row_index:,} rows", flush=True)

    flush_customer_ste(conn, customer_ste_buffer)
    flush_customer_category(conn, customer_category_buffer)
    flush_supplier_ste(conn, supplier_ste_buffer)
    flush_supplier_category(conn, supplier_category_buffer)
    flush_region_category(conn, region_category_buffer)
    flush_contract_key_stats(conn, contract_key_buffer)

    supplier_region_rows = []
    for supplier_inn, counter in supplier_region_counter.items():
        supplier_region, frequency = counter.most_common(1)[0]
        supplier_region_rows.append((supplier_inn, supplier_region, frequency))
    conn.execute("DELETE FROM supplier_region_lookup")
    conn.executemany(
        "INSERT INTO supplier_region_lookup (supplier_inn, supplier_region, frequency) VALUES (?, ?, ?)",
        supplier_region_rows,
    )
    conn.execute("DELETE FROM customer_name_lookup")
    conn.executemany(
        "INSERT INTO customer_name_lookup (customer_inn, customer_name) VALUES (?, ?)",
        sorted(customer_name_lookup.items()),
    )
    conn.execute("DELETE FROM supplier_name_lookup")
    conn.executemany(
        "INSERT INTO supplier_name_lookup (supplier_inn, supplier_name) VALUES (?, ?)",
        sorted(supplier_name_lookup.items()),
    )
    conn.commit()

    duplicate_contract_keys = conn.execute(
        "SELECT COALESCE(SUM(row_count - 1), 0) FROM contract_key_stats WHERE row_count > 1"
    ).fetchone()[0]

    stats["duplicate_contract_keys"] = int(duplicate_contract_keys or 0)
    stats["unique_customers"] = len(unique_customers)
    stats["unique_regions"] = len(unique_regions)
    if stats["rows_valid"]:
        stats["avg_contract_amount"] = round(total_amount / stats["rows_valid"], 2)
    stats["top_customer_regions"] = [{"region": region, "count": count} for region, count in region_counter.most_common(15)]
    stats["top_contract_years"] = [{"year": year, "count": count} for year, count in year_counter.most_common()]
    stats["top_categories"] = [
        {"category": category_id_to_name.get(category_id, "UNKNOWN"), "count": count}
        for category_id, count in category_counter.most_common(15)
    ]
    return stats


def export_relations(conn: sqlite3.Connection, output_dir: Path) -> None:
    export_query_to_csv(
        conn,
        """
        SELECT
            cs.customer_inn,
            cs.ste_id,
            cl.category,
            cs.purchase_count,
            ROUND(cs.total_amount, 2) AS total_amount,
            cs.first_purchase_dt,
            cs.last_purchase_dt
        FROM customer_ste_stats cs
        LEFT JOIN category_lookup cl ON cl.category_id = cs.category_id
        ORDER BY cs.purchase_count DESC, cs.total_amount DESC
        """,
        output_dir / "customer_ste_stats.csv",
    )
    export_query_to_csv(
        conn,
        """
        SELECT
            cc.customer_inn,
            cl.category,
            cc.purchase_count,
            ROUND(cc.total_amount, 2) AS total_amount,
            cc.first_purchase_dt,
            cc.last_purchase_dt
        FROM customer_category_stats cc
        LEFT JOIN category_lookup cl ON cl.category_id = cc.category_id
        ORDER BY cc.purchase_count DESC, cc.total_amount DESC
        """,
        output_dir / "customer_category_stats.csv",
    )
    export_query_to_csv(
        conn,
        """
        SELECT
            ss.supplier_inn,
            ss.ste_id,
            cl.category,
            ss.purchase_count,
            ROUND(ss.total_amount, 2) AS total_amount,
            ss.first_purchase_dt,
            ss.last_purchase_dt
        FROM supplier_ste_stats ss
        LEFT JOIN category_lookup cl ON cl.category_id = ss.category_id
        ORDER BY ss.purchase_count DESC, ss.total_amount DESC
        """,
        output_dir / "supplier_ste_stats.csv",
    )
    export_query_to_csv(
        conn,
        """
        SELECT
            sc.supplier_inn,
            cl.category,
            sc.purchase_count,
            ROUND(sc.total_amount, 2) AS total_amount,
            sc.first_purchase_dt,
            sc.last_purchase_dt
        FROM supplier_category_stats sc
        LEFT JOIN category_lookup cl ON cl.category_id = sc.category_id
        ORDER BY sc.purchase_count DESC, sc.total_amount DESC
        """,
        output_dir / "supplier_category_stats.csv",
    )
    export_query_to_csv(
        conn,
        """
        SELECT
            rc.customer_region,
            cl.category,
            rc.purchase_count,
            ROUND(rc.total_amount, 2) AS total_amount,
            rc.first_purchase_dt,
            rc.last_purchase_dt
        FROM region_category_stats rc
        LEFT JOIN category_lookup cl ON cl.category_id = rc.category_id
        ORDER BY rc.purchase_count DESC, rc.total_amount DESC
        """,
        output_dir / "region_category_stats.csv",
    )


def build_report(summary: dict, report_path: Path) -> None:
    ste_stats = summary["ste_catalog"]
    contract_stats = summary["contracts"]
    report_lines = [
        "# Отчёт по подготовке данных",
        "",
        "## Входные файлы",
        "",
        f"- Каталог СТЕ: `{summary['input_files']['ste_catalog']}`",
        f"- Контракты: `{summary['input_files']['contracts']}`",
        "",
        "## Проверка формата",
        "",
        f"- `СТЕ`: кодировка `{ste_stats['encoding']}`, разделитель `{ste_stats['delimiter']}`, ожидается {ste_stats['expected_columns']} колонки.",
        f"- `Контракты`: кодировка `{contract_stats['encoding']}`, разделитель `{contract_stats['delimiter']}`, ожидается {contract_stats['expected_columns']} колонок.",
        "",
        "## Каталог СТЕ",
        "",
        f"- Строк всего: {ste_stats['rows_total']:,}",
        f"- Валидных строк: {ste_stats['rows_valid']:,}",
        f"- Некорректных строк: {ste_stats['rows_invalid']:,}",
        f"- Уникальных `ste_id`: {ste_stats['unique_ste_ids']:,}",
        f"- Дубликатов по `ste_id`: {ste_stats['duplicate_ste_id_rows']:,}",
        f"- Пустых названий: {ste_stats['rows_with_empty_name']:,}",
        f"- Пустых категорий: {ste_stats['rows_with_empty_category']:,}",
        f"- Пустых атрибутов: {ste_stats['rows_with_empty_attributes']:,}",
        f"- Средняя длина названия: {ste_stats['avg_name_length']}",
        f"- Среднее число атрибутов: {ste_stats['avg_attribute_count']}",
        "",
        "Топ категорий по каталогу:",
        "",
    ]
    for item in ste_stats["top_categories"]:
        report_lines.append(f"- {item['category']}: {item['count']:,}")

    report_lines.extend(
        [
            "",
            "## Контракты",
            "",
            f"- Строк всего: {contract_stats['rows_total']:,}",
            f"- Валидных строк: {contract_stats['rows_valid']:,}",
            f"- Некорректных строк: {contract_stats['rows_invalid']:,}",
            f"- Пустых названий позиций: {contract_stats['rows_with_empty_contract_item_name']:,}",
            f"- Пустых ИНН заказчика: {contract_stats['rows_with_empty_customer_inn']:,}",
            f"- Пустых регионов заказчика: {contract_stats['rows_with_empty_customer_region']:,}",
            f"- Пустых сумм: {contract_stats['rows_with_empty_amount']:,}",
            f"- Строк без матча по `ste_id` в каталоге: {contract_stats['rows_with_missing_ste_reference']:,}",
            f"- Уникальных заказчиков: {contract_stats['unique_customers']:,}",
            f"- Уникальных регионов: {contract_stats['unique_regions']:,}",
            f"- Средняя сумма контракта: {contract_stats['avg_contract_amount']}",
            f"- Дубликатов по ключу (`contract_id`, `ste_id`, `customer_inn`): {contract_stats['duplicate_contract_keys']:,}",
            "",
            "Топ регионов заказчиков:",
            "",
        ]
    )
    for item in contract_stats["top_customer_regions"]:
        report_lines.append(f"- {item['region']}: {item['count']:,}")

    report_lines.extend(
        [
            "",
            "Распределение по годам:",
            "",
        ]
    )
    for item in contract_stats["top_contract_years"]:
        report_lines.append(f"- {item['year']}: {item['count']:,}")

    report_lines.extend(
        [
            "",
            "Топ категорий в контрактах:",
            "",
        ]
    )
    for item in contract_stats["top_categories"]:
        report_lines.append(f"- {item['category']}: {item['count']:,}")

    report_lines.extend(
        [
            "",
            "## Подготовленные артефакты",
            "",
            "- `data/processed/ste_catalog_clean.csv`",
            "- `data/processed/category_lookup.csv`",
            "- `data/processed/customer_ste_stats.csv`",
            "- `data/processed/customer_category_stats.csv`",
            "- `data/processed/region_category_stats.csv`",
            "- `data/processed/tenderhack_preprocessed.sqlite`",
            "- `reports/data_prep_eda_report.md`",
            "- `reports/data_prep_summary.json`",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tender Hack datasets for search and personalization.")
    parser.add_argument(
        "--ste-path",
        default="СТЕ_20260403.csv",
        help="Path to the raw STE catalog CSV.",
    )
    parser.add_argument(
        "--contracts-path",
        default="Контракты_20260403.csv",
        help="Path to the raw contracts CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed outputs.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory for EDA and quality reports.",
    )
    args = parser.parse_args()

    ste_path = Path(args.ste_path)
    contracts_path = Path(args.contracts_path)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "tenderhack_preprocessed.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        build_sqlite_schema(conn)
        ste_stats, ste_id_to_category_id, category_id_to_name = process_ste_catalog(ste_path, output_dir, conn)
        contract_stats = process_contracts(contracts_path, conn, ste_id_to_category_id, category_id_to_name)
        export_relations(conn, output_dir)

        summary = {
            "input_files": {
                "ste_catalog": str(ste_path),
                "contracts": str(contracts_path),
            },
            "ste_catalog": ste_stats,
            "contracts": contract_stats,
        }

        summary_path = report_dir / "data_prep_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        build_report(summary, report_dir / "data_prep_eda_report.md")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
