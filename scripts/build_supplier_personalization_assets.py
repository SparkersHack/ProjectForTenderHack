#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from preprocess_data import (
    AggregateStats,
    CONTRACT_COLUMNS,
    build_sqlite_schema,
    clean_text,
    parse_float,
    flush_supplier_category,
    flush_supplier_ste,
)


def _load_ste_to_category_id(*, search_db_path: Path, preprocessed_db_path: Path) -> dict[str, int]:
    preprocessed_conn = sqlite3.connect(preprocessed_db_path)
    search_conn = sqlite3.connect(search_db_path)
    try:
        category_lookup = {
            str(row[1]): int(row[0])
            for row in preprocessed_conn.execute(
                "SELECT category_id, category FROM category_lookup"
            ).fetchall()
        }
        return {
            str(row[0]): int(category_lookup.get(str(row[1]), 0))
            for row in search_conn.execute(
                "SELECT ste_id, category FROM ste_catalog"
            ).fetchall()
            if row[0]
        }
    finally:
        search_conn.close()
        preprocessed_conn.close()


def build_supplier_personalization_assets(
    *,
    contracts_path: Path,
    search_db_path: Path,
    preprocessed_db_path: Path,
    flush_threshold: int = 200_000,
) -> None:
    ste_to_category_id = _load_ste_to_category_id(
        search_db_path=search_db_path,
        preprocessed_db_path=preprocessed_db_path,
    )

    conn = sqlite3.connect(preprocessed_db_path)
    try:
        build_sqlite_schema(conn)
        conn.execute("DELETE FROM supplier_ste_stats")
        conn.execute("DELETE FROM supplier_category_stats")
        conn.execute("DELETE FROM supplier_region_lookup")
        conn.execute("DELETE FROM customer_name_lookup")
        conn.execute("DELETE FROM supplier_name_lookup")
        conn.commit()

        supplier_ste_buffer: dict[tuple[str, str], AggregateStats] = {}
        supplier_category_buffer: dict[tuple[str, int], AggregateStats] = {}
        supplier_region_counter: dict[str, Counter[str]] = defaultdict(Counter)
        customer_name_lookup: dict[str, str] = {}
        supplier_name_lookup: dict[str, str] = {}

        with contracts_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, delimiter=";", quotechar='"')
            for row_index, row in enumerate(reader, start=1):
                if len(row) != len(CONTRACT_COLUMNS):
                    continue

                (
                    _contract_item_name,
                    _contract_id,
                    ste_id,
                    contract_datetime,
                    contract_amount,
                    customer_inn,
                    customer_name,
                    _customer_region,
                    supplier_inn,
                    supplier_name,
                    supplier_region,
                ) = [clean_text(value) for value in row]

                customer_inn = customer_inn or "UNKNOWN"
                supplier_inn = supplier_inn or "UNKNOWN"
                supplier_region = supplier_region or "UNKNOWN"
                if customer_name and len(customer_name) >= len(customer_name_lookup.get(customer_inn, "")):
                    customer_name_lookup[customer_inn] = customer_name
                if supplier_name and len(supplier_name) >= len(supplier_name_lookup.get(supplier_inn, "")):
                    supplier_name_lookup[supplier_inn] = supplier_name
                category_id = int(ste_to_category_id.get(ste_id, 0))
                amount = parse_float(contract_amount)
                contract_date = contract_datetime[:10] if contract_datetime else ""

                supplier_ste_key = (supplier_inn, ste_id)
                if supplier_ste_key not in supplier_ste_buffer:
                    supplier_ste_buffer[supplier_ste_key] = AggregateStats(category_id=category_id)
                supplier_ste_buffer[supplier_ste_key].update(amount, contract_date, category_id)

                supplier_category_key = (supplier_inn, category_id)
                if supplier_category_key not in supplier_category_buffer:
                    supplier_category_buffer[supplier_category_key] = AggregateStats(category_id=category_id)
                supplier_category_buffer[supplier_category_key].update(amount, contract_date, category_id)

                supplier_region_counter[supplier_inn][supplier_region] += 1

                if len(supplier_ste_buffer) >= flush_threshold:
                    flush_supplier_ste(conn, supplier_ste_buffer)
                if len(supplier_category_buffer) >= flush_threshold:
                    flush_supplier_category(conn, supplier_category_buffer)

                if row_index % 250_000 == 0:
                    print(f"[Supplier Assets] processed {row_index:,} contract rows", flush=True)

        flush_supplier_ste(conn, supplier_ste_buffer)
        flush_supplier_category(conn, supplier_category_buffer)

        supplier_region_rows = []
        for supplier_inn, counter in supplier_region_counter.items():
            supplier_region, frequency = counter.most_common(1)[0]
            supplier_region_rows.append((supplier_inn, supplier_region, frequency))
        conn.executemany(
            "INSERT INTO supplier_region_lookup (supplier_inn, supplier_region, frequency) VALUES (?, ?, ?)",
            supplier_region_rows,
        )
        conn.executemany(
            "INSERT INTO customer_name_lookup (customer_inn, customer_name) VALUES (?, ?)",
            sorted(customer_name_lookup.items()),
        )
        conn.executemany(
            "INSERT INTO supplier_name_lookup (supplier_inn, supplier_name) VALUES (?, ?)",
            sorted(supplier_name_lookup.items()),
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supplier personalization assets in SQLite.")
    parser.add_argument("--contracts-path", default="Контракты_20260403.csv")
    parser.add_argument("--search-db-path", default="data/processed/tenderhack_search.sqlite")
    parser.add_argument("--preprocessed-db-path", default="data/processed/tenderhack_preprocessed.sqlite")
    parser.add_argument("--flush-threshold", type=int, default=200_000)
    args = parser.parse_args()

    build_supplier_personalization_assets(
        contracts_path=Path(args.contracts_path),
        search_db_path=Path(args.search_db_path),
        preprocessed_db_path=Path(args.preprocessed_db_path),
        flush_threshold=args.flush_threshold,
    )


if __name__ == "__main__":
    main()
