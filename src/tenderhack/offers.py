from __future__ import annotations

import csv
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .cache import CacheService
from .dataset_layout import resolve_preprocessed_db_path, resolve_raw_contracts_path

DEFAULT_PREPROCESSED_DB = resolve_preprocessed_db_path()
DEFAULT_CONTRACTS_PATH = resolve_raw_contracts_path()


def _chunked(values: List[str], size: int = 800) -> Iterable[List[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


class OfferLookupService:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_PREPROCESSED_DB,
        cache_service: CacheService | None = None,
        lookup_ttl_seconds: int = 1800,
    ) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cache_service = cache_service
        self.lookup_ttl_seconds = int(lookup_ttl_seconds)

    def close(self) -> None:
        self.conn.close()

    def has_offer_lookup(self) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'ste_offer_lookup'
            LIMIT 1
            """
        ).fetchone()
        return row is not None

    def has_offer_candidates(self) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'ste_offer_candidates'
            LIMIT 1
            """
        ).fetchone()
        return row is not None

    def get_offer_lookup(self, ste_ids: Iterable[str]) -> Dict[str, Dict[str, object]]:
        normalized_ids = [str(value) for value in ste_ids if value]
        if not normalized_ids:
            return {}
        result: Dict[str, Dict[str, object]] = {}
        missing_ids: List[str] = []

        if self.cache_service and self.cache_service.enabled:
            for ste_id in normalized_ids:
                cache_key = self.cache_service.build_key("offer", suffix=ste_id)
                cached_value = self.cache_service.get_json(cache_key)
                if isinstance(cached_value, dict):
                    result[ste_id] = cached_value
                else:
                    missing_ids.append(ste_id)
        else:
            missing_ids = normalized_ids

        loaded: Dict[str, Dict[str, object]]
        if not missing_ids:
            return result
        if self.has_offer_lookup():
            loaded = self._load_offer_lookup(missing_ids)
        else:
            loaded = self._load_estimated_lookup(missing_ids)

        if self.cache_service and self.cache_service.enabled:
            for ste_id, payload in loaded.items():
                cache_key = self.cache_service.build_key("offer", suffix=ste_id)
                self.cache_service.set_json(cache_key, payload, ttl_seconds=self.lookup_ttl_seconds)

        result.update(loaded)
        return result

    def _load_offer_lookup(self, ste_ids: List[str]) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        for chunk in _chunked(ste_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    ste_id,
                    supplier_inn,
                    supplier_region,
                    offer_count,
                    avg_price,
                    min_price,
                    last_contract_dt
                FROM ste_offer_lookup
                WHERE ste_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                result[row["ste_id"]] = {
                    "supplier_inn": row["supplier_inn"] or "не указан",
                    "supplier_region": row["supplier_region"] or "",
                    "offer_count": int(row["offer_count"] or 0),
                    "avg_price": round(float(row["avg_price"] or 0.0), 2),
                    "min_price": round(float(row["min_price"] or 0.0), 2),
                    "price": round(float(row["min_price"] or row["avg_price"] or 0.0), 2),
                    "last_contract_dt": row["last_contract_dt"],
                    "price_source": "contracts_lookup",
                }
        return result

    def _load_estimated_lookup(self, ste_ids: List[str]) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        for chunk in _chunked(ste_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    ste_id,
                    SUM(purchase_count) AS purchase_count,
                    SUM(total_amount) AS total_amount
                FROM customer_ste_stats
                WHERE ste_id IN ({placeholders})
                GROUP BY ste_id
                """,
                chunk,
            ).fetchall()
            for row in rows:
                purchase_count = int(row["purchase_count"] or 0)
                total_amount = float(row["total_amount"] or 0.0)
                avg_price = total_amount / purchase_count if purchase_count else 0.0
                result[row["ste_id"]] = {
                    "supplier_inn": "не указан",
                    "supplier_region": "",
                    "offer_count": purchase_count,
                    "avg_price": round(avg_price, 2),
                    "min_price": round(avg_price, 2),
                    "price": round(avg_price, 2),
                    "last_contract_dt": None,
                    "price_source": "estimated_from_history",
                }
        return result

    def get_offer_candidates(
        self,
        ste_id: str,
        *,
        category: str = "",
        limit: int = 20,
    ) -> List[Dict[str, object]]:
        normalized_ste_id = str(ste_id).strip()
        if not normalized_ste_id:
            return []

        if self.has_offer_candidates():
            offers = self._load_offer_candidates(normalized_ste_id, category=category, limit=limit)
            if offers:
                return offers

        lookup = self.get_offer_lookup([normalized_ste_id]).get(normalized_ste_id)
        if not lookup:
            return []
        price = float(lookup.get("min_price") or lookup.get("avg_price") or lookup.get("price") or 0.0)
        if price <= 0:
            return []
        return [
            {
                "offer_id": f"{normalized_ste_id}::primary",
                "ste_id": normalized_ste_id,
                "category": category,
                "supplier_inn": str(lookup.get("supplier_inn") or "не указан"),
                "supplier_region": str(lookup.get("supplier_region") or ""),
                "unit_price": round(price, 2),
                "offer_score": 1.0,
                "contract_count": int(lookup.get("offer_count") or 0),
                "price_source": str(lookup.get("price_source") or "contracts_lookup"),
            }
        ]

    def _load_offer_candidates(
        self,
        ste_id: str,
        *,
        category: str,
        limit: int,
    ) -> List[Dict[str, object]]:
        rows = self.conn.execute(
            """
            SELECT
                offer_id,
                ste_id,
                supplier_inn,
                supplier_region,
                contract_count,
                avg_price,
                min_price,
                last_contract_dt
            FROM ste_offer_candidates
            WHERE ste_id = ?
            ORDER BY min_price ASC, contract_count DESC, supplier_inn ASC
            LIMIT ?
            """,
            (ste_id, max(1, int(limit))),
        ).fetchall()
        if not rows:
            return []

        max_contract_count = max(int(row["contract_count"] or 0) for row in rows) or 1
        max_price = max(float(row["min_price"] or row["avg_price"] or 0.0) for row in rows) or 1.0

        offers: List[Dict[str, object]] = []
        for row in rows:
            unit_price = float(row["min_price"] or row["avg_price"] or 0.0)
            if unit_price <= 0:
                continue
            contract_count = int(row["contract_count"] or 0)
            activity_score = min(1.0, math.log1p(contract_count) / math.log1p(max_contract_count))
            price_score = max(0.0, 1.0 - (unit_price / max_price))
            offer_score = round(4.0 * activity_score + 6.0 * price_score, 4)
            offers.append(
                {
                    "offer_id": str(row["offer_id"]),
                    "ste_id": str(row["ste_id"]),
                    "category": category,
                    "supplier_inn": str(row["supplier_inn"] or "не указан"),
                    "supplier_region": str(row["supplier_region"] or ""),
                    "unit_price": round(unit_price, 2),
                    "offer_score": offer_score,
                    "contract_count": contract_count,
                    "avg_price": round(float(row["avg_price"] or 0.0), 2),
                    "min_price": round(unit_price, 2),
                    "last_contract_dt": row["last_contract_dt"],
                    "price_source": "contracts_history",
                }
            )
        return offers


def build_offer_tables(
    contracts_path: Path | str = DEFAULT_CONTRACTS_PATH,
    db_path: Path | str = DEFAULT_PREPROCESSED_DB,
) -> Dict[str, int]:
    contracts_path = Path(contracts_path)
    db_path = Path(db_path)

    ste_aggregates: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "offer_count": 0,
            "total_amount": 0.0,
            "min_price": None,
            "supplier_inn": "не указан",
            "supplier_region": "",
            "last_contract_dt": None,
        }
    )
    supplier_aggregates: dict[Tuple[str, str, str], dict[str, object]] = defaultdict(
        lambda: {
            "contract_count": 0,
            "total_amount": 0.0,
            "min_price": None,
            "last_contract_dt": None,
        }
    )

    with contracts_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";", quotechar='"')
        for row in reader:
            if len(row) != 11:
                continue
            ste_id = row[2].strip()
            if not ste_id:
                continue
            contract_dt = row[3].strip()[:10] if row[3].strip() else None
            try:
                amount = float(row[4].strip() or 0.0)
            except ValueError:
                amount = 0.0
            supplier_inn = row[8].strip() or "не указан"
            supplier_region = row[10].strip()

            payload = ste_aggregates[ste_id]
            payload["offer_count"] = int(payload["offer_count"]) + 1
            payload["total_amount"] = float(payload["total_amount"]) + amount

            min_price = payload["min_price"]
            if min_price is None or amount < float(min_price):
                payload["min_price"] = amount
                payload["supplier_inn"] = supplier_inn
                payload["supplier_region"] = supplier_region

            last_contract_dt = payload["last_contract_dt"]
            if contract_dt and (last_contract_dt is None or contract_dt > str(last_contract_dt)):
                payload["last_contract_dt"] = contract_dt

            supplier_payload = supplier_aggregates[(ste_id, supplier_inn, supplier_region)]
            supplier_payload["contract_count"] = int(supplier_payload["contract_count"]) + 1
            supplier_payload["total_amount"] = float(supplier_payload["total_amount"]) + amount
            supplier_min_price = supplier_payload["min_price"]
            if supplier_min_price is None or amount < float(supplier_min_price):
                supplier_payload["min_price"] = amount
            supplier_last_dt = supplier_payload["last_contract_dt"]
            if contract_dt and (supplier_last_dt is None or contract_dt > str(supplier_last_dt)):
                supplier_payload["last_contract_dt"] = contract_dt

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS ste_offer_lookup;
            DROP TABLE IF EXISTS ste_offer_candidates;
            CREATE TABLE ste_offer_lookup (
                ste_id TEXT PRIMARY KEY,
                supplier_inn TEXT NOT NULL,
                supplier_region TEXT,
                offer_count INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                min_price REAL NOT NULL,
                last_contract_dt TEXT
            );
            CREATE INDEX ste_offer_lookup_supplier_idx
            ON ste_offer_lookup(supplier_inn);

            CREATE TABLE ste_offer_candidates (
                offer_id TEXT PRIMARY KEY,
                ste_id TEXT NOT NULL,
                supplier_inn TEXT NOT NULL,
                supplier_region TEXT,
                contract_count INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                min_price REAL NOT NULL,
                last_contract_dt TEXT
            );
            CREATE INDEX ste_offer_candidates_ste_idx
            ON ste_offer_candidates(ste_id, min_price ASC, contract_count DESC);
            """
        )

        lookup_rows = []
        for ste_id, payload in ste_aggregates.items():
            offer_count = int(payload["offer_count"])
            total_amount = float(payload["total_amount"])
            avg_price = total_amount / offer_count if offer_count else 0.0
            lookup_rows.append(
                (
                    ste_id,
                    str(payload["supplier_inn"] or "не указан"),
                    str(payload["supplier_region"] or ""),
                    offer_count,
                    round(avg_price, 2),
                    round(float(payload["min_price"] or avg_price or 0.0), 2),
                    payload["last_contract_dt"],
                )
            )
        conn.executemany(
            """
            INSERT INTO ste_offer_lookup (
                ste_id, supplier_inn, supplier_region, offer_count, avg_price, min_price, last_contract_dt
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            lookup_rows,
        )

        candidate_rows = []
        for (ste_id, supplier_inn, supplier_region), payload in supplier_aggregates.items():
            contract_count = int(payload["contract_count"])
            total_amount = float(payload["total_amount"])
            avg_price = total_amount / contract_count if contract_count else 0.0
            min_price = round(float(payload["min_price"] or avg_price or 0.0), 2)
            if min_price <= 0:
                continue
            offer_id = f"{ste_id}::{supplier_inn}::{supplier_region or '-'}"
            candidate_rows.append(
                (
                    offer_id,
                    ste_id,
                    supplier_inn or "не указан",
                    supplier_region or "",
                    contract_count,
                    round(avg_price, 2),
                    min_price,
                    payload["last_contract_dt"],
                )
            )
        conn.executemany(
            """
            INSERT INTO ste_offer_candidates (
                offer_id, ste_id, supplier_inn, supplier_region, contract_count, avg_price, min_price, last_contract_dt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            candidate_rows,
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "ste_lookup_count": len(ste_aggregates),
        "offer_candidate_count": len(candidate_rows),
    }


def build_offer_lookup_table(
    contracts_path: Path | str = DEFAULT_CONTRACTS_PATH,
    db_path: Path | str = DEFAULT_PREPROCESSED_DB,
) -> int:
    counts = build_offer_tables(contracts_path=contracts_path, db_path=db_path)
    return int(counts["ste_lookup_count"])
