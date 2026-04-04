from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from tenderhack.dataset_layout import (
    contracts_candidates,
    raw_contract_candidates,
    raw_ste_candidates,
    ste_catalog_candidates,
)
from tenderhack.text import normalize_text, tokenize


REQUIRED_STE_COLUMNS = [
    "ste_id",
    "clean_name",
    "normalized_name",
    "category",
    "normalized_category",
    "attribute_keys",
    "attribute_count",
    "key_tokens",
]
RAW_STE_COLUMNS = ["ste_id", "raw_name", "raw_category", "raw_attributes"]

REQUIRED_CONTRACT_COLUMNS = [
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

STE_DATASET_CANDIDATES = [
    Path("datasets/processed/ste_catalog_clean.csv"),
    Path("datasets/processed/ste_catalog_search_ready.csv"),
    Path("data/processed/ste_catalog_clean.csv"),
    Path("data/processed/ste_catalog_search_ready.csv"),
]
CONTRACT_DATASET_CANDIDATES = [
    Path("datasets/processed/contracts_clean.csv"),
    Path("datasets/processed/contracts_flat.csv"),
    Path("datasets/processed/contracts.csv"),
    Path("data/processed/contracts_clean.csv"),
    Path("data/processed/contracts_flat.csv"),
    Path("data/processed/contracts.csv"),
]

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d.%m.%Y",
    "%d.%m.%Y %H:%M:%S",
)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    result = str(value).replace("\ufeff", " ").replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
    return " ".join(result.split())


def _normalize_header(value: object) -> str:
    return _clean_text(value).lower().replace(" ", "_")


def _parse_float(value: object) -> tuple[float, bool]:
    cleaned = _clean_text(value).replace(",", ".")
    if not cleaned:
        return 0.0, False
    try:
        return float(cleaned), True
    except ValueError:
        return 0.0, False


def _parse_int(value: object) -> tuple[int, bool]:
    cleaned = _clean_text(value)
    if not cleaned:
        return 0, False
    try:
        return int(cleaned), True
    except ValueError:
        return 0, False


def parse_date(value: object) -> Optional[date]:
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, date_format).date()
        except ValueError:
            continue
    if len(cleaned) >= 10:
        for date_format in DATE_FORMATS:
            try:
                return datetime.strptime(cleaned[:10], date_format).date()
            except ValueError:
                continue
    return None


def _extract_keywords(*parts: str, limit: int = 24) -> str:
    seen = set()
    result = []
    for part in parts:
        for token in tokenize(part):
            if token in seen:
                continue
            seen.add(token)
            result.append(token)
            if len(result) >= limit:
                return " ".join(result)
    return " ".join(result)


def _parse_attributes(raw_attributes: str) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    values: list[str] = []
    for chunk in _clean_text(raw_attributes).split(";"):
        chunk = _clean_text(chunk)
        if not chunk:
            continue
        if ":" in chunk:
            key, value = chunk.split(":", 1)
        else:
            key, value = chunk, ""
        key = _clean_text(key)
        value = _clean_text(value)
        if key:
            keys.append(key)
        if value:
            values.append(value)
    return keys, values


@dataclass(frozen=True)
class STERecord:
    ste_id: str
    clean_name: str
    normalized_name: str
    category: str
    normalized_category: str
    attribute_keys: str
    attribute_count: int
    key_tokens: str

    @property
    def name_tokens(self) -> list[str]:
        return tokenize(self.normalized_name or self.clean_name)

    @property
    def category_tokens(self) -> list[str]:
        return tokenize(self.normalized_category or self.category)

    @property
    def attribute_tokens(self) -> list[str]:
        return tokenize(self.attribute_keys)


@dataclass(frozen=True)
class ContractRecord:
    contract_item_name: str
    contract_id: str
    ste_id: str
    contract_datetime: str
    contract_date: date
    contract_amount: float
    customer_inn: str
    customer_name: str
    customer_region: str
    supplier_inn: str
    supplier_name: str
    supplier_region: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.contract_id, self.ste_id, self.customer_inn)


@dataclass(frozen=True)
class DatasetPaths:
    ste_catalog_path: Optional[Path]
    contracts_path: Optional[Path]


@dataclass(frozen=True)
class LoadedDatasets:
    catalog_by_id: Dict[str, STERecord]
    contracts: list[ContractRecord]
    validation_summary: dict
    paths: DatasetPaths


def resolve_dataset_paths(project_root: Path | str = ".") -> DatasetPaths:
    root = Path(project_root)
    ste_candidates = ste_catalog_candidates(root)
    contract_candidates = contracts_candidates(root)
    ste_candidates.extend(raw_ste_candidates(root))
    contract_candidates.extend(raw_contract_candidates(root))

    ste_path = next((path for path in ste_candidates if path.exists()), None)
    contracts_path = next((path for path in contract_candidates if path.exists()), None)
    return DatasetPaths(ste_catalog_path=ste_path, contracts_path=contracts_path)


def _infer_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    return ";" if sample.count(";") >= sample.count(",") else ","


def _iter_headered_rows(path: Path, required_columns: list[str]) -> Iterator[dict[str, str]]:
    delimiter = _infer_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter, quotechar='"')
        fieldnames = [_normalize_header(name) for name in (reader.fieldnames or [])]
        row_fieldnames = reader.fieldnames or []
        mapping = {normalized: original for normalized, original in zip(fieldnames, row_fieldnames)}
        if not set(required_columns).issubset(mapping):
            raise ValueError(f"{path} does not contain required header columns: {required_columns}")
        for row in reader:
            yield {column: _clean_text(row.get(mapping[column], "")) for column in required_columns}


def _iter_raw_rows(path: Path, column_names: list[str]) -> Iterator[dict[str, str]]:
    delimiter = _infer_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter, quotechar='"')
        for row in reader:
            if not row:
                continue
            cleaned = [_clean_text(value) for value in row]
            if len(cleaned) != len(column_names):
                continue
            yield dict(zip(column_names, cleaned))


def _is_headered(path: Path, expected_columns: list[str]) -> bool:
    delimiter = _infer_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter, quotechar='"')
        first_row = next(reader, [])
    normalized = {_normalize_header(value) for value in first_row}
    return set(expected_columns).issubset(normalized)


def _new_validation_block(expected_columns: list[str], key_name: str) -> dict:
    return {
        "row_count": 0,
        "required_columns": expected_columns,
        "missing_by_column": {column: 0 for column in expected_columns},
        "invalid_type_by_column": {column: 0 for column in expected_columns},
        "duplicate_keys": 0,
        "unique_keys": 0,
        "key_name": key_name,
        "sample_rows": [],
    }


def load_ste_catalog(path: Path) -> tuple[dict[str, STERecord], dict]:
    summary = _new_validation_block(REQUIRED_STE_COLUMNS, "ste_id")
    catalog_by_id: dict[str, STERecord] = {}
    seen_ste_ids = set()
    row_iterator = _iter_headered_rows(path, REQUIRED_STE_COLUMNS) if _is_headered(path, REQUIRED_STE_COLUMNS) else _iter_raw_rows(path, RAW_STE_COLUMNS)

    for row in row_iterator:
        if "raw_name" in row:
            attribute_keys, attribute_values = _parse_attributes(row["raw_attributes"])
            clean_name = row["raw_name"]
            category = row["raw_category"]
            normalized_name = normalize_text(clean_name)
            normalized_category = normalize_text(category)
            attribute_keys_joined = " | ".join(attribute_keys)
            key_tokens = _extract_keywords(clean_name, category, attribute_keys_joined, " ".join(attribute_values))
            materialized = {
                "ste_id": row["ste_id"],
                "clean_name": clean_name,
                "normalized_name": normalized_name,
                "category": category,
                "normalized_category": normalized_category,
                "attribute_keys": attribute_keys_joined,
                "attribute_count": str(len(attribute_keys)),
                "key_tokens": key_tokens,
            }
        else:
            materialized = row

        summary["row_count"] += 1
        for column in REQUIRED_STE_COLUMNS:
            if not _clean_text(materialized.get(column, "")):
                summary["missing_by_column"][column] += 1

        attribute_count, is_attribute_count_valid = _parse_int(materialized.get("attribute_count", "0"))
        if not is_attribute_count_valid and _clean_text(materialized.get("attribute_count", "")):
            summary["invalid_type_by_column"]["attribute_count"] += 1

        record = STERecord(
            ste_id=_clean_text(materialized["ste_id"]),
            clean_name=_clean_text(materialized["clean_name"]),
            normalized_name=_clean_text(materialized["normalized_name"]) or normalize_text(materialized["clean_name"]),
            category=_clean_text(materialized["category"]),
            normalized_category=_clean_text(materialized["normalized_category"]) or normalize_text(materialized["category"]),
            attribute_keys=_clean_text(materialized.get("attribute_keys", "")),
            attribute_count=attribute_count,
            key_tokens=_clean_text(materialized.get("key_tokens", "")),
        )

        if not record.ste_id:
            summary["invalid_type_by_column"]["ste_id"] += 1
            continue
        if record.ste_id in seen_ste_ids:
            summary["duplicate_keys"] += 1
        else:
            seen_ste_ids.add(record.ste_id)
        catalog_by_id[record.ste_id] = record
        if len(summary["sample_rows"]) < 5:
            summary["sample_rows"].append(
                {
                    "ste_id": record.ste_id,
                    "clean_name": record.clean_name,
                    "category": record.category,
                    "attribute_count": record.attribute_count,
                }
            )

    summary["unique_keys"] = len(catalog_by_id)
    return catalog_by_id, summary


def load_contracts(path: Path) -> tuple[list[ContractRecord], dict]:
    summary = _new_validation_block(REQUIRED_CONTRACT_COLUMNS, "contract_id+ste_id+customer_inn")
    contracts: list[ContractRecord] = []
    seen_keys = set()
    row_iterator = _iter_headered_rows(path, REQUIRED_CONTRACT_COLUMNS) if _is_headered(path, REQUIRED_CONTRACT_COLUMNS) else _iter_raw_rows(path, REQUIRED_CONTRACT_COLUMNS)

    for row in row_iterator:
        summary["row_count"] += 1
        for column in REQUIRED_CONTRACT_COLUMNS:
            if not _clean_text(row.get(column, "")):
                summary["missing_by_column"][column] += 1

        contract_amount, is_amount_valid = _parse_float(row.get("contract_amount", ""))
        if not is_amount_valid and _clean_text(row.get("contract_amount", "")):
            summary["invalid_type_by_column"]["contract_amount"] += 1

        contract_date = parse_date(row.get("contract_datetime", ""))
        if contract_date is None:
            summary["invalid_type_by_column"]["contract_datetime"] += 1
            continue

        record = ContractRecord(
            contract_item_name=_clean_text(row["contract_item_name"]),
            contract_id=_clean_text(row["contract_id"]),
            ste_id=_clean_text(row["ste_id"]),
            contract_datetime=_clean_text(row["contract_datetime"]),
            contract_date=contract_date,
            contract_amount=contract_amount,
            customer_inn=_clean_text(row["customer_inn"]) or "UNKNOWN",
            customer_name=_clean_text(row["customer_name"]),
            customer_region=_clean_text(row["customer_region"]) or "UNKNOWN",
            supplier_inn=_clean_text(row["supplier_inn"]),
            supplier_name=_clean_text(row["supplier_name"]),
            supplier_region=_clean_text(row["supplier_region"]) or "UNKNOWN",
        )

        if not record.ste_id:
            summary["invalid_type_by_column"]["ste_id"] += 1
            continue
        if record.key in seen_keys:
            summary["duplicate_keys"] += 1
        else:
            seen_keys.add(record.key)
        contracts.append(record)
        if len(summary["sample_rows"]) < 5:
            summary["sample_rows"].append(
                {
                    "contract_id": record.contract_id,
                    "ste_id": record.ste_id,
                    "customer_inn": record.customer_inn,
                    "contract_date": record.contract_date.isoformat(),
                    "contract_amount": round(record.contract_amount, 2),
                }
            )

    contracts.sort(key=lambda item: (item.contract_date, item.contract_id, item.customer_inn, item.ste_id))
    summary["unique_keys"] = len({record.key for record in contracts})
    return contracts, summary


def _build_join_summary(catalog_by_id: dict[str, STERecord], contracts: Iterable[ContractRecord]) -> dict:
    contract_ste_ids = {record.ste_id for record in contracts}
    catalog_ste_ids = set(catalog_by_id)
    contracts_without_match = sorted(contract_ste_ids - catalog_ste_ids)
    catalog_without_history = sorted(catalog_ste_ids - contract_ste_ids)
    return {
        "join_key": "ste_id",
        "contracts_without_catalog_match": len(contracts_without_match),
        "catalog_without_contract_history": len(catalog_without_history),
        "sample_missing_ste_ids": contracts_without_match[:10],
    }


def load_and_validate_datasets(
    project_root: Path | str = ".",
    dataset_paths: Optional[DatasetPaths] = None,
    strict: bool = False,
) -> LoadedDatasets:
    root = Path(project_root)
    paths = dataset_paths or resolve_dataset_paths(root)
    validation_summary = {
        "status": "ready",
        "paths": {
            "ste_catalog_path": str(paths.ste_catalog_path) if paths.ste_catalog_path else None,
            "contracts_path": str(paths.contracts_path) if paths.contracts_path else None,
        },
        "ste_catalog": _new_validation_block(REQUIRED_STE_COLUMNS, "ste_id"),
        "contracts": _new_validation_block(REQUIRED_CONTRACT_COLUMNS, "contract_id+ste_id+customer_inn"),
        "joins": {
            "join_key": "ste_id",
            "contracts_without_catalog_match": None,
            "catalog_without_contract_history": None,
            "sample_missing_ste_ids": [],
        },
    }

    if not paths.ste_catalog_path or not paths.contracts_path:
        validation_summary["status"] = "missing_input"
        missing = []
        if not paths.ste_catalog_path:
            missing.append("ste_catalog")
        if not paths.contracts_path:
            missing.append("contracts")
        validation_summary["missing_inputs"] = missing
        if strict:
            raise FileNotFoundError(f"Missing required datasets: {', '.join(missing)}")
        return LoadedDatasets(catalog_by_id={}, contracts=[], validation_summary=validation_summary, paths=paths)

    catalog_by_id, ste_summary = load_ste_catalog(paths.ste_catalog_path)
    contracts, contract_summary = load_contracts(paths.contracts_path)
    validation_summary["ste_catalog"] = ste_summary
    validation_summary["contracts"] = contract_summary
    validation_summary["joins"] = _build_join_summary(catalog_by_id, contracts)
    return LoadedDatasets(catalog_by_id=catalog_by_id, contracts=contracts, validation_summary=validation_summary, paths=paths)


def write_data_contract_report(validation_summary: dict, report_path: Path | str) -> None:
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ste_summary = validation_summary["ste_catalog"]
    contract_summary = validation_summary["contracts"]
    join_summary = validation_summary["joins"]

    lines = [
        "# Personalization Data Contract",
        "",
        "## Expected Inputs",
        "",
        "- STE catalog: `datasets/processed/ste_catalog_clean.csv` or `datasets/raw/ste_catalog.csv`.",
        "- Contracts: `datasets/processed/contracts_clean.csv` or `datasets/raw/contracts.csv`.",
        "- Primary join key between contracts and STE catalog: `ste_id`.",
        "",
        "## Required STE Columns",
        "",
    ]
    for column in REQUIRED_STE_COLUMNS:
        lines.append(f"- `{column}`")

    lines.extend(
        [
            "",
            "## Required Contract Columns",
            "",
        ]
    )
    for column in REQUIRED_CONTRACT_COLUMNS:
        lines.append(f"- `{column}`")

    lines.extend(
        [
            "",
            "## Validation Status",
            "",
            f"- Status: `{validation_summary.get('status', 'unknown')}`",
            f"- STE path: `{validation_summary['paths'].get('ste_catalog_path')}`",
            f"- Contracts path: `{validation_summary['paths'].get('contracts_path')}`",
            "",
            "## Observed STE Dataset",
            "",
            f"- Rows: {ste_summary['row_count']:,}",
            f"- Unique `ste_id`: {ste_summary['unique_keys']:,}",
            f"- Duplicate `ste_id`: {ste_summary['duplicate_keys']:,}",
        ]
    )
    for column, count in ste_summary["missing_by_column"].items():
        lines.append(f"- Missing `{column}`: {count:,}")
    for column, count in ste_summary["invalid_type_by_column"].items():
        if count:
            lines.append(f"- Invalid type in `{column}`: {count:,}")

    lines.extend(
        [
            "",
            "## Observed Contracts Dataset",
            "",
            f"- Rows: {contract_summary['row_count']:,}",
            f"- Unique compound keys (`contract_id`, `ste_id`, `customer_inn`): {contract_summary['unique_keys']:,}",
            f"- Duplicate compound keys: {contract_summary['duplicate_keys']:,}",
        ]
    )
    for column, count in contract_summary["missing_by_column"].items():
        lines.append(f"- Missing `{column}`: {count:,}")
    for column, count in contract_summary["invalid_type_by_column"].items():
        if count:
            lines.append(f"- Invalid type in `{column}`: {count:,}")

    lines.extend(
        [
            "",
            "## Join Checks",
            "",
            f"- Join key: `{join_summary['join_key']}`",
            f"- Contract rows without STE catalog match: {join_summary['contracts_without_catalog_match']}",
            f"- Catalog STE without contract history: {join_summary['catalog_without_contract_history']}",
            "",
            "## Missing Data Notes",
            "",
            "- Empty `customer_inn` is replaced with `UNKNOWN` during loading.",
            "- Empty `customer_region` and `supplier_region` are replaced with `UNKNOWN` during loading.",
            "- Invalid `contract_datetime` rows are skipped because ranking splits are time-based.",
            "- Invalid or empty `contract_amount` is retained as `0.0` and flagged in the contract summary.",
            "",
            "## Samples",
            "",
            "### STE",
            "",
        ]
    )
    for row in ste_summary["sample_rows"]:
        lines.append(f"- `{row}`")
    lines.extend(["", "### Contracts", ""])
    for row in contract_summary["sample_rows"]:
        lines.append(f"- `{row}`")

    if validation_summary.get("status") == "missing_input":
        lines.extend(
            [
                "",
                "## Pending Inputs",
                "",
                "- Реальный запуск pipeline заблокирован до появления обоих входных файлов.",
                "- Второй участник команды должен положить очищенный контрактный датасет в один из ожидаемых путей либо в корень репозитория по маске `Контракты_*.csv`.",
            ]
        )

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
