#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.tenderhack.dense_retrieval import (
    DEFAULT_DENSE_EMBEDDING_DIM,
    build_dense_index,
    default_dense_index_path,
    rebuild_dense_index,
)
from src.tenderhack.learned_dense_retrieval import (
    DEFAULT_LEARNED_DENSE_DIM,
    DEFAULT_LEARNED_DENSE_SAMPLE_SIZE,
    build_learned_dense_artifacts,
    default_learned_dense_index_path,
    default_learned_dense_model_path,
    rebuild_learned_dense_artifacts,
)


PROGRESS_EVERY = 100_000
SEMANTIC_PRUNE_WINDOW = 48


def build_search_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;

        DROP TABLE IF EXISTS ste_catalog;
        DROP TABLE IF EXISTS token_frequency;
        DROP TABLE IF EXISTS search_metadata;
        DROP TABLE IF EXISTS semantic_neighbors;
        DROP TABLE IF EXISTS ste_catalog_fts;

        CREATE TABLE ste_catalog (
            ste_id TEXT PRIMARY KEY,
            clean_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            category TEXT NOT NULL,
            normalized_category TEXT NOT NULL,
            attribute_keys TEXT,
            attribute_count INTEGER NOT NULL,
            key_tokens TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE ste_catalog_fts USING fts5(
            clean_name,
            normalized_name,
            category,
            normalized_category,
            key_tokens,
            content='ste_catalog',
            content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        );

        CREATE TABLE token_frequency (
            token TEXT PRIMARY KEY,
            first_char TEXT NOT NULL,
            token_length INTEGER NOT NULL,
            frequency INTEGER NOT NULL
        );

        CREATE INDEX token_frequency_lookup_idx
        ON token_frequency(first_char, token_length, frequency DESC);

        CREATE TABLE semantic_neighbors (
            token TEXT NOT NULL,
            neighbor TEXT NOT NULL,
            score REAL NOT NULL,
            cooccurrence INTEGER NOT NULL,
            PRIMARY KEY (token, neighbor)
        );

        CREATE INDEX semantic_neighbors_token_idx
        ON semantic_neighbors(token, score DESC);

        CREATE TABLE search_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.commit()


def ensure_semantic_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS semantic_neighbors (
            token TEXT NOT NULL,
            neighbor TEXT NOT NULL,
            score REAL NOT NULL,
            cooccurrence INTEGER NOT NULL,
            PRIMARY KEY (token, neighbor)
        );

        CREATE INDEX IF NOT EXISTS semantic_neighbors_token_idx
        ON semantic_neighbors(token, score DESC);
        """
    )
    conn.commit()


def prune_neighbor_counts(neighbor_counts: dict[str, Counter[str]], keep_limit: int) -> None:
    for token, counter in list(neighbor_counts.items()):
        if len(counter) <= keep_limit * 2:
            continue
        neighbor_counts[token] = Counter(dict(counter.most_common(keep_limit)))


def build_semantic_neighbors(
    conn: sqlite3.Connection,
    token_counter: Counter[str],
    semantic_min_frequency: int = 10,
    semantic_neighbors_per_token: int = 8,
) -> tuple[int, int]:
    ensure_semantic_schema(conn)
    conn.execute("DELETE FROM semantic_neighbors")
    conn.execute("DELETE FROM search_metadata WHERE key IN ('semantic_vocab_size', 'semantic_edge_count')")
    conn.commit()

    semantic_vocab = {
        token
        for token, frequency in token_counter.items()
        if len(token) >= 3 and not token.isdigit() and frequency >= semantic_min_frequency
    }
    if not semantic_vocab:
        conn.executemany(
            "INSERT INTO search_metadata (key, value) VALUES (?, ?)",
            [
                ("semantic_vocab_size", "0"),
                ("semantic_edge_count", "0"),
            ],
        )
        conn.commit()
        return 0, 0

    neighbor_counts: dict[str, Counter[str]] = defaultdict(Counter)
    row_count = 0
    for normalized_name, normalized_category, key_tokens in conn.execute(
        "SELECT normalized_name, normalized_category, key_tokens FROM ste_catalog"
    ):
        row_count += 1
        row_tokens = unique_preserve_order(
            [
                token
                for token in tokenize(f"{normalized_name} {normalized_category} {key_tokens}")
                if token in semantic_vocab
            ]
        )
        if len(row_tokens) < 2:
            continue
        if len(row_tokens) > 14:
            row_tokens = sorted(
                row_tokens,
                key=lambda token: (token_counter[token], -len(token), token),
            )[:14]
        for token in row_tokens:
            counter = neighbor_counts[token]
            for neighbor in row_tokens:
                if token == neighbor:
                    continue
                counter[neighbor] += 1
        if row_count % PROGRESS_EVERY == 0:
            prune_neighbor_counts(neighbor_counts, keep_limit=max(semantic_neighbors_per_token * 4, SEMANTIC_PRUNE_WINDOW))
            print(f"[Semantic] processed {row_count:,} deduped rows", flush=True)

    prune_neighbor_counts(neighbor_counts, keep_limit=max(semantic_neighbors_per_token * 4, SEMANTIC_PRUNE_WINDOW))

    semantic_rows = []
    for token, counter in neighbor_counts.items():
        token_frequency = token_counter[token]
        scored_neighbors = []
        for neighbor, cooccurrence in counter.items():
            neighbor_frequency = token_counter.get(neighbor, 0)
            if neighbor_frequency == 0:
                continue
            if cooccurrence < max(1, min(2, semantic_min_frequency)):
                continue
            association = cooccurrence / math.sqrt(token_frequency * neighbor_frequency)
            score = association + 0.2 * ngram_jaccard(token, neighbor)
            if score <= 0.05:
                continue
            scored_neighbors.append((neighbor, score, cooccurrence))
        scored_neighbors.sort(
            key=lambda item: (item[1], item[2], token_counter[item[0]], item[0]),
            reverse=True,
        )
        for neighbor, score, cooccurrence in scored_neighbors[:semantic_neighbors_per_token]:
            semantic_rows.append((token, neighbor, round(score, 6), cooccurrence))

    conn.executemany(
        "INSERT INTO semantic_neighbors (token, neighbor, score, cooccurrence) VALUES (?, ?, ?, ?)",
        semantic_rows,
    )
    conn.executemany(
        "INSERT INTO search_metadata (key, value) VALUES (?, ?)",
        [
            ("semantic_vocab_size", str(len(semantic_vocab))),
            ("semantic_edge_count", str(len(semantic_rows))),
        ],
    )
    conn.commit()
    return len(semantic_vocab), len(semantic_rows)


def tokenize(value: str) -> list[str]:
    return [token for token in value.split() if token]


def unique_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def char_ngrams(value: str, min_n: int = 3, max_n: int = 5) -> set[str]:
    padded = f"<{value}>"
    ngrams: set[str] = set()
    for size in range(min_n, max_n + 1):
        if len(padded) < size:
            continue
        for index in range(len(padded) - size + 1):
            ngrams.add(padded[index : index + size])
    return ngrams


def ngram_jaccard(left: str, right: str) -> float:
    left_ngrams = char_ngrams(left)
    right_ngrams = char_ngrams(right)
    if not left_ngrams or not right_ngrams:
        return 0.0
    union = len(left_ngrams | right_ngrams)
    if union == 0:
        return 0.0
    return len(left_ngrams & right_ngrams) / union


def build_search_db(
    catalog_path: Path,
    search_db_path: Path,
    semantic_min_frequency: int = 10,
    semantic_neighbors_per_token: int = 8,
    dense_embedding_dim: int = DEFAULT_DENSE_EMBEDDING_DIM,
    build_learned_dense: bool = False,
    learned_dense_dim: int = DEFAULT_LEARNED_DENSE_DIM,
    learned_dense_sample_size: int = DEFAULT_LEARNED_DENSE_SAMPLE_SIZE,
) -> None:
    if search_db_path.exists():
        search_db_path.unlink()
    conn = sqlite3.connect(search_db_path)
    conn.row_factory = sqlite3.Row
    try:
        build_search_schema(conn)
        rows_seen = 0
        with catalog_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            batch = []
            for row in reader:
                rows_seen += 1
                batch.append(
                    (
                        row["ste_id"],
                        row["clean_name"],
                        row["normalized_name"],
                        row["category"],
                        row["normalized_category"],
                        row["attribute_keys"],
                        int(row["attribute_count"] or 0),
                        row["key_tokens"],
                    )
                )
                if len(batch) >= 10_000:
                    conn.executemany(
                        """
                        INSERT INTO ste_catalog (
                            ste_id, clean_name, normalized_name, category, normalized_category,
                            attribute_keys, attribute_count, key_tokens
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ste_id) DO UPDATE SET
                            clean_name = CASE
                                WHEN length(excluded.clean_name) > length(ste_catalog.clean_name) THEN excluded.clean_name
                                ELSE ste_catalog.clean_name
                            END,
                            normalized_name = CASE
                                WHEN length(excluded.normalized_name) > length(ste_catalog.normalized_name) THEN excluded.normalized_name
                                ELSE ste_catalog.normalized_name
                            END,
                            category = CASE
                                WHEN length(excluded.category) > length(ste_catalog.category) THEN excluded.category
                                ELSE ste_catalog.category
                            END,
                            normalized_category = CASE
                                WHEN length(excluded.normalized_category) > length(ste_catalog.normalized_category) THEN excluded.normalized_category
                                ELSE ste_catalog.normalized_category
                            END,
                            attribute_keys = CASE
                                WHEN length(excluded.attribute_keys) > length(COALESCE(ste_catalog.attribute_keys, '')) THEN excluded.attribute_keys
                                ELSE ste_catalog.attribute_keys
                            END,
                            attribute_count = MAX(ste_catalog.attribute_count, excluded.attribute_count),
                            key_tokens = CASE
                                WHEN length(excluded.key_tokens) > length(ste_catalog.key_tokens) THEN excluded.key_tokens
                                ELSE ste_catalog.key_tokens
                            END
                        """,
                        batch,
                    )
                    conn.commit()
                    batch.clear()
                if rows_seen % PROGRESS_EVERY == 0:
                    print(f"[Search DB] processed {rows_seen:,} source rows", flush=True)
            if batch:
                conn.executemany(
                    """
                    INSERT INTO ste_catalog (
                        ste_id, clean_name, normalized_name, category, normalized_category,
                        attribute_keys, attribute_count, key_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ste_id) DO UPDATE SET
                        clean_name = CASE
                            WHEN length(excluded.clean_name) > length(ste_catalog.clean_name) THEN excluded.clean_name
                            ELSE ste_catalog.clean_name
                        END,
                        normalized_name = CASE
                            WHEN length(excluded.normalized_name) > length(ste_catalog.normalized_name) THEN excluded.normalized_name
                            ELSE ste_catalog.normalized_name
                        END,
                        category = CASE
                            WHEN length(excluded.category) > length(ste_catalog.category) THEN excluded.category
                            ELSE ste_catalog.category
                        END,
                        normalized_category = CASE
                            WHEN length(excluded.normalized_category) > length(ste_catalog.normalized_category) THEN excluded.normalized_category
                            ELSE ste_catalog.normalized_category
                        END,
                        attribute_keys = CASE
                            WHEN length(excluded.attribute_keys) > length(COALESCE(ste_catalog.attribute_keys, '')) THEN excluded.attribute_keys
                            ELSE ste_catalog.attribute_keys
                        END,
                        attribute_count = MAX(ste_catalog.attribute_count, excluded.attribute_count),
                        key_tokens = CASE
                            WHEN length(excluded.key_tokens) > length(ste_catalog.key_tokens) THEN excluded.key_tokens
                            ELSE ste_catalog.key_tokens
                        END
                    """,
                    batch,
                )
                conn.commit()

        conn.execute("INSERT INTO ste_catalog_fts(ste_catalog_fts) VALUES ('rebuild')")
        conn.commit()

        token_counter: Counter[str] = Counter()
        row_count = 0
        for normalized_name, normalized_category, key_tokens in conn.execute(
            "SELECT normalized_name, normalized_category, key_tokens FROM ste_catalog"
        ):
            row_count += 1
            token_counter.update(tokenize(normalized_name))
            token_counter.update(tokenize(normalized_category))
            token_counter.update(tokenize(key_tokens))
            if row_count % PROGRESS_EVERY == 0:
                print(f"[Search DB] tokenized {row_count:,} deduped rows", flush=True)

        token_rows = []
        for token, frequency in token_counter.items():
            if len(token) <= 1:
                continue
            token_rows.append((token, token[0], len(token), frequency))
        conn.executemany(
            "INSERT INTO token_frequency (token, first_char, token_length, frequency) VALUES (?, ?, ?, ?)",
            token_rows,
        )
        build_semantic_neighbors(
            conn,
            token_counter=token_counter,
            semantic_min_frequency=semantic_min_frequency,
            semantic_neighbors_per_token=semantic_neighbors_per_token,
        )
        dense_index_path = default_dense_index_path(search_db_path)
        build_dense_index(
            conn,
            index_path=dense_index_path,
            embedding_dim=dense_embedding_dim,
        )
        learned_model_path = default_learned_dense_model_path(search_db_path)
        learned_index_path = default_learned_dense_index_path(search_db_path)
        if build_learned_dense:
            build_learned_dense_artifacts(
                conn,
                model_path=learned_model_path,
                index_path=learned_index_path,
                embedding_dim=learned_dense_dim,
                sample_size=learned_dense_sample_size,
            )
        conn.executemany(
            "INSERT INTO search_metadata (key, value) VALUES (?, ?)",
            [
                ("source_rows", str(rows_seen)),
                ("deduped_rows", str(conn.execute("SELECT COUNT(*) FROM ste_catalog").fetchone()[0])),
                ("token_count", str(conn.execute("SELECT COUNT(*) FROM token_frequency").fetchone()[0])),
                ("dense_index_path", str(dense_index_path)),
                ("dense_embedding_dim", str(dense_embedding_dim)),
                ("learned_dense_model_path", str(learned_model_path)),
                ("learned_dense_index_path", str(learned_index_path)),
                ("learned_dense_enabled", "1" if build_learned_dense else "0"),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def rebuild_semantic_assets(
    search_db_path: Path,
    semantic_min_frequency: int = 10,
    semantic_neighbors_per_token: int = 8,
) -> tuple[int, int]:
    conn = sqlite3.connect(search_db_path)
    try:
        token_counter: Counter[str] = Counter(
            {
                row[0]: int(row[1])
                for row in conn.execute(
                    "SELECT token, frequency FROM token_frequency"
                ).fetchall()
            }
        )
        return build_semantic_neighbors(
            conn,
            token_counter=token_counter,
            semantic_min_frequency=semantic_min_frequency,
            semantic_neighbors_per_token=semantic_neighbors_per_token,
        )
    finally:
        conn.close()


def rebuild_dense_assets(
    search_db_path: Path,
    dense_embedding_dim: int = DEFAULT_DENSE_EMBEDDING_DIM,
) -> tuple[int, int]:
    return rebuild_dense_index(
        search_db_path,
        index_path=default_dense_index_path(search_db_path),
        embedding_dim=dense_embedding_dim,
    )


def rebuild_learned_dense_assets(
    search_db_path: Path,
    learned_dense_dim: int = DEFAULT_LEARNED_DENSE_DIM,
    learned_dense_sample_size: int = DEFAULT_LEARNED_DENSE_SAMPLE_SIZE,
) -> tuple[int, int]:
    return rebuild_learned_dense_artifacts(
        search_db_path,
        model_path=default_learned_dense_model_path(search_db_path),
        index_path=default_learned_dense_index_path(search_db_path),
        embedding_dim=learned_dense_dim,
        sample_size=learned_dense_sample_size,
    )


def build_customer_region_lookup(contracts_path: Path, preprocessed_db_path: Path, output_csv_path: Path) -> None:
    region_counter: dict[str, Counter[str]] = defaultdict(Counter)
    rows_seen = 0
    with contracts_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";", quotechar='"')
        for row in reader:
            rows_seen += 1
            if len(row) != 11:
                continue
            customer_inn = row[5].strip()
            customer_region = row[7].strip()
            if customer_inn and customer_region:
                region_counter[customer_inn][customer_region] += 1
            if rows_seen % 250_000 == 0:
                print(f"[Customer Region] processed {rows_seen:,} contract rows", flush=True)

    records = []
    for customer_inn, counter in region_counter.items():
        customer_region, frequency = counter.most_common(1)[0]
        records.append((customer_inn, customer_region, frequency))
    records.sort(key=lambda item: (item[1], item[0]))

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["customer_inn", "customer_region", "frequency"])
        writer.writerows(records)

    conn = sqlite3.connect(preprocessed_db_path)
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS customer_region_lookup;
            CREATE TABLE customer_region_lookup (
                customer_inn TEXT PRIMARY KEY,
                customer_region TEXT NOT NULL,
                frequency INTEGER NOT NULL
            );
            CREATE INDEX customer_region_lookup_region_idx
            ON customer_region_lookup(customer_region);
            """
        )
        conn.executemany(
            "INSERT INTO customer_region_lookup (customer_inn, customer_region, frequency) VALUES (?, ?, ?)",
            records,
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build search and personalization assets.")
    parser.add_argument("--catalog-path", default="data/processed/ste_catalog_search_ready.csv")
    parser.add_argument("--contracts-path", default="Контракты_20260403.csv")
    parser.add_argument("--search-db-path", default="data/processed/tenderhack_search.sqlite")
    parser.add_argument("--preprocessed-db-path", default="data/processed/tenderhack_preprocessed.sqlite")
    parser.add_argument("--customer-region-output", default="data/processed/customer_region_lookup.csv")
    parser.add_argument("--semantic-min-frequency", type=int, default=10)
    parser.add_argument("--semantic-neighbors-per-token", type=int, default=8)
    parser.add_argument("--dense-embedding-dim", type=int, default=DEFAULT_DENSE_EMBEDDING_DIM)
    parser.add_argument("--learned-dense-dim", type=int, default=DEFAULT_LEARNED_DENSE_DIM)
    parser.add_argument("--learned-dense-sample-size", type=int, default=DEFAULT_LEARNED_DENSE_SAMPLE_SIZE)
    parser.add_argument("--semantic-only", action="store_true", help="Rebuild only semantic neighbors on an existing search DB.")
    parser.add_argument("--dense-only", action="store_true", help="Rebuild only dense retrieval vectors on an existing search DB.")
    parser.add_argument("--learned-dense-only", action="store_true", help="Rebuild only learned dense retrieval artifacts on an existing search DB.")
    parser.add_argument("--build-learned-dense", action="store_true", help="Build learned dense retrieval artifacts during a full search DB rebuild.")
    args = parser.parse_args()

    if args.semantic_only:
        rebuild_semantic_assets(
            Path(args.search_db_path),
            semantic_min_frequency=args.semantic_min_frequency,
            semantic_neighbors_per_token=args.semantic_neighbors_per_token,
        )
        return
    if args.dense_only:
        rebuild_dense_assets(
            Path(args.search_db_path),
            dense_embedding_dim=args.dense_embedding_dim,
        )
        return
    if args.learned_dense_only:
        rebuild_learned_dense_assets(
            Path(args.search_db_path),
            learned_dense_dim=args.learned_dense_dim,
            learned_dense_sample_size=args.learned_dense_sample_size,
        )
        return

    build_search_db(
        Path(args.catalog_path),
        Path(args.search_db_path),
        semantic_min_frequency=args.semantic_min_frequency,
        semantic_neighbors_per_token=args.semantic_neighbors_per_token,
        dense_embedding_dim=args.dense_embedding_dim,
        build_learned_dense=bool(args.build_learned_dense),
        learned_dense_dim=args.learned_dense_dim,
        learned_dense_sample_size=args.learned_dense_sample_size,
    )
    build_customer_region_lookup(
        contracts_path=Path(args.contracts_path),
        preprocessed_db_path=Path(args.preprocessed_db_path),
        output_csv_path=Path(args.customer_region_output),
    )


if __name__ == "__main__":
    main()
