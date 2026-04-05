from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.user_profile_scorer import (
    InMemoryUserHistoryRepository,
    SQLiteUserHistoryRepository,
    UserProfileScorer,
    apply_personalization,
)


class UserProfileScorerTests(unittest.TestCase):
    def test_single_category_user_gets_high_boost(self) -> None:
        repository = InMemoryUserHistoryRepository(
            user_history={
                "user-a": {
                    "cat-stationery": 1,
                }
            }
        )
        scorer = UserProfileScorer(repository)

        weights = scorer.compute_category_weights("user-a")

        self.assertEqual(set(weights), {"cat-stationery"})
        self.assertAlmostEqual(weights["cat-stationery"], 1.0)

    def test_fully_diffuse_user_gets_zero_boosts(self) -> None:
        repository = InMemoryUserHistoryRepository(
            user_history={
                "user-a": {"cat-stationery": 1},
                "user-b": {f"cat-{index}": 1 for index in range(25)},
            }
        )
        scorer = UserProfileScorer(repository)

        user_a_weights = scorer.compute_category_weights("user-a")
        user_b_weights = scorer.compute_category_weights("user-b")

        self.assertAlmostEqual(user_a_weights["cat-stationery"], 1.0)
        self.assertTrue(all(weight == 0.0 for weight in user_b_weights.values()))

    def test_repeated_categories_keep_signal_and_one_off_tail_drops_to_zero(self) -> None:
        repository = InMemoryUserHistoryRepository(
            user_history={
                "user-mixed": {
                    "cat-core": 18,
                    "cat-secondary": 6,
                    "cat-one-off": 1,
                }
            }
        )
        scorer = UserProfileScorer(repository)

        weights = scorer.compute_category_weights("user-mixed")

        self.assertGreater(weights["cat-core"], weights["cat-secondary"])
        self.assertGreater(weights["cat-secondary"], 0.0)
        self.assertEqual(weights["cat-one-off"], 0.0)

    def test_ste_weights_keep_repeated_items_and_drop_one_off_items(self) -> None:
        repository = InMemoryUserHistoryRepository(
            user_history={"user-ste": {"cat-stationery": 9}},
            ste_history={
                "user-ste": {
                    "ste-repeat": 5,
                    "ste-one-off": 1,
                }
            },
        )
        scorer = UserProfileScorer(repository)

        weights = scorer.compute_ste_weights("user-ste")

        self.assertGreater(weights["ste-repeat"], 0.0)
        self.assertEqual(weights["ste-one-off"], 0.0)

    def test_apply_personalization_multiplies_base_score_by_category_weight(self) -> None:
        search_results = [
            {"id": "ste-1", "category_id": "cat-stationery", "base_score": 10.0},
            {"id": "ste-2", "category_id": "cat-rare", "base_score": 10.0},
            {"id": "ste-3", "category_id": "cat-unknown", "base_score": 10.0},
        ]

        personalized = apply_personalization(
            search_results,
            user_weights={
                "cat-stationery": 1.0,
                "cat-rare": 0.2,
            },
        )

        self.assertEqual([item["id"] for item in personalized], ["ste-1", "ste-2", "ste-3"])
        self.assertAlmostEqual(personalized[0]["final_score"], 20.0)
        self.assertAlmostEqual(personalized[1]["final_score"], 12.0)
        self.assertAlmostEqual(personalized[2]["final_score"], 10.0)

    def test_zero_unique_categories_returns_empty_weights(self) -> None:
        repository = InMemoryUserHistoryRepository(user_history={"user-empty": {}})
        scorer = UserProfileScorer(repository)

        weights = scorer.compute_category_weights("user-empty")

        self.assertEqual(weights, {})

    def test_sqlite_repository_falls_back_to_supplier_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "supplier-history.sqlite"
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                conn.executescript(
                    """
                    CREATE TABLE category_lookup (
                        category_id INTEGER PRIMARY KEY,
                        category TEXT NOT NULL,
                        normalized_category TEXT NOT NULL
                    );

                    CREATE TABLE customer_category_stats (
                        customer_inn TEXT NOT NULL,
                        category_id INTEGER NOT NULL,
                        purchase_count INTEGER NOT NULL,
                        total_amount REAL NOT NULL,
                        first_purchase_dt TEXT,
                        last_purchase_dt TEXT,
                        PRIMARY KEY (customer_inn, category_id)
                    );

                    CREATE TABLE customer_ste_stats (
                        customer_inn TEXT NOT NULL,
                        ste_id TEXT NOT NULL,
                        category_id INTEGER NOT NULL,
                        purchase_count INTEGER NOT NULL,
                        total_amount REAL NOT NULL,
                        first_purchase_dt TEXT,
                        last_purchase_dt TEXT,
                        PRIMARY KEY (customer_inn, ste_id)
                    );

                    CREATE TABLE supplier_category_stats (
                        supplier_inn TEXT NOT NULL,
                        category_id INTEGER NOT NULL,
                        purchase_count INTEGER NOT NULL,
                        total_amount REAL NOT NULL,
                        first_purchase_dt TEXT,
                        last_purchase_dt TEXT,
                        PRIMARY KEY (supplier_inn, category_id)
                    );

                    CREATE TABLE supplier_ste_stats (
                        supplier_inn TEXT NOT NULL,
                        ste_id TEXT NOT NULL,
                        category_id INTEGER NOT NULL,
                        purchase_count INTEGER NOT NULL,
                        total_amount REAL NOT NULL,
                        first_purchase_dt TEXT,
                        last_purchase_dt TEXT,
                        PRIMARY KEY (supplier_inn, ste_id)
                    );
                    """
                )
                conn.executemany(
                    "INSERT INTO category_lookup (category_id, category, normalized_category) VALUES (?, ?, ?)",
                    [
                        (1, "Ручки канцелярские", "ручки канцелярские"),
                        (2, "Бумага офисная", "бумага офисная"),
                    ],
                )
                conn.executemany(
                    """
                    INSERT INTO supplier_category_stats (
                        supplier_inn, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ("supp-1", 1, 5, 1000.0, "2024-01-01", "2025-01-01"),
                        ("supp-1", 2, 1, 200.0, "2024-01-01", "2024-02-01"),
                    ],
                )
                conn.executemany(
                    """
                    INSERT INTO supplier_ste_stats (
                        supplier_inn, ste_id, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ("supp-1", "ste-repeat", 1, 4, 800.0, "2024-01-01", "2025-01-01"),
                        ("supp-1", "ste-one-off", 2, 1, 200.0, "2024-01-01", "2024-02-01"),
                    ],
                )
                conn.commit()

                scorer = UserProfileScorer(SQLiteUserHistoryRepository(conn))
                category_weights = scorer.compute_category_weights("supp-1")
                ste_weights = scorer.compute_ste_weights("supp-1")

                self.assertGreater(category_weights["ручки канцелярские"], 0.0)
                self.assertEqual(category_weights["бумага офисная"], 0.0)
                self.assertGreater(ste_weights["ste-repeat"], 0.0)
                self.assertEqual(ste_weights["ste-one-off"], 0.0)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
