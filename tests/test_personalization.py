from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.personalization import PersonalizationService


class PersonalizationServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp_dir.name) / "preprocessed.sqlite"
        conn = sqlite3.connect(cls.db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE category_lookup (
                    category_id INTEGER PRIMARY KEY,
                    category TEXT NOT NULL,
                    normalized_category TEXT NOT NULL
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

                CREATE TABLE customer_category_stats (
                    customer_inn TEXT NOT NULL,
                    category_id INTEGER NOT NULL,
                    purchase_count INTEGER NOT NULL,
                    total_amount REAL NOT NULL,
                    first_purchase_dt TEXT,
                    last_purchase_dt TEXT,
                    PRIMARY KEY (customer_inn, category_id)
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

                CREATE TABLE supplier_category_stats (
                    supplier_inn TEXT NOT NULL,
                    category_id INTEGER NOT NULL,
                    purchase_count INTEGER NOT NULL,
                    total_amount REAL NOT NULL,
                    first_purchase_dt TEXT,
                    last_purchase_dt TEXT,
                    PRIMARY KEY (supplier_inn, category_id)
                );

                CREATE TABLE region_category_stats (
                    customer_region TEXT NOT NULL,
                    category_id INTEGER NOT NULL,
                    purchase_count INTEGER NOT NULL,
                    total_amount REAL NOT NULL,
                    first_purchase_dt TEXT,
                    last_purchase_dt TEXT,
                    PRIMARY KEY (customer_region, category_id)
                );

                CREATE TABLE customer_region_lookup (
                    customer_inn TEXT PRIMARY KEY,
                    customer_region TEXT NOT NULL,
                    frequency INTEGER NOT NULL
                );

                CREATE TABLE supplier_region_lookup (
                    supplier_inn TEXT PRIMARY KEY,
                    supplier_region TEXT NOT NULL,
                    frequency INTEGER NOT NULL
                );

                CREATE TABLE customer_name_lookup (
                    customer_inn TEXT PRIMARY KEY,
                    customer_name TEXT NOT NULL
                );

                CREATE TABLE supplier_name_lookup (
                    supplier_inn TEXT PRIMARY KEY,
                    supplier_name TEXT NOT NULL
                );
                """
            )

            conn.executemany(
                "INSERT INTO category_lookup (category_id, category, normalized_category) VALUES (?, ?, ?)",
                [
                    (1, "ИММУНОДЕПРЕССАНТЫ,L04", "иммунодепрессанты l04"),
                    (2, "Ручки канцелярские", "ручки канцелярские"),
                    (3, "Расходные материалы и комплектующие для лазерных принтеров и МФУ", "расходные материалы и комплектующие для лазерных принтеров и мфу"),
                    (4, "Дезинфицирующие средства медицинские", "дезинфицирующие средства медицинские"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO customer_category_stats (
                    customer_inn, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ("cust-1", 1, 100, 100000.0, "2024-01-01", "2025-01-01"),
                    ("cust-1", 2, 10, 1500.0, "2024-02-01", "2025-02-01"),
                    ("cust-2", 1, 45, 52000.0, "2024-01-01", "2025-01-01"),
                    ("cust-2", 3, 70, 68000.0, "2024-02-01", "2025-02-15"),
                    ("cust-3", 1, 35, 41000.0, "2024-01-01", "2025-01-20"),
                    ("cust-3", 3, 55, 72000.0, "2024-02-15", "2025-02-20"),
                    ("cust-4", 4, 65, 39000.0, "2024-04-01", "2025-03-10"),
                    ("cust-sparse", 1, 3, 1800.0, "2024-05-01", "2025-03-05"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO customer_ste_stats (
                    customer_inn, ste_id, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("cust-1", "ste-immune-1", 1, 20, 25000.0, "2024-03-01", "2025-03-01"),
                    ("cust-1", "ste-pen-1", 2, 4, 400.0, "2024-03-01", "2025-03-01"),
                    ("cust-2", "ste-immune-2", 1, 9, 11000.0, "2024-03-01", "2025-03-01"),
                    ("cust-2", "ste-printer-1", 3, 18, 28000.0, "2024-03-01", "2025-03-10"),
                    ("cust-3", "ste-immune-3", 1, 8, 9800.0, "2024-03-01", "2025-03-01"),
                    ("cust-3", "ste-printer-2", 3, 16, 25000.0, "2024-03-05", "2025-03-12"),
                    ("cust-4", "ste-disinfect-1", 4, 14, 12000.0, "2024-04-10", "2025-03-12"),
                    ("cust-sparse", "ste-immune-sparse", 1, 2, 900.0, "2024-06-01", "2025-03-05"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO supplier_category_stats (
                    supplier_inn, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ("supp-1", 1, 14, 18000.0, "2024-01-01", "2025-03-10"),
                    ("supp-1", 2, 5, 900.0, "2024-02-01", "2025-03-01"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO supplier_ste_stats (
                    supplier_inn, ste_id, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("supp-1", "ste-immune-1", 1, 12, 15600.0, "2024-01-01", "2025-03-10"),
                    ("supp-1", "ste-pen-1", 2, 5, 900.0, "2024-02-01", "2025-03-01"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO region_category_stats (
                    customer_region, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ("Москва", 1, 250, 300000.0, "2024-01-01", "2025-01-01"),
                    ("Москва", 3, 180, 150000.0, "2024-01-01", "2025-01-01"),
                ],
            )
            conn.execute(
                "INSERT INTO customer_region_lookup (customer_inn, customer_region, frequency) VALUES (?, ?, ?)",
                ("cust-1", "Москва", 12),
            )
            conn.executemany(
                "INSERT INTO customer_region_lookup (customer_inn, customer_region, frequency) VALUES (?, ?, ?)",
                [
                    ("cust-2", "Москва", 8),
                    ("cust-3", "Москва", 7),
                    ("cust-4", "Казань", 4),
                    ("cust-sparse", "Санкт-Петербург", 2),
                ],
            )
            conn.execute(
                "INSERT INTO supplier_region_lookup (supplier_inn, supplier_region, frequency) VALUES (?, ?, ?)",
                ("supp-1", "Москва", 9),
            )
            conn.executemany(
                "INSERT INTO customer_name_lookup (customer_inn, customer_name) VALUES (?, ?)",
                [
                    ("cust-1", "ГБУЗ Городская поликлиника № 1"),
                    ("cust-2", "ГБУЗ Клиническая больница № 2"),
                    ("cust-3", "ГБУЗ Диагностический центр № 3"),
                    ("cust-4", "МБОУ Школа № 7"),
                    ("cust-sparse", "ГБУЗ Детская поликлиника № 5"),
                ],
            )
            conn.execute(
                "INSERT INTO supplier_name_lookup (supplier_inn, supplier_name) VALUES (?, ?)",
                ("supp-1", "Поставщик 1"),
            )
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        self.service = PersonalizationService(db_path=self.db_path)

    def tearDown(self) -> None:
        self.service.close()

    def test_build_customer_profile_infers_region_and_preferences(self) -> None:
        profile = self.service.build_customer_profile("cust-1")
        self.assertEqual(profile["customer_region"], "Москва")
        self.assertEqual(profile["institution_archetype"], "healthcare")
        self.assertEqual(profile["top_categories"][0]["category"], "ИММУНОДЕПРЕССАНТЫ,L04")
        self.assertEqual(profile["top_ste"][0]["ste_id"], "ste-immune-1")

    def test_build_customer_profile_generates_peer_and_type_backfill_recommendations(self) -> None:
        profile = self.service.build_customer_profile("cust-1")
        recommended_categories = profile["recommended_categories"]
        recommended_ste = profile["recommended_ste"]

        self.assertEqual(recommended_categories[0]["category"], "ИММУНОДЕПРЕССАНТЫ,L04")
        self.assertTrue(all(float(item.get("region_weight", 0.0) or 0.0) == 0.0 for item in recommended_categories))
        self.assertTrue(all(float(item.get("region_weight", 0.0) or 0.0) == 0.0 for item in recommended_ste))
        self.assertTrue(
            any(
                item["category"] == "Расходные материалы и комплектующие для лазерных принтеров и МФУ"
                and "медицин" in str(item.get("reason", "")).lower()
                for item in recommended_categories
            )
        )
        self.assertEqual(recommended_ste[0]["ste_id"], "ste-immune-1")
        self.assertTrue(
            any(
                item["ste_id"] == "ste-printer-1"
                and "медицин" in str(item.get("reason", "")).lower()
                for item in recommended_ste
            )
        )

    def test_build_customer_profile_adds_archetype_based_recommendations_for_sparse_history(self) -> None:
        profile = self.service.build_customer_profile("cust-sparse")
        recommended_categories = profile["recommended_categories"]
        recommended_ste = profile["recommended_ste"]

        self.assertEqual(profile["institution_archetype"], "healthcare")
        self.assertTrue(profile["archetype_categories"])
        self.assertTrue(profile["archetype_ste"])
        self.assertIn("cust-4", profile["same_type_peer_inns"])
        self.assertTrue(
            any(
                item["category"] == "Дезинфицирующие средства медицинские"
                and "того же типа" in str(item.get("reason", "")).lower()
                for item in recommended_categories
            )
        )
        self.assertTrue(
            any(
                item["ste_id"] == "ste-disinfect-1"
                and "того же типа" in str(item.get("reason", "")).lower()
                for item in recommended_ste
            )
        )

    def test_build_profile_by_inn_returns_supplier_profile(self) -> None:
        profile = self.service.build_profile_by_inn("supp-1")
        self.assertEqual(profile["entity_type"], "supplier")
        self.assertEqual(profile["supplier_inn"], "supp-1")
        self.assertEqual(profile["customer_region"], "Москва")
        self.assertEqual(profile["top_categories"][0]["category"], "ИММУНОДЕПРЕССАНТЫ,L04")
        self.assertEqual(profile["top_ste"][0]["ste_id"], "ste-immune-1")
        self.assertEqual(profile["recommended_ste"][0]["ste_id"], "ste-immune-1")

    def test_supplier_name_context_uses_sqlite_lookup_without_scanning_contracts(self) -> None:
        from unittest.mock import patch

        with patch.object(
            self.service,
            "_ensure_customer_name_index_loaded",
            side_effect=AssertionError("customer contracts scan should not run for supplier login"),
        ):
            payload = self.service.get_entity_name_context("supp-1")

        self.assertEqual(payload["entity_type"], "supplier")
        self.assertEqual(payload["customer_name"], "Поставщик 1")
        self.assertEqual(payload["institution_name_archetype_label"], "Поставщик")

    def test_customer_name_context_uses_sqlite_lookup_without_scanning_contracts(self) -> None:
        from unittest.mock import patch

        with patch.object(
            self.service,
            "_infer_csv_delimiter",
            side_effect=AssertionError("contracts csv scan should not run when customer_name_lookup is available"),
        ):
            payload = self.service.get_customer_name_context("cust-1")

        self.assertEqual(payload["customer_name"], "ГБУЗ Городская поликлиника № 1")
        self.assertEqual(payload["institution_name_archetype"], "healthcare")

    def test_customer_name_archetype_detects_healthcare_for_drug_supply_center(self) -> None:
        archetype, signal_stems, scores = self.service._infer_customer_name_archetype(
            "Государственное бюджетное учреждение здравоохранения города Москвы Центр лекарственного обеспечения Департамента здравоохранения города Москвы"
        )

        self.assertEqual(archetype, "healthcare")
        self.assertTrue(signal_stems)
        self.assertGreater(scores.get("healthcare", 0.0), scores.get("office_admin", 0.0))

    def test_rerank_ste_boosts_matching_history_and_category(self) -> None:
        profile = self.service.build_customer_profile("cust-1")
        results = [
            {
                "ste_id": "ste-other-1",
                "clean_name": "Картридж для лазерного принтера",
                "category": "Расходные материалы и комплектующие для лазерных принтеров и МФУ",
                "search_score": 8.0,
            },
            {
                "ste_id": "ste-immune-1",
                "clean_name": "Препарат иммунодепрессант",
                "category": "ИММУНОДЕПРЕССАНТЫ,L04",
                "search_score": 7.0,
            },
        ]
        reranked = self.service.rerank_ste(
            results,
            profile,
            session_state={"clicked_ste_ids": ["ste-immune-1"], "cart_ste_ids": [], "recent_categories": ["иммунодепрессанты l04"]},
        )
        self.assertEqual(reranked[0]["ste_id"], "ste-immune-1")
        self.assertIn("часто закупалось этой организацией", reranked[0]["explanation"])
        self.assertGreater(reranked[0]["final_score"], reranked[1]["final_score"])

    def test_rerank_offers_prefers_matching_offer_with_history_and_session(self) -> None:
        profile = self.service.build_customer_profile("cust-1")
        offers = [
            {
                "offer_id": "offer-1",
                "ste_id": "ste-immune-1",
                "category": "ИММУНОДЕПРЕССАНТЫ,L04",
                "supplier_region": "Москва",
                "unit_price": 1200.0,
                "offer_score": 5.0,
            },
            {
                "offer_id": "offer-2",
                "ste_id": "ste-other-1",
                "category": "Расходные материалы и комплектующие для лазерных принтеров и МФУ",
                "supplier_region": "Пермский край",
                "unit_price": 900.0,
                "offer_score": 5.0,
            },
        ]
        reranked = self.service.rerank_offers(
            offers,
            profile,
            session_state={"clicked_ste_ids": [], "cart_ste_ids": ["ste-immune-1"], "recent_categories": ["иммунодепрессанты l04"]},
        )
        self.assertEqual(reranked[0]["offer_id"], "offer-1")
        self.assertIn("СТЕ уже часто закупалось этой организацией", reranked[0]["offer_explanation"])
        self.assertEqual(reranked[0]["offer_personalization_features"]["region_match_boost"], 0.0)
        self.assertEqual(reranked[0]["offer_personalization_features"]["region_affinity"], 0.0)
        self.assertGreater(reranked[0]["final_offer_score"], reranked[1]["final_offer_score"])


if __name__ == "__main__":
    unittest.main()
