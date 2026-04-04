from __future__ import annotations

import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from backend.main import AppSettings, SearchRequest, create_app
from scripts.build_search_assets import build_search_db
from tenderhack.personalization_runtime import PersonalizationRuntimeService
from tenderhack.search import SearchService
from tenderhack.text import unique_preserve_order


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(cls.temp_dir.name)
        cls.catalog_path = base_path / "catalog.csv"
        cls.raw_catalog_path = base_path / "raw_ste.csv"
        cls.search_db_path = base_path / "search.sqlite"
        cls.preprocessed_db_path = base_path / "preprocessed.sqlite"
        cls.synonyms_path = base_path / "synonyms.json"

        with cls.catalog_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
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
            writer.writerow(
                [
                    "ste-1",
                    "Ручка канцелярская синяя",
                    "ручка канцелярская синяя",
                    "Ручки канцелярские",
                    "ручки канцелярские",
                    "Цвет | Тип",
                    "2",
                    "ручка канцелярская синяя шариковая",
                ]
            )
            writer.writerow(
                [
                    "ste-2",
                    "Флеш накопитель 16 ГБ USB 3.0",
                    "флеш накопитель 16 гб usb 3 0",
                    "Usb-накопители твердотельные (флеш-драйвы)",
                    "usb накопители твердотельные флеш драйвы",
                    "Объем | Интерфейс",
                    "2",
                    "флеш накопитель 16 гб usb накопитель",
                ]
            )
            writer.writerow(
                [
                    "ste-3",
                    "Парацетамол таблетки 500 мг №10",
                    "парацетамол таблетки 500 мг 10",
                    "Анальгетики и антипиретики (n02bg)",
                    "анальгетики и антипиретики n02bg",
                    "Дозировка | Форма",
                    "2",
                    "парацетамол таблетки 500 мг анальгетики",
                ]
            )
            writer.writerow(
                [
                    "ste-4",
                    "Ручка офисная красная",
                    "ручка офисная красная",
                    "Ручки канцелярские",
                    "ручки канцелярские",
                    "Цвет | Тип",
                    "2",
                    "ручка офисная красная шариковая",
                ]
            )

        with cls.raw_catalog_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(
                [
                    "ste-1",
                    "Ручка канцелярская синяя",
                    "Ручки канцелярские",
                    "Цвет:синий;Тип:шариковая;Материал:пластик",
                ]
            )
            writer.writerow(
                [
                    "ste-2",
                    "Флеш накопитель 16 ГБ USB 3.0",
                    "Usb-накопители твердотельные (флеш-драйвы)",
                    "Объем накопителя:16.00000;Интерфейс подключения:USB 3.0;Цвет:черный",
                ]
            )
            writer.writerow(
                [
                    "ste-3",
                    "Парацетамол таблетки 500 мг №10",
                    "Анальгетики и антипиретики (n02bg)",
                    "Дозировка:500 мг;Лекарственная форма:таблетки;Количество в упаковке:10",
                ]
            )
            writer.writerow(
                [
                    "ste-4",
                    "Ручка офисная красная",
                    "Ручки канцелярские",
                    "Цвет:красный;Тип:шариковая;Материал:пластик",
                ]
            )

        cls.synonyms_path.write_text(
            json.dumps(
                {
                    "phrase_synonyms": {
                        "флешка": ["флеш накопитель", "usb накопитель"],
                    },
                    "token_synonyms": {
                        "флешка": ["накопитель", "usb"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        build_search_db(
            cls.catalog_path,
            cls.search_db_path,
            semantic_min_frequency=1,
            semantic_neighbors_per_token=4,
        )

        conn = sqlite3.connect(cls.preprocessed_db_path)
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

                CREATE TABLE ste_offer_lookup (
                    ste_id TEXT PRIMARY KEY,
                    supplier_inn TEXT NOT NULL,
                    supplier_region TEXT,
                    offer_count INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    min_price REAL NOT NULL,
                    last_contract_dt TEXT
                );
                """
            )

            conn.executemany(
                "INSERT INTO category_lookup (category_id, category, normalized_category) VALUES (?, ?, ?)",
                [
                    (1, "Ручки канцелярские", "ручки канцелярские"),
                    (2, "Usb-накопители твердотельные (флеш-драйвы)", "usb накопители твердотельные флеш драйвы"),
                    (3, "Анальгетики и антипиретики (n02bg)", "анальгетики и антипиретики n02bg"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO customer_category_stats (
                    customer_inn, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ("7701234567", 1, 5, 1500.0, "2024-01-01", "2025-01-10"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO customer_ste_stats (
                    customer_inn, ste_id, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("7701234567", "ste-1", 1, 4, 900.0, "2024-01-01", "2025-01-10"),
                    ("7701234567", "ste-3", 3, 1, 110.0, "2024-06-10", "2024-06-10"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO region_category_stats (
                    customer_region, category_id, purchase_count, total_amount, first_purchase_dt, last_purchase_dt
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ("Москва", 1, 11, 3300.0, "2024-01-01", "2025-01-10"),
                    ("Москва", 3, 8, 980.0, "2024-01-01", "2025-01-10"),
                ],
            )
            conn.execute(
                "INSERT INTO customer_region_lookup (customer_inn, customer_region, frequency) VALUES (?, ?, ?)",
                ("7701234567", "Москва", 6),
            )
            conn.executemany(
                """
                INSERT INTO ste_offer_lookup (
                    ste_id, supplier_inn, supplier_region, offer_count, avg_price, min_price, last_contract_dt
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("ste-1", "1234567890", "Москва", 4, 225.0, 199.99, "2025-01-10"),
                    ("ste-2", "5555555555", "Москва", 2, 599.0, 549.0, "2025-01-11"),
                    ("ste-3", "7777777777", "Москва", 3, 120.0, 99.0, "2025-01-12"),
                    ("ste-4", "8888888888", "Москва", 2, 240.0, 219.0, "2025-01-15"),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        settings = AppSettings(
            search_db_path=cls.search_db_path,
            preprocessed_db_path=cls.preprocessed_db_path,
            synonyms_path=cls.synonyms_path,
            semantic_backend="sqlite",
            raw_ste_catalog_path=cls.raw_catalog_path,
            redis_url="memory://",
        )
        cls.client_cm = TestClient(create_app(settings=settings))
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_cm.__exit__(None, None, None)
        cls.temp_dir.cleanup()

    def test_login_returns_region_and_seeded_categories(self) -> None:
        response = self.client.post("/api/auth/login", json={"inn": "7701234567"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["inn"], "7701234567")
        self.assertEqual(payload["region"], "Москва")
        self.assertTrue(payload["viewedCategories"])

        service = self.client.app.state.service
        cache_key = service.cache_service.build_key("login", data={"inn": "7701234567"})
        cached_payload = service.cache_service.get_json(cache_key)
        self.assertEqual(service.cache_service.backend_name, "memory")
        self.assertIsInstance(cached_payload, dict)
        self.assertEqual(cached_payload["inn"], "7701234567")

    def test_health_returns_readiness_report(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "ready")
        self.assertTrue(payload["assets"]["ready"])
        self.assertTrue(payload["assets"]["required"]["search_db"]["exists"])

    def test_runtime_personalization_predictor_returns_reason_codes(self) -> None:
        search_service = SearchService(
            search_db_path=self.search_db_path,
            synonyms_path=self.synonyms_path,
            semantic_backend="sqlite",
        )
        runtime_service = PersonalizationRuntimeService(db_path=self.preprocessed_db_path)
        try:
            search_payload = search_service.search(query="канцелярские ручки", top_k=5)
            reranked = runtime_service.rerank_candidates(
                query=str(search_payload["query"]["corrected_query"] or search_payload["query"]["normalized_query"]),
                candidates=list(search_payload["results"]),
                user_id="user-7701234567",
                customer_inn="7701234567",
                customer_region="Москва",
                session_categories=["Ручки канцелярские"],
            )
            self.assertTrue(reranked)
            self.assertEqual(reranked[0]["ste_id"], "ste-1")
            self.assertGreater(reranked[0]["personalization_score"], 0.0)
            self.assertIn("USER_CATEGORY_AFFINITY", reranked[0]["top_reason_codes"])
        finally:
            search_service.close()
            runtime_service.close()

    def test_search_returns_personalized_product_shape(self) -> None:
        request_payload = {
            "query": "канцелярские ручки",
            "userContext": {
                "id": "user-7701234567",
                "inn": "7701234567",
                "region": "Москва",
                "viewedCategories": ["Ручки канцелярские"],
            },
            "viewedCategories": ["Ручки канцелярские"],
            "bouncedCategories": [],
            "topK": 5,
        }
        response = self.client.post(
            "/api/search",
            json=request_payload,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(payload["totalCount"], 1)
        self.assertEqual(payload["items"][0]["id"], "ste-1")
        self.assertEqual(payload["items"][0]["supplierInn"], "1234567890")
        self.assertAlmostEqual(payload["items"][0]["price"], 199.99)
        self.assertIn("Цвет", payload["items"][0]["descriptionPreview"])
        self.assertEqual(payload["items"][0]["reasonToShow"], "На основе ваших закупок")

        service = self.client.app.state.service
        server_session = service.online_state_service.get_session_state(
            user_id="user-7701234567",
            customer_inn="7701234567",
            customer_region="Москва",
        )
        merged_session = {
            "recent_categories": unique_preserve_order(
                list(server_session.get("recent_categories", []))
                + list(request_payload["viewedCategories"])
                + list(request_payload["userContext"]["viewedCategories"])
            ),
            "clicked_ste_ids": list(server_session.get("clicked_ste_ids", [])),
            "cart_ste_ids": list(server_session.get("cart_ste_ids", [])),
            "bounced_categories": list(server_session.get("bounced_categories", [])),
            "version": int(server_session.get("version", 0) or 0),
        }
        cache_key = service.cache_service.build_key(
            "search",
            data=service._search_cache_data(
                SearchRequest(**request_payload),
                server_session=merged_session,
            ),
        )
        cached_payload = service.cache_service.get_json(cache_key)
        self.assertIsInstance(cached_payload, dict)
        self.assertEqual(cached_payload["items"][0]["id"], "ste-1")

    def test_event_updates_session_and_dynamic_search_uses_it(self) -> None:
        dynamic_user_id = "user-7701234567-dyn"
        search_payload = {
            "query": "ручка канцелярская",
            "userContext": {
                "id": dynamic_user_id,
                "inn": "7701234567",
                "region": "Москва",
                "viewedCategories": [],
            },
            "viewedCategories": [],
            "bouncedCategories": [],
            "topK": 5,
        }
        baseline = self.client.post("/api/search", json=search_payload)
        self.assertEqual(baseline.status_code, 200)
        baseline_items = baseline.json()["items"]
        self.assertGreaterEqual(len(baseline_items), 2)
        self.assertEqual(baseline_items[0]["id"], "ste-1")

        event_response = self.client.post(
            "/api/event",
            json={
                "userId": dynamic_user_id,
                "inn": "7701234567",
                "region": "Москва",
                "eventType": "cart_add",
                "steId": "ste-4",
                "category": "Ручки канцелярские",
            },
        )
        self.assertEqual(event_response.status_code, 200)
        event_payload = event_response.json()
        self.assertGreaterEqual(event_payload["sessionVersion"], 1)
        self.assertIn("ste-4", event_payload["cartSteIds"])

        reranked = self.client.post("/api/search", json=search_payload)
        self.assertEqual(reranked.status_code, 200)
        reranked_items = reranked.json()["items"]
        self.assertGreaterEqual(len(reranked_items), 2)
        self.assertEqual(reranked_items[0]["id"], "ste-4")
        self.assertEqual(reranked_items[0]["reasonToShow"], "Продолжить подбор в этой категории")

    def test_search_returns_corrected_query(self) -> None:
        response = self.client.post(
            "/api/search",
            json={
                "query": "парацетомол 500 мг",
                "userContext": None,
                "viewedCategories": [],
                "bouncedCategories": [],
                "topK": 5,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["correctedQuery"], "парацетамол 500 мг")

    def test_suggestions_return_correction_and_abstract_phrases(self) -> None:
        response = self.client.get("/api/search/suggestions", params={"q": "флешка"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload)
        self.assertIn("флеш накопитель", payload)
        self.assertNotIn("Флеш накопитель 16 ГБ USB 3.0", payload)

    def test_session_event_persists_runtime_state(self) -> None:
        response = self.client.post(
            "/api/event",
            json={
                "userId": "user-session-1",
                "eventType": "click",
                "steId": "ste-1",
                "category": "Ручки канцелярские",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("ste-1", payload["clickedSteIds"])
        self.assertIn("ручки канцелярские", payload["recentCategories"])

    def test_search_for_anonymous_user_changes_after_session_event(self) -> None:
        before = self.client.post(
            "/api/search",
            json={
                "query": "канцелярские ручки",
                "userContext": {"id": "anon-1", "inn": None, "region": None, "viewedCategories": []},
                "viewedCategories": [],
                "bouncedCategories": [],
                "topK": 5,
            },
        )
        self.assertEqual(before.status_code, 200)
        before_payload = before.json()
        self.assertIsNone(before_payload["items"][0]["reasonToShow"])

        event_response = self.client.post(
            "/api/event",
            json={
                "userId": "anon-1",
                "eventType": "click",
                "steId": "ste-1",
                "category": "Ручки канцелярские",
            },
        )
        self.assertEqual(event_response.status_code, 200)

        after = self.client.post(
            "/api/search",
            json={
                "query": "канцелярские ручки",
                "userContext": {"id": "anon-1", "inn": None, "region": None, "viewedCategories": []},
                "viewedCategories": [],
                "bouncedCategories": [],
                "topK": 5,
            },
        )
        self.assertEqual(after.status_code, 200)
        after_payload = after.json()
        self.assertEqual(after_payload["items"][0]["reasonToShow"], "На основе ваших закупок")

    def test_item_endpoint_returns_catalog_details(self) -> None:
        response = self.client.get("/api/items/ste-1")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["id"], "ste-1")
        self.assertEqual(payload["supplierInn"], "1234567890")
        self.assertGreaterEqual(payload["offerCount"], 1)
        self.assertIn("Цвет", payload["attributeKeys"])

    def test_offers_endpoint_returns_ranked_offers(self) -> None:
        response = self.client.get(
            "/api/items/ste-1/offers",
            params={
                "user_id": "user-7701234567",
                "customer_inn": "7701234567",
                "customer_region": "Москва",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["itemId"], "ste-1")
        self.assertTrue(payload["offers"])
        self.assertEqual(payload["offers"][0]["steId"], "ste-1")
        self.assertGreater(payload["offers"][0]["offerScore"], 0.0)

    def test_cart_and_procurement_flow(self) -> None:
        add_response = self.client.post(
            "/api/cart/add",
            json={
                "userId": "user-cart-1",
                "steId": "ste-2",
                "quantity": 2,
            },
        )
        self.assertEqual(add_response.status_code, 200)
        add_payload = add_response.json()
        self.assertEqual(add_payload["userId"], "user-cart-1")
        self.assertEqual(add_payload["totalItems"], 2)
        self.assertEqual(add_payload["items"][0]["steId"], "ste-2")

        cart_response = self.client.get("/api/cart", params={"user_id": "user-cart-1"})
        self.assertEqual(cart_response.status_code, 200)
        cart_payload = cart_response.json()
        self.assertEqual(cart_payload["totalItems"], 2)
        self.assertGreater(cart_payload["totalAmount"], 0.0)

        procurement_response = self.client.post(
            "/api/cart/create-procurement",
            json={"userId": "user-cart-1", "procurementType": "direct_purchase"},
        )
        self.assertEqual(procurement_response.status_code, 200)
        procurement_payload = procurement_response.json()
        self.assertEqual(procurement_payload["procurementType"], "direct_purchase")
        self.assertEqual(procurement_payload["status"], "created")
        self.assertEqual(procurement_payload["itemCount"], 2)

        empty_cart_response = self.client.get("/api/cart", params={"user_id": "user-cart-1"})
        self.assertEqual(empty_cart_response.status_code, 200)
        self.assertEqual(empty_cart_response.json()["totalItems"], 0)


if __name__ == "__main__":
    unittest.main()
