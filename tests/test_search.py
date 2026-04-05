from __future__ import annotations

import csv
import json
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

from scripts.build_search_assets import build_search_db
from tenderhack.dense_retrieval import rebuild_dense_index
from tenderhack.learned_dense_retrieval import rebuild_learned_dense_artifacts
from tenderhack.search import SearchService


class SearchServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(cls.temp_dir.name)
        cls.catalog_path = base_path / "catalog.csv"
        cls.search_db_path = base_path / "search.sqlite"
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
                    "Многофункциональное устройство (МФУ) лазерное",
                    "многофункциональное устройство мфу лазерное",
                    "Печатающее оборудование",
                    "печатающее оборудование",
                    "Тип печати | Форм-фактор",
                    "2",
                    "мфу многофункциональное устройство",
                ]
            )
            writer.writerow(
                [
                    "ste-5",
                    "Сифон бутылочный с горлышком",
                    "сифон бутылочный с горлышком",
                    "Сифоны сантехнические",
                    "сифоны сантехнические",
                    "Тип | Конструкция",
                    "2",
                    "сифон бутылочный горлышко",
                ]
            )
            writer.writerow(
                [
                    "ste-6",
                    "Автоматический наружный дефибриллятор A15",
                    "автоматический наружный дефибриллятор a15",
                    "Медицинское оборудование",
                    "медицинское оборудование",
                    "Тип | Назначение",
                    "2",
                    "дефибриллятор автоматический наружный a15",
                ]
            )
            writer.writerow(
                [
                    "ste-7",
                    "Defi-B наружный автоматический",
                    "defi b наружный автоматический",
                    "Медицинское оборудование",
                    "медицинское оборудование",
                    "Тип | Назначение",
                    "2",
                    "defi b aed автоматический наружный",
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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        self.service = SearchService(
            search_db_path=self.search_db_path,
            synonyms_path=self.synonyms_path,
            semantic_backend="sqlite",
        )

    def tearDown(self) -> None:
        self.service.close()

    def test_exact_search_returns_relevant_ste(self) -> None:
        results = self.service.search_ste("парацетамол 500 мг", top_k=3, min_score=0.0)
        self.assertTrue(results)
        self.assertEqual(results[0]["ste_id"], "ste-3")

    def test_typo_correction_updates_query_and_result(self) -> None:
        payload = self.service.search("парацетомол 500 мг", top_k=3, min_score=0.0)
        self.assertEqual(payload["query"]["corrected_query"], "парацетамол 500 мг")
        self.assertEqual(payload["query"]["applied_corrections"][0]["target"], "парацетамол")
        self.assertEqual(payload["results"][0]["ste_id"], "ste-3")

    def test_synonym_query_finds_flash_drive(self) -> None:
        payload = self.service.search("флешка 16 гб", top_k=3, min_score=0.0)
        self.assertTrue(payload["query"]["applied_synonyms"])
        self.assertEqual(payload["results"][0]["ste_id"], "ste-2")

    def test_wordform_query_matches_stationery_pen_category(self) -> None:
        payload = self.service.search("канцелярские ручки", top_k=3, min_score=0.0)
        self.assertEqual(payload["results"][0]["ste_id"], "ste-1")
        self.assertEqual(payload["results"][0]["category"], "Ручки канцелярские")

    def test_semantic_neighbors_are_applied_for_acronym_query(self) -> None:
        payload = self.service.search("мфу", top_k=3, min_score=0.0)
        self.assertEqual(payload["results"][0]["ste_id"], "ste-4")
        self.assertTrue(payload["query"]["applied_semantic_neighbors"])
        semantic_targets = {
            target
            for item in payload["query"]["applied_semantic_neighbors"]
            if item["source"] == "мфу"
            for target in item["targets"]
        }
        self.assertTrue({"многофункциональное", "устройство"} & semantic_targets)
        self.assertGreater(payload["results"][0]["search_features"]["semantic_name_overlap"], 0.0)

    def test_strong_lexical_anchor_survives_without_semantic_expansion(self) -> None:
        payload = self.service.search("мфу для офиса", top_k=3)
        self.assertTrue(payload["results"])
        self.assertEqual(payload["results"][0]["ste_id"], "ste-4")
        semantic_sources = {item["source"] for item in payload["query"]["applied_semantic_neighbors"]}
        self.assertNotIn("офиса", semantic_sources)

    def test_min_score_filters_low_semantic_matches_but_keeps_exact_lexical_hits(self) -> None:
        self.assertFalse(
            self.service._passes_min_score(
                {
                    "search_features": {
                        "semantic_vector_similarity": 0.18,
                        "semantic_name_overlap": 0.12,
                        "semantic_category_overlap": 0.14,
                        "semantic_key_overlap": 0.10,
                        "exact_phrase": 0.0,
                        "full_name_cover": 0.0,
                        "full_category_cover": 0.0,
                        "corrected_token_overlap": 0.4,
                    }
                },
                min_score=0.55,
            )
        )
        self.assertTrue(
            self.service._passes_min_score(
                {
                    "search_features": {
                        "semantic_vector_similarity": 0.05,
                        "semantic_name_overlap": 0.0,
                        "semantic_category_overlap": 0.0,
                        "semantic_key_overlap": 0.0,
                        "exact_phrase": 1.0,
                        "full_name_cover": 0.0,
                        "full_category_cover": 0.0,
                        "corrected_token_overlap": 0.0,
                    }
                },
                min_score=0.55,
            )
        )

    def test_incomplete_word_query_uses_completions_without_bad_short_correction(self) -> None:
        payload = self.service.search("мног", top_k=3, min_score=0.0)
        self.assertEqual(payload["results"][0]["ste_id"], "ste-4")
        self.assertEqual(payload["query"]["corrected_query"], "мног")
        self.assertFalse(payload["query"]["applied_corrections"])
        completion_targets = {
            target
            for item in payload["query"]["applied_completions"]
            if item["source"] == "мног"
            for target in item["targets"]
        }
        self.assertIn("многофункциональное", completion_targets)

    def test_incomplete_category_prefix_adds_completion_variants(self) -> None:
        payload = self.service.search("канц", top_k=3, min_score=0.0)
        self.assertEqual(payload["results"][0]["ste_id"], "ste-1")
        completion_targets = {
            target
            for item in payload["query"]["applied_completions"]
            if item["source"] == "канц"
            for target in item["targets"]
        }
        self.assertTrue({"канцелярская", "канцелярские"} & completion_targets)

    def test_inflectional_variant_is_not_forced_into_different_wordform(self) -> None:
        payload = self.service.search("бутылочное горлышко", top_k=3, min_score=0.0)
        self.assertEqual(payload["query"]["corrected_query"], "бутылочное горлышко")
        self.assertFalse(payload["query"]["applied_corrections"])
        self.assertEqual(payload["results"][0]["ste_id"], "ste-5")

    def test_search_returns_pagination_metadata(self) -> None:
        payload = self.service.search("ручка", limit=1, offset=0, min_score=0.0)
        self.assertGreaterEqual(payload["total_found"], 1)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["has_more"], payload["total_found"] > 1)

    def test_phrase_synonym_does_not_match_inside_larger_token(self) -> None:
        payload = self.service.search("суперфлешка", top_k=3, min_score=0.0)
        self.assertFalse(payload["query"]["applied_synonyms"])

    def test_typo_query_does_not_keep_original_misspelled_token_in_expansions(self) -> None:
        payload = self.service.search("парацетомол 500 мг", top_k=3, min_score=0.0)
        self.assertIn("парацетамол", payload["query"]["expanded_tokens"])
        self.assertNotIn("парацетомол", payload["query"]["expanded_tokens"])

    def test_sqlite_semantic_backend_exposes_sentence_similarity_signal(self) -> None:
        similarity = self.service.semantic_expander.sentence_similarity(
            "мфу лазерное",
            "многофункциональное устройство лазерное",
        )
        self.assertGreater(similarity, 0.0)

    def test_sqlite_semantic_backend_filters_generic_modifier_noise(self) -> None:
        self.service.conn.executemany(
            """
            INSERT OR REPLACE INTO token_frequency (token, first_char, token_length, frequency)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("автоматический", "а", 13, 4000),
                ("выключатели", "в", 11, 700),
                ("дефибриллятор", "д", 13, 24),
                ("defi", "d", 4, 3),
            ],
        )
        self.service.conn.executemany(
            """
            INSERT OR REPLACE INTO semantic_neighbors (token, neighbor, score, cooccurrence)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("автоматический", "выключатели", 0.62, 5),
                ("дефибриллятор", "defi", 0.95, 3),
                ("defi", "дефибриллятор", 0.95, 3),
            ],
        )
        self.service.conn.commit()

        expansions, applied = self.service.semantic_expander.expand_tokens(["автоматический", "дефибриллятор"])

        self.assertIn("defi", expansions)
        self.assertNotIn("выключатели", expansions)
        applied_sources = {item["source"] for item in applied}
        self.assertEqual(applied_sources, {"дефибриллятор"})

    def test_sqlite_sentence_similarity_uses_semantic_graph_links(self) -> None:
        self.service.conn.executemany(
            """
            INSERT OR REPLACE INTO token_frequency (token, first_char, token_length, frequency)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("дефибриллятор", "д", 13, 24),
                ("defi", "d", 4, 3),
            ],
        )
        self.service.conn.executemany(
            """
            INSERT OR REPLACE INTO semantic_neighbors (token, neighbor, score, cooccurrence)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("дефибриллятор", "defi", 0.18, 3),
                ("defi", "дефибриллятор", 0.18, 3),
            ],
        )
        self.service.conn.commit()

        similarity = self.service.semantic_expander.sentence_similarity(
            "наружный дефибриллятор",
            "defi b",
        )

        self.assertGreater(similarity, 0.25)

    def test_dense_retrieval_backfills_alias_only_candidate(self) -> None:
        self.service.close()

        conn = sqlite3.connect(self.search_db_path)
        conn.row_factory = sqlite3.Row
        conn.executemany(
            """
            INSERT OR REPLACE INTO semantic_neighbors (token, neighbor, score, cooccurrence)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("defi", "дефибриллятор", 0.24, 4),
            ],
        )
        conn.commit()
        conn.close()

        rebuild_dense_index(self.search_db_path)
        self.service = SearchService(
            search_db_path=self.search_db_path,
            synonyms_path=self.synonyms_path,
            semantic_backend="sqlite",
        )

        payload = self.service.search("дефибриллятор", top_k=5, min_score=0.0)

        result_ids = [item["ste_id"] for item in payload["results"][:5]]
        self.assertTrue(payload["query"]["dense_retrieval_enabled"])
        self.assertIn("ste-7", result_ids)
        alias_item = next(item for item in payload["results"] if item["ste_id"] == "ste-7")
        self.assertGreater(alias_item["search_features"]["dense_vector_similarity"], 0.0)

    def test_learned_dense_retrieval_backend_is_used_when_artifacts_exist(self) -> None:
        self.service.close()
        rebuild_learned_dense_artifacts(
            self.search_db_path,
            embedding_dim=16,
            sample_size=32,
            random_state=7,
        )
        self.service = SearchService(
            search_db_path=self.search_db_path,
            synonyms_path=self.synonyms_path,
            semantic_backend="sqlite",
        )

        payload = self.service.search("дефибриллятор", top_k=5, min_score=0.0)

        self.assertTrue(payload["query"]["dense_retrieval_enabled"])
        self.assertEqual(payload["query"]["dense_retrieval_backend"], "learned")
        result_ids = [item["ste_id"] for item in payload["results"][:5]]
        self.assertIn("ste-7", result_ids)
        learned_item = next(item for item in payload["results"] if item["ste_id"] == "ste-7")
        self.assertGreater(learned_item["search_features"]["dense_vector_similarity"], 0.0)


if __name__ == "__main__":
    unittest.main()
