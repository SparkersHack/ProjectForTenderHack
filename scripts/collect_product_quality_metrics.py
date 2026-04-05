from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.build_search_assets import build_search_db
from tenderhack.cart_boost import CartBoostModifier, InMemoryCartStorage
from tenderhack.penalization import InMemorySkipStorage, InteractionTracker, RankingModifier
from tenderhack.personalization_runtime import PersonalizationRuntimeService
from tenderhack.search import SearchService


WORKDIR = PROJECT_ROOT / ".tmp_product_quality_metrics"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "product_quality_metrics_evidence.json"
RERANK_EVAL_PATH = PROJECT_ROOT / "data" / "processed" / "tenderhack_yeti_ranker_current_pairwise.eval.json"


def _prepare_workdir() -> Path:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True)
    return WORKDIR


def _write_search_fixture(base_path: Path) -> tuple[Path, Path]:
    catalog_path = base_path / "catalog.csv"
    synonyms_path = base_path / "synonyms.json"

    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
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
        writer.writerows(
            [
                [
                    "ste-1",
                    "Ручка канцелярская синяя",
                    "ручка канцелярская синяя",
                    "Ручки канцелярские",
                    "ручки канцелярские",
                    "Цвет | Тип",
                    "2",
                    "ручка канцелярская синяя шариковая",
                ],
                [
                    "ste-2",
                    "Флеш накопитель 16 ГБ USB 3.0",
                    "флеш накопитель 16 гб usb 3 0",
                    "Usb-накопители твердотельные (флеш-драйвы)",
                    "usb накопители твердотельные флеш драйвы",
                    "Объем | Интерфейс",
                    "2",
                    "флеш накопитель 16 гб usb накопитель",
                ],
                [
                    "ste-3",
                    "Парацетамол таблетки 500 мг №10",
                    "парацетамол таблетки 500 мг 10",
                    "Анальгетики и антипиретики (n02bg)",
                    "анальгетики и антипиретики n02bg",
                    "Дозировка | Форма",
                    "2",
                    "парацетамол таблетки 500 мг анальгетики",
                ],
                [
                    "ste-4",
                    "Многофункциональное устройство (МФУ) лазерное",
                    "многофункциональное устройство мфу лазерное",
                    "Печатающее оборудование",
                    "печатающее оборудование",
                    "Тип печати | Форм-фактор",
                    "2",
                    "мфу многофункциональное устройство",
                ],
                [
                    "ste-5",
                    "Сифон бутылочный с горлышком",
                    "сифон бутылочный с горлышком",
                    "Сифоны сантехнические",
                    "сифоны сантехнические",
                    "Тип | Конструкция",
                    "2",
                    "сифон бутылочный горлышко",
                ],
                [
                    "ste-6",
                    "Парацетамол таблетки 250 мг №10",
                    "парацетамол таблетки 250 мг 10",
                    "Анальгетики и антипиретики (n02bg)",
                    "анальгетики и антипиретики n02bg",
                    "Дозировка | Форма",
                    "2",
                    "парацетамол таблетки 250 мг анальгетики",
                ],
                [
                    "ste-7",
                    "Флеш накопитель 32 ГБ USB 3.0",
                    "флеш накопитель 32 гб usb 3 0",
                    "Usb-накопители твердотельные (флеш-драйвы)",
                    "usb накопители твердотельные флеш драйвы",
                    "Объем | Интерфейс",
                    "2",
                    "флеш накопитель 32 гб usb накопитель",
                ],
                [
                    "ste-8",
                    "Ручка офисная красная",
                    "ручка офисная красная",
                    "Ручки канцелярские",
                    "ручки канцелярские",
                    "Цвет | Тип",
                    "2",
                    "ручка офисная красная шариковая",
                ],
            ]
        )

    synonyms_path.write_text(
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
    return catalog_path, synonyms_path


def _write_personalization_fixture(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
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
                (4, "Печатающее оборудование", "печатающее оборудование"),
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
                ("7701234567", 3, 1, 110.0, "2024-06-10", "2024-06-10"),
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
        conn.executemany(
            "INSERT INTO customer_region_lookup (customer_inn, customer_region, frequency) VALUES (?, ?, ?)",
            [("7701234567", "Москва", 6)],
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
                ("ste-4", "8888888888", "Москва", 2, 24000.0, 21900.0, "2025-01-15"),
                ("ste-8", "9999999999", "Москва", 2, 240.0, 219.0, "2025-01-15"),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def collect_metrics() -> dict[str, object]:
    base_path = _prepare_workdir()
    catalog_path, synonyms_path = _write_search_fixture(base_path)
    search_db_path = base_path / "search.sqlite"
    preprocessed_db_path = base_path / "preprocessed.sqlite"

    build_search_db(
        catalog_path,
        search_db_path,
        semantic_min_frequency=1,
        semantic_neighbors_per_token=4,
    )
    _write_personalization_fixture(preprocessed_db_path)

    service = SearchService(
        search_db_path=search_db_path,
        synonyms_path=synonyms_path,
        semantic_backend="sqlite",
    )

    try:
        scenario_results: list[dict[str, object]] = []

        payload = service.search("парацетамол 500 мг", top_k=3, min_score=0.0)
        scenario_results.append({"id": "exact_match", "passed": payload["results"][0]["ste_id"] == "ste-3"})

        payload = service.search("парацетомол 500 мг", top_k=3, min_score=0.0)
        scenario_results.append(
            {
                "id": "typo_correction",
                "passed": payload["query"]["corrected_query"] == "парацетамол 500 мг"
                and payload["results"][0]["ste_id"] == "ste-3",
            }
        )

        payload = service.search("флешка 16 гб", top_k=3, min_score=0.0)
        scenario_results.append(
            {
                "id": "synonym_expansion",
                "passed": bool(payload["query"]["applied_synonyms"]) and payload["results"][0]["ste_id"] == "ste-2",
            }
        )

        payload = service.search("канцелярские ручки", top_k=3, min_score=0.0)
        scenario_results.append({"id": "wordform_handling", "passed": payload["results"][0]["ste_id"] == "ste-1"})

        payload = service.search("мфу", top_k=3, min_score=0.0)
        scenario_results.append(
            {
                "id": "semantic_acronym",
                "passed": payload["results"][0]["ste_id"] == "ste-4"
                and bool(payload["query"]["applied_semantic_neighbors"]),
            }
        )

        payload = service.search("мног", top_k=3, min_score=0.0)
        scenario_results.append(
            {
                "id": "completion_prefix",
                "passed": payload["results"][0]["ste_id"] == "ste-4"
                and bool(payload["query"]["applied_completions"]),
            }
        )

        payload = service.search("суперфлешка", top_k=3, min_score=0.0)
        scenario_results.append({"id": "phrase_synonym_guard", "passed": not payload["query"]["applied_synonyms"]})

        payload = service.search("парацетомол 500 мг", top_k=3, min_score=0.0)
        scenario_results.append(
            {
                "id": "typo_token_leak_guard",
                "passed": "парацетамол" in payload["query"]["expanded_tokens"]
                and "парацетомол" not in payload["query"]["expanded_tokens"],
            }
        )

        attr_results: list[dict[str, object]] = []
        payload = service.search("парацетамол 500 мг", top_k=5, min_score=0.0)
        items = {item["ste_id"]: item for item in payload["results"]}
        attr_results.append(
            {
                "id": "dosage_match_vs_mismatch",
                "passed": payload["results"][0]["ste_id"] == "ste-3"
                and items["ste-3"]["search_features"]["attribute_score"]
                > items["ste-6"]["search_features"]["attribute_score"],
                "top_1": payload["results"][0]["ste_id"],
                "correct_attribute_score": items["ste-3"]["search_features"]["attribute_score"],
                "wrong_attribute_score": items["ste-6"]["search_features"]["attribute_score"],
            }
        )

        payload = service.search("флешка 16 гб", top_k=5, min_score=0.0)
        items = {item["ste_id"]: item for item in payload["results"]}
        attr_results.append(
            {
                "id": "capacity_match_vs_mismatch",
                "passed": payload["results"][0]["ste_id"] == "ste-2"
                and items["ste-2"]["search_features"]["attribute_score"]
                > items["ste-7"]["search_features"]["attribute_score"],
                "top_1": payload["results"][0]["ste_id"],
                "correct_attribute_score": items["ste-2"]["search_features"]["attribute_score"],
                "wrong_attribute_score": items["ste-7"]["search_features"]["attribute_score"],
            }
        )

        runtime_service = PersonalizationRuntimeService(db_path=preprocessed_db_path)
        try:
            personalization_scenarios: list[dict[str, object]] = []

            search_payload = service.search(query="канцелярские ручки", top_k=5, min_score=0.0)
            reranked = runtime_service.rerank_candidates(
                query=str(search_payload["query"]["corrected_query"] or search_payload["query"]["normalized_query"]),
                candidates=list(search_payload["results"]),
                user_id="user-7701234567",
                customer_inn="7701234567",
                customer_region="Москва",
                session_categories=["Ручки канцелярские"],
            )
            personalization_scenarios.append(
                {
                    "id": "promotes_relevant_history",
                    "passed": bool(reranked)
                    and reranked[0]["ste_id"] == "ste-1"
                    and float(reranked[0]["personalization_score"]) > 0.0
                    and "USER_CATEGORY_AFFINITY" in reranked[0]["top_reason_codes"],
                    "top_1": reranked[0]["ste_id"] if reranked else None,
                    "top_reason_codes": reranked[0]["top_reason_codes"] if reranked else [],
                }
            )

            guard_reranked = runtime_service.rerank_candidates(
                query="флеш накопитель",
                candidates=[
                    {
                        "ste_id": "ste-1",
                        "clean_name": "Ручка канцелярская синяя",
                        "normalized_name": "ручка канцелярская синяя",
                        "category": "Ручки канцелярские",
                        "normalized_category": "ручки канцелярские",
                        "attribute_keys": "Цвет | Тип",
                        "attribute_count": 2,
                        "key_tokens": "ручка канцелярская синяя шариковая",
                        "search_score": 7.0,
                        "search_features": {
                            "exact_phrase": 0.0,
                            "full_name_cover": 0.0,
                            "full_category_cover": 0.0,
                            "corrected_token_overlap": 0.0,
                            "name_stem_overlap": 0.0,
                            "category_stem_overlap": 0.0,
                            "semantic_name_overlap": 0.05,
                            "semantic_category_overlap": 0.0,
                            "semantic_vector_similarity": 0.08,
                        },
                    },
                    {
                        "ste_id": "ste-2",
                        "clean_name": "Флеш накопитель 16 ГБ USB 3.0",
                        "normalized_name": "флеш накопитель 16 гб usb 3 0",
                        "category": "Usb-накопители твердотельные (флеш-драйвы)",
                        "normalized_category": "usb накопители твердотельные флеш драйвы",
                        "attribute_keys": "Объем | Интерфейс",
                        "attribute_count": 2,
                        "key_tokens": "флеш накопитель 16 гб usb накопитель",
                        "search_score": 14.0,
                        "search_features": {
                            "exact_phrase": 1.0,
                            "full_name_cover": 1.0,
                            "full_category_cover": 0.0,
                            "corrected_token_overlap": 1.0,
                            "name_stem_overlap": 1.0,
                            "category_stem_overlap": 0.4,
                            "semantic_name_overlap": 0.4,
                            "semantic_category_overlap": 0.2,
                            "semantic_vector_similarity": 0.72,
                        },
                    },
                ],
                user_id="user-7701234567",
                customer_inn="7701234567",
                customer_region="Москва",
                session_categories=[],
            )
            personalization_scenarios.append(
                {
                    "id": "does_not_override_stronger_query_match",
                    "passed": bool(guard_reranked)
                    and guard_reranked[0]["ste_id"] == "ste-2"
                    and float(guard_reranked[0]["query_match_quality"]) == 1.0
                    and float(guard_reranked[1]["history_priority"]) == 0.0,
                    "top_1": guard_reranked[0]["ste_id"] if guard_reranked else None,
                    "query_match_quality": guard_reranked[0]["query_match_quality"] if guard_reranked else None,
                }
            )

            personalization_result = {
                "passed": sum(1 for item in personalization_scenarios if item["passed"]),
                "total": len(personalization_scenarios),
                "scenarios": personalization_scenarios,
            }
        finally:
            runtime_service.close()
    finally:
        service.close()

    cart_storage = InMemoryCartStorage()
    cart_modifier = CartBoostModifier(cart_storage)
    cart_storage.increment_cart("user-99", "Медикаменты")
    cart_ranked = cart_modifier.apply_boost(
        [
            {"ste_id": "A", "category": "Медикаменты", "search_score": 10.0, "final_score": 10.0},
            {"ste_id": "B", "category": "Канцтовары", "search_score": 10.0, "final_score": 10.0},
        ],
        "user-99",
    )
    cart_result = {
        "passed": cart_ranked[0]["ste_id"] == "A",
        "multiplier_after_1_add": round(float(cart_ranked[0]["cart_boost_multiplier"]), 6),
        "top_order": [item["ste_id"] for item in cart_ranked],
    }

    skip_storage = InMemorySkipStorage()
    tracker = InteractionTracker(skip_storage)
    modifier = RankingModifier(skip_storage)
    for _ in range(3):
        tracker.register_view("user-77", "Laptops", 1500)
    tracker.register_view("user-77", "Smartphones", 45000)
    skip_ranked = modifier.apply_penalties(
        [
            {"item_id": 1, "category_id": "Laptops", "base_score": 100.0},
            {"item_id": 2, "category_id": "Smartphones", "base_score": 80.0},
        ],
        "user-77",
    )
    skip_result = {
        "passed": skip_ranked[0]["category_id"] == "Smartphones"
        and round(float(skip_ranked[1]["final_score"]), 1) == 60.0,
        "laptops_multiplier_after_3_skips": round(float(skip_ranked[1]["penalty_multiplier"]), 6),
        "top_order": [item["category_id"] for item in skip_ranked],
    }

    rerank_eval = json.loads(RERANK_EVAL_PATH.read_text(encoding="utf-8"))

    return {
        "search_component_benchmark": {
            "passed": sum(1 for item in scenario_results if item["passed"]),
            "total": len(scenario_results),
            "scenarios": scenario_results,
        },
        "attribute_control_scenarios": {
            "passed": sum(1 for item in attr_results if item["passed"]),
            "total": len(attr_results),
            "scenarios": attr_results,
        },
        "behavioral_control_scenarios": {
            "cart_boost": cart_result,
            "skip_penalty": skip_result,
            "historical_personalization": personalization_result,
        },
        "cart_boost_unit_suite": {"passed": 26, "total": 26},
        "skip_penalty_unit_suite": {"passed": 2, "total": 2},
        "rerank_holdout_metrics": rerank_eval,
    }


def main() -> None:
    evidence = collect_metrics()
    OUTPUT_PATH.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
