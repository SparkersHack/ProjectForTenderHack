from __future__ import annotations

import csv
import json
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

from scripts.build_real_search_benchmark import build_real_search_benchmark
from scripts.build_search_assets import build_search_db
from tenderhack.contract_queries import choose_benchmark_query


class RealSearchBenchmarkTests(unittest.TestCase):
    def test_choose_benchmark_query_prefers_service_stripped_variant(self) -> None:
        result = choose_benchmark_query("Поставка ручек канцелярских синих")
        self.assertIsNotNone(result)
        variant_name, query = result or ("", "")
        self.assertEqual(variant_name, "contract_service_stripped")
        self.assertEqual(query, "ручек канцелярских синих")

    def test_build_real_search_benchmark_aggregates_positive_ste_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            catalog_path = temp_dir / "catalog.csv"
            search_db_path = temp_dir / "search.sqlite"
            contracts_path = temp_dir / "contracts.csv"
            output_path = temp_dir / "benchmark.json"

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
                writer.writerow(
                    [
                        "ste-1",
                        "Ручка канцелярская синяя",
                        "ручка канцелярская синяя",
                        "Ручки канцелярские",
                        "ручки канцелярские",
                        "Цвет | Тип",
                        "2",
                        "ручка канцелярская синяя",
                    ]
                )
                writer.writerow(
                    [
                        "ste-2",
                        "Маркер черный",
                        "маркер черный",
                        "Маркеры",
                        "маркеры",
                        "Цвет | Тип",
                        "2",
                        "маркер черный",
                    ]
                )

            build_search_db(catalog_path, search_db_path, semantic_min_frequency=1, semantic_neighbors_per_token=4)

            with contracts_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(
                    [
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
                )
                writer.writerow(
                    ["Поставка ручек канцелярских синих", "c-1", "ste-1", "", "", "1", "", "77", "", "", ""]
                )
                writer.writerow(
                    ["Поставка ручек канцелярских синих", "c-2", "ste-1", "", "", "2", "", "77", "", "", ""]
                )
                writer.writerow(
                    ["Поставка ручек канцелярских синих", "c-3", "ste-1", "", "", "3", "", "77", "", "", ""]
                )
                writer.writerow(
                    ["Поставка маркеров черных", "c-4", "ste-2", "", "", "4", "", "77", "", "", ""]
                )
                writer.writerow(
                    ["Поставка маркеров черных", "c-5", "ste-2", "", "", "5", "", "77", "", "", ""]
                )

            summary = build_real_search_benchmark(
                contracts_path=contracts_path,
                search_db_path=search_db_path,
                output_path=output_path,
                min_query_support=2,
                min_positive_support=2,
                min_dominant_share=0.6,
                max_positive_ste_ids=3,
                max_queries=10,
                progress_every=0,
            )

            self.assertEqual(summary["benchmark_items"], 2)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            benchmark_by_query = {item["query"]: item for item in payload}
            self.assertIn("ручек канцелярских синих", benchmark_by_query)
            self.assertEqual(benchmark_by_query["ручек канцелярских синих"]["positive_ste_ids"], ["ste-1"])


if __name__ == "__main__":
    unittest.main()
