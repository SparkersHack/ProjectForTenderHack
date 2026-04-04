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

from scripts.build_search_assets import build_search_db
from scripts.generate_search_synonyms import generate_synonyms_payload
from tenderhack.search import SearchService


class SearchSynonymGeneratorTests(unittest.TestCase):
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
                    "ste-phone",
                    "Телефон сотовой связи мобильный",
                    "телефон сотовой связи мобильный",
                    "Телефоны мобильные",
                    "телефоны мобильные",
                    "Тип связи | Цвет",
                    "2",
                    "телефон мобильный сотовый смартфон",
                ]
            )
            writer.writerow(
                [
                    "ste-mfu",
                    "Многофункциональное устройство (МФУ) лазерное",
                    "многофункциональное устройство мфу лазерное",
                    "Многофункциональные устройства (МФУ)",
                    "многофункциональные устройства мфу",
                    "Тип печати | Форм-фактор",
                    "2",
                    "мфу многофункциональное устройство принтер сканер копир",
                ]
            )
            writer.writerow(
                [
                    "ste-sand",
                    "Аппарат абразивоструйный",
                    "аппарат абразивоструйный",
                    "Аппараты абразивоструйные (пескоструйные)",
                    "аппараты абразивоструйные пескоструйные",
                    "Тип",
                    "1",
                    "аппарат абразивоструйный пескоструйный",
                ]
            )
            writer.writerow(
                [
                    "ste-reagent",
                    "Кислота борная реактив",
                    "кислота борная реактив",
                    "Химические реактивы",
                    "химические реактивы",
                    "Тип",
                    "1",
                    "кислота борная реактив",
                ]
            )
            writer.writerow(
                [
                    "ste-chair",
                    "Стул офисный",
                    "стул офисный",
                    "Кресла и стулья",
                    "кресла и стулья",
                    "Материал",
                    "1",
                    "стул кресло офисный",
                ]
            )
            writer.writerow(
                [
                    "ste-software",
                    "Программное обеспечение антивирусное",
                    "программное обеспечение антивирусное",
                    "Программное обеспечение",
                    "программное обеспечение",
                    "Тип лицензии",
                    "1",
                    "программное обеспечение софт software",
                ]
            )
            writer.writerow(
                [
                    "ste-iron",
                    "Железо",
                    "железо",
                    "Железо (III)",
                    "железо iii",
                    "Форма",
                    "1",
                    "железо iii",
                ]
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

    def _write_generated_synonyms(self) -> dict[str, object]:
        payload = generate_synonyms_payload(self.catalog_path, min_auto_pair_count=1, max_targets_per_source=8)
        self.synonyms_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def test_generator_keeps_user_phone_synonyms(self) -> None:
        payload = self._write_generated_synonyms()
        phone_targets = payload["token_synonyms"].get("мобильник", [])
        self.assertIn("мобильный телефон", phone_targets)

    def test_generator_extracts_parenthetical_alias(self) -> None:
        payload = self._write_generated_synonyms()
        mfu_targets = payload["token_synonyms"].get("мфу", [])
        self.assertIn("многофункциональное устройство", mfu_targets)

    def test_generator_rejects_adjectival_parenthetical_alias(self) -> None:
        payload = self._write_generated_synonyms()
        self.assertNotIn("пескоструйные", payload["token_synonyms"])
        self.assertNotIn("пескоструйные", payload["phrase_synonyms"])

    def test_generator_does_not_emit_generic_or_ambiguous_relations(self) -> None:
        payload = self._write_generated_synonyms()
        self.assertNotIn("реактив", payload["token_synonyms"])
        self.assertNotIn("по", payload["token_synonyms"])
        self.assertNotIn("кресло", payload["token_synonyms"].get("стул", []))
        self.assertNotIn("iii", payload["token_synonyms"])

    def test_generated_synonyms_help_search_runtime(self) -> None:
        self._write_generated_synonyms()
        service = SearchService(
            search_db_path=self.search_db_path,
            synonyms_path=self.synonyms_path,
            semantic_backend="sqlite",
        )
        try:
            payload = service.search("мобильник", top_k=3, min_score=0.0)
        finally:
            service.close()
        self.assertEqual(payload["results"][0]["ste_id"], "ste-phone")
        self.assertTrue(payload["query"]["applied_synonyms"])


if __name__ == "__main__":
    unittest.main()
