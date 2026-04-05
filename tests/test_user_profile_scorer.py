from __future__ import annotations

import sys
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


if __name__ == "__main__":
    unittest.main()
