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

from tenderhack.penalization import InMemorySkipStorage, InteractionTracker, RankingModifier


class PenalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = InMemorySkipStorage()
        self.tracker = InteractionTracker(self.storage)
        self.modifier = RankingModifier(self.storage)
        self.user_id = "test-user"

    def test_multiplier_math(self) -> None:
        self.assertAlmostEqual(self.modifier.calculate_multiplier(0), 1.0)
        self.assertAlmostEqual(self.modifier.calculate_multiplier(3), 0.6)
        self.assertAlmostEqual(self.modifier.calculate_multiplier(10), 0.4)

    def test_category_ranking_penalty(self) -> None:
        base_recommendations = [
            {"item_id": 1, "category_id": "Laptops", "base_score": 100.0},
            {"item_id": 2, "category_id": "Smartphones", "base_score": 80.0},
            {"item_id": 3, "category_id": "Accessories", "base_score": 50.0},
        ]

        for _ in range(4):
            self.tracker.register_view(self.user_id, "Laptops", 1500)

        self.tracker.register_view(self.user_id, "Smartphones", 45000)

        final_recommendations = self.modifier.apply_penalties(base_recommendations, self.user_id)

        self.assertEqual(final_recommendations[0]["category_id"], "Smartphones")
        self.assertEqual(final_recommendations[1]["category_id"], "Laptops")
        self.assertEqual(final_recommendations[2]["category_id"], "Accessories")
        self.assertAlmostEqual(final_recommendations[1]["final_score"], 60.0)

    def test_first_quick_exit_is_forgiven(self) -> None:
        outcome = self.tracker.register_view(self.user_id, "Laptops", 1500)
        self.assertEqual(outcome, "forgiven")
        self.assertEqual(self.storage.get_skips(self.user_id, "Laptops"), 0)

        outcome = self.tracker.register_view(self.user_id, "Laptops", 1500)
        self.assertEqual(outcome, "applied")
        self.assertEqual(self.storage.get_skips(self.user_id, "Laptops"), 1)

    def test_long_view_does_not_create_skip_outcome(self) -> None:
        outcome = self.tracker.register_view(self.user_id, "Laptops", 4500)
        self.assertEqual(outcome, "none")
        self.assertEqual(self.storage.get_skips(self.user_id, "Laptops"), 0)


if __name__ == "__main__":
    unittest.main()
