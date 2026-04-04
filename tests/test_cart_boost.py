"""
tests/test_cart_boost.py
========================
Unit-тесты для CartBoostModifier и InMemoryCartStorage.

Доказывают:
  - математику асимптотической функции
  - корректную работу хранилища (increment / decrement / TTL / bulk)
  - что товар с бустом обходит товар с таким же base_score без буста
  - комбинирование с RankingModifier (буст → штраф)
"""
from __future__ import annotations

import math
import sys
import threading
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.cart_boost import (
    CartBoostModifier,
    InMemoryCartStorage,
)
from tenderhack.penalization import InMemorySkipStorage, RankingModifier


# ---------------------------------------------------------------------------
# Вспомогательные фабричные функции
# ---------------------------------------------------------------------------

def make_modifier(
    max_boost: float = 0.15,
    saturation_rate: float = 1.0,
) -> tuple[CartBoostModifier, InMemoryCartStorage]:
    storage = InMemoryCartStorage()
    modifier = CartBoostModifier(storage, max_boost=max_boost, saturation_rate=saturation_rate)
    return modifier, storage


# ---------------------------------------------------------------------------
# 1. Математика мультипликатора
# ---------------------------------------------------------------------------

class TestCalculateMultiplier(unittest.TestCase):
    """multiplier = 1.0 + M * (1.0 - exp(-k * n)), M=0.15, k=1.0"""

    def setUp(self) -> None:
        self.modifier, _ = make_modifier()

    def test_zero_adds_returns_one(self) -> None:
        """0 добавлений → множитель ровно 1.0 (нет буста)."""
        self.assertEqual(self.modifier.calculate_multiplier(0), 1.0)

    def test_negative_adds_returns_one(self) -> None:
        """Отрицательные значения трактуются как 0."""
        self.assertEqual(self.modifier.calculate_multiplier(-5), 1.0)

    def test_one_add_approx(self) -> None:
        """1 добавление → множитель ≈ 1.0948 (≈ +9.5 % буста)."""
        # 1.0 + 0.15 * (1 - exp(-1)) ≈ 1.0 + 0.15 * 0.6321 ≈ 1.0948
        expected = 1.0 + 0.15 * (1.0 - math.exp(-1.0))
        self.assertAlmostEqual(
            self.modifier.calculate_multiplier(1),
            expected,
            places=9,
            msg="1 добавление должно давать ~9.5 % буста",
        )
        # Проверяем, что значение действительно ~1.09
        self.assertGreater(self.modifier.calculate_multiplier(1), 1.09)
        self.assertLess(self.modifier.calculate_multiplier(1), 1.10)

    def test_ten_adds_asymptote(self) -> None:
        """10 добавлений → множитель < 1.15 (асимптота работает)."""
        result = self.modifier.calculate_multiplier(10)
        self.assertLess(
            result,
            1.15,
            msg="При 10 добавлениях множитель не должен превышать 1.15",
        )
        # Должен быть очень близко к потолку
        self.assertGreater(result, 1.149)

    def test_large_n_stays_below_ceiling(self) -> None:
        """При очень большом n множитель стремится к потолку, но не превышает 1.0+M."""
        ceiling = 1.0 + self.modifier.max_boost
        for n in [50, 100, 1000]:
            result = self.modifier.calculate_multiplier(n)
            # При n ≥ 50, exp(-n) ≈ 0 в float64, поэтому результат == ceiling.
            # Используем assertLessEqual — множитель НИКОГДА не превышает потолок.
            self.assertLessEqual(
                result,
                ceiling,
                msg=f"При n={n} множитель {result} превысил потолок {ceiling}",
            )

    def test_monotonically_increasing(self) -> None:
        """Больше добавлений → больше буст (монотонно возрастает)."""
        prev = 1.0
        for n in range(1, 20):
            curr = self.modifier.calculate_multiplier(n)
            self.assertGreater(curr, prev, msg=f"Нарушена монотонность при n={n}")
            prev = curr

    def test_custom_parameters(self) -> None:
        """Кастомные M и k работают корректно."""
        modifier = CartBoostModifier(InMemoryCartStorage(), max_boost=0.3, saturation_rate=0.5)
        # 1.0 + 0.3 * (1 - exp(-0.5 * 2)) = 1.0 + 0.3 * (1 - exp(-1))
        expected = 1.0 + 0.3 * (1.0 - math.exp(-1.0))
        self.assertAlmostEqual(modifier.calculate_multiplier(2), expected, places=9)


# ---------------------------------------------------------------------------
# 2. InMemoryCartStorage — функциональность
# ---------------------------------------------------------------------------

class TestInMemoryCartStorage(unittest.TestCase):

    def setUp(self) -> None:
        self.storage = InMemoryCartStorage(ttl_seconds=3600)
        self.user = "user-42"
        self.cat = "Лекарственные препараты"

    def test_initial_count_is_zero(self) -> None:
        self.assertEqual(self.storage.get_cart_adds(self.user, self.cat), 0)

    def test_increment_returns_new_count(self) -> None:
        self.assertEqual(self.storage.increment_cart(self.user, self.cat), 1)
        self.assertEqual(self.storage.increment_cart(self.user, self.cat), 2)
        self.assertEqual(self.storage.increment_cart(self.user, self.cat), 3)

    def test_get_after_increment(self) -> None:
        self.storage.increment_cart(self.user, self.cat)
        self.storage.increment_cart(self.user, self.cat)
        self.assertEqual(self.storage.get_cart_adds(self.user, self.cat), 2)

    def test_decrement_reduces_count(self) -> None:
        self.storage.increment_cart(self.user, self.cat)
        self.storage.increment_cart(self.user, self.cat)
        self.assertEqual(self.storage.decrement_cart(self.user, self.cat), 1)

    def test_decrement_never_below_zero(self) -> None:
        """Декремент ниже нуля невозможен."""
        self.assertEqual(self.storage.decrement_cart(self.user, self.cat), 0)
        self.assertEqual(self.storage.decrement_cart(self.user, self.cat), 0)
        self.assertEqual(self.storage.get_cart_adds(self.user, self.cat), 0)

    def test_decrement_to_zero_removes_key(self) -> None:
        self.storage.increment_cart(self.user, self.cat)
        self.storage.decrement_cart(self.user, self.cat)
        # После удаления всех добавлений ключ больше не должен занимать память
        key = f"user:{self.user}:cat:{self.cat}:cart"
        self.assertNotIn(key, self.storage._store)

    def test_different_users_isolated(self) -> None:
        self.storage.increment_cart("user-A", self.cat)
        self.storage.increment_cart("user-A", self.cat)
        self.storage.increment_cart("user-B", self.cat)
        self.assertEqual(self.storage.get_cart_adds("user-A", self.cat), 2)
        self.assertEqual(self.storage.get_cart_adds("user-B", self.cat), 1)

    def test_different_categories_isolated(self) -> None:
        self.storage.increment_cart(self.user, "cat-A")
        self.storage.increment_cart(self.user, "cat-A")
        self.storage.increment_cart(self.user, "cat-B")
        self.assertEqual(self.storage.get_cart_adds(self.user, "cat-A"), 2)
        self.assertEqual(self.storage.get_cart_adds(self.user, "cat-B"), 1)

    def test_bulk_get(self) -> None:
        self.storage.increment_cart(self.user, "cat-X")
        self.storage.increment_cart(self.user, "cat-X")
        self.storage.increment_cart(self.user, "cat-Y")
        result = self.storage.get_bulk_cart_adds(self.user, ["cat-X", "cat-Y", "cat-Z"])
        self.assertEqual(result["cat-X"], 2)
        self.assertEqual(result["cat-Y"], 1)
        self.assertEqual(result["cat-Z"], 0)

    def test_ttl_expiry(self) -> None:
        """Записи с истёкшим TTL возвращают 0."""
        storage = InMemoryCartStorage(ttl_seconds=0)  # TTL = максимум 1 секунда
        # Записываем напрямую с прошедшим временем
        key = f"user:{self.user}:cat:{self.cat}:cart"
        storage._store[key] = (5, time.time() - 1)  # уже истёк

        self.assertEqual(storage.get_cart_adds(self.user, self.cat), 0)

    def test_thread_safety(self) -> None:
        """Конкурентные инкременты из разных потоков не теряют данные."""
        n_threads = 20
        n_per_thread = 50
        threads = [
            threading.Thread(
                target=lambda: [self.storage.increment_cart(self.user, self.cat) for _ in range(n_per_thread)]
            )
            for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(
            self.storage.get_cart_adds(self.user, self.cat),
            n_threads * n_per_thread,
        )


# ---------------------------------------------------------------------------
# 3. apply_boost — интеграционный тест ранжирования
# ---------------------------------------------------------------------------

class TestApplyBoost(unittest.TestCase):

    def setUp(self) -> None:
        self.modifier, self.storage = make_modifier()
        self.user = "user-99"

    def _make_recs(self) -> list[dict]:
        return [
            {"ste_id": "A", "category": "Медикаменты",  "search_score": 10.0, "final_score": 10.0},
            {"ste_id": "B", "category": "Канцтовары",   "search_score": 10.0, "final_score": 10.0},
            {"ste_id": "C", "category": "Оргтехника",   "search_score":  8.0, "final_score":  8.0},
        ]

    def test_no_cart_adds_no_change(self) -> None:
        """Без добавлений в корзину final_score не меняется."""
        recs = self._make_recs()
        result = self.modifier.apply_boost(recs, self.user)
        scores = {r["ste_id"]: r["final_score"] for r in result}
        self.assertEqual(scores["A"], 10.0)
        self.assertEqual(scores["B"], 10.0)
        self.assertEqual(scores["C"], 8.0)

    def test_boosted_item_overtakes_equal_score(self) -> None:
        """Товар с бустом обходит товар с таким же базовым скором без буста."""
        # Добавляем "Медикаменты" в корзину один раз
        self.storage.increment_cart(self.user, "Медикаменты")

        recs = self._make_recs()
        result = self.modifier.apply_boost(recs, self.user)

        # A (Медикаменты) должен быть выше B (Канцтовары) несмотря на одинаковый base_score
        ids_in_order = [r["ste_id"] for r in result]
        self.assertEqual(ids_in_order[0], "A", "Медикамент с бустом должен быть первым")
        self.assertEqual(ids_in_order[1], "B")

    def test_boost_multiplier_recorded(self) -> None:
        """Множитель и количество добавлений сохраняются в записи для дебага."""
        self.storage.increment_cart(self.user, "Медикаменты")
        recs = self._make_recs()
        result = self.modifier.apply_boost(recs, self.user)

        boosted = next(r for r in result if r["ste_id"] == "A")
        self.assertIn("cart_boost_multiplier", boosted)
        self.assertIn("cart_adds", boosted)
        self.assertEqual(boosted["cart_adds"], 1)
        self.assertAlmostEqual(
            boosted["cart_boost_multiplier"],
            1.0 + 0.15 * (1.0 - math.exp(-1.0)),
            places=5,
        )

    def test_no_boost_fields_on_non_boosted(self) -> None:
        """Записи без буста не получают лишних полей."""
        self.storage.increment_cart(self.user, "Медикаменты")
        recs = self._make_recs()
        result = self.modifier.apply_boost(recs, self.user)

        non_boosted = next(r for r in result if r["ste_id"] == "B")
        self.assertNotIn("cart_boost_multiplier", non_boosted)

    def test_multiple_cart_adds_increase_boost(self) -> None:
        """Несколько добавлений → больший буст (в пределах асимптоты)."""
        for _ in range(5):
            self.storage.increment_cart(self.user, "Медикаменты")

        recs = self._make_recs()
        result = self.modifier.apply_boost(recs, self.user)
        boosted = next(r for r in result if r["ste_id"] == "A")

        # Максимальный буст — 15 %, то есть final_score ≤ 10.0 * 1.15 = 11.5
        self.assertLessEqual(boosted["final_score"], 11.5)
        self.assertGreater(boosted["final_score"], 10.0)

    def test_missing_category_field_handled(self) -> None:
        """Записи без поля category не вызывают ошибок."""
        recs = [
            {"ste_id": "X", "search_score": 5.0, "final_score": 5.0},  # нет category
        ]
        result = self.modifier.apply_boost(recs, self.user)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ste_id"], "X")


# ---------------------------------------------------------------------------
# 4. Комбинирование: CartBoostModifier → RankingModifier (пессимизация)
# ---------------------------------------------------------------------------

class TestCombinedBoostandPenalty(unittest.TestCase):
    """Буст за корзину выполняется ДО штрафа за скипы."""

    def setUp(self) -> None:
        # Буст
        self.cart_storage = InMemoryCartStorage()
        self.boost_modifier = CartBoostModifier(self.cart_storage)

        # Пессимизация
        self.skip_storage = InMemorySkipStorage()
        self.penalty_modifier = RankingModifier(self.skip_storage)

        self.user = "combined-user"

    def test_boost_then_penalty_correct_order(self) -> None:
        """Пессимизированный товар с бустом не обходит чистый товар из другой категории."""
        # "Медикаменты" добавлены в корзину (буст)
        self.cart_storage.increment_cart(self.user, "Медикаменты")

        # "Медикаменты" также получили 3 скипа (штраф)
        for _ in range(3):
            self.skip_storage.increment_skip(self.user, "Медикаменты")

        recs = [
            {"ste_id": "A", "category": "Медикаменты", "category_id": "Медикаменты",
             "search_score": 10.0, "final_score": 10.0},
            {"ste_id": "B", "category": "Канцтовары", "category_id": "Канцтовары",
             "search_score": 10.0, "final_score": 10.0},
        ]

        # Шаг 1: применяем буст
        after_boost = self.boost_modifier.apply_boost(recs, self.user)

        # Шаг 2: применяем штраф
        final = self.penalty_modifier.apply_penalties(after_boost, self.user)

        # "Медикаменты": буст +~9.5%, штраф ×0.6 → ~10.948 * 0.6 ≈ 6.57
        # "Канцтовары": без изменений → 10.0
        med_item = next(r for r in final if r["ste_id"] == "A")
        kan_item = next(r for r in final if r["ste_id"] == "B")

        self.assertLess(
            med_item["final_score"],
            kan_item["final_score"],
            msg="Товар с 3 скипами должен быть пессимизирован даже при бусте из корзины",
        )
        self.assertEqual(final[0]["ste_id"], "B")

    def test_boost_without_penalty_works(self) -> None:
        """Буст без пессимизации корректно повышает итоговый скор."""
        self.cart_storage.increment_cart(self.user, "cat-A")

        recs = [
            {"ste_id": "A", "category": "cat-A", "category_id": "cat-A",
             "search_score": 10.0, "final_score": 10.0},
            {"ste_id": "B", "category": "cat-B", "category_id": "cat-B",
             "search_score": 10.0, "final_score": 10.0},
        ]

        after_boost = self.boost_modifier.apply_boost(recs, self.user)
        final = self.penalty_modifier.apply_penalties(after_boost, self.user)

        self.assertEqual(final[0]["ste_id"], "A")
        self.assertGreater(
            next(r for r in final if r["ste_id"] == "A")["final_score"],
            10.0,
        )


if __name__ == "__main__":
    unittest.main()
