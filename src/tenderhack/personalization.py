from __future__ import annotations

import csv
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .text import normalize_text, stem_tokens, tokenize


DEFAULT_PREPROCESSED_DB = Path("data/processed/tenderhack_preprocessed.sqlite")

INSTITUTION_ARCHETYPE_LABELS = {
    "healthcare": "медицинских учреждений",
    "education": "образовательных учреждений",
    "office_admin": "офисно-административных учреждений",
    "facilities": "эксплуатационно-хозяйственных учреждений",
    "security_it": "учреждений с повышенным ИТ/безопасностным профилем",
    "general": "похожих учреждений",
}

INSTITUTION_ARCHETYPE_PROFILE_LABELS = {
    "healthcare": "Медицинская организация",
    "education": "Образовательная организация",
    "office_admin": "Административная организация",
    "facilities": "Эксплуатационно-хозяйственная организация",
    "security_it": "ИТ / безопасность",
    "general": "Общий профиль",
}

INSTITUTION_ARCHETYPE_KEYWORDS = {
    "healthcare": {
        "медицинские",
        "препарат",
        "лекарственные",
        "дезинфицирующие",
        "санитарно-противоэпидемические",
        "иммунодепрессанты",
        "анальгетики",
        "фармацевтические",
        "шприцы",
        "инсулин",
    },
    "education": {
        "обучение",
        "квалификация",
        "образование",
        "учебные",
        "школьные",
        "учитель",
        "педагогические",
        "воспитание",
    },
    "office_admin": {
        "канцелярские",
        "бумага",
        "ручки",
        "папки",
        "файлы",
        "маркеры",
        "карандаши",
        "степлеры",
        "скрепки",
        "картриджи",
        "принтеров",
        "мфу",
        "документообороту",
    },
    "facilities": {
        "уборочный",
        "мешки",
        "бумажные",
        "туалетная",
        "мыло",
        "уборки",
        "отходами",
        "ремонту",
        "строительные",
        "полотенца",
        "ветошь",
    },
    "security_it": {
        "пожарной",
        "безопасности",
        "охраны",
        "охранный",
        "информационной",
        "техническое",
        "мониторинг",
        "технологического",
        "оборудования",
        "сертификации",
    },
}

ARCHETYPE_CATEGORY_BLEND_WEIGHTS = {
    "healthcare": {"institution": 4.9, "peer": 2.7, "region": 0.0, "archetype": 1.9, "diversity": 0.12},
    "education": {"institution": 4.5, "peer": 2.5, "region": 0.0, "archetype": 1.7, "diversity": 0.12},
    "office_admin": {"institution": 4.1, "peer": 1.9, "region": 0.0, "archetype": 1.4, "diversity": 0.10},
    "facilities": {"institution": 4.0, "peer": 1.8, "region": 0.0, "archetype": 1.4, "diversity": 0.10},
    "security_it": {"institution": 4.7, "peer": 2.4, "region": 0.0, "archetype": 1.8, "diversity": 0.11},
    "general": {"institution": 4.3, "peer": 2.1, "region": 0.0, "archetype": 0.0, "diversity": 0.10},
}

ARCHETYPE_STE_BLEND_WEIGHTS = {
    "healthcare": {"institution": 5.2, "peer": 2.6, "region": 0.0, "archetype": 1.8, "diversity": 0.10},
    "education": {"institution": 4.8, "peer": 2.4, "region": 0.0, "archetype": 1.6, "diversity": 0.10},
    "office_admin": {"institution": 4.4, "peer": 1.8, "region": 0.0, "archetype": 1.3, "diversity": 0.09},
    "facilities": {"institution": 4.3, "peer": 1.8, "region": 0.0, "archetype": 1.3, "diversity": 0.09},
    "security_it": {"institution": 4.9, "peer": 2.3, "region": 0.0, "archetype": 1.7, "diversity": 0.09},
    "general": {"institution": 4.6, "peer": 2.0, "region": 0.0, "archetype": 0.0, "diversity": 0.09},
}

ARCHETYPE_KEYWORD_STEMS = {
    archetype: {stem for keyword in keywords for stem in stem_tokens(tokenize(normalize_text(keyword))) if stem}
    for archetype, keywords in INSTITUTION_ARCHETYPE_KEYWORDS.items()
}

CUSTOMER_NAME_ARCHETYPE_KEYWORDS = {
    "healthcare": {
        "гбуз",
        "обуз",
        "больница",
        "поликлиника",
        "клиническая",
        "госпиталь",
        "диспансер",
        "роддом",
        "родильный",
        "перинатальный",
        "санаторий",
        "медицинский",
        "медико",
        "стоматологическая",
        "амбулатория",
        "фельдшерский",
        "аптека",
    },
    "education": {
        "гбоу",
        "мбоу",
        "сош",
        "школа",
        "гимназия",
        "лицей",
        "колледж",
        "техникум",
        "университет",
        "институт",
        "академия",
        "училище",
        "образовательный",
        "детский",
        "сад",
        "доу",
    },
    "office_admin": {
        "администрация",
        "департамент",
        "комитет",
        "министерство",
        "управление",
        "мэрия",
        "префектура",
        "правительство",
        "казначейство",
        "инспекция",
        "контрольный",
        "архив",
    },
    "facilities": {
        "жилищник",
        "благоустройство",
        "водоканал",
        "теплосеть",
        "эксплуатация",
        "коммунальный",
        "дорожный",
        "автодор",
        "зеленхоз",
        "уборка",
        "хозяйственный",
        "ремонтный",
    },
    "security_it": {
        "информационный",
        "информатизация",
        "цифровой",
        "безопасность",
        "охрана",
        "пожарный",
        "спасательный",
        "связь",
    },
}

CUSTOMER_NAME_ARCHETYPE_STEMS = {
    archetype: {stem for keyword in keywords for stem in stem_tokens(tokenize(normalize_text(keyword))) if stem}
    for archetype, keywords in CUSTOMER_NAME_ARCHETYPE_KEYWORDS.items()
}


@dataclass
class SessionState:
    clicked_ste_ids: List[str]
    cart_ste_ids: List[str]
    recent_categories: List[str]

    @classmethod
    def from_mapping(cls, payload: Optional[Dict[str, object]]) -> "SessionState":
        payload = payload or {}
        clicked_ste_ids = [str(value) for value in payload.get("clicked_ste_ids", [])]
        cart_ste_ids = [str(value) for value in payload.get("cart_ste_ids", [])]
        recent_categories = [normalize_text(str(value)) for value in payload.get("recent_categories", [])]
        return cls(clicked_ste_ids=clicked_ste_ids, cart_ste_ids=cart_ste_ids, recent_categories=recent_categories)


class PersonalizationService:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_PREPROCESSED_DB,
        contracts_path: Path | str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.contracts_path = Path(contracts_path) if contracts_path else None
        self._archetype_category_ids_cache: Dict[str, List[int]] = {}
        self._customer_name_index_loaded = False
        self._customer_name_by_inn: Dict[str, str] = {}
        self._customer_name_archetype_by_inn: Dict[str, str] = {}
        self._customer_name_signal_stems_by_inn: Dict[str, set[str]] = {}
        self._customer_name_archetype_scores_by_inn: Dict[str, Dict[str, float]] = {}

    def close(self) -> None:
        self.conn.close()

    def build_customer_profile(
        self,
        customer_inn: str,
        customer_region: Optional[str] = None,
        top_categories: int = 12,
        top_ste: int = 20,
        top_region_categories: int = 12,
    ) -> Dict[str, object]:
        customer_inn = str(customer_inn)
        customer_region = customer_region or self._infer_customer_region(customer_inn)

        top_customer_categories = self.conn.execute(
            """
            SELECT
                cc.category_id,
                cl.category,
                cl.normalized_category,
                cc.purchase_count,
                cc.total_amount,
                cc.first_purchase_dt,
                cc.last_purchase_dt
            FROM customer_category_stats cc
            JOIN category_lookup cl ON cl.category_id = cc.category_id
            WHERE cc.customer_inn = ?
            ORDER BY cc.purchase_count DESC, cc.total_amount DESC
            LIMIT ?
            """,
            (customer_inn, top_categories),
        ).fetchall()

        top_customer_ste = self.conn.execute(
            """
            SELECT
                cs.ste_id,
                cs.category_id,
                cl.category,
                cl.normalized_category,
                cs.purchase_count,
                cs.total_amount,
                cs.first_purchase_dt,
                cs.last_purchase_dt
            FROM customer_ste_stats cs
            JOIN category_lookup cl ON cl.category_id = cs.category_id
            WHERE cs.customer_inn = ?
            ORDER BY cs.purchase_count DESC, cs.total_amount DESC
            LIMIT ?
            """,
            (customer_inn, top_ste),
        ).fetchall()

        category_preferences = self._weight_category_rows(top_customer_categories)
        ste_preferences = self._weight_ste_rows(top_customer_ste)
        region_preferences: List[Dict[str, object]] = []
        institution_archetype, institution_archetype_scores = self._infer_institution_archetype(category_preferences)
        same_type_peer_customer_inns = self._load_same_type_peer_customer_inns(
            customer_inn=customer_inn,
            customer_region=None,
            archetype=institution_archetype,
            limit=120,
        )
        archetype_category_preferences = self._weight_category_rows(
            self._load_peer_category_rows(same_type_peer_customer_inns, limit=max(top_categories, top_region_categories))
        )
        if archetype_category_preferences:
            archetype_category_ids = [
                int(item["category_id"])
                for item in archetype_category_preferences[: min(len(archetype_category_preferences), max(top_categories, 6))]
                if int(item.get("category_id") or 0) > 0
            ]
        else:
            archetype_category_rows = self._load_archetype_category_rows(
                archetype=institution_archetype,
                customer_region=None,
                limit=max(top_categories, top_region_categories),
            )
            archetype_category_preferences = self._weight_category_rows(archetype_category_rows)
            archetype_category_ids = [
                int(row["category_id"])
                for row in archetype_category_rows[: min(len(archetype_category_rows), max(top_categories, 6))]
            ]
        top_category_ids = [int(row["category_id"]) for row in top_customer_categories[: min(3, len(top_customer_categories))]]
        peer_customer_inns = self._load_peer_customer_inns(
            customer_inn=customer_inn,
            customer_region=None,
            category_ids=top_category_ids,
        )
        peer_category_preferences = self._weight_category_rows(
            self._load_peer_category_rows(peer_customer_inns, limit=max(top_categories, top_region_categories))
        )
        peer_ste_preferences = self._weight_ste_rows(
            self._load_peer_ste_rows(peer_customer_inns, limit=max(top_ste, 24))
        )
        regional_ste_preferences: List[Dict[str, object]] = []
        archetype_ste_preferences = self._weight_ste_rows(
            self._load_peer_ste_rows(same_type_peer_customer_inns, limit=max(top_ste, 24))
        )
        if not archetype_ste_preferences:
            archetype_ste_preferences = self._weight_ste_rows(
                self._load_archetype_ste_rows(
                    archetype=institution_archetype,
                    customer_region=None,
                    category_ids=archetype_category_ids,
                    limit=max(top_ste, 24),
                )
            )
        recommended_categories = self._merge_category_preferences(
            archetype=institution_archetype,
            institution=category_preferences,
            peers=peer_category_preferences,
            region=region_preferences,
            archetype_items=archetype_category_preferences,
            limit=max(top_categories, top_region_categories),
        )
        recommended_ste = self._merge_ste_preferences(
            archetype=institution_archetype,
            institution=ste_preferences,
            peers=peer_ste_preferences,
            region=regional_ste_preferences,
            archetype_items=archetype_ste_preferences,
            limit=max(top_ste, 24),
        )

        return {
            "customer_inn": customer_inn,
            "customer_region": customer_region,
            "institution_archetype": institution_archetype,
            "institution_archetype_label": INSTITUTION_ARCHETYPE_LABELS.get(
                institution_archetype,
                INSTITUTION_ARCHETYPE_LABELS["general"],
            ),
            "institution_archetype_scores": institution_archetype_scores,
            "top_categories": category_preferences,
            "top_ste": ste_preferences,
            "regional_categories": region_preferences,
            "peer_categories": peer_category_preferences,
            "peer_ste": peer_ste_preferences,
            "same_type_peer_inns": same_type_peer_customer_inns,
            "archetype_categories": archetype_category_preferences,
            "archetype_ste": archetype_ste_preferences,
            "recommended_categories": recommended_categories,
            "recommended_ste": recommended_ste,
            "category_affinity": {item["normalized_category"]: item["weight"] for item in category_preferences},
            "ste_affinity": {item["ste_id"]: item["weight"] for item in ste_preferences},
            "regional_affinity": {item["normalized_category"]: item["weight"] for item in region_preferences},
        }

    def _infer_customer_region(self, customer_inn: str) -> Optional[str]:
        try:
            row = self.conn.execute(
                """
                SELECT customer_region
                FROM customer_region_lookup
                WHERE customer_inn = ?
                LIMIT 1
                """,
                (customer_inn,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return row["customer_region"] if row else None

    @staticmethod
    def _clean_contract_lookup_text(value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).replace("\ufeff", " ").replace("\t", " ").replace("\n", " ").split())

    @staticmethod
    def _infer_csv_delimiter(path: Path) -> str:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                sample = handle.read(4096)
        except OSError:
            return ";"
        return ";" if sample.count(";") >= sample.count(",") else ","

    def _ensure_customer_name_index_loaded(self) -> None:
        if self._customer_name_index_loaded:
            return

        self._customer_name_index_loaded = True
        contracts_path = self.contracts_path
        if not contracts_path or not contracts_path.exists():
            return

        delimiter = self._infer_csv_delimiter(contracts_path)
        try:
            with contracts_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter, quotechar='"')
                for row in reader:
                    if len(row) < 7:
                        continue
                    customer_inn = self._clean_contract_lookup_text(row[5])
                    customer_name = self._clean_contract_lookup_text(row[6])
                    if (
                        not customer_inn
                        or not customer_name
                        or customer_inn.lower() == "customer_inn"
                        or normalize_text(customer_name) == "customer_name"
                    ):
                        continue

                    current_name = self._customer_name_by_inn.get(customer_inn, "")
                    if len(normalize_text(customer_name)) >= len(normalize_text(current_name)):
                        self._customer_name_by_inn[customer_inn] = customer_name
        except OSError:
            return

        for customer_inn, customer_name in self._customer_name_by_inn.items():
            archetype, signal_stems, scores = self._infer_customer_name_archetype(customer_name)
            self._customer_name_archetype_by_inn[customer_inn] = archetype
            self._customer_name_signal_stems_by_inn[customer_inn] = signal_stems
            self._customer_name_archetype_scores_by_inn[customer_inn] = scores

    @classmethod
    def _infer_customer_name_archetype(cls, customer_name: str) -> tuple[str, set[str], Dict[str, float]]:
        customer_name_stems = set(stem_tokens(tokenize(normalize_text(customer_name))))
        if not customer_name_stems:
            return "general", set(), {}

        scores = {archetype: 0.0 for archetype in CUSTOMER_NAME_ARCHETYPE_STEMS}
        overlap_by_archetype: Dict[str, set[str]] = {}
        for archetype, keyword_stems in CUSTOMER_NAME_ARCHETYPE_STEMS.items():
            overlap = customer_name_stems & keyword_stems
            if not overlap:
                continue
            overlap_by_archetype[archetype] = overlap
            scores[archetype] = float(len(overlap))

        rounded_scores = {key: round(value, 4) for key, value in scores.items() if value > 0}
        if not rounded_scores:
            return "general", set(), {}

        archetype, archetype_score = max(scores.items(), key=lambda item: item[1])
        if archetype_score < 1.0:
            return "general", set(), rounded_scores
        return archetype, overlap_by_archetype.get(archetype, set()), rounded_scores

    @classmethod
    def customer_name_archetype_match_score(cls, *, archetype: str, texts: Iterable[str]) -> float:
        keyword_stems = CUSTOMER_NAME_ARCHETYPE_STEMS.get(archetype, set())
        if not keyword_stems:
            return 0.0

        candidate_stems: set[str] = set()
        for text in texts:
            candidate_stems.update(stem_tokens(tokenize(normalize_text(text))))
        if not candidate_stems:
            return 0.0

        overlap = candidate_stems & keyword_stems
        if not overlap:
            return 0.0
        return round(min(1.0, len(overlap) / 2.0), 4)

    def get_customer_name_context(self, customer_inn: str, *, limit: int = 120) -> Dict[str, object]:
        customer_inn = str(customer_inn or "")
        self._ensure_customer_name_index_loaded()

        customer_name = self._customer_name_by_inn.get(customer_inn, "")
        archetype = self._customer_name_archetype_by_inn.get(customer_inn, "general")
        signal_stems = set(self._customer_name_signal_stems_by_inn.get(customer_inn, set()))
        scores = dict(self._customer_name_archetype_scores_by_inn.get(customer_inn, {}))

        peer_candidates: List[tuple[int, str]] = []
        if archetype != "general":
            for peer_inn, peer_archetype in self._customer_name_archetype_by_inn.items():
                if peer_inn == customer_inn or peer_archetype != archetype:
                    continue
                peer_signal_stems = set(self._customer_name_signal_stems_by_inn.get(peer_inn, set()))
                if signal_stems and peer_signal_stems and not (signal_stems & peer_signal_stems):
                    continue
                peer_score = len(signal_stems & peer_signal_stems) * 10 + len(peer_signal_stems)
                peer_candidates.append((peer_score, peer_inn))

        peer_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        same_type_peer_inns = [peer_inn for _, peer_inn in peer_candidates[:limit]]

        return {
            "customer_name": customer_name,
            "institution_name_archetype": archetype,
            "institution_name_archetype_label": INSTITUTION_ARCHETYPE_PROFILE_LABELS.get(
                archetype,
                INSTITUTION_ARCHETYPE_PROFILE_LABELS["general"],
            ),
            "institution_name_archetype_scores": scores,
            "name_signal_stems": sorted(signal_stems),
            "same_type_peer_inns": same_type_peer_inns,
        }

    def _load_archetype_category_ids(self, archetype: str) -> List[int]:
        if archetype in self._archetype_category_ids_cache:
            return list(self._archetype_category_ids_cache[archetype])
        keyword_stems = ARCHETYPE_KEYWORD_STEMS.get(archetype, set())
        if not keyword_stems:
            self._archetype_category_ids_cache[archetype] = []
            return []
        rows = self.conn.execute(
            """
            SELECT category_id, normalized_category
            FROM category_lookup
            """
        ).fetchall()
        category_ids = [
            int(row["category_id"])
            for row in rows
            if self._category_matches_archetype(str(row["normalized_category"] or ""), archetype)
        ]
        self._archetype_category_ids_cache[archetype] = category_ids
        return list(category_ids)

    def _category_matches_archetype(self, normalized_category: str, archetype: str) -> bool:
        keyword_stems = ARCHETYPE_KEYWORD_STEMS.get(archetype, set())
        if not normalized_category or not keyword_stems:
            return False
        category_stems = set(stem_tokens(tokenize(normalized_category)))
        return self._match_keyword_stems(category_stems, keyword_stems)

    def _load_archetype_category_rows(
        self,
        *,
        archetype: str,
        customer_region: Optional[str],
        limit: int,
    ) -> List[sqlite3.Row]:
        category_ids = self._load_archetype_category_ids(archetype)
        if not category_ids:
            return []
        placeholders = ", ".join("?" for _ in category_ids)
        if customer_region:
            rows = self.conn.execute(
                f"""
                SELECT
                    rc.category_id,
                    cl.category,
                    cl.normalized_category,
                    rc.purchase_count,
                    rc.total_amount,
                    rc.first_purchase_dt,
                    rc.last_purchase_dt
                FROM region_category_stats rc
                JOIN category_lookup cl ON cl.category_id = rc.category_id
                WHERE rc.customer_region = ?
                  AND rc.category_id IN ({placeholders})
                ORDER BY rc.purchase_count DESC, rc.total_amount DESC
                LIMIT ?
                """,
                [customer_region, *category_ids, limit],
            ).fetchall()
            if len(rows) >= limit:
                return rows[:limit]
            fallback_rows = self.conn.execute(
                f"""
                SELECT
                    cc.category_id,
                    cl.category,
                    cl.normalized_category,
                    SUM(cc.purchase_count) AS purchase_count,
                    SUM(cc.total_amount) AS total_amount,
                    MIN(cc.first_purchase_dt) AS first_purchase_dt,
                    MAX(cc.last_purchase_dt) AS last_purchase_dt
                FROM customer_category_stats cc
                JOIN category_lookup cl ON cl.category_id = cc.category_id
                WHERE cc.category_id IN ({placeholders})
                GROUP BY cc.category_id, cl.category, cl.normalized_category
                ORDER BY purchase_count DESC, total_amount DESC
                LIMIT ?
                """,
                [*category_ids, limit],
            ).fetchall()
            merged_rows = {int(row["category_id"]): row for row in rows}
            for row in fallback_rows:
                merged_rows.setdefault(int(row["category_id"]), row)
            return list(merged_rows.values())[:limit]
        return self.conn.execute(
            f"""
            SELECT
                cc.category_id,
                cl.category,
                cl.normalized_category,
                SUM(cc.purchase_count) AS purchase_count,
                SUM(cc.total_amount) AS total_amount,
                MIN(cc.first_purchase_dt) AS first_purchase_dt,
                MAX(cc.last_purchase_dt) AS last_purchase_dt
            FROM customer_category_stats cc
            JOIN category_lookup cl ON cl.category_id = cc.category_id
            WHERE cc.category_id IN ({placeholders})
            GROUP BY cc.category_id, cl.category, cl.normalized_category
            ORDER BY purchase_count DESC, total_amount DESC
            LIMIT ?
            """,
            [*category_ids, limit],
        ).fetchall()

    def _load_archetype_ste_rows(
        self,
        *,
        archetype: str,
        customer_region: Optional[str],
        category_ids: List[int],
        limit: int,
    ) -> List[sqlite3.Row]:
        if archetype == "general":
            return []
        effective_category_ids = category_ids or self._load_archetype_category_ids(archetype)
        if not effective_category_ids:
            return []
        placeholders = ", ".join("?" for _ in effective_category_ids)
        if customer_region:
            rows = self.conn.execute(
                f"""
                SELECT
                    cs.ste_id,
                    cs.category_id,
                    cl.category,
                    cl.normalized_category,
                    SUM(cs.purchase_count) AS purchase_count,
                    SUM(cs.total_amount) AS total_amount,
                    MIN(cs.first_purchase_dt) AS first_purchase_dt,
                    MAX(cs.last_purchase_dt) AS last_purchase_dt
                FROM customer_ste_stats cs
                JOIN customer_region_lookup cr ON cr.customer_inn = cs.customer_inn
                JOIN category_lookup cl ON cl.category_id = cs.category_id
                WHERE cr.customer_region = ?
                  AND cs.category_id IN ({placeholders})
                GROUP BY cs.ste_id, cs.category_id, cl.category, cl.normalized_category
                ORDER BY purchase_count DESC, total_amount DESC
                LIMIT ?
                """,
                [customer_region, *effective_category_ids, limit],
            ).fetchall()
            if len(rows) >= limit:
                return rows[:limit]
            fallback_rows = self.conn.execute(
                f"""
                SELECT
                    cs.ste_id,
                    cs.category_id,
                    cl.category,
                    cl.normalized_category,
                    SUM(cs.purchase_count) AS purchase_count,
                    SUM(cs.total_amount) AS total_amount,
                    MIN(cs.first_purchase_dt) AS first_purchase_dt,
                    MAX(cs.last_purchase_dt) AS last_purchase_dt
                FROM customer_ste_stats cs
                JOIN category_lookup cl ON cl.category_id = cs.category_id
                WHERE cs.category_id IN ({placeholders})
                GROUP BY cs.ste_id, cs.category_id, cl.category, cl.normalized_category
                ORDER BY purchase_count DESC, total_amount DESC
                LIMIT ?
                """,
                [*effective_category_ids, limit],
            ).fetchall()
            merged_rows = {str(row["ste_id"]): row for row in rows}
            for row in fallback_rows:
                merged_rows.setdefault(str(row["ste_id"]), row)
            return list(merged_rows.values())[:limit]
        return self.conn.execute(
            f"""
            SELECT
                cs.ste_id,
                cs.category_id,
                cl.category,
                cl.normalized_category,
                SUM(cs.purchase_count) AS purchase_count,
                SUM(cs.total_amount) AS total_amount,
                MIN(cs.first_purchase_dt) AS first_purchase_dt,
                MAX(cs.last_purchase_dt) AS last_purchase_dt
            FROM customer_ste_stats cs
            JOIN category_lookup cl ON cl.category_id = cs.category_id
            WHERE cs.category_id IN ({placeholders})
            GROUP BY cs.ste_id, cs.category_id, cl.category, cl.normalized_category
            ORDER BY purchase_count DESC, total_amount DESC
            LIMIT ?
            """,
            [*effective_category_ids, limit],
        ).fetchall()

    @staticmethod
    def _blend_weights(archetype: str, *, kind: str) -> Dict[str, float]:
        if kind == "category":
            return ARCHETYPE_CATEGORY_BLEND_WEIGHTS.get(archetype, ARCHETYPE_CATEGORY_BLEND_WEIGHTS["general"])
        return ARCHETYPE_STE_BLEND_WEIGHTS.get(archetype, ARCHETYPE_STE_BLEND_WEIGHTS["general"])

    @staticmethod
    def _match_keyword_stems(category_stems: set[str], keyword_stems: set[str]) -> bool:
        if not category_stems or not keyword_stems:
            return False
        for category_stem in category_stems:
            for keyword_stem in keyword_stems:
                if category_stem.startswith(keyword_stem) or keyword_stem.startswith(category_stem):
                    return True
        return False

    def _infer_institution_archetype(self, category_preferences: List[Dict[str, object]]) -> tuple[str, Dict[str, float]]:
        if not category_preferences:
            return "general", {}

        scores = {archetype: 0.0 for archetype in ARCHETYPE_KEYWORD_STEMS}
        for item in category_preferences[:8]:
            normalized_category = str(item.get("normalized_category") or item.get("category") or "")
            category_stems = set(stem_tokens(tokenize(normalized_category)))
            if not category_stems:
                continue
            signal_strength = float(item.get("weight", 0.0) or 0.0) + min(float(item.get("purchase_count", 0) or 0.0) / 40.0, 2.0)
            for archetype, keyword_stems in ARCHETYPE_KEYWORD_STEMS.items():
                if self._match_keyword_stems(category_stems, keyword_stems):
                    scores[archetype] += signal_strength

        rounded_scores = {key: round(value, 4) for key, value in scores.items() if value > 0}
        if not rounded_scores:
            return "general", {}

        archetype, archetype_score = max(scores.items(), key=lambda item: item[1])
        if archetype_score < 0.75:
            return "general", rounded_scores
        return archetype, rounded_scores

    def _load_peer_customer_inns(
        self,
        *,
        customer_inn: str,
        customer_region: Optional[str],
        category_ids: List[int],
        limit: int = 120,
    ) -> List[str]:
        if not customer_region or not category_ids:
            return []
        placeholders = ", ".join("?" for _ in category_ids)
        rows = self.conn.execute(
            f"""
            SELECT
                cc.customer_inn,
                COUNT(DISTINCT cc.category_id) AS overlap_category_count,
                SUM(cc.purchase_count) AS overlap_purchase_count
            FROM customer_category_stats cc
            JOIN customer_region_lookup cr ON cr.customer_inn = cc.customer_inn
            WHERE cc.customer_inn <> ?
              AND cr.customer_region = ?
              AND cc.category_id IN ({placeholders})
            GROUP BY cc.customer_inn
            ORDER BY overlap_category_count DESC, overlap_purchase_count DESC, cc.customer_inn ASC
            LIMIT ?
            """,
            [customer_inn, customer_region, *category_ids, limit],
        ).fetchall()
        return [str(row["customer_inn"]) for row in rows if row["customer_inn"]]

    def _load_same_type_peer_customer_inns(
        self,
        *,
        customer_inn: str,
        customer_region: Optional[str],
        archetype: str,
        limit: int = 120,
    ) -> List[str]:
        category_ids = self._load_archetype_category_ids(archetype)
        if not category_ids:
            return []
        placeholders = ", ".join("?" for _ in category_ids)
        result: List[str] = []
        seen: set[str] = set()

        if customer_region:
            regional_rows = self.conn.execute(
                f"""
                SELECT
                    cc.customer_inn,
                    COUNT(DISTINCT cc.category_id) AS overlap_category_count,
                    SUM(cc.purchase_count) AS overlap_purchase_count
                FROM customer_category_stats cc
                JOIN customer_region_lookup cr ON cr.customer_inn = cc.customer_inn
                WHERE cc.customer_inn <> ?
                  AND cr.customer_region = ?
                  AND cc.category_id IN ({placeholders})
                GROUP BY cc.customer_inn
                ORDER BY overlap_category_count DESC, overlap_purchase_count DESC, cc.customer_inn ASC
                LIMIT ?
                """,
                [customer_inn, customer_region, *category_ids, limit],
            ).fetchall()
            for row in regional_rows:
                peer_inn = str(row["customer_inn"] or "")
                if peer_inn and peer_inn not in seen:
                    seen.add(peer_inn)
                    result.append(peer_inn)
            if len(result) >= limit:
                return result[:limit]

        global_rows = self.conn.execute(
            f"""
            SELECT
                cc.customer_inn,
                COUNT(DISTINCT cc.category_id) AS overlap_category_count,
                SUM(cc.purchase_count) AS overlap_purchase_count
            FROM customer_category_stats cc
            WHERE cc.customer_inn <> ?
              AND cc.category_id IN ({placeholders})
            GROUP BY cc.customer_inn
            ORDER BY overlap_category_count DESC, overlap_purchase_count DESC, cc.customer_inn ASC
            LIMIT ?
            """,
            [customer_inn, *category_ids, limit],
        ).fetchall()
        for row in global_rows:
            peer_inn = str(row["customer_inn"] or "")
            if peer_inn and peer_inn not in seen:
                seen.add(peer_inn)
                result.append(peer_inn)
                if len(result) >= limit:
                    break
        return result

    def _load_peer_category_rows(self, peer_customer_inns: List[str], *, limit: int) -> List[sqlite3.Row]:
        if not peer_customer_inns:
            return []
        placeholders = ", ".join("?" for _ in peer_customer_inns)
        return self.conn.execute(
            f"""
            SELECT
                cc.category_id,
                cl.category,
                cl.normalized_category,
                SUM(cc.purchase_count) AS purchase_count,
                SUM(cc.total_amount) AS total_amount,
                MIN(cc.first_purchase_dt) AS first_purchase_dt,
                MAX(cc.last_purchase_dt) AS last_purchase_dt
            FROM customer_category_stats cc
            JOIN category_lookup cl ON cl.category_id = cc.category_id
            WHERE cc.customer_inn IN ({placeholders})
            GROUP BY cc.category_id, cl.category, cl.normalized_category
            ORDER BY purchase_count DESC, total_amount DESC
            LIMIT ?
            """,
            [*peer_customer_inns, limit],
        ).fetchall()

    def _load_peer_ste_rows(self, peer_customer_inns: List[str], *, limit: int) -> List[sqlite3.Row]:
        if not peer_customer_inns:
            return []
        placeholders = ", ".join("?" for _ in peer_customer_inns)
        return self.conn.execute(
            f"""
            SELECT
                cs.ste_id,
                cs.category_id,
                cl.category,
                cl.normalized_category,
                SUM(cs.purchase_count) AS purchase_count,
                SUM(cs.total_amount) AS total_amount,
                MIN(cs.first_purchase_dt) AS first_purchase_dt,
                MAX(cs.last_purchase_dt) AS last_purchase_dt
            FROM customer_ste_stats cs
            JOIN category_lookup cl ON cl.category_id = cs.category_id
            WHERE cs.customer_inn IN ({placeholders})
            GROUP BY cs.ste_id, cs.category_id, cl.category, cl.normalized_category
            ORDER BY purchase_count DESC, total_amount DESC
            LIMIT ?
            """,
            [*peer_customer_inns, limit],
        ).fetchall()

    def _load_regional_ste_rows(
        self,
        *,
        customer_region: Optional[str],
        category_ids: List[int],
        limit: int,
    ) -> List[sqlite3.Row]:
        if not customer_region:
            return []
        params: List[object] = [customer_region]
        category_filter = ""
        if category_ids:
            placeholders = ", ".join("?" for _ in category_ids)
            category_filter = f" AND cs.category_id IN ({placeholders})"
            params.extend(category_ids)
        params.append(limit)
        return self.conn.execute(
            f"""
            SELECT
                cs.ste_id,
                cs.category_id,
                cl.category,
                cl.normalized_category,
                SUM(cs.purchase_count) AS purchase_count,
                SUM(cs.total_amount) AS total_amount,
                MIN(cs.first_purchase_dt) AS first_purchase_dt,
                MAX(cs.last_purchase_dt) AS last_purchase_dt
            FROM customer_ste_stats cs
            JOIN customer_region_lookup cr ON cr.customer_inn = cs.customer_inn
            JOIN category_lookup cl ON cl.category_id = cs.category_id
            WHERE cr.customer_region = ?
            {category_filter}
            GROUP BY cs.ste_id, cs.category_id, cl.category, cl.normalized_category
            ORDER BY purchase_count DESC, total_amount DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    @staticmethod
    def _category_reason(
        archetype: str,
        institution_weight: float,
        peer_weight: float,
        region_weight: float,
        archetype_weight: float,
    ) -> str:
        archetype_label = INSTITUTION_ARCHETYPE_LABELS.get(archetype, INSTITUTION_ARCHETYPE_LABELS["general"])
        if institution_weight > 0 and (peer_weight > 0 or archetype_weight > 0):
            return "Часто закупалось учреждением и поддержано закупками учреждений того же типа"
        if institution_weight > 0:
            return "Часто закупалось учреждением"
        if peer_weight > 0:
            return f"Популярно у похожих {archetype_label}"
        if archetype_weight > 0:
            return f"Популярно у учреждений того же типа ({archetype_label})"
        return "Популярно у похожих учреждений"

    @classmethod
    def _merge_category_preferences(
        cls,
        *,
        archetype: str,
        institution: List[Dict[str, object]],
        peers: List[Dict[str, object]],
        region: List[Dict[str, object]],
        archetype_items: List[Dict[str, object]],
        limit: int,
    ) -> List[Dict[str, object]]:
        merged: Dict[str, Dict[str, object]] = {}

        def upsert(items: List[Dict[str, object]], source: str) -> None:
            for item in items:
                normalized_category = str(item.get("normalized_category") or "")
                if not normalized_category:
                    continue
                payload = merged.setdefault(
                    normalized_category,
                    {
                        "category": str(item.get("category") or ""),
                        "normalized_category": normalized_category,
                        "purchase_count": 0,
                        "total_amount": 0.0,
                        "institution_weight": 0.0,
                        "peer_weight": 0.0,
                        "region_weight": 0.0,
                        "archetype_weight": 0.0,
                    },
                )
                payload["category"] = payload.get("category") or str(item.get("category") or "")
                payload["purchase_count"] = max(int(payload.get("purchase_count", 0) or 0), int(item.get("purchase_count") or 0))
                payload["total_amount"] = max(float(payload.get("total_amount", 0.0) or 0.0), float(item.get("total_amount") or 0.0))
                payload[f"{source}_weight"] = max(float(payload.get(f"{source}_weight", 0.0) or 0.0), float(item.get("weight") or 0.0))

        upsert(institution, "institution")
        upsert(peers, "peer")
        upsert(region, "region")
        upsert(archetype_items, "archetype")

        ranked: List[Dict[str, object]] = []
        weights = cls._blend_weights(archetype, kind="category")
        for payload in merged.values():
            institution_weight = float(payload.get("institution_weight", 0.0) or 0.0)
            peer_weight = float(payload.get("peer_weight", 0.0) or 0.0)
            region_weight = float(payload.get("region_weight", 0.0) or 0.0)
            archetype_weight = float(payload.get("archetype_weight", 0.0) or 0.0)
            recommendation_score = (
                float(weights["institution"]) * institution_weight
                + float(weights["peer"]) * peer_weight
                + float(weights["region"]) * region_weight
                + float(weights["archetype"]) * archetype_weight
                + float(weights["diversity"])
                * sum(1 for value in [institution_weight, peer_weight, archetype_weight] if value > 0)
            )
            payload["recommendation_score"] = round(recommendation_score, 4)
            payload["reason"] = cls._category_reason(
                archetype,
                institution_weight,
                peer_weight,
                region_weight,
                archetype_weight,
            )
            ranked.append(payload)

        ranked.sort(
            key=lambda item: (
                float(item.get("recommendation_score", 0.0)),
                float(item.get("institution_weight", 0.0)),
                float(item.get("peer_weight", 0.0)),
                float(item.get("region_weight", 0.0)),
                float(item.get("archetype_weight", 0.0)),
                int(item.get("purchase_count", 0)),
            ),
            reverse=True,
        )
        return ranked[:limit]

    @staticmethod
    def _ste_reason(
        archetype: str,
        institution_weight: float,
        peer_weight: float,
        region_weight: float,
        archetype_weight: float,
    ) -> str:
        archetype_label = INSTITUTION_ARCHETYPE_LABELS.get(archetype, INSTITUTION_ARCHETYPE_LABELS["general"])
        if institution_weight > 0 and (peer_weight > 0 or archetype_weight > 0):
            return "Часто закупалось учреждением и поддержано закупками учреждений того же типа"
        if institution_weight > 0:
            return "Часто закупалось учреждением"
        if peer_weight > 0:
            return f"Популярно у похожих {archetype_label}"
        if archetype_weight > 0:
            return f"Популярно у учреждений того же типа ({archetype_label})"
        return "Популярно у похожих учреждений"

    @classmethod
    def _merge_ste_preferences(
        cls,
        *,
        archetype: str,
        institution: List[Dict[str, object]],
        peers: List[Dict[str, object]],
        region: List[Dict[str, object]],
        archetype_items: List[Dict[str, object]],
        limit: int,
    ) -> List[Dict[str, object]]:
        merged: Dict[str, Dict[str, object]] = {}

        def upsert(items: List[Dict[str, object]], source: str) -> None:
            for item in items:
                ste_id = str(item.get("ste_id") or "")
                if not ste_id:
                    continue
                payload = merged.setdefault(
                    ste_id,
                    {
                        "ste_id": ste_id,
                        "category": str(item.get("category") or ""),
                        "normalized_category": str(item.get("normalized_category") or ""),
                        "purchase_count": 0,
                        "total_amount": 0.0,
                        "institution_weight": 0.0,
                        "peer_weight": 0.0,
                        "region_weight": 0.0,
                        "archetype_weight": 0.0,
                    },
                )
                payload["category"] = payload.get("category") or str(item.get("category") or "")
                payload["normalized_category"] = payload.get("normalized_category") or str(item.get("normalized_category") or "")
                payload["purchase_count"] = max(int(payload.get("purchase_count", 0) or 0), int(item.get("purchase_count") or 0))
                payload["total_amount"] = max(float(payload.get("total_amount", 0.0) or 0.0), float(item.get("total_amount") or 0.0))
                payload[f"{source}_weight"] = max(float(payload.get(f"{source}_weight", 0.0) or 0.0), float(item.get("weight") or 0.0))

        upsert(institution, "institution")
        upsert(peers, "peer")
        upsert(region, "region")
        upsert(archetype_items, "archetype")

        ranked: List[Dict[str, object]] = []
        weights = cls._blend_weights(archetype, kind="ste")
        for payload in merged.values():
            institution_weight = float(payload.get("institution_weight", 0.0) or 0.0)
            peer_weight = float(payload.get("peer_weight", 0.0) or 0.0)
            region_weight = float(payload.get("region_weight", 0.0) or 0.0)
            archetype_weight = float(payload.get("archetype_weight", 0.0) or 0.0)
            recommendation_score = (
                float(weights["institution"]) * institution_weight
                + float(weights["peer"]) * peer_weight
                + float(weights["region"]) * region_weight
                + float(weights["archetype"]) * archetype_weight
                + float(weights["diversity"])
                * sum(1 for value in [institution_weight, peer_weight, archetype_weight] if value > 0)
            )
            payload["weight"] = round(max(institution_weight, peer_weight, region_weight, archetype_weight), 4)
            payload["recommendation_score"] = round(recommendation_score, 4)
            payload["reason"] = cls._ste_reason(
                archetype,
                institution_weight,
                peer_weight,
                region_weight,
                archetype_weight,
            )
            ranked.append(payload)

        ranked.sort(
            key=lambda item: (
                float(item.get("recommendation_score", 0.0)),
                float(item.get("institution_weight", 0.0)),
                float(item.get("peer_weight", 0.0)),
                float(item.get("region_weight", 0.0)),
                float(item.get("archetype_weight", 0.0)),
                int(item.get("purchase_count", 0)),
            ),
            reverse=True,
        )
        return ranked[:limit]

    def _weight_category_rows(self, rows: Iterable[sqlite3.Row]) -> List[Dict[str, object]]:
        rows = list(rows)
        if not rows:
            return []
        max_count = max(row["purchase_count"] for row in rows) or 1
        max_amount_log = max(math.log1p(float(row["total_amount"])) for row in rows) or 1.0
        result = []
        for rank, row in enumerate(rows, start=1):
            count_component = float(row["purchase_count"]) / max_count
            amount_component = math.log1p(float(row["total_amount"])) / max_amount_log if max_amount_log else 0.0
            rank_component = 1.0 / rank
            weight = 0.55 * count_component + 0.25 * amount_component + 0.20 * rank_component
            result.append(
                {
                    "category": row["category"],
                    "normalized_category": str(row["normalized_category"] or normalize_text(row["category"])),
                    "category_id": int(row["category_id"] or 0) if "category_id" in row.keys() else 0,
                    "purchase_count": int(row["purchase_count"]),
                    "total_amount": round(float(row["total_amount"]), 2),
                    "first_purchase_dt": row["first_purchase_dt"],
                    "last_purchase_dt": row["last_purchase_dt"],
                    "weight": round(weight, 4),
                }
            )
        return result

    def _weight_ste_rows(self, rows: Iterable[sqlite3.Row]) -> List[Dict[str, object]]:
        rows = list(rows)
        if not rows:
            return []
        max_count = max(row["purchase_count"] for row in rows) or 1
        max_amount_log = max(math.log1p(float(row["total_amount"])) for row in rows) or 1.0
        result = []
        for rank, row in enumerate(rows, start=1):
            count_component = float(row["purchase_count"]) / max_count
            amount_component = math.log1p(float(row["total_amount"])) / max_amount_log if max_amount_log else 0.0
            rank_component = 1.0 / rank
            weight = 0.60 * count_component + 0.25 * amount_component + 0.15 * rank_component
            result.append(
                {
                    "ste_id": row["ste_id"],
                    "category": row["category"],
                    "normalized_category": str(row["normalized_category"] or normalize_text(row["category"])),
                    "category_id": int(row["category_id"] or 0) if "category_id" in row.keys() else 0,
                    "purchase_count": int(row["purchase_count"]),
                    "total_amount": round(float(row["total_amount"]), 2),
                    "first_purchase_dt": row["first_purchase_dt"],
                    "last_purchase_dt": row["last_purchase_dt"],
                    "weight": round(weight, 4),
                }
            )
        return result

    def rerank_ste(
        self,
        results: List[Dict[str, object]],
        customer_profile: Dict[str, object],
        session_state: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, object]]:
        session = SessionState.from_mapping(session_state)
        category_affinity = customer_profile.get("category_affinity", {})
        ste_affinity = customer_profile.get("ste_affinity", {})
        reranked: List[Dict[str, object]] = []
        for index, result in enumerate(results, start=1):
            category_norm = normalize_text(str(result.get("category", "")))
            ste_id = str(result.get("ste_id"))
            base_score = float(result.get("search_score", 0.0))

            history_affinity = float(ste_affinity.get(ste_id, 0.0))
            category_score = self._best_category_affinity(category_norm, customer_profile.get("top_categories", []))
            region_score = 0.0
            session_boost = self._session_boost(result, session)

            final_score = (
                base_score
                + 5.0 * history_affinity
                + 3.0 * category_score
                + 4.0 * session_boost
            )

            explanation = self._build_explanation(
                result=result,
                history_affinity=history_affinity,
                category_affinity=category_score,
                region_affinity=region_score,
                session_boost=session_boost,
            )

            enriched = dict(result)
            enriched["base_search_rank"] = index
            enriched["personalization_features"] = {
                "history_affinity": round(history_affinity, 4),
                "category_affinity": round(category_score, 4),
                "region_affinity": round(region_score, 4),
                "session_action_boost": round(session_boost, 4),
            }
            enriched["final_score"] = round(final_score, 4)
            enriched["explanation"] = explanation
            reranked.append(enriched)

        reranked.sort(
            key=lambda item: (
                item["final_score"],
                item["personalization_features"]["history_affinity"],
                item["personalization_features"]["category_affinity"],
                item["search_score"],
            ),
            reverse=True,
        )
        return reranked

    def rerank_offers(
        self,
        offers: List[Dict[str, object]],
        customer_profile: Dict[str, object],
        session_state: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, object]]:
        session = SessionState.from_mapping(session_state)
        max_price = max((float(offer.get("unit_price", 0.0) or 0.0) for offer in offers), default=0.0)
        reranked: List[Dict[str, object]] = []
        for index, offer in enumerate(offers, start=1):
            ste_id = str(offer.get("ste_id", ""))
            category_norm = normalize_text(str(offer.get("category", "")))
            base_score = float(offer.get("offer_score", offer.get("search_score", 0.0)))

            history_affinity = float(customer_profile.get("ste_affinity", {}).get(ste_id, 0.0))
            category_affinity = self._best_category_affinity(category_norm, customer_profile.get("top_categories", []))
            region_affinity = 0.0
            session_boost = self._session_boost({"ste_id": ste_id, "category": category_norm}, session)

            region_match_boost = 0.0

            unit_price = float(offer.get("unit_price", 0.0) or 0.0)
            price_bonus = 0.0
            if max_price > 0 and unit_price > 0:
                price_bonus = max(0.0, 1.0 - (unit_price / max_price))

            final_score = (
                base_score
                + 4.0 * history_affinity
                + 3.0 * category_affinity
                + 3.0 * session_boost
                + 1.5 * price_bonus
            )

            explanation = []
            if history_affinity >= 0.20:
                explanation.append("СТЕ уже часто закупалось этой организацией")
            if category_affinity >= 0.20:
                explanation.append("оффер относится к предпочитаемой категории")
            if price_bonus >= 0.20:
                explanation.append("цена выгоднее части альтернатив")
            if session_boost >= 0.35:
                explanation.append("поднято после действий пользователя в текущей сессии")
            if not explanation:
                explanation.append("оставлено выше за счёт базовой релевантности оферты")

            enriched = dict(offer)
            enriched["base_offer_rank"] = index
            enriched["offer_personalization_features"] = {
                "history_affinity": round(history_affinity, 4),
                "category_affinity": round(category_affinity, 4),
                "region_affinity": round(region_affinity, 4),
                "session_action_boost": round(session_boost, 4),
                "region_match_boost": round(region_match_boost, 4),
                "price_bonus": round(price_bonus, 4),
            }
            enriched["final_offer_score"] = round(final_score, 4)
            enriched["offer_explanation"] = explanation
            reranked.append(enriched)

        reranked.sort(
            key=lambda item: (
                item["final_offer_score"],
                item["offer_personalization_features"]["history_affinity"],
                item["offer_personalization_features"]["category_affinity"],
                item["offer_personalization_features"]["price_bonus"],
            ),
            reverse=True,
        )
        return reranked

    def _session_boost(self, result: Dict[str, object], session: SessionState) -> float:
        category_norm = normalize_text(str(result.get("category", "")))
        ste_id = str(result.get("ste_id"))
        boost = 0.0
        if ste_id in session.clicked_ste_ids:
            boost += 0.45
        if ste_id in session.cart_ste_ids:
            boost += 0.75
        if category_norm and category_norm in session.recent_categories:
            boost += 0.35
        return min(boost, 1.5)

    def _build_explanation(
        self,
        result: Dict[str, object],
        history_affinity: float,
        category_affinity: float,
        region_affinity: float,
        session_boost: float,
    ) -> List[str]:
        explanation: List[str] = []
        if history_affinity >= 0.25:
            explanation.append("часто закупалось этой организацией")
        if category_affinity >= 0.20:
            explanation.append("похоже на ранее выбранные категории")
        if session_boost >= 0.35:
            explanation.append("поднято после клика или добавления в корзину")
        if not explanation:
            explanation.append("оставлено выше за счёт базовой текстовой релевантности")
        return explanation

    def _best_category_affinity(self, result_category: str, profile_categories: List[Dict[str, object]]) -> float:
        result_stems = set(stem_tokens(tokenize(result_category)))
        if not result_stems:
            return 0.0
        best = 0.0
        for item in profile_categories:
            profile_stems = set(stem_tokens(tokenize(str(item.get("normalized_category", "")))))
            if not profile_stems:
                continue
            overlap = len(result_stems & profile_stems) / max(1, min(len(result_stems), len(profile_stems)))
            best = max(best, float(item.get("weight", 0.0)) * overlap)
        return round(best, 4)


def build_customer_profile(
    customer_inn: str,
    customer_region: Optional[str] = None,
    db_path: Path | str = DEFAULT_PREPROCESSED_DB,
) -> Dict[str, object]:
    service = PersonalizationService(db_path=db_path)
    try:
        return service.build_customer_profile(customer_inn=customer_inn, customer_region=customer_region)
    finally:
        service.close()


def rerank_ste(
    results: List[Dict[str, object]],
    customer_profile: Dict[str, object],
    session_state: Optional[Dict[str, object]] = None,
    db_path: Path | str = DEFAULT_PREPROCESSED_DB,
) -> List[Dict[str, object]]:
    service = PersonalizationService(db_path=db_path)
    try:
        return service.rerank_ste(results=results, customer_profile=customer_profile, session_state=session_state)
    finally:
        service.close()


def rerank_offers(
    offers: List[Dict[str, object]],
    customer_profile: Dict[str, object],
    session_state: Optional[Dict[str, object]] = None,
    db_path: Path | str = DEFAULT_PREPROCESSED_DB,
) -> List[Dict[str, object]]:
    service = PersonalizationService(db_path=db_path)
    try:
        return service.rerank_offers(offers=offers, customer_profile=customer_profile, session_state=session_state)
    finally:
        service.close()
