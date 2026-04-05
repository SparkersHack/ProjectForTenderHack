from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .text import normalize_text, normalize_tokens, stem_token, tokenize, unique_preserve_order


DEFAULT_FASTTEXT_MODEL_PATH = Path("data/processed/tenderhack_fasttext.bin")

try:
    import fasttext as fasttext_module
except ImportError:
    fasttext_module = None


ALPHA_TOKEN_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]+$")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")
VOWEL_RE = re.compile(r"[аеёиоуыэюяaeiou]")
ADJECTIVE_LIKE_ENDINGS = (
    "ический",
    "ическая",
    "ическое",
    "ические",
    "ический",
    "овый",
    "овая",
    "овое",
    "овые",
    "ий",
    "ый",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
)


def _load_fasttext_model_silently(model_path: Path) -> object | None:
    if fasttext_module is None:
        return None
    fasttext_submodule = getattr(fasttext_module, "FastText", None)
    fasttext_class = getattr(fasttext_submodule, "_FastText", None)
    if fasttext_class is not None:
        return fasttext_class(model_path=str(model_path))
    return fasttext_module.load_model(str(model_path))


def char_ngrams(value: str, min_n: int = 3, max_n: int = 5) -> List[str]:
    padded = f"<{value}>"
    ngrams: List[str] = []
    for size in range(min_n, max_n + 1):
        if len(padded) < size:
            continue
        for index in range(len(padded) - size + 1):
            ngrams.append(padded[index : index + size])
    return ngrams


def ngram_jaccard(left: str, right: str) -> float:
    left_ngrams = set(char_ngrams(left))
    right_ngrams = set(char_ngrams(right))
    if not left_ngrams or not right_ngrams:
        return 0.0
    intersection = len(left_ngrams & right_ngrams)
    union = len(left_ngrams | right_ngrams)
    if union == 0:
        return 0.0
    return intersection / union


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_tokens(left.split()))
    right_tokens = set(normalize_tokens(right.split()))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    if union == 0:
        return 0.0
    return intersection / union


def cosine_similarity(left_vector: Iterable[float], right_vector: Iterable[float]) -> float:
    dot_product = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left_vector, right_vector):
        dot_product += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (math.sqrt(left_norm) * math.sqrt(right_norm))


class SqliteSemanticBackend:
    def __init__(self, conn: sqlite3.Connection, top_n: int = 4) -> None:
        self.conn = conn
        self.top_n = top_n
        self._cache: Dict[str, List[sqlite3.Row]] = {}
        self._edge_cache: Dict[Tuple[str, str], float] = {}
        self._frequency_cache: Dict[str, int] = {}
        self._sentence_cache: Dict[Tuple[str, str], float] = {}
        self._pair_similarity_cache: Dict[Tuple[str, str], float] = {}
        self._row_count = self._load_row_count()
        self.enabled = self._has_assets()

    def _load_row_count(self) -> int:
        metadata_row = self.conn.execute(
            "SELECT value FROM search_metadata WHERE key = 'deduped_rows' LIMIT 1"
        ).fetchone()
        if metadata_row is not None:
            try:
                return max(1, int(metadata_row[0]))
            except (TypeError, ValueError):
                pass
        table_row = self.conn.execute("SELECT COUNT(*) FROM ste_catalog").fetchone()
        return max(1, int(table_row[0] if table_row is not None else 1))

    def _has_assets(self) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'semantic_neighbors'
            LIMIT 1
            """
        ).fetchone()
        return row is not None

    @staticmethod
    def _looks_like_abbreviation(token: str) -> bool:
        if len(token) > 5 or not token.isalpha():
            return False
        vowel_count = len(VOWEL_RE.findall(token))
        return vowel_count <= 1

    @staticmethod
    def _is_adjective_like(token: str) -> bool:
        return any(token.endswith(ending) for ending in ADJECTIVE_LIKE_ENDINGS)

    def _token_frequency(self, token: str) -> int:
        cached = self._frequency_cache.get(token)
        if cached is not None:
            return cached
        row = self.conn.execute(
            "SELECT frequency FROM token_frequency WHERE token = ? LIMIT 1",
            (token,),
        ).fetchone()
        frequency = int(row[0]) if row is not None else 0
        self._frequency_cache[token] = frequency
        return frequency

    def _idf(self, token: str) -> float:
        frequency = self._token_frequency(token)
        return 1.0 + math.log1p(self._row_count / max(1, frequency))

    def _edge_score(self, token: str, neighbor: str) -> float:
        key = (token, neighbor)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached
        row = self.conn.execute(
            """
            SELECT score
            FROM semantic_neighbors
            WHERE token = ? AND neighbor = ?
            LIMIT 1
            """,
            (token, neighbor),
        ).fetchone()
        score = float(row[0]) if row is not None else 0.0
        self._edge_cache[key] = score
        return score

    def _source_is_informative(self, token: str) -> bool:
        frequency = self._token_frequency(token)
        if frequency <= 0:
            return False
        if self._looks_like_abbreviation(token):
            return True
        if frequency >= 15_000:
            return False
        if frequency >= 2_000 and self._is_adjective_like(token):
            return False
        return True

    def _source_priority(self, token: str) -> float:
        priority = self._idf(token)
        if self._looks_like_abbreviation(token):
            priority += 1.5
        if self._is_adjective_like(token):
            priority -= 0.75
        if len(token) <= 4 and not token.isdigit():
            priority += 0.15
        return priority

    def _select_source_tokens(self, tokens: List[str]) -> List[str]:
        informative_tokens = [token for token in tokens if self._source_is_informative(token)]
        if len(informative_tokens) <= 1:
            return informative_tokens
        abbreviation_tokens = [token for token in informative_tokens if self._looks_like_abbreviation(token)]
        if abbreviation_tokens:
            informative_tokens = abbreviation_tokens

        ranked_tokens = sorted(
            informative_tokens,
            key=lambda token: (self._source_priority(token), self._idf(token), -self._token_frequency(token), token),
            reverse=True,
        )
        best_priority = self._source_priority(ranked_tokens[0])
        max_sources = 1 if len(tokens) <= 2 else 2
        selected = [
            token
            for token in ranked_tokens
            if self._source_priority(token) >= best_priority - 0.45
        ]
        return selected[:max_sources]

    def _target_is_allowed(
        self,
        source_token: str,
        neighbor_token: str,
        score: float,
        cooccurrence: int,
    ) -> bool:
        if not neighbor_token or neighbor_token == source_token:
            return False

        source_frequency = self._token_frequency(source_token)
        neighbor_frequency = self._token_frequency(neighbor_token)
        if neighbor_frequency <= 0:
            return False
        if neighbor_frequency >= 20_000 and not self._looks_like_abbreviation(neighbor_token):
            return False

        stem_match = stem_token(source_token) == stem_token(neighbor_token)
        surface_similarity = ngram_jaccard(source_token, neighbor_token)
        reverse_score = self._edge_score(neighbor_token, source_token)

        if stem_match:
            return True
        if surface_similarity >= 0.82:
            return True
        if reverse_score == 0.0 and score < 0.35:
            return False
        if reverse_score == 0.0 and neighbor_frequency > max(2_500, source_frequency * 1.4):
            return False
        if cooccurrence < 2 and score < 0.16:
            return False
        if score < 0.08:
            return False
        return True

    def _neighbors(self, token: str) -> List[sqlite3.Row]:
        if not self.enabled:
            return []
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        rows = self.conn.execute(
            """
            SELECT neighbor, score, cooccurrence
            FROM semantic_neighbors
            WHERE token = ?
            ORDER BY score DESC, cooccurrence DESC, neighbor
            LIMIT ?
            """,
            (token, max(self.top_n * 8, 24)),
        ).fetchall()
        self._cache[token] = rows
        return rows

    def expand_tokens(self, tokens: Iterable[str]) -> tuple[List[str], List[Dict[str, object]]]:
        normalized = normalize_tokens(tokens)
        expansions: List[str] = []
        applied: List[Dict[str, object]] = []
        for token in self._select_source_tokens(normalized):
            rows = self._neighbors(token)
            if not rows:
                continue
            targets: List[str] = []
            for row in rows:
                neighbor = str(row["neighbor"] or "")
                if not self._target_is_allowed(token, neighbor, float(row["score"]), int(row["cooccurrence"])):
                    continue
                targets.append(neighbor)
                if len(unique_preserve_order(targets)) >= self.top_n:
                    break
            if not targets:
                continue
            normalized_targets = unique_preserve_order(normalize_tokens(targets))
            if not normalized_targets:
                continue
            expansions.extend(normalized_targets)
            applied.append(
                {
                    "source": token,
                    "targets": normalized_targets,
                    "backend": "sqlite",
                }
            )
        return unique_preserve_order(expansions), applied

    def _token_pair_similarity(self, left_token: str, right_token: str) -> float:
        if not left_token or not right_token:
            return 0.0
        key = (left_token, right_token)
        cached = self._pair_similarity_cache.get(key)
        if cached is not None:
            return cached

        if stem_token(left_token) == stem_token(right_token):
            score = 1.0
        else:
            surface_similarity = ngram_jaccard(left_token, right_token)
            semantic_edge = max(
                self._edge_score(left_token, right_token),
                self._edge_score(right_token, left_token),
            )
            score = 0.0
            if surface_similarity >= 0.82:
                score = max(score, min(0.96, 0.55 + 0.45 * surface_similarity))
            if semantic_edge > 0.0:
                score = max(score, min(0.88, 0.22 + 1.8 * semantic_edge))
        self._pair_similarity_cache[key] = score
        return score

    def _soft_directional_overlap(self, left_tokens: List[str], right_tokens: List[str]) -> float:
        if not left_tokens or not right_tokens:
            return 0.0
        weighted_total = 0.0
        weighted_hits = 0.0
        for token in left_tokens:
            weight = self._idf(token)
            best_similarity = 0.0
            for candidate in right_tokens:
                best_similarity = max(best_similarity, self._token_pair_similarity(token, candidate))
                if best_similarity >= 1.0:
                    break
            weighted_total += weight
            weighted_hits += weight * best_similarity
        if weighted_total == 0.0:
            return 0.0
        return weighted_hits / weighted_total

    def sentence_similarity(self, left_text: str, right_text: str) -> float:
        left = normalize_text(left_text)
        right = normalize_text(right_text)
        if not left or not right:
            return 0.0
        cache_key = (left, right)
        cached = self._sentence_cache.get(cache_key)
        if cached is not None:
            return cached

        left_tokens = normalize_tokens(tokenize(left))
        right_tokens = normalize_tokens(tokenize(right))
        token_overlap = token_jaccard(left, right)
        char_overlap = ngram_jaccard(left, right)
        directional_overlap = (
            self._soft_directional_overlap(left_tokens, right_tokens)
            + self._soft_directional_overlap(right_tokens, left_tokens)
        ) / 2.0
        similarity = max(token_overlap, char_overlap * 0.75, directional_overlap)
        self._sentence_cache[cache_key] = similarity
        self._sentence_cache[(right, left)] = similarity
        return similarity


class FastTextSemanticBackend:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_FASTTEXT_MODEL_PATH,
        top_n: int = 4,
        similarity_threshold: float = 0.55,
    ) -> None:
        self.model_path = Path(model_path)
        self.top_n = top_n
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.enabled = False
        self._sentence_cache: Dict[str, object] = {}
        if fasttext_module is not None and self.model_path.exists():
            self.model = _load_fasttext_model_silently(self.model_path)
            self.enabled = True

    def _normalize_neighbor(self, source_token: str, candidate: str) -> str:
        normalized_candidates = normalize_tokens([candidate])
        if not normalized_candidates:
            return ""
        normalized = normalized_candidates[0]
        if normalized == source_token:
            return ""
        if normalized.isdigit():
            return ""
        if len(normalized) <= 2:
            return ""
        if not ALPHA_TOKEN_RE.fullmatch(normalized):
            return ""
        if CYRILLIC_RE.search(normalized) and LATIN_RE.search(normalized):
            return ""
        return normalized

    def expand_tokens(self, tokens: Iterable[str]) -> tuple[List[str], List[Dict[str, object]]]:
        if not self.enabled or self.model is None:
            return [], []

        normalized = normalize_tokens(tokens)
        expansions: List[str] = []
        applied: List[Dict[str, object]] = []
        for token in normalized:
            if len(token) <= 2:
                continue
            rows = self.model.get_nearest_neighbors(token, k=max(self.top_n * 4, 16))
            targets: List[str] = []
            for similarity, candidate in rows:
                if similarity < self.similarity_threshold:
                    continue
                normalized_candidate = self._normalize_neighbor(token, candidate)
                if not normalized_candidate:
                    continue
                targets.append(normalized_candidate)
                if len(unique_preserve_order(targets)) >= self.top_n:
                    break
            normalized_targets = unique_preserve_order(targets)
            if not normalized_targets:
                continue
            expansions.extend(normalized_targets)
            applied.append(
                {
                    "source": token,
                    "targets": normalized_targets,
                    "backend": "fasttext",
                }
            )
        return unique_preserve_order(expansions), applied

    def _sentence_vector(self, text: str) -> object:
        normalized = normalize_text(text)
        cached = self._sentence_cache.get(normalized)
        if cached is not None:
            return cached
        if not self.enabled or self.model is None or not normalized:
            return []
        vector = self.model.get_sentence_vector(normalized)
        self._sentence_cache[normalized] = vector
        return vector

    def sentence_similarity(self, left_text: str, right_text: str) -> float:
        if not self.enabled or self.model is None:
            return 0.0
        left_vector = self._sentence_vector(left_text)
        right_vector = self._sentence_vector(right_text)
        return cosine_similarity(left_vector, right_vector)


class SemanticExpander:
    def __init__(
        self,
        conn: sqlite3.Connection,
        top_n: int = 4,
        backend: str = "auto",
        fasttext_model_path: Path | str = DEFAULT_FASTTEXT_MODEL_PATH,
        fasttext_similarity_threshold: float = 0.55,
    ) -> None:
        self.conn = conn
        self.top_n = top_n
        self.backend_name = "none"
        self.fasttext_backend = None
        self.sqlite_backend = None

        if backend not in {"auto", "fasttext", "sqlite"}:
            raise ValueError("semantic backend must be one of: auto, fasttext, sqlite")

        if backend in {"auto", "fasttext"}:
            self.fasttext_backend = FastTextSemanticBackend(
                model_path=fasttext_model_path,
                top_n=top_n,
                similarity_threshold=fasttext_similarity_threshold,
            )
            if self.fasttext_backend.enabled:
                self.backend_name = "fasttext"

        if self.backend_name == "none" and backend in {"auto", "sqlite"}:
            self.sqlite_backend = SqliteSemanticBackend(conn=conn, top_n=top_n)
            if self.sqlite_backend.enabled:
                self.backend_name = "sqlite"

    @property
    def enabled(self) -> bool:
        return self.backend_name != "none"

    def expand_tokens(self, tokens: Iterable[str]) -> tuple[List[str], List[Dict[str, object]]]:
        if self.backend_name == "fasttext" and self.fasttext_backend is not None:
            return self.fasttext_backend.expand_tokens(tokens)
        if self.backend_name == "sqlite" and self.sqlite_backend is not None:
            return self.sqlite_backend.expand_tokens(tokens)
        return [], []

    def sentence_similarity(self, left_text: str, right_text: str) -> float:
        if self.backend_name == "fasttext" and self.fasttext_backend is not None:
            return self.fasttext_backend.sentence_similarity(left_text, right_text)
        if self.backend_name == "sqlite" and self.sqlite_backend is not None:
            return self.sqlite_backend.sentence_similarity(left_text, right_text)
        return 0.0
