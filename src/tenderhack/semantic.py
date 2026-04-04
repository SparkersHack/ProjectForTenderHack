from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List

from .dataset_layout import resolve_fasttext_model_path
from .text import normalize_text, normalize_tokens, unique_preserve_order


DEFAULT_FASTTEXT_MODEL_PATH = resolve_fasttext_model_path()

try:
    import fasttext as fasttext_module
except ImportError:
    fasttext_module = None


ALPHA_TOKEN_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]+$")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")


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
        self.enabled = self._has_assets()

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
            (token, self.top_n),
        ).fetchall()
        self._cache[token] = rows
        return rows

    def expand_tokens(self, tokens: Iterable[str]) -> tuple[List[str], List[Dict[str, object]]]:
        normalized = normalize_tokens(tokens)
        expansions: List[str] = []
        applied: List[Dict[str, object]] = []
        for token in normalized:
            rows = self._neighbors(token)
            if not rows:
                continue
            targets = [row["neighbor"] for row in rows if row["neighbor"]]
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

    def sentence_similarity(self, left_text: str, right_text: str) -> float:
        return 0.0


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
