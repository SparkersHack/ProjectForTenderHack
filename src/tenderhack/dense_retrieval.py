from __future__ import annotations

import hashlib
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .text import normalize_tokens, tokenize, unique_preserve_order


DENSE_INDEX_VERSION = 1
DEFAULT_DENSE_EMBEDDING_DIM = 48
DEFAULT_DENSE_TOP_K = 80
DEFAULT_DENSE_SEMANTIC_NEIGHBORS = 4


def default_dense_index_path(search_db_path: Path | str) -> Path:
    return Path(search_db_path).with_suffix(".dense.npz")


def _stable_seed(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _char_ngrams(value: str, min_n: int = 3, max_n: int = 5) -> List[str]:
    padded = f"<{value}>"
    ngrams: List[str] = []
    for size in range(min_n, max_n + 1):
        if len(padded) < size:
            continue
        for index in range(len(padded) - size + 1):
            ngrams.append(padded[index : index + size])
    return ngrams


class DenseRetriever:
    def __init__(
        self,
        conn: sqlite3.Connection,
        index_path: Path | str,
        embedding_dim: int = DEFAULT_DENSE_EMBEDDING_DIM,
        semantic_neighbors_per_token: int = DEFAULT_DENSE_SEMANTIC_NEIGHBORS,
    ) -> None:
        self.conn = conn
        self.index_path = Path(index_path)
        self.embedding_dim = embedding_dim
        self.semantic_neighbors_per_token = semantic_neighbors_per_token
        self.backend_name = "graph"
        self.enabled = False

        self.row_ids = np.zeros(0, dtype=np.int64)
        self.doc_vectors = np.zeros((0, embedding_dim), dtype=np.float32)
        self._row_id_to_pos: Dict[int, int] = {}
        self._token_frequency_cache: Dict[str, int] = {}
        self._idf_cache: Dict[str, float] = {}
        self._neighbor_cache: Dict[str, List[Tuple[str, float]]] = {}
        self._base_vector_cache: Dict[str, np.ndarray] = {}
        self._token_vector_cache: Dict[str, np.ndarray] = {}
        self._row_count = self._load_row_count()

        self._load_index()

    def _load_row_count(self) -> int:
        metadata_row = self.conn.execute(
            "SELECT value FROM search_metadata WHERE key = 'deduped_rows' LIMIT 1"
        ).fetchone()
        if metadata_row is not None:
            try:
                return max(1, int(metadata_row[0]))
            except (TypeError, ValueError):
                pass
        row = self.conn.execute("SELECT COUNT(*) FROM ste_catalog").fetchone()
        return max(1, int(row[0] if row is not None else 1))

    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        payload = np.load(self.index_path, allow_pickle=False)
        version = int(payload["version"][0]) if "version" in payload else 0
        dimension = int(payload["embedding_dim"][0]) if "embedding_dim" in payload else self.embedding_dim
        if version != DENSE_INDEX_VERSION or dimension != self.embedding_dim:
            return
        self.row_ids = payload["row_ids"].astype(np.int64, copy=False)
        self.doc_vectors = payload["vectors"].astype(np.float32, copy=False)
        self._row_id_to_pos = {int(row_id): index for index, row_id in enumerate(self.row_ids.tolist())}
        self.enabled = bool(len(self.row_ids) and self.doc_vectors.size)

    def _token_frequency(self, token: str) -> int:
        cached = self._token_frequency_cache.get(token)
        if cached is not None:
            return cached
        row = self.conn.execute(
            "SELECT frequency FROM token_frequency WHERE token = ? LIMIT 1",
            (token,),
        ).fetchone()
        frequency = int(row[0]) if row is not None else 0
        self._token_frequency_cache[token] = frequency
        return frequency

    def _idf(self, token: str) -> float:
        cached = self._idf_cache.get(token)
        if cached is not None:
            return cached
        frequency = max(1, self._token_frequency(token))
        value = 1.0 + math.log1p(self._row_count / frequency)
        self._idf_cache[token] = value
        return value

    def _hash_vector(self, key: str) -> np.ndarray:
        cached = self._base_vector_cache.get(key)
        if cached is not None:
            return cached
        seed = _stable_seed(key)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.embedding_dim, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        self._base_vector_cache[key] = vector
        return vector

    def _surface_vector(self, token: str) -> np.ndarray:
        normalized = normalize_tokens([token])
        if not normalized:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        token = normalized[0]
        token_vector = self._hash_vector(f"tok:{token}")
        ngrams = unique_preserve_order(_char_ngrams(token))
        if not ngrams:
            return token_vector
        accumulator = token_vector.astype(np.float32, copy=True) * 0.7
        for ngram in ngrams:
            accumulator += 0.3 * self._hash_vector(f"ng:{ngram}")
        norm = float(np.linalg.norm(accumulator))
        if norm > 0.0:
            accumulator /= norm
        return accumulator.astype(np.float32, copy=False)

    def _semantic_neighbors(self, token: str) -> List[Tuple[str, float]]:
        cached = self._neighbor_cache.get(token)
        if cached is not None:
            return cached
        rows = self.conn.execute(
            """
            SELECT neighbor, score
            FROM semantic_neighbors
            WHERE token = ?
            ORDER BY score DESC, cooccurrence DESC, neighbor
            LIMIT ?
            """,
            (token, max(self.semantic_neighbors_per_token * 4, 12)),
        ).fetchall()
        neighbors: List[Tuple[str, float]] = []
        for row in rows:
            neighbor = normalize_tokens([str(row[0] or "")])
            if not neighbor:
                continue
            score = float(row[1] or 0.0)
            if score < 0.08:
                continue
            neighbors.append((neighbor[0], score))
            if len(neighbors) >= self.semantic_neighbors_per_token:
                break
        self._neighbor_cache[token] = neighbors
        return neighbors

    def _token_vector(self, token: str) -> np.ndarray:
        cached = self._token_vector_cache.get(token)
        if cached is not None:
            return cached
        surface = self._surface_vector(token)
        accumulator = surface.astype(np.float32, copy=True)
        weight_total = 1.0
        for neighbor, score in self._semantic_neighbors(token):
            neighbor_surface = self._surface_vector(neighbor)
            weight = min(0.35, 0.08 + score)
            accumulator += weight * neighbor_surface
            weight_total += weight
        if weight_total > 0.0:
            accumulator /= weight_total
        norm = float(np.linalg.norm(accumulator))
        if norm > 0.0:
            accumulator /= norm
        accumulator = accumulator.astype(np.float32, copy=False)
        self._token_vector_cache[token] = accumulator
        return accumulator

    def _compose_vector(self, weighted_tokens: Sequence[Tuple[str, float]]) -> np.ndarray:
        token_weights: Dict[str, float] = defaultdict(float)
        for token, base_weight in weighted_tokens:
            normalized = normalize_tokens([token])
            if not normalized:
                continue
            token_weights[normalized[0]] += float(base_weight)
        if not token_weights:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        total_weight = 0.0
        for token, base_weight in token_weights.items():
            token_weight = base_weight * self._idf(token)
            vector += token_weight * self._token_vector(token)
            total_weight += token_weight
        if total_weight > 0.0:
            vector /= total_weight
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector.astype(np.float32, copy=False)

    def build_query_vector(self, analysis: object) -> np.ndarray:
        weighted_tokens: List[Tuple[str, float]] = []
        corrected_tokens = list(getattr(analysis, "corrected_tokens", []) or [])
        synonym_expansions = list(getattr(analysis, "synonym_expansions", []) or [])
        completion_expansions = list(getattr(analysis, "completion_expansions", []) or [])
        semantic_expansions = list(getattr(analysis, "semantic_expansions", []) or [])

        weighted_tokens.extend((token, 2.4) for token in corrected_tokens)
        weighted_tokens.extend((token, 1.5) for token in synonym_expansions if token not in corrected_tokens)
        weighted_tokens.extend((token, 1.0) for token in completion_expansions if token not in corrected_tokens)
        weighted_tokens.extend((token, 1.1) for token in semantic_expansions if token not in corrected_tokens)
        return self._compose_vector(weighted_tokens)

    def search(self, analysis: object, top_k: int = DEFAULT_DENSE_TOP_K) -> List[Tuple[int, float]]:
        if not self.enabled or not len(self.row_ids):
            return []
        query_vector = self.build_query_vector(analysis)
        if not float(np.linalg.norm(query_vector)):
            return []
        with np.errstate(over="ignore", invalid="ignore"):
            scores = np.einsum("ij,j->i", self.doc_vectors, query_vector, optimize=True)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        if scores.size == 0:
            return []
        top_k = max(1, min(int(top_k), int(scores.shape[0])))
        if top_k >= scores.shape[0]:
            top_indices = np.argsort(scores)[::-1]
        else:
            candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        results: List[Tuple[int, float]] = []
        for index in top_indices.tolist():
            score = float(scores[index])
            if score <= 0.0:
                continue
            results.append((int(self.row_ids[index]), score))
        return results

    def score_row_ids(self, analysis: object, row_ids: Iterable[int]) -> Dict[int, float]:
        if not self.enabled or not len(self.row_ids):
            return {}
        query_vector = self.build_query_vector(analysis)
        if not float(np.linalg.norm(query_vector)):
            return {}
        results: Dict[int, float] = {}
        for row_id in row_ids:
            position = self._row_id_to_pos.get(int(row_id))
            if position is None:
                continue
            score = float(self.doc_vectors[position] @ query_vector)
            if score > 0.0:
                results[int(row_id)] = score
        return results


def _document_weighted_tokens(row: sqlite3.Row) -> List[Tuple[str, float]]:
    weighted_tokens: List[Tuple[str, float]] = []
    weighted_tokens.extend((token, 2.2) for token in tokenize(str(row["normalized_name"] or "")))
    weighted_tokens.extend((token, 1.2) for token in tokenize(str(row["normalized_category"] or "")))
    weighted_tokens.extend((token, 0.9) for token in tokenize(str(row["key_tokens"] or "")))
    return weighted_tokens


def build_dense_index(
    conn: sqlite3.Connection,
    index_path: Path | str,
    embedding_dim: int = DEFAULT_DENSE_EMBEDDING_DIM,
    semantic_neighbors_per_token: int = DEFAULT_DENSE_SEMANTIC_NEIGHBORS,
) -> Tuple[int, int]:
    retriever = DenseRetriever(
        conn=conn,
        index_path=index_path,
        embedding_dim=embedding_dim,
        semantic_neighbors_per_token=semantic_neighbors_per_token,
    )
    row_ids: List[int] = []
    vectors: List[np.ndarray] = []
    for row in conn.execute(
        """
        SELECT rowid, normalized_name, normalized_category, key_tokens
        FROM ste_catalog
        ORDER BY rowid
        """
    ):
        vector = retriever._compose_vector(_document_weighted_tokens(row))
        row_ids.append(int(row["rowid"]))
        vectors.append(vector)
    matrix = (
        np.vstack(vectors).astype(np.float32, copy=False)
        if vectors
        else np.zeros((0, embedding_dim), dtype=np.float32)
    )
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_path,
        version=np.asarray([DENSE_INDEX_VERSION], dtype=np.int32),
        embedding_dim=np.asarray([embedding_dim], dtype=np.int32),
        row_ids=np.asarray(row_ids, dtype=np.int64),
        vectors=matrix,
    )
    return len(row_ids), embedding_dim


def rebuild_dense_index(
    search_db_path: Path | str,
    index_path: Path | str | None = None,
    embedding_dim: int = DEFAULT_DENSE_EMBEDDING_DIM,
    semantic_neighbors_per_token: int = DEFAULT_DENSE_SEMANTIC_NEIGHBORS,
) -> Tuple[int, int]:
    search_db_path = Path(search_db_path)
    resolved_index_path = Path(index_path) if index_path is not None else default_dense_index_path(search_db_path)
    conn = sqlite3.connect(search_db_path)
    conn.row_factory = sqlite3.Row
    try:
        return build_dense_index(
            conn,
            index_path=resolved_index_path,
            embedding_dim=embedding_dim,
            semantic_neighbors_per_token=semantic_neighbors_per_token,
        )
    finally:
        conn.close()
