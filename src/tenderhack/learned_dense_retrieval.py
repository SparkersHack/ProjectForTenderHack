from __future__ import annotations

import math
import sqlite3
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer

from .text import normalize_text, normalize_tokens, tokenize, unique_preserve_order


LEARNED_DENSE_VERSION = 1
DEFAULT_LEARNED_DENSE_DIM = 128
DEFAULT_LEARNED_DENSE_SAMPLE_SIZE = 100_000
DEFAULT_LEARNED_DENSE_TOP_K = 80


def default_learned_dense_model_path(search_db_path: Path | str) -> Path:
    return Path(search_db_path).with_suffix(".learned_dense.joblib")


def default_learned_dense_index_path(search_db_path: Path | str) -> Path:
    return Path(search_db_path).with_suffix(".learned_dense.npz")


def _document_text(normalized_name: str, normalized_category: str, key_tokens: str) -> str:
    parts: List[str] = []
    if normalized_name:
        parts.extend([normalized_name, normalized_name])
    if normalized_category:
        parts.append(normalized_category)
    if key_tokens:
        parts.append(key_tokens)
    return " ".join(part for part in parts if part)


def _analysis_to_query_text(analysis: object) -> str:
    weighted_tokens: List[str] = []
    corrected_tokens = list(getattr(analysis, "corrected_tokens", []) or [])
    synonym_expansions = list(getattr(analysis, "synonym_expansions", []) or [])
    completion_expansions = list(getattr(analysis, "completion_expansions", []) or [])
    semantic_expansions = list(getattr(analysis, "semantic_expansions", []) or [])

    weighted_tokens.extend(corrected_tokens * 3)
    weighted_tokens.extend(synonym_expansions * 2)
    weighted_tokens.extend(completion_expansions)
    weighted_tokens.extend(semantic_expansions)

    normalized = normalize_tokens(weighted_tokens)
    if normalized:
        return " ".join(normalized)
    corrected_query = normalize_text(str(getattr(analysis, "corrected_query", "") or ""))
    if corrected_query:
        return corrected_query
    return normalize_text(str(getattr(analysis, "normalized_query", "") or ""))


class LearnedDenseRetriever:
    def __init__(
        self,
        conn: sqlite3.Connection,
        model_path: Path | str,
        index_path: Path | str,
        top_k: int = DEFAULT_LEARNED_DENSE_TOP_K,
    ) -> None:
        self.conn = conn
        self.model_path = Path(model_path)
        self.index_path = Path(index_path)
        self.top_k = top_k
        self.enabled = False
        self.backend_name = "learned"

        self._word_vectorizer: HashingVectorizer | None = None
        self._char_vectorizer: HashingVectorizer | None = None
        self._tfidf_transformer: TfidfTransformer | None = None
        self._svd: TruncatedSVD | None = None
        self._normalizer: Normalizer | None = None
        self.row_ids = np.zeros(0, dtype=np.int64)
        self.doc_vectors = np.zeros((0, 1), dtype=np.float32)
        self._row_id_to_pos: Dict[int, int] = {}

        self._load()

    def _load(self) -> None:
        if not self.model_path.exists() or not self.index_path.exists():
            return
        payload = joblib.load(self.model_path)
        if int(payload.get("version", 0)) != LEARNED_DENSE_VERSION:
            return
        self._word_vectorizer = payload["word_vectorizer"]
        self._char_vectorizer = payload["char_vectorizer"]
        self._tfidf_transformer = payload["tfidf_transformer"]
        self._svd = payload["svd"]
        self._normalizer = payload["normalizer"]

        index_payload = np.load(self.index_path, allow_pickle=False)
        version = int(index_payload["version"][0]) if "version" in index_payload else 0
        if version != LEARNED_DENSE_VERSION:
            return
        self.row_ids = index_payload["row_ids"].astype(np.int64, copy=False)
        self.doc_vectors = index_payload["vectors"].astype(np.float32, copy=False)
        self._row_id_to_pos = {int(row_id): index for index, row_id in enumerate(self.row_ids.tolist())}
        self.enabled = bool(len(self.row_ids) and self.doc_vectors.size)

    def _base_matrix(self, texts: Sequence[str]) -> sparse.spmatrix:
        assert self._word_vectorizer is not None
        assert self._char_vectorizer is not None
        word_matrix = self._word_vectorizer.transform(texts)
        char_matrix = self._char_vectorizer.transform(texts)
        return sparse.hstack([word_matrix, char_matrix], format="csr")

    def _transform_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts or not self.enabled:
            return np.zeros((0, self.doc_vectors.shape[1] if self.doc_vectors.ndim == 2 else 1), dtype=np.float32)
        assert self._tfidf_transformer is not None
        assert self._svd is not None
        assert self._normalizer is not None
        base_matrix = self._base_matrix(texts)
        tfidf_matrix = self._tfidf_transformer.transform(base_matrix)
        dense_matrix = self._svd.transform(tfidf_matrix)
        dense_matrix = self._normalizer.transform(dense_matrix)
        dense_matrix = np.nan_to_num(dense_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return dense_matrix.astype(np.float32, copy=False)

    def build_query_vector(self, analysis: object) -> np.ndarray:
        if not self.enabled:
            return np.zeros(0, dtype=np.float32)
        query_text = _analysis_to_query_text(analysis)
        if not query_text:
            return np.zeros(self.doc_vectors.shape[1], dtype=np.float32)
        return self._transform_texts([query_text])[0]

    def search(self, analysis: object, top_k: int | None = None) -> List[Tuple[int, float]]:
        if not self.enabled or not len(self.row_ids):
            return []
        query_vector = self.build_query_vector(analysis)
        if not query_vector.size or not float(np.linalg.norm(query_vector)):
            return []
        scores = np.einsum("ij,j->i", self.doc_vectors, query_vector, optimize=True)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        limit = max(1, min(int(top_k or self.top_k), int(scores.shape[0])))
        if limit >= scores.shape[0]:
            top_indices = np.argsort(scores)[::-1]
        else:
            candidate_indices = np.argpartition(scores, -limit)[-limit:]
            top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        return [
            (int(self.row_ids[index]), float(scores[index]))
            for index in top_indices.tolist()
            if float(scores[index]) > 0.0
        ]

    def score_row_ids(self, analysis: object, row_ids: Iterable[int]) -> Dict[int, float]:
        if not self.enabled:
            return {}
        query_vector = self.build_query_vector(analysis)
        if not query_vector.size or not float(np.linalg.norm(query_vector)):
            return {}
        results: Dict[int, float] = {}
        for row_id in row_ids:
            position = self._row_id_to_pos.get(int(row_id))
            if position is None:
                continue
            score = float(np.dot(self.doc_vectors[position], query_vector))
            if math.isfinite(score) and score > 0.0:
                results[int(row_id)] = score
        return results


def build_learned_dense_artifacts(
    conn: sqlite3.Connection,
    model_path: Path | str,
    index_path: Path | str,
    embedding_dim: int = DEFAULT_LEARNED_DENSE_DIM,
    sample_size: int = DEFAULT_LEARNED_DENSE_SAMPLE_SIZE,
    random_state: int = 42,
) -> Tuple[int, int]:
    rows = conn.execute(
        """
        SELECT rowid, normalized_name, normalized_category, key_tokens
        FROM ste_catalog
        ORDER BY rowid
        """
    ).fetchall()
    row_ids = [int(row["rowid"]) for row in rows]
    texts = [
        _document_text(
            str(row["normalized_name"] or ""),
            str(row["normalized_category"] or ""),
            str(row["key_tokens"] or ""),
        )
        for row in rows
    ]

    word_vectorizer = HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=2**18,
        alternate_sign=False,
        norm=None,
        lowercase=False,
    )
    char_vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=2**17,
        alternate_sign=False,
        norm=None,
        lowercase=False,
    )

    word_matrix = word_vectorizer.transform(texts)
    char_matrix = char_vectorizer.transform(texts)
    base_matrix = sparse.hstack([word_matrix, char_matrix], format="csr")

    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    tfidf_matrix = tfidf_transformer.fit_transform(base_matrix)

    rng = np.random.default_rng(random_state)
    sample_size = min(max(1, int(sample_size)), tfidf_matrix.shape[0])
    if sample_size < tfidf_matrix.shape[0]:
        sample_indices = np.sort(rng.choice(tfidf_matrix.shape[0], size=sample_size, replace=False))
        fit_matrix = tfidf_matrix[sample_indices]
    else:
        fit_matrix = tfidf_matrix

    max_components = max(2, min(fit_matrix.shape[0] - 1, fit_matrix.shape[1] - 1))
    svd_algorithm = "arpack" if fit_matrix.shape[0] <= 512 else "randomized"
    svd = TruncatedSVD(
        n_components=min(int(embedding_dim), max_components),
        algorithm=svd_algorithm,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
        svd.fit(fit_matrix)
    normalizer = Normalizer(copy=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
        dense_vectors = normalizer.transform(svd.transform(tfidf_matrix))
    dense_vectors = np.nan_to_num(dense_vectors, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    model_path = Path(model_path)
    index_path = Path(index_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": LEARNED_DENSE_VERSION,
            "word_vectorizer": word_vectorizer,
            "char_vectorizer": char_vectorizer,
            "tfidf_transformer": tfidf_transformer,
            "svd": svd,
            "normalizer": normalizer,
        },
        model_path,
    )
    np.savez_compressed(
        index_path,
        version=np.asarray([LEARNED_DENSE_VERSION], dtype=np.int32),
        row_ids=np.asarray(row_ids, dtype=np.int64),
        vectors=dense_vectors,
    )
    return len(row_ids), int(dense_vectors.shape[1] if dense_vectors.ndim == 2 else embedding_dim)


def rebuild_learned_dense_artifacts(
    search_db_path: Path | str,
    model_path: Path | str | None = None,
    index_path: Path | str | None = None,
    embedding_dim: int = DEFAULT_LEARNED_DENSE_DIM,
    sample_size: int = DEFAULT_LEARNED_DENSE_SAMPLE_SIZE,
    random_state: int = 42,
) -> Tuple[int, int]:
    search_db_path = Path(search_db_path)
    resolved_model_path = (
        Path(model_path) if model_path is not None else default_learned_dense_model_path(search_db_path)
    )
    resolved_index_path = (
        Path(index_path) if index_path is not None else default_learned_dense_index_path(search_db_path)
    )
    conn = sqlite3.connect(search_db_path)
    conn.row_factory = sqlite3.Row
    try:
        return build_learned_dense_artifacts(
            conn,
            model_path=resolved_model_path,
            index_path=resolved_index_path,
            embedding_dim=embedding_dim,
            sample_size=sample_size,
            random_state=random_state,
        )
    finally:
        conn.close()
