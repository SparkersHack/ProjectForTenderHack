from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .dense_retrieval import (
    DEFAULT_DENSE_EMBEDDING_DIM,
    DEFAULT_DENSE_TOP_K,
    DenseRetriever,
    default_dense_index_path,
)
from .learned_dense_retrieval import (
    DEFAULT_LEARNED_DENSE_DIM,
    LearnedDenseRetriever,
    default_learned_dense_index_path,
    default_learned_dense_model_path,
)
from .semantic import DEFAULT_FASTTEXT_MODEL_PATH, SemanticExpander
from .text import (
    extract_attribute_spans,
    normalize_text,
    normalize_tokens,
    stem_token,
    stem_tokens,
    tokenize,
    unique_preserve_order,
)


DEFAULT_SEARCH_DB = Path("data/processed/tenderhack_search.sqlite")
DEFAULT_SYNONYMS_PATH = Path("data/reference/search_synonyms.json")


def _token_sequence_contains(tokens: List[str], phrase_tokens: List[str]) -> bool:
    if not tokens or not phrase_tokens or len(phrase_tokens) > len(tokens):
        return False
    phrase_length = len(phrase_tokens)
    for index in range(len(tokens) - phrase_length + 1):
        if tokens[index : index + phrase_length] == phrase_tokens:
            return True
    return False


def _edit_distance(left: str, right: str, max_distance: int = 2) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    previous_previous: list[int] | None = None
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        best = current[0]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            if (
                previous_previous is not None
                and i > 1
                and j > 1
                and left[i - 1] == right[j - 2]
                and left[i - 2] == right[j - 1]
            ):
                value = min(value, previous_previous[j - 2] + 1)
            current.append(value)
            best = min(best, value)
        if best > max_distance:
            return max_distance + 1
        previous_previous, previous = previous, current
    return previous[-1]


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    corrected_query: str
    original_tokens: List[str]
    corrected_tokens: List[str]
    stemmed_tokens: List[str]
    lexical_expanded_tokens: List[str]
    expanded_tokens: List[str]
    expanded_stems: List[str]
    synonym_expansions: List[str]
    semantic_expansions: List[str]
    completion_expansions: List[str]
    applied_corrections: List[Dict[str, str]]
    applied_completions: List[Dict[str, List[str]]]
    applied_synonyms: List[Dict[str, List[str]]]
    applied_semantic_neighbors: List[Dict[str, object]]
    # Structured attribute pairs extracted from the query, e.g. [("500", "мг"), ("16", "гб")]
    attribute_spans: List[tuple]


class TypoCorrector:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @staticmethod
    def _is_inflection_only_variant(source: str, candidate: str) -> bool:
        if source == candidate:
            return False
        if not re.fullmatch(r"[а-я]+", source) or not re.fullmatch(r"[а-я]+", candidate):
            return False
        if len(source) <= 4 or len(candidate) <= 4:
            return False
        return stem_token(source) == stem_token(candidate)

    def completion_candidates(self, token: str, limit: int = 6) -> List[str]:
        token = token.strip()
        if len(token) < 3 or token.isdigit():
            return []
        exists = self.conn.execute(
            "SELECT 1 FROM token_frequency WHERE token = ? LIMIT 1",
            (token,),
        ).fetchone()
        if exists and len(token) > 4:
            return []
        rows = self.conn.execute(
            """
            SELECT token, frequency
            FROM token_frequency
            WHERE token LIKE ?
              AND token_length > ?
            ORDER BY frequency DESC, token_length ASC, token ASC
            LIMIT ?
            """,
            (f"{token}%", len(token), limit * 3),
        ).fetchall()
        completions: List[str] = []
        for row in rows:
            candidate = str(row["token"])
            if not candidate or candidate == token:
                continue
            completions.append(candidate)
            if len(completions) >= limit:
                break
        return completions

    def _candidate_tokens(self, token: str) -> List[sqlite3.Row]:
        first_char = token[0]
        length = len(token)
        candidates: Dict[str, sqlite3.Row] = {}

        prefix = token[: min(4, len(token))]
        for row in self.conn.execute(
            """
            SELECT token, frequency
            FROM token_frequency
            WHERE token LIKE ?
              AND token_length BETWEEN ? AND ?
            ORDER BY frequency DESC
            LIMIT 150
            """,
            (f"{prefix}%", max(1, length - 2), length + 2),
        ).fetchall():
            candidates[row["token"]] = row

        for row in self.conn.execute(
            """
            SELECT token, frequency
            FROM token_frequency
            WHERE first_char = ?
              AND token_length BETWEEN ? AND ?
            ORDER BY frequency DESC
            LIMIT 250
            """,
            (first_char, max(1, length - 2), length + 2),
        ).fetchall():
            candidates.setdefault(row["token"], row)

        if candidates:
            return sorted(candidates.values(), key=lambda row: row["frequency"], reverse=True)

        return self.conn.execute(
            """
            SELECT token, frequency
            FROM token_frequency
            WHERE token_length BETWEEN ? AND ?
            ORDER BY frequency DESC
            LIMIT 250
            """,
            (max(1, length - 1), length + 1),
        ).fetchall()

    def correct_tokens(self, tokens: Iterable[str]) -> tuple[List[str], List[Dict[str, str]]]:
        corrected: List[str] = []
        applied: List[Dict[str, str]] = []
        for token in tokens:
            if len(token) <= 2 or token.isdigit():
                corrected.append(token)
                continue
            if self.completion_candidates(token, limit=1):
                corrected.append(token)
                continue
            exists = self.conn.execute(
                "SELECT 1 FROM token_frequency WHERE token = ? LIMIT 1",
                (token,),
            ).fetchone()
            if exists:
                corrected.append(token)
                continue
            best_token = token
            best_distance = 99
            best_frequency = -1
            for row in self._candidate_tokens(token):
                candidate = row["token"]
                distance = _edit_distance(token, candidate, max_distance=2)
                if distance > 2:
                    continue
                if distance < best_distance or (distance == best_distance and row["frequency"] > best_frequency):
                    best_token = candidate
                    best_distance = distance
                    best_frequency = row["frequency"]
            if self._is_inflection_only_variant(token, best_token):
                best_token = token
            corrected.append(best_token)
            if best_token != token:
                applied.append({"source": token, "target": best_token})
        return corrected, applied


class SearchService:
    def __init__(
        self,
        search_db_path: Path | str = DEFAULT_SEARCH_DB,
        synonyms_path: Path | str = DEFAULT_SYNONYMS_PATH,
        semantic_top_n: int = 4,
        semantic_backend: str = "auto",
        fasttext_model_path: Path | str = DEFAULT_FASTTEXT_MODEL_PATH,
        fasttext_similarity_threshold: float = 0.55,
        dense_index_path: Path | str | None = None,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        dense_embedding_dim: int = DEFAULT_DENSE_EMBEDDING_DIM,
        learned_dense_model_path: Path | str | None = None,
        learned_dense_index_path: Path | str | None = None,
    ) -> None:
        self.search_db_path = Path(search_db_path)
        self.synonyms_path = Path(synonyms_path)
        self.dense_index_path = (
            Path(dense_index_path)
            if dense_index_path is not None
            else default_dense_index_path(self.search_db_path)
        )
        self.dense_top_k = dense_top_k
        self.learned_dense_model_path = (
            Path(learned_dense_model_path)
            if learned_dense_model_path is not None
            else default_learned_dense_model_path(self.search_db_path)
        )
        self.learned_dense_index_path = (
            Path(learned_dense_index_path)
            if learned_dense_index_path is not None
            else default_learned_dense_index_path(self.search_db_path)
        )
        self.conn = sqlite3.connect(self.search_db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.corrector = TypoCorrector(self.conn)
        self.synonyms = self._load_synonyms()
        self.semantic_expander = SemanticExpander(
            self.conn,
            top_n=semantic_top_n,
            backend=semantic_backend,
            fasttext_model_path=fasttext_model_path,
            fasttext_similarity_threshold=fasttext_similarity_threshold,
        )
        self.learned_dense_retriever = LearnedDenseRetriever(
            self.conn,
            model_path=self.learned_dense_model_path,
            index_path=self.learned_dense_index_path,
            top_k=dense_top_k,
        )
        self.dense_retriever = (
            self.learned_dense_retriever
            if self.learned_dense_retriever.enabled
            else DenseRetriever(
                self.conn,
                index_path=self.dense_index_path,
                embedding_dim=dense_embedding_dim,
            )
        )

    def close(self) -> None:
        self.conn.close()

    def _load_synonyms(self) -> Dict[str, Dict[str, List[str]]]:
        payload = json.loads(self.synonyms_path.read_text(encoding="utf-8"))
        phrase_synonyms = {normalize_text(key): [normalize_text(value) for value in values] for key, values in payload.get("phrase_synonyms", {}).items()}
        token_synonyms = {normalize_text(key): [normalize_text(value) for value in values] for key, values in payload.get("token_synonyms", {}).items()}
        return {
            "phrase_synonyms": phrase_synonyms,
            "token_synonyms": token_synonyms,
        }

    def _apply_synonyms(self, normalized_query: str, corrected_tokens: List[str]) -> tuple[List[str], List[Dict[str, List[str]]]]:
        expanded: List[str] = list(corrected_tokens)
        applied: List[Dict[str, List[str]]] = []
        phrase_synonyms = self.synonyms["phrase_synonyms"]
        token_synonyms = self.synonyms["token_synonyms"]
        query_tokens = normalize_tokens(tokenize(normalized_query))

        for phrase, replacements in phrase_synonyms.items():
            phrase_tokens = normalize_tokens(tokenize(phrase))
            if phrase_tokens and _token_sequence_contains(query_tokens, phrase_tokens):
                expanded.extend(replacements)
                applied.append({"source": phrase, "targets": replacements})

        for token in corrected_tokens:
            replacements = token_synonyms.get(token)
            if replacements:
                expanded.extend(replacements)
                applied.append({"source": token, "targets": replacements})

        expanded_tokens: List[str] = []
        for item in expanded:
            expanded_tokens.extend(tokenize(item))
        return unique_preserve_order(expanded_tokens), applied

    def _expand_completion_tokens(self, corrected_tokens: Iterable[str]) -> tuple[List[str], List[Dict[str, List[str]]]]:
        expanded: List[str] = []
        applied: List[Dict[str, List[str]]] = []
        for token in corrected_tokens:
            completions = self.corrector.completion_candidates(token)
            if not completions:
                continue
            expanded.extend(completions)
            applied.append({"source": token, "targets": completions})
        return unique_preserve_order(expanded), applied

    @staticmethod
    def _dedupe_applied_targets(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
        result: List[Dict[str, object]] = []
        seen = set()
        for item in items:
            source = item.get("source")
            targets = tuple(item.get("targets", []))
            key = (source, targets)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def analyze_query(self, query: str) -> QueryAnalysis:
        normalized_query = normalize_text(query)
        original_tokens = normalize_tokens(tokenize(normalized_query))

        corrected_tokens: List[str] = []
        corrections: List[Dict[str, str]] = []
        synonym_keys = set(self.synonyms["token_synonyms"].keys())
        for token in original_tokens:
            if token in synonym_keys:
                corrected_tokens.append(token)
                continue
            corrected_part, applied_part = self.corrector.correct_tokens([token])
            corrected_tokens.extend(corrected_part)
            corrections.extend(applied_part)

        corrected_tokens = normalize_tokens(corrected_tokens)
        corrected_query = " ".join(corrected_tokens)

        completion_expansions, applied_completions = self._expand_completion_tokens(corrected_tokens)
        expansion_query = corrected_query or normalized_query
        synonym_expansions, applied_synonyms = self._apply_synonyms(expansion_query, corrected_tokens)
        applied_synonyms = self._dedupe_applied_targets(applied_synonyms)
        semantic_expansions, applied_semantic_neighbors = self.semantic_expander.expand_tokens(corrected_tokens)
        lexical_expanded_tokens = unique_preserve_order(corrected_tokens + synonym_expansions + completion_expansions)
        merged_tokens = unique_preserve_order(lexical_expanded_tokens + semantic_expansions)
        stemmed_tokens = stem_tokens(corrected_tokens)
        expanded_stems = stem_tokens(merged_tokens)
        # Extract structured attribute spans (value+unit pairs) from the original query
        attribute_spans = extract_attribute_spans(query)
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            corrected_query=corrected_query,
            original_tokens=original_tokens,
            corrected_tokens=corrected_tokens,
            stemmed_tokens=stemmed_tokens,
            lexical_expanded_tokens=lexical_expanded_tokens,
            expanded_tokens=merged_tokens,
            expanded_stems=expanded_stems,
            synonym_expansions=synonym_expansions,
            semantic_expansions=semantic_expansions,
            completion_expansions=completion_expansions,
            applied_corrections=corrections,
            applied_completions=applied_completions,
            applied_synonyms=applied_synonyms,
            applied_semantic_neighbors=applied_semantic_neighbors,
            attribute_spans=attribute_spans,
        )

    @staticmethod
    def _build_match_query_from_tokens(tokens: Iterable[str], stems: Iterable[str]) -> str:
        terms: List[str] = []
        for token in tokens:
            if token:
                terms.append(f"{token}*")
        for stem in stems:
            if len(stem) >= 3:
                terms.append(f"{stem}*")
        terms = unique_preserve_order(terms)
        if not terms:
            return ""
        return " OR ".join(terms)

    def _fetch_match_candidates(self, match_query: str, candidate_limit: int) -> List[Dict[str, object]]:
        if not match_query:
            return []
        rows = self.conn.execute(
            """
            SELECT
                ste_catalog.rowid AS row_id,
                ste_catalog.ste_id,
                ste_catalog.clean_name,
                ste_catalog.normalized_name,
                ste_catalog.category,
                ste_catalog.normalized_category,
                ste_catalog.attribute_keys,
                ste_catalog.attribute_count,
                ste_catalog.key_tokens,
                bm25(ste_catalog_fts, 1.6, 2.1, 0.9, 1.2, 0.7) AS bm25_score
            FROM ste_catalog_fts
            JOIN ste_catalog ON ste_catalog_fts.rowid = ste_catalog.rowid
            WHERE ste_catalog_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (match_query, candidate_limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def _fetch_rows_by_ids(self, row_ids: List[int], bm25_score: float | None = None) -> List[Dict[str, object]]:
        if not row_ids:
            return []
        placeholders = ", ".join("?" for _ in row_ids)
        score_sql = "NULL" if bm25_score is None else str(float(bm25_score))
        rows = self.conn.execute(
            f"""
            SELECT
                ste_catalog.rowid AS row_id,
                ste_catalog.ste_id,
                ste_catalog.clean_name,
                ste_catalog.normalized_name,
                ste_catalog.category,
                ste_catalog.normalized_category,
                ste_catalog.attribute_keys,
                ste_catalog.attribute_count,
                ste_catalog.key_tokens,
                {score_sql} AS bm25_score
            FROM ste_catalog
            WHERE ste_catalog.rowid IN ({placeholders})
            """,
            tuple(row_ids),
        ).fetchall()
        rows_by_id = {int(row["row_id"]): dict(row) for row in rows}
        return [rows_by_id[row_id] for row_id in row_ids if row_id in rows_by_id]

    @staticmethod
    def _semantic_candidate_limit(analysis: QueryAnalysis, candidate_limit: int, lexical_hits: int) -> int:
        if not analysis.semantic_expansions:
            return 0
        if lexical_hits == 0:
            return max(candidate_limit, 80)
        if len(analysis.corrected_tokens) <= 1:
            return min(max(40, candidate_limit // 2), 120)
        return min(max(30, candidate_limit // 3), 80)

    @staticmethod
    def _dense_candidate_limit(analysis: QueryAnalysis, candidate_limit: int, lexical_hits: int) -> int:
        if lexical_hits == 0:
            return min(max(80, candidate_limit), 160)
        # Dense backfill helps when lexical retrieval is weak or when the query
        # already carries explicit semantic hints. It should not flood simple
        # single-token lexical queries with loosely related items.
        if (
            len(analysis.corrected_tokens) == 1
            and not analysis.semantic_expansions
            and not analysis.applied_synonyms
            and lexical_hits > 0
        ):
            return 0
        if analysis.semantic_expansions or analysis.applied_synonyms:
            return min(max(24, candidate_limit // 3), 80)
        return min(max(16, candidate_limit // 4), 48)

    def _fetch_candidates(self, analysis: QueryAnalysis, candidate_limit: int = 250) -> List[Dict[str, object]]:
        lexical_query = self._build_match_query_from_tokens(
            analysis.corrected_tokens,
            stem_tokens(analysis.lexical_expanded_tokens),
        )
        lexical_rows = self._fetch_match_candidates(lexical_query, candidate_limit)

        semantic_query = self._build_match_query_from_tokens([], stem_tokens(analysis.semantic_expansions))
        semantic_limit = self._semantic_candidate_limit(analysis, candidate_limit, len(lexical_rows))
        semantic_rows = self._fetch_match_candidates(semantic_query, semantic_limit) if semantic_limit else []
        dense_rows: List[Dict[str, object]] = []
        if self.dense_retriever.enabled:
            dense_limit = self._dense_candidate_limit(analysis, candidate_limit, len(lexical_rows))
            dense_hits = self.dense_retriever.search(analysis, top_k=min(self.dense_top_k, dense_limit)) if dense_limit else []
            dense_row_ids = [row_id for row_id, _score in dense_hits]
            dense_rows = self._fetch_rows_by_ids(dense_row_ids, bm25_score=1_000_000.0)

        merged_rows: List[sqlite3.Row] = []
        seen_row_ids = set()
        for row in lexical_rows + semantic_rows + dense_rows:
            row_id = int(row["row_id"])
            if row_id in seen_row_ids:
                continue
            seen_row_ids.add(row_id)
            merged_rows.append(row)
        return merged_rows

    @staticmethod
    def _needs_broader_retrieval(analysis: QueryAnalysis) -> bool:
        if analysis.applied_completions:
            return True
        return any(len(token) <= 4 and not token.isdigit() for token in analysis.corrected_tokens)

    @staticmethod
    def _compute_attribute_score(row: sqlite3.Row, analysis: QueryAnalysis) -> float:
        """Score bonus/penalty based on structured attribute matching.

        Strategy:
        - For each (value, unit) pair from the query:
            - Check the candidate's normalized_name + attribute_keys for the
              pair and for *any* other value with the same unit.
            - +5.0 exact value+unit match in name/keys
            - +2.0 value appears anywhere in name (unit match not needed)
            - -3.0 same unit present but a *different* numeric value is adjacent
              (e.g. query asks 500мг but candidate says 250мг)
        - Cap total penalty to avoid burying otherwise-good candidates.
        """
        if not analysis.attribute_spans:
            return 0.0

        candidate_text = " ".join(filter(None, [
            row["normalized_name"] or "",
            row["attribute_keys"] or "",
        ])).lower()

        total = 0.0
        for value, unit in analysis.attribute_spans:
            # Build pattern to find unit with its adjacent number in the candidate
            unit_re = re.compile(
                r"(\d+(?:[.,]\d+)?)\s*" + re.escape(unit) + r"\b",
                re.IGNORECASE,
            )
            matches_in_candidate = unit_re.findall(candidate_text)

            if matches_in_candidate:
                # Normalise candidate values (replace comma decimal separator)
                candidate_values = [v.replace(",", ".") for v in matches_in_candidate]
                if value in candidate_values:
                    # Exact match — strong boost
                    total += 5.0
                else:
                    # Same unit, different value — penalise
                    total -= 3.0
            else:
                # Unit not found in candidate — check whether the bare value is
                # present (could be written differently)
                if re.search(r"\b" + re.escape(value) + r"\b", candidate_text):
                    total += 2.0

        # Never let attribute penalty alone push score to extremely negative
        return max(total, -6.0)

    def _score_candidate(
        self,
        row: Dict[str, object] | sqlite3.Row,
        analysis: QueryAnalysis,
        dense_similarity: float = 0.0,
    ) -> tuple[float, Dict[str, float]]:
        name_tokens = set(tokenize(row["normalized_name"]))
        category_tokens = set(tokenize(row["normalized_category"]))
        key_tokens = set(tokenize(row["key_tokens"]))

        name_stems = set(stem_tokens(name_tokens))
        category_stems = set(stem_tokens(category_tokens))
        key_stems = set(stem_tokens(key_tokens))

        corrected_set = set(analysis.corrected_tokens)
        stem_set = set(analysis.stemmed_tokens)
        expanded_stem_set = set(analysis.expanded_stems)
        semantic_stem_set = set(stem_tokens(analysis.semantic_expansions))

        query_target = analysis.corrected_query or analysis.normalized_query
        exact_phrase = 1.0 if query_target and query_target in row["normalized_name"] else 0.0
        corrected_hits = len(corrected_set & name_tokens)
        stem_hits_name = len(stem_set & name_stems)
        stem_hits_category = len(expanded_stem_set & category_stems)
        stem_hits_key = len(expanded_stem_set & key_stems)
        semantic_hits_name = len(semantic_stem_set & name_stems)
        semantic_hits_category = len(semantic_stem_set & category_stems)
        semantic_hits_key = len(semantic_stem_set & key_stems)
        full_name_cover = 1.0 if stem_set and stem_set.issubset(name_stems) else 0.0
        full_category_cover = 1.0 if stem_set and stem_set.issubset(category_stems) else 0.0
        synonym_bonus = 1.0 if analysis.applied_synonyms and stem_hits_name + stem_hits_category > 0 else 0.0
        semantic_query = analysis.corrected_query or analysis.normalized_query
        semantic_vector_similarity = self.semantic_expander.sentence_similarity(
            semantic_query,
            f"{row['normalized_name']} {row['normalized_category']} {row['key_tokens']}",
        )
        attribute_score = self._compute_attribute_score(row, analysis)

        coverage_denominator = max(1, len(corrected_set))
        stem_denominator = max(1, len(stem_set))
        expanded_denominator = max(1, len(expanded_stem_set))
        semantic_denominator = max(1, len(semantic_stem_set))

        bm25_score = row["bm25_score"]
        bm25_component = 0.0 if bm25_score is None else 1.0 / (1.0 + max(0.0, float(bm25_score)))

        score = 0.0
        score += 12.0 * exact_phrase
        score += 4.0 * full_name_cover
        score += 6.0 * full_category_cover
        score += 6.0 * (corrected_hits / coverage_denominator)
        score += 4.0 * (stem_hits_name / stem_denominator)
        score += 3.0 * (stem_hits_category / expanded_denominator)
        score += 2.0 * (stem_hits_key / expanded_denominator)
        score += 1.75 * (semantic_hits_name / semantic_denominator)
        score += 1.25 * (semantic_hits_category / semantic_denominator)
        score += 1.0 * (semantic_hits_key / semantic_denominator)
        score += 1.5 * synonym_bonus
        score += 3.0 * semantic_vector_similarity
        score += 3.5 * dense_similarity
        score += 2.0 * bm25_component
        score += attribute_score  # structured attribute bonus/penalty

        features = {
            "exact_phrase": round(exact_phrase, 4),
            "full_name_cover": round(full_name_cover, 4),
            "full_category_cover": round(full_category_cover, 4),
            "corrected_token_overlap": round(corrected_hits / coverage_denominator, 4),
            "name_stem_overlap": round(stem_hits_name / stem_denominator, 4),
            "category_stem_overlap": round(stem_hits_category / expanded_denominator, 4),
            "key_token_overlap": round(stem_hits_key / expanded_denominator, 4),
            "semantic_name_overlap": round(semantic_hits_name / semantic_denominator, 4),
            "semantic_category_overlap": round(semantic_hits_category / semantic_denominator, 4),
            "semantic_key_overlap": round(semantic_hits_key / semantic_denominator, 4),
            "semantic_vector_similarity": round(semantic_vector_similarity, 4),
            "dense_vector_similarity": round(dense_similarity, 4),
            "synonym_bonus": round(synonym_bonus, 4),
            "bm25_component": round(bm25_component, 4),
            "attribute_score": round(attribute_score, 4),
        }
        return score, features

    @staticmethod
    def _semantic_score(item: Dict[str, object]) -> float:
        search_features = dict(item.get("search_features") or {})
        return max(
            float(search_features.get("semantic_vector_similarity", 0.0) or 0.0),
            float(search_features.get("semantic_name_overlap", 0.0) or 0.0),
            float(search_features.get("semantic_category_overlap", 0.0) or 0.0),
            float(search_features.get("semantic_key_overlap", 0.0) or 0.0),
            float(search_features.get("dense_vector_similarity", 0.0) or 0.0),
        )

    @classmethod
    def _passes_min_score(cls, item: Dict[str, object], min_score: float) -> bool:
        search_features = dict(item.get("search_features") or {})
        semantic_score = cls._semantic_score(item)
        corrected_token_overlap = float(search_features.get("corrected_token_overlap", 0.0) or 0.0)
        name_stem_overlap = float(search_features.get("name_stem_overlap", 0.0) or 0.0)
        category_stem_overlap = float(search_features.get("category_stem_overlap", 0.0) or 0.0)
        key_token_overlap = float(search_features.get("key_token_overlap", 0.0) or 0.0)
        exact_lexical_match = max(
            float(search_features.get("exact_phrase", 0.0) or 0.0),
            float(search_features.get("full_name_cover", 0.0) or 0.0),
            float(search_features.get("full_category_cover", 0.0) or 0.0),
            corrected_token_overlap,
        )
        blended_lexical_match = max(
            exact_lexical_match,
            0.45 * corrected_token_overlap
            + 0.25 * name_stem_overlap
            + 0.15 * category_stem_overlap
            + 0.15 * key_token_overlap,
        )
        # We cut low-confidence semantic noise, but we keep exact lexical hits:
        # model numbers, article codes and short catalog queries are often valid
        # even when the semantic backend gives them a low vector similarity.
        return semantic_score >= min_score or exact_lexical_match >= 1.0 or blended_lexical_match >= max(0.52, min_score * 0.95)

    def search(
        self,
        query: str,
        top_k: int = 20,
        candidate_limit: int = 250,
        limit: int | None = None,
        offset: int = 0,
        min_score: float = 0.55,
    ) -> Dict[str, object]:
        analysis = self.analyze_query(query)
        retrieval_limit = max(candidate_limit, 400) if self._needs_broader_retrieval(analysis) else candidate_limit
        candidates = self._fetch_candidates(analysis, candidate_limit=retrieval_limit)
        dense_scores = self.dense_retriever.score_row_ids(
            analysis,
            [int(row["row_id"]) for row in candidates],
        ) if self.dense_retriever.enabled else {}
        scored_results: List[Dict[str, object]] = []
        for row in candidates:
            lexical_score, features = self._score_candidate(
                row,
                analysis,
                dense_similarity=float(dense_scores.get(int(row["row_id"]), 0.0)),
            )
            scored_results.append(
                {
                    "ste_id": row["ste_id"],
                    "clean_name": row["clean_name"],
                    "normalized_name": row["normalized_name"],
                    "category": row["category"],
                    "normalized_category": row["normalized_category"],
                    "attribute_keys": row["attribute_keys"],
                    "attribute_count": int(row["attribute_count"] or 0),
                    "key_tokens": row["key_tokens"],
                    "semantic_score": round(
                        max(
                            float(features.get("semantic_vector_similarity", 0.0) or 0.0),
                            float(features.get("semantic_name_overlap", 0.0) or 0.0),
                            float(features.get("semantic_category_overlap", 0.0) or 0.0),
                            float(features.get("semantic_key_overlap", 0.0) or 0.0),
                            float(features.get("dense_vector_similarity", 0.0) or 0.0),
                        ),
                        4,
                    ),
                    "search_score": round(lexical_score, 4),
                    "search_features": features,
                }
            )

        scored_results.sort(
            key=lambda item: (
                item["search_score"],
                item["search_features"]["exact_phrase"],
                item["search_features"]["corrected_token_overlap"],
                item["search_features"]["category_stem_overlap"],
            ),
            reverse=True,
        )
        filtered_results = [item for item in scored_results if self._passes_min_score(item, min_score)]
        page_limit = max(1, limit or top_k)
        paginated_results = filtered_results[offset : offset + page_limit]
        return {
            "query": {
                "original_query": analysis.original_query,
                "normalized_query": analysis.normalized_query,
                "corrected_query": analysis.corrected_query,
                "applied_corrections": analysis.applied_corrections,
                "applied_completions": analysis.applied_completions,
                "applied_synonyms": analysis.applied_synonyms,
                "applied_semantic_neighbors": analysis.applied_semantic_neighbors,
                "semantic_backend": self.semantic_expander.backend_name,
                "dense_retrieval_enabled": self.dense_retriever.enabled,
                "dense_retrieval_backend": getattr(self.dense_retriever, "backend_name", "graph"),
                "expanded_tokens": analysis.expanded_tokens,
                "completion_expansions": analysis.completion_expansions,
                "semantic_expansions": analysis.semantic_expansions,
            },
            "results": paginated_results,
            "total_found": len(filtered_results),
            "has_more": offset + page_limit < len(filtered_results),
        }

    def search_ste(self, query: str, top_k: int = 20, min_score: float = 0.55) -> List[Dict[str, object]]:
        return self.search(query, top_k=top_k, min_score=min_score)["results"]


def search_ste(
    query: str,
    top_k: int = 20,
    min_score: float = 0.55,
    search_db_path: Path | str = DEFAULT_SEARCH_DB,
    synonyms_path: Path | str = DEFAULT_SYNONYMS_PATH,
    semantic_backend: str = "auto",
    fasttext_model_path: Path | str = DEFAULT_FASTTEXT_MODEL_PATH,
) -> List[Dict[str, object]]:
    service = SearchService(
        search_db_path=search_db_path,
        synonyms_path=synonyms_path,
        semantic_backend=semantic_backend,
        fasttext_model_path=fasttext_model_path,
    )
    try:
        return service.search_ste(query=query, top_k=top_k, min_score=min_score)
    finally:
        service.close()
