from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

from .dataset_layout import resolve_search_db_path
from .query_understanding import DEFAULT_SYNONYMS_PATH, QueryAnalysis, QueryUnderstandingService
from .retrieval import CandidateRetriever
from .semantic import DEFAULT_FASTTEXT_MODEL_PATH
from .text import stem_tokens, tokenize


DEFAULT_SEARCH_DB = resolve_search_db_path()


class SearchService:
    def __init__(
        self,
        search_db_path: Path | str = DEFAULT_SEARCH_DB,
        synonyms_path: Path | str = DEFAULT_SYNONYMS_PATH,
        semantic_top_n: int = 4,
        semantic_backend: str = "auto",
        fasttext_model_path: Path | str = DEFAULT_FASTTEXT_MODEL_PATH,
        fasttext_similarity_threshold: float = 0.55,
    ) -> None:
        self.search_db_path = Path(search_db_path)
        self.synonyms_path = Path(synonyms_path)
        self.conn = sqlite3.connect(self.search_db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.query_understanding = QueryUnderstandingService(
            self.conn,
            synonyms_path=self.synonyms_path,
            semantic_top_n=semantic_top_n,
            semantic_backend=semantic_backend,
            fasttext_model_path=fasttext_model_path,
            fasttext_similarity_threshold=fasttext_similarity_threshold,
        )
        self.candidate_retriever = CandidateRetriever(self.conn)

    def close(self) -> None:
        self.conn.close()

    def analyze_query(self, query: str) -> QueryAnalysis:
        return self.query_understanding.analyze_query(query)

    def get_ste_item(self, ste_id: str) -> Dict[str, object] | None:
        row = self.conn.execute(
            """
            SELECT
                ste_id,
                clean_name,
                normalized_name,
                category,
                normalized_category,
                attribute_keys,
                attribute_count,
                key_tokens
            FROM ste_catalog
            WHERE ste_id = ?
            LIMIT 1
            """,
            (str(ste_id),),
        ).fetchone()
        if row is None:
            return None
        return {
            "ste_id": row["ste_id"],
            "clean_name": row["clean_name"],
            "normalized_name": row["normalized_name"],
            "category": row["category"],
            "normalized_category": row["normalized_category"],
            "attribute_keys": row["attribute_keys"],
            "attribute_count": int(row["attribute_count"] or 0),
            "key_tokens": row["key_tokens"],
        }

    def _fetch_candidates(self, analysis: QueryAnalysis, candidate_limit: int = 250) -> List[Dict[str, object]]:
        return self.candidate_retriever.retrieve(analysis, candidate_limit=candidate_limit)

    def _score_candidate(self, row: Dict[str, object], analysis: QueryAnalysis) -> tuple[float, Dict[str, float]]:
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
        entity_hits = len(set(analysis.entity_stems) & (name_stems | category_stems | key_stems))
        semantic_query = analysis.corrected_query or analysis.normalized_query
        semantic_vector_similarity = self.query_understanding.semantic_expander.sentence_similarity(
            semantic_query,
            f"{row['normalized_name']} {row['normalized_category']} {row['key_tokens']}",
        )

        coverage_denominator = max(1, len(corrected_set))
        stem_denominator = max(1, len(stem_set))
        expanded_denominator = max(1, len(expanded_stem_set))
        semantic_denominator = max(1, len(semantic_stem_set))
        entity_denominator = max(1, len(analysis.entity_stems))

        bm25_score = row["bm25_score"] if row["bm25_score"] is not None else 0.0
        bm25_component = 1.0 / (1.0 + max(0.0, bm25_score))
        retrieval_sources = set(row.get("retrieval_sources", []))
        retrieval_features = row.get("retrieval_features", {})
        precise_retrieval_match = 1.0 if "precise" in retrieval_sources else 0.0
        entity_retrieval_match = 1.0 if "entities" in retrieval_sources else 0.0
        multi_source_bonus = min(float(retrieval_features.get("source_count", 1)), 3.0) / 3.0

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
        score += 2.0 * (entity_hits / entity_denominator)
        score += 3.0 * semantic_vector_similarity
        score += 2.0 * bm25_component
        score += 1.0 * precise_retrieval_match
        score += 0.75 * entity_retrieval_match
        score += 0.5 * multi_source_bonus

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
            "synonym_bonus": round(synonym_bonus, 4),
            "entity_overlap": round(entity_hits / entity_denominator, 4),
            "bm25_component": round(bm25_component, 4),
            "precise_retrieval_match": round(precise_retrieval_match, 4),
            "entity_retrieval_match": round(entity_retrieval_match, 4),
            "multi_source_bonus": round(multi_source_bonus, 4),
        }
        return score, features

    def search(self, query: str, top_k: int = 20, candidate_limit: int = 250) -> Dict[str, object]:
        analysis = self.analyze_query(query)
        candidates = self._fetch_candidates(analysis, candidate_limit=candidate_limit)
        scored_results: List[Dict[str, object]] = []
        for row in candidates:
            lexical_score, features = self._score_candidate(row, analysis)
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
                    "search_score": round(lexical_score, 4),
                    "search_features": features,
                    "retrieval_sources": row.get("retrieval_sources", []),
                    "retrieval_features": row.get("retrieval_features", {}),
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
        return {
            "query": {
                "original_query": analysis.original_query,
                "normalized_query": analysis.normalized_query,
                "corrected_query": analysis.corrected_query,
                "applied_corrections": analysis.applied_corrections,
                "applied_synonyms": analysis.applied_synonyms,
                "applied_semantic_neighbors": analysis.applied_semantic_neighbors,
                "semantic_backend": self.query_understanding.semantic_expander.backend_name,
                "expanded_tokens": analysis.expanded_tokens,
                "semantic_expansions": analysis.semantic_expansions,
                "entities": [entity.to_mapping() for entity in analysis.entities],
                "entity_tokens": analysis.entity_tokens,
            },
            "results": scored_results[:top_k],
        }

    def search_ste(self, query: str, top_k: int = 20) -> List[Dict[str, object]]:
        return self.search(query, top_k=top_k)["results"]


def search_ste(
    query: str,
    top_k: int = 20,
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
        return service.search_ste(query=query, top_k=top_k)
    finally:
        service.close()
