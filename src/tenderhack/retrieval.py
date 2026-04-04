from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .query_understanding import QueryAnalysis
from .text import unique_preserve_order


@dataclass
class RetrievalStrategy:
    name: str
    tokens: List[str]
    stems: List[str]
    phrase: str
    limit: int


class CandidateRetriever:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def retrieve(self, analysis: QueryAnalysis, candidate_limit: int = 250) -> List[Dict[str, object]]:
        strategies = self._build_strategies(analysis, candidate_limit)
        merged: Dict[str, Dict[str, object]] = {}

        for strategy in strategies:
            match_query = self._build_match_query(
                tokens=strategy.tokens,
                stems=strategy.stems,
                phrase=strategy.phrase,
            )
            if not match_query:
                continue
            rows = self._run_match_query(match_query=match_query, limit=strategy.limit)
            for row in rows:
                self._merge_candidate(merged, row, strategy.name)

        candidates = list(merged.values())
        candidates.sort(
            key=lambda item: (
                item["retrieval_features"]["best_bm25_component"],
                item["retrieval_features"]["source_count"],
                item["retrieval_features"]["source_priority"],
            ),
            reverse=True,
        )
        return candidates[:candidate_limit]

    def _build_strategies(self, analysis: QueryAnalysis, candidate_limit: int) -> List[RetrievalStrategy]:
        strategies: List[RetrievalStrategy] = []
        corrected_phrase = analysis.corrected_query or analysis.normalized_query

        if analysis.corrected_tokens:
            strategies.append(
                RetrievalStrategy(
                    name="precise",
                    tokens=unique_preserve_order(analysis.corrected_tokens + analysis.entity_tokens),
                    stems=unique_preserve_order(analysis.stemmed_tokens + analysis.entity_stems),
                    phrase=corrected_phrase,
                    limit=min(candidate_limit, max(80, candidate_limit // 2)),
                )
            )

        if analysis.expanded_tokens:
            strategies.append(
                RetrievalStrategy(
                    name="expanded",
                    tokens=analysis.expanded_tokens,
                    stems=analysis.expanded_stems,
                    phrase="",
                    limit=min(candidate_limit, max(120, int(candidate_limit * 0.75))),
                )
            )

        if analysis.entity_tokens:
            strategies.append(
                RetrievalStrategy(
                    name="entities",
                    tokens=analysis.entity_tokens,
                    stems=analysis.entity_stems,
                    phrase=" ".join(analysis.entity_tokens) if len(analysis.entity_tokens) > 1 else "",
                    limit=min(candidate_limit, max(60, candidate_limit // 3)),
                )
            )

        if analysis.original_tokens and analysis.original_tokens != analysis.corrected_tokens:
            strategies.append(
                RetrievalStrategy(
                    name="fallback",
                    tokens=analysis.original_tokens,
                    stems=[],
                    phrase=analysis.normalized_query,
                    limit=min(candidate_limit, max(40, candidate_limit // 4)),
                )
            )

        return strategies

    def _build_match_query(self, tokens: Iterable[str], stems: Iterable[str], phrase: str) -> str:
        terms: List[str] = []

        normalized_phrase = " ".join(token for token in phrase.split() if token)
        if normalized_phrase and len(normalized_phrase.split()) >= 2:
            terms.append(f'"{normalized_phrase}"')

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

    def _run_match_query(self, match_query: str, limit: int) -> List[sqlite3.Row]:
        return self.conn.execute(
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
            (match_query, limit),
        ).fetchall()

    def _merge_candidate(self, merged: Dict[str, Dict[str, object]], row: sqlite3.Row, source: str) -> None:
        ste_id = row["ste_id"]
        bm25_score = float(row["bm25_score"] if row["bm25_score"] is not None else 0.0)
        bm25_component = self._bm25_component(bm25_score)
        source_priority = self._source_priority(source)

        existing = merged.get(ste_id)
        if existing is None:
            merged[ste_id] = {
                "row_id": row["row_id"],
                "ste_id": ste_id,
                "clean_name": row["clean_name"],
                "normalized_name": row["normalized_name"],
                "category": row["category"],
                "normalized_category": row["normalized_category"],
                "attribute_keys": row["attribute_keys"],
                "attribute_count": int(row["attribute_count"] or 0),
                "key_tokens": row["key_tokens"],
                "bm25_score": bm25_score,
                "retrieval_sources": [source],
                "retrieval_features": {
                    f"{source}_bm25_component": round(bm25_component, 4),
                    "best_bm25_component": round(bm25_component, 4),
                    "source_count": 1,
                    "source_priority": source_priority,
                },
            }
            return

        existing["bm25_score"] = min(float(existing.get("bm25_score", 0.0)), bm25_score)
        retrieval_sources = set(existing.get("retrieval_sources", []))
        retrieval_sources.add(source)
        existing["retrieval_sources"] = sorted(retrieval_sources)

        retrieval_features = dict(existing.get("retrieval_features", {}))
        retrieval_features[f"{source}_bm25_component"] = round(bm25_component, 4)
        retrieval_features["best_bm25_component"] = round(
            max(float(retrieval_features.get("best_bm25_component", 0.0)), bm25_component),
            4,
        )
        retrieval_features["source_count"] = len(retrieval_sources)
        retrieval_features["source_priority"] = max(int(retrieval_features.get("source_priority", 0)), source_priority)
        existing["retrieval_features"] = retrieval_features

    @staticmethod
    def _bm25_component(bm25_score: float) -> float:
        return 1.0 / (1.0 + max(0.0, bm25_score))

    @staticmethod
    def _source_priority(source: str) -> int:
        priorities = {
            "precise": 4,
            "entities": 3,
            "expanded": 2,
            "fallback": 1,
        }
        return priorities.get(source, 0)
