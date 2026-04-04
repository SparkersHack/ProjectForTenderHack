from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .rerank_dataset import build_rerank_row

try:
    from catboost import CatBoostRanker

    CATBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    CatBoostRanker = None  # type: ignore[assignment]
    CATBOOST_AVAILABLE = False


DEFAULT_SEARCH_RERANK_MODEL_PATH = Path("data/processed/tenderhack_yeti_ranker.cbm")
DEFAULT_SEARCH_RERANK_METADATA_PATH = Path("data/processed/tenderhack_yeti_ranker.json")


class SearchRerankPredictor:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_SEARCH_RERANK_MODEL_PATH,
        metadata_path: Path | str = DEFAULT_SEARCH_RERANK_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model = None
        self.feature_order: List[str] = []

        if self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.feature_order = [str(name) for name in metadata.get("feature_names", []) if str(name)]

        if CATBOOST_AVAILABLE and self.model_path.exists() and self.feature_order:
            self.model = CatBoostRanker()
            self.model.load_model(str(self.model_path))

    @property
    def enabled(self) -> bool:
        return self.model is not None and bool(self.feature_order)

    def rerank_candidates(
        self,
        *,
        query: str,
        query_meta: Dict[str, object],
        candidates: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        if not candidates or not self.enabled:
            return list(candidates)

        feature_matrix: List[List[float]] = []
        for rank, candidate in enumerate(candidates, start=1):
            row = build_rerank_row(
                group_id="inference",
                query=query,
                query_meta=query_meta,
                contract_id="",
                customer_inn="",
                customer_region="",
                positive_ste_id="",
                candidate=candidate,
                candidate_rank=rank,
            )
            feature_matrix.append([float(row.get(name, 0.0) or 0.0) for name in self.feature_order])

        predictions = self.model.predict(feature_matrix)
        if not isinstance(predictions, list):
            predictions = list(predictions)

        reranked: List[Dict[str, object]] = []
        for candidate, prediction in zip(candidates, predictions):
            enriched = dict(candidate)
            original_search_score = float(enriched.get("search_score", 0.0) or 0.0)
            ml_score = float(prediction)
            enriched["retrieval_score"] = round(original_search_score, 6)
            enriched["ml_rerank_score"] = round(ml_score, 6)
            enriched["search_score"] = round(ml_score, 6)
            reranked.append(enriched)

        reranked.sort(
            key=lambda item: (
                float(item.get("ml_rerank_score", item.get("search_score", 0.0))),
                float(item.get("retrieval_score", 0.0)),
            ),
            reverse=True,
        )
        return reranked


def rerank_search_candidates(
    *,
    query: str,
    query_meta: Dict[str, object],
    candidates: List[Dict[str, object]],
    model_path: Path | str = DEFAULT_SEARCH_RERANK_MODEL_PATH,
    metadata_path: Path | str = DEFAULT_SEARCH_RERANK_METADATA_PATH,
) -> List[Dict[str, object]]:
    predictor = SearchRerankPredictor(model_path=model_path, metadata_path=metadata_path)
    return predictor.rerank_candidates(query=query, query_meta=query_meta, candidates=candidates)
