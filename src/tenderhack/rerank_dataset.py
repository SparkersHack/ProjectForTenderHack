from __future__ import annotations

from typing import Dict, List

from .text import normalize_text, tokenize


NON_FEATURE_COLUMNS = {
    "group_id",
    "query",
    "normalized_query",
    "corrected_query",
    "contract_id",
    "customer_inn",
    "customer_region",
    "positive_ste_id",
    "candidate_ste_id",
    "candidate_name",
    "candidate_category",
    "label",
}


def build_rerank_row(
    *,
    group_id: str,
    query: str,
    query_meta: Dict[str, object],
    contract_id: str,
    customer_inn: str,
    customer_region: str,
    positive_ste_id: str,
    candidate: Dict[str, object],
    candidate_rank: int,
) -> Dict[str, object]:
    normalized_query = str(query_meta.get("normalized_query") or "")
    corrected_query = str(query_meta.get("corrected_query") or "")
    expanded_tokens = list(query_meta.get("expanded_tokens") or [])
    applied_corrections = list(query_meta.get("applied_corrections") or [])
    applied_synonyms = list(query_meta.get("applied_synonyms") or [])
    applied_semantic_neighbors = list(query_meta.get("applied_semantic_neighbors") or [])
    semantic_backend = str(query_meta.get("semantic_backend") or "none")

    clean_name = str(candidate.get("clean_name") or candidate.get("normalized_name") or "")
    category = str(candidate.get("category") or "")
    key_tokens = str(candidate.get("key_tokens") or "")
    search_features = dict(candidate.get("search_features") or {})

    row = {
        "group_id": group_id,
        "query": query,
        "normalized_query": normalized_query,
        "corrected_query": corrected_query,
        "contract_id": contract_id,
        "customer_inn": customer_inn,
        "customer_region": customer_region,
        "positive_ste_id": positive_ste_id,
        "candidate_ste_id": str(candidate.get("ste_id") or ""),
        "candidate_name": clean_name,
        "candidate_category": category,
        "label": 1 if str(candidate.get("ste_id") or "") == positive_ste_id else 0,
        "candidate_rank": candidate_rank,
        "candidate_reciprocal_rank": round(1.0 / max(candidate_rank, 1), 6),
        "search_score": round(float(candidate.get("search_score") or 0.0), 6),
        "attribute_count": int(candidate.get("attribute_count") or 0),
        "query_token_count": len(tokenize(normalized_query)),
        "query_has_digits": 1 if any(char.isdigit() for char in normalized_query) else 0,
        "query_changed_by_correction": 1 if corrected_query and corrected_query != normalized_query else 0,
        "applied_correction_count": len(applied_corrections),
        "applied_synonym_count": len(applied_synonyms),
        "applied_semantic_count": len(applied_semantic_neighbors),
        "expanded_token_count": len(expanded_tokens),
        "semantic_backend_sqlite": 1 if semantic_backend == "sqlite" else 0,
        "semantic_backend_fasttext": 1 if semantic_backend == "fasttext" else 0,
        "candidate_name_token_count": len(tokenize(clean_name)),
        "candidate_category_token_count": len(tokenize(category)),
        "candidate_key_token_count": len(tokenize(key_tokens)),
        "candidate_name_char_count": len(clean_name),
        "candidate_category_char_count": len(category),
        "query_name_jaccard": _token_jaccard(normalized_query, normalize_text(clean_name)),
        "query_category_jaccard": _token_jaccard(normalized_query, normalize_text(category)),
    }

    for feature_name, feature_value in search_features.items():
        row[feature_name] = round(float(feature_value or 0.0), 6)
    return row


def infer_feature_columns(fieldnames: List[str]) -> List[str]:
    return [name for name in fieldnames if name not in NON_FEATURE_COLUMNS]


def _token_jaccard(left_text: str, right_text: str) -> float:
    left_tokens = set(tokenize(left_text))
    right_tokens = set(tokenize(right_text))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    if union == 0:
        return 0.0
    return round(intersection / union, 6)
