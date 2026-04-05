# Offline Evaluation

- Status: `partial`
- Selected pseudo-query variant: `contract_item_name`

## Ranking Dataset

- processed_positive_events: `2009457`
- emitted_positive_events: `60000`
- skipped_missing_catalog_match: `3824`
- skipped_empty_query: `0`
- rows_total: `2100000`
- groups_total: `300000`
- rows_by_split: `{'train': 2100000}`

## Pseudo-Query Benchmark

- `contract_item_name`: NDCG@10=0.0, NDCG@5=0.0
- `contract_plus_category`: NDCG@10=0.0, NDCG@5=0.0
- `ste_name`: NDCG@10=0.0, NDCG@5=0.0
- `ste_name_category_attributes`: NDCG@10=0.0, NDCG@5=0.0
- `ste_name_plus_category`: NDCG@10=0.0, NDCG@5=0.0

## Model Comparison

### baseline_non_personalized

- overall: NDCG@5=0.0, NDCG@10=0.0, MRR@10=0.0, Recall@10=0.0, HitRate@10=0.0
- new_users: NDCG@5=0.0, NDCG@10=0.0, MRR@10=0.0, Recall@10=0.0, HitRate@10=0.0
- active_users: NDCG@5=0.0, NDCG@10=0.0, MRR@10=0.0, Recall@10=0.0, HitRate@10=0.0
- frequent_categories: NDCG@5=0.0, NDCG@10=0.0, MRR@10=0.0, Recall@10=0.0, HitRate@10=0.0
- rare_categories: NDCG@5=0.0, NDCG@10=0.0, MRR@10=0.0, Recall@10=0.0, HitRate@10=0.0

### baseline_rule_based

- overall: NDCG@5=0.0, NDCG@10=0.0, MRR@10=0.0, Recall@10=0.0, HitRate@10=0.0
- new_users: NDCG@5=0.0, NDCG@10=0.0, MRR@10=0.0, Recall@10=0.0, HitRate@10=0.0
- active_users: NDCG@5=0.0, NDCG@10=0.0, MRR@10=0.0, Recall@10=0.0, HitRate@10=0.0
- frequent_categories: NDCG@5=0.0, NDCG@10=0.0, MRR@10=0.0, Recall@10=0.0, HitRate@10=0.0
- rare_categories: NDCG@5=0.0, NDCG@10=0.0, MRR@10=0.0, Recall@10=0.0, HitRate@10=0.0

## Global Feature Importance


## Integration Contract

- Input: `user_id`, `query_features`, `candidates`, `user_profile`.
- Output: `candidate_id`, `personalization_score`, `top_reason_codes`, `reasons`.
- Stable inference entrypoint: `predict_personalization(candidates, user_profile, query_features)`.

## Notes

- CatBoost недоступен в текущем окружении: ModuleNotFoundError: No module named 'catboost'
