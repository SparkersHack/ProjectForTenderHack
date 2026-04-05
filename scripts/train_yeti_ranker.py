#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval.ranking_metrics import evaluate_grouped_rows
from tenderhack.rerank_dataset import infer_feature_columns

try:
    from catboost import CatBoostRanker, Pool
except Exception as exc:  # pragma: no cover - optional runtime dependency
    CatBoostRanker = None  # type: ignore[assignment]
    Pool = None  # type: ignore[assignment]
    CATBOOST_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    CATBOOST_IMPORT_ERROR = ""


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

DEFAULT_DATASET_PATH = "data/processed/rerank_train_current.csv"
DEFAULT_MODEL_PATH = "data/processed/tenderhack_yeti_ranker_current.cbm"
DEFAULT_METADATA_PATH = "data/processed/tenderhack_yeti_ranker_current.json"


def _resolve_output_path(path_value: str, *, loss_function: str, default_value: str, suffix: str) -> Path:
    path = Path(path_value)
    if loss_function != "YetiRankPairwise" or path_value != default_value:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _require_dataset_path(dataset_path: Path) -> None:
    if dataset_path.exists():
        return

    default_dataset = PROJECT_ROOT / DEFAULT_DATASET_PATH
    build_hint = (
        "Dataset for rerank training was not found.\n"
        f"Expected: {dataset_path}\n"
        "Build it first, for example:\n"
        "  venv/bin/python scripts/build_rerank_dataset.py "
        "--contracts-path Контракты_20260403.csv "
        "--output-path data/processed/rerank_train_current.csv\n"
    )
    if dataset_path == Path(DEFAULT_DATASET_PATH) or dataset_path == default_dataset:
        raise FileNotFoundError(build_hint)
    raise FileNotFoundError(f"{build_hint}Or pass an existing CSV via --dataset-path.")


def _load_rows(dataset_path: Path) -> tuple[List[Dict[str, object]], List[str]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        feature_names = infer_feature_columns(fieldnames)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed = dict(row)
            parsed["label"] = float(row["label"])
            for feature_name in feature_names:
                parsed[feature_name] = float(row.get(feature_name, 0.0) or 0.0)
            rows.append(parsed)
    return rows, feature_names


def _split_group_ids(group_ids: Sequence[str], valid_fraction: float, test_fraction: float, seed: int) -> dict[str, set[str]]:
    unique_group_ids = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_group_ids)

    total = len(unique_group_ids)
    test_count = int(total * test_fraction)
    valid_count = int(total * valid_fraction)
    if total >= 3 and test_count == 0 and test_fraction > 0:
        test_count = 1
    if total - test_count >= 2 and valid_count == 0 and valid_fraction > 0:
        valid_count = 1
    if valid_count + test_count >= total:
        overflow = valid_count + test_count - max(0, total - 1)
        if overflow > 0:
            if test_count >= overflow:
                test_count -= overflow
            else:
                valid_count = max(0, valid_count - (overflow - test_count))
                test_count = 0

    test_groups = set(unique_group_ids[:test_count])
    valid_groups = set(unique_group_ids[test_count : test_count + valid_count])
    train_groups = set(unique_group_ids[test_count + valid_count :])
    return {
        "train": train_groups,
        "valid": valid_groups,
        "test": test_groups,
    }


def _build_pool(rows: List[Dict[str, object]], feature_names: List[str]) -> Pool:
    matrices: List[List[float]] = []
    labels: List[float] = []
    group_ids: List[int] = []
    ordered_rows = sorted(rows, key=lambda item: (str(item["group_id"]), -float(item["label"]), str(item["candidate_ste_id"])))
    group_lookup: Dict[str, int] = {}
    for row in ordered_rows:
        group_id = str(row["group_id"])
        if group_id not in group_lookup:
            group_lookup[group_id] = len(group_lookup) + 1
        matrices.append([float(row.get(feature_name, 0.0) or 0.0) for feature_name in feature_names])
        labels.append(float(row["label"]))
        group_ids.append(group_lookup[group_id])
    return Pool(data=matrices, label=labels, group_id=group_ids)


def _evaluate_scored_rows(rows: Iterable[Dict[str, object]], score_key: str) -> dict[str, float]:
    payload = [
        {
            "group_id": row["group_id"],
            "label": row["label"],
            "score": row[score_key],
        }
        for row in rows
    ]
    return evaluate_grouped_rows(payload)


def train_yeti_ranker(
    *,
    dataset_path: Path,
    model_path: Path,
    metadata_path: Path,
    loss_function: str,
    iterations: int,
    depth: int,
    learning_rate: float,
    valid_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict:
    if CatBoostRanker is None or Pool is None:
        raise RuntimeError(f"CatBoost is not available: {CATBOOST_IMPORT_ERROR}")

    _require_dataset_path(dataset_path)
    rows, feature_names = _load_rows(dataset_path)
    splits = _split_group_ids(
        [str(row["group_id"]) for row in rows],
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
        seed=random_seed,
    )

    train_rows = [row for row in rows if str(row["group_id"]) in splits["train"]]
    valid_rows = [row for row in rows if str(row["group_id"]) in splits["valid"]]
    test_rows = [row for row in rows if str(row["group_id"]) in splits["test"]]

    if not train_rows or not valid_rows:
        raise ValueError("Not enough ranking groups for train/valid split.")

    model = CatBoostRanker(
        loss_function=loss_function,
        eval_metric="NDCG:top=10",
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=float(learning_rate),
        random_seed=int(random_seed),
        verbose=50,
    )
    train_pool = _build_pool(train_rows, feature_names)
    valid_pool = _build_pool(valid_rows, feature_names)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=50,
        verbose=50,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    def attach_prediction(target_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not target_rows:
            return []
        matrices = [[float(row.get(feature_name, 0.0) or 0.0) for feature_name in feature_names] for row in target_rows]
        predictions = model.predict(matrices)
        if not isinstance(predictions, list):
            predictions = list(predictions)
        enriched: List[Dict[str, object]] = []
        for row, prediction in zip(target_rows, predictions):
            payload = dict(row)
            payload["ml_score"] = float(prediction)
            payload["baseline_score"] = float(row.get("search_score", 0.0) or 0.0)
            enriched.append(payload)
        return enriched

    valid_scored = attach_prediction(valid_rows)
    test_scored = attach_prediction(test_rows)
    metrics = {
        "valid_baseline": _evaluate_scored_rows(valid_scored, "baseline_score"),
        "valid_ml": _evaluate_scored_rows(valid_scored, "ml_score"),
        "test_baseline": _evaluate_scored_rows(test_scored, "baseline_score"),
        "test_ml": _evaluate_scored_rows(test_scored, "ml_score"),
    }

    metadata = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "loss_function": loss_function,
        "feature_names": feature_names,
        "train_groups": len(splits["train"]),
        "valid_groups": len(splits["valid"]),
        "test_groups": len(splits["test"]),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "train_positives": sum(1 for row in train_rows if float(row["label"]) > 0),
        "valid_positives": sum(1 for row in valid_rows if float(row["label"]) > 0),
        "test_positives": sum(1 for row in test_rows if float(row["label"]) > 0),
        "best_iteration": int(model.get_best_iteration()),
        "best_score": model.get_best_score(),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost Yeti ranker on rerank dataset.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--loss-function", default="YetiRank", choices=["YetiRank", "YetiRankPairwise"])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    resolved_model_path = _resolve_output_path(
        args.model_path,
        loss_function=str(args.loss_function),
        default_value=DEFAULT_MODEL_PATH,
        suffix="_pairwise",
    )
    resolved_metadata_path = _resolve_output_path(
        args.metadata_path,
        loss_function=str(args.loss_function),
        default_value=DEFAULT_METADATA_PATH,
        suffix="_pairwise",
    )

    metadata = train_yeti_ranker(
        dataset_path=Path(args.dataset_path),
        model_path=resolved_model_path,
        metadata_path=resolved_metadata_path,
        loss_function=str(args.loss_function),
        iterations=int(args.iterations),
        depth=int(args.depth),
        learning_rate=float(args.learning_rate),
        valid_fraction=float(args.valid_fraction),
        test_fraction=float(args.test_fraction),
        random_seed=int(args.random_seed),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
