#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval.ranking_metrics import evaluate_grouped_rows
from tenderhack.rerank_dataset import infer_feature_columns

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover - optional runtime dependency
    lgb = None  # type: ignore[assignment]
    LIGHTGBM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    LIGHTGBM_IMPORT_ERROR = ""


DEFAULT_DATASET_PATH = "data/processed/rerank_train_current.csv"
DEFAULT_MODEL_PATH = "data/processed/tenderhack_lightgbm_ranker_current.txt"
DEFAULT_METADATA_PATH = "data/processed/tenderhack_lightgbm_ranker_current.json"


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


def _ordered_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(rows, key=lambda item: (str(item["group_id"]), -float(item["label"]), str(item["candidate_ste_id"])))


def _build_matrix(
    rows: List[Dict[str, object]],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    ordered = _ordered_rows(rows)
    matrix = np.asarray(
        [[float(row.get(feature_name, 0.0) or 0.0) for feature_name in feature_names] for row in ordered],
        dtype=np.float32,
    )
    labels = np.asarray([float(row["label"]) for row in ordered], dtype=np.float32)
    group_sizes: List[int] = []
    current_group_id = None
    current_group_size = 0
    for row in ordered:
        group_id = str(row["group_id"])
        if group_id != current_group_id:
            if current_group_size:
                group_sizes.append(current_group_size)
            current_group_id = group_id
            current_group_size = 1
        else:
            current_group_size += 1
    if current_group_size:
        group_sizes.append(current_group_size)
    return matrix, labels, group_sizes


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


def train_lightgbm_ranker(
    *,
    dataset_path: Path,
    model_path: Path,
    metadata_path: Path,
    objective: str,
    num_boost_round: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    valid_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict:
    if lgb is None:
        raise RuntimeError(f"LightGBM is not available: {LIGHTGBM_IMPORT_ERROR}")

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

    train_matrix, train_labels, train_groups = _build_matrix(train_rows, feature_names)
    valid_matrix, valid_labels, valid_groups = _build_matrix(valid_rows, feature_names)

    train_dataset = lgb.Dataset(
        train_matrix,
        label=train_labels,
        group=train_groups,
        feature_name=feature_names,
        free_raw_data=False,
    )
    valid_dataset = lgb.Dataset(
        valid_matrix,
        label=valid_labels,
        group=valid_groups,
        feature_name=feature_names,
        free_raw_data=False,
        reference=train_dataset,
    )

    params = {
        "objective": objective,
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "feature_fraction": float(feature_fraction),
        "bagging_fraction": float(bagging_fraction),
        "bagging_freq": int(bagging_freq),
        "verbosity": -1,
        "seed": int(random_seed),
        "feature_fraction_seed": int(random_seed),
        "bagging_seed": int(random_seed),
        "data_random_seed": int(random_seed),
    }

    evaluation_results: Dict[str, Dict[str, List[float]]] = {}
    booster = lgb.train(
        params=params,
        train_set=train_dataset,
        num_boost_round=int(num_boost_round),
        valid_sets=[valid_dataset],
        valid_names=["valid"],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.record_evaluation(evaluation_results),
        ],
    )

    best_iteration = int(booster.best_iteration or num_boost_round)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path), num_iteration=best_iteration)

    def attach_prediction(target_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not target_rows:
            return []
        matrix = np.asarray(
            [[float(row.get(feature_name, 0.0) or 0.0) for feature_name in feature_names] for row in target_rows],
            dtype=np.float32,
        )
        predictions = booster.predict(matrix, num_iteration=best_iteration)
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
        "model_type": "lightgbm",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "objective": objective,
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
        "best_iteration": best_iteration,
        "best_score": booster.best_score,
        "metrics": metrics,
        "params": params,
        "evaluation_results": evaluation_results,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM ranker on rerank dataset.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--objective", default="lambdarank", choices=["lambdarank", "rank_xendcg"])
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    metadata = train_lightgbm_ranker(
        dataset_path=Path(args.dataset_path),
        model_path=Path(args.model_path),
        metadata_path=Path(args.metadata_path),
        objective=str(args.objective),
        num_boost_round=int(args.num_boost_round),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_data_in_leaf=int(args.min_data_in_leaf),
        feature_fraction=float(args.feature_fraction),
        bagging_fraction=float(args.bagging_fraction),
        bagging_freq=int(args.bagging_freq),
        valid_fraction=float(args.valid_fraction),
        test_fraction=float(args.test_fraction),
        random_seed=int(args.random_seed),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
