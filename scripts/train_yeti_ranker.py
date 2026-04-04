#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_rows(dataset_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def split_group_ids(group_ids: List[str], valid_fraction: float, seed: int) -> Tuple[set[str], set[str]]:
    ordered = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    valid_size = int(len(ordered) * valid_fraction)
    if valid_fraction > 0 and valid_size == 0 and len(ordered) > 1:
        valid_size = 1
    valid_groups = set(ordered[:valid_size])
    train_groups = set(ordered[valid_size:])
    return train_groups, valid_groups


def build_pools(
    *,
    rows: List[Dict[str, str]],
    feature_names: List[str],
    train_groups: set[str],
    valid_groups: set[str],
):
    from catboost import Pool

    grouped_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["group_id"]].append(row)

    def materialize(target_groups: set[str]):
        features: List[List[float]] = []
        labels: List[float] = []
        group_id: List[str] = []
        for group in sorted(target_groups):
            group_rows = grouped_rows[group]
            for row in group_rows:
                features.append([float(row[name] or 0.0) for name in feature_names])
                labels.append(float(row["label"] or 0.0))
                group_id.append(group)
        if not features:
            return None, 0, 0
        positives = sum(1 for value in labels if value > 0)
        return Pool(data=features, label=labels, group_id=group_id, feature_names=feature_names), len(group_id), positives

    train_pool, train_rows, train_positives = materialize(train_groups)
    valid_pool, valid_rows, valid_positives = materialize(valid_groups)
    return {
        "train_pool": train_pool,
        "valid_pool": valid_pool,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "train_positives": train_positives,
        "valid_positives": valid_positives,
    }


def infer_feature_names(fieldnames: List[str]) -> List[str]:
    from tenderhack.rerank_dataset import infer_feature_columns

    return infer_feature_columns(fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline CatBoostRanker with YetiRank.")
    parser.add_argument("--dataset-path", default="data/processed/rerank_train.csv")
    parser.add_argument("--model-path", default="data/processed/tenderhack_yeti_ranker.cbm")
    parser.add_argument("--metadata-path", default="data/processed/tenderhack_yeti_ranker.json")
    parser.add_argument("--loss-function", choices=["YetiRank", "YetiRankPairwise"], default="YetiRank")
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from catboost import CatBoostRanker
    except ImportError as exc:
        raise SystemExit(
            "catboost is not installed. Install it first, for example:\n"
            "pip install catboost"
        ) from exc

    dataset_path = Path(args.dataset_path)
    model_path = Path(args.model_path)
    metadata_path = Path(args.metadata_path)
    rows, fieldnames = load_rows(dataset_path)
    if not rows:
        raise SystemExit(f"Dataset is empty: {dataset_path}")

    feature_names = infer_feature_names(fieldnames)
    group_ids = [row["group_id"] for row in rows]
    train_groups, valid_groups = split_group_ids(group_ids, valid_fraction=args.valid_fraction, seed=args.random_seed)
    pools = build_pools(
        rows=rows,
        feature_names=feature_names,
        train_groups=train_groups,
        valid_groups=valid_groups,
    )
    if pools["train_pool"] is None:
        raise SystemExit("Training split is empty.")

    model = CatBoostRanker(
        loss_function=args.loss_function,
        eval_metric="NDCG:top=10",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        verbose=50,
    )
    model.fit(
        pools["train_pool"],
        eval_set=pools["valid_pool"],
        use_best_model=bool(pools["valid_pool"]),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    metadata = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "loss_function": args.loss_function,
        "feature_names": feature_names,
        "train_groups": len(train_groups),
        "valid_groups": len(valid_groups),
        "train_rows": pools["train_rows"],
        "valid_rows": pools["valid_rows"],
        "train_positives": pools["train_positives"],
        "valid_positives": pools["valid_positives"],
        "best_iteration": model.get_best_iteration(),
        "best_score": model.get_best_score(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
