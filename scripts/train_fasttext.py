#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

try:
    import fasttext
except ImportError as exc:
    raise SystemExit(
        "fasttext не установлен. Установите зависимости:\n"
        "python3 -m pip install --user -r requirements.txt"
    ) from exc


PROGRESS_EVERY = 100_000


def build_corpus_line(row: dict[str, str]) -> str:
    normalized_name = row.get("normalized_name", "").strip()
    normalized_category = row.get("normalized_category", "").strip()
    key_tokens = row.get("key_tokens", "").strip()

    weighted_parts = []
    if normalized_name:
        weighted_parts.extend([normalized_name, normalized_name])
    if normalized_category:
        weighted_parts.append(normalized_category)
    if key_tokens:
        weighted_parts.append(key_tokens)
    return " ".join(part for part in weighted_parts if part)


def build_fasttext_corpus(catalog_path: Path, corpus_path: Path) -> int:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with catalog_path.open("r", encoding="utf-8", newline="") as source_handle, corpus_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target_handle:
        reader = csv.DictReader(source_handle)
        for row in reader:
            line = build_corpus_line(row)
            if not line:
                continue
            target_handle.write(line)
            target_handle.write("\n")
            rows_written += 1
            if rows_written % PROGRESS_EVERY == 0:
                print(f"[fastText] prepared {rows_written:,} corpus rows", flush=True)
    return rows_written


def train_fasttext_model(
    corpus_path: Path,
    model_path: Path,
    model_type: str = "skipgram",
    dim: int = 100,
    epoch: int = 10,
    ws: int = 6,
    min_count: int = 3,
    minn: int = 3,
    maxn: int = 5,
    lr: float = 0.05,
    thread: int | None = None,
) -> None:
    if thread is None:
        cpu_count = os.cpu_count() or 4
        thread = max(1, min(cpu_count, 8))
    print(
        f"[fastText] training model={model_type} dim={dim} epoch={epoch} ws={ws} "
        f"minCount={min_count} minn={minn} maxn={maxn} thread={thread}",
        flush=True,
    )
    model = fasttext.train_unsupervised(
        input=str(corpus_path),
        model=model_type,
        dim=dim,
        epoch=epoch,
        ws=ws,
        minCount=min_count,
        minn=minn,
        maxn=maxn,
        lr=lr,
        thread=thread,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"[fastText] saved model to {model_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fastText model for Tender Hack STE semantic search.")
    parser.add_argument("--catalog-path", default="datasets/processed/ste_catalog_search_ready.csv")
    parser.add_argument("--corpus-path", default="datasets/processed/tenderhack_fasttext_corpus.txt")
    parser.add_argument("--model-path", default="datasets/processed/tenderhack_fasttext.bin")
    parser.add_argument("--model-type", choices=["skipgram", "cbow"], default="skipgram")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--ws", type=int, default=6)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--minn", type=int, default=3)
    parser.add_argument("--maxn", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--thread", type=int, default=None)
    parser.add_argument("--skip-corpus-build", action="store_true")
    args = parser.parse_args()

    catalog_path = Path(args.catalog_path)
    corpus_path = Path(args.corpus_path)
    model_path = Path(args.model_path)

    if not args.skip_corpus_build:
        rows_written = build_fasttext_corpus(catalog_path, corpus_path)
        print(f"[fastText] corpus rows written: {rows_written:,}", flush=True)

    train_fasttext_model(
        corpus_path=corpus_path,
        model_path=model_path,
        model_type=args.model_type,
        dim=args.dim,
        epoch=args.epoch,
        ws=args.ws,
        min_count=args.min_count,
        minn=args.minn,
        maxn=args.maxn,
        lr=args.lr,
        thread=args.thread,
    )


if __name__ == "__main__":
    main()
