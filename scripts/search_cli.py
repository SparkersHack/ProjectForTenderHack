#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.dataset_layout import (
    resolve_fasttext_model_path,
    resolve_preprocessed_db_path,
    resolve_search_db_path,
    resolve_synonyms_path,
)
from tenderhack.personalization import PersonalizationService
from tenderhack.search import SearchService


SEARCH_DB_PATH = resolve_search_db_path(PROJECT_ROOT)
PREPROCESSED_DB_PATH = resolve_preprocessed_db_path(PROJECT_ROOT)
SYNONYMS_PATH = resolve_synonyms_path(PROJECT_ROOT)
FASTTEXT_MODEL_PATH = resolve_fasttext_model_path(PROJECT_ROOT)


def ensure_required_files() -> None:
    missing = []
    for path in [SEARCH_DB_PATH, PREPROCESSED_DB_PATH, SYNONYMS_PATH]:
        if not path.exists():
            missing.append(path)
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "Не найдены нужные файлы для поиска:\n"
            f"{missing_list}\n\n"
            "Сначала соберите данные:\n"
            "python3 scripts/normalize_datasets.py\n"
            "python3 scripts/bootstrap_local.py"
        )


def render_payload(payload: dict, query: str) -> None:
    query_meta = payload["query"]
    results = payload["results"]

    print(f"\nQUERY: {query}")
    print(f"NORMALIZED: {query_meta['normalized_query']}")
    print(f"CORRECTED: {query_meta['corrected_query'] or query_meta['normalized_query']}")
    print(f"SEMANTIC BACKEND: {query_meta.get('semantic_backend', 'none')}")
    if query_meta["applied_corrections"]:
        print(f"CORRECTIONS: {query_meta['applied_corrections']}")
    if query_meta["applied_synonyms"]:
        print(f"SYNONYMS: {query_meta['applied_synonyms']}")
    if query_meta.get("applied_semantic_neighbors"):
        print(f"SEMANTIC: {query_meta['applied_semantic_neighbors']}")
    if query_meta.get("entities"):
        print(f"ENTITIES: {query_meta['entities']}")

    if not results:
        print("RESULTS: nothing found")
        return

    print("\nRESULTS:")
    for index, item in enumerate(results, start=1):
        category = item.get("category", "")
        score = item.get("final_score", item.get("search_score"))
        print(f"{index:02d}. {item['ste_id']} | {score}")
        print(f"    {item['clean_name']}")
        print(f"    category: {category}")
        if item.get("retrieval_sources"):
            print(f"    retrieval: {', '.join(item['retrieval_sources'])}")
        if item.get("explanation"):
            print(f"    explanation: {', '.join(item['explanation'])}")


def run_search(
    query: str,
    top_k: int,
    customer_inn: str | None,
    customer_region: str | None,
    clicked_ste_ids: list[str],
    cart_ste_ids: list[str],
    recent_categories: list[str],
    semantic_backend: str,
    fasttext_model_path: Path,
) -> None:
    ensure_required_files()

    search_service = SearchService(
        search_db_path=SEARCH_DB_PATH,
        synonyms_path=SYNONYMS_PATH,
        semantic_backend=semantic_backend,
        fasttext_model_path=fasttext_model_path,
    )
    personalization_service = None
    try:
        payload = search_service.search(query=query, top_k=top_k)

        if customer_inn:
            personalization_service = PersonalizationService(db_path=PREPROCESSED_DB_PATH)
            profile = personalization_service.build_customer_profile(
                customer_inn=customer_inn,
                customer_region=customer_region,
            )
            reranked = personalization_service.rerank_ste(
                payload["results"],
                profile,
                session_state={
                    "clicked_ste_ids": clicked_ste_ids,
                    "cart_ste_ids": cart_ste_ids,
                    "recent_categories": recent_categories,
                },
            )
            payload["results"] = reranked[:top_k]

        render_payload(payload, query)
    finally:
        search_service.close()
        if personalization_service:
            personalization_service.close()


def interactive_loop(args: argparse.Namespace) -> None:
    print("Interactive search mode. Empty query or 'exit' stops the session.")
    while True:
        try:
            query = input("\nsearch> ").strip()
        except EOFError:
            print()
            return
        if not query or query.lower() in {"exit", "quit"}:
            return
        run_search(
            query=query,
            top_k=args.top_k,
            customer_inn=args.customer_inn,
            customer_region=args.customer_region,
            clicked_ste_ids=args.clicked_ste_id,
            cart_ste_ids=args.cart_ste_id,
            recent_categories=args.recent_category,
            semantic_backend=args.semantic_backend,
            fasttext_model_path=Path(args.fasttext_model_path),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI for manual Tender Hack STE search checks.")
    parser.add_argument("--query", help="Single query to run. If omitted, interactive mode starts.")
    parser.add_argument("--top-k", type=int, default=10, help="How many results to print.")
    parser.add_argument("--customer-inn", help="Optional customer INN for personalized rerank.")
    parser.add_argument("--customer-region", help="Optional customer region override for personalization.")
    parser.add_argument("--clicked-ste-id", action="append", default=[], help="Session click signal. Can be repeated.")
    parser.add_argument("--cart-ste-id", action="append", default=[], help="Session cart signal. Can be repeated.")
    parser.add_argument("--recent-category", action="append", default=[], help="Recent category signal. Can be repeated.")
    parser.add_argument("--semantic-backend", choices=["auto", "fasttext", "sqlite"], default="auto")
    parser.add_argument("--fasttext-model-path", default=str(FASTTEXT_MODEL_PATH))
    args = parser.parse_args()

    if args.query:
        run_search(
            query=args.query,
            top_k=args.top_k,
            customer_inn=args.customer_inn,
            customer_region=args.customer_region,
            clicked_ste_ids=args.clicked_ste_id,
            cart_ste_ids=args.cart_ste_id,
            recent_categories=args.recent_category,
            semantic_backend=args.semantic_backend,
            fasttext_model_path=Path(args.fasttext_model_path),
        )
        return

    interactive_loop(args)


if __name__ == "__main__":
    main()
