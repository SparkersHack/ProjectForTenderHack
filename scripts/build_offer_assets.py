#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.dataset_layout import resolve_preprocessed_db_path, resolve_raw_contracts_path
from tenderhack.offers import build_offer_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build STE offer lookup table from raw contracts.")
    parser.add_argument("--contracts-path", default=str(resolve_raw_contracts_path(PROJECT_ROOT)))
    parser.add_argument("--preprocessed-db-path", default=str(resolve_preprocessed_db_path(PROJECT_ROOT)))
    args = parser.parse_args()

    counts = build_offer_tables(
        contracts_path=Path(args.contracts_path),
        db_path=Path(args.preprocessed_db_path),
    )
    print(
        "Built offer assets: "
        f"ste_offer_lookup={counts['ste_lookup_count']:,}, "
        f"ste_offer_candidates={counts['offer_candidate_count']:,}"
    )


if __name__ == "__main__":
    main()
