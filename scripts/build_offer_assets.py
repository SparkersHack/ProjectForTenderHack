#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.offers import build_offer_lookup_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build STE offer lookup table from raw contracts.")
    parser.add_argument("--contracts-path", default="data/Контракты_20260403.csv")
    parser.add_argument("--preprocessed-db-path", default="data/processed/tenderhack_preprocessed.sqlite")
    args = parser.parse_args()

    count = build_offer_lookup_table(
        contracts_path=Path(args.contracts_path),
        db_path=Path(args.preprocessed_db_path),
    )
    print(f"Built ste_offer_lookup for {count:,} STE ids")


if __name__ == "__main__":
    main()
