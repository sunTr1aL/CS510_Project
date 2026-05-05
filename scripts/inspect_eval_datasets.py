"""Print small schema samples for staged benchmark parquet files."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for key in args.keys:
        files = sorted(glob.glob(str(args.root / key / "data" / "*.parquet")))
        print("KEY", key, "files", len(files))
        if not files:
            continue
        table = pq.read_table(files[0])
        print(table.schema)
        row = table.slice(0, 1).to_pylist()[0]
        preview = {name: str(value)[:300] for name, value in row.items()}
        print(json.dumps(preview, indent=2, sort_keys=True)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
