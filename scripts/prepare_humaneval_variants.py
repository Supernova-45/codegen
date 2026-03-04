#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.humaneval_loader import build_humaneval_variants_rows


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robustness-dir",
        default=str(ROOT / "data/sources/robustness_humaneval"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data/humaneval_variants.jsonl"),
    )
    args = parser.parse_args()

    d = Path(args.robustness_dir)
    original = read_json(d / "HumanEval.json")
    incomplete = read_json(d / "incomplete_humaneval.json")
    ambiguous = read_json(d / "ambiguous_humaneval.json")
    contradictory_path = d / "contradictory_humaneval.json"
    contradictory = read_json(contradictory_path) if contradictory_path.exists() else None

    rows = build_humaneval_variants_rows(
        original=original,
        incomplete=incomplete,
        ambiguous=ambiguous,
        contradictory=contradictory,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
