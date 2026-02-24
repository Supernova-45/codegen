#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.mbpp_loader import extract_function_name


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_test_list(s: str) -> list[str]:
    return [line.strip() for line in s.splitlines() if line.strip().startswith("assert ")]


def infer_function_name(oracle_row: dict[str, Any], oracle_tests: list[str]) -> str:
    try:
        return extract_function_name(oracle_tests)
    except ValueError:
        defs = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", oracle_row["code"])
        if defs:
            return defs[-1]
        raise


def build_task_rows(
    ticode_mbpp: list[dict[str, Any]],
    original: list[dict[str, Any]],
    incomplete: list[dict[str, Any]],
    ambiguous: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    oracle_by_task = {int(x["task_id"]): x for x in ticode_mbpp}
    cond_maps = {
        "original": {int(x["task_id"]): x for x in original},
        "incomplete": {int(x["task_id"]): x for x in incomplete},
        "ambiguous": {int(x["task_id"]): x for x in ambiguous},
    }

    rows: list[dict[str, Any]] = []
    for task_id, oracle_row in sorted(oracle_by_task.items(), key=lambda x: x[0]):
        oracle_tests = list(oracle_row.get("test_list", []))
        challenge = list(oracle_row.get("challenge_test_list", []))
        hidden_tests = challenge if challenge else oracle_tests[1:] if len(oracle_tests) > 1 else oracle_tests
        visible_tests = oracle_tests[:1]
        function_name = infer_function_name(oracle_row, oracle_tests)

        for condition, cond_map in cond_maps.items():
            if task_id not in cond_map:
                continue
            prompt = cond_map[task_id]["prompt"]
            rows.append(
                {
                    "task_id": task_id,
                    "condition": condition,
                    "prompt": prompt,
                    "oracle_code": oracle_row["code"],
                    "visible_tests": visible_tests,
                    "hidden_tests": hidden_tests,
                    "function_name": function_name,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ticode-mbpp",
        default=str(ROOT / "data/sources/ticode_mbpp/mbpp.jsonl"),
    )
    parser.add_argument(
        "--robustness-dir",
        default=str(ROOT / "data/sources/robustness_mbpp"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ticode_mbpp = read_jsonl(Path(args.ticode_mbpp))
    robustness_dir = Path(args.robustness_dir)
    original = read_json(robustness_dir / "MBPP.json")
    incomplete = read_json(robustness_dir / "incomplete_MBPP.json")
    ambiguous = read_json(robustness_dir / "ambiguous_MBPP.json")

    out_rows = build_task_rows(ticode_mbpp, original, incomplete, ambiguous)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(out_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
