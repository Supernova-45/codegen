from __future__ import annotations

from collections import defaultdict
from typing import Any


def aggregate_pass_at_1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["strategy"])
        grouped[key].append(1 if row["pass_at_1"] else 0)

    summary: list[dict[str, Any]] = []
    for (condition, strategy), vals in grouped.items():
        summary.append(
            {
                "condition": condition,
                "strategy": strategy,
                "n": len(vals),
                "pass_at_1": sum(vals) / max(1, len(vals)),
            }
        )
    summary.sort(key=lambda x: (x["condition"], x["strategy"]))
    return summary


def average_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["strategy"])
        grouped[key].append(int(row.get("questions_asked", 0)))

    out: list[dict[str, Any]] = []
    for (condition, strategy), vals in grouped.items():
        out.append(
            {
                "condition": condition,
                "strategy": strategy,
                "avg_questions": sum(vals) / max(1, len(vals)),
            }
        )
    out.sort(key=lambda x: (x["condition"], x["strategy"]))
    return out
