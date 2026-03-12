#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _total_tokens(row: dict[str, Any]) -> int:
    return int(row.get("total_prompt_tokens", 0)) + int(row.get("total_completion_tokens", 0))


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "pass_at_1": 0.0,
            "mbppplus_pass_at_1": None,
            "mbppplus_n": 0,
            "avg_questions": 0.0,
            "avg_total_tokens": 0.0,
        }
    p1 = sum(1 for r in rows if bool(r.get("pass_at_1"))) / n
    qs = [int(r.get("questions_asked", 0)) for r in rows]
    toks = [_total_tokens(r) for r in rows]
    mb = [r.get("mbppplus_pass_at_1") for r in rows if r.get("mbppplus_pass_at_1") is not None]
    mb_n = len(mb)
    mb_p1 = (sum(1 for v in mb if bool(v)) / mb_n) if mb_n else None
    return {
        "n": n,
        "pass_at_1": p1,
        "mbppplus_pass_at_1": mb_p1,
        "mbppplus_n": mb_n,
        "avg_questions": statistics.mean(qs),
        "avg_total_tokens": statistics.mean(toks),
    }


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _write_markdown(
    *,
    out_path: Path,
    title: str,
    left_label: str,
    right_label: str,
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> None:
    left_overall = _agg(left_rows)
    right_overall = _agg(right_rows)

    conditions = sorted({r["condition"] for r in left_rows + right_rows if "condition" in r})

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| variant | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, overall in [(left_label, left_overall), (right_label, right_overall)]:
        lines.append(
            "| {variant} | {n} | {p1} | {mbp1} | {mbn} | {q} | {tok} |".format(
                variant=label,
                n=overall["n"],
                p1=_fmt(overall["pass_at_1"]),
                mbp1=_fmt(overall["mbppplus_pass_at_1"]),
                mbn=overall["mbppplus_n"],
                q=_fmt(overall["avg_questions"]),
                tok=f"{overall['avg_total_tokens']:.1f}" if overall["n"] else "0.0",
            )
        )
    lines.append("")
    lines.append("## By Condition")
    lines.append("")
    lines.append("| condition | variant | n | pass@1 | avg_questions | avg_total_tokens |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cond in conditions:
        l = [r for r in left_rows if r.get("condition") == cond]
        r = [r for r in right_rows if r.get("condition") == cond]
        for label, rows in [(left_label, l), (right_label, r)]:
            overall = _agg(rows)
            lines.append(
                "| {cond} | {variant} | {n} | {p1} | {q} | {tok} |".format(
                    cond=cond,
                    variant=label,
                    n=overall["n"],
                    p1=_fmt(overall["pass_at_1"]),
                    q=_fmt(overall["avg_questions"]),
                    tok=f"{overall['avg_total_tokens']:.1f}" if overall["n"] else "0.0",
                )
            )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--left-results", required=True)
    parser.add_argument("--right-results", required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    left_path = Path(args.left_results)
    right_path = Path(args.right_results)
    out_path = Path(args.output_path)

    left_rows = _read_rows(left_path)
    right_rows = _read_rows(right_path)
    _write_markdown(
        out_path=out_path,
        title=str(args.title),
        left_label=str(args.left_label),
        right_label=str(args.right_label),
        left_rows=left_rows,
        right_rows=right_rows,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

