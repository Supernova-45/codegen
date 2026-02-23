#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.metrics import aggregate_optional_boolean_metric, aggregate_pass_at_1, average_questions


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pass_at_1_vs_k(rows: list[dict]) -> list[dict]:
    # Uses number of asked questions as realized K for adaptive strategy.
    grouped: dict[tuple[str, str, int], list[int]] = {}
    for row in rows:
        k = int(row.get("questions_asked", 0))
        key = (row["condition"], row["strategy"], k)
        grouped.setdefault(key, []).append(1 if row["pass_at_1"] else 0)
    out: list[dict] = []
    for (condition, strategy, k), vals in sorted(grouped.items()):
        out.append(
            {
                "condition": condition,
                "strategy": strategy,
                "k": k,
                "n": len(vals),
                "pass_at_1": sum(vals) / len(vals),
            }
        )
    return out


def token_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        total = int(row.get("total_prompt_tokens", 0)) + int(
            row.get("total_completion_tokens", 0)
        )
        key = (row["condition"], row["strategy"])
        grouped.setdefault(key, []).append(total)
    out: list[dict] = []
    for (condition, strategy), vals in sorted(grouped.items()):
        out.append(
            {
                "condition": condition,
                "strategy": strategy,
                "avg_total_tokens": statistics.mean(vals),
                "median_total_tokens": statistics.median(vals),
            }
        )
    return out


def overall_strategy_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["strategy"], []).append(row)
    out: list[dict] = []
    for strategy, items in sorted(grouped.items()):
        pass_vals = [1 if x["pass_at_1"] else 0 for x in items]
        q_vals = [int(x.get("questions_asked", 0)) for x in items]
        tok_vals = [
            int(x.get("total_prompt_tokens", 0)) + int(x.get("total_completion_tokens", 0))
            for x in items
        ]
        mbppplus_vals = [x.get("mbppplus_pass_at_1") for x in items if x.get("mbppplus_pass_at_1") is not None]
        out.append(
            {
                "strategy": strategy,
                "n": len(items),
                "pass_at_1": sum(pass_vals) / len(pass_vals),
                "mbppplus_pass_at_1": (sum(1 if x else 0 for x in mbppplus_vals) / len(mbppplus_vals))
                if mbppplus_vals
                else None,
                "mbppplus_n": len(mbppplus_vals),
                "avg_questions": sum(q_vals) / len(q_vals),
                "avg_total_tokens": statistics.mean(tok_vals),
            }
        )
    return out


def write_markdown_table(
    path: Path,
    overall: list[dict],
    by_condition: list[dict],
    mbppplus_by_condition: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# Strategy Comparison")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in overall:
        mbppplus_display = "n/a" if row["mbppplus_pass_at_1"] is None else f"{row['mbppplus_pass_at_1']:.3f}"
        lines.append(
            "| {strategy} | {n} | {p:.3f} | {mp} | {mn} | {q:.3f} | {t:.1f} |".format(
                strategy=row["strategy"],
                n=row["n"],
                p=row["pass_at_1"],
                mp=mbppplus_display,
                mn=row["mbppplus_n"],
                q=row["avg_questions"],
                t=row["avg_total_tokens"],
            )
        )
    lines.append("")
    lines.append("## By Condition")
    lines.append("")
    lines.append("| condition | strategy | n | pass@1 |")
    lines.append("|---|---|---:|---:|")
    for row in by_condition:
        lines.append(
            "| {condition} | {strategy} | {n} | {p:.3f} |".format(
                condition=row["condition"],
                strategy=row["strategy"],
                n=row["n"],
                p=row["pass_at_1"],
            )
        )
    lines.append("")
    lines.append("## MBPP+ By Condition")
    lines.append("")
    lines.append("| condition | strategy | n | MBPP+ pass@1 |")
    lines.append("|---|---|---:|---:|")
    for row in mbppplus_by_condition:
        lines.append(
            "| {condition} | {strategy} | {n} | {p:.3f} |".format(
                condition=row["condition"],
                strategy=row["strategy"],
                n=row["n"],
                p=row["mbppplus_pass_at_1"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(results_path)
    p1 = aggregate_pass_at_1(rows)
    q = average_questions(rows)
    by_k = pass_at_1_vs_k(rows)
    tok = token_summary(rows)
    overall = overall_strategy_summary(rows)
    mbppplus_by_condition = aggregate_optional_boolean_metric(
        rows,
        metric_key="mbppplus_pass_at_1",
        output_key="mbppplus_pass_at_1",
    )

    write_csv(out_dir / "summary_pass_at_1.csv", p1)
    write_csv(out_dir / "summary_mbppplus_pass_at_1.csv", mbppplus_by_condition)
    write_csv(out_dir / "summary_questions.csv", q)
    write_csv(out_dir / "summary_pass_at_1_vs_k.csv", by_k)
    write_csv(out_dir / "summary_tokens.csv", tok)
    write_csv(out_dir / "summary_overall_strategy.csv", overall)
    write_markdown_table(out_dir / "comparison_table.md", overall, p1, mbppplus_by_condition)

    print("Wrote:")
    print(f"- {out_dir / 'summary_pass_at_1.csv'}")
    print(f"- {out_dir / 'summary_mbppplus_pass_at_1.csv'}")
    print(f"- {out_dir / 'summary_questions.csv'}")
    print(f"- {out_dir / 'summary_pass_at_1_vs_k.csv'}")
    print(f"- {out_dir / 'summary_tokens.csv'}")
    print(f"- {out_dir / 'summary_overall_strategy.csv'}")
    print(f"- {out_dir / 'comparison_table.md'}")


if __name__ == "__main__":
    main()
