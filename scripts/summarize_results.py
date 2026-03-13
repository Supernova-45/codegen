#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
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


def _bootstrap_ci_binary(values: list[int], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    n = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int(math.floor((alpha / 2.0) * len(means))) - 1)
    hi_idx = min(len(means) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(means))) - 1)
    return means[lo_idx], means[hi_idx]


def bootstrap_pass_at_1(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[int]] = {}
    overall_grouped: dict[str, list[int]] = {}
    for row in rows:
        val = 1 if row["pass_at_1"] else 0
        grouped.setdefault((row["condition"], row["strategy"]), []).append(val)
        overall_grouped.setdefault(row["strategy"], []).append(val)
    out: list[dict] = []
    for (condition, strategy), vals in sorted(grouped.items()):
        lo, hi = _bootstrap_ci_binary(vals)
        out.append(
            {
                "scope": "condition",
                "condition": condition,
                "strategy": strategy,
                "n": len(vals),
                "pass_at_1": sum(vals) / len(vals),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    for strategy, vals in sorted(overall_grouped.items()):
        lo, hi = _bootstrap_ci_binary(vals)
        out.append(
            {
                "scope": "overall",
                "condition": "all",
                "strategy": strategy,
                "n": len(vals),
                "pass_at_1": sum(vals) / len(vals),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    return out


def cost_efficiency_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(("condition", str(row["condition"]), str(row["strategy"])), []).append(row)
        grouped.setdefault(("overall", "all", str(row["strategy"])), []).append(row)
    one_shot_baseline: dict[tuple[str, str], float] = {}
    for (scope, condition, strategy), items in grouped.items():
        if strategy == "one-shot":
            one_shot_baseline[(scope, condition)] = (
                sum(1 if x["pass_at_1"] else 0 for x in items) / len(items)
            )
    out: list[dict] = []
    for (scope, condition, strategy), items in sorted(grouped.items()):
        p = sum(1 if x["pass_at_1"] else 0 for x in items) / len(items)
        avg_tokens = statistics.mean(
            int(x.get("total_prompt_tokens", 0)) + int(x.get("total_completion_tokens", 0))
            for x in items
        )
        baseline = one_shot_baseline.get((scope, condition), 0.0)
        out.append(
            {
                "scope": scope,
                "condition": condition,
                "strategy": strategy,
                "n": len(items),
                "pass_at_1": p,
                "avg_total_tokens": avg_tokens,
                "pass_per_1k_tokens": (1000.0 * p / avg_tokens) if avg_tokens > 0 else 0.0,
                "delta_pass_vs_oneshot": p - baseline,
                "delta_pass_per_1k_tokens_vs_oneshot": (1000.0 * (p - baseline) / avg_tokens)
                if avg_tokens > 0
                else 0.0,
            }
        )
    return out


def fixed_budget_pass_summary(rows: list[dict], budgets: list[int] | None = None) -> list[dict]:
    budgets = budgets or [500, 1000, 2000, 4000, 8000]
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(("condition", str(row["condition"]), str(row["strategy"])), []).append(row)
        grouped.setdefault(("overall", "all", str(row["strategy"])), []).append(row)
    out: list[dict] = []
    for (scope, condition, strategy), items in sorted(grouped.items()):
        with_tokens = [
            (
                int(x.get("total_prompt_tokens", 0)) + int(x.get("total_completion_tokens", 0)),
                1 if x["pass_at_1"] else 0,
            )
            for x in items
        ]
        for budget in budgets:
            eligible = [p for tok, p in with_tokens if tok <= budget]
            out.append(
                {
                    "scope": scope,
                    "condition": condition,
                    "strategy": strategy,
                    "budget_tokens": budget,
                    "eligible_n": len(eligible),
                    "coverage": len(eligible) / max(1, len(with_tokens)),
                    "pass_at_1_at_budget": (sum(eligible) / len(eligible)) if eligible else None,
                }
            )
    return out


def pareto_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["strategy"], []).append(row)
    points: list[dict] = []
    for strategy, items in sorted(grouped.items()):
        p = sum(1 if x["pass_at_1"] else 0 for x in items) / len(items)
        avg_tokens = statistics.mean(
            int(x.get("total_prompt_tokens", 0)) + int(x.get("total_completion_tokens", 0))
            for x in items
        )
        avg_questions = statistics.mean(int(x.get("questions_asked", 0)) for x in items)
        points.append(
            {
                "strategy": strategy,
                "n": len(items),
                "pass_at_1": p,
                "avg_total_tokens": avg_tokens,
                "avg_questions": avg_questions,
                "pareto_efficient": True,
            }
        )
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i == j:
                continue
            dominates = (
                b["avg_total_tokens"] <= a["avg_total_tokens"]
                and b["avg_questions"] <= a["avg_questions"]
                and b["pass_at_1"] >= a["pass_at_1"]
                and (
                    b["avg_total_tokens"] < a["avg_total_tokens"]
                    or b["avg_questions"] < a["avg_questions"]
                    or b["pass_at_1"] > a["pass_at_1"]
                )
            )
            if dominates:
                a["pareto_efficient"] = False
                break
    return points


def eig_diagnostics(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    counts_by_key: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["condition"], row["strategy"])
        counts_by_key[key] = counts_by_key.get(key, 0) + 1
        grouped.setdefault(
            key,
            {
                "rounds": 0.0,
                "generated_tests": 0.0,
                "selected_tests": 0.0,
                "sum_selected_score": 0.0,
                "zero_score_selected": 0.0,
                "filtered_non_discriminative": 0.0,
                "filtered_universal_runtime_error": 0.0,
                "filtered_low_defined_coverage": 0.0,
                "filtered_low_candidate_coverage": 0.0,
                "filtered_timeout": 0.0,
                "filtered_non_deterministic": 0.0,
                "filtered_other_invalid": 0.0,
                "filtered_signature_mismatch": 0.0,
                "filtered_keyword_arguments": 0.0,
                "filtered_extra_function_calls": 0.0,
                "filter_assert_lines": 0.0,
                "filter_accepted": 0.0,
                "type_errors": 0.0,
                "noresults": 0.0,
                "process_crashes": 0.0,
                "adapter_applied_count": 0.0,
                "adapter_success_count": 0.0,
                "invalid_tests_total": 0.0,
                "valid_tests_total": 0.0,
                "generated_tests_total": 0.0,
                "generated_unique_total": 0.0,
            },
        )
        stats = grouped[key]
        adapter_info = row.get("adapter_info", {})
        if adapter_info.get("adapter_applied"):
            stats["adapter_applied_count"] += 1
        if adapter_info.get("adapter_success"):
            stats["adapter_success_count"] += 1
        for err in row.get("eval_errors", []):
            err_s = str(err)
            if "TypeError" in err_s:
                stats["type_errors"] += 1
            if "NoResult" in err_s:
                stats["noresults"] += 1
            if "ProcessCrashed" in err_s:
                stats["process_crashes"] += 1
        for step in row.get("interaction_trace", []):
            if "round" not in step:
                continue
            stats["rounds"] += 1
            test_validation = step.get("test_validation", [])
            generated_tests = step.get("generated_tests", [])
            if isinstance(generated_tests, list):
                stats["generated_tests_total"] += float(len(generated_tests))
                stats["generated_unique_total"] += float(len({str(x).strip() for x in generated_tests}))
            stats["valid_tests_total"] += float(len(step.get("valid_tests", [])))
            stats["generated_tests"] += len(step.get("generated_tests", []))
            for tv in test_validation:
                if tv.get("valid"):
                    continue
                stats["invalid_tests_total"] += 1
                if tv.get("invalid_reason") == "non_discriminative":
                    stats["filtered_non_discriminative"] += 1
                if tv.get("invalid_reason") == "universal_runtime_error":
                    stats["filtered_universal_runtime_error"] += 1
                if tv.get("invalid_reason") == "low_defined_coverage":
                    stats["filtered_low_defined_coverage"] += 1
                if tv.get("invalid_reason") == "low_candidate_coverage":
                    stats["filtered_low_candidate_coverage"] += 1
                if tv.get("invalid_reason") == "timeout":
                    stats["filtered_timeout"] += 1
                if tv.get("invalid_reason") == "non_deterministic":
                    stats["filtered_non_deterministic"] += 1
                if tv.get("invalid_reason") not in {
                    "non_discriminative",
                    "universal_runtime_error",
                    "low_defined_coverage",
                    "low_candidate_coverage",
                    "timeout",
                    "non_deterministic",
                }:
                    stats["filtered_other_invalid"] += 1
            if step.get("decision") == "ask_and_update":
                stats["selected_tests"] += 1
                sel_score = step.get("selected_test_score")
                if isinstance(sel_score, (int, float)):
                    stats["sum_selected_score"] += float(sel_score)
                    if abs(float(sel_score)) < 1e-12:
                        stats["zero_score_selected"] += 1
            for gen in step.get("test_generation_stats", []):
                filt = gen.get("filter_stats", {})
                stats["filtered_signature_mismatch"] += float(filt.get("signature_mismatch", 0))
                stats["filtered_keyword_arguments"] += float(
                    filt.get("keyword_arguments_not_allowed", 0)
                )
                stats["filtered_extra_function_calls"] += float(
                    filt.get("extra_function_calls", 0)
                )
                stats["filter_assert_lines"] += float(filt.get("assert_lines", 0))
                stats["filter_accepted"] += float(filt.get("accepted", 0))

    out: list[dict] = []
    for (condition, strategy), s in sorted(grouped.items()):
        selected = max(1.0, s["selected_tests"])
        asserted = max(1.0, s["filter_assert_lines"])
        run_count = max(1.0, float(counts_by_key.get((condition, strategy), 0)))
        out.append(
            {
                "condition": condition,
                "strategy": strategy,
                "rounds": int(s["rounds"]),
                "avg_generated_tests_per_round": s["generated_tests"] / max(1.0, s["rounds"]),
                "selected_tests": int(s["selected_tests"]),
                "avg_selected_eig_score": s["sum_selected_score"] / selected,
                "selected_zero_score_frac": s["zero_score_selected"] / selected,
                "filtered_non_discriminative": int(s["filtered_non_discriminative"]),
                "filtered_universal_runtime_error": int(s["filtered_universal_runtime_error"]),
                "filtered_low_defined_coverage": int(s["filtered_low_defined_coverage"]),
                "filtered_low_candidate_coverage": int(s["filtered_low_candidate_coverage"]),
                "filtered_timeout": int(s["filtered_timeout"]),
                "filtered_non_deterministic": int(s["filtered_non_deterministic"]),
                "filtered_other_invalid": int(s["filtered_other_invalid"]),
                "filtered_signature_mismatch": int(s["filtered_signature_mismatch"]),
                "filtered_keyword_arguments": int(s["filtered_keyword_arguments"]),
                "filtered_extra_function_calls": int(s["filtered_extra_function_calls"]),
                "signature_mismatch_rate": s["filtered_signature_mismatch"] / asserted,
                "accepted_assert_rate": s["filter_accepted"] / asserted,
                "adapter_applied_rate": s["adapter_applied_count"] / run_count,
                "adapter_success_rate": s["adapter_success_count"] / run_count,
                "type_error_rate": s["type_errors"] / run_count,
                "noresult_rate": s["noresults"] / run_count,
                "process_crash_rate": s["process_crashes"] / run_count,
                "test_executable_rate": s["valid_tests_total"] / max(1.0, s["generated_tests_total"]),
                "test_redundancy_rate": 1.0
                - (s["generated_unique_total"] / max(1.0, s["generated_tests_total"])),
                "invalid_test_rate": s["invalid_tests_total"] / max(1.0, s["generated_tests_total"]),
                "invalid_timeout_rate": s["filtered_timeout"] / max(1.0, s["generated_tests_total"]),
                "invalid_non_deterministic_rate": s["filtered_non_deterministic"]
                / max(1.0, s["generated_tests_total"]),
                "invalid_non_discriminative_rate": s["filtered_non_discriminative"]
                / max(1.0, s["generated_tests_total"]),
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
        humanevalplus_vals = [
            x.get("humanevalplus_pass_at_1")
            for x in items
            if x.get("humanevalplus_pass_at_1") is not None
        ]
        out.append(
            {
                "strategy": strategy,
                "n": len(items),
                "pass_at_1": sum(pass_vals) / len(pass_vals),
                "mbppplus_pass_at_1": (sum(1 if x else 0 for x in mbppplus_vals) / len(mbppplus_vals))
                if mbppplus_vals
                else None,
                "mbppplus_n": len(mbppplus_vals),
                "humanevalplus_pass_at_1": (
                    sum(1 if x else 0 for x in humanevalplus_vals) / len(humanevalplus_vals)
                )
                if humanevalplus_vals
                else None,
                "humanevalplus_n": len(humanevalplus_vals),
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
    humanevalplus_by_condition: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# Strategy Comparison")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | HumanEval+ pass@1 | "
        "HumanEval+ n | avg_questions | avg_total_tokens |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in overall:
        mbppplus_display = "n/a" if row["mbppplus_pass_at_1"] is None else f"{row['mbppplus_pass_at_1']:.3f}"
        humanevalplus_display = (
            "n/a"
            if row["humanevalplus_pass_at_1"] is None
            else f"{row['humanevalplus_pass_at_1']:.3f}"
        )
        lines.append(
            "| {strategy} | {n} | {p:.3f} | {mp} | {mn} | {hp} | {hn} | {q:.3f} | {t:.1f} |".format(
                strategy=row["strategy"],
                n=row["n"],
                p=row["pass_at_1"],
                mp=mbppplus_display,
                mn=row["mbppplus_n"],
                hp=humanevalplus_display,
                hn=row["humanevalplus_n"],
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
    lines.append("## HumanEval+ By Condition")
    lines.append("")
    lines.append("| condition | strategy | n | HumanEval+ pass@1 |")
    lines.append("|---|---|---:|---:|")
    for row in humanevalplus_by_condition:
        lines.append(
            "| {condition} | {strategy} | {n} | {p:.3f} |".format(
                condition=row["condition"],
                strategy=row["strategy"],
                n=row["n"],
                p=row["humanevalplus_pass_at_1"],
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
    humanevalplus_by_condition = aggregate_optional_boolean_metric(
        rows,
        metric_key="humanevalplus_pass_at_1",
        output_key="humanevalplus_pass_at_1",
    )

    write_csv(out_dir / "summary_pass_at_1.csv", p1)
    write_csv(out_dir / "summary_mbppplus_pass_at_1.csv", mbppplus_by_condition)
    write_csv(out_dir / "summary_humanevalplus_pass_at_1.csv", humanevalplus_by_condition)
    write_csv(out_dir / "summary_questions.csv", q)
    write_csv(out_dir / "summary_pass_at_1_vs_k.csv", by_k)
    write_csv(out_dir / "summary_tokens.csv", tok)
    write_csv(out_dir / "summary_overall_strategy.csv", overall)
    write_csv(out_dir / "summary_eig_diagnostics.csv", eig_diagnostics(rows))
    write_csv(out_dir / "summary_bootstrap_ci.csv", bootstrap_pass_at_1(rows))
    write_csv(out_dir / "summary_cost_efficiency.csv", cost_efficiency_summary(rows))
    write_csv(out_dir / "summary_fixed_budget.csv", fixed_budget_pass_summary(rows))
    write_csv(out_dir / "summary_pareto.csv", pareto_summary(rows))
    write_markdown_table(
        out_dir / "comparison_table.md",
        overall,
        p1,
        mbppplus_by_condition,
        humanevalplus_by_condition,
    )

    print("Wrote:")
    print(f"- {out_dir / 'summary_pass_at_1.csv'}")
    print(f"- {out_dir / 'summary_mbppplus_pass_at_1.csv'}")
    print(f"- {out_dir / 'summary_humanevalplus_pass_at_1.csv'}")
    print(f"- {out_dir / 'summary_questions.csv'}")
    print(f"- {out_dir / 'summary_pass_at_1_vs_k.csv'}")
    print(f"- {out_dir / 'summary_tokens.csv'}")
    print(f"- {out_dir / 'summary_overall_strategy.csv'}")
    print(f"- {out_dir / 'summary_eig_diagnostics.csv'}")
    print(f"- {out_dir / 'summary_bootstrap_ci.csv'}")
    print(f"- {out_dir / 'summary_cost_efficiency.csv'}")
    print(f"- {out_dir / 'summary_fixed_budget.csv'}")
    print(f"- {out_dir / 'summary_pareto.csv'}")
    print(f"- {out_dir / 'comparison_table.md'}")


if __name__ == "__main__":
    main()
