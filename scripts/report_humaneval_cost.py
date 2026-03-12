#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cost-focused report artifacts from run_model_matrix outputs "
            "for HumanEval cheap-vs-large comparisons."
        )
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help=(
            "Path to a run_model_matrix output directory containing "
            "model_comparison.csv and model_budget_comparison.csv"
        ),
    )
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--cheap-profile", required=True)
    parser.add_argument("--large-profile", required=True)
    parser.add_argument("--cheap-strategy", default="eig-tests")
    parser.add_argument("--large-strategy", default="one-shot")
    parser.add_argument(
        "--line-strategies",
        nargs="+",
        default=["one-shot", "eig-tests"],
        help="Strategies to include in fixed-budget line chart for each profile.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to <run-root>/humaneval_cost_report.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s in {"", "None", "null", "nan"}:
        return None
    return float(s)


def _to_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out: list[str] = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _build_budget_grid(budget_rows: list[dict[str, str]], dataset: str) -> list[int]:
    vals = {
        int(row["budget_tokens"])
        for row in budget_rows
        if row.get("dataset") == dataset and row.get("budget_tokens")
    }
    return sorted(vals)


def _first_budget_geq(budgets: list[int], target: float) -> int:
    if not budgets:
        raise ValueError("No budget points found in budget comparison CSV.")
    for b in budgets:
        if float(b) >= target:
            return b
    return budgets[-1]


def _key(profile: str, strategy: str) -> tuple[str, str]:
    return (profile, strategy)


def _format_f(value: float | None, ndigits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{ndigits}f}"


def maybe_make_plots(
    *,
    output_dir: Path,
    comparison_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    profiles: list[str],
    line_strategies: list[str],
    dataset: str,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    plot_paths: list[str] = []

    # Plot 1: fixed-budget pass@1 curves for selected profile/strategy lines.
    budget_grid = sorted({int(r["budget_tokens"]) for r in budget_rows})
    line_rows = [
        r
        for r in budget_rows
        if str(r["profile"]) in profiles and str(r["strategy"]) in line_strategies
    ]
    if line_rows and budget_grid:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for profile in profiles:
            for strategy in line_strategies:
                pts = [
                    r
                    for r in line_rows
                    if str(r["profile"]) == profile and str(r["strategy"]) == strategy
                ]
                by_budget = {int(p["budget_tokens"]): p for p in pts}
                xs: list[int] = []
                ys: list[float] = []
                for b in budget_grid:
                    p = by_budget.get(b)
                    if not p:
                        continue
                    y = p.get("pass_at_1_at_budget")
                    if y is None:
                        continue
                    xs.append(b)
                    ys.append(float(y))
                if xs:
                    ax.plot(xs, ys, marker="o", label=f"{profile} / {strategy}")
        ax.set_title(f"{dataset}: pass@1 at fixed token budgets")
        ax.set_xlabel("Token budget")
        ax.set_ylabel("pass@1")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        p = output_dir / "plot_budget_curve.png"
        fig.tight_layout()
        fig.savefig(p, dpi=180)
        plt.close(fig)
        plot_paths.append(str(p))

    # Plot 2: Pareto-style scatter (avg tokens vs pass@1).
    if comparison_rows:
        fig, ax = plt.subplots(figsize=(8, 5.0))
        for row in comparison_rows:
            profile = str(row["profile"])
            strategy = str(row["strategy"])
            x = float(row["avg_total_tokens"])
            y = float(row["pass_at_1"])
            pareto = bool(row["pareto_efficient"])
            marker = "D" if pareto else "o"
            ax.scatter(
                x,
                y,
                marker=marker,
                s=70,
                alpha=0.85,
                label=f"{profile}/{strategy}",
            )
            ax.annotate(strategy, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(f"{dataset}: pass@1 vs avg tokens (Pareto points in diamond)")
        ax.set_xlabel("Average total tokens")
        ax.set_ylabel("pass@1")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)

        # De-duplicate legend entries.
        handles, labels = ax.get_legend_handles_labels()
        dedup: dict[str, Any] = {}
        for h, l in zip(handles, labels):
            if l not in dedup:
                dedup[l] = h
        ax.legend(dedup.values(), dedup.keys(), fontsize=7, ncol=2)

        p = output_dir / "plot_pareto_tokens_vs_pass.png"
        fig.tight_layout()
        fig.savefig(p, dpi=180)
        plt.close(fig)
        plot_paths.append(str(p))

    # Plot 3: pass per 1k tokens bar chart.
    if comparison_rows:
        rows = sorted(
            comparison_rows,
            key=lambda r: (str(r["profile"]), str(r["strategy"])),
        )
        labels = [f"{r['profile']}\n{r['strategy']}" for r in rows]
        values = [float(r["pass_per_1k_tokens"]) for r in rows]
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(rows)), 4.8))
        ax.bar(labels, values)
        ax.set_title(f"{dataset}: accuracy gain density (pass@1 per 1k tokens)")
        ax.set_ylabel("pass@1 per 1k tokens")
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        p = output_dir / "plot_pass_per_1k_tokens.png"
        fig.tight_layout()
        fig.savefig(p, dpi=180)
        plt.close(fig)
        plot_paths.append(str(p))

    return plot_paths


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.output_dir) if args.output_dir else run_root / "humaneval_cost_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_raw = read_csv_rows(run_root / "model_comparison.csv")
    budget_raw = read_csv_rows(run_root / "model_budget_comparison.csv")

    profiles_of_interest = [args.cheap_profile, args.large_profile]

    comparison_rows: list[dict[str, Any]] = []
    for row in comparison_raw:
        if row.get("dataset") != args.dataset:
            continue
        profile = str(row.get("profile", ""))
        if profile not in profiles_of_interest:
            continue
        comparison_rows.append(
            {
                "dataset": args.dataset,
                "profile": profile,
                "strategy": str(row["strategy"]),
                "n": int(row["n"]),
                "pass_at_1": float(row["pass_at_1"]),
                "avg_total_tokens": float(row["avg_total_tokens"]),
                "avg_questions": float(row["avg_questions"]),
                "pass_per_1k_tokens": float(row["pass_per_1k_tokens"]),
                "delta_pass_vs_oneshot": float(row["delta_pass_vs_oneshot"]),
                "pareto_efficient": _to_bool(row.get("pareto_efficient")),
                "ci95_low": _to_float(row.get("ci95_low")),
                "ci95_high": _to_float(row.get("ci95_high")),
            }
        )

    budget_rows: list[dict[str, Any]] = []
    for row in budget_raw:
        if row.get("dataset") != args.dataset:
            continue
        profile = str(row.get("profile", ""))
        if profile not in profiles_of_interest:
            continue
        budget_rows.append(
            {
                "dataset": args.dataset,
                "profile": profile,
                "strategy": str(row["strategy"]),
                "budget_tokens": int(row["budget_tokens"]),
                "eligible_n": int(row["eligible_n"]),
                "coverage": float(row["coverage"]),
                "pass_at_1_at_budget": _to_float(row.get("pass_at_1_at_budget")),
            }
        )

    if not comparison_rows:
        raise ValueError(
            "No comparison rows after filtering. Check --dataset and profile names."
        )

    comparison_rows.sort(key=lambda r: (str(r["profile"]), str(r["strategy"])))
    budget_rows.sort(
        key=lambda r: (
            str(r["profile"]),
            str(r["strategy"]),
            int(r["budget_tokens"]),
        )
    )

    comp_by_key = {
        _key(str(r["profile"]), str(r["strategy"])): r for r in comparison_rows
    }
    budget_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for r in budget_rows:
        budget_by_key[
            (str(r["profile"]), str(r["strategy"]), int(r["budget_tokens"]))
        ] = r

    large_base_key = _key(args.large_profile, args.large_strategy)
    cheap_eig_key = _key(args.cheap_profile, args.cheap_strategy)
    if large_base_key not in comp_by_key:
        raise ValueError(
            f"Missing large baseline row for profile={args.large_profile} strategy={args.large_strategy}."
        )
    if cheap_eig_key not in comp_by_key:
        raise ValueError(
            f"Missing cheap EIG row for profile={args.cheap_profile} strategy={args.cheap_strategy}."
        )

    large_base = comp_by_key[large_base_key]
    cheap_eig = comp_by_key[cheap_eig_key]

    budget_grid = _build_budget_grid(budget_raw, dataset=args.dataset)
    match_budget = _first_budget_geq(budget_grid, target=float(large_base["avg_total_tokens"]))

    match_rows: list[dict[str, Any]] = []
    for row in comparison_rows:
        k = (str(row["profile"]), str(row["strategy"]), match_budget)
        b = budget_by_key.get(k)
        match_rows.append(
            {
                "dataset": args.dataset,
                "profile": row["profile"],
                "strategy": row["strategy"],
                "target_budget_tokens": match_budget,
                "pass_at_1_at_target_budget": (
                    None if b is None else b.get("pass_at_1_at_budget")
                ),
                "coverage_at_target_budget": (None if b is None else b.get("coverage")),
                "overall_pass_at_1": row["pass_at_1"],
                "avg_total_tokens": row["avg_total_tokens"],
                "avg_questions": row["avg_questions"],
                "pass_per_1k_tokens": row["pass_per_1k_tokens"],
                "delta_pass_vs_oneshot": row["delta_pass_vs_oneshot"],
            }
        )

    write_csv(out_dir / "cost_summary.csv", comparison_rows)
    write_csv(out_dir / "fixed_budget_curve.csv", budget_rows)
    write_csv(out_dir / "budget_match_summary.csv", match_rows)

    plot_paths = maybe_make_plots(
        output_dir=out_dir,
        comparison_rows=comparison_rows,
        budget_rows=budget_rows,
        profiles=profiles_of_interest,
        line_strategies=list(args.line_strategies),
        dataset=args.dataset,
    )

    md: list[str] = []
    md.append(f"# Cost-Focused HumanEval Report ({args.dataset})")
    md.append("")
    md.append("## Key Comparison")
    md.append("")
    md.append(
        "- Cheap EIG: "
        f"{args.cheap_profile}/{args.cheap_strategy}: pass@1={cheap_eig['pass_at_1']:.3f}, "
        f"avg_tokens={cheap_eig['avg_total_tokens']:.1f}, "
        f"pass/1k={cheap_eig['pass_per_1k_tokens']:.4f}"
    )
    md.append(
        "- Large baseline: "
        f"{args.large_profile}/{args.large_strategy}: pass@1={large_base['pass_at_1']:.3f}, "
        f"avg_tokens={large_base['avg_total_tokens']:.1f}, "
        f"pass/1k={large_base['pass_per_1k_tokens']:.4f}"
    )
    md.append(
        "- Accuracy gap (cheap EIG - large baseline): "
        f"{cheap_eig['pass_at_1'] - large_base['pass_at_1']:+.3f}"
    )
    md.append(
        "- Token ratio (cheap EIG / large baseline): "
        f"{cheap_eig['avg_total_tokens'] / max(1.0, large_base['avg_total_tokens']):.2f}x"
    )
    md.append(
        f"- Compute-matched budget used in table below: {match_budget} tokens "
        f"(first budget >= large baseline avg token use {large_base['avg_total_tokens']:.1f})."
    )
    md.append("")

    md.append("## Overall Cost Table")
    md.append("")
    table_rows: list[list[str]] = []
    for row in comparison_rows:
        table_rows.append(
            [
                str(row["profile"]),
                str(row["strategy"]),
                str(row["n"]),
                _format_f(float(row["pass_at_1"]), 3),
                _format_f(float(row["avg_total_tokens"]), 1),
                _format_f(float(row["avg_questions"]), 2),
                _format_f(float(row["pass_per_1k_tokens"]), 4),
                _format_f(float(row["delta_pass_vs_oneshot"]), 3),
                "yes" if bool(row["pareto_efficient"]) else "no",
            ]
        )
    md.append(
        _md_table(
            [
                "profile",
                "strategy",
                "n",
                "pass@1",
                "avg_tokens",
                "avg_questions",
                "pass_per_1k",
                "delta_pass_vs_oneshot",
                "pareto",
            ],
            table_rows,
        )
    )
    md.append("")

    md.append("## Compute-Matched Table")
    md.append("")
    match_table_rows: list[list[str]] = []
    for row in sorted(match_rows, key=lambda r: (str(r["profile"]), str(r["strategy"]))):
        match_table_rows.append(
            [
                str(row["profile"]),
                str(row["strategy"]),
                str(row["target_budget_tokens"]),
                _format_f(_to_float(str(row["pass_at_1_at_target_budget"])), 3),
                _format_f(_to_float(str(row["coverage_at_target_budget"])), 3),
                _format_f(float(row["overall_pass_at_1"]), 3),
                _format_f(float(row["avg_total_tokens"]), 1),
            ]
        )
    md.append(
        _md_table(
            [
                "profile",
                "strategy",
                "budget",
                "pass@1_at_budget",
                "coverage",
                "overall_pass@1",
                "avg_tokens",
            ],
            match_table_rows,
        )
    )
    md.append("")

    md.append("## Fixed-Budget Curve Data")
    md.append("")
    md.append("Use `fixed_budget_curve.csv` directly in your paper tables/figures.")
    md.append("")

    if plot_paths:
        md.append("## Generated Plots")
        md.append("")
        for p in plot_paths:
            md.append(f"- `{p}`")
    else:
        md.append("## Plots")
        md.append("")
        md.append(
            "Matplotlib was not available in this environment, so plot files were not generated. "
            "Install matplotlib and rerun this script to produce PNG figures."
        )

    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Wrote report artifacts:")
    print(f"- {out_dir / 'cost_summary.csv'}")
    print(f"- {out_dir / 'fixed_budget_curve.csv'}")
    print(f"- {out_dir / 'budget_match_summary.csv'}")
    print(f"- {out_dir / 'report.md'}")
    if plot_paths:
        for p in plot_paths:
            print(f"- {p}")


if __name__ == "__main__":
    main()
