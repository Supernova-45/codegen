#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results" / "model_matrix"
DEFAULT_STRATEGIES = [
    "one-shot",
    "random-tests",
    "eig-tests",
    "self-consistency",
    "repair",
]
DEFAULT_CONFIGS = {
    "mbpp": ROOT / "configs" / "mvp_mbpp.yaml",
    "humaneval": ROOT / "configs" / "mvp_humaneval.yaml",
}
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full benchmark matrix across model profiles and summarize "
            "cheap-vs-strong comparisons."
        )
    )
    parser.add_argument(
        "--profiles-config",
        default=str(ROOT / "configs" / "model_profiles.yaml"),
        help="YAML file with profile env mappings.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="Profile names to execute from --profiles-config.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mbpp", "humaneval"],
        choices=["mbpp", "humaneval"],
    )
    parser.add_argument("--mbpp-config", default=str(DEFAULT_CONFIGS["mbpp"]))
    parser.add_argument("--humaneval-config", default=str(DEFAULT_CONFIGS["humaneval"]))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-mbppplus", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--pipeline-overrides", nargs="*", default=[])
    parser.add_argument("--parallel-shards", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Profile config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("profiles-config must contain a top-level 'profiles' mapping.")
    return profiles


def expand_profile_env(raw_env: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in raw_env.items():
        text = str(value)

        def _replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            resolved = os.environ.get(env_name, "").strip()
            if not resolved:
                raise ValueError(
                    f"Required environment variable '{env_name}' is missing for profile expansion."
                )
            return resolved

        out[key] = ENV_VAR_PATTERN.sub(_replace, text)
    return out


def run_dataset_profile(
    *,
    profile_name: str,
    profile_env: dict[str, str],
    dataset: str,
    config_path: Path,
    run_root: Path,
    strategies: list[str],
    num_shards: int,
    resume: bool,
    skip_mbppplus: bool,
    max_examples: int,
    parallel_shards: bool,
    pipeline_overrides: list[str],
    dry_run: bool,
) -> tuple[Path, Path]:
    profile_dir = run_root / profile_name / dataset
    shard_dir = profile_dir / "shards"
    summary_dir = profile_dir / "summary"
    merged_path = profile_dir / f"{dataset}_merged.jsonl"
    shard_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(profile_env)

    commands: list[list[str]] = []
    shard_files: list[Path] = []
    for shard_idx in range(num_shards):
        shard_out = shard_dir / f"shard_{shard_idx}.jsonl"
        shard_files.append(shard_out)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            "--config",
            str(config_path),
            "--strategies",
            *strategies,
            "--num-shards",
            str(num_shards),
            "--shard-index",
            str(shard_idx),
            "--output-file",
            str(shard_out),
        ]
        if resume:
            cmd.append("--resume")
        if skip_mbppplus:
            cmd.append("--skip-mbppplus")
        if max_examples > 0:
            cmd.extend(["--max-examples", str(max_examples)])
        if pipeline_overrides:
            cmd.extend(["--pipeline-overrides", *pipeline_overrides])
        commands.append(cmd)

    for cmd in commands:
        print(" ".join(cmd))
    if dry_run:
        return merged_path, summary_dir

    if parallel_shards:
        procs = [subprocess.Popen(cmd, cwd=str(ROOT), env=env) for cmd in commands]
        failed = False
        for proc in procs:
            code = proc.wait()
            if code != 0:
                failed = True
        if failed:
            raise RuntimeError(f"One or more shards failed for profile={profile_name} dataset={dataset}")
    else:
        for cmd in commands:
            subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as out_f:
        for shard_path in shard_files:
            if not shard_path.exists():
                continue
            with shard_path.open("r", encoding="utf-8", errors="replace") as in_f:
                for line in in_f:
                    s = line.strip()
                    if s:
                        out_f.write(s + "\n")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_results.py"),
            "--results",
            str(merged_path),
            "--output-dir",
            str(summary_dir),
        ],
        cwd=str(ROOT),
        env=env,
        check=True,
    )
    return merged_path, summary_dir


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    lines: list[str] = []
    lines.append("# Model Matrix Comparison")
    lines.append("")
    lines.append(
        "| dataset | profile | strategy | pass@1 | ci95 | avg_tokens | avg_questions | pass/1k tokens | delta pass vs one-shot |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        ci_text = "n/a"
        if row["ci95_low"] is not None and row["ci95_high"] is not None:
            ci_text = f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
        lines.append(
            "| {dataset} | {profile} | {strategy} | {pass_at_1:.3f} | {ci} | {avg_tokens:.1f} | "
            "{avg_questions:.2f} | {pass_per_1k_tokens:.4f} | {delta_pass_vs_oneshot:.3f} |".format(
                dataset=row["dataset"],
                profile=row["profile"],
                strategy=row["strategy"],
                pass_at_1=row["pass_at_1"],
                ci=ci_text,
                avg_tokens=row["avg_total_tokens"],
                avg_questions=row["avg_questions"],
                pass_per_1k_tokens=row["pass_per_1k_tokens"],
                delta_pass_vs_oneshot=row["delta_pass_vs_oneshot"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cross_model_summary(
    run_index: list[tuple[str, str, Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparison_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []

    for profile, dataset, summary_dir in run_index:
        overall_rows = read_csv_rows(summary_dir / "summary_overall_strategy.csv")
        ci_rows = [
            x for x in read_csv_rows(summary_dir / "summary_bootstrap_ci.csv")
            if x.get("scope") == "overall"
        ]
        ci_by_strategy = {x["strategy"]: x for x in ci_rows}
        cost_rows = [
            x for x in read_csv_rows(summary_dir / "summary_cost_efficiency.csv")
            if x.get("scope") == "overall" and x.get("condition") == "all"
        ]
        cost_by_strategy = {x["strategy"]: x for x in cost_rows}
        fixed_budget = [
            x for x in read_csv_rows(summary_dir / "summary_fixed_budget.csv")
            if x.get("scope") == "overall" and x.get("condition") == "all"
        ]
        for row in fixed_budget:
            budget_rows.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "strategy": row["strategy"],
                    "budget_tokens": int(row["budget_tokens"]),
                    "eligible_n": int(row["eligible_n"]),
                    "coverage": float(row["coverage"]),
                    "pass_at_1_at_budget": (
                        None
                        if row["pass_at_1_at_budget"] in {"", "None", "null"}
                        else float(row["pass_at_1_at_budget"])
                    ),
                }
            )

        for row in overall_rows:
            strategy = row["strategy"]
            ci = ci_by_strategy.get(strategy, {})
            cost = cost_by_strategy.get(strategy, {})
            avg_tokens = float(row["avg_total_tokens"])
            pass_at_1 = float(row["pass_at_1"])
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "strategy": strategy,
                    "n": int(row["n"]),
                    "pass_at_1": pass_at_1,
                    "ci95_low": (
                        None
                        if not ci.get("ci95_low")
                        else float(ci["ci95_low"])
                    ),
                    "ci95_high": (
                        None
                        if not ci.get("ci95_high")
                        else float(ci["ci95_high"])
                    ),
                    "avg_questions": float(row["avg_questions"]),
                    "avg_total_tokens": avg_tokens,
                    "pass_per_1k_tokens": (
                        float(cost["pass_per_1k_tokens"])
                        if cost and cost.get("pass_per_1k_tokens")
                        else (1000.0 * pass_at_1 / avg_tokens if avg_tokens > 0 else 0.0)
                    ),
                    "delta_pass_vs_oneshot": (
                        float(cost["delta_pass_vs_oneshot"])
                        if cost and cost.get("delta_pass_vs_oneshot")
                        else 0.0
                    ),
                }
            )

    comparison_rows.sort(
        key=lambda x: (x["dataset"], x["profile"], _strategy_sort_key(str(x["strategy"])))
    )
    budget_rows.sort(
        key=lambda x: (x["dataset"], x["profile"], _strategy_sort_key(str(x["strategy"])), x["budget_tokens"])
    )
    _append_pareto_flags(comparison_rows)
    return comparison_rows, budget_rows


def _strategy_sort_key(strategy: str) -> tuple[int, str]:
    order = {
        "one-shot": 0,
        "random-tests": 1,
        "eig-tests": 2,
        "self-consistency": 3,
        "repair": 4,
        "ticode-tests": 5,
    }
    return (order.get(strategy, 99), strategy)


def _append_pareto_flags(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["profile"])), []).append(row)
    for _, points in grouped.items():
        for point in points:
            point["pareto_efficient"] = True
        for i, a in enumerate(points):
            for j, b in enumerate(points):
                if i == j:
                    continue
                dominates = (
                    float(b["avg_total_tokens"]) <= float(a["avg_total_tokens"])
                    and float(b["avg_questions"]) <= float(a["avg_questions"])
                    and float(b["pass_at_1"]) >= float(a["pass_at_1"])
                    and (
                        float(b["avg_total_tokens"]) < float(a["avg_total_tokens"])
                        or float(b["avg_questions"]) < float(a["avg_questions"])
                        or float(b["pass_at_1"]) > float(a["pass_at_1"])
                    )
                )
                if dominates:
                    a["pareto_efficient"] = False
                    break


def _infer_default_run_id() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    args = parse_args()
    load_dotenv()
    profiles = load_profiles(Path(args.profiles_config))
    run_id = args.run_id.strip() or _infer_default_run_id()
    run_root = Path(args.results_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    configs = {
        "mbpp": Path(args.mbpp_config),
        "humaneval": Path(args.humaneval_config),
    }
    run_index: list[tuple[str, str, Path]] = []

    for profile_name in args.profiles:
        if profile_name not in profiles:
            raise ValueError(f"Unknown profile '{profile_name}' in {args.profiles_config}")
        raw_profile = profiles[profile_name]
        raw_env = raw_profile.get("env", {})
        if not isinstance(raw_env, dict):
            raise ValueError(f"profile '{profile_name}' env must be a mapping.")
        profile_env = expand_profile_env(raw_env)
        print(f"\n=== Profile: {profile_name} ===")
        for dataset in args.datasets:
            print(f"--- Dataset: {dataset} ---")
            merged_path, summary_dir = run_dataset_profile(
                profile_name=profile_name,
                profile_env=profile_env,
                dataset=dataset,
                config_path=configs[dataset],
                run_root=run_root,
                strategies=list(args.strategies),
                num_shards=args.num_shards,
                resume=bool(args.resume),
                skip_mbppplus=bool(args.skip_mbppplus),
                max_examples=int(args.max_examples),
                pipeline_overrides=list(args.pipeline_overrides),
                parallel_shards=bool(args.parallel_shards),
                dry_run=bool(args.dry_run),
            )
            if not args.dry_run:
                run_index.append((profile_name, dataset, summary_dir))
                print(f"Merged results: {merged_path}")
                print(f"Summary dir: {summary_dir}")

    if args.dry_run:
        print(f"\nDry run complete. Planned outputs under: {run_root}")
        return

    comparison_rows, budget_rows = build_cross_model_summary(run_index)
    comparison_csv = run_root / "model_comparison.csv"
    comparison_md = run_root / "model_comparison.md"
    budget_csv = run_root / "model_budget_comparison.csv"
    write_csv(comparison_csv, comparison_rows)
    write_csv(budget_csv, budget_rows)
    write_markdown(comparison_md, comparison_rows)

    print("\nWrote model matrix outputs:")
    print(f"- {comparison_csv}")
    print(f"- {comparison_md}")
    print(f"- {budget_csv}")
    if comparison_rows:
        eig_rows = [x for x in comparison_rows if x["strategy"] == "eig-tests"]
        if eig_rows:
            best = max(eig_rows, key=lambda x: (x["pass_at_1"], -x["avg_total_tokens"]))
            print(
                "Best EIG row by pass@1: "
                f"profile={best['profile']} dataset={best['dataset']} "
                f"pass@1={best['pass_at_1']:.3f} avg_tokens={best['avg_total_tokens']:.1f}"
            )


if __name__ == "__main__":
    main()
