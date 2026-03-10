#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Token-aware random search for EIG hyperparameters against "
            "one-shot/random baselines."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-trials", type=int, default=24)
    parser.add_argument("--max-examples", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", default=str(ROOT / "results" / "eig_tuning"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--token-budget", type=float, default=8000.0)
    parser.add_argument(
        "--objective-gain-weight",
        type=float,
        default=1.0,
        help="Weight for absolute pass@1 gain over max(one-shot, random).",
    )
    parser.add_argument(
        "--objective-efficiency-weight",
        type=float,
        default=0.35,
        help="Weight for gain-per-1k-token term.",
    )
    parser.add_argument(
        "--objective-budget-penalty",
        type=float,
        default=0.75,
        help="Penalty weight if avg tokens exceed token budget.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Shard each experiment run across this many worker processes.",
    )
    parser.add_argument(
        "--parallel-shards",
        action="store_true",
        help="Run shard processes concurrently (faster, higher endpoint load/cost).",
    )
    parser.add_argument(
        "--skip-mbppplus",
        action="store_true",
        help="Skip MBPP+ re-scoring during tuning for faster iteration.",
    )
    parser.add_argument(
        "--baseline-max-examples",
        type=int,
        default=0,
        help="Optional smaller max_examples for baseline run only (0 uses --max-examples).",
    )
    parser.add_argument(
        "--fast-search-space",
        action="store_true",
        help="Sample a smaller, cheaper EIG hyperparameter space.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _infer_default_run_id() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _run(
    cmd: list[str],
    *,
    dry_run: bool,
) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _run_and_summarize(
    *,
    config: str,
    strategies: list[str],
    output_jsonl: Path,
    summary_dir: Path,
    max_examples: int,
    overrides: list[str] | None,
    num_shards: int,
    parallel_shards: bool,
    skip_mbppplus: bool,
    dry_run: bool,
) -> None:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    if num_shards == 1:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            "--config",
            config,
            "--strategies",
            *strategies,
            "--output-file",
            str(output_jsonl),
            "--max-examples",
            str(max_examples),
        ]
        if overrides:
            cmd.extend(["--pipeline-overrides", *overrides])
        if skip_mbppplus:
            cmd.append("--skip-mbppplus")
        _run(cmd, dry_run=dry_run)
    else:
        shard_dir = output_jsonl.parent / f"{output_jsonl.stem}_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_files: list[Path] = []
        shard_cmds: list[list[str]] = []
        for shard_idx in range(num_shards):
            shard_out = shard_dir / f"shard_{shard_idx}.jsonl"
            shard_files.append(shard_out)
            shard_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_experiment.py"),
                "--config",
                config,
                "--strategies",
                *strategies,
                "--output-file",
                str(shard_out),
                "--max-examples",
                str(max_examples),
                "--num-shards",
                str(num_shards),
                "--shard-index",
                str(shard_idx),
            ]
            if overrides:
                shard_cmd.extend(["--pipeline-overrides", *overrides])
            if skip_mbppplus:
                shard_cmd.append("--skip-mbppplus")
            shard_cmds.append(shard_cmd)

        for shard_cmd in shard_cmds:
            print(" ".join(shard_cmd))
        if not dry_run:
            if parallel_shards:
                procs = [subprocess.Popen(cmd, cwd=str(ROOT)) for cmd in shard_cmds]
                failed = False
                for proc in procs:
                    if proc.wait() != 0:
                        failed = True
                if failed:
                    raise RuntimeError("One or more shard processes failed during optimize_eig run.")
            else:
                for shard_cmd in shard_cmds:
                    subprocess.run(shard_cmd, cwd=str(ROOT), check=True)

            with output_jsonl.open("w", encoding="utf-8") as out_f:
                for shard_file in shard_files:
                    if not shard_file.exists():
                        continue
                    with shard_file.open("r", encoding="utf-8", errors="replace") as in_f:
                        for line in in_f:
                            s = line.strip()
                            if s:
                                out_f.write(s + "\n")

    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_results.py"),
            "--results",
            str(output_jsonl),
            "--output-dir",
            str(summary_dir),
        ],
        dry_run=dry_run,
    )


def _sample_overrides(
    rng: random.Random,
    *,
    fast_search_space: bool,
) -> dict[str, Any]:
    if fast_search_space:
        n_candidates = rng.choice([6, 8, 10, 12])
        tests_per_round = rng.choice([4, 6, 8, 10])
        k_max = rng.choice([1, 2, 3])
        epsilon_choices = [0.01, 0.02, 0.04]
        gamma_choices = [0.82, 0.88, 0.92, 0.95]
        min_q_choices = [1, 2]
        min_cov_choices = [0.55, 0.65, 0.75]
        min_eig_choices = [0.0, 0.01, 0.02, 0.03]
        regen_choices = [1, 2, 3]
        eig_q_choices = [1]
        cand_temp_choices = [0.6, 0.7, 0.8]
        reprompt_temp_choices = [0.05, 0.1, 0.2]
        reprompt_min_q_choices = [1, 2]
        reprompt_match_choices = [0.5, 0.67, 0.75]
        discrim_choices = [0.35, 0.5, 0.65]
        err_penalty_choices = [0.5, 0.7, 0.9]
        undefined_like_choices = [0.2, 0.4, 0.6]
        shared_pool_size_choices = [24, 32, 48]
        shared_pool_regen_choices = [1, 2]
    else:
        n_candidates = rng.choice([10, 12, 16, 20, 24, 28])
        tests_per_round = rng.choice([8, 12, 16, 20, 24])
        k_max = rng.choice([2, 3, 4, 5])
        epsilon_choices = [0.01, 0.02, 0.04, 0.06]
        gamma_choices = [0.82, 0.88, 0.92, 0.95, 0.97]
        min_q_choices = [1, 2, 3]
        min_cov_choices = [0.55, 0.65, 0.75, 0.8]
        min_eig_choices = [0.0, 0.01, 0.02, 0.03, 0.05]
        regen_choices = [2, 4, 6, 8]
        eig_q_choices = [1, 2]
        cand_temp_choices = [0.6, 0.7, 0.8, 0.9]
        reprompt_temp_choices = [0.05, 0.1, 0.2, 0.3]
        reprompt_min_q_choices = [1, 2, 3]
        reprompt_match_choices = [0.5, 0.67, 0.75, 0.85]
        discrim_choices = [0.35, 0.5, 0.65, 0.8]
        err_penalty_choices = [0.3, 0.5, 0.7, 0.9]
        undefined_like_choices = [0.2, 0.4, 0.6, 0.85]
        shared_pool_size_choices = [32, 48, 64, 96]
        shared_pool_regen_choices = [1, 2, 3]

    hard_prune = rng.choice([True, False])
    disable_voi_stop = rng.choice([False, True])
    force_full_budget = False if disable_voi_stop is False else rng.choice([False, True])
    shared_pool = rng.choice([False, True])
    return {
        "n_candidates": n_candidates,
        "tests_per_round": tests_per_round,
        "k_max": k_max,
        "epsilon": rng.choice(epsilon_choices),
        "gamma": rng.choice(gamma_choices),
        "min_questions_if_valid": rng.choice(min_q_choices),
        "min_valid_candidate_coverage": rng.choice(min_cov_choices),
        "min_eig_score": rng.choice(min_eig_choices),
        "max_test_regen_attempts": rng.choice(regen_choices),
        "eig_questions_per_round": rng.choice(eig_q_choices),
        "candidate_temperature": rng.choice(cand_temp_choices),
        "reprompt_temperature": rng.choice(reprompt_temp_choices),
        "run_reprompt": rng.choice([True, False]),
        "reprompt_min_questions": rng.choice(reprompt_min_q_choices),
        "reprompt_require_mixed_outcomes": rng.choice([True, False]),
        "reprompt_min_constraint_match_rate": rng.choice(reprompt_match_choices),
        "eig_discriminative_weight": rng.choice(discrim_choices),
        "eig_runtime_error_penalty": rng.choice(err_penalty_choices),
        "undefined_outcome_likelihood": rng.choice(undefined_like_choices),
        "skip_posterior_update_on_undefined_oracle": rng.choice([True, False]),
        "disable_voi_stop": disable_voi_stop,
        "force_full_question_budget": force_full_budget,
        "shared_test_pool": shared_pool,
        "shared_test_pool_size": rng.choice(shared_pool_size_choices),
        "shared_test_pool_regen_rounds": rng.choice(shared_pool_regen_choices),
        "hard_prune_update": hard_prune,
    }


def _to_override_list(overrides: dict[str, Any]) -> list[str]:
    vals: list[str] = []
    for key in sorted(overrides.keys()):
        value = overrides[key]
        if isinstance(value, bool):
            txt = "true" if value else "false"
        else:
            txt = str(value)
        vals.append(f"{key}={txt}")
    return vals


def _extract_overall_metrics(summary_dir: Path, strategy: str) -> dict[str, float] | None:
    rows = _read_csv(summary_dir / "summary_overall_strategy.csv")
    for row in rows:
        if row.get("strategy") == strategy:
            return {
                "pass_at_1": float(row["pass_at_1"]),
                "avg_questions": float(row["avg_questions"]),
                "avg_total_tokens": float(row["avg_total_tokens"]),
            }
    return None


def _score_trial(
    eig: dict[str, float],
    baseline_one_shot: dict[str, float],
    baseline_random: dict[str, float],
    token_budget: float,
    gain_weight: float,
    efficiency_weight: float,
    budget_penalty: float,
) -> dict[str, float]:
    baseline_pass = max(baseline_one_shot["pass_at_1"], baseline_random["pass_at_1"])
    gain = eig["pass_at_1"] - baseline_pass
    gain_per_1k = (1000.0 * gain / eig["avg_total_tokens"]) if eig["avg_total_tokens"] > 0 else 0.0
    budget_over = max(0.0, eig["avg_total_tokens"] - token_budget) / max(1.0, token_budget)
    score = (gain_weight * gain) + (efficiency_weight * gain_per_1k) - (budget_penalty * budget_over)
    return {
        "baseline_pass": baseline_pass,
        "gain": gain,
        "gain_per_1k_tokens": gain_per_1k,
        "budget_over_ratio": budget_over,
        "objective": score,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    rng = random.Random(args.seed)
    run_id = args.run_id.strip() or _infer_default_run_id()
    run_root = Path(args.results_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    baseline_examples = args.baseline_max_examples if args.baseline_max_examples > 0 else args.max_examples

    baseline_jsonl = run_root / "baseline.jsonl"
    baseline_summary = run_root / "baseline_summary"
    print("\n=== Baseline run (one-shot + random-tests) ===")
    _run_and_summarize(
        config=args.config,
        strategies=["one-shot", "random-tests"],
        output_jsonl=baseline_jsonl,
        summary_dir=baseline_summary,
        max_examples=baseline_examples,
        overrides=None,
        num_shards=args.num_shards,
        parallel_shards=args.parallel_shards,
        skip_mbppplus=bool(args.skip_mbppplus),
        dry_run=bool(args.dry_run),
    )

    if args.dry_run:
        print(f"\nDry run complete. Planned outputs under: {run_root}")
        return

    baseline_one_shot = _extract_overall_metrics(baseline_summary, "one-shot")
    baseline_random = _extract_overall_metrics(baseline_summary, "random-tests")
    if baseline_one_shot is None or baseline_random is None:
        raise RuntimeError(
            "Baseline summary is missing one-shot/random-tests rows. "
            f"Check: {baseline_summary / 'summary_overall_strategy.csv'}"
        )
    print(
        "Baselines: one-shot pass@1={:.3f}, random-tests pass@1={:.3f}".format(
            baseline_one_shot["pass_at_1"],
            baseline_random["pass_at_1"],
        )
    )

    leaderboard: list[dict[str, Any]] = []
    seen_override_strings: set[str] = set()
    for trial_idx in range(args.num_trials):
        overrides = _sample_overrides(rng, fast_search_space=bool(args.fast_search_space))
        override_list = _to_override_list(overrides)
        override_sig = "|".join(override_list)
        if override_sig in seen_override_strings:
            continue
        seen_override_strings.add(override_sig)

        trial_name = f"trial_{trial_idx:03d}"
        trial_jsonl = run_root / "trials" / f"{trial_name}.jsonl"
        trial_summary = run_root / "trials" / f"{trial_name}_summary"
        print(f"\n=== {trial_name} ===")
        _run_and_summarize(
            config=args.config,
            strategies=["eig-tests"],
            output_jsonl=trial_jsonl,
            summary_dir=trial_summary,
            max_examples=args.max_examples,
            overrides=override_list,
            num_shards=args.num_shards,
            parallel_shards=args.parallel_shards,
            skip_mbppplus=bool(args.skip_mbppplus),
            dry_run=False,
        )
        eig_metrics = _extract_overall_metrics(trial_summary, "eig-tests")
        if eig_metrics is None:
            row = {
                "trial": trial_name,
                "pass_at_1": None,
                "avg_questions": None,
                "avg_total_tokens": None,
                "baseline_pass": max(baseline_one_shot["pass_at_1"], baseline_random["pass_at_1"]),
                "gain_vs_best_baseline": None,
                "gain_per_1k_tokens": None,
                "budget_over_ratio": None,
                "objective": float("-inf"),
                "overrides_json": json.dumps(overrides, sort_keys=True),
                "overrides_cli": " ".join(override_list),
                "result_file": str(trial_jsonl),
                "status": "missing_eig_row",
            }
            leaderboard.append(row)
            leaderboard.sort(
                key=lambda x: float(x["objective"]) if x["objective"] is not None else float("-inf"),
                reverse=True,
            )
            _write_csv(run_root / "leaderboard.csv", leaderboard)
            print(
                f"Skipping {trial_name}: missing eig-tests row in "
                f"{trial_summary / 'summary_overall_strategy.csv'}"
            )
            continue
        score = _score_trial(
            eig=eig_metrics,
            baseline_one_shot=baseline_one_shot,
            baseline_random=baseline_random,
            token_budget=args.token_budget,
            gain_weight=args.objective_gain_weight,
            efficiency_weight=args.objective_efficiency_weight,
            budget_penalty=args.objective_budget_penalty,
        )
        row = {
            "trial": trial_name,
            "pass_at_1": eig_metrics["pass_at_1"],
            "avg_questions": eig_metrics["avg_questions"],
            "avg_total_tokens": eig_metrics["avg_total_tokens"],
            "baseline_pass": score["baseline_pass"],
            "gain_vs_best_baseline": score["gain"],
            "gain_per_1k_tokens": score["gain_per_1k_tokens"],
            "budget_over_ratio": score["budget_over_ratio"],
            "objective": score["objective"],
            "overrides_json": json.dumps(overrides, sort_keys=True),
            "overrides_cli": " ".join(override_list),
            "result_file": str(trial_jsonl),
            "status": "ok",
        }
        leaderboard.append(row)
        leaderboard.sort(key=lambda x: float(x["objective"]), reverse=True)
        _write_csv(run_root / "leaderboard.csv", leaderboard)
        best = leaderboard[0]
        print(
            "Current best: trial={} objective={:.4f} pass@1={:.3f} avg_tokens={:.1f}".format(
                best["trial"],
                float(best["objective"]),
                float(best["pass_at_1"]),
                float(best["avg_total_tokens"]),
            )
        )

    if not leaderboard:
        raise RuntimeError("No trials executed.")
    best = leaderboard[0]
    best_overrides = json.loads(str(best["overrides_json"]))
    best_cli = str(best["overrides_cli"])
    (run_root / "best_overrides.json").write_text(
        json.dumps(best_overrides, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_root / "best_overrides.txt").write_text(best_cli + "\n", encoding="utf-8")

    print("\nBest trial summary:")
    print(
        "- trial={trial} objective={objective:.4f} pass@1={pass_at_1:.3f} "
        "avg_tokens={avg_total_tokens:.1f} gain_vs_best_baseline={gain:.3f}".format(
            trial=best["trial"],
            objective=float(best["objective"]),
            pass_at_1=float(best["pass_at_1"]),
            avg_total_tokens=float(best["avg_total_tokens"]),
            gain=float(best["gain_vs_best_baseline"]),
        )
    )
    print(f"- overrides file: {run_root / 'best_overrides.json'}")
    print(f"- CLI overrides: {best_cli}")


if __name__ == "__main__":
    main()
