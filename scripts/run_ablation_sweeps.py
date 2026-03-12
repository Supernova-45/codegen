#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_sweep_jobs(*, include_sensitivity: bool) -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    # -------------------------------------------------------------------------
    # Core ablations should be *paired* so the comparison is complete:
    # - scorer: EIG vs TiCode (same budget + stopping)
    # - update: soft vs hard-prune (same scorer + stopping)
    # - stopping: VOI on vs off (same scorer + update)
    #
    # We keep reprompt disabled for these ablations to isolate the effect.
    # -------------------------------------------------------------------------
    jobs.append(
        (
            "ablation_scorer_eig",
            [
                "query_scorer=eig",
                "disable_voi_stop=true",
                "run_reprompt=false",
            ],
        )
    )
    jobs.append(
        (
            "ablation_scorer_ticode",
            [
                "query_scorer=ticode",
                "disable_voi_stop=true",
                "run_reprompt=false",
            ],
        )
    )
    jobs.append(
        (
            "ablation_update_soft",
            [
                "hard_prune_update=false",
                "query_scorer=eig",
                "disable_voi_stop=true",
                "run_reprompt=false",
            ],
        )
    )
    jobs.append(
        (
            "ablation_update_hard_prune",
            [
                "hard_prune_update=true",
                "query_scorer=eig",
                "disable_voi_stop=true",
                "run_reprompt=false",
            ],
        )
    )
    jobs.append(
        (
            "ablation_stop_voi_on",
            [
                "disable_voi_stop=false",
                "query_scorer=eig",
                "hard_prune_update=false",
                "run_reprompt=false",
            ],
        )
    )
    jobs.append(
        (
            "ablation_stop_voi_off",
            [
                "disable_voi_stop=true",
                "query_scorer=eig",
                "hard_prune_update=false",
                "run_reprompt=false",
            ],
        )
    )

    if include_sensitivity:
        for n_candidates, tests_per_round, k_max in product([8, 16, 24], [8, 16], [2, 4]):
            jobs.append(
                (
                    f"sens_n{n_candidates}_t{tests_per_round}_k{k_max}",
                    [
                        f"n_candidates={n_candidates}",
                        f"tests_per_round={tests_per_round}",
                        f"k_max={k_max}",
                    ],
                )
            )
    return jobs


def _pair_summaries(output_subdir: str) -> list[tuple[str, str, str, str, str]]:
    """
    Returns (title, left_name, right_name, left_label, right_label) for core ablation pairs.
    Names refer to the JSONL job names (without directory / extension).
    """
    return [
        ("Ablation: Query Scorer", "ablation_scorer_eig", "ablation_scorer_ticode", "EIG scorer", "TiCode scorer"),
        (
            "Ablation: Belief Update",
            "ablation_update_soft",
            "ablation_update_hard_prune",
            "Soft update",
            "Hard prune",
        ),
        ("Ablation: Stopping Rule", "ablation_stop_voi_on", "ablation_stop_voi_off", "VOI on", "VOI off"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-subdir", default="ablations")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only the 3 core ablation pairs (6 jobs). Skips sensitivity sweeps.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each job's JSONL output if it already exists.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="After each job, write results/<subdir>/<name>_summary/comparison_table.md.",
    )
    parser.add_argument(
        "--summarize-pairs",
        action="store_true",
        help="Write one combined comparison_table.md per core ablation study.",
    )
    parser.add_argument(
        "--skip-mbppplus",
        action="store_true",
        help="Disable MBPP+ rescoring for faster sweeps.",
    )
    parser.add_argument(
        "--live-profiler",
        action="store_true",
        help="Print per-run profiling lines (useful for long sweeps).",
    )
    args = parser.parse_args()

    jobs = build_sweep_jobs(include_sensitivity=not args.core_only)
    for name, overrides in jobs:
        output_file = f"{args.output_subdir}/{name}.jsonl"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            "--config",
            args.config,
            "--strategies",
            "eig-tests",
            "--output-file",
            output_file,
            "--pipeline-overrides",
            *overrides,
        ]
        if args.resume:
            cmd.append("--resume")
        if args.skip_mbppplus:
            cmd.append("--skip-mbppplus")
        if args.live_profiler:
            cmd.append("--live-profiler")
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, cwd=str(ROOT), check=True)
            if args.summarize:
                out_dir = f"results/{args.output_subdir}/{name}_summary"
                summarize_cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_results.py"),
                    "--results",
                    f"results/{output_file}",
                    "--output-dir",
                    out_dir,
                ]
                print(" ".join(summarize_cmd))
                subprocess.run(summarize_cmd, cwd=str(ROOT), check=True)

    if args.execute and args.summarize_pairs:
        for title, left_name, right_name, left_label, right_label in _pair_summaries(args.output_subdir):
            left_path = f"results/{args.output_subdir}/{left_name}.jsonl"
            right_path = f"results/{args.output_subdir}/{right_name}.jsonl"
            out_path = f"results/{args.output_subdir}/{left_name}_VS_{right_name}/comparison_table.md"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "summarize_ablation_pair.py"),
                "--title",
                title,
                "--left-results",
                left_path,
                "--right-results",
                right_path,
                "--left-label",
                left_label,
                "--right-label",
                right_label,
                "--output-path",
                out_path,
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
