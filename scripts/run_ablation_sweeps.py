#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_sweep_jobs() -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
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
            "ablation_hard_prune",
            [
                "hard_prune_update=true",
                "disable_voi_stop=true",
                "run_reprompt=false",
            ],
        )
    )
    jobs.append(
        (
            "ablation_no_voi_stop",
            [
                "disable_voi_stop=true",
                "run_reprompt=true",
            ],
        )
    )

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-subdir", default="ablations")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    jobs = build_sweep_jobs()
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
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
