#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ensure_output_path, load_config
from data.mbpp_loader import filter_tasks, load_variant_file
from models.openai_compatible import OpenAICompatibleClient
from pipeline.run_problem import run_problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["one-shot", "random-tests", "eig-tests"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg.seed)

    all_tasks = load_variant_file(cfg.variants_path)
    tasks = filter_tasks(
        all_tasks,
        conditions=cfg.conditions,
        max_examples=cfg.max_examples,
        seed=cfg.seed,
        shuffle=cfg.shuffle,
    )
    model = OpenAICompatibleClient(cfg.model)
    output_path = ensure_output_path(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for strategy in args.strategies:
            for task in tasks:
                result = run_problem(
                    task=task,
                    strategy=strategy,
                    model=model,
                    cfg=cfg.pipeline,
                    seed=cfg.seed,
                )
                f.write(json.dumps(result.to_dict()) + "\n")
                run_count += 1
                if run_count % 10 == 0:
                    print(f"Completed {run_count} runs")

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
