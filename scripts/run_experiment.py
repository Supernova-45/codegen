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
from data.mbppplus_loader import load_mbppplus_tests
from data.mbpp_loader import filter_tasks, load_variant_file
from execution.adapter import build_effective_code
from execution.sandbox import run_test_script
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
    parser.add_argument("--output-file")
    parser.add_argument("--pipeline-overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_file:
        cfg.output_file = args.output_file
    for raw in args.pipeline_overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid --pipeline-overrides entry: {raw}. Use key=value format.")
        key, value = raw.split("=", 1)
        if not hasattr(cfg.pipeline, key):
            raise ValueError(f"Unknown pipeline override key: {key}")
        cur = getattr(cfg.pipeline, key)
        if isinstance(cur, bool):
            parsed = value.lower() in {"1", "true", "yes", "y", "on"}
        elif isinstance(cur, int) and not isinstance(cur, bool):
            parsed = int(value)
        elif isinstance(cur, float):
            parsed = float(value)
        else:
            parsed = value
        setattr(cfg.pipeline, key, parsed)
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
    mbppplus_by_task: dict[int, str] = {}
    if cfg.mbppplus_enabled:
        mbppplus_rows = load_mbppplus_tests(
            dataset=cfg.mbppplus_dataset,
            split=cfg.mbppplus_split,
        )
        mbppplus_by_task = {task_id: row.test_script for task_id, row in mbppplus_rows.items()}
        print(f"Loaded {len(mbppplus_by_task)} MBPP+ tasks from {cfg.mbppplus_dataset}/{cfg.mbppplus_split}")

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
                if cfg.mbppplus_enabled:
                    mbppplus_script = mbppplus_by_task.get(task.task_id)
                    if mbppplus_script:
                        effective_mbpp_code, _ = build_effective_code(
                            result.final_code,
                            expected_function_name=task.function_name,
                            expected_arity=None,
                            enabled=cfg.pipeline.eval_with_adapter,
                        )
                        mbppplus_ok, mbppplus_err = run_test_script(
                            effective_mbpp_code,
                            mbppplus_script,
                            cfg.mbppplus_timeout_s,
                        )
                        result.mbppplus_pass_at_1 = mbppplus_ok
                        result.mbppplus_error = "" if mbppplus_ok else mbppplus_err
                    else:
                        result.mbppplus_pass_at_1 = None
                        result.mbppplus_error = "TaskMissingInMBPPPlus"
                f.write(json.dumps(result.to_dict()) + "\n")
                run_count += 1
                if run_count % 10 == 0:
                    print(f"Completed {run_count} runs")

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
