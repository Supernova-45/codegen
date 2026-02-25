#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

from dotenv import load_dotenv

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
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-mbppplus", action="store_true")
    parser.add_argument(
        "--live-profiler",
        action="store_true",
        help=(
            "Print per-run profiling metrics: HTTP 429 retries, timeout count, and average "
            "seconds per clarification round."
        ),
    )
    parser.add_argument(
        "--api-key-env-pool",
        nargs="*",
        default=[],
        help=(
            "Optional list of environment variable names containing API keys. "
            "If sharding is enabled, each shard uses one key by shard index; "
            "otherwise calls rotate across all provided keys."
        ),
    )
    args = parser.parse_args()
    load_dotenv()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    resolved_pool_keys: list[str] = []
    if args.api_key_env_pool:
        for env_name in args.api_key_env_pool:
            value = os.environ.get(env_name, "").strip()
            if not value:
                raise ValueError(f"API key env var '{env_name}' is missing or empty.")
            resolved_pool_keys.append(value)
        # load_config validates CLARIFYCODE_API_KEY; seed it from pool so config loading succeeds.
        os.environ["CLARIFYCODE_API_KEY"] = resolved_pool_keys[0]
    cfg = load_config(args.config)
    if args.output_file:
        cfg.output_file = args.output_file
    if args.skip_mbppplus:
        cfg.mbppplus_enabled = False
    shard_api_keys: list[str] | None = None
    if resolved_pool_keys:
        if args.num_shards > 1:
            selected_idx = args.shard_index % len(resolved_pool_keys)
            selected_env = args.api_key_env_pool[selected_idx]
            shard_api_keys = [resolved_pool_keys[selected_idx]]
            cfg.model.api_key = shard_api_keys[0]
            print(
                f"Shard {args.shard_index + 1}/{args.num_shards} pinned to API key from "
                f"${selected_env}"
            )
        else:
            shard_api_keys = resolved_pool_keys
            cfg.model.api_key = shard_api_keys[0]
            print(
                f"Single process rotating across {len(shard_api_keys)} API keys "
                f"from --api-key-env-pool"
            )
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
    if args.num_shards > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % args.num_shards == args.shard_index]
        print(
            f"Shard {args.shard_index + 1}/{args.num_shards}: "
            f"{len(tasks)} tasks selected after sharding"
        )
    model = OpenAICompatibleClient(cfg.model, api_keys=shard_api_keys)
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
                run_started_at = time.perf_counter()
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
                run_elapsed_s = time.perf_counter() - run_started_at
                f.write(json.dumps(result.to_dict()) + "\n")
                f.flush()
                run_count += 1
                if args.live_profiler:
                    _print_live_profile(
                        run_count=run_count,
                        result=result.to_dict(),
                        run_elapsed_s=run_elapsed_s,
                    )
                if run_count % 10 == 0:
                    print(f"Completed {run_count} runs")

    print(f"Saved results to {output_path}")


def _print_live_profile(run_count: int, result: dict[str, object], run_elapsed_s: float) -> None:
    model_trace = result.get("model_trace", [])
    interaction_trace = result.get("interaction_trace", [])
    eval_errors = result.get("eval_errors", [])
    mbppplus_error = str(result.get("mbppplus_error", ""))
    status_429 = 0
    timeout_count = 0
    for event in model_trace if isinstance(model_trace, list) else []:
        attempts = event.get("attempts", []) if isinstance(event, dict) else []
        for attempt in attempts:
            if isinstance(attempt, dict) and attempt.get("status_code") == 429:
                status_429 += 1
    round_count = 0
    for step in interaction_trace if isinstance(interaction_trace, list) else []:
        if not isinstance(step, dict) or "round" not in step:
            continue
        round_count += 1
        for tv in step.get("test_validation", []):
            if not isinstance(tv, dict):
                continue
            for check in tv.get("candidate_checks", []):
                if not isinstance(check, dict):
                    continue
                if check.get("first_error") == "Timeout":
                    timeout_count += 1
                if check.get("second_error") == "Timeout":
                    timeout_count += 1
    for err in eval_errors if isinstance(eval_errors, list) else []:
        if "Timeout" in str(err):
            timeout_count += 1
    if "Timeout" in mbppplus_error:
        timeout_count += 1
    avg_round_s = run_elapsed_s / max(1, round_count)
    print(
        "PROFILE run={run} strategy={strategy} task_id={task} rounds={rounds} "
        "avg_round_s={avg:.2f} run_s={run_s:.2f} retries_429={r429} timeouts={to} "
        "questions={q} pass_at_1={p1}".format(
            run=run_count,
            strategy=result.get("strategy"),
            task=result.get("task_id"),
            rounds=round_count,
            avg=avg_round_s,
            run_s=run_elapsed_s,
            r429=status_429,
            to=timeout_count,
            q=result.get("questions_asked"),
            p1=result.get("pass_at_1"),
        )
    )


if __name__ == "__main__":
    main()
