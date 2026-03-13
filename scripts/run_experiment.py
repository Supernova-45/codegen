#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Iterable, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ModelConfig, ensure_output_path, load_config
from data.humaneval_loader import load_variant_file as load_humaneval_variants
from data.mbppplus_loader import load_mbppplus_tests
from data.mbpp_loader import filter_tasks, load_variant_file
from execution.adapter import build_effective_code
from execution.sandbox import run_test_script
from models.openai_compatible import OpenAICompatibleClient
from models.routed_client import RoutedModelClient
from pipeline.run_problem import run_problem


def _env_with_legacy(name: str) -> tuple[str, str]:
    value = os.environ.get(name, "").strip()
    if value:
        return value, name
    if name.startswith("CODEGEN_"):
        legacy_name = f"CLARIFYCODE_{name[len('CODEGEN_'):]}"
        legacy_value = os.environ.get(legacy_name, "").strip()
        if legacy_value:
            return legacy_value, legacy_name
    return "", ""


def _split_env_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.replace(";", ",").replace(" ", ",").split(",") if x.strip()]


def _resolve_api_key_pool(cli_env_names: list[str]) -> tuple[list[str], list[str], str]:
    # Priority: explicit CLI list -> .env env-name list -> .env direct key pool.
    env_names = list(cli_env_names)
    source = "--api-key-env-pool"
    if not env_names:
        raw_pool_names, source_name = _env_with_legacy("CODEGEN_API_KEY_ENV_POOL")
        from_env_names = _split_env_list(raw_pool_names)
        if from_env_names:
            env_names = from_env_names
            source = source_name
    if env_names:
        keys: list[str] = []
        for env_name in env_names:
            value = os.environ.get(env_name, "").strip()
            if not value:
                raise ValueError(f"API key env var '{env_name}' is missing or empty.")
            keys.append(value)
        return keys, env_names, source

    raw_key_pool, source_name = _env_with_legacy("CODEGEN_API_KEY_POOL")
    pooled_keys = _split_env_list(raw_key_pool)
    if pooled_keys:
        return pooled_keys, [], source_name
    return [], [], ""


def _optional_testgen_model_cfg(primary: ModelConfig) -> ModelConfig | None:
    testgen_api_key, _ = _env_with_legacy("CODEGEN_TESTGEN_API_KEY")
    if not testgen_api_key:
        return None
    base_url, _ = _env_with_legacy("CODEGEN_TESTGEN_BASE_URL")
    model_name, _ = _env_with_legacy("CODEGEN_TESTGEN_MODEL")
    timeout_raw, _ = _env_with_legacy("CODEGEN_TESTGEN_REQUEST_TIMEOUT_S")
    temperature_raw, _ = _env_with_legacy("CODEGEN_TESTGEN_TEMPERATURE")
    if not base_url:
        base_url = primary.base_url
    if not model_name:
        model_name = primary.model
    timeout_s = int(timeout_raw) if timeout_raw else primary.request_timeout_s
    temperature = float(temperature_raw) if temperature_raw else primary.temperature
    return ModelConfig(
        base_url=base_url,
        api_key=testgen_api_key,
        model=model_name,
        temperature=temperature,
        request_timeout_s=timeout_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["one-shot", "random-tests", "eig-tests", "self-consistency", "repair"],
    )
    parser.add_argument("--output-file")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional override for config.max_examples (0 keeps config value).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to output file and skip already-completed rows found in it.",
    )
    parser.add_argument(
        "--skip-existing-files",
        nargs="*",
        default=[],
        help=(
            "Optional JSONL result files to scan for completed (strategy, task_id, condition) "
            "rows and skip re-running them."
        ),
    )
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
    resolved_pool_keys, resolved_pool_env_names, pool_source = _resolve_api_key_pool(
        args.api_key_env_pool
    )
    if resolved_pool_keys:
        # load_config validates CODEGEN_API_KEY; seed it from pool so config loading succeeds.
        os.environ["CODEGEN_API_KEY"] = resolved_pool_keys[0]
    cfg = load_config(args.config)
    if args.output_file:
        cfg.output_file = args.output_file
    if args.max_examples > 0:
        cfg.max_examples = args.max_examples
    if args.skip_mbppplus:
        cfg.mbppplus_enabled = False
    shard_api_keys: Optional[list[str]] = None
    if resolved_pool_keys:
        if args.num_shards > 1:
            selected_idx = args.shard_index % len(resolved_pool_keys)
            shard_api_keys = [resolved_pool_keys[selected_idx]]
            cfg.model.api_key = shard_api_keys[0]
            if resolved_pool_env_names:
                selected_env = resolved_pool_env_names[selected_idx]
                print(
                    f"Shard {args.shard_index + 1}/{args.num_shards} pinned to API key from "
                    f"${selected_env} ({pool_source})"
                )
            else:
                print(
                    f"Shard {args.shard_index + 1}/{args.num_shards} pinned to pooled key "
                    f"index {selected_idx + 1}/{len(resolved_pool_keys)} ({pool_source})"
                )
        else:
            shard_api_keys = resolved_pool_keys
            cfg.model.api_key = shard_api_keys[0]
            print(
                f"Single process rotating across {len(shard_api_keys)} API keys "
                f"from {pool_source}"
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

    if cfg.benchmark == "humaneval":
        all_tasks = load_humaneval_variants(cfg.variants_path)
    elif cfg.benchmark == "mbpp":
        all_tasks = load_variant_file(cfg.variants_path)
    else:
        raise ValueError(f"Unsupported dataset benchmark: {cfg.benchmark}")
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
    codegen_client = OpenAICompatibleClient(cfg.model, api_keys=shard_api_keys)
    testgen_cfg = _optional_testgen_model_cfg(cfg.model)
    testgen_client = OpenAICompatibleClient(testgen_cfg) if testgen_cfg else None
    model = RoutedModelClient(code_client=codegen_client, test_client=testgen_client)
    if testgen_cfg:
        print(
            "Using separate test-generation model: "
            f"{testgen_cfg.model} @ {testgen_cfg.base_url}"
        )
    output_path = ensure_output_path(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {output_path}")
    completed_keys = _load_completed_keys(
        [output_path] if args.resume else [],
        extra_paths=[Path(x) for x in args.skip_existing_files],
    )
    mbppplus_by_task: dict[int, str] = {}
    if cfg.mbppplus_enabled and cfg.benchmark == "mbpp":
        mbppplus_rows = load_mbppplus_tests(
            dataset=cfg.mbppplus_dataset,
            split=cfg.mbppplus_split,
        )
        mbppplus_by_task = {task_id: row.test_script for task_id, row in mbppplus_rows.items()}
        print(f"Loaded {len(mbppplus_by_task)} MBPP+ tasks from {cfg.mbppplus_dataset}/{cfg.mbppplus_split}")

    run_count = 0
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for strategy in args.strategies:
            for task in tasks:
                key = _result_key(strategy=strategy, task_id=task.task_id, condition=task.condition)
                if key in completed_keys:
                    continue
                run_started_at = time.perf_counter()
                result = run_problem(
                    task=task,
                    strategy=strategy,
                    model=model,
                    cfg=cfg.pipeline,
                    seed=cfg.seed,
                )
                if cfg.mbppplus_enabled and cfg.benchmark == "mbpp":
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
                result_row = result.to_dict()
                result_row["codegen_model"] = cfg.model.model
                result_row["codegen_base_url"] = cfg.model.base_url
                if testgen_cfg:
                    result_row["testgen_model"] = testgen_cfg.model
                    result_row["testgen_base_url"] = testgen_cfg.base_url
                else:
                    result_row["testgen_model"] = None
                    result_row["testgen_base_url"] = None
                f.write(json.dumps(result_row) + "\n")
                f.flush()
                os.fsync(f.fileno())
                run_count += 1
                completed_keys.add(key)
                if args.live_profiler:
                    _print_live_profile(
                        run_count=run_count,
                        result=result_row,
                        run_elapsed_s=run_elapsed_s,
                    )
                if run_count % 10 == 0:
                    print(f"Completed {run_count} runs")

    print(f"Saved results to {output_path}")


def _result_key(strategy: str, task_id: int, condition: str) -> tuple[str, int, str]:
    return (strategy, int(task_id), str(condition))


def _load_completed_keys(
    primary_paths: Iterable[Path],
    extra_paths: Optional[Iterable[Path]] = None,
) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    paths = list(primary_paths) + list(extra_paths or [])
    seen_paths: set[Path] = set()
    for path in paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except json.JSONDecodeError:
                    # Ignore truncated/corrupt lines from interrupted runs.
                    continue
                strategy = row.get("strategy")
                task_id = row.get("task_id")
                condition = row.get("condition")
                if strategy is None or task_id is None or condition is None:
                    continue
                try:
                    keys.add(_result_key(strategy=str(strategy), task_id=int(task_id), condition=str(condition)))
                except (TypeError, ValueError):
                    continue
    return keys


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
