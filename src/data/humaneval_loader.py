from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from data.task_schema import BenchmarkTask


def load_humaneval_json(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_hidden_assertions(test_source: str, entry_point: str) -> list[str]:
    try:
        tree = ast.parse(test_source, mode="exec")
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            out: list[str] = []
            for stmt in node.body:
                if isinstance(stmt, ast.Assert):
                    expr = ast.unparse(stmt.test).replace("candidate(", f"{entry_point}(")
                    out.append(f"assert {expr}")
            return out
    return []


def build_humaneval_variants_rows(
    original: list[dict[str, Any]],
    incomplete: list[dict[str, Any]],
    ambiguous: list[dict[str, Any]],
    contradictory: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cond_maps: dict[str, dict[str, dict[str, Any]]] = {
        "original": {str(x["task_id"]): x for x in original},
        "incomplete": {str(x["task_id"]): x for x in incomplete},
        "ambiguous": {str(x["task_id"]): x for x in ambiguous},
    }
    if contradictory:
        cond_maps["contradictory"] = {str(x["task_id"]): x for x in contradictory}

    canonical = cond_maps["original"]
    rows: list[dict[str, Any]] = []
    for task_key, base in sorted(canonical.items(), key=lambda kv: kv[0]):
        entry_point = str(base["entry_point"])
        hidden_tests = extract_hidden_assertions(str(base["test"]), entry_point=entry_point)
        visible_tests = hidden_tests[:1]
        for condition, cond_map in cond_maps.items():
            item = cond_map.get(task_key)
            if not item:
                continue
            rows.append(
                {
                    "benchmark": "humaneval",
                    "task_id": _numeric_task_id(task_key),
                    "task_key": task_key,
                    "condition": condition,
                    "prompt": str(item["prompt"]),
                    "oracle_code": _build_oracle_code(
                        prompt=str(base["prompt"]),
                        canonical_solution=str(base["canonical_solution"]),
                    ),
                    "visible_tests": visible_tests,
                    "hidden_tests": hidden_tests,
                    "function_name": entry_point,
                }
            )
    return rows


def load_variant_file(path: str) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tasks.append(
                BenchmarkTask(
                    benchmark=str(row.get("benchmark", "humaneval")),
                    task_id=int(row["task_id"]),
                    task_key=str(row.get("task_key", row["task_id"])),
                    condition=str(row["condition"]),
                    prompt=str(row["prompt"]),
                    oracle_code=str(row["oracle_code"]),
                    visible_tests=list(row["visible_tests"]),
                    hidden_tests=list(row["hidden_tests"]),
                    function_name=str(row["function_name"]),
                )
            )
    return tasks


def _numeric_task_id(task_key: str) -> int:
    if "/" in task_key:
        _, suffix = task_key.split("/", 1)
        if suffix.isdigit():
            return int(suffix)
    return int(task_key)


def _build_oracle_code(prompt: str, canonical_solution: str) -> str:
    sol = canonical_solution
    if sol and not sol.startswith((" ", "\t")):
        sol = "    " + sol
    return f"{prompt.rstrip()}\n{sol}\n"
