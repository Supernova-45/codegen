from __future__ import annotations

from dataclasses import dataclass
import ast
import json
import random
import re
from typing import Iterable


@dataclass
class MBPPTask:
    task_id: int
    condition: str
    prompt: str
    oracle_code: str
    visible_tests: list[str]
    hidden_tests: list[str]
    function_name: str


def load_variant_file(path: str) -> list[MBPPTask]:
    tasks: list[MBPPTask] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tasks.append(
                MBPPTask(
                    task_id=int(row["task_id"]),
                    condition=row["condition"],
                    prompt=row["prompt"],
                    oracle_code=row["oracle_code"],
                    visible_tests=row["visible_tests"],
                    hidden_tests=row["hidden_tests"],
                    function_name=row["function_name"],
                )
            )
    return tasks


def filter_tasks(
    tasks: Iterable[MBPPTask],
    conditions: list[str],
    max_examples: int,
    seed: int,
    shuffle: bool,
) -> list[MBPPTask]:
    filtered = [t for t in tasks if t.condition in conditions]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(filtered)
    return filtered[:max_examples]


def extract_function_name(test_lines: list[str]) -> str:
    for test in test_lines:
        m = re.search(r"assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", test)
        if m:
            return m.group(1)
    raise ValueError("Could not infer function name from tests.")


def infer_signature_hint(
    test_lines: list[str],
    function_name: str,
) -> tuple[str | None, int | None]:
    for test in test_lines:
        try:
            tree = ast.parse(test, mode="exec")
        except SyntaxError:
            continue
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
            continue
        calls = _find_calls(tree.body[0].test, function_name=function_name)
        if len(calls) != 1:
            continue
        call = calls[0]
        arg_count = len(call.args)
        arg_names = ", ".join(f"arg{i + 1}" for i in range(arg_count))
        return f"{function_name}({arg_names})", arg_count
    return None, None


def _find_calls(expr: ast.AST, function_name: str) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == function_name:
            calls.append(node)
    return calls
