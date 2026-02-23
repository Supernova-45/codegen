from __future__ import annotations

from dataclasses import dataclass
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
