from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkTask:
    benchmark: str
    task_id: int
    task_key: str
    condition: str
    prompt: str
    oracle_code: str
    visible_tests: list[str]
    hidden_tests: list[str]
    function_name: str
