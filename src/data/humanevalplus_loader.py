from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import requests

from data.humaneval_loader import extract_hidden_assertions


_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


@dataclass
class HumanEvalPlusTask:
    task_id: int
    test_script: str


def _normalize_task_id(raw: Any) -> int | None:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.isdigit():
            return int(s)
        if "/" in s:
            _, suffix = s.rsplit("/", 1)
            if suffix.isdigit():
                return int(suffix)
    return None


def _build_oracle_code(prompt: str, canonical_solution: str) -> str:
    sol = canonical_solution
    if sol and not sol.startswith((" ", "\t")):
        sol = "    " + sol
    return f"{prompt.rstrip()}\n{sol}\n"


def _build_script_from_inputs(row: dict[str, Any], entry_point: str) -> str | None:
    prompt = str(row.get("prompt", "")).strip()
    canonical_solution = str(row.get("canonical_solution", "")).strip()
    if not prompt or not canonical_solution:
        return None
    raw_cases: list[Any] = []
    for key in ("base_input", "plus_input"):
        val = row.get(key)
        if isinstance(val, list):
            raw_cases.extend(val)
    if not raw_cases:
        return None
    unique_cases: list[Any] = []
    seen: set[str] = set()
    for case in raw_cases:
        try:
            key = json.dumps(case, sort_keys=True)
        except TypeError:
            key = repr(case)
        if key in seen:
            continue
        seen.add(key)
        unique_cases.append(case)
    if not unique_cases:
        return None

    atol_raw = row.get("atol", 0.0)
    try:
        atol = float(atol_raw)
    except (TypeError, ValueError):
        atol = 0.0
    oracle_code = _build_oracle_code(prompt=prompt, canonical_solution=canonical_solution)
    return (
        "import copy\n"
        "import math\n\n"
        f"_ORACLE_SOURCE = {oracle_code!r}\n"
        "_ORACLE_NS: dict[str, object] = {}\n"
        "exec(_ORACLE_SOURCE, _ORACLE_NS)\n"
        f"_ORACLE_FN = _ORACLE_NS[{entry_point!r}]\n"
        f"_CASES = {unique_cases!r}\n"
        f"_ATOL = {atol!r}\n\n"
        "def _eq(a, b, atol):\n"
        "    if isinstance(a, float) or isinstance(b, float):\n"
        "        try:\n"
        "            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)\n"
        "        except Exception:\n"
        "            return False\n"
        "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):\n"
        "        return len(a) == len(b) and all(_eq(x, y, atol) for x, y in zip(a, b))\n"
        "    if isinstance(a, dict) and isinstance(b, dict):\n"
        "        if set(a.keys()) != set(b.keys()):\n"
        "            return False\n"
        "        return all(_eq(a[k], b[k], atol) for k in a)\n"
        "    return a == b\n\n"
        "def _to_args(raw):\n"
        "    if isinstance(raw, tuple):\n"
        "        return list(raw)\n"
        "    if isinstance(raw, list):\n"
        "        return list(raw)\n"
        "    return [raw]\n\n"
        "for _case in _CASES:\n"
        "    _args_candidate = _to_args(copy.deepcopy(_case))\n"
        "    _args_oracle = _to_args(copy.deepcopy(_case))\n"
        f"    _cand_val = {entry_point}(*_args_candidate)\n"
        "    _oracle_val = _ORACLE_FN(*_args_oracle)\n"
        "    assert _eq(_cand_val, _oracle_val, _ATOL)\n"
    )


def _extract_test_script(row: dict[str, Any]) -> str | None:
    entry_point_raw = row.get("entry_point")
    entry_point = str(entry_point_raw).strip() if entry_point_raw is not None else ""

    test_raw = row.get("test")
    if isinstance(test_raw, str) and test_raw.strip():
        if entry_point:
            assertions = extract_hidden_assertions(test_raw, entry_point=entry_point)
            if assertions:
                return "\n".join(assertions)
            return test_raw.replace("candidate(", f"{entry_point}(")
        return test_raw

    if entry_point:
        return _build_script_from_inputs(row, entry_point=entry_point)
    return None


def load_humanevalplus_tests(
    dataset: str = "evalplus/humanevalplus",
    split: str = "test",
    page_size: int = 100,
    timeout_s: int = 30,
) -> dict[int, HumanEvalPlusTask]:
    tasks: dict[int, HumanEvalPlusTask] = {}
    offset = 0

    while True:
        params = {
            "dataset": dataset,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": page_size,
        }
        response = requests.get(_ROWS_ENDPOINT, params=params, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows", [])
        if not rows:
            break

        for wrapped_row in rows:
            row = wrapped_row.get("row", {})
            task_id = _normalize_task_id(row.get("task_id"))
            if task_id is None:
                continue
            test_script = _extract_test_script(row)
            if not test_script or not test_script.strip():
                continue
            tasks[task_id] = HumanEvalPlusTask(task_id=task_id, test_script=test_script)

        offset += len(rows)
        if len(rows) < page_size:
            break

    return tasks
