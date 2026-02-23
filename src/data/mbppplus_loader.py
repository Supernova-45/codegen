from __future__ import annotations

from dataclasses import dataclass

import requests


_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


@dataclass
class MBPPPlusTask:
    task_id: int
    test_script: str


def load_mbppplus_tests(
    dataset: str = "evalplus/mbppplus",
    split: str = "test",
    page_size: int = 100,
    timeout_s: int = 30,
) -> dict[int, MBPPPlusTask]:
    tasks: dict[int, MBPPPlusTask] = {}
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
            task_id_raw = row.get("task_id")
            test_script = row.get("test")
            if task_id_raw is None or not isinstance(test_script, str) or not test_script.strip():
                continue
            task_id = int(task_id_raw)
            tasks[task_id] = MBPPPlusTask(task_id=task_id, test_script=test_script)

        offset += len(rows)
        if len(rows) < page_size:
            break

    return tasks
