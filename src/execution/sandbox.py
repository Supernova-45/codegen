from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any


ALLOWED_IMPORTS = {
    "math",
    "re",
    "heapq",
    "collections",
    "itertools",
    "functools",
    "string",
    "bisect",
    "numpy",
}


def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"Import blocked in sandbox: {name}")
    return __import__(name, globals_, locals_, fromlist, level)


def _run_in_child(code: str, test: str, q: mp.Queue) -> None:
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__import__": _safe_import,
    }
    env: dict[str, Any] = {"__builtins__": safe_builtins}
    try:
        exec(code, env, env)
        exec(test, env, env)
        q.put({"ok": True, "error": ""})
    except Exception as exc:  # noqa: BLE001
        q.put({"ok": False, "error": repr(exc)})


def run_assertion(code: str, assertion: str, timeout_s: int) -> tuple[bool, str]:
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_run_in_child, args=(code, assertion, q))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, "Timeout"
    try:
        out = q.get_nowait()
    except queue.Empty:
        return False, "NoResult"
    return bool(out["ok"]), str(out["error"])


def run_tests(code: str, tests: list[str], timeout_s: int) -> tuple[int, int, list[str]]:
    passed = 0
    errors: list[str] = []
    for t in tests:
        ok, err = run_assertion(code, t, timeout_s)
        if ok:
            passed += 1
        else:
            errors.append(err)
    return passed, len(tests), errors


def run_test_script(code: str, script: str, timeout_s: int) -> tuple[bool, str]:
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_run_in_child, args=(code, script, q))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, "Timeout"
    try:
        out = q.get_nowait()
    except queue.Empty:
        return False, "NoResult"
    return bool(out["ok"]), str(out["error"])
