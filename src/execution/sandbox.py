from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any
import warnings


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
    # Generated snippets occasionally include invalid escape sequences like "\d";
    # suppress syntax warnings to keep long-run logs readable.
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "float": float,
        "filter": filter,
        "hex": hex,
        "isinstance": isinstance,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "oct": oct,
        "ord": ord,
        "print": print,
        "pow": pow,
        "range": range,
        "round": round,
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
    except BaseException as exc:  # noqa: BLE001
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
    for _ in range(2):
        try:
            out = q.get(timeout=0.1)
            return bool(out["ok"]), str(out["error"])
        except queue.Empty:
            continue
    if proc.exitcode not in {0, None}:
        return False, f"ProcessCrashed(exitcode={proc.exitcode})"
    return False, "NoResult"


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
    for _ in range(2):
        try:
            out = q.get(timeout=0.1)
            return bool(out["ok"]), str(out["error"])
        except queue.Empty:
            continue
    if proc.exitcode not in {0, None}:
        return False, f"ProcessCrashed(exitcode={proc.exitcode})"
    return False, "NoResult"
