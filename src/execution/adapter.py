from __future__ import annotations

from dataclasses import dataclass
import ast
import difflib
from typing import Iterable


@dataclass
class AdapterInfo:
    adapter_applied: bool
    adapter_success: bool
    expected_function_name: str
    expected_arity: int | None
    selected_callable: str | None
    selected_arity: int | None
    adapter_failed_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_applied": self.adapter_applied,
            "adapter_success": self.adapter_success,
            "expected_function_name": self.expected_function_name,
            "expected_arity": self.expected_arity,
            "selected_callable": self.selected_callable,
            "selected_arity": self.selected_arity,
            "adapter_failed_reason": self.adapter_failed_reason,
        }


@dataclass
class _CallableDef:
    name: str
    arity: int


def build_effective_code(
    code: str,
    expected_function_name: str,
    expected_arity: int | None,
    enabled: bool,
) -> tuple[str, AdapterInfo]:
    if not enabled:
        return code, AdapterInfo(
            adapter_applied=False,
            adapter_success=False,
            expected_function_name=expected_function_name,
            expected_arity=expected_arity,
            selected_callable=None,
            selected_arity=None,
            adapter_failed_reason="adapter_disabled",
        )
    parsed, parse_error = _parse_module(code)
    if parsed is None:
        return code, AdapterInfo(
            adapter_applied=True,
            adapter_success=False,
            expected_function_name=expected_function_name,
            expected_arity=expected_arity,
            selected_callable=None,
            selected_arity=None,
            adapter_failed_reason=f"parse_error:{parse_error}",
        )
    callables = _collect_top_level_callables(parsed)
    if not callables:
        return code, AdapterInfo(
            adapter_applied=True,
            adapter_success=False,
            expected_function_name=expected_function_name,
            expected_arity=expected_arity,
            selected_callable=None,
            selected_arity=None,
            adapter_failed_reason="no_top_level_callables",
        )
    selected = _select_best_callable(
        callables,
        expected_function_name=expected_function_name,
        expected_arity=expected_arity,
    )
    if selected is None:
        return code, AdapterInfo(
            adapter_applied=True,
            adapter_success=False,
            expected_function_name=expected_function_name,
            expected_arity=expected_arity,
            selected_callable=None,
            selected_arity=None,
            adapter_failed_reason="ambiguous_or_unfit_candidate",
        )
    if selected.name == expected_function_name:
        return code, AdapterInfo(
            adapter_applied=True,
            adapter_success=True,
            expected_function_name=expected_function_name,
            expected_arity=expected_arity,
            selected_callable=selected.name,
            selected_arity=selected.arity,
            adapter_failed_reason="",
        )
    wrapper = _build_wrapper(
        expected_function_name=expected_function_name,
        expected_arity=expected_arity,
        target_name=selected.name,
    )
    if not wrapper:
        return code, AdapterInfo(
            adapter_applied=True,
            adapter_success=False,
            expected_function_name=expected_function_name,
            expected_arity=expected_arity,
            selected_callable=selected.name,
            selected_arity=selected.arity,
            adapter_failed_reason="arity_mismatch_no_varargs_adapter",
        )
    effective = f"{code.rstrip()}\n\n{wrapper}\n"
    return effective, AdapterInfo(
        adapter_applied=True,
        adapter_success=True,
        expected_function_name=expected_function_name,
        expected_arity=expected_arity,
        selected_callable=selected.name,
        selected_arity=selected.arity,
        adapter_failed_reason="",
    )


def _parse_module(code: str) -> tuple[ast.Module | None, str]:
    try:
        return ast.parse(code, mode="exec"), ""
    except SyntaxError as exc:
        return None, repr(exc)


def _collect_top_level_callables(tree: ast.Module) -> list[_CallableDef]:
    out: list[_CallableDef] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        if _has_unsupported_signature(node.args):
            continue
        out.append(_CallableDef(name=node.name, arity=len(node.args.args)))
    return out


def _has_unsupported_signature(args: ast.arguments) -> bool:
    if args.kwonlyargs:
        return True
    if args.kwarg is not None:
        return True
    return False


def _select_best_callable(
    callables: Iterable[_CallableDef],
    expected_function_name: str,
    expected_arity: int | None,
) -> _CallableDef | None:
    ranked: list[tuple[float, _CallableDef]] = []
    for fn in callables:
        score = _callable_score(
            fn,
            expected_function_name=expected_function_name,
            expected_arity=expected_arity,
        )
        ranked.append((score, fn))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    top_score, top_fn = ranked[0]
    if top_score < 0.45:
        return None
    if len(ranked) > 1 and abs(top_score - ranked[1][0]) < 0.03:
        return None
    return top_fn


def _callable_score(
    fn: _CallableDef,
    expected_function_name: str,
    expected_arity: int | None,
) -> float:
    name_sim = difflib.SequenceMatcher(
        None,
        fn.name.lower(),
        expected_function_name.lower(),
    ).ratio()
    if expected_arity is None:
        arity_score = 0.6
    else:
        diff = abs(fn.arity - expected_arity)
        arity_score = max(0.0, 1.0 - 0.5 * diff)
    exact_bonus = 0.3 if fn.name == expected_function_name else 0.0
    return 0.55 * name_sim + 0.45 * arity_score + exact_bonus


def _build_wrapper(
    expected_function_name: str,
    expected_arity: int | None,
    target_name: str,
) -> str:
    if expected_arity is None:
        return (
            _adapter_helper_source()
            + "\n"
            f"def {expected_function_name}(*args):\n"
            f"    return _adapter_call({target_name}, *args)\n"
        )
    if expected_arity == 0:
        return (
            _adapter_helper_source()
            + "\n"
            f"def {expected_function_name}():\n"
            f"    return _adapter_call({target_name})\n"
        )
    args = ", ".join(f"arg{i + 1}" for i in range(expected_arity))
    return (
        _adapter_helper_source()
        + "\n"
        f"def {expected_function_name}({args}):\n"
        f"    return _adapter_call({target_name}, {args})\n"
    )


def _adapter_helper_source() -> str:
    return (
        "def _adapter_call(fn, *args):\n"
        "    attempts = []\n"
        "    attempts.append(args)\n"
        "    # Try simple argument-order permutations first.\n"
        "    if len(args) == 2:\n"
        "        attempts.append((args[1], args[0]))\n"
        "    if len(args) == 3:\n"
        "        attempts.append((args[0], args[2], args[1]))\n"
        "        attempts.append((args[1], args[0], args[2]))\n"
        "        attempts.append((args[1], args[2], args[0]))\n"
        "        attempts.append((args[2], args[0], args[1]))\n"
        "        attempts.append((args[2], args[1], args[0]))\n"
        "    if len(args) > 1:\n"
        "        attempts.append(args[:1])\n"
        "        attempts.append(args[:-1])\n"
        "        attempts.append(tuple(args))\n"
        "        attempts.append(list(args))\n"
        "    if len(args) == 1 and isinstance(args[0], (list, tuple)):\n"
        "        seq = args[0]\n"
        "        if len(seq) >= 2:\n"
        "            attempts.append(tuple(seq))\n"
        "            attempts.append(tuple(seq[:2]))\n"
        "            attempts.append(tuple(seq[:3]))\n"
        "            attempts.append(tuple(seq[:4]))\n"
        "    last_exc = None\n"
        "    seen = set()\n"
        "    for cand in attempts:\n"
        "        key = repr(cand)\n"
        "        if key in seen:\n"
        "            continue\n"
        "        seen.add(key)\n"
        "        try:\n"
        "            if isinstance(cand, tuple):\n"
        "                return fn(*cand)\n"
        "            return fn(cand)\n"
        "        except TypeError as exc:\n"
        "            last_exc = exc\n"
        "    if last_exc is not None:\n"
        "        raise last_exc\n"
        "    return fn(*args)\n"
    )
