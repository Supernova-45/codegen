from __future__ import annotations

import ast
from dataclasses import dataclass
import random
import re
import time
from typing import Any

import requests

from config import ModelConfig


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class OpenAICompatibleClient:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self._trace_events: list[dict[str, Any]] = []

    def clear_trace(self) -> None:
        self._trace_events = []

    def get_trace(self) -> list[dict[str, Any]]:
        return list(self._trace_events)

    def _chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> tuple[str, Usage]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        data: dict[str, Any] | None = None
        max_attempts = 6
        attempts: list[dict[str, Any]] = []
        for attempt in range(max_attempts):
            resp = requests.post(
                f"{self.cfg.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.cfg.request_timeout_s,
            )
            attempts.append({"attempt": attempt + 1, "status_code": resp.status_code})
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = min(60.0, float(retry_after))
                else:
                    sleep_s = min(30.0, float(2**attempt) + random.uniform(0.0, 0.5))
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        if data is None:
            raise RuntimeError("Model request failed after retries.")
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self._trace_events.append(
            {
                "kind": "chat_completion",
                "meta": meta or {},
                "request": {
                    "url": f"{self.cfg.base_url.rstrip('/')}/chat/completions",
                    "model": payload["model"],
                    "temperature": payload["temperature"],
                    "messages": messages,
                },
                "attempts": attempts,
                "response": {
                    "content": text,
                    "usage": usage,
                    "finish_reason": (
                        data.get("choices", [{}])[0].get("finish_reason")
                        if data.get("choices")
                        else None
                    ),
                },
            }
        )
        return text, Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    def generate_code_candidates(
        self,
        prompt: str,
        n: int,
        function_name: str | None = None,
        signature_hint: str | None = None,
        constraints: list[tuple[str, bool]] | None = None,
        temperature: float | None = None,
    ) -> tuple[list[str], Usage]:
        constraints = constraints or []
        constraint_lines = "\n".join(
            f"- {test} => {answer}" for test, answer in constraints
        )
        user_prompt = (
            "Write only Python code implementing the requested function.\n"
            "Do not include markdown fences.\n\n"
            f"Task:\n{prompt}\n"
        )
        if function_name:
            user_prompt += f"\nThe function name must be exactly: {function_name}\n"
        if signature_hint:
            user_prompt += (
                f"\nUse this call shape for the target function: {signature_hint}\n"
                "Keep parameter count and order consistent with that signature.\n"
            )
        if constraint_lines:
            user_prompt += f"\nObserved clarification tests:\n{constraint_lines}\n"
        user_prompt += "\nReturn only code."

        outputs: list[str] = []
        usage = Usage()
        for idx in range(n):
            txt, u = self._chat(
                [
                    {"role": "system", "content": "You are a careful Python coding assistant."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                meta={
                    "call_type": "generate_code_candidate",
                    "candidate_index": idx,
                    "total_candidates": n,
                    "function_name": function_name,
                },
            )
            outputs.append(extract_python(txt))
            usage.prompt_tokens += u.prompt_tokens
            usage.completion_tokens += u.completion_tokens
        return outputs, usage

    def generate_candidate_tests(
        self,
        prompt: str,
        function_name: str | None,
        signature_hint: str | None,
        expected_arity: int | None,
        asked_tests: list[str],
        n_tests: int,
    ) -> tuple[list[str], Usage, dict[str, int]]:
        already = "\n".join(f"- {x}" for x in asked_tests) if asked_tests else "- none"
        signature_text = signature_hint or f"{function_name or 'target_function'}(...)"
        txt, usage = self._chat(
            [
                {"role": "system", "content": "You design discriminative Python unit tests."},
                {
                    "role": "user",
                    "content": (
                        "Given the task, propose candidate binary clarification tests.\n"
                        "Return exactly one assert per line.\n"
                        "No prose, no numbering, no markdown.\n"
                        "Each assert must be standalone and executable by itself.\n"
                        "Use only Python literals in inputs (ints, floats, strings, bools, lists, tuples, dicts).\n"
                        "Do not use helper variables, random values, external files, or extra function calls.\n"
                        "Do not reference any names except the target function.\n"
                        "Use this exact function-call shape when writing asserts:\n"
                        f"{signature_text}\n"
                        "Prefer edge cases that split plausible implementations.\n"
                        f"Task:\n{prompt}\n\n"
                        f"Function under test: {function_name or 'unknown'}\n"
                        f"Already asked tests:\n{already}\n\n"
                        f"Need {n_tests} candidate asserts."
                    ),
                },
            ],
            meta={
                "call_type": "generate_candidate_tests",
                "requested_n_tests": n_tests,
                "function_name": function_name,
                "asked_tests_count": len(asked_tests),
            },
        )
        lines = [line.strip() for line in txt.splitlines()]
        asserts = [line for line in lines if line.startswith("assert ")]
        filtered, stats = _filter_assert_lines(
            asserts,
            function_name=function_name,
            expected_arity=expected_arity,
        )
        stats["raw_lines"] = len(lines)
        stats["assert_lines"] = len(asserts)
        stats["accepted"] = min(n_tests, len(filtered))
        return filtered[:n_tests], usage, stats


def extract_python(text: str) -> str:
    block = re.findall(r"```python(.*?)```", text, flags=re.DOTALL)
    if block:
        return block[0].strip()
    block2 = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    if block2:
        return block2[0].strip()
    return text.strip()


def _filter_assert_lines(
    lines: list[str],
    function_name: str | None,
    expected_arity: int | None,
) -> tuple[list[str], dict[str, int]]:
    out: list[str] = []
    seen: set[str] = set()
    stats: dict[str, int] = {
        "duplicates": 0,
        "invalid_assert_syntax": 0,
        "wrong_function_name": 0,
        "signature_mismatch": 0,
    }
    for line in lines:
        if line in seen:
            stats["duplicates"] += 1
            continue
        valid, reason = _is_valid_assert_line(
            line,
            function_name=function_name,
            expected_arity=expected_arity,
        )
        if not valid:
            stats[reason] = stats.get(reason, 0) + 1
            continue
        out.append(line)
        seen.add(line)
    return out, stats


def _is_valid_assert_line(
    line: str,
    function_name: str | None,
    expected_arity: int | None,
) -> tuple[bool, str]:
    try:
        tree = ast.parse(line, mode="exec")
    except SyntaxError:
        return False, "invalid_assert_syntax"
    if len(tree.body) != 1:
        return False, "invalid_assert_syntax"
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Assert):
        return False, "invalid_assert_syntax"

    if not function_name:
        return True, "accepted"

    calls = _find_target_calls(stmt.test, function_name=function_name)
    if len(calls) != 1:
        return False, "wrong_function_name"
    if expected_arity is not None and len(calls[0].args) != expected_arity:
        return False, "signature_mismatch"
    return True, "accepted"


def _find_target_calls(expr: ast.AST, function_name: str) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == function_name:
            calls.append(node)
    return calls
