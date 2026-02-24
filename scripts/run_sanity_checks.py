#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PipelineConfig
from data.mbpp_loader import MBPPTask
from execution.sandbox import run_assertion
import pipeline.run_problem as run_problem_module


class FakeModel:
    def __init__(self) -> None:
        self.test_generation_calls = 0
        self._trace: list[dict[str, object]] = []

    def clear_trace(self) -> None:
        self._trace = []

    def get_trace(self) -> list[dict[str, object]]:
        return list(self._trace)

    def generate_code_candidates(
        self,
        prompt: str,
        n: int,
        function_name: str | None = None,
        signature_hint: str | None = None,
        visible_tests: list[str] | None = None,
        constraints: list[tuple[str, bool]] | None = None,
        temperature: float | None = None,
    ) -> tuple[list[str], object]:
        del prompt, function_name, signature_hint, visible_tests, constraints, temperature
        code = "def foo(x):\n    return x\n"
        return [code for _ in range(n)], _Usage()

    def generate_candidate_tests(
        self,
        prompt: str,
        function_name: str | None,
        signature_hint: str | None,
        expected_arity: int | None,
        asked_tests: list[str],
        visible_tests: list[str] | None,
        n_tests: int,
    ) -> tuple[list[str], object, dict[str, int]]:
        del prompt, function_name, signature_hint, expected_arity, asked_tests, visible_tests, n_tests
        self.test_generation_calls += 1
        return ["assert foo(1) == 1"], _Usage(), {"accepted": 1, "assert_lines": 1, "raw_lines": 1}


class FakeRepromptModel:
    def __init__(self) -> None:
        self._trace: list[dict[str, object]] = []

    def clear_trace(self) -> None:
        self._trace = []

    def get_trace(self) -> list[dict[str, object]]:
        return list(self._trace)

    def generate_code_candidates(
        self,
        prompt: str,
        n: int,
        function_name: str | None = None,
        signature_hint: str | None = None,
        visible_tests: list[str] | None = None,
        constraints: list[tuple[str, bool]] | None = None,
        temperature: float | None = None,
    ) -> tuple[list[str], object]:
        del prompt, function_name, signature_hint, visible_tests, temperature
        good = "def foo(x):\n    return 1\n"
        bad = "def foo(x):\n    return 0\n"
        if constraints:
            return [bad], _Usage()
        if n == 2:
            return [good, bad], _Usage()
        return [good for _ in range(n)], _Usage()

    def generate_candidate_tests(
        self,
        prompt: str,
        function_name: str | None,
        signature_hint: str | None,
        expected_arity: int | None,
        asked_tests: list[str],
        visible_tests: list[str] | None,
        n_tests: int,
    ) -> tuple[list[str], object, dict[str, int]]:
        del prompt, function_name, signature_hint, expected_arity, asked_tests, visible_tests, n_tests
        return ["assert foo(1) == 1"], _Usage(), {"accepted": 1, "assert_lines": 1, "raw_lines": 1}


class _Usage:
    prompt_tokens = 0
    completion_tokens = 0


def _base_cfg() -> PipelineConfig:
    return PipelineConfig(
        strategy="eig-tests",
        n_candidates=2,
        tests_per_round=1,
        k_max=1,
        epsilon=0.02,
        gamma=0.85,
        min_questions_if_valid=1,
        min_valid_candidate_coverage=0.6,
        enforce_test_signature_arity=True,
        filter_non_discriminative=True,
        min_eig_score=0.0,
        max_test_regen_attempts=2,
        eig_questions_per_round=1,
        candidate_temperature=0.7,
        reprompt_temperature=0.1,
        eval_with_adapter=True,
        run_reprompt=True,
        reprompt_min_questions=2,
        reprompt_require_mixed_outcomes=True,
        reprompt_max_false_rate=0.75,
        reprompt_max_runtime_error_rate=0.25,
        reprompt_min_constraint_match_rate=0.75,
        sandbox_timeout_s=3,
    )


def _task() -> MBPPTask:
    return MBPPTask(
        task_id=999001,
        condition="original",
        prompt="Write foo.",
        oracle_code="def foo(x):\n    return 1\n",
        visible_tests=["assert foo(1) == 1"],
        hidden_tests=["assert foo(1) == 1"],
        function_name="foo",
    )


def check_sandbox_builtins() -> None:
    code = (
        "def uses_isinstance(x):\n"
        "    return isinstance(x, list)\n\n"
        "def uses_map_filter(xs):\n"
        "    return list(map(lambda z: z + 1, filter(lambda y: y % 2 == 0, xs)))\n"
    )
    ok1, err1 = run_assertion(code, "assert uses_isinstance([1, 2]) is True", timeout_s=3)
    ok2, err2 = run_assertion(code, "assert uses_map_filter([1, 2, 3, 4]) == [3, 5]", timeout_s=3)
    if not ok1 or not ok2:
        raise AssertionError(f"sandbox builtins check failed: {err1} / {err2}")


def check_eig_regen_retries() -> None:
    model = FakeModel()
    cfg = _base_cfg()
    cfg.run_reprompt = False
    task = _task()

    old_eval = run_problem_module._evaluate_test_matrix
    try:
        run_problem_module._evaluate_test_matrix = lambda *args, **kwargs: ([], [], [])
        result = run_problem_module.run_problem(
            task=task,
            strategy="eig-tests",
            model=model,
            cfg=cfg,
            seed=42,
        )
    finally:
        run_problem_module._evaluate_test_matrix = old_eval

    if model.test_generation_calls != cfg.max_test_regen_attempts + 1:
        raise AssertionError(
            "EIG regen did not retry expected number of times: "
            f"got {model.test_generation_calls}, want {cfg.max_test_regen_attempts + 1}"
        )
    round_steps = [x for x in result.interaction_trace if "round" in x]
    if not round_steps:
        raise AssertionError("expected at least one round trace entry")
    first_round = round_steps[0]
    if first_round.get("decision") != "no_valid_tests_stop":
        raise AssertionError(f"unexpected decision for no-valid-tests path: {first_round.get('decision')}")
    if first_round.get("regen_attempts_used") != cfg.max_test_regen_attempts:
        raise AssertionError("regen_attempts_used did not reach configured max on empty-valid path")


def check_reprompt_fallback_on_constraint_mismatch() -> None:
    model = FakeRepromptModel()
    cfg = _base_cfg()
    cfg.reprompt_min_questions = 1
    cfg.reprompt_require_mixed_outcomes = False
    cfg.reprompt_max_false_rate = 1.0
    cfg.reprompt_max_runtime_error_rate = 1.0
    cfg.reprompt_min_constraint_match_rate = 1.0
    task = _task()

    old_eval = run_problem_module._evaluate_test_matrix
    try:
        run_problem_module._evaluate_test_matrix = (
            lambda *args, **kwargs: (
                ["assert foo(1) == 1"],
                [[True, False]],
                [
                    {
                        "test": "assert foo(1) == 1",
                        "valid": True,
                        "invalid_reason": "",
                        "outcomes_if_valid": [True, False],
                        "coverage": 1.0,
                        "runtime_error_runs": 0,
                        "candidate_checks": [],
                    }
                ],
            )
        )
        result = run_problem_module.run_problem(
            task=task,
            strategy="eig-tests",
            model=model,
            cfg=cfg,
            seed=42,
        )
    finally:
        run_problem_module._evaluate_test_matrix = old_eval

    if "return 1" not in result.final_code:
        raise AssertionError("expected fallback to MAP candidate when reprompt constraints mismatch")
    reprompt_steps = [x for x in result.interaction_trace if x.get("step") == "reprompt_decision"]
    if not reprompt_steps:
        raise AssertionError("missing reprompt_decision trace entry")
    reprompt = reprompt_steps[-1]
    if reprompt.get("decision") != "fallback_to_map_candidate":
        raise AssertionError(f"unexpected reprompt decision: {reprompt.get('decision')}")


def main() -> None:
    checks = [
        ("sandbox_builtins", check_sandbox_builtins),
        ("eig_regen_retries", check_eig_regen_retries),
        ("reprompt_fallback", check_reprompt_fallback_on_constraint_mismatch),
    ]
    for name, check in checks:
        check()
        print(f"PASS {name}")
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
