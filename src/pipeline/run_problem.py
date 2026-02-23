from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from config import PipelineConfig
from data.mbpp_loader import MBPPTask
from execution.sandbox import run_assertion, run_tests
from models.openai_compatible import OpenAICompatibleClient, Usage
from posterior.particle_posterior import ParticlePosterior
from query.ask_or_submit import should_ask
from query.eig_selector import select_max_eig


@dataclass
class ProblemResult:
    task_id: int
    condition: str
    strategy: str
    pass_at_1: bool
    questions_asked: int
    chosen_tests: list[dict[str, Any]]
    total_prompt_tokens: int
    total_completion_tokens: int
    final_code: str
    eval_passed: int
    eval_total: int
    eval_errors: list[str]
    mbppplus_pass_at_1: bool | None
    mbppplus_error: str | None
    interaction_trace: list[dict[str, Any]]
    model_trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "condition": self.condition,
            "strategy": self.strategy,
            "pass_at_1": self.pass_at_1,
            "questions_asked": self.questions_asked,
            "chosen_tests": self.chosen_tests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "final_code": self.final_code,
            "eval_passed": self.eval_passed,
            "eval_total": self.eval_total,
            "eval_errors": self.eval_errors,
            "mbppplus_pass_at_1": self.mbppplus_pass_at_1,
            "mbppplus_error": self.mbppplus_error,
            "interaction_trace": self.interaction_trace,
            "model_trace": self.model_trace,
        }


def run_problem(
    task: MBPPTask,
    strategy: str,
    model: OpenAICompatibleClient,
    cfg: PipelineConfig,
    seed: int,
) -> ProblemResult:
    rng = random.Random(seed + task.task_id)
    usage = Usage()
    model.clear_trace()
    interaction_trace: list[dict[str, Any]] = []

    if strategy not in {"one-shot", "random-tests", "eig-tests"}:
        raise ValueError(f"Unsupported strategy: {strategy}")

    if strategy == "one-shot":
        code_list, u = model.generate_code_candidates(
            task.prompt, n=1, function_name=task.function_name, temperature=0.2
        )
        usage.prompt_tokens += u.prompt_tokens
        usage.completion_tokens += u.completion_tokens
        final_code = code_list[0]
        passed, total, errors = run_tests(final_code, task.hidden_tests, cfg.sandbox_timeout_s)
        interaction_trace.append(
            {
                "step": "one_shot_generation",
                "prompt": task.prompt,
                "function_name": task.function_name,
                "candidate_count": 1,
                "selected_code": final_code,
            }
        )
        return ProblemResult(
            task_id=task.task_id,
            condition=task.condition,
            strategy=strategy,
            pass_at_1=passed == total,
            questions_asked=0,
            chosen_tests=[],
            total_prompt_tokens=usage.prompt_tokens,
            total_completion_tokens=usage.completion_tokens,
            final_code=final_code,
            eval_passed=passed,
            eval_total=total,
            eval_errors=errors,
            mbppplus_pass_at_1=None,
            mbppplus_error=None,
            interaction_trace=interaction_trace,
            model_trace=model.get_trace(),
        )

    candidates, u_codes = model.generate_code_candidates(
        task.prompt,
        n=cfg.n_candidates,
        function_name=task.function_name,
        temperature=0.8,
    )
    usage.prompt_tokens += u_codes.prompt_tokens
    usage.completion_tokens += u_codes.completion_tokens
    posterior = ParticlePosterior.uniform(candidates)
    asked: list[tuple[str, bool]] = []
    chosen_tests: list[dict[str, Any]] = []

    for round_idx in range(cfg.k_max):
        test_candidates, u_tests = model.generate_candidate_tests(
            task.prompt,
            function_name=task.function_name,
            asked_tests=[x[0] for x in asked],
            n_tests=cfg.tests_per_round,
        )
        usage.prompt_tokens += u_tests.prompt_tokens
        usage.completion_tokens += u_tests.completion_tokens
        valid_tests, outcomes_by_test, test_eval_logs = _evaluate_test_matrix(
            test_candidates,
            candidates,
            cfg.sandbox_timeout_s,
            min_coverage=cfg.min_valid_candidate_coverage,
        )
        round_log: dict[str, Any] = {
            "round": round_idx + 1,
            "generated_tests": test_candidates,
            "test_validation": test_eval_logs,
            "valid_tests": valid_tests,
            "strategy": strategy,
            "asked_so_far": [x[0] for x in asked],
        }
        if not valid_tests:
            round_log["decision"] = "no_valid_tests_stop"
            interaction_trace.append(round_log)
            break

        if strategy == "random-tests":
            idx = rng.randrange(len(valid_tests))
            scores = []
        else:
            idx, scores = select_max_eig(valid_tests, outcomes_by_test, posterior, cfg.epsilon)

        selected_test = valid_tests[idx]
        outcomes = outcomes_by_test[idx]

        must_ask_for_minimum = len(chosen_tests) < cfg.min_questions_if_valid
        if strategy == "eig-tests" and not must_ask_for_minimum:
            p_current = posterior.map_confidence()
            p_next = posterior.expected_map_after_question(outcomes, cfg.epsilon)
            if not should_ask(p_current, p_next, cfg.gamma):
                round_log["selected_test"] = selected_test
                round_log["selected_test_score"] = scores[idx] if scores else None
                round_log["map_before"] = p_current
                round_log["map_expected_after"] = p_next
                round_log["decision"] = "submit_without_asking"
                interaction_trace.append(round_log)
                break
        else:
            p_current = posterior.map_confidence()
            p_next = p_current

        observed, _ = run_assertion(task.oracle_code, selected_test, cfg.sandbox_timeout_s)
        posterior.update(outcomes, observed, cfg.epsilon)
        asked.append((selected_test, observed))
        chosen_tests.append(
            {
                "test": selected_test,
                "observed": observed,
                "map_before": p_current,
                "map_expected_after": p_next,
                "score": scores[idx] if scores else None,
            }
        )
        round_log["selected_test"] = selected_test
        round_log["selected_test_score"] = scores[idx] if scores else None
        round_log["map_before"] = p_current
        round_log["map_expected_after"] = p_next
        round_log["oracle_observed"] = observed
        round_log["decision"] = "ask_and_update"
        interaction_trace.append(round_log)

    if cfg.run_reprompt and asked:
        constraints = [(t, a) for t, a in asked]
        new_code, u = model.generate_code_candidates(
            task.prompt,
            n=1,
            function_name=task.function_name,
            constraints=constraints,
            temperature=0.2,
        )
        usage.prompt_tokens += u.prompt_tokens
        usage.completion_tokens += u.completion_tokens
        final_code = new_code[0]
    else:
        final_code = candidates[posterior.map_index()]

    passed, total, errors = run_tests(final_code, task.hidden_tests, cfg.sandbox_timeout_s)
    return ProblemResult(
        task_id=task.task_id,
        condition=task.condition,
        strategy=strategy,
        pass_at_1=passed == total,
        questions_asked=len(chosen_tests),
        chosen_tests=chosen_tests,
        total_prompt_tokens=usage.prompt_tokens,
        total_completion_tokens=usage.completion_tokens,
        final_code=final_code,
        eval_passed=passed,
        eval_total=total,
        eval_errors=errors,
        mbppplus_pass_at_1=None,
        mbppplus_error=None,
        interaction_trace=interaction_trace,
        model_trace=model.get_trace(),
    )


def _evaluate_test_matrix(
    tests: list[str],
    candidates: list[str],
    timeout_s: int,
    min_coverage: float,
) -> tuple[list[str], list[list[bool]], list[dict[str, Any]]]:
    valid_tests: list[str] = []
    matrix: list[list[bool]] = []
    logs: list[dict[str, Any]] = []
    for test in tests:
        outcomes: list[bool] = []
        valid = True
        invalid_reason = ""
        candidate_checks: list[dict[str, Any]] = []
        deterministic_runs = 0
        for code in candidates:
            ok1, err1 = run_assertion(code, test, timeout_s)
            ok2, err2 = run_assertion(code, test, timeout_s)
            candidate_checks.append(
                {
                    "first_run_ok": ok1,
                    "second_run_ok": ok2,
                    "first_error": err1,
                    "second_error": err2,
                }
            )
            if err1 == "Timeout" or err2 == "Timeout":
                valid = False
                invalid_reason = "timeout"
                break
            if ok1 != ok2:
                valid = False
                invalid_reason = "non_deterministic"
                break
            if not ok1 and "AssertionError" not in err1:
                # Stable runtime failures still provide discrimination signal:
                # candidates that crash on this behavior are effectively "False".
                deterministic_runs += 1
                outcomes.append(False)
                continue
            deterministic_runs += 1
            outcomes.append(ok1)
        coverage = deterministic_runs / max(1, len(candidates))
        if valid and outcomes and coverage >= min_coverage:
            valid_tests.append(test)
            matrix.append(outcomes)
        elif valid and outcomes:
            valid = False
            invalid_reason = "low_candidate_coverage"
        logs.append(
            {
                "test": test,
                "valid": valid and bool(outcomes),
                "invalid_reason": invalid_reason,
                "outcomes_if_valid": outcomes if valid else [],
                "coverage": coverage,
                "candidate_checks": candidate_checks,
            }
        )
    return valid_tests, matrix, logs
