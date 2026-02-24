from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from config import PipelineConfig
from data.mbpp_loader import MBPPTask, infer_signature_hint
from execution.adapter import build_effective_code
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
    adapter_info: dict[str, Any]
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
            "adapter_info": self.adapter_info,
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
    signature_hint, expected_arity = infer_signature_hint(task.visible_tests, task.function_name)
    test_arity = expected_arity if cfg.enforce_test_signature_arity else None

    if strategy not in {"one-shot", "random-tests", "eig-tests"}:
        raise ValueError(f"Unsupported strategy: {strategy}")

    if strategy == "one-shot":
        code_list, u = model.generate_code_candidates(
            task.prompt,
            n=1,
            function_name=task.function_name,
            signature_hint=signature_hint,
            visible_tests=task.visible_tests,
            temperature=0.2,
        )
        usage.prompt_tokens += u.prompt_tokens
        usage.completion_tokens += u.completion_tokens
        final_code = code_list[0]
        effective_code, adapter_info = build_effective_code(
            final_code,
            expected_function_name=task.function_name,
            expected_arity=expected_arity,
            enabled=cfg.eval_with_adapter,
        )
        passed, total, errors = run_tests(effective_code, task.hidden_tests, cfg.sandbox_timeout_s)
        interaction_trace.append(
            {
                "step": "one_shot_generation",
                "prompt": task.prompt,
                "function_name": task.function_name,
                "candidate_count": 1,
                "selected_code": final_code,
                "adapter_info": adapter_info.to_dict(),
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
            adapter_info=adapter_info.to_dict(),
            mbppplus_pass_at_1=None,
            mbppplus_error=None,
            interaction_trace=interaction_trace,
            model_trace=model.get_trace(),
        )

    candidates, u_codes = model.generate_code_candidates(
        task.prompt,
        n=cfg.n_candidates,
        function_name=task.function_name,
        signature_hint=signature_hint,
        visible_tests=task.visible_tests,
        temperature=cfg.candidate_temperature,
    )
    usage.prompt_tokens += u_codes.prompt_tokens
    usage.completion_tokens += u_codes.completion_tokens
    candidate_adapters: list[dict[str, Any]] = []
    effective_candidates: list[str] = []
    for code in candidates:
        effective, adapter_info = build_effective_code(
            code,
            expected_function_name=task.function_name,
            expected_arity=expected_arity,
            enabled=cfg.eval_with_adapter,
        )
        effective_candidates.append(effective)
        candidate_adapters.append(adapter_info.to_dict())
    posterior = ParticlePosterior.uniform(candidates)
    asked: list[tuple[str, bool]] = []
    asked_details: list[dict[str, Any]] = []
    chosen_tests: list[dict[str, Any]] = []

    eig_score_floor = cfg.min_eig_score if strategy == "eig-tests" else None
    for round_idx in range(cfg.k_max):
        test_candidates: list[str] = []
        valid_tests: list[str] = []
        outcomes_by_test: list[list[bool]] = []
        test_eval_logs: list[dict[str, Any]] = []
        generation_stats: list[dict[str, Any]] = []
        regen_count = 0
        while True:
            test_candidates, u_tests, test_filter_stats = model.generate_candidate_tests(
                task.prompt,
                function_name=task.function_name,
                signature_hint=signature_hint,
                expected_arity=test_arity,
                asked_tests=[x[0] for x in asked],
                visible_tests=task.visible_tests,
                n_tests=cfg.tests_per_round,
            )
            usage.prompt_tokens += u_tests.prompt_tokens
            usage.completion_tokens += u_tests.completion_tokens
            valid_tests, outcomes_by_test, test_eval_logs = _evaluate_test_matrix(
                test_candidates,
                effective_candidates,
                cfg.sandbox_timeout_s,
                min_coverage=cfg.min_valid_candidate_coverage,
                filter_non_discriminative=cfg.filter_non_discriminative,
            )
            generation_stats.append(
                {
                    "regen_attempt": regen_count,
                    "filter_stats": test_filter_stats,
                    "generated_tests": test_candidates,
                    "valid_count": len(valid_tests),
                }
            )
            if strategy != "eig-tests" or eig_score_floor is None:
                break
            if not valid_tests:
                if regen_count >= cfg.max_test_regen_attempts:
                    break
                regen_count += 1
                continue
            _, regen_scores = select_max_eig(valid_tests, outcomes_by_test, posterior, cfg.epsilon)
            best_score = max(regen_scores) if regen_scores else 0.0
            high_value_count = sum(1 for s in regen_scores if s >= eig_score_floor)
            if (
                (best_score >= eig_score_floor and high_value_count >= cfg.eig_questions_per_round)
                or regen_count >= cfg.max_test_regen_attempts
            ):
                break
            regen_count += 1
        round_log: dict[str, Any] = {
            "round": round_idx + 1,
            "generated_tests": test_candidates,
            "test_validation": test_eval_logs,
            "valid_tests": valid_tests,
            "strategy": strategy,
            "asked_so_far": [x[0] for x in asked],
            "signature_hint": signature_hint,
            "expected_arity": expected_arity,
            "enforce_test_signature_arity": cfg.enforce_test_signature_arity,
            "filter_non_discriminative": cfg.filter_non_discriminative,
            "test_generation_stats": generation_stats,
            "regen_attempts_used": regen_count,
            "candidate_adapter_info": candidate_adapters,
        }
        if not valid_tests:
            round_log["decision"] = "no_valid_tests_stop"
            interaction_trace.append(round_log)
            break

        if strategy == "random-tests":
            scores = []
            selected_indices = [rng.randrange(len(valid_tests))]
        else:
            _, scores = select_max_eig(valid_tests, outcomes_by_test, posterior, cfg.epsilon)
            if eig_score_floor is not None:
                valid_by_score = [i for i, s in enumerate(scores) if s >= eig_score_floor]
                valid_tests = [valid_tests[i] for i in valid_by_score]
                outcomes_by_test = [outcomes_by_test[i] for i in valid_by_score]
                scores = [scores[i] for i in valid_by_score]
            if not valid_tests:
                round_log["decision"] = "no_high_value_tests_stop"
                round_log["eig_score_floor"] = eig_score_floor
                interaction_trace.append(round_log)
                break
            order = sorted(range(len(valid_tests)), key=lambda i: scores[i], reverse=True)
            selected_indices = order[: max(1, cfg.eig_questions_per_round)]

        round_log["asked_in_round"] = []
        stop_after_round = False
        for idx in selected_indices:
            selected_test = valid_tests[idx]
            outcomes = outcomes_by_test[idx]
            selected_score = scores[idx] if scores else None

            p_current = posterior.map_confidence()
            p_next = posterior.expected_map_after_question(outcomes, cfg.epsilon)
            if len(chosen_tests) >= cfg.k_max:
                stop_after_round = True
                break

            observed, oracle_error = run_assertion(task.oracle_code, selected_test, cfg.sandbox_timeout_s)
            oracle_runtime_error = _is_runtime_error(observed, oracle_error)
            posterior.update(outcomes, observed, cfg.epsilon)
            asked.append((selected_test, observed))
            asked_details.append(
                {
                    "test": selected_test,
                    "observed": observed,
                    "oracle_error": oracle_error,
                    "oracle_runtime_error": oracle_runtime_error,
                }
            )
            chosen_tests.append(
                {
                    "test": selected_test,
                    "observed": observed,
                    "oracle_error": oracle_error,
                    "oracle_runtime_error": oracle_runtime_error,
                    "map_before": p_current,
                    "map_expected_after": p_next,
                    "score": selected_score,
                }
            )
            round_log["asked_in_round"].append(
                {
                    "test": selected_test,
                    "score": selected_score,
                    "map_before": p_current,
                    "map_expected_after": p_next,
                    "oracle_observed": observed,
                    "oracle_error": oracle_error,
                }
            )

        if round_log["asked_in_round"]:
            first = round_log["asked_in_round"][0]
            round_log["selected_test"] = first["test"]
            round_log["selected_test_score"] = first["score"]
            round_log["map_before"] = first["map_before"]
            round_log["map_expected_after"] = first["map_expected_after"]
            round_log["oracle_observed"] = first["oracle_observed"]
            if "decision" not in round_log:
                round_log["decision"] = "ask_and_update"
        elif strategy == "eig-tests":
            round_log["decision"] = "submit_without_asking"
            stop_after_round = True

        interaction_trace.append(round_log)
        if stop_after_round:
            break

    map_candidate_index = posterior.map_index()
    final_code = candidates[map_candidate_index]
    reprompt_log: dict[str, Any] = {
        "step": "reprompt_decision",
        "run_reprompt_enabled": cfg.run_reprompt,
        "asked_count": len(asked),
        "map_candidate_index": map_candidate_index,
        "used_map_candidate_fallback": True,
    }
    if cfg.run_reprompt and asked:
        observed_values = [entry["observed"] for entry in asked_details]
        false_rate = sum(1 for v in observed_values if not v) / len(observed_values)
        runtime_error_rate = (
            sum(1 for x in asked_details if x["oracle_runtime_error"]) / len(asked_details)
        )
        reason_codes: list[str] = []
        if len(asked_details) < cfg.reprompt_min_questions:
            reason_codes.append("below_min_questions")
        if cfg.reprompt_require_mixed_outcomes and not (
            any(observed_values) and any(not x for x in observed_values)
        ):
            reason_codes.append("missing_mixed_outcomes")
        if false_rate > cfg.reprompt_max_false_rate:
            reason_codes.append("false_rate_above_threshold")
        if runtime_error_rate > cfg.reprompt_max_runtime_error_rate:
            reason_codes.append("runtime_error_rate_above_threshold")

        reprompt_log.update(
            {
                "false_rate": false_rate,
                "runtime_error_rate": runtime_error_rate,
                "reason_codes": reason_codes,
            }
        )
        if not reason_codes:
            constraints = [(t, a) for t, a in asked]
            new_code, u = model.generate_code_candidates(
                task.prompt,
                n=1,
                function_name=task.function_name,
                signature_hint=signature_hint,
                visible_tests=task.visible_tests,
                constraints=constraints,
                temperature=cfg.reprompt_temperature,
            )
            usage.prompt_tokens += u.prompt_tokens
            usage.completion_tokens += u.completion_tokens
            reprompt_candidate = new_code[0]
            effective_candidate_code, candidate_adapter_info = build_effective_code(
                reprompt_candidate,
                expected_function_name=task.function_name,
                expected_arity=expected_arity,
                enabled=cfg.eval_with_adapter,
            )

            validation_checks: list[dict[str, Any]] = []
            matched = 0
            for entry in asked_details:
                got_ok, got_error = run_assertion(
                    effective_candidate_code,
                    entry["test"],
                    cfg.sandbox_timeout_s,
                )
                is_match = got_ok == entry["observed"]
                if is_match:
                    matched += 1
                validation_checks.append(
                    {
                        "test": entry["test"],
                        "expected_observed": entry["observed"],
                        "candidate_observed": got_ok,
                        "candidate_error": got_error,
                        "matches": is_match,
                    }
                )
            match_rate = matched / max(1, len(asked_details))
            reprompt_log.update(
                {
                    "reprompt_attempted": True,
                    "reprompt_candidate": reprompt_candidate,
                    "reprompt_candidate_adapter_info": candidate_adapter_info.to_dict(),
                    "constraint_validation": validation_checks,
                    "constraint_match_rate": match_rate,
                    "constraint_match_threshold": cfg.reprompt_min_constraint_match_rate,
                }
            )
            if match_rate >= cfg.reprompt_min_constraint_match_rate:
                final_code = reprompt_candidate
                reprompt_log["decision"] = "use_reprompt_candidate"
                reprompt_log["used_map_candidate_fallback"] = False
            else:
                reprompt_log["decision"] = "fallback_to_map_candidate"
                reprompt_log["reason_codes"] = ["constraint_match_below_threshold"]
        else:
            reprompt_log["decision"] = "skip_reprompt"
    elif cfg.run_reprompt:
        reprompt_log["decision"] = "skip_reprompt"
        reprompt_log["reason_codes"] = ["no_clarification_questions"]
    else:
        reprompt_log["decision"] = "skip_reprompt"
        reprompt_log["reason_codes"] = ["reprompt_disabled"]
    interaction_trace.append(reprompt_log)

    effective_final_code, final_adapter_info = build_effective_code(
        final_code,
        expected_function_name=task.function_name,
        expected_arity=expected_arity,
        enabled=cfg.eval_with_adapter,
    )
    passed, total, errors = run_tests(effective_final_code, task.hidden_tests, cfg.sandbox_timeout_s)
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
        adapter_info=final_adapter_info.to_dict(),
        mbppplus_pass_at_1=None,
        mbppplus_error=None,
        interaction_trace=interaction_trace,
        model_trace=model.get_trace(),
    )


def _is_runtime_error(ok: bool, error: str) -> bool:
    if ok:
        return False
    return "AssertionError" not in error


def _evaluate_test_matrix(
    tests: list[str],
    candidates: list[str],
    timeout_s: int,
    min_coverage: float,
    filter_non_discriminative: bool,
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
        runtime_error_runs = 0
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
                runtime_error_runs += 1
                deterministic_runs += 1
                outcomes.append(False)
                continue
            deterministic_runs += 1
            outcomes.append(ok1)
        coverage = deterministic_runs / max(1, len(candidates))
        non_discriminative = bool(outcomes) and (all(outcomes) or not any(outcomes))
        universal_runtime_error = runtime_error_runs == len(candidates) and len(candidates) > 0
        if (
            valid
            and outcomes
            and coverage >= min_coverage
            and not universal_runtime_error
            and (not filter_non_discriminative or not non_discriminative)
        ):
            valid_tests.append(test)
            matrix.append(outcomes)
        elif valid and outcomes:
            valid = False
            if universal_runtime_error:
                invalid_reason = "universal_runtime_error"
            elif non_discriminative:
                invalid_reason = "non_discriminative"
            else:
                invalid_reason = "low_candidate_coverage"
        logs.append(
            {
                "test": test,
                "valid": valid and bool(outcomes),
                "invalid_reason": invalid_reason,
                "outcomes_if_valid": outcomes if valid else [],
                "coverage": coverage,
                "runtime_error_runs": runtime_error_runs,
                "candidate_checks": candidate_checks,
            }
        )
    return valid_tests, matrix, logs
