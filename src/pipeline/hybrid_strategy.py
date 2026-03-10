"""Hybrid algorithm: EIG for test selection, hard pruning, pass-count final choice"""

from __future__ import annotations

from typing import Any

from config import PipelineConfig
from data.task_schema import BenchmarkTask
from execution.adapter import build_effective_code
from execution.sandbox import run_assertion, run_tests
from models.client_protocol import ModelClient
from models.openai_compatible import Usage
from posterior.particle_posterior import ParticlePosterior, TestOutcome
from query.eig_selector import select_max_eig

from pipeline.run_problem import (
    ProblemResult,
    _evaluate_test_matrix,
    _is_runtime_error,
    _ticode_rank_alive_by_passing_tests,
)

def run_hybrid_tests(
    task: BenchmarkTask,
    cfg: PipelineConfig,
    model: ModelClient,
    usage: Usage,
    interaction_trace: list[dict[str, Any]],
    signature_hint: str | None,
    expected_arity: int | None,
    test_arity: int | None,
    candidates: list[str],
    effective_candidates: list[str],
    candidate_adapters: list[dict[str, Any]],
) -> ProblemResult:
    keep_indices = list(range(len(candidates)))
    chosen_tests: list[dict[str, Any]] = []
    already_asked: list[dict[str, Any]] = []

    for round_idx in range(cfg.k_max):
        if len(keep_indices) <= 1:
            break

        weights = [0.0] * len(candidates)
        for i in keep_indices:
            weights[i] = 1.0 / len(keep_indices)
        posterior = ParticlePosterior(candidates=list(candidates), weights=weights)

        test_candidates, u_tests, test_filter_stats = model.generate_candidate_tests(
            task.prompt,
            function_name=task.function_name,
            signature_hint=signature_hint,
            expected_arity=test_arity,
            asked_tests=[x["test"] for x in already_asked],
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
            determinism_repeats=cfg.assertion_determinism_repeats,
        )

        round_log: dict[str, Any] = {
            "round": round_idx + 1,
            "generated_tests": test_candidates,
            "test_validation": test_eval_logs,
            "valid_tests": valid_tests,
            "strategy": "hybrid-tests",
            "asked_so_far": [x["test"] for x in already_asked],
            "signature_hint": signature_hint,
            "expected_arity": expected_arity,
            "enforce_test_signature_arity": cfg.enforce_test_signature_arity,
            "filter_non_discriminative": cfg.filter_non_discriminative,
            "candidate_adapter_info": candidate_adapters,
        }

        if not valid_tests:
            round_log["decision"] = "no_valid_tests_stop"
            round_log["test_generation_stats"] = [{"filter_stats": test_filter_stats}]
            interaction_trace.append(round_log)
            break

        # EIG test selection
        _, scores, score_details = select_max_eig(
            valid_tests,
            outcomes_by_test,
            posterior,
            cfg.epsilon,
            undefined_likelihood=cfg.undefined_outcome_likelihood,
            discriminative_weight=cfg.eig_discriminative_weight,
            runtime_error_penalty=cfg.eig_runtime_error_penalty,
        )
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_idx] <= 0.0:
            round_log["decision"] = "no_discriminative_tests_stop"
            round_log["eig_scores"] = scores
            interaction_trace.append(round_log)
            break

        selected_test = valid_tests[best_idx]
        selected_outcomes = outcomes_by_test[best_idx]
        selected_score = scores[best_idx]

        observed_bool, oracle_error = run_assertion(
            task.oracle_code,
            selected_test,
            cfg.sandbox_timeout_s,
        )
        oracle_runtime_error = _is_runtime_error(observed_bool, oracle_error)

        if oracle_runtime_error:
            observed: TestOutcome = None
        else:
            observed = observed_bool

        # hard prune
        new_keep: list[int] = []
        if observed is not None:
            for idx in keep_indices:
                out = selected_outcomes[idx]
                if out is True or out is False:
                    if out == observed:
                        new_keep.append(idx)
        else:
            new_keep = list(keep_indices)

        keep_indices = new_keep

        already_asked.append(
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
                "score": selected_score,
                "score_components": score_details[best_idx] if score_details else None,
                "posterior_updated": False,  # we use hard prune, not soft
            }
        )
        round_log["asked_in_round"] = [
            {
                "test": selected_test,
                "score": selected_score,
                "score_components": score_details[best_idx] if score_details else None,
                "oracle_observed": observed,
                "oracle_error": oracle_error,
            }
        ]
        round_log["selected_test"] = selected_test
        round_log["selected_test_score"] = selected_score
        round_log["decision"] = "ask_and_prune"
        round_log["alive_candidate_count_after"] = len(keep_indices)
        interaction_trace.append(round_log)

        if len(keep_indices) <= 1:
            break

    if keep_indices:
        ranked = _ticode_rank_alive_by_passing_tests(
            keep_indices=keep_indices,
            effective_candidates=effective_candidates,
            asked_tests=already_asked,
            timeout_s=cfg.sandbox_timeout_s,
        )
        final_index = ranked[0]
    else:
        final_index = 0

    final_code = candidates[final_index]
    effective_final_code, final_adapter_info = build_effective_code(
        final_code,
        expected_function_name=task.function_name,
        expected_arity=expected_arity,
        enabled=cfg.eval_with_adapter,
    )
    passed, total, errors = run_tests(
        effective_final_code,
        task.hidden_tests,
        cfg.sandbox_timeout_s,
    )

    return ProblemResult(
        task_id=task.task_id,
        condition=task.condition,
        strategy="hybrid-tests",
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
