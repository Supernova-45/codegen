from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

from dotenv import load_dotenv
import yaml


@dataclass
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    request_timeout_s: int


@dataclass
class PipelineConfig:
    strategy: str
    n_candidates: int
    tests_per_round: int
    k_max: int
    epsilon: float
    gamma: float
    min_questions_if_valid: int
    min_valid_candidate_coverage: float
    enforce_test_signature_arity: bool
    filter_non_discriminative: bool
    min_eig_score: float
    max_test_regen_attempts: int
    eig_questions_per_round: int
    candidate_temperature: float
    reprompt_temperature: float
    eval_with_adapter: bool
    run_reprompt: bool
    reprompt_min_questions: int
    reprompt_require_mixed_outcomes: bool
    reprompt_max_false_rate: float
    reprompt_max_runtime_error_rate: float
    reprompt_min_constraint_match_rate: float
    eig_discriminative_weight: float
    eig_runtime_error_penalty: float
    undefined_outcome_likelihood: float
    sandbox_timeout_s: int


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    max_examples: int
    shuffle: bool
    variants_path: str
    conditions: list[str]
    mbppplus_enabled: bool
    mbppplus_dataset: str
    mbppplus_split: str
    mbppplus_timeout_s: int
    model: ModelConfig
    pipeline: PipelineConfig
    results_dir: str
    output_file: str


def _must_env(var_name: str) -> str:
    value = os.environ.get(var_name, "").strip()
    if not value:
        raise ValueError(
            f"Missing required environment variable: {var_name}. "
            "Create a .env file (see .env.example) or export it in your shell."
        )
    return value


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_config(path: str) -> ExperimentConfig:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    model = raw["model"]
    pipeline = raw["pipeline"]
    output = raw["output"]
    dataset = raw["dataset"]

    mbppplus = dataset.get("mbppplus", {})
    mbppplus_enabled = bool(mbppplus.get("enabled", False))
    mbppplus_dataset = str(mbppplus.get("dataset", "evalplus/mbppplus"))
    mbppplus_split = str(mbppplus.get("split", "test"))
    mbppplus_timeout_s = int(mbppplus.get("timeout_s", max(10, int(pipeline["sandbox_timeout_s"]))))

    base_url_var = str(model["base_url_env"])
    api_key_var = str(model["api_key_env"])
    model_var = str(model["model_env"])

    model_cfg = ModelConfig(
        base_url=os.environ.get(base_url_var, "https://api.openai.com/v1"),
        api_key=_must_env(api_key_var),
        model=_must_env(model_var),
        temperature=float(model["temperature"]),
        request_timeout_s=int(model["request_timeout_s"]),
    )
    pipe_cfg = PipelineConfig(
        strategy=str(pipeline["strategy"]),
        n_candidates=int(pipeline["n_candidates"]),
        tests_per_round=int(pipeline["tests_per_round"]),
        k_max=int(pipeline["k_max"]),
        epsilon=float(pipeline["epsilon"]),
        gamma=float(pipeline["gamma"]),
        min_questions_if_valid=int(pipeline.get("min_questions_if_valid", 1)),
        min_valid_candidate_coverage=float(pipeline.get("min_valid_candidate_coverage", 0.6)),
        enforce_test_signature_arity=bool(pipeline.get("enforce_test_signature_arity", True)),
        filter_non_discriminative=bool(pipeline.get("filter_non_discriminative", True)),
        min_eig_score=float(pipeline.get("min_eig_score", 0.02)),
        max_test_regen_attempts=int(pipeline.get("max_test_regen_attempts", 1)),
        eig_questions_per_round=int(pipeline.get("eig_questions_per_round", 2)),
        candidate_temperature=float(pipeline.get("candidate_temperature", 0.8)),
        reprompt_temperature=float(pipeline.get("reprompt_temperature", 0.2)),
        eval_with_adapter=bool(pipeline.get("eval_with_adapter", True)),
        run_reprompt=bool(pipeline["run_reprompt"]),
        reprompt_min_questions=int(pipeline.get("reprompt_min_questions", 2)),
        reprompt_require_mixed_outcomes=bool(pipeline.get("reprompt_require_mixed_outcomes", True)),
        reprompt_max_false_rate=float(pipeline.get("reprompt_max_false_rate", 0.75)),
        reprompt_max_runtime_error_rate=float(pipeline.get("reprompt_max_runtime_error_rate", 0.25)),
        reprompt_min_constraint_match_rate=float(
            pipeline.get("reprompt_min_constraint_match_rate", 0.75)
        ),
        eig_discriminative_weight=_clamp(
            float(pipeline.get("eig_discriminative_weight", 0.35)),
            0.0,
            1.0,
        ),
        eig_runtime_error_penalty=_clamp(
            float(pipeline.get("eig_runtime_error_penalty", 0.5)),
            0.0,
            1.0,
        ),
        undefined_outcome_likelihood=_clamp(
            float(pipeline.get("undefined_outcome_likelihood", 0.85)),
            0.0,
            1.0,
        ),
        sandbox_timeout_s=int(pipeline["sandbox_timeout_s"]),
    )

    return ExperimentConfig(
        experiment_name=str(raw["experiment_name"]),
        seed=int(raw["seed"]),
        max_examples=int(raw["max_examples"]),
        shuffle=bool(raw["shuffle"]),
        variants_path=str(dataset["variants_path"]),
        conditions=list(dataset["conditions"]),
        mbppplus_enabled=mbppplus_enabled,
        mbppplus_dataset=mbppplus_dataset,
        mbppplus_split=mbppplus_split,
        mbppplus_timeout_s=mbppplus_timeout_s,
        model=model_cfg,
        pipeline=pipe_cfg,
        results_dir=str(output["results_dir"]),
        output_file=str(output["output_file"]),
    )


def ensure_output_path(cfg: ExperimentConfig) -> Path:
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / cfg.output_file
