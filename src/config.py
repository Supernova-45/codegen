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
    run_reprompt: bool
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
        run_reprompt=bool(pipeline["run_reprompt"]),
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
