# Scripts Index

## Core execution

- `run_experiment.py`: main runner for benchmark experiments.
- `summarize_results.py`: aggregates JSONL runs into CSV/Markdown summaries.
- `run_sanity_checks.py`: quick consistency checks for result files and metrics.

## Dataset preparation

- `prepare_mbpp_variants.py`: builds MBPP variant prompt file.
- `prepare_humaneval_variants.py`: builds HumanEval variant prompt file.

## Optimization and ablations

- `optimize_eig.py`: random search for EIG hyperparameters.
- `run_ablation_sweeps.py`: generates/executes ablation command matrix.
- `summarize_ablation_pair.py`: pairwise summary helper for ablations.

## Multi-profile runs

- `run_model_matrix.py`: executes model-profile matrix and merges comparisons.
- `report_humaneval_cost.py`: cost-focused reporting for HumanEval runs.

## Infrastructure helpers

- `setup_env.sh`: bootstraps local Python environment.
- `deploy_modal_qwen.sh`: deploys OpenAI-compatible Qwen server on Modal (`servers/modal_qwen_openai_server.py`).
- `run_google_humaneval_oneshot.sh`: one-shot HumanEval run via local Gemini wrapper (`servers/gemini_vertex_openai_server.py`).
- `run_full_6shards.sh`: convenience launcher for 6-way sharded run.
