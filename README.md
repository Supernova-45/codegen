# Twenty Questions for Code

Interactive code-generation experiments with clarification questions (yes/no tests).

The core loop is:
1. sample candidate programs,
2. ask informative test questions,
3. update a particle posterior over candidates,
4. stop and submit when expected value no longer justifies asking.

## Quick Start

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternative:

```bash
bash scripts/setup_env.sh --venv
# or
bash scripts/setup_env.sh --conda
```

### 2) Configure API access

```bash
cp .env.example .env
```

Set these core variables in `.env`:
- `CODEGEN_API_KEY`
- `CODEGEN_MODEL`
- `CODEGEN_BASE_URL` (optional; defaults to OpenAI if omitted)

Common optional routing variables:
- `CODEGEN_API_KEY_ENV_POOL`
- `CODEGEN_API_KEY_POOL`
- `CODEGEN_TESTGEN_API_KEY`
- `CODEGEN_TESTGEN_BASE_URL`
- `CODEGEN_TESTGEN_MODEL`

Provider-specific endpoint variables are grouped in `.env.example` under:
- `Modal Endpoint`
- `Cross-Model Profile Variables`
- `Local Gemini Vertex Proxy`

### 3) Build datasets (one-time per source refresh)

```bash
python scripts/prepare_mbpp_variants.py --output data/mbpp_variants.jsonl
python scripts/prepare_humaneval_variants.py --output data/humaneval_variants.jsonl
```

### 4) Run and summarize

```bash
python scripts/run_experiment.py --config configs/mvp_mbpp.yaml
python scripts/summarize_results.py --results results/latest.jsonl --output-dir results
```

## Standard Workflows

### Smoke run (fast sanity check)

```bash
python scripts/run_experiment.py --config configs/mvp_mbpp_ab_smoke.yaml --max-examples 8
```

### Full benchmark (MBPP or HumanEval)

```bash
python scripts/run_experiment.py --config configs/mvp_mbpp_rigorous.yaml
python scripts/run_experiment.py --config configs/mvp_humaneval.yaml
```

### Sharded run

```bash
python scripts/run_experiment.py \
  --config configs/mvp_mbpp_rigorous.yaml \
  --num-shards 6 \
  --shard-index 0
```

### EIG tuning

```bash
python scripts/optimize_eig.py \
  --config configs/mvp_mbpp_rigorous.yaml \
  --num-trials 24 \
  --max-examples 240 \
  --token-budget 8000
```

### Cross-model matrix

```bash
python scripts/run_model_matrix.py \
  --profiles modal_qwen_only openai_large google_large_openai_compat \
  --datasets mbpp humaneval \
  --num-shards 8 \
  --parallel-shards \
  --resume
```

## Repository Layout

```text
configs/   Experiment presets and profile mappings
scripts/   Entry points for runs, tuning, summaries, and helpers
src/       Runtime library (data loading, querying, execution, posterior, pipeline)
data/      Source datasets and generated variant files
results/   Run artifacts (ignored by git)
```

- Script index: `scripts/README.md`
- Config index: `configs/README.md`

## Configuration Notes

Common strategy labels used by `run_experiment.py`:
- `one-shot`
- `random-tests`
- `eig-tests`
- `self-consistency`
- `repair`
- `ticode-tests`

Useful `pipeline` knobs:
- `gamma`: ask-vs-submit threshold.
- `min_questions_if_valid`: force a minimum number of questions if valid tests exist.
- `min_valid_candidate_coverage`: minimum candidate coverage for generated tests.
- `min_eig_score`: minimum EIG before asking.
- `max_test_regen_attempts`: retry count for generating stronger tests.
- `query_scorer`: `eig` or `ticode`.
- `hard_prune_update`: hard elimination vs soft Bayesian reweighting.

## Outputs

Typical artifacts under `results/`:
- raw run JSONL,
- `summary_pass_at_1.csv`,
- `summary_questions.csv`,
- `summary_tokens.csv`,
- `summary_eig_diagnostics.csv`,
- `summary_bootstrap_ci.csv`,
- `summary_cost_efficiency.csv`,
- `summary_fixed_budget.csv`,
- `summary_pareto.csv`.

## Optional Integrations

- Modal Qwen endpoint: `scripts/deploy_modal_qwen.sh`, `servers/modal_qwen_openai_server.py`
- Vertex Gemini wrapper: `scripts/run_google_humaneval_oneshot.sh`, `servers/gemini_vertex_openai_server.py`

Modal quick wiring:
```bash
CODEGEN_BASE_URL=https://<modal-endpoint>.modal.run/v1
CODEGEN_API_KEY=dummy
CODEGEN_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Local Gemini wrapper quick wiring:
```bash
GCP_PROJECT_NAME=<your-gcp-project>
GEMINI_VERTEX_MODEL=gemini-3-pro-preview
bash scripts/run_google_humaneval_oneshot.sh 20
```

## References

- Grand et al., *Shoot First, Ask Questions Later?* (2025): https://arxiv.org/pdf/2510.20886
- Fakhoury et al., *LLM-Based Test-Driven Interactive Code Generation* (2024): https://arxiv.org/pdf/2404.10100
- Larbi et al., *When Prompts Go Wrong* (2025): https://arxiv.org/pdf/2507.20439
