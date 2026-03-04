# Twenty Questions for Code

Interactive code generation with binary unit-test clarification.

This project implements the loop described in `proposal.md`:
- sample candidate programs
- ask informative yes/no unit-test questions
- update a Bayesian particle posterior
- decide whether to ask again or submit code

## Scope

Currently supports:
- MBPP and HumanEval variant datasets
- Optional MBPP+ strict re-scoring (`evalplus/mbppplus`)
- prompt conditions: `original`, `incomplete`, `ambiguous`, `contradictory` (if available in source variants)
- strategies: `one-shot`, `random-tests`, `eig-tests`, `self-consistency`, `repair`, `ticode-tests`

## Environment Setup

Use one of the following before anything else.

### Option A: `venv`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: `conda`

```bash
conda create -n clarifycode python=3.11 -y
conda activate clarifycode
pip install -r requirements.txt
```

You can also run a helper script:

```bash
bash scripts/setup_env.sh --venv
# or
bash scripts/setup_env.sh --conda
```

## Configure (.env)

Create a local `.env` file (never commit it):

```bash
cp .env.example .env
```

Then edit `.env` with your key and model.

Required environment variables:
- `CLARIFYCODE_API_KEY`
- `CLARIFYCODE_MODEL`

Optional:
- `CLARIFYCODE_BASE_URL` (defaults to OpenAI public endpoint)
- `CLARIFYCODE_API_KEY_ENV_POOL` (comma-separated env var names, e.g. `GROK_KEY_1,...,GROK_KEY_6`)
- `CLARIFYCODE_API_KEY_POOL` (comma-separated raw keys)

If `--api-key-env-pool` is not provided, `scripts/run_experiment.py` auto-loads keys from `.env` in this order:
1. `CLARIFYCODE_API_KEY_ENV_POOL`
2. `CLARIFYCODE_API_KEY_POOL`

## Prepare MBPP Variants (Example)

```bash
python scripts/prepare_mbpp_variants.py \
  --output data/mbpp_variants.jsonl
```

Default source paths are now vendored in this repo:
- `data/sources/ticode_mbpp/mbpp.jsonl`
- `data/sources/robustness_mbpp/{MBPP.json,incomplete_MBPP.json,ambiguous_MBPP.json,contradictory_MBPP.json}`

You can still override with `--ticode-mbpp` and `--robustness-dir` if needed.

## Prepare HumanEval Variants

```bash
python scripts/prepare_humaneval_variants.py \
  --output data/humaneval_variants.jsonl
```

Source files are expected under:
- `data/sources/robustness_humaneval/HumanEval.json`
- `data/sources/robustness_humaneval/incomplete_humaneval.json`
- `data/sources/robustness_humaneval/ambiguous_humaneval.json`
- `data/sources/robustness_humaneval/contradictory_humaneval.json` (optional)

`scripts/run_experiment.py` automatically loads `.env`.

## Run Experiments

```bash
python scripts/run_experiment.py --config configs/mvp_mbpp.yaml
```

This keeps optimization on current hidden tests and, when enabled in config, re-scores the
same `final_code` on MBPP+ for stricter reporting.

HumanEval run:

```bash
python scripts/run_experiment.py --config configs/mvp_humaneval.yaml
```

Sharded run using key pool from `.env` (no CLI key flags needed):

```bash
python scripts/run_experiment.py \
  --config configs/mvp_mbpp_rigorous.yaml \
  --num-shards 6 \
  --shard-index 0
```

EIG tuning knobs in `pipeline`:
- `gamma`: ask-vs-submit threshold (lower asks more often)
- `min_questions_if_valid`: force at least this many asks when valid tests exist
- `min_valid_candidate_coverage`: fraction of candidate programs a generated test must execute on
- `min_eig_score`: minimum EIG score required before asking a clarification test
- `max_test_regen_attempts`: retries for generating stronger clarification tests if scores are too low
- `eval_with_adapter`: evaluate with compatibility wrapper so correct logic can pass despite naming/signature variance
- `eig_questions_per_round`: number of top-ranked EIG questions asked each round (no diversity constraint)
- `candidate_temperature`: sampling temperature for candidate program generation
- `reprompt_temperature`: sampling temperature for post-clarification regeneration
- `query_scorer`: `eig` or `ticode` (ablation for selection rule)
- `hard_prune_update`: hard candidate elimination instead of soft posterior reweighting
- `repair_rounds`: number of generate->test->repair rounds in repair baseline
- `self_consistency_min_coverage`: minimum defined-outcome coverage for self-consistency test scoring

## Summarize Results

```bash
python scripts/summarize_results.py \
  --results results/latest.jsonl \
  --output-dir results
```

Additional MBPP+ summary output:
- `results/summary_mbppplus_pass_at_1.csv`
- `results/summary_eig_diagnostics.csv`
- `results/summary_bootstrap_ci.csv`
- `results/summary_cost_efficiency.csv`
- `results/summary_fixed_budget.csv`
- `results/summary_pareto.csv`

## Ablation Sweeps

Generate the sweep command matrix (scorer, update mode, VOI stop, and N/tests/k sensitivity):

```bash
python scripts/run_ablation_sweeps.py --config configs/mvp_mbpp_rigorous.yaml
```

Execute the sweep:

```bash
python scripts/run_ablation_sweeps.py \
  --config configs/mvp_mbpp_rigorous.yaml \
  --execute
```

## Reporting Checklist

When preparing final tables/figures, include:
- `pass@1` overall and by condition (`original`, `incomplete`, `ambiguous`, `contradictory`)
- bootstrap uncertainty from `summary_bootstrap_ci.csv`
- cost-normalized metrics from `summary_cost_efficiency.csv`
- fixed-budget comparisons from `summary_fixed_budget.csv`
- Pareto frontier points from `summary_pareto.csv`
- test quality diagnostics from `summary_eig_diagnostics.csv`

## References

- **Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People**  
  Gabriel Grand, Valerio Pepe, Jacob Andreas, Joshua B. Tenenbaum (2025-10-23)  
  https://arxiv.org/pdf/2510.20886
- **LLM-Based Test-Driven Interactive Code Generation: User Study and Empirical Evaluation**  
  Sarah Fakhoury, Aaditya Naik, Georgios Sakkas, Saikat Chakraborty, Shuvendu K. Lahiri (2024-04-15)  
  https://arxiv.org/pdf/2404.10100
- **When Prompts Go Wrong: Evaluating Code Model Robustness to Ambiguous, Contradictory, and Incomplete Task Descriptions**  
  Maya Larbi, Amal Akli, Mike Papadakis, Rihab Bouyousfi, Maxime Cordy, Federica Sarro, Yves Le Traon (2025-07-27)  
  https://arxiv.org/pdf/2507.20439
