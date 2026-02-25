# Twenty Questions for Code

Interactive code generation with binary unit-test clarification.

This project implements the loop described in `proposal.md`:
- sample candidate programs
- ask informative yes/no unit-test questions
- update a Bayesian particle posterior
- decide whether to ask again or submit code

## Scope

Currently supports:
- MBPP dataset
- Optional MBPP+ strict re-scoring (`evalplus/mbppplus`)
- prompt conditions: `original`, `incomplete`, `ambiguous`
- strategies: `one-shot`, `random-tests`, `eig-tests`

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

## Prepare MBPP Variants (Example)

```bash
python scripts/prepare_mbpp_variants.py \
  --output data/mbpp_variants.jsonl
```

Default source paths are now vendored in this repo:
- `data/sources/ticode_mbpp/mbpp.jsonl`
- `data/sources/robustness_mbpp/{MBPP.json,incomplete_MBPP.json,ambiguous_MBPP.json}`

You can still override with `--ticode-mbpp` and `--robustness-dir` if needed.

`scripts/run_experiment.py` automatically loads `.env`.

## Run Experiments

```bash
python scripts/run_experiment.py --config configs/mvp_mbpp.yaml
```

This keeps optimization on current hidden tests and, when enabled in config, re-scores the
same `final_code` on MBPP+ for stricter reporting.

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

## Summarize Results

```bash
python scripts/summarize_results.py \
  --results results/latest.jsonl \
  --output-dir results
```

Additional MBPP+ summary output:
- `results/summary_mbppplus_pass_at_1.csv`
- `results/summary_eig_diagnostics.csv`

## Preliminary Results

Using the following config:
- `gamma=0.95`
- `min_questions_if_valid=2`
- `filter_non_discriminative=true`
- `min_valid_candidate_coverage=0.5`
- `disable_voi_stop=false`
- `force_full_question_budget=false`
- `k_max=4`
- `run_reprompt=true`
- `shared_test_pool=true`
- `shared_test_pool_size=24`
- `shared_test_pool_regen_rounds=1`
- `skip_posterior_update_on_undefined_oracle=true` 

Here are the results:

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| eig-tests | 30 | 0.733 | n/a | 0 | 2.367 | 7961.0 |
| one-shot | 30 | 0.567 | n/a | 0 | 0.000 | 209.8 |
| random-tests | 30 | 0.700 | n/a | 0 | 3.333 | 8101.0 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 10 | 0.700 |
| ambiguous | one-shot | 10 | 0.500 |
| ambiguous | random-tests | 10 | 0.700 |
| incomplete | eig-tests | 10 | 0.700 |
| incomplete | one-shot | 10 | 0.500 |
| incomplete | random-tests | 10 | 0.700 |
| original | eig-tests | 10 | 0.800 |
| original | one-shot | 10 | 0.700 |
| original | random-tests | 10 | 0.700 |

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
