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
- OpenAI-compatible chat-completions backend

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

## Prepare MBPP Variants

```bash
python scripts/prepare_mbpp_variants.py \
  --ticode-mbpp /Users/alexandrakim/Desktop/codegen/TiCoder/datasets/mbpp/mbpp.jsonl \
  --robustness-dir /Users/alexandrakim/Desktop/codegen/Robustness-of-LLMs-to-prompt-imperfections/datasets/mbpp \
  --output data/mbpp_variants.jsonl
```

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

## Summarize Results

```bash
python scripts/summarize_results.py \
  --results results/latest.jsonl \
  --output-dir results
```

Additional MBPP+ summary output:
- `results/summary_mbppplus_pass_at_1.csv`
- `results/summary_eig_diagnostics.csv`
