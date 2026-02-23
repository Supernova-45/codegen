# ClarifyCode MVP

Clean-room MVP for interactive code generation with binary unit-test clarification.

This project implements the core loop described in `proposal.md`:
- sample candidate programs
- ask informative yes/no unit-test questions
- update a Bayesian particle posterior
- decide whether to ask again or submit code

## Scope

Current MVP supports:
- MBPP dataset
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

## Summarize Results

```bash
python scripts/summarize_results.py \
  --results results/latest.jsonl \
  --output-dir results
```

## Notes

- API keys are read from env vars only. Never hardcode secrets.
- Sandbox execution is best-effort for MVP; use isolated containers for stronger safety in production.
