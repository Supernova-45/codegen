# Twenty Questions for Code

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use:

```bash
bash scripts/setup_env.sh --venv
# or
bash scripts/setup_env.sh --conda
```

## Environment Variables

Create your local env file:

```bash
cp .env.example .env
```

Required in `.env`:
- `CODEGEN_API_KEY`
- `CODEGEN_MODEL`

Optional:
- `CODEGEN_BASE_URL` 
- `CODEGEN_TESTGEN_API_KEY`
- `CODEGEN_TESTGEN_BASE_URL`
- `CODEGEN_TESTGEN_MODEL`

## Prepare Datasets

```bash
python scripts/prepare_mbpp_variants.py --output data/mbpp_variants.jsonl
python scripts/prepare_humaneval_variants.py --output data/humaneval_variants.jsonl
```

## Run Experiments

MBPP:

```bash
python scripts/run_experiment.py --config configs/mvp_mbpp.yaml
```

HumanEval:

```bash
python scripts/run_experiment.py --config configs/mvp_humaneval.yaml
```

HumanEval with HumanEval+ re-scoring:

```bash
python scripts/run_experiment.py --config configs/mvp_humaneval.yaml --enable-humanevalplus
```

Google Vertex one-shot wrapper:

```bash
bash scripts/run_google_humaneval_oneshot.sh 20
```

Google wrapper with HumanEval+ re-scoring:

```bash
ENABLE_HUMANEVALPLUS=1 bash scripts/run_google_humaneval_oneshot.sh 20
```

## Summarize Results

```bash
python scripts/summarize_results.py \
  --results results/latest.jsonl \
  --output-dir results
```

