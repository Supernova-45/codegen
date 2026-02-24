# Adapter A/B Validation (Provisional)

Runs were partially completed due API rate limits (`429`). Results below are directional, not final.

## strict_partial

- File: `results/strict_eval_ab_small.jsonl`
- Rows: `37`
- Strategy counts: `{'one-shot': 30, 'random-tests': 7}`


| strategy     | n   | pass@1 | type_error_rate | noresult_rate | process_crash_rate | adapter_applied_rate | adapter_success_rate |
| ------------ | --- | ------ | --------------- | ------------- | ------------------ | -------------------- | -------------------- |
| one-shot     | 30  | 0.267  | 0.200           | 0.000         | 0.000              | 0.000                | 0.000                |
| random-tests | 7   | 0.143  | 0.429           | 0.000         | 0.000              | 0.000                | 0.000                |


## adapter_eig_partial

- File: `results/adapter_eval_eig_only_small.jsonl`
- Rows: `3`
- Strategy counts: `{'eig-tests': 3}`


| strategy  | n   | pass@1 | type_error_rate | noresult_rate | process_crash_rate | adapter_applied_rate | adapter_success_rate |
| --------- | --- | ------ | --------------- | ------------- | ------------------ | -------------------- | -------------------- |
| eig-tests | 3   | 0.000  | 0.333           | 0.000         | 0.000              | 1.000                | 1.000                |


## Interpretation

- Sandbox hardening appears to eliminate `NoResult` in available runs (`noresult_rate=0.0`).
- Adapter path is active where executed (`adapter_applied_rate=1.0`, `adapter_success_rate=1.0` in adapter sample).
- Full strict-vs-adapter comparison across all 3 strategies remains incomplete and should be re-run once rate limits reset.

