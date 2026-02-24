# EIG Incremental Redesign Ablation Report

Runs use the same config and seed, with incremental toggles by stage.

| stage | strategy | n | pass@1 | avg_questions | avg_total_tokens |
|---|---|---:|---:|---:|---:|
| stage0 | eig-tests | 30 | 0.400 | 1.233 | 1666.6 |
| stage0 | random-tests | 30 | 0.233 | 2.000 | 1677.8 |
| stage1 | eig-tests | 30 | 0.333 | 1.200 | 1634.6 |
| stage2 | eig-tests | 30 | 0.233 | 0.600 | 1481.1 |
| stage3 | eig-tests | 30 | 0.233 | 0.867 | 1575.8 |
| final | eig-tests | 30 | 0.233 | 1.200 | 1519.9 |
| final | random-tests | 30 | 0.233 | 1.967 | 1523.6 |

## EIG Diagnostics by Stage

| stage | avg_selected_eig_score | selected_zero_score_frac | filtered_non_discriminative | signature_mismatch_rate | accepted_assert_rate |
|---|---:|---:|---:|---:|---:|
| stage0 | 0.500 | 0.324 | 0 | 0.000 | 0.991 |
| stage1 | 0.566 | 0.278 | 0 | 0.043 | 0.957 |
| stage2 | 0.764 | 0.000 | 102 | 0.055 | 0.923 |
| stage3 | 0.785 | 0.000 | 137 | 0.020 | 0.944 |
| final | 0.498 | 0.361 | 0 | 0.000 | 0.000 |

## Stage Definitions

- `stage0`: baseline-like (no signature arity filter, no non-discriminative filter, no EIG score gating, k=2, gamma=0.85, min_questions=1).
- `stage1`: + signature-aware arity filtering only.
- `stage2`: + non-discriminative filtering.
- `stage3`: + EIG score floor/regeneration (`min_eig_score=0.05`, `max_test_regen_attempts=2`).
- `final`: full current config (`k_max=3`, `gamma=0.8`, `min_questions_if_valid=2`) plus prior stages.
