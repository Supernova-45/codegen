# Strategy Comparison

## Overall

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| eig-tests | 3 | 0.333 | n/a | 0 | 1.333 | 3057.3 |
| random-tests | 3 | 0.000 | n/a | 0 | 1.333 | 2309.7 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 1 | 0.000 |
| ambiguous | random-tests | 1 | 0.000 |
| incomplete | eig-tests | 1 | 1.000 |
| incomplete | random-tests | 1 | 0.000 |
| original | eig-tests | 1 | 0.000 |
| original | random-tests | 1 | 0.000 |

## MBPP+ By Condition

| condition | strategy | n | MBPP+ pass@1 |
|---|---|---:|---:|
