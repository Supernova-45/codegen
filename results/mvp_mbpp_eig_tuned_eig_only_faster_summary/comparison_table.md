# Strategy Comparison

## Overall

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| eig-tests | 30 | 0.767 | 0.000 | 21 | 1.867 | 9626.1 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 10 | 0.800 |
| incomplete | eig-tests | 10 | 0.700 |
| original | eig-tests | 10 | 0.800 |

## MBPP+ By Condition

| condition | strategy | n | MBPP+ pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 7 | 0.000 |
| incomplete | eig-tests | 7 | 0.000 |
| original | eig-tests | 7 | 0.000 |
