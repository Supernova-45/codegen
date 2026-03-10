# Strategy Comparison

## Overall

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| eig-tests | 242 | 0.558 | 0.000 | 108 | 2.062 | 11488.3 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 45 | 0.600 |
| contradictory | eig-tests | 74 | 0.392 |
| incomplete | eig-tests | 79 | 0.633 |
| original | eig-tests | 44 | 0.659 |

## MBPP+ By Condition

| condition | strategy | n | MBPP+ pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 9 | 0.000 |
| contradictory | eig-tests | 45 | 0.000 |
| incomplete | eig-tests | 45 | 0.000 |
| original | eig-tests | 9 | 0.000 |
