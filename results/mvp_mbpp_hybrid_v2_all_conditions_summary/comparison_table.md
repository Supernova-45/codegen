# Strategy Comparison

## Overall

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

## MBPP+ By Condition

| condition | strategy | n | MBPP+ pass@1 |
|---|---|---:|---:|
