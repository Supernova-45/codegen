# Strategy Comparison

## Overall

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| eig-tests | 30 | 0.300 | 0.000 | 21 | 2.400 | 4929.7 |
| random-tests | 30 | 0.300 | 0.000 | 21 | 2.567 | 4479.1 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 10 | 0.200 |
| ambiguous | random-tests | 10 | 0.300 |
| incomplete | eig-tests | 10 | 0.300 |
| incomplete | random-tests | 10 | 0.100 |
| original | eig-tests | 10 | 0.400 |
| original | random-tests | 10 | 0.500 |

## MBPP+ By Condition

| condition | strategy | n | MBPP+ pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 7 | 0.000 |
| ambiguous | random-tests | 7 | 0.000 |
| incomplete | eig-tests | 7 | 0.000 |
| incomplete | random-tests | 7 | 0.000 |
| original | eig-tests | 7 | 0.000 |
| original | random-tests | 7 | 0.000 |
