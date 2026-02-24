# Strategy Comparison

## Overall

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| eig-tests | 30 | 0.300 | 0.000 | 21 | 0.933 | 1818.3 |
| one-shot | 30 | 0.233 | 0.000 | 21 | 0.000 | 168.5 |
| random-tests | 30 | 0.200 | 0.000 | 21 | 1.300 | 1761.5 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 10 | 0.200 |
| ambiguous | one-shot | 10 | 0.200 |
| ambiguous | random-tests | 10 | 0.200 |
| incomplete | eig-tests | 10 | 0.400 |
| incomplete | one-shot | 10 | 0.300 |
| incomplete | random-tests | 10 | 0.100 |
| original | eig-tests | 10 | 0.300 |
| original | one-shot | 10 | 0.200 |
| original | random-tests | 10 | 0.300 |

## MBPP+ By Condition

| condition | strategy | n | MBPP+ pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 7 | 0.000 |
| ambiguous | one-shot | 7 | 0.000 |
| ambiguous | random-tests | 7 | 0.000 |
| incomplete | eig-tests | 7 | 0.000 |
| incomplete | one-shot | 7 | 0.000 |
| incomplete | random-tests | 7 | 0.000 |
| original | eig-tests | 7 | 0.000 |
| original | one-shot | 7 | 0.000 |
| original | random-tests | 7 | 0.000 |
