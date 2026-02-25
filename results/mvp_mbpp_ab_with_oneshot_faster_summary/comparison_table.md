# Strategy Comparison

## Overall

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| eig-tests | 30 | 0.767 | 0.000 | 21 | 1.867 | 9626.1 |
| one-shot | 30 | 0.633 | 0.000 | 21 | 0.000 | 209.1 |
| random-tests | 30 | 0.700 | 0.000 | 21 | 2.200 | 7789.4 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 10 | 0.800 |
| ambiguous | one-shot | 10 | 0.600 |
| ambiguous | random-tests | 10 | 0.600 |
| incomplete | eig-tests | 10 | 0.700 |
| incomplete | one-shot | 10 | 0.600 |
| incomplete | random-tests | 10 | 0.700 |
| original | eig-tests | 10 | 0.800 |
| original | one-shot | 10 | 0.700 |
| original | random-tests | 10 | 0.800 |

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
