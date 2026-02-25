# Strategy Comparison

## Overall

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| random-tests | 17 | 0.706 | 0.000 | 11 | 3.118 | 8721.9 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | random-tests | 5 | 0.600 |
| incomplete | random-tests | 6 | 0.833 |
| original | random-tests | 6 | 0.667 |

## MBPP+ By Condition

| condition | strategy | n | MBPP+ pass@1 |
|---|---|---:|---:|
| ambiguous | random-tests | 3 | 0.000 |
| incomplete | random-tests | 4 | 0.000 |
| original | random-tests | 4 | 0.000 |
