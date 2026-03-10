# Strategy Comparison

## Overall

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| one-shot | 664 | 0.441 | 0.000 | 375 | 0.000 | 199.2 |
| random-tests | 664 | 0.535 | 0.000 | 375 | 3.108 | 9160.3 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | one-shot | 175 | 0.503 |
| ambiguous | random-tests | 175 | 0.611 |
| contradictory | one-shot | 160 | 0.263 |
| contradictory | random-tests | 160 | 0.375 |
| incomplete | one-shot | 174 | 0.460 |
| incomplete | random-tests | 174 | 0.552 |
| original | one-shot | 155 | 0.535 |
| original | random-tests | 155 | 0.594 |

## MBPP+ By Condition

| condition | strategy | n | MBPP+ pass@1 |
|---|---|---:|---:|
| ambiguous | one-shot | 97 | 0.000 |
| ambiguous | random-tests | 97 | 0.000 |
| contradictory | one-shot | 91 | 0.000 |
| contradictory | random-tests | 91 | 0.000 |
| incomplete | one-shot | 97 | 0.000 |
| incomplete | random-tests | 97 | 0.000 |
| original | one-shot | 90 | 0.000 |
| original | random-tests | 90 | 0.000 |
