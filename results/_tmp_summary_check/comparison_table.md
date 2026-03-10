# Strategy Comparison

## Overall

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| one-shot | 3896 | 0.454 | 0.000 | 1512 | 0.000 | 203.0 |
| random-tests | 664 | 0.535 | 0.000 | 375 | 3.108 | 9160.3 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | one-shot | 974 | 0.473 |
| ambiguous | random-tests | 175 | 0.611 |
| contradictory | one-shot | 974 | 0.296 |
| contradictory | random-tests | 160 | 0.375 |
| incomplete | one-shot | 974 | 0.473 |
| incomplete | random-tests | 174 | 0.552 |
| original | one-shot | 974 | 0.574 |
| original | random-tests | 155 | 0.594 |

## MBPP+ By Condition

| condition | strategy | n | MBPP+ pass@1 |
|---|---|---:|---:|
| ambiguous | one-shot | 378 | 0.000 |
| ambiguous | random-tests | 97 | 0.000 |
| contradictory | one-shot | 378 | 0.000 |
| contradictory | random-tests | 91 | 0.000 |
| incomplete | one-shot | 378 | 0.000 |
| incomplete | random-tests | 97 | 0.000 |
| original | one-shot | 378 | 0.000 |
| original | random-tests | 90 | 0.000 |
