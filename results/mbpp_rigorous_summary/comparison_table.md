# Strategy Comparison

## Overall

| strategy | n | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| eig-tests | 20 | 0.600 | 0.000 | 10 | 1.650 | 9913.4 |
| one-shot | 20 | 0.550 | 0.000 | 10 | 0.000 | 208.4 |
| random-tests | 20 | 0.550 | 0.000 | 10 | 2.450 | 8235.2 |
| repair | 20 | 0.550 | 0.000 | 10 | 0.000 | 2141.4 |
| self-consistency | 20 | 0.500 | 0.000 | 10 | 0.000 | 6605.2 |

## By Condition

| condition | strategy | n | pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 10 | 0.700 |
| ambiguous | one-shot | 10 | 0.600 |
| ambiguous | random-tests | 10 | 0.600 |
| ambiguous | repair | 10 | 0.700 |
| ambiguous | self-consistency | 10 | 0.500 |
| original | eig-tests | 10 | 0.500 |
| original | one-shot | 10 | 0.500 |
| original | random-tests | 10 | 0.500 |
| original | repair | 10 | 0.400 |
| original | self-consistency | 10 | 0.500 |

## MBPP+ By Condition

| condition | strategy | n | MBPP+ pass@1 |
|---|---|---:|---:|
| ambiguous | eig-tests | 6 | 0.000 |
| ambiguous | one-shot | 6 | 0.000 |
| ambiguous | random-tests | 6 | 0.000 |
| ambiguous | repair | 6 | 0.000 |
| ambiguous | self-consistency | 6 | 0.000 |
| original | eig-tests | 4 | 0.000 |
| original | one-shot | 4 | 0.000 |
| original | random-tests | 4 | 0.000 |
| original | repair | 4 | 0.000 |
| original | self-consistency | 4 | 0.000 |
