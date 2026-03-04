# Strategy Comparison

## Overall


| strategy     | n   | pass@1 | MBPP+ pass@1 | MBPP+ n | avg_questions | avg_total_tokens |
| ------------ | --- | ------ | ------------ | ------- | ------------- | ---------------- |
| eig-tests    | 30  | 0.700  | n/a          | 0       | 1.333         | 5836.5           |
| ticode-tests | 30  | 0.733  | n/a          | 0       | 1.233         | 3988.8           |


## By Condition


| condition  | strategy     | n   | pass@1 |
| ---------- | ------------ | --- | ------ |
| ambiguous  | eig-tests    | 10  | 0.700  |
| ambiguous  | ticode-tests | 10  | 0.800  |
| incomplete | eig-tests    | 10  | 0.600  |
| incomplete | ticode-tests | 10  | 0.700  |
| original   | eig-tests    | 10  | 0.800  |
| original   | ticode-tests | 10  | 0.700  |


## MBPP+ By Condition


| condition | strategy | n   | MBPP+ pass@1 |
| --------- | -------- | --- | ------------ |


