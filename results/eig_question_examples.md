# EIG Clarification Examples

Examples from `results/latest.jsonl` comparing `eig-tests` vs `one-shot`.

## EIG Correct, One-shot Incorrect

### Task 3 (ambiguous)

- Task prompts (by condition):
  - Original task prompt:

```text
Write a python function to identify non-prime numbers.
```
  - Mutated incomplete prompt:

```text
Write a function to identify numbers in a list.
```
  - Mutated ambiguous prompt:

```text
Write a function to pick out non prime values from a set.
```
  - Mutated contradictory prompt: `_not available in dataset_`
- One-shot pass@1: `False` (eval: 0/2)
- EIG pass@1: `True` (eval: 2/2)
- EIG questions asked: `1`
- EIG clarification tests and oracle responses:
  - Q1: `assert is_not_prime(4) == True`
    - Oracle response: `True`; score: `0.7834`

### Task 9 (original)

- Task prompts (by condition):
  - Original task prompt:

```text
Write a python function to find the minimum number of rotations required to get the same string.
```
  - Mutated incomplete prompt:

```text
Write a function to find how many times a string can be rotated.
```
  - Mutated ambiguous prompt:

```text
Write a function that finds the number  of rotations changes on text until something gets the same .
```
  - Mutated contradictory prompt: `_not available in dataset_`
- One-shot pass@1: `False` (eval: 0/2)
- EIG pass@1: `True` (eval: 2/2)
- EIG questions asked: `2`
- EIG clarification tests and oracle responses:
  - Q1: `assert find_Rotations("abcdeabcde") == 5`
    - Oracle response: `True`; score: `0.8586`
  - Q2: `assert find_Rotations("aaaaaa") == 0`
    - Oracle response: `False`; score: `0.7951`

## EIG Incorrect After Clarification

### Task 1 (ambiguous)

- Task prompts (by condition):
  - Original task prompt:

```text
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
```
  - Mutated incomplete prompt:

```text
Write a function to find the minimum cost path to a position in a given cost matrix.
```
  - Mutated ambiguous prompt:

```text
Create a function to locate a less expensive route to reach a specific position (m, n) from the starting point (0, 0) within a given cost matrix cost[][].
```
  - Mutated contradictory prompt: `_not available in dataset_`
- One-shot pass@1: `False` (eval: 0/2)
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG clarification tests and oracle responses:
  - Q1: `assert min_cost(2, 2, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == (0, 0, 0)`
    - Oracle response: `False`; score: `0.0000`

### Task 1 (incomplete)

- Task prompts (by condition):
  - Original task prompt:

```text
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
```
  - Mutated incomplete prompt:

```text
Write a function to find the minimum cost path to a position in a given cost matrix.
```
  - Mutated ambiguous prompt:

```text
Create a function to locate a less expensive route to reach a specific position (m, n) from the starting point (0, 0) within a given cost matrix cost[][].
```
  - Mutated contradictory prompt: `_not available in dataset_`
- One-shot pass@1: `False` (eval: 0/2)
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG clarification tests and oracle responses:
  - Q1: `assert min_cost([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 1 + 4 + 7`
    - Oracle response: `False`; score: `0.0000`

### Task 1 (original)

- Task prompts (by condition):
  - Original task prompt:

```text
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
```
  - Mutated incomplete prompt:

```text
Write a function to find the minimum cost path to a position in a given cost matrix.
```
  - Mutated ambiguous prompt:

```text
Create a function to locate a less expensive route to reach a specific position (m, n) from the starting point (0, 0) within a given cost matrix cost[][].
```
  - Mutated contradictory prompt: `_not available in dataset_`
- One-shot pass@1: `False` (eval: 0/2)
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG clarification tests and oracle responses:
  - Q1: `assert min_cost([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (2, 2)) == 15`
    - Oracle response: `False`; score: `0.0000`

### Task 2 (ambiguous)

- Task prompts (by condition):
  - Original task prompt:

```text
Write a function to find the similar elements from the given two tuple lists.
```
  - Mutated incomplete prompt:

```text
Write a function to find similar elements shared between lists.
```
  - Mutated ambiguous prompt:

```text
Write a function to compare groups and find equal items.
```
  - Mutated contradictory prompt: `_not available in dataset_`
- One-shot pass@1: `False` (eval: 0/2)
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG clarification tests and oracle responses:
  - Q1: `assert similar_elements([1, 2, 3], [1, 2, 3]) == [1, 2, 3]`
    - Oracle response: `False`; score: `0.7834`

### Task 2 (incomplete)

- Task prompts (by condition):
  - Original task prompt:

```text
Write a function to find the similar elements from the given two tuple lists.
```
  - Mutated incomplete prompt:

```text
Write a function to find similar elements shared between lists.
```
  - Mutated ambiguous prompt:

```text
Write a function to compare groups and find equal items.
```
  - Mutated contradictory prompt: `_not available in dataset_`
- One-shot pass@1: `False` (eval: 0/2)
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `2`
- EIG clarification tests and oracle responses:
  - Q1: `assert similar_elements([1, 2, 2], [2, 2, 3]) == [2, 2]`
    - Oracle response: `False`; score: `0.7834`
  - Q2: `assert similar_elements([1, 2, 2], [2, 3, 3]) == [2]`
    - Oracle response: `False`; score: `0.8575`

### Task 2 (original)

- Task prompts (by condition):
  - Original task prompt:

```text
Write a function to find the similar elements from the given two tuple lists.
```
  - Mutated incomplete prompt:

```text
Write a function to find similar elements shared between lists.
```
  - Mutated ambiguous prompt:

```text
Write a function to compare groups and find equal items.
```
  - Mutated contradictory prompt: `_not available in dataset_`
- One-shot pass@1: `False` (eval: 0/2)
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG clarification tests and oracle responses:
  - Q1: `assert similar_elements(([1, 2, 3], [4, 5, 6])) == []`
    - Oracle response: `False`; score: `0.0000`

## Notes

- `Oracle response` is the boolean result from running each clarification test on oracle code.
- This report only includes examples where EIG actually asked at least one clarification question.
- `contradictory` prompts are shown when present in the dataset; otherwise marked unavailable.
