# EIG Failure Casebook (Strict Eval)

Source: `results/strict_eval_final.jsonl` (strategy=`eig-tests`, failures only).

This casebook is intended to diagnose why EIG underperforms by showing prompts, asked vs unasked candidate tests, and resulting code/errors.

## Case 1: Task 5 (ambiguous)

- Outcome: `pass_at_1=False`; eval `0/2`; questions asked `3`
- Original prompt:
```text
Write a function to find the number of ways to fill it with 2 x 1 dominoes for the given 3 x n board.
```
- Modified prompt (ambiguous):
```text
Create a program that determines the quantity of possibilities to populate it with 2 x 1 units for a provided 3 x variable sized game board.
```
- Tests EIG asked (with oracle response):
  - Q1: `assert count_ways(0) == 0`
    - observed: `False`; score: `0.8586`
  - Q2: `assert count_ways(-1) == 0`
    - observed: `False`; score: `0.7951`
  - Q3: `assert count_ways(2) == 3`
    - observed: `True`; score: `0.8585`
- Other valid tests it could have asked (from same rounds):
  - `assert count_ways(3) == 2`
  - `assert count_ways(1) == 0`
  - `assert count_ways(-1) == 0`
  - `assert count_ways(3.5) == 0 # edge case for non-integer value`
  - `assert count_ways(1) == 1`
- Final generated code submitted by EIG:
```python
def count_ways(n):
    if n < 0:
        return 0
    elif n == 0:
        return 1
    else:
        return count_ways(n-1) + count_ways(n-2) + count_ways(n-3)
```
- Hidden tests for this task variant:
  - `assert count_ways(8) == 153`
  - `assert count_ways(12) == 2131`
- Errors recorded during eval:
  - `AssertionError()`
  - `AssertionError()`
- Likely failure mode(s): logic mismatch on hidden tests

## Case 2: Task 9 (ambiguous)

- Outcome: `pass_at_1=False`; eval `0/2`; questions asked `3`
- Original prompt:
```text
Write a python function to find the minimum number of rotations required to get the same string.
```
- Modified prompt (ambiguous):
```text
Write a function that finds the number  of rotations changes on text until something gets the same .
```
- Tests EIG asked (with oracle response):
  - Q1: `assert find_Rotations("abc") == 0`
    - observed: `False`; score: `0.5386`
  - Q2: `assert find_Rotations("abca") == 1`
    - observed: `False`; score: `0.8309`
  - Q3: `assert find_Rotations("") == 0`
    - observed: `True`; score: `0.7834`
- Other valid tests it could have asked (from same rounds):
  - `assert find_Rotations("abcabc") == 0`
  - `assert find_Rotations("abcaba") == 0`
  - `assert find_Rotations("a") == 0`
  - `assert find_Rotations("abcabcabc") == 0`
  - `assert find_Rotations("ab") == 0`
- Final generated code submitted by EIG:
```python
def find_Rotations(s):
    if len(s) == 0:
        return 0
    for i in range(len(s)):
        if s == s[i:] + s[:i]:
            return i
    return -1
```
- Hidden tests for this task variant:
  - `assert find_Rotations("ab") == 2`
  - `assert find_Rotations("abc") == 3`
- Errors recorded during eval:
  - `AssertionError()`
  - `AssertionError()`
- Likely failure mode(s): logic mismatch on hidden tests

## Case 3: Task 2 (ambiguous)

- Outcome: `pass_at_1=False`; eval `0/2`; questions asked `2`
- Original prompt:
```text
Write a function to find the similar elements from the given two tuple lists.
```
- Modified prompt (ambiguous):
```text
Write a function to compare groups and find equal items.
```
- Tests EIG asked (with oracle response):
  - Q1: `assert similar_elements([1, 1, 1], [1, 1, 1]) == [1, 1, 1]`
    - observed: `False`; score: `0.7834`
  - Q2: `assert similar_elements([1, 2, 1], [2, 1]) == [1, 2]`
    - observed: `False`; score: `0.8575`
- Other valid tests it could have asked (from same rounds):
  - `assert similar_elements([1, 2, 3], [1, 2, 4]) == [1, 2]`
  - `assert similar_elements([1, 2, 3], [1, 2, 2]) == [1, 2]`
  - `assert similar_elements([], []) == []`
  - `assert similar_elements("abc", "def") == []`
  - `assert similar_elements([1, 1, 1], [1, 1, 1]) == [1, 1, 1]`
- Final generated code submitted by EIG:
```python
def similar_elements(arg1, arg2):
    return list(set(arg1) & set(arg2))
```
- Hidden tests for this task variant:
  - `assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)`
  - `assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)`
- Errors recorded during eval:
  - `AssertionError()`
  - `AssertionError()`
- Likely failure mode(s): logic mismatch on hidden tests

## Case 4: Task 1 (incomplete)

- Outcome: `pass_at_1=False`; eval `0/2`; questions asked `2`
- Original prompt:
```text
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
```
- Modified prompt (incomplete):
```text
Write a function to find the minimum cost path to a position in a given cost matrix.
```
- Tests EIG asked (with oracle response):
  - Q1: `assert min_cost(0, 0, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 1`
    - observed: `False`; score: `0.5386`
  - Q2: `assert min_cost(0, 0, []) == 0`
    - observed: `False`; score: `0.6024`
- Other valid tests it could have asked (from same rounds):
  - `assert min_cost(2, 2, [[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == 0`
  - `assert min_cost(0, 0, [[1], [2]]) == 1`
  - `assert min_cost(0, 0, [[1, 2, 3]]) == 1`
  - `assert min_cost(0, 0, [[1, 1, 1], [1, 1, 1], [1, 1, 1]]) == 1`
  - `assert min_cost(0, 0, [[1, 2], [3, 4]]) == 1`
- Final generated code submitted by EIG:
```python
def min_cost(x, y, cost_matrix):
    if not cost_matrix or x < 0 or y < 0 or x >= len(cost_matrix) or y >= len(cost_matrix[0]):
        return float('inf')
    if x == 0 and y == 0:
        return cost_matrix[x][y]
    return cost_matrix[x][y] + min(min_cost(x-1, y, cost_matrix), min_cost(x, y-1, cost_matrix))
```
- Hidden tests for this task variant:
  - `assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12`
  - `assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16`
- Errors recorded during eval:
  - `TypeError("'<' not supported between instances of 'list' and 'int'")`
  - `TypeError("'<' not supported between instances of 'list' and 'int'")`
- Likely failure mode(s): signature/argument mismatch

## Case 5: Task 7 (incomplete)

- Outcome: `pass_at_1=False`; eval `1/2`; questions asked `2`
- Original prompt:
```text
Write a function to find all words which are at least 4 characters long in a string by using regex.
```
- Modified prompt (incomplete):
```text
Write a function to find long words in a string.
```
- Tests EIG asked (with oracle response):
  - Q1: `assert find_char_long("abc def ghi") == ["abc", "def", "ghi"]`
    - observed: `False`; score: `0.8586`
  - Q2: `assert find_char_long("abcdef") == ["abcdef"]`
    - observed: `True`; score: `0.7951`
- Other valid tests it could have asked (from same rounds):
  - `assert find_char_long("a b c") == []`
  - `assert find_char_long("abc") == []`
  - `assert find_char_long("abcdefghij") == ["abcdefghij"]`
  - `assert find_char_long("") == []`
  - `assert find_char_long("a") == []`
- Final generated code submitted by EIG:
```python
def find_char_long(s):
    words = s.split()
    return [word for word in words if len(word) > len(max(words, key=len)) / 2]
```
- Hidden tests for this task variant:
  - `assert find_char_long('Jing Eco and Tech') == ['Jing', 'Tech']`
  - `assert find_char_long('Jhingai wulu road Zone 3') == ['Jhingai', 'wulu', 'road', 'Zone']`
- Errors recorded during eval:
  - `AssertionError()`
- Likely failure mode(s): logic mismatch on hidden tests

## Case 6: Task 2 (incomplete)

- Outcome: `pass_at_1=False`; eval `0/2`; questions asked `1`
- Original prompt:
```text
Write a function to find the similar elements from the given two tuple lists.
```
- Modified prompt (incomplete):
```text
Write a function to find similar elements shared between lists.
```
- Tests EIG asked (with oracle response):
  - Q1: `assert similar_elements([1, 1, 1], [1, 1, 1]) == [1, 1, 1]`
    - observed: `False`; score: `0.5386`
- Other valid tests it could have asked (from same rounds):
  - _none (no additional valid tests logged)_
- Final generated code submitted by EIG:
```python
def similar_elements(arg1, arg2):
    return list(set(arg1) & set(arg2))
```
- Hidden tests for this task variant:
  - `assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)`
  - `assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)`
- Errors recorded during eval:
  - `AssertionError()`
  - `AssertionError()`
- Likely failure mode(s): logic mismatch on hidden tests

## Case 7: Task 9 (original)

- Outcome: `pass_at_1=False`; eval `0/2`; questions asked `3`
- Original prompt:
```text
Write a python function to find the minimum number of rotations required to get the same string.
```
- Modified prompt (original):
```text
Write a python function to find the minimum number of rotations required to get the same string.
```
- Tests EIG asked (with oracle response):
  - Q1: `assert find_Rotations("abc") == 0`
    - observed: `False`; score: `0.8586`
  - Q2: `assert find_Rotations("aaaa") == 0`
    - observed: `False`; score: `0.7951`
  - Q3: `assert find_Rotations("a") == 0`
    - observed: `False`; score: `0.8585`
- Other valid tests it could have asked (from same rounds):
  - `assert find_Rotations("xyz") == 0`
  - `assert find_Rotations("a") == 0`
  - `assert find_Rotations("xyz" * 1000) == 0`
  - `assert find_Rotations("abcabc") == 0`
  - `assert find_Rotations("abcabca") == 0`
- Final generated code submitted by EIG:
```python
def find_Rotations(s):
    for i in range(len(s)):
        if s == s[i:] + s[:i]:
            return i
    return -1
```
- Hidden tests for this task variant:
  - `assert find_Rotations("ab") == 2`
  - `assert find_Rotations("abc") == 3`
- Errors recorded during eval:
  - `AssertionError()`
  - `AssertionError()`
- Likely failure mode(s): logic mismatch on hidden tests

## Case 8: Task 5 (original)

- Outcome: `pass_at_1=False`; eval `0/2`; questions asked `2`
- Original prompt:
```text
Write a function to find the number of ways to fill it with 2 x 1 dominoes for the given 3 x n board.
```
- Modified prompt (original):
```text
Write a function to find the number of ways to fill it with 2 x 1 dominoes for the given 3 x n board.
```
- Tests EIG asked (with oracle response):
  - Q1: `assert count_ways(1) == 1`
    - observed: `False`; score: `0.7834`
  - Q2: `assert count_ways(0) == 0`
    - observed: `False`; score: `0.8583`
- Other valid tests it could have asked (from same rounds):
  - `assert count_ways(3) == 3`
  - `assert count_ways(2) == 1`
  - `assert count_ways(3) == 2`
  - `assert count_ways(4) == 3`
  - `assert count_ways(5) == 5`
- Final generated code submitted by EIG:
```python
def count_ways(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return count_ways(n-1) + count_ways(n-2)
```
- Hidden tests for this task variant:
  - `assert count_ways(8) == 153`
  - `assert count_ways(12) == 2131`
- Errors recorded during eval:
  - `AssertionError()`
  - `AssertionError()`
- Likely failure mode(s): logic mismatch on hidden tests

## Case 9: Task 6 (original)

- Outcome: `pass_at_1=False`; eval `0/2`; questions asked `2`
- Original prompt:
```text
Write a python function to check whether the two numbers differ at one bit position only or not.
```
- Modified prompt (original):
```text
Write a python function to check whether the two numbers differ at one bit position only or not.
```
- Tests EIG asked (with oracle response):
  - Q1: `assert differ_At_One_Bit_Pos(10, 5) == True`
    - observed: `False`; score: `0.8586`
  - Q2: `assert differ_At_One_Bit_Pos(0, 1) == True`
    - observed: `True`; score: `0.7834`
- Other valid tests it could have asked (from same rounds):
  - `assert differ_At_One_Bit_Pos(1, 0) == True`
  - `assert differ_At_One_Bit_Pos(0, 0) == False`
- Final generated code submitted by EIG:
```python
def differ_At_One_Bit_Pos(arg1, arg2):
    return bin(arg1 ^ arg2).count('1') == 1
```
- Hidden tests for this task variant:
  - `assert differ_At_One_Bit_Pos(15,8) == False`
  - `assert differ_At_One_Bit_Pos(2,4) == False`
- Errors recorded during eval:
  - `NameError("name 'bin' is not defined")`
  - `NameError("name 'bin' is not defined")`
- Likely failure mode(s): mixed/unclear (inspect code + hidden tests)

## Case 10: Task 1 (ambiguous)

- Outcome: `pass_at_1=False`; eval `0/2`; questions asked `1`
- Original prompt:
```text
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
```
- Modified prompt (ambiguous):
```text
Create a function to locate a less expensive route to reach a specific position (m, n) from the starting point (0, 0) within a given cost matrix cost[][].
```
- Tests EIG asked (with oracle response):
  - Q1: `assert min_cost(0, 0, [[0, 1, 2], [3, 4, 5], [6, 7, 8]]) == 0`
    - observed: `False`; score: `0.5386`
- Other valid tests it could have asked (from same rounds):
  - `assert min_cost(0, 0, [[0, 1, 1], [3, 4, 5], [6, 7, 8]]) == 0`
  - `assert min_cost(0, 0, [[0, float('inf')], [0, 0]]) == 0`
- Final generated code submitted by EIG:
```python
def min_cost(m, n, cost):
    dp = [[0]*n for _ in range(m)]
    dp[0][0] = cost[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + cost[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + cost[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = cost[i][j] + min(dp[i-1][j], dp[i][j-1])
    return dp[m-1][n-1]
```
- Hidden tests for this task variant:
  - `assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12`
  - `assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16`
- Errors recorded during eval:
  - `TypeError("'list' object cannot be interpreted as an integer")`
  - `TypeError("'list' object cannot be interpreted as an integer")`
- Likely failure mode(s): signature/argument mismatch

## Cross-Case Patterns

- Many failures include `TypeError` -> generated code/test call shape mismatch, not just wrong algorithm.
- Some asked tests are low-information (score near 0), which does not improve posterior selection much.
- In several cases, there were additional valid tests available but EIG selected only one/few.
