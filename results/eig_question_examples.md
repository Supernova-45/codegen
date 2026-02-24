# EIG Clarification Examples (Expanded)

Source run: `results/ablation_stage0.jsonl` (completed stage0 ablation).

## EIG Correct After Clarification (reference)

### Win 1: Task 3 (original)

- Original prompt:

```text
Write a python function to identify non-prime numbers.
```
- Modified prompt (original condition; unchanged):

```text
Write a python function to identify non-prime numbers.
```
- EIG pass@1: `True` (eval: 2/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert is_not_prime(4) == True`
    - Oracle response: `True`; EIG score: `0.0000`
- Generated code (EIG final submission):

```python
def is_not_prime(n):
    if n <= 1:
        return True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return True
    return False
```
- Hidden tests failed by EIG submission:
  - `assert is_not_prime(10) == True`
    - Error: `NoResult`
  - `assert is_not_prime(35) == True`
    - Error: `NoResult`

### Win 2: Task 3 (incomplete)

- Original prompt:

```text
Write a python function to identify non-prime numbers.
```
- Modified prompt (incomplete):

```text
Write a function to identify numbers in a list.
```
- EIG pass@1: `True` (eval: 2/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert is_not_prime(2) == False  # Prime numbers are greater than 1`
    - Oracle response: `True`; EIG score: `0.7834`
- Generated code (EIG final submission):

```python
def is_not_prime(n):
    if n < 2:
        return True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return True
    return False
```
- Hidden tests failed by EIG submission:
  - `assert is_not_prime(10) == True`
    - Error: `NoResult`
  - `assert is_not_prime(35) == True`
    - Error: `NoResult`

### Win 3: Task 3 (ambiguous)

- Original prompt:

```text
Write a python function to identify non-prime numbers.
```
- Modified prompt (ambiguous):

```text
Write a function to pick out non prime values from a set.
```
- EIG pass@1: `True` (eval: 2/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert not is_not_prime(2)`
    - Oracle response: `True`; EIG score: `0.5386`
- Generated code (EIG final submission):

```python
def is_not_prime(n):
    if n < 2:
        return True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return True
    return False
```
- Hidden tests failed by EIG submission:
  - `assert is_not_prime(10) == True`
    - Error: `NoResult`
  - `assert is_not_prime(35) == True`
    - Error: `NoResult`

### Win 4: Task 4 (incomplete)

- Original prompt:

```text
Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
```
- Modified prompt (incomplete):

```text
Write a function to find the largest integers from a given list of numbers.
```
- EIG pass@1: `True` (eval: 2/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert heap_queue_largest([2, 3, 4, 5], 5)`
    - Oracle response: `True`; EIG score: `0.7834`
- Generated code (EIG final submission):

```python
import heapq

def heap_queue_largest(numbers, k):
    return heapq.nlargest(k, numbers)
```
- Hidden tests failed by EIG submission:
  - `assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] `
    - Error: `NoResult`
  - `assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]`
    - Error: `NoResult`

## EIG Incorrect After Clarification (expanded failure set)

### Failure 1: Task 1 (original)

- Original prompt:

```text
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
```
- Modified prompt (original condition; unchanged):

```text
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `2`
- EIG questions + oracle responses:
  - Q1: `assert min_cost(2, 2, [[10, 15, 20], [20, 25, 30], [30, 35, 40]]) == 130`
    - Oracle response: `False`; EIG score: `0.0000`
  - Q2: `assert min_cost(1, 1, [[0]]) == 0`
    - Oracle response: `False`; EIG score: `0.5386`
- Generated code (EIG final submission):

```python
def min_cost(cost, m, n):
    dp = [[0 for _ in range(n)] for _ in range(m)]

    dp[0][0] = cost[0][0]

    # initialize first row
    for i in range(1, n):
        dp[0][i] = dp[0][i-1] + cost[0][i]

    # initialize first column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + cost[i][0]

    # fill rest of matrix
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = cost[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]
```
- Hidden tests failed by EIG submission:
  - `assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12`
    - Error: `NoResult`
  - `assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `AssertionError()`
  - `AssertionError()`

### Failure 2: Task 1 (ambiguous)

- Original prompt:

```text
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
```
- Modified prompt (ambiguous):

```text
Create a function to locate a less expensive route to reach a specific position (m, n) from the starting point (0, 0) within a given cost matrix cost[][].
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `2`
- EIG questions + oracle responses:
  - Q1: `assert min_cost(0, 0, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 1`
    - Oracle response: `False`; EIG score: `0.0000`
  - Q2: `assert min_cost(1, 2, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 3`
    - Oracle response: `False`; EIG score: `0.7834`
- Generated code (EIG final submission):

```python
def min_cost(cost, m, n):
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = cost[i-1][j-1] + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```
- Hidden tests failed by EIG submission:
  - `assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12`
    - Error: `NoResult`
  - `assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `AssertionError()`
  - `AssertionError()`

### Failure 3: Task 2 (original)

- Original prompt:

```text
Write a function to find the similar elements from the given two tuple lists.
```
- Modified prompt (original condition; unchanged):

```text
Write a function to find the similar elements from the given two tuple lists.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `2`
- EIG questions + oracle responses:
  - Q1: `assert similar_elements((1, 2, 3), (2, 3, 4)) == {(1, 2, 3), (2, 3, 4)}`
    - Oracle response: `False`; EIG score: `0.0000`
  - Q2: `assert similar_elements((1, 2, 3), (4, 5, 6)) == set()`
    - Oracle response: `False`; EIG score: `0.5386`
- Generated code (EIG final submission):

```python
def similar_elements(tuple_list1, tuple_list2):
    set1 = set(tuple(t) for t in tuple_list1)
    set2 = set(tuple(t) for t in tuple_list2)
    return [t for t in tuple_list1 if t in set2]
```
- Hidden tests failed by EIG submission:
  - `assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)`
    - Error: `NoResult`
  - `assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `TypeError("'int' object is not iterable")`
  - `TypeError("'int' object is not iterable")`

### Failure 4: Task 5 (ambiguous)

- Original prompt:

```text
Write a function to find the number of ways to fill it with 2 x 1 dominoes for the given 3 x n board.
```
- Modified prompt (ambiguous):

```text
Create a program that determines the quantity of possibilities to populate it with 2 x 1 units for a provided 3 x variable sized game board.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `2`
- EIG questions + oracle responses:
  - Q1: `assert count_ways(3) == 2`
    - Oracle response: `False`; EIG score: `0.8586`
  - Q2: `assert count_ways(1) == 0`
    - Oracle response: `True`; EIG score: `0.7951`
- Generated code (EIG final submission):

```python
def count_ways(n):
    if n <= 1:
        return 0
    if n == 2:
        return 1
    ways = [0]*(n+1)
    ways[1] = 0
    ways[2] = 1
    for i in range(3, n+1):
        ways[i] = ways[i-1] + ways[i-2]
    return ways[n]
```
- Hidden tests failed by EIG submission:
  - `assert count_ways(8) == 153`
    - Error: `NoResult`
  - `assert count_ways(12) == 2131`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `AssertionError()`
  - `AssertionError()`

### Failure 5: Task 7 (ambiguous)

- Original prompt:

```text
Write a function to find all words which are at least 4 characters long in a string by using regex.
```
- Modified prompt (ambiguous):

```text
Craft a function to identify all groupings of letters in a text string by employing a pattern recognition method. These groupings should be of some length using regex.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `2`
- EIG questions + oracle responses:
  - Q1: `assert find_char_long("a") == []`
    - Oracle response: `True`; EIG score: `0.8586`
  - Q2: `assert find_char_long("ab") == []`
    - Oracle response: `True`; EIG score: `0.7951`
- Generated code (EIG final submission):

```python
import re

def find_char_long(text):
    return re.findall(r"([a-zA-Z])\1+", text)
```
- Hidden tests failed by EIG submission:
  - `assert find_char_long('Jing Eco and Tech') == ['Jing', 'Tech']`
    - Error: `NoResult`
  - `assert find_char_long('Jhingai wulu road Zone 3') == ['Jhingai', 'wulu', 'road', 'Zone']`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `AssertionError()`
  - `AssertionError()`

### Failure 6: Task 1 (incomplete)

- Original prompt:

```text
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].
```
- Modified prompt (incomplete):

```text
Write a function to find the minimum cost path to a position in a given cost matrix.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert min_cost(1, 1, [[0, 0], [0, 0]]) == 0`
    - Oracle response: `False`; EIG score: `0.7834`
- Generated code (EIG final submission):

```python
def min_cost(matrix, x, y, cost_func):
    rows, cols = len(matrix), len(matrix[0])
    dp = [[float('inf')] * cols for _ in range(rows)]

    dp[0][0] = matrix[0][0]
    for i in range(1, rows):
        dp[i][0] = dp[i-1][0] + matrix[i][0]
    for j in range(1, cols):
        dp[0][j] = dp[0][j-1] + matrix[0][j]

    for i in range(1, rows):
        for j in range(1, cols):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + matrix[i][j]

    return dp[x][y]
```
- Hidden tests failed by EIG submission:
  - `assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12`
    - Error: `NoResult`
  - `assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `TypeError("min_cost() missing 1 required positional argument: 'cost_func'")`
  - `TypeError("min_cost() missing 1 required positional argument: 'cost_func'")`

### Failure 7: Task 2 (incomplete)

- Original prompt:

```text
Write a function to find the similar elements from the given two tuple lists.
```
- Modified prompt (incomplete):

```text
Write a function to find similar elements shared between lists.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert similar_elements([1, 2, 3], [2, 3, 4]) == [2, 3]`
    - Oracle response: `False`; EIG score: `0.8586`
- Generated code (EIG final submission):

```python
def similar_elements(arg1, arg2):
    return set(arg1) & set(arg2)
```
- Hidden tests failed by EIG submission:
  - `assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)`
    - Error: `NoResult`
  - `assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `AssertionError()`
  - `AssertionError()`

### Failure 8: Task 2 (ambiguous)

- Original prompt:

```text
Write a function to find the similar elements from the given two tuple lists.
```
- Modified prompt (ambiguous):

```text
Write a function to compare groups and find equal items.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert similar_elements([1, 2, 3], [1, 2, 3]) == True`
    - Oracle response: `False`; EIG score: `0.0000`
- Generated code (EIG final submission):

```python
def similar_elements(arg1, arg2):
    return set(arg1).intersection(set(arg2))
```
- Hidden tests failed by EIG submission:
  - `assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)`
    - Error: `NoResult`
  - `assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `AssertionError()`
  - `AssertionError()`

### Failure 9: Task 4 (original)

- Original prompt:

```text
Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
```
- Modified prompt (original condition; unchanged):

```text
Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert heap_queue_largest([3, 2, 5, 1, 4]) == 5`
    - Oracle response: `False`; EIG score: `0.0000`
- Generated code (EIG final submission):

```python
import heapq

def heap_queue_largest(num1, num2):
    nums = [num1, num2]
    heapq.heapify(nums)
    return heapq.heappop(nums), heapq.heappop(nums)
```
- Hidden tests failed by EIG submission:
  - `assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] `
    - Error: `NoResult`
  - `assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `TypeError("'<' not supported between instances of 'list' and 'int'")`
  - `TypeError("'<' not supported between instances of 'list' and 'int'")`

### Failure 10: Task 4 (ambiguous)

- Original prompt:

```text
Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
```
- Modified prompt (ambiguous):

```text
Create a method that can determine the bigger values in a selection of numerical values, with the help of a common heap queue method.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert heap_queue_largest([], []) == []`
    - Oracle response: `False`; EIG score: `0.7834`
- Generated code (EIG final submission):

```python
import heapq

def heap_queue_largest(arg1, arg2):
    heap = []
    heapq.heappush(heap, arg1)
    heapq.heappush(heap, arg2)
    return heapq.nlargest(2, heap)
```
- Hidden tests failed by EIG submission:
  - `assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] `
    - Error: `NoResult`
  - `assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `TypeError("'<' not supported between instances of 'int' and 'list'")`
  - `TypeError("'<' not supported between instances of 'int' and 'list'")`

### Failure 11: Task 6 (incomplete)

- Original prompt:

```text
Write a python function to check whether the two numbers differ at one bit position only or not.
```
- Modified prompt (incomplete):

```text
Write a function to check if two numbers differ.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert differ_At_One_Bit_Pos(1, 0)`
    - Oracle response: `True`; EIG score: `0.7834`
- Generated code (EIG final submission):

```python
def differ_At_One_Bit_Pos(arg1, arg2):
    return (arg1 ^ arg2) != 0
```
- Hidden tests failed by EIG submission:
  - `assert differ_At_One_Bit_Pos(15,8) == False`
    - Error: `NoResult`
  - `assert differ_At_One_Bit_Pos(2,4) == False`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `AssertionError()`
  - `AssertionError()`

### Failure 12: Task 6 (ambiguous)

- Original prompt:

```text
Write a python function to check whether the two numbers differ at one bit position only or not.
```
- Modified prompt (ambiguous):

```text
Write a function that checks if two numerical values are distinguishable through a minimal binary alteration
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert differ_At_One_Bit_Pos(0, 1)`
    - Oracle response: `True`; EIG score: `0.5386`
- Generated code (EIG final submission):

```python
def differ_At_One_Bit_Pos(arg1, arg2):
    if arg1 != arg2:
        return True
    max_val = max(arg1, arg2)
    while max_val > 0:
        if (arg1 & max_val) != (arg2 & max_val):
            return True
        max_val >>= 1
    return False
```
- Hidden tests failed by EIG submission:
  - `assert differ_At_One_Bit_Pos(15,8) == False`
    - Error: `NoResult`
  - `assert differ_At_One_Bit_Pos(2,4) == False`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `AssertionError()`
  - `AssertionError()`

### Failure 13: Task 8 (original)

- Original prompt:

```text
Write a function to find squares of individual elements in a list using lambda function.
```
- Modified prompt (original condition; unchanged):

```text
Write a function to find squares of individual elements in a list using lambda function.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert square_nums([1, 2, 3]) == [1, 4, 9]`
    - Oracle response: `False`; EIG score: `0.0000`
- Generated code (EIG final submission):

```python
def square_nums(lst):
    return list(map(lambda x: x**2, lst))
```
- Hidden tests failed by EIG submission:
  - `assert square_nums([10,20,30])==([100,400,900])`
    - Error: `NoResult`
  - `assert square_nums([12,15])==([144,225])`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `NameError("name 'map' is not defined")`
  - `NameError("name 'map' is not defined")`

### Failure 14: Task 10 (incomplete)

- Original prompt:

```text
Write a function to get the n smallest items from a dataset.
```
- Modified prompt (incomplete):

```text
Write a function to get items from a dataset.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert small_nnum(1, {"a": 1, "b": 2}) == {"a"}`
    - Oracle response: `False`; EIG score: `0.0000`
- Generated code (EIG final submission):

```python
def small_nnum(dataset, n):
    return dataset[:n]
```
- Hidden tests failed by EIG submission:
  - `assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],5)==[10,20,20,40,50]`
    - Error: `NoResult`
  - `assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],3)==[10,20,20]`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `Timeout`
  - `Timeout`

### Failure 15: Task 10 (ambiguous)

- Original prompt:

```text
Write a function to get the n smallest items from a dataset.
```
- Modified prompt (ambiguous):

```text
Write a function to select the smallest of a data group.
```
- EIG pass@1: `False` (eval: 0/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert small_nnum(1, 2) == 1`
    - Oracle response: `False`; EIG score: `0.5386`
- Generated code (EIG final submission):

```python
def small_nnum(arr1, arr2):
    return min(min(arr1), min(arr2))
```
- Hidden tests failed by EIG submission:
  - `assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],5)==[10,20,20,40,50]`
    - Error: `NoResult`
  - `assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],3)==[10,20,20]`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `TypeError("'int' object is not iterable")`
  - `TypeError("'int' object is not iterable")`

### Failure 16: Task 7 (incomplete)

- Original prompt:

```text
Write a function to find all words which are at least 4 characters long in a string by using regex.
```
- Modified prompt (incomplete):

```text
Write a function to find long words in a string.
```
- EIG pass@1: `False` (eval: 1/2)
- EIG questions asked: `1`
- EIG questions + oracle responses:
  - Q1: `assert find_char_long("longest") == ["longest"]`
    - Oracle response: `True`; EIG score: `0.5386`
- Generated code (EIG final submission):

```python
def find_char_long(s):
    max_len = 0
    long_words = []
    for word in s.split():
        if len(word) > max_len:
            max_len = len(word)
            long_words = [word]
        elif len(word) == max_len:
            long_words.append(word)
    return long_words
```
- Hidden tests failed by EIG submission:
  - `assert find_char_long('Jing Eco and Tech') == ['Jing', 'Tech']`
    - Error: `NoResult`
  - `assert find_char_long('Jhingai wulu road Zone 3') == ['Jhingai', 'wulu', 'road', 'Zone']`
    - Error: `NoResult`
- Recorded eval errors from run:
  - `AssertionError()`

## Notes

- `Original prompt` comes from the `original` condition for the same task ID.
- `Modified prompt` is the prompt used for the example condition (`incomplete`/`ambiguous`/`original`).
- `Hidden tests failed` are recomputed from `data/mbpp_variants.jsonl` hidden tests against EIG final code.
- Code snippets are truncated for readability when long.
- This version focuses on EIG behavior only and does not require one-shot rows to be present.
