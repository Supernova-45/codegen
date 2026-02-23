## Goal

Large language models have become astonishingly good at generating working code. Yet even when LLM-generated code appears correct and passes basic sanity checks, it can fail on edge cases not explicitly clarified by the user. For instance, if a user asks for a script to remove duplicate names from a list, should it treat "Bob" and "bob" as the same name? Human programmers resolve these ambiguities by asking follow-up questions before writing any code. LLMs, on the other hand, make assumptions about intent, producing code misaligned with the task at hand. [Larbi et al.](https://arxiv.org/abs/2507.20439) quantify this cost: even minor ambiguities or omissions in task descriptions cause 20-40% absolute drops in pass@1 across state-of-the-art code models, and 60-90% of the code that compiles under unclear descriptions is semantically incorrect.

Our project asks: **can a code generator improve correctness by asking binary yes/no questions in the form of unit tests before producing code, and can it select those questions to maximize information gain?**

This project builds on two lines of recent work. [Grand et al.](https://arxiv.org/abs/2510.20886) develop Bayesian Experimental Design (BED) strategies for agents playing Collaborative Battleship, showing that selecting questions to maximize expected information gain (EIG) enables weak models to outperform both humans and frontier LLMs at a fraction of the cost. Concurrently, [TiCoder (Fakhoury et al., 2024)](https://arxiv.org/abs/2404.10100) demonstrated that test-driven intent clarification can improve pass@1 by up to 45.97% within 5 interactions. However, TiCoder either requires a human in the loop or simulates one with an oracle, and its test-ranking heuristic is not information-theoretically grounded. We bring *principled, fully automated test selection* based on BED to the code generation setting.

We make three contributions:

1. **EIG-based test selection** adapted from Grand et al.'s BED framework, replacing TiCoder's heuristic discriminative ranking with information-theoretically optimal query selection over a Bayesian posterior of candidate programs.
2. **An adaptive ask-or-submit decision**, adapted from the exploration/exploitation tradeoff in Grand et al., where the model decides at each step whether another test query is worth its cost or whether to submit code immediately.
3. **Evaluation on systematically underspecified prompts**, using the controlled mutation methodology of [Larbi et al.](https://arxiv.org/abs/2507.20439) to create incomplete and ambiguous task descriptions where clarification provides maximal value.

Can we get smaller, cheaper models to match larger ones by having them ask a few well-chosen questions? If structured test queries can recover accuracy lost to prompt underspecification, we gain insight into how to guide LLMs to behave less like autocomplete and more like human engineers.

## Task

Given a natural-language description of a Python function (signature and docstring), the model must output an implementation that passes a hidden test suite. Unlike standard one-shot code generation, the model is allowed to ask questions before submitting its final program. It may ask up to K_max binary yes/no questions, where each question is an executable unit test proposed by the model and answered by running it against a reference implementation. Critically, the model may also choose to *stop asking early* and submit code when it is sufficiently confident, making the number of questions adaptive rather than fixed.

Each question takes the form of an assertion:

```
assert f(input) == output
```

The answer is `True` if the assertion passes on the reference solution and `False` otherwise. We model the reference solution as a noisy oracle -- a binary symmetric channel BSC(epsilon) with small flip probability epsilon -- to account for flaky tests, non-determinism, or floating-point imprecision.

For example, suppose the prompt is: "Write a function `dedup(names)` that removes duplicate names from a list." One area of ambiguity is case sensitivity. The model might ask:

```
assert dedup(["Bob", "bob"]) == ["Bob"]
```

If the answer is `True`, the intended behavior is case-insensitive; if `False`, `"Bob"` and `"bob"` are treated as distinct. After asking a small number of such tests, the model outputs its implementation and is scored on the hidden test suite.

## Data

We evaluate on two standard Python code-generation benchmarks, plus systematically underspecified variants of each.

**HumanEval+** ([Chen et al., 2021](https://arxiv.org/abs/2107.03374); [EvalPlus](https://arxiv.org/abs/2305.01210)): 164 hand-written Python problems, each with a function signature, natural-language docstring, canonical solution, and unit tests. We use the EvalPlus HumanEval+ variant, which extends the original test suites by 80x for more comprehensive evaluation.

**MBPP** ([Austin et al., 2021](https://arxiv.org/abs/2108.07732)): 974 crowd-sourced Python problems with short natural-language descriptions (typically 1-2 sentences), canonical solutions, and test cases. MBPP descriptions are naturally terser and more ambiguous than HumanEval, providing a complementary evaluation setting.

**Underspecified variants.** A key challenge is that HumanEval docstrings are often detailed enough that test queries may add little value. Following [Larbi et al.](https://arxiv.org/abs/2507.20439), we apply controlled mutations to the task descriptions of both benchmarks to create two degraded conditions:

- **Incomplete:** Strip edge-case specifications, remove illustrative examples, omit parameter types or return-value constraints. (e.g., "Write a function to find a missing value in a list" instead of "Write a function to find the missing number in a sorted array.")
- **Ambiguous:** Introduce vague wording with multiple plausible interpretations. (e.g., "Write a function to select the smallest of a data group" instead of "Write a function to get the n smallest items from a dataset.")

Following Larbi et al.'s methodology, we use GPT-4 as the mutation engine with structured mutation guidelines, followed by manual expert validation to ensure naturalness and that the intended quality issue is present. The original, well-specified descriptions serve as the reference oracle defining "intended behavior." This creates a three-level spectrum -- *original*, *incomplete*, *ambiguous* -- over which we can measure how much test queries help as a function of prompt clarity.

## Methods

Given an underspecified prompt, there are often multiple plausible interpretations of the intended behavior. A well-chosen unit test rules out many interpretations at once. Our system should choose the test whose outcome would maximally reduce uncertainty about which interpretation is correct, then generate code consistent with the discovered constraints.

**Model and hyperparameters.** We use Qwen2.5-Coder-7B-Instruct as our base code LLM, a strong open-source coding model that runs on a single A100 GPU via vLLM. We sample N = 50 candidate programs per problem at temperature T = 0.8 to approximate the space of plausible solutions, and generate 15 candidate tests per round. The maximum question budget is K_max = 5.

Our pipeline has four components:

1. A code LLM (Qwen2.5-Coder-7B-Instruct) to generate candidate solutions and propose candidate unit tests.
2. A **weighted particle posterior** over plausible intended behaviors, represented as a set of N candidate programs {(c_j, w_j)} with weights summing to 1.
3. A **query-selection rule** based on expected information gain (EIG) that chooses the most informative test under the current posterior.
4. An **ask-or-submit decision rule** that determines whether to ask another question or submit code, based on a one-step lookahead over the posterior.

For each code generation task, we proceed as follows:

### Step 1: Initialize the posterior

Sample N = 50 candidate implementations from the LLM given the prompt (temperature T = 0.8). Initialize uniform weights w_j = 1/N.

### Step 2: Iterative query selection

For up to K_max rounds:

**(a) Generate candidate tests.** Prompt the LLM to propose 15 candidate unit tests targeting edge cases and ambiguities.

**(b) Validate tests.** Execute each candidate test against all N candidate programs in a sandbox (5-second timeout, no filesystem/network access). Discard tests that throw exceptions on any candidate. Run each test twice to check for non-determinism; discard non-deterministic tests.

**(c) Score tests by EIG.** For each surviving candidate test t, compute the predictive pass probability under the current posterior:

> p_t = sum over j of w_j * 1{f_t(c_j) = 1}

where f_t(c_j) in {0, 1} indicates whether test t passes on candidate c_j. Then compute:

> EIG_epsilon(t) = H_b(epsilon + (1 - 2 * epsilon) * p_t) - H_b(epsilon)

where H_b(p) = -p log2(p) - (1-p) log2(1-p) is binary entropy and epsilon is the noise parameter of the oracle channel. EIG is maximized when p_t is near 0.5, i.e., when the candidate programs are maximally split on the test outcome. Select the test t* = argmax_t EIG_epsilon(t).

**(d) Ask-or-submit decision.** Before asking t*, check whether asking is worth the cost. Let p_best = max_j w_j be the current MAP confidence. Compute the expected post-question MAP confidence p_best_next(t*) by marginalizing over both possible answers (pass/fail). If p_best > gamma * p_best_next(t*), skip the remaining questions and proceed to submission. The discount factor gamma in [0, 1] trades off the cost of asking against the value of information.

**(e) Ask and update.** Submit t* to the reference oracle and observe the answer a_tilde in {0, 1}. Update the posterior via Bayesian reweighting:

> w_j_new is proportional to w_j * [(1 - epsilon) * 1{a_tilde = f_t*(c_j)} + epsilon * 1{a_tilde != f_t*(c_j)}]

This soft update downweights rather than eliminates inconsistent candidates, making the system robust to noisy or flaky test outcomes -- a key advantage over TiCoder's hard pruning.

### Step 3: Generate the final program

We compare two strategies:

- *Best-candidate selection:* Output the candidate program with the highest posterior weight.
- *Constraint-conditioned re-prompting:* Re-prompt the LLM with the original task description augmented by the discovered test constraints (the sequence of test/answer pairs), generating a fresh solution conditioned on all observed evidence.

## Baselines

To isolate the value of (i) clarification itself, (ii) the test format, and (iii) principled test selection, we compare against four baselines:

**One-shot generation:** Generate a solution directly from the prompt with no interaction.

**Random test querying:** Generate candidate tests with the LLM, select K tests uniformly at random (ignoring EIG), observe their answers, and generate a final program conditioned on the results. This controls for the effect of arbitrary interaction.

**Fixed edge-case templates:** Apply a hand-crafted library of common edge-case tests (empty input, single-element input, type boundary values, large inputs, duplicate elements) without any task-specific test generation. This isolates whether generic probing helps.

**Natural-language clarifying questions:** Instead of unit tests, the model asks free-form natural-language clarifying questions (e.g., "Should the function be case-sensitive?"), answered by an LLM judge with access to the reference solution. This isolates the value of the executable test format specifically versus natural-language clarification.

Each baseline is crossed with both final-generation strategies (best-candidate selection and constraint-conditioned re-prompting), yielding the following condition table:

| | Best candidate | Re-prompt |
|---|---|---|
| One-shot (no questions) | yes | -- |
| Random tests | yes | yes |
| Fixed edge-case templates | yes | yes |
| Natural-language questions | -- | yes |
| EIG-selected tests (ours) | yes | yes |

## Evaluation

**Primary metric.** We report *pass@1* on HumanEval+ and MBPP, computed using the EvalPlus evaluation harness.

**Performance vs. question budget.** We plot pass@1 as a function of questions asked (K = 0, 1, 2, ..., 5) to show the marginal value of each additional query. For the adaptive ask-or-submit condition, we report the empirical distribution of questions asked.

**Query efficiency.** We report accuracy gain per question and per token (total tokens consumed by test generation and LLM queries), enabling cost-normalized comparisons. This answers: is it cheaper to ask 3 well-chosen test queries or to use a model 10x larger?

**Breakdown by description quality.** We report pass@1 separately for the three description conditions -- original, incomplete, ambiguous -- to test our central hypothesis that test queries help more on underspecified prompts.

**Ambiguity-type analysis.** We categorize the underspecified prompts by the type of information removed (edge-case behavior, parameter types, ordering constraints, boundary conditions) and measure which categories are most "query-resolvable" via unit tests versus not. This identifies the boundary of what binary test queries can and cannot clarify.

**Cost comparison.** We compare the total inference cost (tokens x price) of our 7B model with K test queries against using a larger model (e.g., 70B) with no questions, to determine whether structured interaction is a cost-effective substitute for scale.

## Related Work

**Test-driven interactive code generation.** [TiCoder (Fakhoury et al., 2024)](https://arxiv.org/abs/2404.10100) pioneered the workflow of generating candidate tests, presenting them to a user for validation, and pruning inconsistent code suggestions. TiCoder ranks tests by a discriminative heuristic s_discr(t) = min(|G_t+|, |G_t-|) / max(|G_t+|, |G_t-|), which prefers tests that split candidates evenly but is not information-theoretically grounded. Our work replaces this with EIG-based selection from Bayesian Experimental Design and adds a soft posterior update (vs. hard pruning) and an adaptive stopping rule. [ClarifyCoder (2025)](https://arxiv.org/abs/2504.16331) fine-tunes models to ask natural-language clarifying questions, but does not use executable tests or information-theoretic selection.

**Bayesian Experimental Design for agents.** [Grand et al. (2024)](https://arxiv.org/abs/2510.20886) develop BED strategies for Collaborative Battleship, showing that EIG-based question selection with Monte Carlo posterior inference enables weak LMs to achieve superhuman performance at ~1% of frontier model cost. We adapt their framework -- weighted particle posteriors, EIG scoring, noisy-channel observation model, and the ask-or-act decision -- from spatial reasoning over hidden board states to reasoning over the space of plausible program behaviors.

**Robustness to underspecified prompts.** [Larbi et al. (2025)](https://arxiv.org/abs/2507.20439) systematically mutate HumanEval and MBPP descriptions to introduce ambiguity, incompleteness, and contradiction, showing 20-40% pass@1 drops. We adopt their mutation methodology to construct evaluation conditions where clarification is maximally valuable, and test whether our interactive pipeline can recover the lost accuracy.

## Compute Budget

We estimate the following resource requirements for the full experiment:

- **Model:** Qwen2.5-Coder-7B-Instruct, served via vLLM on 1x A100 (80GB) at ~30 tokens/sec.
- **Candidate programs:** 50 samples x 1,138 problems = ~57K generations (~200 tokens each = ~11.4M tokens).
- **Candidate tests:** 15 tests x 5 rounds x 1,138 problems = ~85K generations (~100 tokens each = ~8.5M tokens).
- **Total per condition:** ~20M tokens, roughly 3-5 hours on a single A100.
- **Full experiment** (5 conditions x 3 description types x 2 final-generation strategies): ~100-150 GPU-hours.
