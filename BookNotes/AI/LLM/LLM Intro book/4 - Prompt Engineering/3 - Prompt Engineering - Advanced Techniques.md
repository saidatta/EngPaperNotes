In this note, we delve into advanced prompt engineering techniques to enhance the performance of Large Language Models (LLMs). Specifically, we focus on:
- Overcoming **recency bias** by repeating instructions at the end of prompts.
- Utilizing **delimiters** to structure prompts effectively.
These strategies help in guiding LLMs to produce more accurate and consistent outputs, which is crucial for developing robust LLM-powered applications.

---
## Table of Contents
1. [Understanding Recency Bias](#understanding-recency-bias)
2. [Overcoming Recency Bias by Repeating Instructions](#overcoming-recency-bias-by-repeating-instructions)
   - [Example: Sentiment Analysis](#example-sentiment-analysis)
3. [Using Delimiters in Prompts](#using-delimiters-in-prompts)
   - [Benefits of Delimiters](#benefits-of-delimiters)
   - [Example: Code Generation](#example-code-generation)
4. [Conclusion](#conclusion)
5. [References](#references)
---
## Understanding Recency Bias
**Recency bias** is the tendency of LLMs to give more weight to information presented at the end of a prompt, potentially ignoring or forgetting earlier instructions. This can lead to:
- **Inaccurate responses**: The model may not consider the full context.
- **Inconsistent outputs**: Instructions at the beginning may be overlooked.
### Definition

> **Recency Bias**: In the context of LLMs, it refers to the model's propensity to prioritize information near the end of a prompt over earlier content.

**Mathematical Perspective**:

LLMs process input tokens sequentially, updating their hidden state at each step. Given the sequential nature and the attention mechanisms focusing on recent tokens, the model may disproportionately focus on later tokens.

---

## Overcoming Recency Bias by Repeating Instructions

One effective strategy to mitigate recency bias is **repeating key instructions at the end of the prompt**. This reinforces important information, ensuring the model considers it when generating a response.

### Example: Sentiment Analysis

**Objective**: Analyze the sentiment of a conversation and output only the sentiment in lowercase without punctuation.

**Initial Prompt**:

```python
system_message = """
You are a sentiment analyzer. You classify conversations into three categories: positive, negative, or neutral.
Return only the sentiment, in lowercase and without punctuation.
Conversation:
"""
```

**Conversation**:

```python
conversation = """
Customer: Hi, I need some help with my order.
AI agent: Hello, welcome to our online store. I'm an AI agent and I'm here to assist you.
Customer: I ordered a pair of shoes yesterday, but I haven't received a confirmation email yet. Can you check the status of my order?
[...]
"""
```

**Model Invocation**:

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": conversation},
    ]
)

print(response['choices'][0]['message']['content'])
```

**Output**:

```
Neutral
```

**Issue**: The output is capitalized, contrary to the instruction.

### Enhancing the Prompt

To address this, we repeat the key instruction at the end:

```python
system_message = f"""
You are a sentiment analyzer. You classify conversations into three categories: positive, negative, or neutral.
Return only the sentiment, in lowercase and without punctuation.
Conversation:
{conversation}
Remember to return only the sentiment, in lowercase and without punctuation.
"""
```

**Updated Model Invocation**:

```python
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": system_message},
    ]
)

print(response['choices'][0]['message']['content'])
```

**Output**:

```
neutral
```

**Explanation**:

By repeating the instruction at the end, the model adheres to the requirement of outputting the sentiment in lowercase without punctuation.

**Key Takeaways**:

- Reiterating critical instructions at the end of the prompt helps counteract recency bias.
- This technique ensures that essential details are fresh in the model's context during response generation.

---

## Using Delimiters in Prompts

**Delimiters** are sequences of characters or symbols used to clearly separate different sections within a prompt. They help structure the prompt, making it easier for the model to parse and understand.

### Common Delimiters

- `>>>`
- `===`
- `---`
- `####`
- ````

### Benefits of Delimiters

1. **Clear Separation**: Distinctly marks different sections (e.g., instructions, examples, user input).
2. **Guidance for LLMs**: Reduces ambiguity, helping the model interpret the prompt correctly.
3. **Enhanced Precision**: Leads to more relevant and accurate responses.
4. **Improved Coherence**: Organizes the prompt logically, improving the overall quality of the output.

### Example: Code Generation

**Objective**: Instruct the model to generate Python code based on the user's request, using an example for one-shot learning.

**Prompt with Delimiters**:

```python
system_message = """
You are a Python expert who produces Python code as per the user's request.
===>START EXAMPLE
---User Query---
Give me a function to print a string of text.
---User Output---
Below you can find the described function:
```python
def my_print(text):
    print(text)
```
<===END EXAMPLE
"""
```

**User Query**:

```python
query = "Generate a Python function to calculate the nth Fibonacci number."
```

**Model Invocation**:

```python
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]
)

print(response['choices'][0]['message']['content'])
```

**Output**:

```
Below you can find the described function:
```python
def fibonacci(n):
    if n <= 0:
        return None
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```
```

**Explanation**:

- The model follows the format provided in the example.
- It uses delimiters to encapsulate the code snippet, maintaining consistency.

### Mathematical Explanation: Fibonacci Function

The Fibonacci sequence is defined recursively:

\[
F(n) = \begin{cases}
0 & \text{if } n = 1 \\
1 & \text{if } n = 2 \\
F(n - 1) + F(n - 2) & \text{if } n > 2
\end{cases}
\]

**Code Explanation**:

- **Base Cases**:
  - If `n <= 0`: Return `None` (invalid input).
  - If `n == 1`: Return `0`.
  - If `n == 2`: Return `1`.
- **Recursive Case**:
  - Return the sum of `fibonacci(n-1)` and `fibonacci(n-2)`.

---

## Conclusion

- **Repeating Instructions**: Reinforces critical information, mitigating recency bias in LLMs.
- **Using Delimiters**: Structures prompts effectively, guiding LLMs to produce accurate and coherent responses.
- These techniques enhance the reliability and performance of LLM-powered applications, especially in complex tasks.

---

## References

1. **John Stewart's Blog Post**: [Large Language Model Prompt Engineering for Complex Summarization](https://devblogs.microsoft.com/ise/gpt-summary-prompt-engineering/)
2. **OpenAI API Documentation**: [OpenAI ChatCompletion API](https://platform.openai.com/docs/api-reference/chat/create)
3. **Fibonacci Sequence**: [Wikipedia - Fibonacci Number](https://en.wikipedia.org/wiki/Fibonacci_number)

---

*Note: This document is intended for Staff+ engineers seeking a deep understanding of advanced prompt engineering techniques, including practical code examples and mathematical explanations.*