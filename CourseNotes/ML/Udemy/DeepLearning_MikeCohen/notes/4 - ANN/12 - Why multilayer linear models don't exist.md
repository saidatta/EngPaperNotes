Below is an extensive set of Obsidian notes in Markdown format that cover the lecture “ANN: Why multilayer linear models don't exist.” These notes include theoretical explanations, a numerical example, code snippets, and visualizations. You can paste the following into your Obsidian vault as a new note.

---

# ANN: Why Multilayer Linear Models Don't Exist

**Lecture Overview:**  
In this lecture, Mike explains that if you build a neural network by stacking multiple _linear_ layers **without** any intervening non-linear activation functions (or any other form of non-linearity like pooling), the entire network is mathematically equivalent to a single linear layer. This is because the composition of linear functions remains linear.

---

## Table of Contents
1. [Key Concepts](#key-concepts)
2. [Mathematical Explanation](#mathematical-explanation)
3. [Numerical Example: Linear vs. Non-linear Operations](#numerical-example)
4. [Code Example in PyTorch](#code-example)
5. [Visualization with Mermaid](#visualization)
6. [Takeaways and Conclusion](#takeaways-and-conclusion)
7. [References](#references)

---

## Key Concepts

- **Linear Transformation:**  
  A function of the form  
  \[
  f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}
  \]
  where \(\mathbf{W}\) is a weight matrix and \(\mathbf{b}\) is a bias vector.

- **Activation Function (Non-linearity):**  
  Functions such as ReLU, sigmoid, or tanh introduce non-linearity into the network. Their role is critical because they allow the network to learn and represent complex patterns.

- **Composition of Functions:**  
  - *Linear Composition:* If you have two linear functions \(f(x)=\mathbf{A}x\) and \(g(x)=\mathbf{B}x\), their composition is:  
    \[
    g(f(x)) = \mathbf{B}(\mathbf{A}x) = (\mathbf{B}\mathbf{A})x,
    \]
    which is still a linear function.
  - *Non-linear Composition:* Non-linear functions do not have this property. For example,  
    \[
    \log(a+b) \neq \log(a) + \log(b),
    \]
    which illustrates the breakdown of distributivity in non-linear operations.

---

## Mathematical Explanation

### With Non-linear Activation
Consider a two-layer network:
1. **Layer 1:**  
   \[
   \mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1
   \]
   \[
   \mathbf{a}_1 = \sigma(\mathbf{z}_1)
   \]
   where \(\sigma\) is a non-linear activation function (e.g., ReLU, sigmoid).

2. **Layer 2:**  
   \[
   \mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2
   \]
   \[
   \mathbf{y} = \sigma(\mathbf{z}_2)
   \]

Because of the non-linear activation \(\sigma\), the overall mapping from \(\mathbf{x}\) to \(\mathbf{y}\) can approximate complex functions.

### Without Non-linear Activation
If you **omit** the activation functions (or use the identity function), the layers become purely linear:

1. **Layer 1:**  
   \[
   \mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1
   \]

2. **Layer 2:**  
   \[
   \mathbf{y} = \mathbf{W}_2 \mathbf{z}_1 + \mathbf{b}_2 = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
   \]
   \[
   \mathbf{y} = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2)
   \]

This is equivalent to a single linear transformation with effective weight matrix:
\[
\mathbf{W}_{\text{eff}} = \mathbf{W}_2 \mathbf{W}_1
\]
and effective bias:
\[
\mathbf{b}_{\text{eff}} = \mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2.
\]

Thus, stacking linear layers without non-linearity **collapses** the network into one equivalent linear layer.

---

## Numerical Example

### The Logarithm Analogy

- **Non-linear operation:**  
  Consider the logarithm function, which is non-linear:
  - Compute:  
    \[
    \log_{10}(5 + 5) = \log_{10}(10) = 1.
    \]
  - Separately applying the logarithm:  
    \[
    \log_{10}(5) + \log_{10}(5) \neq 1,
    \]
    because in general,  
    \[
    \log_{10}(a+b) \neq \log_{10}(a) + \log_{10}(b).
    \]

- **Linear operation:**  
  Now consider a linear function:
  - Let \(A\) be a constant (say, 2). Then:
    \[
    A \times (5 + 5) = A \times 10.
    \]
  - Separately,  
    \[
    A \times 5 + A \times 5 = A \times 10.
    \]
  
The **distributive property** holds for linear operations but fails for non-linear ones. This example underscores why a non-linear activation is critical between layers: it prevents the collapse of multiple layers into a single linear transformation.

---

## Code Example in PyTorch

Below is a simple PyTorch example illustrating a two-layer network **without** any non-linear activation functions.

```python
import torch
import torch.nn as nn

class LinearNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearNetwork, self).__init__()
        # Define two linear layers without non-linear activations
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # No non-linear activation is applied between the layers
        out1 = self.layer1(x)  # Linear transformation: W1 * x + b1
        out2 = self.layer2(out1)  # Linear transformation: W2 * out1 + b2
        return out2

# Sample input (batch of 1, 10 features)
x = torch.randn(1, 10)
model = LinearNetwork(input_dim=10, hidden_dim=5, output_dim=2)

# Compute the output
output = model(x)
print("Network output:", output)

# Note:
# The two linear layers are equivalent to a single linear layer with weights:
# W_eff = W2 @ W1 and bias: b_eff = W2 @ b1 + b2.
```

*Key observation:*  
Without activation functions, this network can be reduced to a single linear transformation.

---

## Visualization with Mermaid

You can add this diagram to your Obsidian note to visualize the flow:

```mermaid
flowchart LR
    A[Input X] --> B[Layer 1: Linear Transformation (W₁)]
    B --> C{Activation Function?}
    C -- Yes --> D[Non-linearity: σ]
    C -- No --> E[Layer 2: Linear Transformation (W₂)]
    D --> E
    E --> F[Output Y]
```

**Diagram Explanation:**
- **With Activation:** When a non-linear function \( \sigma \) is applied after Layer 1, the network can represent more complex functions.
- **Without Activation:** If the activation is omitted (or is simply the identity), then Layer 2 just applies another linear transformation, and the network is equivalent to one linear layer.

---

## Takeaways and Conclusion

- **Non-linearity is Essential:**  
  Without non-linear activations (or other forms of non-linear processing like pooling), multiple linear layers collapse into a single linear transformation.
  
- **Representation Power:**  
  The power of deep networks comes from stacking non-linear functions. This enables them to approximate highly complex, non-linear functions.

- **Practical Implication:**  
  Always include a non-linear activation between layers when designing deep learning architectures; otherwise, you are not increasing the model’s expressive power beyond that of a single-layer model.

---

## References

- **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
- [PyTorch Documentation](https://pytorch.org/docs/stable/nn.html#linear).
- Lecture notes and blog posts on the importance of activation functions in neural networks.

---

*End of Note*

Feel free to modify and expand these notes as you continue your studies in deep learning. Happy learning!