aliases: [Activation Functions, Non-linearities, ReLU, Sigmoid, Tanh]
tags: [Deep Learning, Lecture Notes, Neural Networks, Meta Parameters]

**Activation functions** introduce **non-linearity** into neural networks, enabling them to learn and represent complex relationships. This note delves into:
1. **Why** non-linear activations are necessary
2. **Key properties** of commonly used activation functions (Sigmoid, Tanh, ReLU variants)
3. **When** to use each function
4. **Practical tips** for choosing or experimenting with activation functions

---

## 1. Why We Need Non-Linear Activations

### 1.1 Linear vs. Non-Linear Layers

- A **linear layer** computes \(\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}\).  
- Stacking multiple **linear** layers without any non-linearity is effectively **equivalent** to a **single** linear layer:  
  \[
    \mathbf{W_2}(\mathbf{W_1} \mathbf{x} + \mathbf{b_1}) + \mathbf{b_2} 
    = (\mathbf{W_2}\mathbf{W_1})\mathbf{x} + (\mathbf{W_2}\mathbf{b_1} + \mathbf{b_2})
  \]
- **Non-linear** activations **break** this equivalence, allowing deep networks to learn **complex, hierarchical** features.

### 1.2 Desired Properties for Hidden-Layer Activations

1. **Non-Linear**: Must not be purely linear.  
2. **Computationally Simple**: Because activations run **millions or billions** of times, speed is critical.  
3. **Avoid Overly Saturated Ranges** (if possible): Large saturation leads to **vanishing gradients**.  
4. **Stable Gradients**: Should minimize risk of gradient exploding or vanishing.

### 1.3 Desired Properties for Output-Layer Activations

- **Depends on the task**:
  - **Regression** (\(\mathbb{R}\)): Often **linear** (no activation) or unbounded.  
  - **Binary Classification** (\([0,1]\)): **Sigmoid**—maps to probability.  
  - **Multi-class Classification** (\(\text{Softmax}\)): Output vector summing to 1.  
- Typically, the output activation is chosen based on how we interpret the **final layer** output (probabilities, real-valued predictions, etc.).

---

## 2. Common Activation Functions

Below we review three major families of activation functions—**Sigmoid**, **Tanh**, and **ReLU** (plus variants).

### 2.1 Sigmoid

\[
\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
\]

- **Range**: (0, 1)  
- **Interpretation**: Can be seen as a probability if you interpret outputs \(\ge 0.5\) as “positive class.”  
- **Pros**:
  - Smooth, well-defined between 0 and 1, often used in **output layers** for binary classification.  
- **Cons**:
  - **Saturates** near 0 or 1 → **vanishing gradient** problem for large \(|x|\).  
  - Centered around 0.5 (bias shift).  
  - Almost **linear** for a small range around 0, offering less useful non-linearity in hidden layers.

**Where to use**:
- **Output** for **binary classification** (with `BCELoss` or `BCEWithLogitsLoss` in PyTorch).

### 2.2 Hyperbolic Tangent (Tanh)

\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

- **Range**: \((-1, 1)\)  
- **Interpretation**: Symmetric around 0—values close to 0.0 if input is near 0.  
- **Pros**:
  - Often used historically in hidden layers because it is **zero-centered**.  
- **Cons**:
  - Still **saturates** for large \(|x|\) → potential **vanishing gradients**.  
  - Over certain ranges, \(\tanh\) is almost linear, offering less “strong” non-linearity.

**Where to use**:
- Historically in **hidden layers**, but less common now.

### 2.3 ReLU (Rectified Linear Unit)

\[
\text{ReLU}(x) = \max(0, x)
\]

- **Range**: \([0, \infty)\)  
- **Piecewise linear**:
  - 0 for \(x < 0\)  
  - Identity for \(x \ge 0\)  
- **Pros**:
  - **Very cheap** to compute.  
  - Does **not** saturate for positive \(x\).  
  - Empirically leads to **fast** training in deeper networks.  
- **Cons**:
  - Negative inputs become **0** → “dead ReLU” if weights drive outputs < 0 often.  
  - Not truly differentiable at \(x=0\), but in practice not a real issue.  
  - Possible **exploding gradients** if large positive values are not otherwise constrained.

**Where to use**:
- **Most common** choice for hidden layers in modern deep networks.

---

## 3. ReLU Variants

### 3.1 Leaky ReLU

\[
\text{LeakyReLU}(x) = 
\begin{cases}
x, & \text{if } x \ge 0\\
\alpha x, & \text{if } x < 0
\end{cases}
\]

- Typically \(\alpha\) is small (e.g., 0.01 or 0.1).
- **Advantage**: Doesn’t zero-out negative values entirely → can mitigate “dead ReLU” issue.
- **Disadvantage**: Introduces an extra hyperparameter \(\alpha\); performance gains can be data-dependent.

### 3.2 Capped ReLU (ReLU-n)

\[
\text{ReLU-n}(x) = \min(\max(0, x), n)
\]
- Clamps output to a maximum \(n\) (commonly 6 in some CNN architectures).
- Avoids extremely large outputs → can help control exploding gradients.

**No universal winner** among these ReLU variants—**empirical testing** is recommended.

---

## 4. Summary of Properties

| **Function**   | **Formula**                          | **Range**    | **Pros**                                     | **Cons**                                        | **Typical Use**                  |
|----------------|--------------------------------------|-------------|---------------------------------------------|-------------------------------------------------|----------------------------------|
| **Sigmoid**    | \(1 / (1 + e^{-x})\)                 | (0,1)       | Easy probability interpretation             | Saturates at 0 or 1 → vanishing gradients       | Output for **binary class**      |
| **Tanh**       | \((e^x - e^{-x})/(e^x + e^{-x})\)     | (-1,1)      | Zero-centered, historically popular         | Saturates for large \(|x|\)                    | Occasionally hidden layers, RNNs |
| **ReLU**       | \(\max(0,x)\)                        | [0,∞)       | Simple, fast, non-saturating for x>0        | No negative slope → dead neurons possible       | **Default** in hidden layers     |
| **Leaky ReLU** | \(\max(\alpha x, x)\)                | (-∞,∞)      | Allows negative slope, helps “dead ReLU”    | Extra hyperparam \(\alpha\)                    | Alternate hidden layer function  |
| **ReLU-n**     | \(\min(\max(0, x), n)\)              | [0, n]      | Limits exploding outputs                    | Capping might hamper large signals if needed    | Less common, but used in some CNNs |

---

## 5. Practical Guidelines

1. **Default Choice**: 
   - **ReLU** for hidden layers; **Sigmoid** or **Softmax** for output layers in classification tasks.  
2. **Experiment with Variants**:
   - Leaky ReLU, Parametric ReLU, or Swish/GELU (modern alternatives) if ReLU yields poor performance or many “dead” neurons.  
3. **Watch for Saturation**:
   - If using **Sigmoid**/**Tanh** in hidden layers, monitor training for potential **vanishing gradients**.  
4. **Output Layer**:
   - **Sigmoid** for binary, **Softmax** for multi-class, or no activation (i.e., linear) for regression.

---

## 6. Code Examples

### 6.1 Using Built-In Activations in PyTorch

PyTorch provides both **functional** and **module** forms:
- **Functional**: `torch.nn.functional.relu(x)`, `torch.sigmoid(x)`, `torch.tanh(x)`, etc.
- **Module**: `nn.ReLU()`, `nn.Sigmoid()`, `nn.Tanh()`, `nn.LeakyReLU(negative_slope=0.1)`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)
        
        # We can define a leaky ReLU module if we want
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x):
        # Option 1: Use functional relu
        x = F.relu(self.fc1(x))  
        
        # Option 2: Use module-based LeakyReLU
        x = self.leaky_relu(x)
        
        # Output layer can be linear (e.g., for regression) 
        # or Sigmoid for binary classification
        x = self.fc2(x)
        return x
```

### 6.2 Comparing Different Activations Quickly

```python
activations = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "LeakyReLU": nn.LeakyReLU(0.1)
}

x_vals = torch.linspace(-3, 3, steps=100)
for name, act in activations.items():
    y = act(x_vals)
    plt.plot(x_vals.numpy(), y.detach().numpy(), label=name)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.ylim([-1.2, 3.2])
plt.legend()
plt.title("Activation Functions")
plt.show()
```

*(You’ll see the different shapes and ranges.)*

---

## 7. Other Activation Functions & Current Research

- **Swish** (\(x \cdot \sigma(x)\))  
- **GELU** (Gaussian Error Linear Unit)  
- **ELU**, **SELU**, etc.  

Some are specialized, and others aim for better gradient properties. **Most** mainstream networks still rely on **ReLU** variants due to simplicity and robust results.

---

## 8. Final Thoughts

- **Choice of activation** is a **meta-parameter**.  
- Start with **ReLU** in hidden layers and the **appropriate** activation for your output layer (Sigmoid, Softmax, linear, etc.).  
- If performance or convergence issues arise, **experiment** with alternative non-linearities.  
- Keep an eye on **vanishing** or **exploding** gradients, and consider **batch normalization** or other techniques to help.

---

## 9. References

1. **Deep Learning** – Goodfellow, Bengio, Courville (Ch. 6 – Activation Functions)  
2. **Rectified Linear Units**: [Nair & Hinton, 2010](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)  
3. **Leaky ReLU**: [Maas et al., 2013, “Rectifier Nonlinearities Improve Neural Networks”](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)  
4. **Swish**: [Ramachandran et al., 2017, *Searching for Activation Functions*](https://arxiv.org/abs/1710.05941)

---

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: “Meta parameters: Activation Functions”  
```