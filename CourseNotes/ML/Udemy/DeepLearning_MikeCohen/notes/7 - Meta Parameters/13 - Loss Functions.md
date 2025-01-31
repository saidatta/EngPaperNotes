aliases: [Loss Functions, MSE, Cross Entropy, Log-Softmax, Sigmoid vs. Softmax]
tags: [Deep Learning, Lecture Notes, Meta Parameters, Neural Networks]

**Loss functions** are at the heart of deep learning—guiding how we adjust model parameters to minimize **prediction errors**. This note:

1. Reviews **why** we need loss functions.  
2. Covers **common losses**: Mean Squared Error (MSE) and Cross-Entropy (binary & multi-class).  
3. Discusses **Sigmoid vs. Softmax** connections.  
4. Introduces **Log-Softmax** and why it’s often used in practice.

---
## 1. Why We Need Loss Functions

### 1.1 Forward Pass & Error Computation

1. **Forward Pass**:  
   - A network performs a series of **linear** transformations (weighted sums) followed by **non-linear** activations.  
   - Produces a final **prediction** \(\hat{y}\).  
2. **Compare** \(\hat{y}\) with the real target \(y\). The **difference** is measured by a **loss** or **cost** function.  
3. **Backward Pass**:  
   - Compute gradients \(\nabla_\mathbf{w} \mathcal{L}\) of the loss \(\mathcal{L}\) w.r.t. model parameters \(\mathbf{w}\).  
   - Update weights: \(\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_\mathbf{w}\mathcal{L}\).

### 1.2 Types of Tasks & Corresponding Losses

- **Regression** → Typically **MSE** (Mean Squared Error) or **MAE** (Mean Absolute Error).  
- **Classification** → Typically **Cross-Entropy** (binary or categorical).  

---

## 2. Mean Squared Error (MSE)

### 2.1 Formula

\[
\mathcal{L}_\text{MSE} = \frac{1}{N} \sum_{i=1}^N \Bigl(\hat{y}^{(i)} - y^{(i)}\Bigr)^2
\]

- Suited for **continuous** output (e.g., house price, temperature).  
- **Output layer**: often **linear** (no final activation).  
- In PyTorch, used via `nn.MSELoss()`.

---

## 3. Cross-Entropy (CE)

### 3.1 Binary Cross-Entropy

\[
\mathcal{L}_{\text{BCE}}(y, \hat{y}) 
= -\Bigl[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \Bigr]
\]

- **Binary classification**: \(y\in \{0,1\}\).  
- **Output layer**: typically **Sigmoid** for \(\hat{y}\in (0,1)\).  
- If the prediction \(\hat{y}\approx y\), loss is **low**; if \(\hat{y}\) is far from \(y\), loss is **high**.

#### Visualization
```
y=0 --> Loss is high as y_hat -> 1
y=1 --> Loss is high as y_hat -> 0
```

### 3.2 Categorical Cross-Entropy

For **multi-class** classification with \(C\) classes:

\[
\mathcal{L}_{\text{CE}}( \mathbf{y}, \hat{\mathbf{y}} )
= -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
\]

- **One-hot** labels: \( \mathbf{y} \) has 1 in one position, 0 in others.  
- **Output layer**: typically **Softmax** so that \(\sum_c \hat{y}_c = 1\).  
- In PyTorch, used via `nn.CrossEntropyLoss()` (which combines Softmax + CE).

---

## 4. Kullback–Leibler (KL) Divergence

\[
D_{\text{KL}}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
\]

- Measures “distance” between **two distributions** \(P\) and \(Q\).  
- Used in specialized contexts (e.g., **variational autoencoders**).  
- Not as common for direct classification/regression tasks.

---

## 5. Designing the Output Layer

### 5.1 Regression

- **Output Units**: 1 (or more if multi-output).  
- **Activation**: *Linear* (or identity).  
- **Loss**: MSE (or similar).  

### 5.2 Binary Classification

- **Output Units**: 1  
- **Activation**: Sigmoid (\(\sigma\))  
- **Loss**: Binary Cross-Entropy (`nn.BCEWithLogitsLoss` in PyTorch).

### 5.3 Multi-class Classification

- **Output Units**: \(C\) (one per class).  
- **Activation**: Softmax → \(\hat{y}_c = \frac{e^{z_c}}{\sum_j e^{z_j}}\).  
- **Loss**: Categorical Cross-Entropy (PyTorch `nn.CrossEntropyLoss`).

---

## 6. Sigmoid vs. Softmax

### 6.1 Softmax for 2 Classes = Sigmoid

For two classes (e.g., cat vs. dog), **Softmax** reduces to:

\[
\hat{y}_1 = \frac{e^{a}}{e^a + e^b}, 
\quad
\hat{y}_2 = \frac{e^{b}}{e^a + e^b}
\]
If we define \(z = a - b\), we get:
\[
\hat{y}_1 = \frac{1}{1 + e^{-z}},
\quad
\hat{y}_2 = 1 - \hat{y}_1
\]
Which is essentially **Sigmoid**.

### 6.2 Why Separate Sigmoid & Softmax?

- **Sigmoid** is simpler for the 2-class scenario.  
- **Softmax** ensures **all outputs** sum to 1 (\(\sum_c \hat{y}_c = 1\)) for **multi-class**.  
- Sigmoid on each class individually does **not** produce a valid probability distribution across multiple classes unless exactly 2 classes exist.

---

## 7. Log-Softmax

### 7.1 Definition

**Log-Softmax** applies the log function to each softmax probability:

\[
\log \Bigl(\frac{e^{z_c}}{\sum_j e^{z_j}}\Bigr)
= z_c - \log \Bigl(\sum_j e^{z_j}\Bigr)
\]

### 7.2 Advantages

- Improves numerical **stability** (logarithms of tiny probabilities can be handled better).  
- Amplifies the penalty for **confident but wrong** predictions.  
- Often used internally by PyTorch’s `nn.CrossEntropyLoss` for numerical efficiency.

**Practical Note**: In PyTorch, you might see:
- `nn.LogSoftmax(dim=1)` for the activation.  
- Then `nn.NLLLoss()` for training.  
- Or you can directly call `nn.CrossEntropyLoss()`, which effectively does **Log-Softmax** + **NLL (negative log-likelihood)** internally.

---

## 8. Summary

1. **MSE** for **Regression**:
   - Output layer: **linear**.  
   - Loss function: **mean squared error**.  

2. **Cross-Entropy** for **Classification**:
   - **Binary**:
     - Output layer: 1 unit + **Sigmoid** → BCE loss.  
   - **Multi-class**:
     - Output layer: \(C\) units + **Softmax** → CE loss.  

3. **Sigmoid** = **Softmax** (for 2 classes).  
4. **Log-Softmax**: A numerically stable variant, common in frameworks like PyTorch.

**Takeaways**:
- Choice of **loss function** is tightly coupled to the **output layer** architecture and data type (continuous vs. categorical).  
- For multi-class problems, **Softmax** (or Log-Softmax) is the standard approach.  
- For advanced scenarios (e.g., distributions over outputs), consider other losses like **KL divergence**.

---

## 9. Code Snippets (PyTorch Examples)

### 9.1 MSE for Regression

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 1)  # no activation for regression
)

criterion = nn.MSELoss()  # or L1Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X = torch.randn(32, 10)
y = torch.randn(32, 1)

optimizer.zero_grad()
pred = model(X)
loss = criterion(pred, y)
loss.backward()
optimizer.step()

print("Regression MSE Loss:", loss.item())
```

### 9.2 Cross-Entropy for Multi-class

```python
import torch.nn.functional as F

model = nn.Sequential(
    nn.Linear(10, 5),  # 5 classes
    # no Softmax needed here if we use CrossEntropyLoss
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X = torch.randn(32, 10)
y = torch.randint(0, 5, (32,))  # 0..4 labels

optimizer.zero_grad()
logits = model(X)   # shape [32,5]
loss = criterion(logits, y)
loss.backward()
optimizer.step()

print("Categorical CE Loss:", loss.item())
```

*(`CrossEntropyLoss` in PyTorch applies **log-softmax** internally.)*

---

## 10. References

1. **Deep Learning Book** (Goodfellow, Bengio, Courville) – Chapter 6 (Sections on losses).  
2. **PyTorch Documentation**:
   - [`nn.MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)  
   - [`nn.BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)  
   - [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) (combines `LogSoftmax` + `NLLLoss`).  
3. [Log-Softmax vs. Softmax Discussion](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html)  

```
Created by: [Your Name / Lab / Date]
Lecture Reference: “Meta parameters: Loss functions”
```
```