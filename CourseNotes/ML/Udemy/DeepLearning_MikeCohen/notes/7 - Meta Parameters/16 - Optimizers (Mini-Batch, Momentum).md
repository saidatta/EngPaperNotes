aliases: [Optimizers, SGD, Momentum, Mini-Batch]
tags: [Deep Learning, Lecture Notes, Optimizers, Meta Parameters, Neural Networks]

These notes clarify **what** an optimizer is, **why** optimizers extend basic gradient descent, and **how** mini-batch training and **momentum** smooth out the learning process. In subsequent lectures, we’ll dive into RMSProp and Adam, but for now, we focus on **Mini-Batch** Stochastic Gradient Descent (SGD) and **Momentum**.

---

## 1. Background: Gradient Descent

### 1.1 Standard Gradient Descent

- **Goal**: Find parameters \(\mathbf{w}\) that *minimize* the loss function \(\mathcal{L}(\mathbf{w})\).  
- **Algorithm**: Repeatedly update weights by **descending** along the **negative** gradient.
  
\[
\mathbf{w} \leftarrow \mathbf{w} - \eta \,\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w})
\]

- \(\eta\) = **learning rate**: scales the gradient to control step size.

### 1.2 Stochastic Gradient Descent

- In **pure** SGD, each weight update uses **one sample** (or **one** row of the data) at a time.  
- Pros:
  - Quick updates if data is **homogeneous**.  
  - Useful for streaming data.  
- Cons:
  - **Volatile** updates if data points differ significantly (outliers).  
  - Steps might **bounce around** or overshoot local minima.

---

## 2. Introduction to Optimizers

- **Optimizers** are **modifications** or **extensions** of the basic SGD formula to handle:
  1. **Mini-Batch Training**: average gradients over a small batch of samples.  
  2. **Momentum**: smooth out the path by accumulating gradients from previous steps.  
  3. (Later) **Adaptive Learning Rates** (RMSProp, Adam, etc.).

### 2.1 Why Add Modifications?

- **Vanilla SGD** can be noisy:
  - Single samples can be **unrepresentative**, causing large, erratic updates.  
  - We might see the loss function fluctuate, sometimes even going up from iteration to iteration.  
- **Mini-batches** reduce noise by averaging over \(N\) samples.  
- **Momentum** smooths the weight updates by taking into account previous gradient directions.

---

## 3. Mini-Batch SGD

### 3.1 Motivation

- Instead of updating weights after **each** sample, we:
  - Accumulate the loss/gradients over a batch of **N** samples.  
  - **Average** the gradients → single weight update.
  
\[
\nabla_{\mathbf{w}} \mathcal{L}_{\text{batch}} \;=\; \frac{1}{N} \sum_{i=1}^{N} \nabla_{\mathbf{w}} \mathcal{L}^{(i)}
\]

- This **averaging** dilutes the effect of any single outlier sample, leading to more **stable** steps.

### 3.2 Effects on Training

- **Fewer updates** per epoch (since each update uses a chunk of data).  
- Often **faster** convergence than pure SGD if samples are diverse.  
- But if samples are **already very similar**, pure SGD might converge faster.

#### Example Visualization
If you have a large outlier:
- **SGD** sees the outlier alone, possibly causing a huge step.  
- **Mini-Batch** sees that outlier’s gradient **averaged** with other points, reducing the shock.

---

## 4. Momentum

### 4.1 Concept

- **Momentum** adds a “velocity” term:
  \[
  \mathbf{v}_t \;=\; \beta\, \mathbf{v}_{t-1} \;+\; (1-\beta)\,\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}_t)
  \]
  - \(\beta \in [0,1)\) is the momentum coefficient.  
  - Sometimes equivalently written as:
    \[
    \mathbf{v}_t \;=\; \beta\, \mathbf{v}_{t-1} \;+\; \nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}_t)
    \]
    and factor out learning rate separately.
- Then the **weight update** is:
  \[
  \mathbf{w} \;\leftarrow\; \mathbf{w} \;-\; \eta \,\mathbf{v}_t
  \]
- Interpretation:
  - Current gradient is **smoothed** by previous update directions.  
  - Larger \(\beta\) (e.g., 0.9 to 0.99) → stronger “inertia” from past steps.

### 4.2 Why Momentum?

- **Analogy**: A ball rolling downhill **gains** momentum, so small bumps won’t redirect it completely.  
- **In practice**:
  - Reduces oscillations (especially in ravines, or directions of steep curvature).  
  - Helps the optimizer move **consistently** in a beneficial direction even if individual batches differ slightly.

### 4.3 Visualization

Imagine a 2D loss surface:
- **SGD**: might bounce around from one mini-batch to the next.  
- **SGD+Momentum**: path is more **direct** (accumulates velocity), less random zig-zag.

---

## 5. Putting It All Together

1. **Choose** batch size (mini-batch).  
2. **Update** weights each iteration using the **average** gradient.  
3. **Momentum** accumulates velocity so that updates consider current + past gradients:
   \[
   \mathbf{v}_t \;=\; \beta\,\mathbf{v}_{t-1} \;+\; (1-\beta)\,\nabla_{\mathbf{w}}\mathcal{L}
   \]
4. **Weight step**:
   \[
   \mathbf{w} \;\leftarrow\; \mathbf{w} \;-\; \eta \,\mathbf{v}_t
   \]

**Result**: Smoother descent, potentially faster convergence, less “bouncing”.

---

## 6. Example PyTorch Usage

In PyTorch, you pick an **optimizer** (e.g., `torch.optim.SGD`), specifying:
- **`params`** = model parameters (e.g. `model.parameters()`)  
- **`lr`** = learning rate  
- **`momentum`** = scalar in [0, 1)

**Code**:

```python
import torch.optim as optim

model = MyModel()
criterion = nn.MSELoss()

# standard SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
```

- `momentum=0.9` → means \(\beta=0.9\).  
- If you want **vanilla** SGD, just set `momentum=0.0`.

---

## 7. Key Takeaways

1. **Optimizers** are **extensions** of vanilla SGD.  
2. **Mini-Batch**:
   - Averages gradients over a chunk of data → smoother updates.  
   - Typically the **default** approach in modern deep learning.  
3. **Momentum**:
   - Accumulates velocity to reduce random fluctuations.  
   - Often used with momentum ~ `0.9` or `0.99`.

**Next**: We’ll see **RMSProp** and **Adam**, which adapt the learning rate per parameter dimension.

---

## 8. Additional Resources

- [Deep Learning Book, Goodfellow et al.](https://www.deeplearningbook.org/) – *Chapter on Optimization*.  
- [Stanford CS231n: Optimization notes](http://cs231n.github.io/neural-networks-3/#sgd)  
- [PyTorch docs on SGD](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)

```
Created by: [Your Name / Lab / Date]
Lecture Reference: “Meta parameters: Optimizers (minibatch, momentum)”
```
```