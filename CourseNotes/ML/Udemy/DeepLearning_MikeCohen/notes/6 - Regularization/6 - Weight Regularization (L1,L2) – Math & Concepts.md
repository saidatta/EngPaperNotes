## Table of Contents
1. [[Motivation and Context]]
2. [[Standard Cost Function]]
3. [[Adding the Regularization Term]]
4. [[L2 Regularization (Ridge, Weight Decay)]]
5. [[L1 Regularization (Lasso)]]
6. [[Comparison of L1 vs. L2]]
7. [[Choosing the Regularization Parameter λ]]
8. [[Why Regularization Works]]
9. [[Practical Points & Code Examples]]
10. [[Extensions: Elastic Net & Beyond]]
11. [[Key Takeaways]]

---

## 1. Motivation and Context
**Deep learning** models can easily overfit, especially when they have **many parameters** relative to the size or complexity of the dataset. One way to mitigate overfitting is to **penalize** large or unnecessary weights, guiding the model to **simpler** (less complex) representations that often **generalize** better.

### Weight Regularization Approaches
- **L2**: Penalizes the **sum of squared weights**.  
- **L1**: Penalizes the **sum of absolute values** of weights.  

Both are widely used in machine learning, statistics, optimization, and other fields (e.g., *ridge regression*, *lasso*, or *weight decay* in neural networks).

---

## 2. Standard Cost Function
In a typical deep learning setup, we have a **cost function** (also called **loss** or **objective** function) that measures how well the model’s predictions match the actual target labels. For \(m\) samples:

\[
\mathcal{J}(\mathbf{W}) 
= \frac{1}{m} \sum_{i=1}^m \ell(y^{(i)}, \hat{y}^{(i)}) \,,
\]

where
- \(\ell(\cdot)\) is the **loss** per sample (e.g., mean squared error for regression, cross entropy for classification),
- \(\hat{y}^{(i)}\) = prediction for sample \(i\),
- \(y^{(i)}\) = true label for sample \(i\),
- \(\mathbf{W}\) = all trainable parameters in the network (weights + biases).

---

## 3. Adding the Regularization Term
To **discourage** the model from simply learning large weights, we add a penalty term that depends on \(\mathbf{W}\). The new cost function becomes:

\[
\mathcal{J}_{\text{regularized}}(\mathbf{W}) 
= \frac{1}{m} \sum_{i=1}^m \ell\bigl(y^{(i)}, \hat{y}^{(i)}\bigr)
\,\,+\, \lambda \cdot \Omega(\mathbf{W}) \,,
\]

where 
- \(\Omega(\mathbf{W})\) is the **regularization function** (e.g., L1 or L2 norm), 
- \(\lambda\) is a hyperparameter controlling the **relative importance** of the regularization term.

> **Note**: If \(\lambda\) is **too large**, the model might push all weights **toward zero** (underfitting). If **too small**, it might not mitigate overfitting.

---

## 4. L2 Regularization (Ridge, Weight Decay)
Often called **ridge** in statistics or **weight decay** in neural networks. The L2 penalty is:

\[
\Omega_{L2}(\mathbf{W}) 
= \|\mathbf{W}\|_2^2 
= \sum_j W_j^2 \,.
\]

Hence the cost becomes:

\[
\mathcal{J}_{\text{L2}}(\mathbf{W}) 
= \frac{1}{m}\sum_{i=1}^m \ell(y^{(i)}, \hat{y}^{(i)})
\,\,+\, \lambda \sum_j W_j^2 \,.
\]

### Effect
- **Shrinks** all weights.  
- **Emphasizes penalizing large weights** more (since \((\alpha \cdot W_j)^2 = \alpha^2 W_j^2\)).  
- Tends to produce **more distributed** representations (weights get smaller but not necessarily zero).

**Geometric Interpretation**: A weight going from 1.5 to 1.4 offers a larger decrease in the L2 term than a weight going from 0.2 to 0.1. So **large weights** are more strongly penalized.

---

## 5. L1 Regularization (Lasso)
Called **lasso** in statistics. The L1 penalty is:

\[
\Omega_{L1}(\mathbf{W}) 
= \|\mathbf{W}\|_1 
= \sum_j |W_j| \,.
\]

Hence:

\[
\mathcal{J}_{\text{L1}}(\mathbf{W}) 
= \frac{1}{m}\sum_{i=1}^m \ell(y^{(i)}, \hat{y}^{(i)})
\,\,+\, \lambda \sum_j |W_j|.
\]

### Effect
- **Promotes sparsity**: some weights driven exactly to **zero**.  
- The penalty grows linearly with \(|W_j|\).  
- The slope is **constant** away from 0, so small weights are penalized **as much per unit** as large weights.

**Geometric Interpretation**: If you reduce a weight from 0.2 to 0.1, you gain the same penalty reduction as going from 1.5 to 1.4 (both changes = 0.1 in absolute value).

---

## 6. Comparison of L1 vs. L2

| Aspect                   | **L1** (Lasso)                   | **L2** (Ridge)                 |
|--------------------------|----------------------------------|--------------------------------|
| **Penalty Term**         | \(\sum |W_j|\)                   | \(\sum W_j^2\)                |
| **Effect on Weights**    | Creates **sparse** solutions, zeros out some weights | Shrinks **all** weights, rarely zero |
| **Interpretation**       | Useful for **feature selection** (zero weights = irrelevant features) | Useful for **distributing** weights across features |
| **Penalizes**            | Each weight equally (linear slope) | Large weights more strongly (quadratic slope) |
| **Common Names**         | Lasso regularization | Ridge, Weight Decay |

---

## 7. Choosing the Regularization Parameter λ
- **\(\lambda\)** determines how strongly you penalize large weights vs. focusing on data fit.  
- Typical approach is to **grid search** or **cross-validate** over several \(\lambda\) values and pick the one that yields the best validation metric.  
- **Heuristics**: Some defaults like `0.01` or `1e-4` for weight decay might be good starting points, depending on dataset/model size.  
- If \(\lambda\) is **too high**, you risk underfitting (weights forced to near zero, poor training performance).  
- If \(\lambda\) is **too low**, overfitting might persist.

---

## 8. Why Regularization Works

1. **Discourages Overfitting**: 
   - Large or complex weight patterns might fit noise in the training set.  
   - Weight penalties favor simpler patterns.  

2. **Improves Numerical Stability**: 
   - Extremely large weights can lead to exploding activation values.  
   - Keeping weights smaller helps control variance in outputs.

3. **Simpler Representations**:
   - L1 yields sparse weight vectors (some features or neurons are effectively dropped).  
   - L2 yields smoother, more distributed weights.

---

## 9. Practical Points & Code Examples

### 9.1. Implementing L2 (Weight Decay) in PyTorch
In **PyTorch**, L2 regularization can be **automatically** applied via an optimizer’s `weight_decay` parameter. For example:

```python
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# L2 regularization (weight_decay = lambda)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Now each step will penalize sum of squares of weights
# Great for L2
```

### 9.2. Manual Approach for L1
There is no built-in `weight_decay` for L1 in standard PyTorch optimizers. One approach is:

```python
```python
l1_lambda = 1e-4
loss_fn = nn.MSELoss()  # or CrossEntropy, etc.

for X_batch, y_batch in train_loader:
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = loss_fn(y_pred, y_batch)
    
    # Manually add L1 penalty
    l1_penalty = 0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
    
    total_loss = loss + l1_lambda * l1_penalty
    total_loss.backward()
    optimizer.step()
```

### 9.3. Combined L1 + L2 (Elastic Net)
You can add both terms to your code:

```python
```python
l1_ratio   = 1e-4
weight_dec = 1e-4
...
l1_loss = 0
for p in model.parameters():
    l1_loss += torch.sum(torch.abs(p))

# total_loss = original_loss + weight_dec * sum of squares + l1_ratio * sum of abs
total_loss = loss + weight_dec* sum_of_squares + l1_ratio * l1_loss
```

---

## 10. Extensions: Elastic Net & Beyond
- **Elastic Net**: Combines L1 + L2, often used in linear regression with the parameter \(\alpha\) controlling overall strength, and a ratio controlling L1 vs. L2.  
- **Advanced Norms**: Some tasks require specialized regularizers, e.g. Group Lasso, L0 Approximation, or domain-based constraints (like positivity constraints in medical imaging).  
- **Geometric** or **manifold** constraints can also be used in advanced settings.

---

## 11. Key Takeaways
1. **Weight Regularization**:
   - Adds a term to the cost that **penalizes** weight magnitudes.  
   - Steers the optimizer to smaller or sparser weights.
2. **L2**:
   - Summation of squared weights.  
   - Large weights get heavily penalized; fosters *distributed* solutions.
3. **L1**:
   - Summation of absolute weights.  
   - Encourages *sparsity*, zeroing out some weights.  
4. **\(\lambda\)** Tuning:
   - Cross-validation or empirical heuristics.  
   - If too big => underfitting; if too small => overfitting.  
5. **Real-World Usage**:
   - L2 often default in deep learning (weight decay).  
   - L1 helpful for *feature selection* or interpretability (zero weights).  
   - Combining with **dropout** or other forms of regularization is common.

**Conclusion**: Weight regularization (L1/L2) is a **fundamental** mechanism to control overfitting, ensure numerical stability, and often is a default strategy in *modern* neural network training. The next step is to see **practical implementations** in your code, leveraging frameworks like PyTorch.