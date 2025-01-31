aliases: [CodeChallenge: Adam + L2, Weight Decay, Regularization, Optimizer Integration]
tags: [Deep Learning, Lecture Notes, Meta Parameters, Optimizers, Regularization]

This challenge combines **two** concepts from previous lessons:

1. **Adam Optimizer** from the [Optimizers section](#)  
2. **L2 Regularization** (weight decay) from the [Regularization section](#)

We aim to see how different **L2 penalty strengths** (\(\lambda\)) affect **training** and **test** performance on the **3-class Qwerty** dataset using **Adam** with a fixed learning rate.

---
## 1. Overview

- **Dataset**: Same Qwerty 3-class data.  
- **Model**: A small MLP (2 \(\to\) 8 \(\to\) 3).  
- **Optimizer**: **Adam** at a **fixed** learning rate `lr=1e-3`.  
- **Mini-Batch**: 32.  
- **Key Parameter**: `weight_decay` in the Adam optimizer – a.k.a **L2 regularization** factor (\(\lambda\)).

We will vary \(\lambda \in [0, 0.1]\) (e.g., 6 values) and examine **accuracy curves**.

---

## 2. Setup: Data & Model

**Same** Qwerty dataset generation as in previous code. We jump straight to:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# Qwerty data generation
np.random.seed(42)
n_per_class = 300
means = np.array([[ 2,  1],
                  [-2,  1],
                  [ 0, -3]])
cov   = np.array([[1,0],[0,1]])

X = np.zeros((n_per_class*3, 2), dtype=np.float32)
y = np.zeros(n_per_class*3,       dtype=np.int64)
for i in range(3):
    idx = range(i*n_per_class, (i+1)*n_per_class)
    X[idx,:] = np.random.multivariate_normal(means[i], cov, n_per_class)
    y[idx]   = i

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_t, y_t, test_size=0.2, shuffle=True, random_state=42
)

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)
```

### 2.1 Model Definition

```python
class QwertyNet(nn.Module):
    def __init__(self):
        super(QwertyNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

## 3. Creating the Model + Adam with L2

We add a parameter `l2_lambda` for L2 regularization:

```python
def create_model(l2_lambda=0.0):
    """
    Returns (model, loss_fn, optimizer) using Adam with a fixed lr=1e-3
    and weight_decay = l2_lambda.
    """
    model = QwertyNet()
    loss_fn = nn.CrossEntropyLoss()
    
    # Adam optimizer with weight_decay as L2
    optimizer = optim.Adam(
        model.parameters(), 
        lr=1e-3, 
        weight_decay=l2_lambda
    )
    return model, loss_fn, optimizer
```

---

## 4. Training Function

Identical structure to prior code but includes an argument `l2_lambda`:

```python
def train_model(l2_lambda=0.0, epochs=50):
    model, loss_fn, optimizer = create_model(l2_lambda=l2_lambda)
    
    train_loss_hist = []
    test_loss_hist  = []
    train_acc_hist  = []
    test_acc_hist   = []
    
    for ep in range(epochs):
        # TRAIN
        model.train()
        total_loss = 0
        correct, total = 0, 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total   += len(yb)
        
        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        
        # TEST
        model.eval()
        t_loss = 0
        c_test, tot_test = 0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                logits_test = model(Xb)
                loss_test   = loss_fn(logits_test, yb)
                t_loss     += loss_test.item()
                
                preds_test = torch.argmax(logits_test, dim=1)
                c_test     += (preds_test == yb).sum().item()
                tot_test   += len(yb)
        
        test_loss = t_loss / len(test_loader)
        test_acc  = c_test / tot_test
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
    
    return {
        "train_loss": train_loss_hist,
        "test_loss":  test_loss_hist,
        "train_acc":  train_acc_hist,
        "test_acc":   test_acc_hist,
        "model": model
    }
```

---

## 5. The Experiment: Vary L2 from 0.0 up to 0.1

We pick a range, e.g. 6 steps from 0.0 to 0.1 in increments of 0.02:

```python
l2_values = np.arange(0, 0.12, 0.02)  # 0.00, 0.02, 0.04, 0.06, 0.08, 0.10
results_dict = {}

for lam in l2_values:
    print(f"Training with L2={lam:.2f}")
    outcome = train_model(l2_lambda=lam, epochs=50)
    results_dict[lam] = outcome
    print(f"  Final Test Acc: {outcome['test_acc'][-1]*100:.2f}%\n")
```

---

## 6. Visualization

We can plot the training/test accuracy curves for each L2 setting:

```python
plt.figure(figsize=(12,4))

colors = sns.color_palette("husl", len(l2_values))

for i, lam in enumerate(l2_values):
    # Plot only test accuracies (or both)
    test_acc = results_dict[lam]["test_acc"]
    plt.plot(test_acc, label=f"L2={lam:.2f}", color=colors[i])

plt.title("Test Accuracy with Adam + L2 Regularization")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

Similarly, we can plot **train** accuracy or **loss** if desired.

---

## 7. Observations

- If **L2** is **0.0** (no reg), we typically get maximum or near-max accuracy (the model can freely overfit or fit well).  
- Small L2 can act as a mild **regularizer**, sometimes improving generalization if the model risked overfitting.  
- Larger L2 (like 0.1) might hamper the network from fully learning, resulting in slower/final lower accuracy.  
- Adam is already quite capable, so you might see **little** or **no** improvement from L2 on this easy dataset.

### Example Patterns

1. For some **small** \(\lambda \approx 0.01\), you might see **slightly** improved test accuracy or faster stable convergence.  
2. For **larger** \(\lambda\), model might underfit, failing to reach the same final accuracy.

---

## 8. Conclusion

In this code challenge, we merged **Adam** (with a set `lr=1e-3`) and **L2 regularization** by varying **weight_decay**:

- **Code**: `optim.Adam(..., weight_decay=l2_lambda)`.  
- **Result**: Evaluate how each \(\lambda\) affects training/test accuracy on the Qwerty dataset.  
- **Key Insight**: Adding complexity (like L2) doesn’t always help if the model/data is already easy. Over-regularization can reduce performance.

**Takeaway**: 
- Combining advanced optimizers (Adam) with regularization can be powerful, especially in larger models/datasets prone to overfitting.  
- On simpler tasks, moderate or zero L2 might suffice.

---

## 9. Possible Extensions

1. **Plot** train accuracy vs. \(\lambda\) or final test accuracy vs. \(\lambda\).  
2. Try additional regularizers (e.g., **Dropout**) or more complex architectures.  
3. Explore if a different **learning rate** changes how much \(\lambda\) matters.

---

## 10. References

- [**Deep Learning Book** – Goodfellow et al. Chapters on **Regularization** and **Optimization**.](https://www.deeplearningbook.org/)  
- [PyTorch Docs on `Adam` and `weight_decay`](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)  

```
Created by: [Your Name / Lab / Date]
Lecture Reference: “Meta parameters: CodeChallenge: Adam with L2 regularization”
```
```