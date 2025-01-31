## Table of Contents
1. [[Overview & Motivation]]
2. [[Dataset: The "Qwerties’ Cousins" (Non-linear Problem)]]
3. [[Data Splitting & Loaders]]
4. [[Model Definition & Dropout Integration]]
5. [[Training & Code Snippets]]
6. [[Experiment: Varying Dropout Rates]]
7. [[Visualizing & Smoothing Accuracy]]
8. [[Key Conclusions]]

---
## 1. Overview & Motivation
While dropout is a powerful regularization technique, **it does not always** guarantee performance gains. This lecture demonstrates a **hands-on** approach:
- We create a **non-linear** dataset where features \((x, y)\) do not allow perfect linear separation.
- Implement a **deep feedforward** model with **dropout** in PyTorch.
- Run a **parametric experiment** to see how different dropout rates \( p \in [0,1]\) affect both **training** and **test** accuracy.

---
## 2. Dataset: The "Qwerties’ Cousins" (Non-linear Problem)

```python
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic, non-linear data
N = 2000
thetas = np.linspace(0, 4*np.pi, N)  # angles
# e.g., 2 classes centered around different radii
r1 = 10 + 0.5 * np.random.randn(N//2)
r2 = 15 + 0.5 * np.random.randn(N//2)

# Class 0 (smaller radius)
x0 = r1*np.cos(thetas[:N//2]) + np.random.randn(N//2)*0.5
y0 = r1*np.sin(thetas[:N//2]) + np.random.randn(N//2)*0.5

# Class 1 (larger radius)
x1 = r2*np.cos(thetas[N//2:]) + np.random.randn(N//2)*0.5
y1 = r2*np.sin(thetas[N//2:]) + np.random.randn(N//2)*0.5

# Stack data
X0 = np.vstack((x0, y0)).T
X1 = np.vstack((x1, y1)).T
X = np.concatenate((X0, X1), axis=0)
Y = np.concatenate((np.zeros(len(x0)), np.ones(len(x1))))

# Quick visualization
plt.figure(figsize=(5,5))
plt.scatter(X0[:,0], X0[:,1], c='blue', label='Class 0')
plt.scatter(X1[:,0], X1[:,1], c='black', label='Class 1')
plt.title("Qwerties’ Cousins Dataset")
plt.legend()
plt.show()
```

**Key Features**:
- Two radial “bands” of points, each with some noise.  
- Non-linear separation is required to discriminate the classes.

---

## 3. Data Splitting & Loaders
We use **scikit-learn** to split into train and test sets, then wrap in **PyTorch** `TensorDataset` and `DataLoader`.

```python
```python
# 1) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# 2) Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32)

# 3) Create TensorDatasets
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

# 4) DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test loader: single batch (entire test set)
test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)
```

---

## 4. Model Definition & Dropout Integration
We build a small feedforward network with **two hidden layers** (or one, depending on structure) and incorporate **dropout** in the forward pass.

```python
```python
class QwertyDropoutModel(nn.Module):
    def __init__(self, dropoutRate=0.5):
        super().__init__()
        self.input = nn.Linear(2, 128)
        self.hidden = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)
        self.dr = dropoutRate  # store for usage in forward

    def forward(self, x):
        # Input layer -> ReLU -> Dropout
        x = F.relu(self.input(x))
        x = F.dropout(x, p=self.dr, training=self.training)
        
        # Hidden layer -> ReLU -> Dropout
        x = F.relu(self.hidden(x))
        x = F.dropout(x, p=self.dr, training=self.training)
        
        # Output layer (binary classification => 1 neuron)
        x = self.output(x)
        return x
```

**Note**:  
- `training=self.training` ensures that **dropout** is active only when `model.train()` is set, and **not** in `model.eval()`.

---

## 5. Training & Code Snippets

### 5.1. Training Loop
```python
```python
def create_model(dropout_rate=0.5):
    return QwertyDropoutModel(dropout_rate)

def train_model(model, train_loader, test_loader, epochs=1000):
    loss_fn = nn.BCEWithLogitsLoss()   # for binary classification
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Store accuracies
    train_accs = []
    test_accs  = []

    for epoch in range(epochs):
        # training
        model.train()  # ensure dropout is ON
        batch_accs = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).flatten()
            loss   = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            # batch accuracy
            preds = torch.sigmoid(y_pred) > 0.5
            acc   = (preds == y_batch.bool()).float().mean()
            batch_accs.append(acc.item())

        # average train accuracy over batches
        train_accs.append(np.mean(batch_accs))

        # evaluate on test set
        model.eval()  # dropout OFF
        with torch.no_grad():
            for X_testb, y_testb in test_loader:
                y_pred_test = model(X_testb).flatten()
                preds_test  = torch.sigmoid(y_pred_test) > 0.5
                acc_test    = (preds_test == y_testb.bool()).float().mean()
        test_accs.append(acc_test.item())

    return train_accs, test_accs
```

- We store **train** and **test** accuracy at each epoch.  
- Notice the toggling: `model.train()` vs. `model.eval()` around different loops.

### 5.2. Quick Test
```python
```python
model = create_model(dropout_rate=0.0)
train_accs, test_accs = train_model(model, train_loader, test_loader, epochs=1000)

plt.figure(figsize=(8,4))
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs,  label='Test Acc')
plt.title("Accuracy vs Epoch (No Dropout)")
plt.legend()
plt.show()
```
We might see some **oscillation** or “jitter” in accuracy, especially with small **batch sizes**.

---

## 6. Experiment: Varying Dropout Rates
We systematically vary dropout probability \(p\) from **0.0** to **1.0** (in increments) and average the final accuracy.

```python
```python
dropout_rates = np.linspace(0, 1, 10)
final_train_accuracies = []
final_test_accuracies  = []

for dr in dropout_rates:
    model = create_model(dropout_rate=dr)
    train_acc, test_acc = train_model(model, train_loader, test_loader, epochs=1000)
    
    # average last 100 epochs
    final_train_accuracies.append(np.mean(train_acc[-100:]))
    final_test_accuracies.append(np.mean(test_acc[-100:]))

# Plot results
plt.figure(figsize=(6,5))
plt.plot(dropout_rates, final_train_accuracies, 'o-', label='Train')
plt.plot(dropout_rates, final_test_accuracies, 'o-', label='Test')
plt.xlabel("Dropout Probability p")
plt.ylabel("Accuracy")
plt.title("Effect of Dropout Rate on Train/Test Accuracy")
plt.legend()
plt.show()
```

**Possible Outcomes**:
- **Best** performance might be at **p = 0.0** (no dropout), indicating that dropout doesn’t help in this scenario.  
- In other contexts or data, a moderate \(p \approx 0.5\) might yield better generalization.

---

## 7. Visualizing & Smoothing Accuracy
Because accuracy can **oscillate** with each epoch (due to small batch sizes or noisy updates), we may want to **smooth** the plotted curve using a **moving average** or convolution filter.

```python
```python
def smooth_series(series, k=5):
    # e.g. for each point, average k neighboring points
    box = np.ones(k)/k
    return np.convolve(series, box, mode='same')

# Example usage
epochs = len(train_accs)
train_acc_smooth = smooth_series(train_accs, k=5)
test_acc_smooth  = smooth_series(test_accs,  k=5)

plt.plot(range(epochs), train_acc_smooth, label='Train Smooth')
plt.plot(range(epochs), test_acc_smooth, label='Test Smooth')
plt.legend()
plt.title("Smoothed Accuracy vs Epoch")
plt.show()
```

> **Edge Effects**: Smoothing can distort the first and last few points (boundary artifacts). Sometimes you may just ignore the first/last `k/2` points.

---

## 8. Key Conclusions
1. **Dropout** doesn’t always yield improvements, especially if:
   - The model or data are relatively small or simpler.  
   - The data is not high-dimensional.  
2. **In other scenarios**, dropout can significantly boost generalization.  
3. **Systematic experiments** (as done here) help identify if dropout is beneficial and at what **rate** \(p\).  
4. **Smoothed** accuracy plots are often easier to interpret than highly noisy epoch-wise accuracy lines.  

**Overall**: Always treat dropout as a **hyperparameter**—test different rates and compare train/test performance before deciding on the best setting for your problem.

---

**End of Notes**  
Using this step-by-step approach, you can **confidently** incorporate dropout into practical deep learning pipelines, assess its **efficacy** via systematic experiments, and **interpret** results while mitigating noisy training dynamics.