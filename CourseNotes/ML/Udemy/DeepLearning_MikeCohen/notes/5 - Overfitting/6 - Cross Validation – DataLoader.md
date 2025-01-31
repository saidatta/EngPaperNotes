## Table of Contents
1. [[Introduction to DataLoaders]]
2. [[Review: Splitting Data for Training & Testing]]
3. [[Mini-Batches: Concept & Advantages]]
4. [[Creating PyTorch TensorDataset]]
5. [[Building and Using DataLoader]]
6. [[Code Walkthrough: Training with Mini-Batches]]
7. [[Key Takeaways]]

---
## 1. Introduction to DataLoaders
A **DataLoader** is a PyTorch utility that:
- Converts a dataset into an **iterable**.
- Shuffles and batches the data for convenient **mini-batch** training.
- Allows you to loop over data in training loops without manually indexing.

**In Cross-Validation**:
- You can **still** use scikit-learn’s `train_test_split` (or any other split method) to produce train/test (or train/dev/test) subsets.
- Then wrap each subset in a **`TensorDataset`** → feed that into **`DataLoader`**.

---

## 2. Review: Splitting Data for Training & Testing
1. **Manual** or **scikit-learn** approach:
   - **train_test_split** creates `(X_train, X_test, y_train, y_test)`.
   - Typical ratio: ~80/20 or 90/10, etc.
2. **DataLoader**:
   - Often you’ll do: 
     1. Use scikit-learn to get `X_train, X_test, y_train, y_test`.  
     2. Convert each of these into a **PyTorch** dataset (`TensorDataset`).  
     3. Create **DataLoader** objects for each subset.

---

## 3. Mini-Batches: Concept & Advantages
- **Batch Size**: Number of samples per gradient update step.
- **Trade-off**:
  - **batch_size=1**: Very noisy updates, can take many epochs to converge.  
  - **batch_size = full dataset**: Fewer steps, but can be **slower** to compute each step and might converge less efficiently.
- **Benefits**:
  1. **Faster training** per epoch compared to single-sample updates.  
  2. **Less memory** usage per step vs. using the entire dataset at once (for large datasets).  
  3. **Often better generalization** performance.

---

## 4. Creating PyTorch TensorDataset
**TensorDataset** combines features (`X`) and labels (`y`) into a single object:

```python
```python
import torch
from torch.utils.data import TensorDataset

# Suppose we have X_train (features) and y_train (labels)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

# Create a dataset wrapping tensors
train_data = TensorDataset(X_train_t, y_train_t)
```

- The dataset ensures `(features, labels)` **remain aligned** and accessible by PyTorch.

---

## 5. Building and Using DataLoader
### 5.1. Basic Usage
```python
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_data,
    batch_size=32,    # e.g., 32 samples per mini-batch
    shuffle=True      # randomize order each epoch
)
```
- `DataLoader` is **iterable**:
  ```python
  for batch_X, batch_y in train_loader:
      # batch_X.shape -> [batch_size, n_features]
      # batch_y.shape -> [batch_size]
  ```
- Each iteration yields a **mini-batch** of `(X, y)` pairs.

### 5.2. Example with Test Data
```python
test_data = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=1,     # or len(X_test) to get them all at once
    shuffle=False
)
```

---

## 6. Code Walkthrough: Training with Mini-Batches

Below is a typical workflow using the **Iris dataset**:

### 6.1. Data Preparation & Splitting
```python
```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 1) Load data
iris = load_iris()
X = iris.data  # shape: (150, 4)
y = iris.target  # shape: (150,)

# 2) Split (e.g. 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.8,
    shuffle=True,
    random_state=42
)

# 3) Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# 4) Create TensorDataset
train_data = TensorDataset(X_train_t, y_train_t)
test_data  = TensorDataset(X_test_t, y_test_t)

# 5) Build DataLoader
train_loader = DataLoader(
    dataset=train_data,
    batch_size=12,  # e.g., mini-batches of 12 samples
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=len(test_data), # entire test set in one batch
    shuffle=False
)
```

### 6.2. Defining the Model
```python
```python
model = torch.nn.Sequential(
    torch.nn.Linear(4, 16),  # 4 input features
    torch.nn.ReLU(),
    torch.nn.Linear(16, 3)   # 3 classes (Iris species)
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

### 6.3. Training Loop with Mini-Batches
```python
```python
num_epochs = 200
train_acc_history = []
test_acc_history  = []

for epoch in range(num_epochs):
    # Keep track of batch-level accuracy
    batch_accuracies = []
    
    # 1) Loop over mini-batches from train_loader
    for batch_X, batch_y in train_loader:
        # Forward pass
        y_pred = model(batch_X)
        loss   = loss_fn(y_pred, batch_y)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute batch accuracy
        preds = torch.argmax(y_pred, axis=1)
        batch_acc = (preds == batch_y).float().mean().item()
        batch_accuracies.append(batch_acc)
    
    # 2) Training accuracy for this epoch = avg of batch accuracies
    epoch_train_acc = np.mean(batch_accuracies)
    train_acc_history.append(epoch_train_acc)
    
    # 3) Evaluate on test set
    # Here we only have one batch in test_loader (full test set)
    test_batch_X, test_batch_y = next(iter(test_loader))
    with torch.no_grad():
        test_pred = model(test_batch_X).argmax(dim=1)
        epoch_test_acc = (test_pred == test_batch_y).float().mean().item()
    test_acc_history.append(epoch_test_acc)
    
    # (Optional) Print progress
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Acc: {epoch_train_acc:.3f}, "
              f"Test Acc: {epoch_test_acc:.3f}")
```

### 6.4. Visualizing Accuracy Curves
```python
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(test_acc_history,  label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs. Test Accuracy Over Epochs")
plt.legend()
plt.show()
```

**Expect**:
- **Training** accuracy often rises faster when using **mini-batches** vs. single-sample training.
- **Test** accuracy lags slightly behind, as usual, but typically converges well.

---

## 7. Key Takeaways
1. **DataLoader**:
   - A **convenient** and **essential** PyTorch object for iterating over training data in **batches**.
   - Shuffling helps avoid model overfitting to any given data order.
2. **Mini-Batch Training**:
   - Reduces overall training time (especially for larger datasets).  
   - Often improves generalization and numerical stability.
3. **Workflow**:
   1. **Split** with `train_test_split`.  
   2. Create **`TensorDataset`** from `(features, labels)`.  
   3. Build **DataLoader** for **train** and **test**.  
   4. **Iterate** over DataLoader in the training loop, applying **backprop** on each mini-batch.
4. **Iris Example**:
   - Small dataset, but demonstrates how quickly the model can achieve high accuracy with mini-batch training vs. single-sample updates.

---

**End of Notes**.  
By integrating **DataLoader** objects and mini-batch training into your workflow, you’ll be well-equipped to handle larger datasets and achieve efficient, high-quality training in PyTorch.