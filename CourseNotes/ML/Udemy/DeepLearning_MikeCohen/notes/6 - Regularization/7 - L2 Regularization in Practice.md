## Table of Contents
1. [[Overview]]
2. [[Dataset and Data Preparation]]
3. [[Model Architecture]]
4. [[L2 Regularization in PyTorch (weight_decay)]]
5. [[Training Procedure]]
6. [[Experiment: Varying L2 Regularization]]
7. [[Results and Observations]]
8. [[Key Takeaways]]

---

## 1. Overview
**Objective**: Demonstrate how **easy** it is to apply L2 regularization to a PyTorch model, and explore how different **weight_decay** (L2 penalty) values affect the model’s performance on a small classification task (Iris).

**Data**: The **Iris** dataset (150 samples, 4 features, 3 classes):
- Often used as a **toy** or **intro** dataset, so we do not necessarily expect significant benefits from L2 regularization (the dataset is quite small, and the classification is not very difficult).

---

## 2. Dataset and Data Preparation
We use **scikit-learn** to load Iris and **train_test_split** to create a train/test partition. Then, **DataLoader** wraps the data for mini-batch training.

```python
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 1) Load iris
iris_data = load_iris()
X = iris_data.data          # shape: (150, 4)
y = iris_data.target        # 3 classes: 0, 1, 2

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# 3) Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# 4) Create DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)
```

---

## 3. Model Architecture
We define a simple feedforward network with one hidden layer. Since we only need a small model for Iris, we keep it **lightweight**.

```python
```python
def create_model():
    # 4 features input -> 12 hidden -> 3 output
    model = nn.Sequential(
        nn.Linear(4, 12),
        nn.ReLU(),
        nn.Linear(12, 12),
        nn.ReLU(),
        nn.Linear(12, 3)  # output layer
    )
    return model
```

- We are **not** inserting dropout or batch normalization in this example.  
- We will rely on **L2** regularization alone.

---

## 4. L2 Regularization in PyTorch (weight_decay)
In PyTorch, **L2** regularization is done by setting the **`weight_decay`** parameter of the optimizer. If **`weight_decay`** is set to 0, there is **no** L2 penalty.

```python
```python
def create_optimizer(model, lr=0.005, l2_lambda=0.0):
    # l2_lambda is the weight_decay hyperparameter
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_lambda)
    return optimizer
```

> **Note**: Many PyTorch optimizers (SGD, Adam, RMSProp, etc.) accept a `weight_decay` argument that implements **L2**.

---

## 5. Training Procedure
We write a **training loop** that:
1. Loops over epochs.
2. For each epoch, loops over **mini-batches** in `train_loader`.
3. Computes **loss** (CrossEntropy for multi-class).
4. Updates the model via **optimizer.step()**.
5. Computes **train accuracy** and **test accuracy** per epoch.

```python
```python
def train_model(model, train_loader, test_loader, l2_lambda=0.0, epochs=1000):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, lr=0.005, l2_lambda=l2_lambda)

    train_accs = []
    train_losses = []
    test_accs  = []

    for epoch in range(epochs):
        # 1) Training mode
        model.train()
        
        # track metrics across batches
        batch_accs = []
        batch_losses = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # forward pass
            y_pred = model(X_batch)
            loss   = loss_fn(y_pred, y_batch)

            # backprop
            loss.backward()
            optimizer.step()

            # compute accuracy in this batch
            preds = torch.argmax(y_pred, axis=1)
            acc   = (preds == y_batch).float().mean().item()
            
            batch_losses.append(loss.item())
            batch_accs.append(acc)

        # average across all batches
        train_accs.append(np.mean(batch_accs))
        train_losses.append(np.mean(batch_losses))
        
        # 2) Evaluate on test set
        model.eval()  # not strictly necessary if no dropout/batchnorm, but included for standard practice
        with torch.no_grad():
            for X_testb, y_testb in test_loader:
                y_test_pred = model(X_testb)
                test_preds  = torch.argmax(y_test_pred, axis=1)
                test_acc    = (test_preds == y_testb).float().mean().item()
        test_accs.append(test_acc)
        
        # revert to train mode if needed
        model.train()

    return train_accs, test_accs, train_losses
```

---

## 6. Experiment: Varying L2 Regularization
We systematically vary **l2_lambda** (often denoted \(\lambda\)) from 0.0 (no L2) to higher values (e.g., 0.1). Then measure final train/test accuracies.

```python
```python
l2_values = np.linspace(0, 0.1, 5)  # e.g. [0.0, 0.025, 0.05, 0.075, 0.1]

store_train_accs = {}
store_test_accs  = {}

epochs = 1000

for l2_val in l2_values:
    # create model and train
    model = create_model()
    train_acc, test_acc, train_loss = train_model(
        model, train_loader, test_loader, l2_lambda=l2_val, epochs=epochs
    )

    store_train_accs[l2_val] = train_acc
    store_test_accs[l2_val]  = test_acc
```

---

## 7. Results and Observations

### 7.1. Accuracy vs. Epoch
We can plot the training curves to see how each **L2** value affects convergence.

```python
```python
plt.figure(figsize=(8,6))

for l2_val, accs in store_train_accs.items():
    # optionally smooth
    smoothed_accs = np.convolve(accs, np.ones(10)/10, mode='same')
    plt.plot(smoothed_accs, label=f"L2={l2_val:.3f} (Train)")

plt.title("Train Accuracy vs. Epoch for Different L2")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

A separate plot for **test accuracy**:

```python
```python
plt.figure(figsize=(8,6))

for l2_val, accs in store_test_accs.items():
    smoothed_test = np.convolve(accs, np.ones(10)/10, mode='same')
    plt.plot(smoothed_test, label=f"L2={l2_val:.3f} (Test)")

plt.title("Test Accuracy vs. Epoch for Different L2")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

**Typical** Observations:
- L2 might **not** yield huge gains for small, easy datasets like Iris.  
- You might see slightly lower **train accuracy** for larger \(\lambda\), with test accuracy roughly similar or sometimes even worse, indicating **no** strong benefit from L2 if overfitting is minimal.

### 7.2. Final Accuracy vs. L2
We can also average the last 100 epochs to see the “steady-state” or late-epoch accuracy.

```python
```python
import statistics

avg_train_accs = []
avg_test_accs  = []

for l2_val in l2_values:
    tr = store_train_accs[l2_val]
    te = store_test_accs[l2_val]
    # average last 100 epochs
    avg_train = statistics.mean(tr[-100:])
    avg_test  = statistics.mean(te[-100:])
    avg_train_accs.append(avg_train)
    avg_test_accs.append(avg_test)

plt.figure(figsize=(6,4))
plt.plot(l2_values, avg_train_accs, 'o-', label='Train (avg last 100)')
plt.plot(l2_values, avg_test_accs,  'o-', label='Test (avg last 100)')
plt.xlabel("L2 Regularization Weight (lambda)")
plt.ylabel("Accuracy")
plt.title("L2 Parameter vs. Final Accuracy (Iris)")
plt.legend()
plt.show()
```

**Potential Outcome**: 
- You may see best performance around **\(\lambda=0\)** or a very small value.  
- Larger values degrade performance in both train and test accuracy.

---

## 8. Key Takeaways
1. **Implementation**: Adding **L2** to your PyTorch model is as simple as specifying `weight_decay=<value>` in the optimizer.  
2. **Small Datasets**: For tasks like Iris, you might see **no** real improvement or even **worse** performance with moderate/high L2. Overfitting is not a major issue for this easy dataset.  
3. **Tune**: Typically, you try a **range** of \(\lambda\) values (e.g., `[1e-5, 1e-4, 1e-3, …]`) and pick the best via a dev set.  
4. **Large Models**: L2 often has clearer benefits in **large** or **complex** networks where overfitting is more pronounced.  

**Conclusion**: L2 regularization is extremely straightforward in PyTorch. While the Iris example demonstrates the code mechanics, it also reveals that L2 **won’t always** help if you’re not strongly overfitting to begin with.