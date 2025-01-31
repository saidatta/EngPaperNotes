Below is a **very detailed, Obsidian-friendly** set of notes on a **code challenge** exploring the **effects of mini-batch size** on training performance in a neural network. These notes cover the objectives, code insights, and illustrative results, providing an overview of **why** batch size matters and **how** to run a parameter sweep over different batch sizes.

## Table of Contents
1. [[Goal of the Code Challenge]]
2. [[Dataset and Model Setup]]
3. [[Experiment Design]]
4. [[Implementation Details]]
5. [[Observations and Visualization]]
6. [[Analysis of Results]]
7. [[Further Exploration & Tips]]

---

## 1. Goal of the Code Challenge
- **Objective**: Systematically vary **mini-batch size** (from \(2^1\) to \(2^6\)) in a neural network training loop and measure:
  1. **How fast** the model learns,
  2. **Final** train/test accuracy.

- **Data**: A **circular “qwerties doughnuts”** dataset, a non-linear classification problem—more interesting visually than Iris.

- **Outcome**: Plot train/test accuracy vs. epochs for different batch sizes on the **same figure** to compare **learning speed** and **convergence**.

---

## 2. Dataset and Model Setup
### 2.1. Qwerties Doughnuts
A toy dataset featuring **two circular** rings or arcs:
- Each ring belongs to a **different class**.
- Non-linear separation, but easily handled by a small MLP.

```python
```python
# Pseudocode for generating 'qwerties doughnuts' dataset
N = 2000
thetas = np.linspace(0, 4*np.pi, N)  # angles
# Class 0: smaller radius
r0 = 10 + np.random.randn(N//2)*0.5
x0 = r0*np.cos(thetas[:N//2])
y0 = r0*np.sin(thetas[:N//2])
# Class 1: larger radius
...
```

### 2.2. Model Definition
A **small** MLP typically works:
```python
```python
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)  # binary classification
)
```
*(Exact hidden sizes can vary.)*

---

## 3. Experiment Design
- **Batch sizes**: \(\{2, 4, 8, 16, 32, 64\}\).  
- **Learning rate**: `0.001` (as recommended).
- **Epochs**: Enough (e.g., 300–1000) to see how quickly (or whether) training curves converge.
- **Record**:
  - **Train accuracy** vs. epoch,
  - **Test accuracy** vs. epoch.

**Hypotheses**:
- Smaller batches might **learn faster** initially or show more stability for this dataset, or
- Larger batches could converge slower but possibly reach a similar final accuracy.

---

## 4. Implementation Details
Below is a typical structure:

```python
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) Generate or load "qwerties doughnuts" dataset
# X_train, y_train, X_test, y_test ...

batch_sizes = [2**i for i in range(1,7)]  # [2,4,8,16,32,64]

store_results = {}

for bsize in batch_sizes:
    # Create train_loader and test_loader
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)
    
    # Build model, define loss & optimizer
    model = MyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    train_accs, test_accs = [], []
    epochs = 500

    for epoch in range(epochs):
        # Train mode
        model.train()
        # track train metrics
        batch_accs = []
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            y_pred = model(Xb).flatten()
            loss = loss_fn(y_pred, yb.float())
            loss.backward()
            optimizer.step()
            
            preds = torch.sigmoid(y_pred) > 0.5
            acc   = (preds == yb.bool()).float().mean().item()
            batch_accs.append(acc)
        train_accs.append(np.mean(batch_accs))

        # Evaluate test
        model.eval()
        with torch.no_grad():
            X_testb, y_testb = next(iter(test_loader))
            y_test_pred = model(X_testb).flatten()
            test_loss   = loss_fn(y_test_pred, y_testb.float())
            preds_test  = torch.sigmoid(y_test_pred) > 0.5
            test_acc    = (preds_test == y_testb.bool()).float().mean().item()
        test_accs.append(test_acc)

    # Store the accuracy curves
    store_results[bsize] = (train_accs, test_accs)
```

---

## 5. Observations and Visualization
### 5.1. Plotting Accuracies
After collecting `(train_accs, test_accs)` for each batch size, we can overlay them on a single figure:

```python
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

for bsize, (train_accs, test_accs) in store_results.items():
    plt.plot(train_accs, label=f'batch={bsize} (Train)')
    # or overlay test_accs in a separate plot
    # or as subplots

plt.title("Train Accuracy vs Epoch for Different Mini-Batch Sizes")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

*(**Alternatively**, make two subplots—one for train accuracy, one for test accuracy.)*

---

## 6. Analysis of Results
Typical patterns that might emerge:

1. **Small batch** (e.g. 2, 4):
   - More frequent updates, **often** faster early learning.
   - Might converge quickly to high accuracy for a simpler dataset like “qwerties doughnuts”.

2. **Intermediate batch** (16, 32):
   - Balanced approach, stable learning, might converge well if data is heterogeneous.

3. **Large batch** (64):
   - Fewer updates each epoch, potentially slower to progress initially.
   - Could eventually match or exceed the performance if given **enough epochs**.

**Caution**:  
- For some tasks/datasets, large batch can hamper final performance or require careful tuning of learning rate and epoch count.

---

## 7. Further Exploration & Tips
1. **Extend epochs** for large batch sizes to see if they catch up in final accuracy.  
2. **Try** different **learning rates**: Larger batch sizes often need higher or more carefully tuned LR.  
3. **Plot** not only final accuracies but also **loss curves** to compare convergence.  
4. **Consider** more complex datasets: mini-batch effects can become more pronounced with higher dimensional data or more samples.  
5. **Memory constraints**: If a dataset is **large**, smaller batch sizes might be forced by GPU memory.

**Conclusion**: This code challenge reveals how **mini-batch size** significantly impacts **training dynamics**—smaller batches can learn faster (especially for homogeneous data), while larger batches might show slower but eventually robust convergence with adequate epochs or an adjusted learning rate.