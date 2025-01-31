aliases: [Multi-Output ANNs, Qwerty Dataset, Multi-class Classification, PyTorch Practice]
tags: [Deep Learning, Lecture Notes, Neural Networks, PyTorch]

This notebook revisits a **multi-class classification** problem (the “qwerty” dataset) to reinforce:
1. **Construction** of multi-class MLPs (multi-layer perceptrons) in PyTorch.  
2. **Model training** using `DataLoader` and **CrossEntropyLoss**.  
3. **Visualization** of results, including **accuracy per class** and **error patterns**.

Despite having seen multi-class ANNs before, additional practice helps build **flexibility** and deeper **intuition** for real-world deep learning tasks.

---

## 1. Imports & Basic Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

print("PyTorch version:", torch.__version__)
```

---

## 2. Synthetic Qwerty Data Generation

We'll create a dataset with **3 categories**, each containing qwerty samples in a 2D space (x,y). This is **similar** to previous sections, but we apply some new steps (like train/test splitting with `DataLoader`).

```python
# For reproducibility
np.random.seed(42)

# Number of samples per class
n_per_class = 300

# Means (2D) for each class (3 classes => 3 means)
means = np.array([
    [ 2,  1],
    [-2,  1],
    [ 0, -3]
])

# Covariance matrix (common for all classes)
cov  = np.array([[1, 0],[0, 1]])

# Generate data
X = np.zeros((n_per_class*3, 2))
y = np.zeros(n_per_class*3)

for i in range(3):
    idx = range(i*n_per_class, (i+1)*n_per_class)
    X[idx,:] = np.random.multivariate_normal(means[i], cov, n_per_class)
    y[idx]   = i

print("X shape:", X.shape, "| y shape:", y.shape)

plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap="Set2", alpha=0.7)
plt.title("Qwerty Dataset (3-class)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

---

## 3. Train/Test Split and DataLoaders

We convert `X, y` into PyTorch tensors, do a **train/test** split, and wrap them in a `DataLoader`.

```python
# Convert to tensors
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_t, y_t, test_size=0.2, shuffle=True, random_state=42
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, drop_last=True)

print("Num train batches:", len(train_loader))
print("Num test  batches:", len(test_loader))
```

---

## 4. Defining the Model

A simple MLP with:

- **2 inputs** (the x,y coordinates).
- 1 or 2 hidden layers, e.g. size 8 or 16.
- **3 outputs** (for 3 categories) → final activation *not* needed if using `CrossEntropyLoss` in PyTorch (which applies `LogSoftmax` internally).

```python
class QwertyNet(nn.Module):
    def __init__(self):
        super(QwertyNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 3)  # 3 output classes
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # no relu on final output -> raw logits for CrossEntropy
        x = self.fc2(x)
        return x

# Quick test of dimensions
model = QwertyNet()
x_test_in = torch.randn(10, 2)
y_test_out = model(x_test_in)
print("Model output shape on 10 samples:", y_test_out.shape)  # expect [10,3]
```

### Note on Activations

- We use **`F.relu(...)`** from `torch.nn.functional` rather than instantiating `nn.ReLU()` in `__init__`.  
- Either approach is valid; it’s often a **personal/style choice**.  

---

## 5. Training Function

We’ll define a helper function to train for `epochs` and track:

- **Train Loss** & **Train Accuracy**  
- **Test Loss** & **Test Accuracy**

```python
def train_model(model, train_loader, test_loader, epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_loss_history = []
    train_acc_history  = []
    test_loss_history  = []
    test_acc_history   = []
    
    for ep in range(epochs):
        # ---- Training ----
        model.train()
        batch_loss = 0
        correct    = 0
        total      = 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
            
            # accuracy
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == yb).item()
            total   += len(yb)
        
        train_loss = batch_loss / len(train_loader)
        train_acc  = correct / total
        
        # ---- Testing ----
        model.eval()
        batch_loss_test = 0
        correct_test    = 0
        total_test      = 0
        
        with torch.no_grad():  # no gradient tracking during eval
            for Xb, yb in test_loader:
                logits_test = model(Xb)
                loss_test   = criterion(logits_test, yb)
                batch_loss_test += loss_test.item()
                
                preds_test = torch.argmax(logits_test, dim=1)
                correct_test += torch.sum(preds_test == yb).item()
                total_test   += len(yb)
        
        test_loss = batch_loss_test / len(test_loader)
        test_acc  = correct_test / total_test
        
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        
    return {
        "train_loss": train_loss_history,
        "train_acc":  train_acc_history,
        "test_loss":  test_loss_history,
        "test_acc":   test_acc_history
    }
```

---

## 6. Run Training and Plot Results

```python
model = QwertyNet()

epochs = 100
results = train_model(model, train_loader, test_loader, epochs=epochs, lr=0.01)

# Plot losses and accuracies
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(results["train_loss"], label="Train Loss")
plt.plot(results["test_loss"],  label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(results["train_acc"], label="Train Acc")
plt.plot(results["test_acc"],  label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over epochs")
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 7. Inspecting Model Predictions

After training:

1. Feed **all** data (train + test) through the model.
2. Compare predicted class vs. true class for each sample.

### 7.1 Predicted vs. True Class (all samples)

```python
# Combine train+test
X_all = torch.cat([X_train, X_test], dim=0)
y_all = torch.cat([y_train, y_test], dim=0)

model.eval()
with torch.no_grad():
    logits_all = model(X_all)
    preds_all  = torch.argmax(logits_all, dim=1)
    
print("X_all shape:", X_all.shape, "| preds_all shape:", preds_all.shape)
```

#### Visual Compare per Sample

```python
plt.figure(figsize=(10,4))
plt.plot(preds_all.numpy(), 'bo', label="Preds")
# offset the true labels by +0.2 for clarity
plt.plot(y_all.numpy()+0.2, 'rs', label="True Labels")
plt.title("Predicted vs. True (all samples)")
plt.xlabel("Sample Index")
plt.ylabel("Class Label")
plt.legend()
plt.show()
```

- Blue circles = predicted class
- Red squares = true class (+0.2 offset on y-axis)

Visually, **correct** predictions align on the same integer level; **mismatches** are offset.

---

### 7.2 Accuracy Per Class

We can see if the model is more accurate on some classes vs. others.

```python
accuracy = (preds_all == y_all).numpy()  # array of bool

# overall
overall_acc = accuracy.mean()
print("Overall Accuracy (all data):", f"{overall_acc*100:.2f}%")

# class-wise
for c in [0,1,2]:
    class_mask = (y_all == c).numpy()
    class_acc  = accuracy[class_mask].mean()
    print(f" Class {c} Accuracy: {class_acc*100:.2f}%")
```

---

### 7.3 Visualizing Misclassified Points in x-y Space

```python
misclassified = preds_all != y_all

plt.figure(figsize=(6,6))
plt.scatter(X_all[:,0], X_all[:,1], c=y_all, cmap="Set2", alpha=0.6)
# highlight misclassified
plt.scatter(X_all[misclassified,0],
            X_all[misclassified,1],
            edgecolor="k", facecolors="none", linewidths=1.5,
            s=80, label="Misclassified")

plt.title("x-y space with Misclassifications")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
```

---

## 8. Possible Extensions

1. **Change Architecture**:
   - Add extra hidden layer, or change # neurons.  
   - Try `LeakyReLU` or `ReLU6`.  
2. **Batch Normalization**:
   - Insert BN layers and see if training stabilizes or speeds up.  
3. **no_grad** Exploration:
   - Move or remove `with torch.no_grad():` in testing to see how it affects gradient tracking.

Real-world code is rarely written **from scratch**—it often evolves by **modifying** reference examples like this.

---

## 9. Conclusion

We revisited **3-class classification** with a synthetic **qwerty** dataset, but using more **modern** PyTorch practices (DataLoaders, train/test split, CrossEntropyLoss). Key points:

1. **Multi-class** → **CrossEntropyLoss** with raw **logits**.  
2. **Different** ways to visualize success/fail patterns in multi-class tasks:
   - Checking predicted vs. true labels across all samples.  
   - Accuracy **per class** to identify potential class imbalance or biased predictions.  
   - Marking **misclassified** points in feature space for deeper insight.

This example hopefully strengthens your skill in multi-output MLPs, bridging earlier “ANN” topics with the more **PyTorch**-centric approach used throughout this course.

---

## 10. References

- [PyTorch Dataloader & Dataset Docs](https://pytorch.org/docs/stable/data.html)  
- **Cross Entropy** in PyTorch: [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)  
- **Activation Functions** Reference: [nn.ReLU, nn.LeakyReLU, etc.](https://pytorch.org/docs/stable/nn.html#non-linear-activations)  

```
Created by: [Your Name / Lab / Date]
Based on Lecture: “Meta parameters: More practice with multioutput ANNs”
```
```