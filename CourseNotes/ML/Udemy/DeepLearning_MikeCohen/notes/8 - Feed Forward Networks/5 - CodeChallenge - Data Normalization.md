aliases: [CodeChallenge, Data Normalization, FFN, MNIST]
tags: [Deep Learning, DataNormalization, CodeChallenge, Classification]
## Overview
This challenge explores how **data normalization** (and potential mismatches between training and test normalization) impacts a **Feedforward Network (FFN)** on MNIST. It reinforces three important concepts:
1. **Always normalize data** to avoid scale disparities.
2. **Train and test** must be **scaled consistently**.
3. **Loss values** are **scale-dependent**, while **accuracy** is more robust to input scaling.

You’ll run a classification experiment **three times**:
1. Both **train & test** normalized to \([0,1]\).
2. **Train** normalized \([0,1]\), **test** in original \([0,255]\).
3. **Train** in original \([0,255]\), **test** normalized \([0,1]\).

Observe the **loss** and **accuracy** curves for each scenario and interpret the results.

---
## 1. Background & Motivation

### 1.1 Why Normalize?
- **Loss Scale Dependency**: A model trained on large numeric ranges often yields **high numeric loss** values, even if the task performance (accuracy) is acceptable.
- **Gradient Stability**: Normalizing inputs helps avoid exploding or vanishing gradients.
- **Comparison**: Accuracy is easier to compare across scaling because it’s a discrete measure (correct/incorrect).

### 1.2 Min-Max Scaling Recap
A common approach is **Min-Max scaling** to \([0,1]\):
\[
X_{\text{scaled}} = \frac{X - \min(X)}{\max(X) - \min(X)}
\]
For MNIST specifically, \(\min(X)=0\) and \(\max(X)=255\).

---

## 2. Experiment Outline

Below is an outline showing how to run **three** configurations. We assume we have a partial MNIST dataset loaded from CSV (similar to previous challenges):

| Version | Train Range   | Test Range    | Expected Result                                         |
|---------|---------------|---------------|---------------------------------------------------------|
| #0      | 0 - 1         | 0 - 1         | *Baseline* ~95% accuracy, stable loss.                  |
| #1      | 0 - 1         | 0 - 255       | Accuracy still decent (~95%), but test loss is huge.    |
| #2      | 0 - 255       | 0 - 1         | Accuracy plummets (~20%), training can't generalize.    |

---

## 3. Code Setup & Data Loading

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1. Load partial MNIST (~20k samples)
data_path = "/content/sample_data/mnist_train_small.csv"
data_np = np.loadtxt(data_path, delimiter=",")

# 2. Separate labels and pixels
labels_np = data_np[:, 0].astype(int)
pixels_np = data_np[:, 1:].astype(float)

print("Original data range:", pixels_np.min(), "to", pixels_np.max())
# Should be 0 to 255 for MNIST
```

---

## 4. Train/Test Splits

```python
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    pixels_np, labels_np, test_size=0.10, random_state=42
)

print("Before normalization:")
print("Train data range: ", X_train.min(), "to", X_train.max())
print("Test data range:  ", X_test.min(),  "to", X_test.max())
```

---

## 5. The Three Scenarios

### 5.1 Version #0: Both Train & Test in [0,1]

```python
# Normalize both train & test
X_train_norm = X_train / 255.0
X_test_norm  = X_test  / 255.0

print("After normalization:")
print("Train data range:", X_train_norm.min(), "to", X_train_norm.max())
print("Test data range: ", X_test_norm.min(),  "to", X_test_norm.max())
```

### 5.2 Version #1: Train in [0,1], Test in [0,255]

```python
# Train normalized, test left unnormalized
X_train_norm = X_train / 255.0
X_test_norm  = X_test  # no change

print("After mismatched normalization (train norm, test not):")
print("Train data range:", X_train_norm.min(), "to", X_train_norm.max())
print("Test data range: ", X_test_norm.min(),  "to", X_test_norm.max())
```

### 5.3 Version #2: Train in [0,255], Test in [0,1]

```python
# Train left unnormalized, test normalized
X_train_norm = X_train  # no change
X_test_norm  = X_test / 255.0

print("After mismatched normalization (train not, test norm):")
print("Train data range:", X_train_norm.min(), "to", X_train_norm.max())
print("Test data range: ", X_test_norm.min(),  "to", X_test_norm.max())
```

> **Note**: You only need to comment/uncomment one line for the train set and one for the test set. Each version is simply re-running the script with different normalization.

---

## 6. Convert to Tensors & Dataloaders

For each version, after deciding on `X_train_norm` and `X_test_norm`:

```python
# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_norm,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# Make datasets and dataloaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)
```

---

## 7. Defining the Model

We can switch to **raw outputs** (logits) and let **CrossEntropyLoss** handle the softmax internally. That means **no** `torch.log_softmax` in the forward pass.

```python
class MNIST_FFN(nn.Module):
    def __init__(self):
        super(MNIST_FFN, self).__init__()
        self.fc0 = nn.Linear(784, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # raw logits

model = MNIST_FFN()
```

### 7.1 Loss & Optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

---

## 8. Training Loop

```python
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=60):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        batch_losses, batch_accs = [], []
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            acc = (preds == yb).float().mean().item()
            batch_accs.append(acc)
        
        # End of epoch - compute train metrics
        train_loss = np.mean(batch_losses)
        train_acc = np.mean(batch_accs)
        
        # Evaluate on test
        model.eval()
        test_batch_losses, test_batch_accs = [], []
        with torch.no_grad():
            for Xb_t, yb_t in test_loader:
                logits_t = model(Xb_t)
                loss_t = criterion(logits_t, yb_t)
                test_batch_losses.append(loss_t.item())
                preds_t = torch.argmax(logits_t, dim=1)
                acc_t = (preds_t == yb_t).float().mean().item()
                test_batch_accs.append(acc_t)
        test_loss = np.mean(test_batch_losses)
        test_acc = np.mean(test_batch_accs)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")
            
    return train_losses, test_losses, train_accs, test_accs

# Train
model = MNIST_FFN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_losses, test_losses, train_accs, test_accs = train_model(
    model, train_loader, test_loader, criterion, optimizer, epochs=60
)
```

---

## 9. Plotting Results

```python
epochs_range = range(1, 61)
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, test_losses,  label="Test Loss")
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, train_accs, label="Train Accuracy")
plt.plot(epochs_range, test_accs,  label="Test Accuracy")
plt.title("Accuracy Curves")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 10. Observations

### 10.1 Scenario #0: Both Normalized [0,1]
- **Train** & **Test** have the same scale.
- We get **~95%** test accuracy (typical).
- **Loss** numbers are moderate, decreasing smoothly.

### 10.2 Scenario #1: Train [0,1], Test [0,255]
- Accuracy remains **fairly high** (~95%), because **test data** is effectively *"scaled outward"* from the learned embedding.  
- However, **test loss** can be large. The numeric mismatch inflates the raw logit scale (the model sees bigger inputs than it expects).

### 10.3 Scenario #2: Train [0,255], Test [0,1]
- The model now expects **larger** input values. But **test** data is “shrunken” near the origin.  
- Accuracy **plummets** (around ~20%).  
- Training does not generalize to much smaller input magnitudes.

---

## 11. Geometric Interpretation

Consider a simplified **3D** input space:
- **Normalized** data cluster near the **unit sphere**.
- **Unnormalized** data cluster further out in **radius**.
- When **train** is near the origin and **test** is far away, the model’s learned decision boundaries can still scale outwards (like a fan) → less performance penalty.
- When **train** is far from origin and **test** is at the origin, the learned vectors compress → categories become entangled → poor performance.

---

## 12. Key Takeaways

1. **Always Normalize Both**: Train & test data **must** share the same scaling.
2. **Loss vs. Accuracy**:  
   - **Loss** is sensitive to input scale, so absolute values can vary wildly.  
   - **Accuracy** is discrete and remains within [0,100%]; it’s often more comparable across scaling.
3. **Mismatched Ranges**:  
   - Train [0,1] → Test [0,255]: Could still yield decent accuracy, but inflated test loss.  
   - Train [0,255] → Test [0,1]: Drastically worsens performance (the model is “unprepared” for small input values).
4. **Habit**: Always normalize your data consistently. Even if sometimes it “works” unnormalized, consistent normalization is best practice.

---

## 13. Further Exploration
- Try other scaling strategies: **Standardization** (\(z\)-score), **RobustScaler**, etc.
- Investigate **misclassified examples** in each scenario to see how drastically input scale changes predictions.
- Extend this concept to other datasets (CIFAR, custom images, etc.) and advanced models (CNNs).

---

**End of Notes – "CodeChallenge: Data Normalization"**  
```