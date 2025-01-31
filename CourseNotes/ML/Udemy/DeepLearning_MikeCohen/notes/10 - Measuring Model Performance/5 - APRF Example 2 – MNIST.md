title: "Measuring Model Performance: APRF Example 2 (MNIST)"
tags: [deep-learning, mnist, multi-class-classification, performance-metrics]
## 1. Overview

In this lecture, we extend the **ARPF metrics**—Accuracy, Recall, Precision, and F1—**to a multi-class problem** using the classic **MNIST** dataset (handwritten digits 0–9). We will:

1. **Train** a simple neural network on MNIST.  
2. Compute **multi-class** versions of Accuracy, Precision, Recall, and F1.  
3. Examine a **10×10 confusion matrix** to see how the network confuses digits with each other.

### Key Takeaways
- **Multi-Class ARPF** can be computed for **each class** (digit) or **averaged** across classes in different ways (e.g., *macro*, *weighted*).  
- The **confusion matrix** for multi-class tasks is no longer 2×2 but *C×C* for C classes (10 for MNIST).  
- High-level metrics can **mask** per-class biases; we need a deeper look at **each digit’s** Precision/Recall and the confusion matrix.

---

## 2. Data and Model Setup

### 2.1. Imports and Data Loading

Below is a compact example of setting up MNIST in PyTorch. We also import `scikit-learn.metrics` (abbreviated as `skm`) to compute classification metrics.

```python
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```
```

- **MNIST** is available via `torchvision.datasets.MNIST`.
- We typically **normalize** the images and load them into **DataLoader** objects.

```python
```python
# Transforms for MNIST: converting to tensor, normalizing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download + load MNIST train/test
train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

print(f"Train set size: {len(train_data)}, Test set size: {len(test_data)}")
```
```

### 2.2. Simple Neural Network Model

We define a straightforward **feed-forward** or **convolutional** model. Below is a simple fully connected variant:

```python
```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Flatten 2D images into 1D
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # raw logits for 10 classes
        return x

model = MNISTNet()
print(model)
```
```

---

## 3. Training the Model

### 3.1. Hyperparameters and Setup
We use **Adam** with a relatively higher learning rate (e.g., 0.01) to see how performance quickly saturates.  

```python
```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
epochs = 10
```
```

### 3.2. Training Loop

```python
```python
train_losses = []
test_losses  = []
train_acc    = []
test_acc     = []

for epoch in range(epochs):
    # --- TRAIN ---
    model.train()
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_hat = model(batch_X)
        loss = loss_fn(y_hat, batch_y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # compute training accuracy per batch
        preds = y_hat.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total   += len(batch_y)
    
    epoch_train_acc = correct / total
    train_acc.append(epoch_train_acc)
    
    # --- EVAL ---
    model.eval()
    correct_test = 0
    total_test   = 0
    with torch.no_grad():
        for X_t, y_t in test_loader:
            y_hat_t = model(X_t)
            loss_t  = loss_fn(y_hat_t, y_t)
            test_losses.append(loss_t.item())
            
            preds_t = y_hat_t.argmax(dim=1)
            correct_test += (preds_t == y_t).sum().item()
            total_test   += len(y_t)
    
    epoch_test_acc = correct_test / total_test
    test_acc.append(epoch_test_acc)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Acc: {epoch_train_acc:.3f}, Test Acc: {epoch_test_acc:.3f}")
```
```

**Note**: MNIST typically can reach 95%+ accuracy in a few epochs with this setup.

### 3.3. Plotting Training Curves
```python
```python
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses,  label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Training Iterations')

plt.subplot(1,2,2)
plt.plot(train_acc, label='Train Acc')
plt.plot(test_acc,  label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.tight_layout()
plt.show()
```
```

---

## 4. Generating Predictions for ARPF Metrics

We compute final predictions on **all** train/test samples (not just minibatches):

```python
```python
# Switch to eval mode
model.eval()

# Gather all data from DataLoader
X_train_all = []
y_train_all = []
for Xbatch, ybatch in train_loader:
    X_train_all.append(Xbatch)
    y_train_all.append(ybatch)
X_train_all = torch.cat(X_train_all, dim=0)
y_train_all = torch.cat(y_train_all, dim=0)

X_test_all = []
y_test_all = []
for Xbatch, ybatch in test_loader:
    X_test_all.append(Xbatch)
    y_test_all.append(ybatch)
X_test_all = torch.cat(X_test_all, dim=0)
y_test_all = torch.cat(y_test_all, dim=0)

# Predictions
with torch.no_grad():
    train_logits = model(X_train_all)
    train_preds  = train_logits.argmax(dim=1)
    test_logits  = model(X_test_all)
    test_preds   = test_logits.argmax(dim=1)
    
print("Train predictions shape:", train_preds.shape)
print("Test predictions shape: ", test_preds.shape)
```
```

Now we have **integer** predictions \([0,9]\) for the entire dataset.

---

## 5. Multi-Class Metrics: Accuracy, Precision, Recall, F1

In **multi-class** settings, `scikit-learn` provides different ways to **average** metrics across classes:

- **`average='macro'`**: Unweighted mean across classes.  
- **`average='weighted'`**: Weighted mean by support (the number of samples in each class).  
- **`average=None`**: Returns a *vector of metrics* for each class separately.

### 5.1. Converting Tensors to NumPy

```python
```python
y_train_np = y_train_all.numpy()
y_test_np  = y_test_all.numpy()

train_preds_np = train_preds.numpy()
test_preds_np  = test_preds.numpy()
```
```

### 5.2. Computing Metrics

```python
```python
acc_train   = skm.accuracy_score(y_train_np, train_preds_np)
prec_train  = skm.precision_score(y_train_np, train_preds_np, average='weighted')
rec_train   = skm.recall_score(y_train_np,  train_preds_np, average='weighted')
f1_train    = skm.f1_score(y_train_np,      train_preds_np, average='weighted')

acc_test    = skm.accuracy_score(y_test_np, test_preds_np)
prec_test   = skm.precision_score(y_test_np, test_preds_np, average='weighted')
rec_test    = skm.recall_score(y_test_np,  test_preds_np, average='weighted')
f1_test     = skm.f1_score(y_test_np,      test_preds_np, average='weighted')

print("Train Metrics (Weighted Avg):")
print(f"  Accuracy : {acc_train:.4f}")
print(f"  Precision: {prec_train:.4f}")
print(f"  Recall   : {rec_train:.4f}")
print(f"  F1 Score : {f1_train:.4f}")

print("\nTest Metrics (Weighted Avg):")
print(f"  Accuracy : {acc_test:.4f}")
print(f"  Precision: {prec_test:.4f}")
print(f"  Recall   : {rec_test:.4f}")
print(f"  F1 Score : {f1_test:.4f}")
```
```

**Example Output** (typical ranges):
```
Train Metrics (Weighted Avg):
  Accuracy : 0.9300
  Precision: 0.9301
  Recall   : 0.9300
  F1 Score : 0.9299

Test Metrics (Weighted Avg):
  Accuracy : 0.9200
  Precision: 0.9201
  Recall   : 0.9200
  F1 Score : 0.9198
```
*Note*: The exact numbers depend on network initialization, number of epochs, etc.

---

## 6. Visualizing Metrics

### 6.1. Comparing ARPF in Bar Plot

```python
```python
metrics_train = [acc_train, prec_train, rec_train, f1_train]
metrics_test  = [acc_test,  prec_test,  rec_test,  f1_test]
labels = ['Accuracy','Precision','Recall','F1']

x = np.arange(len(labels))

plt.figure(figsize=(8,5))
plt.bar(x - 0.2, metrics_train,  width=0.4, label='Train')
plt.bar(x + 0.2, metrics_test,   width=0.4, label='Test')
plt.xticks(x, labels)
plt.ylim([0,1])
plt.legend()
plt.title('ARPF metrics for MNIST (Weighted Avg)')
plt.show()
```
```

When **all** four metrics are **similar**, the model is less biased across classes.

### 6.2. Per-Class Precision and Recall

To see **class-wise** metrics, do:

```python
```python
prec_each_class = skm.precision_score(y_test_np, test_preds_np, average=None)
rec_each_class  = skm.recall_score(y_test_np, test_preds_np, average=None)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.bar(range(10), prec_each_class, color='g', alpha=0.6)
plt.xticks(range(10))
plt.ylim([0,1])
plt.title('Precision per digit')

plt.subplot(1,2,2)
plt.bar(range(10), rec_each_class, color='b', alpha=0.6)
plt.xticks(range(10))
plt.ylim([0,1])
plt.title('Recall per digit')

plt.tight_layout()
plt.show()

print("Precision:", prec_each_class)
print("Recall:   ", rec_each_class)
```
```

Observe how certain digits (e.g., **9** or **3**) might have **noticeably** lower precision or recall, indicating a **bias** in predictions for those classes.

---

## 7. Confusion Matrix

With **10 classes**, the confusion matrix is **10×10**. We still want **strong diagonals** (correct classifications) and small off-diagonal elements:

```python
```python
cm_train = skm.confusion_matrix(y_train_np, train_preds_np)
cm_test  = skm.confusion_matrix(y_test_np,  test_preds_np)

fig, ax = plt.subplots(1,2, figsize=(12,5))
skm.ConfusionMatrixDisplay(cm_train, display_labels=range(10)).plot(ax=ax[0], cmap=plt.cm.Blues, colorbar=False)
ax[0].set_title('Train Confusion Matrix')

skm.ConfusionMatrixDisplay(cm_test, display_labels=range(10)).plot(ax=ax[1], cmap=plt.cm.Blues, colorbar=False)
ax[1].set_title('Test Confusion Matrix')

plt.tight_layout()
plt.show()
```
```

- Rows = **true** digits.  
- Columns = **predicted** digits.  
- Each cell \((i,j)\) indicates how many **class i** samples were **predicted as class j**.  

**Interpretation**:
- Large diagonal numbers → correct classification.  
- Off-diagonal elements → misclassifications (e.g., `3 → 5`).  
- Some digits may be frequently confused with each other (e.g., `9 -> 4` or `2 -> 7`).

---

## 8. Observations and Conclusions

1. **High-Level Metrics**: ~92–95% accuracy, with near-identical precision/recall/F1, suggesting a **well-trained** model on MNIST.  
2. **Class-Level Performance**: Some digits show **lower** precision/recall, indicating bias toward or against certain digits.  
3. **Confusion Matrix**: Reveals which **pairs** of digits the model confuses most (e.g., 3 vs. 5, 4 vs. 9).  
4. **Average Types**: *Macro* vs. *Weighted* vs. *None*—multiple ways to **aggregate** multi-class metrics.

---

## 9. Summary

- **MNIST** is a multi-class example where ARPF must be computed **per class** or in an **averaged** fashion.
- **Accuracy** remains a quick measure, but it **doesn’t** reveal **per-digit biases**.
- **Precision** and **Recall** can **vary** significantly across digits, so a single F1 or precision/recall number may hide imbalances.
- The **confusion matrix** is invaluable for **visualizing** which classes are commonly misclassified as others.

---

## 10. References and Further Reading

- **MNIST dataset**: [Yann LeCun’s MNIST page](http://yann.lecun.com/exdb/mnist/)  
- **PyTorch MNIST Example**: [PyTorch Tutorials](https://pytorch.org/tutorials/)  
- **Scikit-learn Metrics**: [sklearn.metrics documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)  
- **Signal Detection Theory**: Green & Swets (1966), *Signal Detection Theory and Psychophysics*.

---

```

**How to Use These Notes in Obsidian**:

1. **Create a new note** in your vault (e.g., `APRF_MNIST_Notes.md`).  
2. **Paste** this entire markdown (including the frontmatter `---`).  
3. Customize or link to related notes (e.g., `[[Wine Quality Example]]`, `[[Confusion Matrix Explanation]]`).  

These notes demonstrate **multi-class ARPF metrics** on MNIST, illustrating how **per-class** performance can vary even when **overall accuracy** is high.