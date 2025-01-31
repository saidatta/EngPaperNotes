aliases: [Code Challenge, Binarized MNIST, Feedforward Network, Data Normalization]
tags: [Deep Learning, MNIST, Code Challenge, Classification]
## Overview
In this challenge, we explore the **impact of data binarization** on FFN classification performance on the MNIST dataset. Instead of using continuous pixel intensity values \([0,1]\), we transform each pixel into a **binary** value (0 or 1). We’ll apply the **same feedforward network** architecture and training regime used previously (e.g., ~95% accuracy on the partial dataset) and see if binarizing the pixels affects performance.

---
## 1. The Binarization Concept

### 1.1 What Does Binarization Mean?
- In typical MNIST preprocessing, each pixel is scaled from \([0,255]\) to \([0,1]\), yielding a **continuous** range.
- In **binarized** MNIST, **every pixel** is forced to be either **0** or **1**:
  - 0 → indicates no "ink" (background).
  - 1 → indicates some "ink" (foreground).

### 1.2 Why Binarize?
- **Simplicity**: Binarized images reduce the grayscale detail. The network sees only two possible intensity values.
- **Curiosity**: Does the network still learn digit shapes effectively without subtle gradients of intensity?

---

## 2. Setup & Data Loading

Below is the **outline** of the code. We load the same partial MNIST dataset (~20k samples) from CSV:

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. Load the CSV data
data_path = "/content/sample_data/mnist_train_small.csv"  # ~20k images
data_np = np.loadtxt(data_path, delimiter=",")

# 2. Separate labels and pixel values
labels_np = data_np[:, 0].astype(int)
pixels_np = data_np[:, 1:].astype(float)

print("Original pixel range:", pixels_np.min(), "to", pixels_np.max())
# e.g., 0 to 255
```

---

## 3. Binarizing the Data

### 3.1 Thresholding
One simple approach:
- `pixels_np > 0`  
  This yields a boolean array: `True` if pixel > 0, `False` if pixel = 0.

Then we convert to float (1.0 for True, 0.0 for False):

```python
# Binarize: anything > 0 becomes 1; else 0
pixels_bin = (pixels_np > 0).astype(float)

print("Binarized pixel range:", np.unique(pixels_bin))
# Should be [0. 1.]
```

> **Other Approaches**:  
> - A small threshold, e.g., `pixels_np > 128`, might yield different binarization.  
> - Experiment with different thresholds if you like.

---

## 4. Preparing Train/Test Sets

We now split the binarized data into training and test partitions:

```python
X_train, X_test, y_train, y_test = train_test_split(
    pixels_bin, labels_np, test_size=0.10, random_state=42
)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

# Create Dataset and DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

print("Train set size:", len(train_loader) * batch_size)
print("Test set size: ", len(test_loader) * batch_size)
```

---

## 5. Visualizing the Binarized Data

Optional but recommended: Inspect a few examples to confirm the pixels are indeed 0 or 1.

```python
def show_sample_images(X, y, num=8):
    indices = np.random.choice(len(X), num, replace=False)
    plt.figure(figsize=(10, 2))
    for i, idx in enumerate(indices):
        img = X[idx].reshape(28, 28)
        plt.subplot(1, num, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {y[idx]}")
        plt.axis('off')
    plt.show()

show_sample_images(X_train, y_train, num=8)
```

You should see **hard-edged**, high-contrast images: all pixels are black (0) or white (1).

---

## 6. Defining the Feedforward Network

We’ll reuse the **same** network from the previous lecture. For completeness:

```python
class MNIST_FFN(nn.Module):
    def __init__(self):
        super(MNIST_FFN, self).__init__()
        self.fc0 = nn.Linear(784, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.fc0(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

model = MNIST_FFN()
```

---

## 7. Training Setup

We keep the **optimizer** (SGD) and **loss function** (NLLLoss) the same. Hyperparameters remain identical, e.g., 60 epochs:

```python
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train_model(model, train_loader, test_loader, epochs=60):
    train_losses, test_accuracies, train_accuracies = [], [], []
    
    for epoch in range(epochs):
        model.train()
        batch_losses, batch_accs = [], []
        
        for Xb, yb in train_loader:
            # Forward
            y_pred_log = model(Xb)
            loss = criterion(y_pred_log, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accuracy in this batch
            _, preds = torch.max(y_pred_log, 1)
            acc = (preds == yb).float().mean()
            
            batch_losses.append(loss.item())
            batch_accs.append(acc.item())
        
        # Train stats
        train_loss = np.mean(batch_losses)
        train_acc  = np.mean(batch_accs)
        
        # Evaluate on test
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for Xb_t, yb_t in test_loader:
                y_pred_log_test = model(Xb_t)
                _, preds_t = torch.max(y_pred_log_test, 1)
                correct += (preds_t == yb_t).sum().item()
                total   += len(yb_t)
        test_acc = correct / total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Test Acc: {test_acc:.4f}")
    
    return train_losses, train_accuracies, test_accuracies

train_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, epochs=60)
```

---

## 8. Results & Analysis

### 8.1 Performance Curves

```python
epochs_range = range(1, 61)

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, train_accs, label="Train Accuracy")
plt.plot(epochs_range, test_accs, label="Test Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
```

**Observation**:
- You’ll likely reach training accuracies near **100%**.
- Test accuracy might stabilize around **94–96%**, similar to when using continuous grayscale in \([0,1]\).

### 8.2 Qualitative Comparison
Comparing to the previous (~95% with continuous [0,1] intensities) result:
- **Binarized** → ~**95%** test accuracy
- **Continuous** → ~**95%** test accuracy

Hence, **binarization** does **not** drastically hurt performance (at least for MNIST with this architecture).

---

## 9. Key Takeaways

1. **Data Range**: Even though binarization discards grayscale details, the **network can still learn** digit classification effectively.
2. **High-Level Structure**: MNIST digits are fairly **robust**; rough shapes suffice for ~95% accuracy with a simple FFN.
3. **Try Different Thresholds**: Another threshold (e.g., 128 in [0,255]) might yield similar results.
4. **Generalization**: In tasks where **fine pixel intensity details** matter, binarization could degrade performance more significantly. But for MNIST digits, the contour alone is highly discriminative.

---

## 10. Next Steps
- Try different binarization thresholds or methods (e.g., Otsu thresholding).
- Switch to **Adam** to see if training converges faster.
- Explore **misclassified** binarized images for insight into the network’s confusion.

---

**End of Notes – “CodeChallenge: Binarized MNIST Images”**
```