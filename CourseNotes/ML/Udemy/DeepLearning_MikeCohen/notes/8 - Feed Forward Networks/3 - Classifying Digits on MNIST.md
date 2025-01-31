aliases: [FFN, MNIST, Classification, Softmax, LogSoftmax, Neural Networks]
tags: [Deep Learning, PyTorch, Feedforward, Classification, MNIST]

## Overview
In this lecture, we implement a **feedforward neural network** (FFN) to classify **MNIST digits** (0–9). The instructor demonstrates that even a **relatively simple** network with two hidden layers can achieve around **95% accuracy** on a **partial MNIST dataset** (only ~20,000 images instead of the full 70k). 

We will cover:
1. **Network architecture**: 784 → 64 → 32 → 10
2. **Softmax vs. LogSoftmax** for multi-class problems
3. **Implementation details**: data loading, normalization, model creation, training loop
4. **Evaluation**: accuracy, confusion analysis, interpreting the model's mistakes

---

## 1. Network Architecture

### 1.1 Input Layer (784)
- Each MNIST image is \(28 \times 28 = 784\) pixels.
- In feedforward networks, these images are **flattened** into a 784-dimensional vector per image.

### 1.2 Hidden Layers
- First hidden layer: **64** neurons
- Second hidden layer: **32** neurons
- **Activation**: ReLU (Rectified Linear Unit) after each hidden layer.

> **Note**: The choice of 64 and 32 hidden units is somewhat arbitrary; we can tweak these hyperparameters.

### 1.3 Output Layer (10)
- We have **10** output units corresponding to the 10 digit classes (0 through 9).
- Outputs typically go through a **Softmax** to produce a probability distribution over the 10 possible digits.

---

## 2. Softmax vs. LogSoftmax

### 2.1 Why LogSoftmax?
- **LogSoftmax** = \(\log(\text{Softmax}(z))\)
  - In practice, we often use `torch.log_softmax(z, dim=...)`.
- Benefits:
  1. **Numerical stability**: Avoids extremely small probability values close to 0.  
  2. **Sharper penalties** for incorrect classifications, which can improve learning when there are **multiple classes**.
- **Traditional Softmax** ("lin-softmax") can still work but is often less stable for larger or multi-class problems (like 10 classes in MNIST).

### 2.2 Negative Log-Likelihood Loss (NLLLoss)
- When using **LogSoftmax**, you typically pair it with **NLLLoss** (negative log-likelihood loss).  
- If you used a raw output layer (no log), you’d use **CrossEntropyLoss**, which internally applies `log_softmax` + `nll_loss`.

> **Side-by-Side Comparison**:  
> - **Linear Softmax** accuracy ~77% (in the instructor’s experiment).  
> - **LogSoftmax** accuracy ~95% (same architecture, same data).

---

## 3. Implementation: Data Loading & Preprocessing

Below is an outline of the steps to load and preprocess the partial MNIST data (here, ~20,000 images).

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. Load data from CSV (partial MNIST: ~20k examples)
train_path = "/content/sample_data/mnist_train_small.csv"
data_np = np.loadtxt(train_path, delimiter=",")

# 2. Separate labels and pixel values
labels_np = data_np[:, 0].astype(int)           # first column -> digit labels
pixels_np = data_np[:, 1:].astype(float)        # remaining 784 columns -> image pixels

# 3. Normalize pixel intensities to [0, 1]
pixels_np /= 255.0

# 4. Create train/test split
X_train, X_test, y_train, y_test = train_test_split(
    pixels_np, labels_np, test_size=0.10, random_state=42
)

# Convert NumPy arrays -> PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 5. Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

print(f"Train set size: {len(train_loader)*batch_size}")
print(f"Test set size:  {len(test_loader)*batch_size}")
```

### 3.1 Data Visualization
Check normalization and distribution:

```python
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(pixels_np.flatten(), bins=50, color="gray")
plt.title("Pixel Intensity Distribution (Linear Scale)")
plt.xlabel("Pixel Value")
plt.ylabel("Count")

plt.subplot(1,2,2)
plt.hist(pixels_np.flatten(), bins=50, log=True, color="gray")
plt.title("Pixel Intensity Distribution (Log Scale)")
plt.xlabel("Pixel Value")
plt.ylabel("Count (log-scale)")
plt.tight_layout()
plt.show()
```

> You’ll see a large peak near 0 (background) and some near 1 (strokes), with a range of intermediate values.

---

## 4. Defining the FFN Model

We define a **PyTorch model** class with two hidden layers (64 and 32 units) and an output of 10 units. In the final layer, we apply **`torch.log_softmax`**. 

```python
class MNIST_FFN(nn.Module):
    def __init__(self):
        super(MNIST_FFN, self).__init__()
        # Layers
        self.fc0 = nn.Linear(784, 64)   # input -> hidden1
        self.fc1 = nn.Linear(64, 32)    # hidden1 -> hidden2
        self.fc2 = nn.Linear(32, 10)    # hidden2 -> output (10 classes)
        
    def forward(self, x):
        # x shape: (batch_size, 784)
        x = self.fc0(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        # Apply LogSoftmax on the output
        # dim=1 means we apply softmax across the "class" dimension
        return torch.log_softmax(x, dim=1)

model = MNIST_FFN()
print(model)
```

**Model architecture**:
1. **fc0**: \(784 \to 64\)
2. **fc1**: \(64 \to 32\)
3. **fc2**: \(32 \to 10\)
4. Output → **LogSoftmax**

---

## 5. Choosing an Optimizer & Loss Function

### 5.1 Loss Function
Since we apply **log_softmax** in the forward pass, we use **`nn.NLLLoss()`**:

```python
criterion = nn.NLLLoss()
```

### 5.2 Optimizer
For demonstration purposes, the instructor uses **basic SGD** (`optim.SGD`) to slow the learning process and highlight differences. You can experiment with **Adam** or **RMSprop** later.

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

---

## 6. Training Loop

Below is a standard PyTorch training loop:

```python
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=60):
    train_losses, test_accuracies, train_accuracies = [], [], []
    
    for epoch in range(epochs):
        model.train()
        batch_losses, batch_accs = [], []
        
        for X_batch, y_batch in train_loader:
            # 1) Forward pass
            y_pred_log_probs = model(X_batch)
            loss = criterion(y_pred_log_probs, y_batch)
            
            # 2) Zero gradients
            optimizer.zero_grad()
            
            # 3) Backprop
            loss.backward()
            
            # 4) Update parameters
            optimizer.step()
            
            # Compute accuracy in this batch
            _, predicted_labels = torch.max(y_pred_log_probs, 1)
            acc = (predicted_labels == y_batch).float().mean()
            
            batch_losses.append(loss.item())
            batch_accs.append(acc.item())
        
        # End of epoch: compute average train loss & accuracy
        train_loss = np.mean(batch_losses)
        train_acc = np.mean(batch_accs)
        
        # Evaluate on the test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch_test, y_batch_test in test_loader:
                y_pred_test_log = model(X_batch_test)
                _, predicted_test_labels = torch.max(y_pred_test_log, 1)
                correct += (predicted_test_labels == y_batch_test).sum().item()
                total += len(y_batch_test)
        test_acc = correct / total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Test Acc: {test_acc:.4f}")
            
    return train_losses, train_accuracies, test_accuracies

# Instantiate and train
model = MNIST_FFN()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_losses, train_accs, test_accs = train_model(
    model, train_loader, test_loader, criterion, optimizer, epochs=60
)
```

> Training might take about **1–2 minutes** (CPU) depending on your setup.

### 6.1 Plotting Loss & Accuracy

```python
epochs_range = range(1, 61)

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, train_accs, label='Train Accuracy')
plt.plot(epochs_range, test_accs, label='Test Accuracy')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
```

- Expect **training loss** to decrease and **accuracy** (train & test) to rise.  
- Final **test accuracy** ~**95%** with the partial dataset.

---

## 7. Evaluation & Model Interpretation

### 7.1 Inspect Model Outputs

After training, we can pass a **batch** from the test set to see how confident the model is:

```python
X_batch_test, y_batch_test = next(iter(test_loader))
y_pred_log_probs = model(X_batch_test)

print("Log probabilities shape:", y_pred_log_probs.shape)  # [batch_size, 10]

# Convert log probabilities to actual probabilities
y_pred_probs = torch.exp(y_pred_log_probs)
print(y_pred_probs[0])  # Probability distribution for the first item in the batch
```

> Most entries might be near 0, with one class near 1 if the model is confident.

### 7.2 Visualizing Predicted Probabilities

Pick a random sample from the batch to see its predicted distribution:

```python
sample_idx = 10  # pick an index in [0, batch_size-1]

log_probs_sample = y_pred_log_probs[sample_idx].detach()
probs_sample = torch.exp(log_probs_sample)
predicted_label = torch.argmax(probs_sample).item()
true_label = y_batch_test[sample_idx].item()

plt.bar(range(10), probs_sample.numpy(), color='steelblue')
plt.title(f"True label: {true_label}, Predicted: {predicted_label}")
plt.xlabel("Digit Class")
plt.ylabel("Probability")
plt.show()
```

### 7.3 Inspecting Model Errors

```python
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        log_probs = model(Xb)
        _, preds = torch.max(log_probs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Find misclassified indices
errors = np.where(all_preds != all_labels)[0]
print("Number of misclassifications:", len(errors))

# Examine a specific error
error_idx = errors[4]  # just an example
print(f"Error index in overall test set: {error_idx}")

# Visualize the misclassified digit
X_mis = X_test[error_idx]  # recall X_test is a np array from train_test_split
label_mis = y_test[error_idx]
pred_mis = all_preds[error_idx]

plt.imshow(X_mis.reshape(28,28), cmap='gray')
plt.title(f"True label: {label_mis}, Predicted: {pred_mis}")
plt.axis('off')
plt.show()
```

You can loop over several misclassifications to see *why* the network might get confused. Some digits are ambiguous even to humans!

---

## 8. Key Takeaways

1. **Architecture**: A straightforward FFN (784 → 64 → 32 → 10) can already achieve ~95% on partial MNIST.
2. **LogSoftmax & NLLLoss**: Crucial for numerical stability and better convergence in multi-class tasks (10 classes).
3. **Data Normalization**: Scaling pixels \([0, 255]\) → \([0, 1]\) aids training.
4. **Inspecting Mistakes**: Always review errors to gain insight into your model’s limitations.

---

## 9. Next Steps
- Experiment with a **deeper or wider** network to see if accuracy improves.
- Switch from **SGD** to **Adam** or **RMSProp** to compare training speed and final performance.
- Try **Dropout** or **Batch Normalization** to see their effects on overfitting and generalization.
- Finally, move on to **Convolutional Neural Networks (CNNs)**, which usually surpass feedforward nets on image data.

---

## 10. References & Further Reading
- **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press.  
- [PyTorch Tutorials](https://pytorch.org/tutorials/): Detailed guides on building neural networks.  
- [Yann LeCun’s MNIST Page](http://yann.lecun.com/exdb/mnist/): Original dataset details and benchmark results.

---

```markdown
**End of Notes – "Feedforward Networks: FFN to Classify Digits"**
```