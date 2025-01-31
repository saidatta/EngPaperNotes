aliases: [Missing 7, Zero-Shot, MNIST, CodeChallenge, FFN]
tags: [Deep Learning, Classification, MNIST, Experimentation, Visualization]
## Overview
In this challenge, we test how a feedforward network (FFN) **behaves when it encounters a category it has never seen** during training. Specifically:

1. We **exclude** all images of the digit “7” from MNIST during training.  
2. We train the FFN on digits \(\{0,1,2,3,4,5,6,8,9\}\).  
3. After training, we **evaluate** the model on *only* the missing digit “7.”  

The key question: **“How will the model label 7s if it has never been trained on them?”**  

We’ll see it often mistakes “7” for certain digits like “9,” “2,” “3,” etc.

---
## 1. Data Setup

### 1.1 Loading and Normalizing MNIST

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 1) Load partial MNIST (e.g., ~20k samples in CSV)
data_path = "/content/sample_data/mnist_train_small.csv"
data_np = np.loadtxt(data_path, delimiter=",")

# Separate labels and pixel values
labels_np = data_np[:, 0].astype(int)
pixels_np = data_np[:, 1:].astype(float)

# Normalize pixel range [0,255] -> [0,1]
pixels_np /= 255.0

```

### 1.2 Splitting Out the Sevens
We identify and **remove** all sevens from the training data, and keep them for testing only.

```python
# Boolean mask to find where labels == 7
where7 = (labels_np == 7)

# Data without sevens (train data)
data_no7 = pixels_np[~where7]
labels_no7 = labels_np[~where7]

# Data with sevens (test data)
data_7 = pixels_np[where7]
labels_7 = labels_np[where7]

print("Shape no7:", data_no7.shape, labels_no7.shape)
print("Shape 7:  ", data_7.shape,  labels_7.shape)
```

---

## 2. Creating DataLoaders

We only need:
- A **train set** (digits 0,1,2,3,4,5,6,8,9)
- A **test set** (digit 7)

```python
# Convert to PyTorch tensors
X_train_t = torch.tensor(data_no7, dtype=torch.float32)
y_train_t = torch.tensor(labels_no7, dtype=torch.long)

X_test_t  = torch.tensor(data_7, dtype=torch.float32)
y_test_t  = torch.tensor(labels_7, dtype=torch.long)

# Create Dataset objects
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, drop_last=False)
```

Check unique labels in the **train set**:

```python
unique_train_labels = np.unique(labels_no7)
print("Unique digits in train set:", unique_train_labels)
# Expect [0 1 2 3 4 5 6 8 9]
```

Check unique label in the **test set**:

```python
unique_test_labels = np.unique(labels_7)
print("Unique digit in test set:", unique_test_labels)
# Expect [7]
```

---

## 3. Defining the Feedforward Model

We use a simple FFN with two hidden layers. Crucially, we keep **10 output neurons** because the original MNIST had 10 digit classes. We’ll see how the model allocates probabilities for the missing class.

```python
class FFN_Missing7(nn.Module):
    def __init__(self):
        super(FFN_Missing7, self).__init__()
        self.fc0 = nn.Linear(784, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.fc0(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        # use log_softmax for interpretability of probabilities
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)
```

> **Why 10 outputs?**  
> Even though we only train on 9 classes, the network architecture is still set up for 10-digit classification. It simply will not have learned any pattern for digit “7.”

---

## 4. Training the Model

We only measure **training loss/accuracy** since our test set is exclusively “7,” which is an unknown class to the model.

```python
def train_model(model, train_loader, epochs=20):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    
    losses, train_accs = [], []
    for epoch in range(epochs):
        model.train()
        batch_losses, batch_correct, batch_total = [], 0, 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            log_probs = model(Xb)
            loss = criterion(log_probs, yb)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            
            _, preds = torch.max(log_probs, dim=1)
            batch_correct += (preds == yb).sum().item()
            batch_total   += len(yb)
        
        epoch_loss = np.mean(batch_losses)
        epoch_acc  = batch_correct / batch_total
        losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.3f}")
    
    return losses, train_accs

model = FFN_Missing7()
losses, train_accs = train_model(model, train_loader, epochs=30)
```

### 4.1 Training Performance

```python
import matplotlib.pyplot as plt

epochs_range = range(1, 31)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(epochs_range, train_accs, label='Train Accuracy')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

print(f"Final Train Accuracy: {train_accs[-1]*100:.2f}%")
```

> Typically, the **train accuracy** can reach near **100%**, because the model only needs to distinguish the 9 digits it’s given.

---

## 5. Testing on Missing Digit "7"

Now we feed **all the 7s** from `test_loader` to our trained model. We examine how the model assigns **log-probabilities** (or equivalently, probabilities) across the 10 output classes.

### 5.1 Generating Predictions

```python
model.eval()
with torch.no_grad():
    all_test_preds = []
    for Xb_test, _ in test_loader:
        log_probs_test = model(Xb_test)
        preds_test = torch.argmax(log_probs_test, dim=1)
        all_test_preds.extend(preds_test.cpu().numpy())

all_test_preds = np.array(all_test_preds)
```

Since all these test samples are **7** in reality, we can see how the model incorrectly classifies them:

```python
from collections import Counter
count_preds = Counter(all_test_preds)
print("Prediction counts on digit 7:\n", count_preds)
```

> For instance, the model may predominantly predict **“9”** for 7.

### 5.2 Visualizing Some 7 Predictions

```python
# Choose random indices from the 7-dataset
import random

indices = random.sample(range(len(X_test_t)), 6)
plt.figure(figsize=(10,3))

for i, idx in enumerate(indices):
    img_2d = X_test_t[idx].reshape(28,28)
    pred_label = all_test_preds[idx]
    
    plt.subplot(2, 3, i+1)
    plt.imshow(img_2d, cmap='gray')
    plt.title(f"Pred: {pred_label} (true:7)")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

You might see many 7s labeled as **9** or **1**, etc.

---

## 6. Analyzing the Distribution of Predictions

To see **which digits** the model confuses for “7” and **how often**, we can compute normalized counts:

```python
unique_preds, counts = np.unique(all_test_preds, return_counts=True)
total_7 = len(all_test_preds)
proportions = counts / total_7

plt.bar(unique_preds, proportions)
plt.xticks(unique_preds)
plt.xlabel("Predicted Digit")
plt.ylabel("Proportion of all 7s")
plt.title("Distribution of 7 misclassifications")
plt.show()

print("Predictions proportions:", dict(zip(unique_preds, proportions)))
```

> Often you’ll see the model confuses 7 with certain digits (e.g., 9, 2, 1, 3) more than others.

---

## 7. Inspecting Probability Outputs for a Single 7

We can also look at the **log_softmax** outputs for a particular sample. Suppose we pick sample #12 (arbitrary):

```python
sample_idx = 12
img_7 = X_test_t[sample_idx].reshape(28,28)
model.eval()

with torch.no_grad():
    log_probs_7 = model(X_test_t[sample_idx])
    probs_7 = torch.exp(log_probs_7)  # convert log_prob -> prob

pred_digit = torch.argmax(log_probs_7).item()

plt.figure()
plt.bar(range(10), probs_7.numpy(), color='steelblue')
plt.title(f"Probability Distribution for a 7 (Predicted: {pred_digit})")
plt.xlabel("Digit Class")
plt.ylabel("Probability")
plt.show()

plt.imshow(img_7, cmap='gray')
plt.title(f"True 7, Predicted {pred_digit}")
plt.axis('off')
plt.show()
```

- See how the model’s confidence distributes among the 10 classes.  
- Usually, one or two classes might dominate (e.g., 9, 2).

---

## 8. Discussion & Key Takeaways

1. **Zero-Shot Learning Scenario**  
   - The model never saw digit “7,” so it has no direct representation for that class.  
   - Instead, it must **force** the input into one of the known 9 categories.

2. **Likely Misclassifications**  
   - Often, “7” might resemble “9,” “1,” “2,” or “3” in certain stroke patterns.  
   - The exact distribution of misclassifications can vary by weight initialization, architecture, etc.

3. **Implication**  
   - Neural networks **cannot** predict categories they have never seen (in naive settings).  
   - In real-world tasks, **unseen** classes remain a major challenge (this leads to areas like **open-set recognition** and **zero-shot learning**).

4. **Visualization**  
   - Inspecting **individual** misclassifications and **probability distributions** often provides deeper insight than just overall metrics.

---

## 9. Extensions & Ideas

- **Add a New Class**: Attempt to label “7” as an explicit “unknown” class by adjusting the architecture or using alternative techniques.  
- **Compare Different Digits**: Omit a different digit each time (e.g., omit “1” or “9”), see if certain digits are more easily replaced.  
- **Overlapping Classes**: What if we omit 2 digits from training?  
- **Data Augmentation**: If you artificially added some “near 7” images, does it help?

---

**End of Notes – "CodeChallenge: The Mystery of the Missing 7"**  
```