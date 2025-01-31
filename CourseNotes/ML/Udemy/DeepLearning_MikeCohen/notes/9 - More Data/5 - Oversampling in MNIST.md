aliases: [Oversampling, MNIST, Data Augmentation, Unbalanced, Double Sampling]
tags: [Data, Deep Learning, PyTorch, Experiment, Classification]
## Overview
In this lecture, we illustrate **oversampling** by **doubling** certain training samples. While MNIST is *not* naturally unbalanced, this exercise shows how oversampling affects training outcomes—and warns about potential pitfalls when **train** and **test** splits can accidentally share duplicates.

We will:
1. Take a **subset** of MNIST data (e.g., 500 to 4000 samples).
2. Train one model on **unique** samples, another on **doubled** samples.
3. Compare performance, discussing **why** oversampling can inflate accuracy and how to properly validate.

---
## 1. Motivational Example

### 1.1 Small Dataset Hypothesis
Imagine you have only **4** MNIST images. If you “oversample” (double) the data, you get **8** total samples, but **no new** information. The network might *appear* to see more data, but it can just memorize the duplicates.

### 1.2 Inadvertent Data Leakage
If you double the **entire** dataset (train+test) *before* splitting, you risk having **identical copies** of an image in **both** train and test sets. This invalidates the test accuracy (the model effectively sees the test images during training).

---

## 2. Experimental Setup

Below, we outline the major steps:
1. **Load partial MNIST** (~20k images if using `mnist_*_small.csv` from Google Colab).
2. Define a function to **slice** the first \(N\) samples & optionally **double** them.
3. Build a feedforward model, train with small subsets of MNIST, compare “Single” vs. “Double” sampling.

---

### 2.1 Data Loading & Slicing

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_mnist_dataset(nSamples, doubleTheData=False):
    """
    1) Loads partial MNIST from Colab
    2) Takes the first nSamples
    3) Optionally doubles (concatenates) the data
    4) Splits into train/test DataLoaders (90%/10%)
    """
    # Load CSV (small MNIST)
    train_path  = "/content/sample_data/mnist_train_small.csv"
    data_np = np.loadtxt(train_path, delimiter=",")
    
    labels = data_np[:, 0].astype(int)
    pixels = data_np[:, 1:].astype(float)
    
    # Restrict to first nSamples
    pixels = pixels[:nSamples]
    labels = labels[:nSamples]
    
    # Normalize
    pixels /= 255.0
    
    # Optionally double the entire set
    # -- approach 1: double data BEFORE train/test split
    if doubleTheData:
        pixels = np.concatenate((pixels, pixels), axis=0)
        labels = np.concatenate((labels, labels), axis=0)
    
    # Convert to tensors
    X_t = torch.tensor(pixels, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.long)
    
    # Alternatively, double only the training set
    # (commented out by default)
    # you can handle this logic after splitting if desired
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_t, y_t, test_size=0.10, random_state=42
    )
    
    # -- approach 2 (uncomment if you want to double ONLY the train set):
    # if doubleTheData:
    #     X_train = torch.cat([X_train, X_train], dim=0)
    #     y_train = torch.cat([y_train, y_train], dim=0)
    
    # Create Datasets & Loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test,  y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    return train_loader, test_loader
```

**Key**: 
- **`doubleTheData=True`** doubles the entire slice of data *before* the split (approach 1). 
- Alternatively, you could **double only the training set** (approach 2 in comments).

#### 2.1.1 Quick Sanity Check

```python
# Just test with small nSamples=200
trainL, testL = create_mnist_dataset(200, doubleTheData=False)
print("Unique approach: Train size =", len(trainL.dataset), "Test size =", len(testL.dataset))

trainL2, testL2 = create_mnist_dataset(200, doubleTheData=True)
print("Doubled approach: Train size =", len(trainL2.dataset), "Test size =", len(testL2.dataset))
```

- We might see something like **Train=180, Test=20** initially, and then after doubling: **Train=360, Test=40**.

---

### 2.2 Feedforward Model & Training Function

Define a **2-layer** feedforward network for MNIST:

```python
class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: [batch_size, 784]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x  # raw logits

def train_model(train_loader, test_loader, epochs=50):
    model = SimpleMNISTNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    train_accs, test_accs = [], []
    train_losses, test_losses = [], []
    
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
            acc   = (preds == yb).float().mean().item()
            batch_accs.append(acc)
        train_losses.append(np.mean(batch_losses))
        train_accs.append(np.mean(batch_accs))
        
        # Evaluate
        model.eval()
        batch_losses_test, batch_accs_test = [], []
        with torch.no_grad():
            for Xb_t, yb_t in test_loader:
                logits_t = model(Xb_t)
                loss_t   = criterion(logits_t, yb_t)
                batch_losses_test.append(loss_t.item())
                
                preds_t = torch.argmax(logits_t, dim=1)
                acc_t   = (preds_t == yb_t).float().mean().item()
                batch_accs_test.append(acc_t)
        test_losses.append(np.mean(batch_losses_test))
        test_accs.append(np.mean(batch_accs_test))
    
    return model, (train_losses, test_losses, train_accs, test_accs)
```

---

## 3. Running a Preliminary Test

```python
# Try nSamples=5000, doubleTheData=False
train_loader, test_loader = create_mnist_dataset(5000, doubleTheData=False)
model, stats = train_model(train_loader, test_loader, epochs=50)
train_losses, test_losses, train_accs, test_accs = stats

print("Final train acc:", train_accs[-1])
print("Final test  acc:", test_accs[-1])
```

- We might see ~**90–92%** test accuracy with only 5000 MNIST samples.

---

## 4. The Main Experiment

We vary **nSamples** in \([500, 1000, 1500, ..., 4000]\), and for each:
1. Train once with **unique** data
2. Train once with **double** data
3. Log final performance

```python
sample_sizes = range(500, 4501, 500)
results_single = np.zeros((len(sample_sizes), 3))  # columns: [loss, train_acc, test_acc]
results_double = np.zeros((len(sample_sizes), 3))

for i, nSamp in enumerate(sample_sizes):
    # (1) Single
    trainL, testL = create_mnist_dataset(nSamp, doubleTheData=False)
    _, stats = train_model(trainL, testL, epochs=50)
    tr_losses, te_losses, tr_accs, te_accs = stats
    results_single[i,0] = np.mean(tr_losses[-5:])  # average last 5 epochs
    results_single[i,1] = np.mean(tr_accs[-5:])
    results_single[i,2] = np.mean(te_accs[-5:])
    
    # (2) Double
    trainLd, testLd = create_mnist_dataset(nSamp, doubleTheData=True)
    _, statsd = train_model(trainLd, testLd, epochs=50)
    tr_losses_d, te_losses_d, tr_accs_d, te_accs_d = statsd
    results_double[i,0] = np.mean(tr_losses_d[-5:])
    results_double[i,1] = np.mean(tr_accs_d[-5:])
    results_double[i,2] = np.mean(te_accs_d[-5:])

print("Experiment complete!")
```

**Important**: With **doubleTheData=True** in the above approach, the doubling is done *before* the train/test split, so **duplicates** can appear in both sets. We’ll see how that affects results.

---

## 5. Visualizing Results

We’ll plot **training** and **test** accuracy for both “Single” and “Double” conditions.

```python
x_vals = list(sample_sizes)

# Plot training accuracy
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(x_vals, results_single[:,1], 'o-', label='Train Acc (Single)')
plt.plot(x_vals, results_double[:,1], 'o-', label='Train Acc (Double)')
plt.xlabel("Number of unique samples")
plt.ylabel("Train Accuracy (avg last 5 epochs)")
plt.legend()

# Plot test accuracy
plt.subplot(1,2,2)
plt.plot(x_vals, results_single[:,2], 'o-', label='Test Acc (Single)')
plt.plot(x_vals, results_double[:,2], 'o-', label='Test Acc (Double)')
plt.xlabel("Number of unique samples")
plt.ylabel("Test Accuracy (avg last 5 epochs)")
plt.legend()

plt.tight_layout()
plt.show()
```

**Note**: 
- **x-axis** is the number of unique samples in the slice.  
- For “Double,” the network effectively sees `2 * nSamples` training examples, but duplicates might **leak** into the test set if we do not carefully handle splitting first.

---

## 6. Discussion & Observations

### 6.1 Suspiciously High Accuracy?
- If the **test** accuracy line for **Double** is significantly above the **Single** line (especially approaching **~100%**), you should suspect **data leakage** (some repeated samples in both train & test).

### 6.2 Proper Approach
- Often, you want to **split** first, then **only** oversample the training set. This ensures test set remains a **clean** hold-out of unique samples.  
- E.g., in `create_mnist_dataset`, you might:
  ```python
  # after splitting:
  if doubleTheData:
      X_train = torch.cat((X_train, X_train))
      y_train = torch.cat((y_train, y_train))
  ```

### 6.3 When Is Oversampling Useful?
- If your dataset is **small** or **unbalanced** in certain classes, oversampling the minority classes (or the entire dataset) can help.  
- However, **pure duplication** doesn’t add new information—just re-exposes the same samples more often.

### 6.4 Overfitting Risks
- The model can memorize repeated samples and become **overconfident** during training.  
- Performance on a proper **unique** test set might be unaffected or show only modest gains.

---

## 7. Key Takeaways

1. **Oversampling**: Straightforward technique where you replicate some or all samples to “increase” your dataset size.  
2. **Data Leakage**: If duplication happens *before* train/test splitting, you can artificially inflate test accuracy.  
3. **Practical Advice**:  
   - Always do **train/test split first**.  
   - Oversample **only** the training set.  
   - Check that the test set contains *no duplicates* from the training set.  
4. **Real Gains?**: Oversampling might help when data is extremely limited or highly imbalanced, but it usually does **not** create new information—only new illusions of sample size.

**End of Notes – "Data: Oversampling in MNIST"**  
```