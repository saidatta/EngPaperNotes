aliases: [Noise Augmentation, Oversampling, MNIST, Data Regularization]
tags: [Data, Augmentation, Deep Learning, PyTorch, Experimentation]
## Overview
This lecture extends the idea of **oversampling** from duplicating data into **creating noisy copies** of images—providing a form of **data augmentation**. Instead of exact duplicates, each repeated sample is *slightly perturbed* by random noise, helping the network generalize better (less overfitting). We also introduce a true **independent test set** to avoid data leakage.

---
## 1. Motivation

### 1.1 From Exact Copies to Noisy Copies
- In standard oversampling, we **duplicate** certain samples, leading to potential overfitting or data leakage.  
- By adding **noise** to duplicates, we reduce the risk of memorization because each new sample, though correlated with its original, is **not exactly identical**.

### 1.2 Benefits
- **Regularization**: Noisy augmentation forces the model to learn robust features.  
- **Effective Data Increase**: For small sample sizes, noise-augmented data can significantly boost performance.

### 1.3 Caveat
- If noise is too large or unrealistic, we risk shifting samples out of their original class distribution.

---

## 2. Data Preparation & Noise Augmentation

Below is a modified version of our MNIST slicing code. We:
1. Take **\(N\) samples** from a partial MNIST dataset.
2. (Optionally) **augment** them by adding random noise in \([0, 0.5]\).
3. Split into **train** and **dev** sets (plus a separate **test** set that’s truly untouched).

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_mnist_dataset_noise(
    nSamples, 
    noiseAugment=False, 
    train_split=0.9
):
    """
    1) Loads partial MNIST from CSV (e.g., mnist_train_small.csv)
    2) Selects the first nSamples
    3) Optionally adds noisy copies of the data (uniform noise [0,0.5])
    4) Splits into (train+dev) and separate large test set (the rest of the data)
    
    Returns:
      train_loader, dev_loader, (X_test_t, y_test_t)
    """
    # (A) Load the full partial MNIST
    full_path = "/content/sample_data/mnist_train_small.csv"
    data_full = np.loadtxt(full_path, delimiter=",")
    
    # Unpack
    labels_full = data_full[:, 0].astype(int)
    pixels_full = data_full[:, 1:].astype(float)
    
    # Normalize to [0,1]
    pixels_full /= 255.0
    
    # (B) Prepare the subset for train/dev
    #     first nSamples => potential training pool
    X_sub = pixels_full[:nSamples]
    y_sub = labels_full[:nSamples]
    
    # If noiseAugment=True, we create a noisy copy
    if noiseAugment:
        noise = np.random.rand(*X_sub.shape) / 2.0  # uniform[0,0.5]
        X_noisy = X_sub + noise
        # optionally, clamp or re-normalize to [0,1] if desired
        # X_noisy = np.clip(X_noisy, 0, 1)
        
        # Concatenate
        X_sub = np.concatenate([X_sub, X_noisy], axis=0)
        y_sub = np.concatenate([y_sub, y_sub], axis=0)
    
    # Convert to tensors
    X_sub_t = torch.tensor(X_sub, dtype=torch.float32)
    y_sub_t = torch.tensor(y_sub, dtype=torch.long)
    
    # (C) The REMAINING images serve as a truly independent test set
    X_test = pixels_full[nSamples:]
    y_test = labels_full[nSamples:]
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # (D) Now split our subset into train+dev
    train_idx, dev_idx = train_test_split(
        np.arange(len(y_sub_t)), test_size=(1-train_split), random_state=42
    )
    
    X_train_t = X_sub_t[train_idx]
    y_train_t = y_sub_t[train_idx]
    X_dev_t   = X_sub_t[dev_idx]
    y_dev_t   = y_sub_t[dev_idx]
    
    # Create Datasets & DataLoaders
    train_data = TensorDataset(X_train_t, y_train_t)
    dev_data   = TensorDataset(X_dev_t,   y_dev_t)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  drop_last=True)
    dev_loader   = DataLoader(dev_data,   batch_size=32, shuffle=False, drop_last=True)
    
    return train_loader, dev_loader, (X_test_t, y_test_t)
```

---

### 2.1 Visualizing Original vs. Noisy Images

Let’s pick a few samples to see how noise transforms them:

```python
def show_original_noisy(X, n=6):
    indices = np.random.choice(len(X), n, replace=False)
    plt.figure(figsize=(10,2))
    for i, idx in enumerate(indices):
        plt.subplot(1,n,i+1)
        img = X[idx].reshape(28,28)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example: get 10 original samples, make 10 noisy copies
nSamples = 10
pixels_sample = np.random.rand(nSamples, 784)  # dummy small random for demonstration
noise       = np.random.rand(nSamples, 784)/2
pixels_noisy = pixels_sample + noise

# Show side by side
show_original_noisy(pixels_sample, n=5)   # original random images
show_original_noisy(pixels_noisy, n=5)    # noisy versions
```

*(In the actual MNIST code, we’d display real digit images, but the concept is the same.)*

---

## 3. Model Architecture & Training

We can reuse a simple feedforward net:

```python
class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x  # raw logits

def train_model(train_loader, dev_loader, epochs=50):
    model = SimpleMNISTNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    train_accs, dev_accs = [], []
    
    for epoch in range(epochs):
        ########################
        # Train
        ########################
        model.train()
        batch_accs = []
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(logits, 1)
            acc   = (preds == yb).float().mean().item()
            batch_accs.append(acc)
        train_accs.append(np.mean(batch_accs))
        
        ########################
        # Dev
        ########################
        model.eval()
        batch_accs_dev = []
        with torch.no_grad():
            for Xb_d, yb_d in dev_loader:
                logits_d = model(Xb_d)
                preds_d  = torch.argmax(logits_d, 1)
                acc_d    = (preds_d == yb_d).float().mean().item()
                batch_accs_dev.append(acc_d)
        dev_accs.append(np.mean(batch_accs_dev))
    
    return model, (train_accs, dev_accs)
```

---

## 4. Main Experiment: Original vs. Noise-Augmented Data

We vary **\(N\)**, the number of unique samples used in training/dev, from **500** to **4,000**. For each \(N\):

1. Create dataset **without** augmentation (`noiseAugment=False`).  
2. Create dataset **with** augmentation (`noiseAugment=True`).  
3. Train each for 50 epochs, log final accuracies on dev set.
4. Evaluate both on a truly **independent** large test set from the remainder of MNIST.

```python
sample_sizes = [500, 1000, 1500, 2000, 2500, 3000, 4000]
results_orig  = []
results_noise = []

for nSamp in sample_sizes:
    # Original
    tr_loader, dv_loader, (X_test, y_test) = create_mnist_dataset_noise(nSamp, noiseAugment=False)
    mod_o, accs_o = train_model(tr_loader, dv_loader, epochs=50)
    tr_accs_o, dv_accs_o = accs_o
    
    # Evaluate on test
    mod_o.eval()
    preds_test_o = mod_o(X_test)
    preds_o      = torch.argmax(preds_test_o, 1)
    test_acc_o   = (preds_o == y_test).float().mean().item()
    
    results_orig.append((nSamp, tr_accs_o[-1], dv_accs_o[-1], test_acc_o))
    
    # Noise-augmented
    tr_loader_N, dv_loader_N, (X_test_N, y_test_N) = create_mnist_dataset_noise(nSamp, noiseAugment=True)
    mod_n, accs_n = train_model(tr_loader_N, dv_loader_N, epochs=50)
    tr_accs_n, dv_accs_n = accs_n
    
    # Evaluate on test
    mod_n.eval()
    preds_test_n = mod_n(X_test_N)
    preds_n      = torch.argmax(preds_test_n, 1)
    test_acc_n   = (preds_n == y_test_N).float().mean().item()
    
    results_noise.append((nSamp, tr_accs_n[-1], dv_accs_n[-1], test_acc_n))

print("Experiment complete!")
```

---

## 5. Plotting the Results

We’ll plot lines for **Original** vs. **NoiseAug** in **train**, **dev**, and **test** performance.

```python
res_o = np.array(results_orig)
res_n = np.array(results_noise)

x_vals = res_o[:,0]  # sample sizes

plt.figure(figsize=(14,4))

# 5.1 Train Accuracy
plt.subplot(1,3,1)
plt.plot(x_vals, res_o[:,1], 'o--', label='Original')
plt.plot(x_vals, res_n[:,1], 'o--', label='NoiseAug')
plt.xlabel("Number of Unique Samples")
plt.ylabel("Train Accuracy (final epoch)")
plt.title("Train Accuracy")
plt.legend()

# 5.2 Dev Accuracy
plt.subplot(1,3,2)
plt.plot(x_vals, res_o[:,2], 'o--', label='Original')
plt.plot(x_vals, res_n[:,2], 'o--', label='NoiseAug')
plt.xlabel("Number of Unique Samples")
plt.ylabel("Dev Accuracy (final epoch)")
plt.title("Dev Accuracy")
plt.legend()

# 5.3 Test Accuracy
plt.subplot(1,3,3)
plt.plot(x_vals, res_o[:,3], 'o--', label='Original')
plt.plot(x_vals, res_n[:,3], 'o--', label='NoiseAug')
plt.xlabel("Number of Unique Samples")
plt.ylabel("Test Accuracy")
plt.title("Independent Test Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
```

**Interpretation**:

- **Train**: Noise augmentation may push training accuracy higher (or occasionally lower) depending on sample size.  
- **Dev**: For smaller subsets, noisy copies can significantly raise dev accuracy.  
- **Test**: Demonstrates the real effect—if test accuracy is also higher, noise augmentation genuinely helps model generalization.

---

## 6. Detailed Example: \(N = 500\)

To confirm the effect, let’s run a final example at **\(N=500\)**:

```python
nSamp = 500

# Original
trL, dvL, (X_ts, y_ts) = create_mnist_dataset_noise(nSamp, noiseAugment=False)
net_o, (trA_o, dvA_o)  = train_model(trL, dvL, epochs=50)

# NoiseAug
trL_n, dvL_n, (X_ts_n, y_ts_n) = create_mnist_dataset_noise(nSamp, noiseAugment=True)
net_n, (trA_n, dvA_n)          = train_model(trL_n, dvL_n, epochs=50)

# Evaluate on dev
final_tr_o, final_dev_o = trA_o[-1], dvA_o[-1]
final_tr_n, final_dev_n = trA_n[-1], dvA_n[-1]

# Evaluate on independent test
def test_accuracy(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        preds_arg = torch.argmax(preds, dim=1)
        acc = (preds_arg == y_test).float().mean().item()
    return acc

test_acc_o = test_accuracy(net_o, X_ts, y_ts)
test_acc_n = test_accuracy(net_n, X_ts_n, y_ts_n)

print(f"\nWithout noise:\n Train={final_tr_o*100:.1f}% Dev={final_dev_o*100:.1f}%, Test={test_acc_o*100:.1f}%")
print(f"With noise:\n Train={final_tr_n*100:.1f}% Dev={final_dev_n*100:.1f}%, Test={test_acc_n*100:.1f}%")
```

- Often you’ll see **With noise** has much higher train & dev accuracy, and also a genuinely improved **test** accuracy. 
- This indicates the augmentation *really did* help the model learn more robust features (not just memorize duplicates).

---

## 7. Key Takeaways

1. **Noisy Oversampling** vs. Exact Duplication  
   - Adding random noise to repeated samples prevents trivial memorization and acts like a **data regularizer**.
2. **True Independent Test Set**  
   - Splitting out a large chunk of data from the start ensures no augmented duplicates contaminate test.  
   - Dev set might still partially overlap or see mild inflation, but test set remains pure.
3. **Practical Gains**  
   - Especially helpful for **small** training subsets.  
   - For larger sample sizes, the effect diminishes because the original data already provides enough variety.
4. **Implementation**  
   - Often done via **transforms** in PyTorch (`torchvision.transforms` or custom augmentations).

---

## 8. Further Exploration

- **Different Noise Distributions**: Instead of uniform [0,0.5], try **Gaussian** \(\mathcal{N}(0,\sigma^2)\) or other distributions.  
- **Clamp or Re-normalize**: Ensure pixel intensities remain in [0,1] if that’s crucial to your model.  
- **Partial Augmentation**: Only noise-augment a fraction of images to see if that balances best.  
- **Advanced Image Aug**: Rotations, flips, crops, color jitter—common in CNN contexts (we’ll revisit in CNN lectures).

---

**End of Notes – “Data: Noise Augmentation in MNIST (Separate Dev+Test)”**  
```