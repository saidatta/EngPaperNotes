## 1. Overview

So far, we’ve **initialized** neural network weights in various ways (Xavier, Kaiming, etc.). The next step is understanding **how** these weights *change* during training. Specifically:

1. **Euclidean (Frobenius) Distance** between weight matrices at successive epochs.  
2. **Condition Number** to measure the “sparsity” or “directional skew” of each layer’s weight matrix over training.

We’ll see how these **linear algebra**-based metrics can shed light on **learning dynamics** across epochs.

---

## 2. Recap: Two Metrics for Weight Changes

### 2.1. Euclidean (Frobenius) Distance

\[
\displaystyle
d \bigl(\mathbf{W}(t), \mathbf{W}(t-1)\bigr) 
\;=\; 
\biggl\|\; \mathbf{W}(t) - \mathbf{W}(t-1)\biggr\|_F
\;=\;
\sqrt{
\sum_{i,j}
\Bigl(
w_{ij}^{(t)} \;-\; w_{ij}^{(t-1)}
\Bigr)^2 
}
\]

- Measures **how much** each layer’s weights move from epoch \(t-1\) to \(t\).  
- **Large** distance = bigger updates (lots of learning).  
- **Small** distance = minimal changes (model saturates or learns slowly).

In **NumPy**, this is `np.linalg.norm(W_t - W_{t-1}, 'fro')`.

### 2.2. Condition Number

\[
\kappa(\mathbf{W}) 
\;=\; 
\frac{\sigma_{\max}(\mathbf{W})}{\sigma_{\min}(\mathbf{W})}
\]

- **Singular Values** (\(\sigma_i\)) come from **Singular Value Decomposition** (SVD).  
- **\(\kappa(\mathbf{W})\)** = ratio of largest to smallest singular value of \(\mathbf{W}\).  
- Large \(\kappa\) often indicates **high directional skew** or “flattening” of the weight matrix (possibly more “sparse”).  
- Can also indicate potential **overfitting** if the matrix grows extremely skewed.

---

## 3. Practical Example in PyTorch (MNIST)

We’ll train a feed-forward MNIST model, but with **slowed** training (small LR, SGD) to highlight weight changes across epochs.

### 3.1. Data Preparation

```python
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# MNIST transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load train/test
train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)
```
```

---

### 3.2. Model Definition

```python
```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)  # raw logits
        return x
```
```

We have 4 **weight-carrying** layers: `fc1`, `fc2`, `fc3`, `out`.

---

### 3.3. Training Function with Weight Tracking

We define a training loop that, at each epoch:

1. Saves each layer’s weight **before** epoch’s mini-batch updates.  
2. Performs standard training across the epoch.  
3. Calculates:
   - **Frobenius norm** between final and initial weight values for that epoch.  
   - **Condition number** of each layer at the epoch’s end.  

```python
```python
def train_and_track_weights(net, train_loader, test_loader, epochs=50, lr=0.005):
    optimizer = optim.SGD(net.parameters(), lr=lr)  # slow learning
    loss_fn   = nn.CrossEntropyLoss()
    
    # We'll store results in arrays: shape (epochs, num_layers)
    #   distance[t, layer_i] = ||W_layer(t) - W_layer(t-1)||_F
    #   condition[t, layer_i] = cond(W_layer(t))
    num_layers = 4
    weightChanges = np.zeros((epochs, num_layers))
    weightConds   = np.zeros((epochs, num_layers))
    
    train_acc_hist = []
    test_acc_hist  = []
    
    for epoch_i in range(epochs):
        # Step A: record initial weights for each layer
        layer_weights_init = []
        with torch.no_grad():
            for name, param in net.named_parameters():
                if 'weight' in name:  # skip biases
                    wcopy = param.detach().cpu().numpy().copy()
                    layer_weights_init.append(wcopy)
        
        # Step B: TRAIN over mini-batches
        net.train()
        correct_train = 0
        total_train   = 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            y_hat = net(Xb)
            loss  = loss_fn(y_hat, yb)
            loss.backward()
            optimizer.step()
            
            # track training accuracy within epoch
            preds = y_hat.argmax(dim=1)
            correct_train += (preds == yb).sum().item()
            total_train   += len(yb)
        
        train_acc = correct_train / total_train
        train_acc_hist.append(train_acc)
        
        # Step C: EVAL on test set
        net.eval()
        correct_test = 0
        total_test   = 0
        with torch.no_grad():
            for X_t, y_t in test_loader:
                out_t = net(X_t)
                preds_t = out_t.argmax(dim=1)
                correct_test += (preds_t == y_t).sum().item()
                total_test   += len(y_t)
        test_acc = correct_test / total_test
        test_acc_hist.append(test_acc)
        
        # Step D: after training is done for this epoch,
        #         compute metrics for each layer
        layer_idx = 0
        with torch.no_grad():
            for name, param in net.named_parameters():
                if 'weight' in name: 
                    # current weight
                    w_curr = param.detach().cpu().numpy()
                    # compute Frobenius distance from init
                    #   note: layer_weights_init[layer_idx] is the same layer's weight
                    dist = np.linalg.norm(w_curr - layer_weights_init[layer_idx], 'fro')
                    weightChanges[epoch_i, layer_idx] = dist
                    
                    # condition number
                    cond = np.linalg.cond(w_curr, p=None)  # default is 2-norm cond
                    weightConds[epoch_i, layer_idx] = cond
                    
                    layer_idx += 1
    
    return (train_acc_hist, test_acc_hist, weightChanges, weightConds)
```
```

**Notes**:
- We only measure *intra-epoch* differences (start vs. end) rather than differences between *epoch i and epoch i+1*.  
- Condition number uses `np.linalg.cond(...)`. By default, `p=None` means 2-norm condition (largest singular value / smallest singular value).  

---

## 4. Running and Visualizing Results

```python
```python
net = MNISTNet()
train_acc, test_acc, weightChanges, weightConds = train_and_track_weights(
    net, train_loader, test_loader, epochs=50, lr=0.005
)

# For labeling convenience
layer_names = ["fc1", "fc2", "fc3", "out"]

# 4.1 Plot Accuracies
epochs_arr = np.arange(1, 51)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_arr, train_acc, label='Train Acc')
plt.plot(epochs_arr, test_acc,  label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

# 4.2 Plot Weight Changes (Frobenius Dist)
plt.subplot(1,2,2)
for i in range(weightChanges.shape[1]):
    plt.plot(epochs_arr, weightChanges[:,i], label=layer_names[i])
plt.xlabel('Epoch')
plt.ylabel('Euclidean Dist from Start of Epoch')
plt.title('Weight Changes over Epochs')
plt.legend()
plt.tight_layout()
plt.show()
```
```

### 4.1. Observations: Weight Changes vs. Epoch

- **Large** distance = big updates that epoch → *rapid learning*.  
- **Small** distance = stable or minimal updates.

---

### 4.2. Plot Condition Numbers

```python
```python
plt.figure(figsize=(8,5))
for i in range(weightConds.shape[1]):
    plt.plot(epochs_arr, weightConds[:,i], label=layer_names[i])
plt.xlabel('Epoch')
plt.ylabel('Condition Number')
plt.title('Condition Number per Layer over Training')
plt.legend()
plt.show()
```
```

**Observations**:
- High condition number \(\Rightarrow\) the matrix is more “flattened” → potential “sparsity” or big directional skew.  
- Some layers might remain fairly **low** condition, others may **explode** to very high values.

---

## 5. Relationship to Performance

We can **compare** the derivative of accuracy to average weight change:

```python
```python
from scipy.stats import zscore

# derivative of train accuracy
acc_train_diff = np.diff(train_acc)  # length = epochs-1
acc_train_diff_z = zscore(acc_train_diff)

# average weight changes across layers
avg_change = weightChanges.mean(axis=1)  # shape = (epochs,)
avg_change_diff_z = zscore(avg_change[1:])  # to align with .diff

plt.figure(figsize=(8,5))
plt.plot(acc_train_diff_z, label='Z-scored derivative of Train Acc')
plt.plot(avg_change_diff_z, label='Z-scored Weight Change')
plt.title('Comparing Learning Dynamics to Weight Update Magnitude')
plt.legend()
plt.show()
```
```

**Often**: We see that *when the model is learning fastest*, the weight changes are **largest**. Over time, updates shrink as the model converges.

---

## 6. Takeaways

1. **Euclidean (Frobenius) Distance**: Reveals **how much** each layer’s weights shift per epoch.  
2. **Condition Number**: Shows how “spherical” (low cond #) or “flattened/sparse” (high cond #) the weight matrix becomes.  
3. **Layer-Specific Behavior**: Input, hidden, and output layers may vary in how quickly or drastically they update.  
4. **Relating to Accuracy**: Large weight changes can align with spikes in learning (performance improvement).  
5. **Interpretation**: 
   - Large \(\kappa\): Possibly indicates a sparse representation, which may be beneficial for certain tasks, or might reflect overfitting.  
   - Over training, we often see condition numbers drift upward in hidden layers as the model finds specialized features.

**Hence**: Investigating weight dynamics can help diagnose if learning is **stalling** or if layers are **overfitting** or **underutilizing** certain directions in parameter space.

---

## 7. Additional Notes

- **Practical Usage**:
  - Could track these metrics as a form of **debugging** or **monitoring** training, ensuring model doesn’t blow up or saturate.  
  - Condition number can get extremely large → might consider **regularization** or alternative initialization if it leads to instability.  
- **SVD Complexity**:
  - For large matrices, computing SVD each epoch can be expensive. Often done for smaller models or only once in a while (not every epoch).  
- **Future**:
  - We can combine these insights with **pruning** (removing weights with small contributions) or advanced optimization strategies.

---

## 8. References

- **PyTorch**: [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)  
- **Linear Algebra** (eigenvalues, singular values): [Strang, Gilbert. *Introduction to Linear Algebra*]  
- **Condition Number** discussion: [https://en.wikipedia.org/wiki/Condition_number](https://en.wikipedia.org/wiki/Condition_number)  
- **Frobenius Norm** usage in Python: [NumPy docs on `np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

---
```

**How to Use These Notes in Obsidian**:

1. **Create a new note** (e.g., `LearningRelatedWeightChanges.md`) in your vault.  
2. **Paste** the entire text above (including frontmatter).  
3. Adjust or add **internal links** (e.g., `[[Transfer Learning]]`, `[[PyTorch Examples]]`).  
4. (Optional) Re-organize headings or tags based on your vault’s structure.

These notes give a **detailed** look at how to **quantify** weight changes (via **Frobenius distance** and **condition numbers**), interpret the results, and see how they correlate with **model accuracy** over training.