
aliases: [SGD with Momentum, Momentum Hyperparameter, Beta in SGD]
tags: [Deep Learning, Lecture Notes, Optimizers, PyTorch]

In this lecture, we implement **SGD + Momentum** in PyTorch and observe how different momentum coefficients (\(\beta\)) affect **training speed** and **final accuracy** on a synthetic 3-class “qwerty” dataset. We’ll see that **moderate** momentum values (\(\beta \approx 0.9\)) often yield **faster** convergence, while excessively large momentum can destabilize learning.

---
## 1. Dataset & Imports

We use the **3-class qwerty** dataset as in previous examples. The code below:

1. **Generates** 3 Gaussian blobs.
2. Splits into **train/test** sets.
3. Wraps in a PyTorch **DataLoader**.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# 1) Create synthetic data (3 Gaussian clouds)
np.random.seed(42)
n_per_class = 300

means = np.array([
    [ 2,  1],
    [-2,  1],
    [ 0, -3]
])
cov = np.array([[1, 0],[0, 1]])

X = np.zeros((n_per_class*3, 2))
y = np.zeros(n_per_class*3)

for i in range(3):
    idx = range(i*n_per_class, (i+1)*n_per_class)
    X[idx,:] = np.random.multivariate_normal(means[i], cov, n_per_class)
    y[idx]   = i

# 2) Convert to tensors
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

# 3) Train/Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_t, y_t, test_size=0.2, shuffle=True, random_state=42
)

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

print("Train samples:", len(train_ds), "Test samples:", len(test_ds))

# Quick plot
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap="Set2", alpha=0.8)
plt.title("Qwerty 3-Class Dataset")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()
```

---

## 2. Defining the Model & Optimizer

We build a **simple** MLP for 3-class classification:

```python
class QwertyNet(nn.Module):
    def __init__(self):
        super(QwertyNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # final layer: raw logits for 3 classes
        x = self.fc2(x)
        return x

def create_network(momentum=0.0, lr=0.01):
    """
    Returns (model, loss_fn, optimizer) given a momentum factor and learning rate.
    """
    model = QwertyNet()
    loss_fn = nn.CrossEntropyLoss()
    
    # The crucial line: using 'SGD' with momentum
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    return model, loss_fn, optimizer
```

**Key**: `momentum=0.0` gives **pure** SGD; nonzero momentum accumulates velocity from previous updates.

---

## 3. Training Function

We use a standard training loop, computing **train** and **test** loss/accuracy each epoch.

```python
def train_model(momentum=0.0, lr=0.01, epochs=50):
    model, loss_fn, optimizer = create_network(momentum, lr)
    
    train_loss_history = []
    test_loss_history  = []
    train_acc_history  = []
    test_acc_history   = []
    
    for ep in range(epochs):
        
        # ---- TRAIN ----
        model.train()
        total_loss = 0
        correct = 0
        total   = 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total   += len(yb)
        
        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total
        
        # ---- TEST ----
        model.eval()
        total_loss_test = 0
        correct_test = 0
        total_test   = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                logits_test = model(Xb)
                loss_test   = loss_fn(logits_test, yb)
                total_loss_test += loss_test.item()
                
                preds_test  = torch.argmax(logits_test, dim=1)
                correct_test += (preds_test == yb).sum().item()
                total_test   += len(yb)
        
        test_loss = total_loss_test / len(test_loader)
        test_acc  = correct_test / total_test
        
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
    
    return {
        "train_loss": train_loss_history,
        "test_loss":  test_loss_history,
        "train_acc":  train_acc_history,
        "test_acc":   test_acc_history
    }
```

---

## 4. Parametric Experiment with Different Momentum Values

We try \(\beta\) in \([0.0, 0.5, 0.9, 0.95, 0.999]\) to see how it impacts **training speed** and final performance.

```python
momentums = [0.0, 0.5, 0.9, 0.95, 0.999]
results = {}

for mom in momentums:
    print(f"Training with momentum={mom}")
    outcome = train_model(momentum=mom, lr=0.01, epochs=50)
    results[mom] = outcome
```

---

## 5. Visualizing the Results

### 5.1 Loss Curves

```python
plt.figure(figsize=(10,4))

for i, mom in enumerate(momentums):
    train_loss = results[mom]["train_loss"]
    test_loss  = results[mom]["test_loss"]
    
    plt.plot(train_loss, label=f"Train (mom={mom})", linestyle="-")
    plt.plot(test_loss,  label=f"Test (mom={mom})", linestyle="--")

plt.title("Loss Curves for Different Momentum Values")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim([0,2])  # adjust if any lines go very high
plt.legend(ncol=2)
plt.show()
```

### 5.2 Accuracy Curves

```python
plt.figure(figsize=(10,4))

for i, mom in enumerate(momentums):
    train_acc = results[mom]["train_acc"]
    test_acc  = results[mom]["test_acc"]
    
    plt.plot(train_acc, label=f"Train (mom={mom})", linestyle="-")
    plt.plot(test_acc,  label=f"Test (mom={mom})", linestyle="--")

plt.title("Accuracy Curves for Different Momentum Values")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0,1])
plt.legend(ncol=2)
plt.show()
```

**Observations**:

- **mom=0.0** (pure SGD): converges slower.  
- **mom ~ 0.9** or 0.95: typically **faster** convergence; good final accuracy.  
- **mom=0.999**: can cause **instability** or very slow adaptation because the algorithm overrelies on previous updates, ignoring current gradient signals → poor performance.

---

## 6. Interpretation

1. **Momentum > 0** typically **speeds** training in early epochs.  
2. Moderately high momentum \(\beta \approx 0.9 \dots 0.95\) often helps for both **faster** and sometimes **better** final performance.  
3. Extremely high momentum \(\beta \approx 0.999\) can overshoot or oscillate (or, in some problems, fail to converge).  
4. In our 3-class “qwerty” dataset, eventually all solutions but \(\beta=0.999\) reach similar final accuracy, but real-world data often show more significant gains from moderate momentum.

---

## 7. Conclusion

- **SGD with momentum** modifies the weight update to include a fraction of **previous** gradient steps, smoothing out the path in the loss landscape.
- In **PyTorch**, it’s as simple as specifying `momentum=0.x` in `torch.optim.SGD(...)`.
- **Best practices**: Momentum around `0.9`–`0.95` is common if not using more advanced optimizers (e.g., Adam).  
- **Future**: We’ll explore **RMSProp** and **Adam**, which combine momentum-like ideas with **adaptive learning rates**.

---

## 8. Further Reading

- [**Momentum** in Deep Learning Book – Goodfellow et al. (Ch. 8.3.2)](https://www.deeplearningbook.org/)  
- [**Stanford CS231n** Momentum discussion](http://cs231n.github.io/neural-networks-3/#sgd)  
- PyTorch Docs: [**SGD** with Momentum](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)

```
Created by: [Your Name / Lab / Date]
Based on Lecture: “Meta parameters: SGD with momentum”
```