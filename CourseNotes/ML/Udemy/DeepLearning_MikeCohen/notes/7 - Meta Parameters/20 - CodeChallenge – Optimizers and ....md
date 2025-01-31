aliases: [Code Challenge: Optimizers and Learning Rates, Optimizer Meta Parameters]
tags: [Deep Learning, Lecture Notes, Meta Parameters, Optimizers, PyTorch]

This challenge extends our **optimizer comparison** from the previous lecture by systematically varying the **learning rate** across several **logarithmically spaced** values. You will:

1. **Compare** how **SGD**, **RMSprop**, and **Adam** handle a wide range of learning rates on the same 3-class Qwerty dataset.
2. **Plot** final (or near-final) performance versus learning rate.
3. Observe differences in **sensitivity** to the chosen learning rate among these optimizers.

---

## 1. Requirements and Overview

- Start with the **previous** code from [“Optimizers comparison”](#).  
- Instead of **one** fixed learning rate, we will test **~20** rates from \(10^{-4}\) to \(10^{-1}\), spaced **logarithmically**.  
- For each **(optimizer, learning_rate)** combination:
  - Train a small MLP on the Qwerty dataset for ~50 epochs.
  - Record the **final** or **average-of-final** (10) epochs test accuracy.
- **Plot** or **tabulate** these accuracies vs. learning rate for each optimizer.

---

## 2. Data and Model Setup

We use the same approach to generate the 3-class Qwerty dataset, define the model, and so on. (All prior code is reused with minimal modifications.)

### 2.1 Qwerty Data Generation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# Generate the data (same as before)
np.random.seed(42)
n_per_class = 300
means = np.array([[2,1], [-2,1], [0,-3]])
cov   = np.array([[1,0],[0,1]])

X = np.zeros((n_per_class*3, 2), dtype=np.float32)
y = np.zeros(n_per_class*3,       dtype=np.int64)
for i in range(3):
    idx = range(i*n_per_class, (i+1)*n_per_class)
    X[idx,:] = np.random.multivariate_normal(means[i], cov, n_per_class)
    y[idx]   = i

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_t, y_t, test_size=0.2, shuffle=True, random_state=42
)

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)
```

### 2.2 Model Definition

```python
class QwertyNet(nn.Module):
    def __init__(self):
        super(QwertyNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # raw logits
        return x

def create_model(optim_algo="SGD", lr=0.01):
    """
    Returns (model, loss_fn, optimizer) for the chosen optimizer with the given LR.
    """
    model = QwertyNet()
    loss_fn = nn.CrossEntropyLoss()
    OptimClass = getattr(optim, optim_algo)  # e.g. 'Adam' -> optim.Adam
    
    optimizer = OptimClass(model.parameters(), lr=lr)
    return model, loss_fn, optimizer
```

---

## 3. Training Function

We include a parameter for **learning rate** so we can systematically vary it:

```python
def train_model(optimizer_algo="SGD", lr=0.01, epochs=50):
    model, loss_fn, optimizer = create_model(optimizer_algo, lr=lr)
    
    train_loss_hist = []
    test_loss_hist  = []
    train_acc_hist  = []
    test_acc_hist   = []
    
    for ep in range(epochs):
        # ---- TRAIN ----
        model.train()
        total_loss = 0
        correct = 0
        total   = 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total   += len(yb)
        
        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total
        
        # ---- TEST ----
        model.eval()
        t_loss = 0
        c_test, tot_test = 0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                logits_test = model(Xb)
                loss_test   = loss_fn(logits_test, yb)
                t_loss     += loss_test.item()
                
                preds_test = torch.argmax(logits_test, dim=1)
                c_test     += (preds_test == yb).sum().item()
                tot_test   += len(yb)
        
        test_loss = t_loss / len(test_loader)
        test_acc  = c_test / tot_test
        
        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)
    
    # we can average the *final 10 epochs* test accuracy as suggested
    final_test_acc_mean = np.mean(test_acc_hist[-10:])
    
    return {
        "train_loss": train_loss_hist,
        "test_loss":  test_loss_hist,
        "train_acc":  train_acc_hist,
        "test_acc":   test_acc_hist,
        "final_test_acc_mean": final_test_acc_mean
    }
```

---

## 4. Running the Parametric Experiment

We test **3 optimizers** over about **20** learning rates from \(10^{-4}\) to \(10^{-1}\), spaced **logarithmically**:

```python
# Logarithmically spaced LRs from 1e-4 to 1e-1
lr_values = np.logspace(np.log10(1e-4), np.log10(1e-1), num=20)

optimizers = ["SGD", "RMSprop", "Adam"]

results_dict = {opt: [] for opt in optimizers}

for opt_name in optimizers:
    print(f"\n=== Optimizer: {opt_name} ===")
    for lr_val in lr_values:
        outcome = train_model(opt_name, lr=lr_val, epochs=50)
        final_acc = outcome["final_test_acc_mean"]
        results_dict[opt_name].append((lr_val, final_acc))
        print(f"  LR={lr_val:.5f}, Final Test Acc (avg last 10 ep)={final_acc:.3f}")
```

### 4.1 Visualizing Results

We can plot **final test accuracy** vs. **learning rate** on a **log scale**:

```python
plt.figure(figsize=(8,6))

for opt_name in optimizers:
    lrs = [res[0] for res in results_dict[opt_name]]
    acc = [res[1] for res in results_dict[opt_name]]
    plt.plot(lrs, acc, marker="o", label=opt_name)

plt.xscale("log")
plt.title("Final Test Accuracy vs. Learning Rate")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Test Accuracy (avg last 10 epochs)")
plt.legend()
plt.show()
```

---

## 5. Interpretation

### 5.1 Typical Outcomes

- **SGD**:
  - More **sensitive** to learning rate.
  - Might fail to converge or do poorly if `lr` is too high; might converge very slowly if `lr` is too low.
- **RMSprop / Adam**:
  - Generally more **robust** to a wide range of learning rates.
  - Often reach decent accuracy even if `lr` is suboptimal.
- Possibly:
  - For extremely high `lr`, all might degrade.
  - For extremely small `lr`, training can be too slow or stuck.

### 5.2 Observations

- If you see **Adam** + **RMSprop** curves nearly flat across wide `lr` range, that highlights their **adaptiveness**.  
- SGD might have a **peak** around some mid-range LR, with poor performance at extremes.

---

## 6. Conclusion

This challenge reveals how **learning rate** interacts with **different optimizers**:

- **SGD** can be **highly** sensitive to LR.  
- **RMSprop** and **Adam** adapt per-parameter step sizes internally, so they are **less sensitive** (though not entirely immune) to LR choices.  
- On more complex tasks, **Adam** often outperforms plain SGD or RMSprop, but one must still consider hyperparameter tuning.

**Key Lesson**: Modern optimizers like **RMSprop** and **Adam** are **more robust** to learning rate. That said, you should still **experiment** with a small range of LRs to see optimal performance.

---

## 7. Possible Extensions

1. **Vary Momentum / betas** in RMSprop or Adam.  
2. **Compare** total training time or iteration count vs. final accuracy.  
3. **Visualize** the entire training curves for each (optimizer, LR) to see convergence patterns.

---

## 8. References

- Goodfellow, Bengio, Courville, *Deep Learning*: Chapter on **Optimization**.  
- [**Adam Paper**: Kingma & Ba (2014)](https://arxiv.org/abs/1412.6980)  
- [**PyTorch**: `torch.optim` docs](https://pytorch.org/docs/stable/optim.html)

```
Created by: [Your Name / Lab / Date]
Lecture Reference: “Meta parameters: CodeChallenge: Optimizers and... something”
```
```