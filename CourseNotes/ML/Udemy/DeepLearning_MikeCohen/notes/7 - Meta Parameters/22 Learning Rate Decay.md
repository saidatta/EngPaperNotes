
aliases: [Learning Rate Decay, LR Scheduler, StepLR, PyTorch]
tags: [Deep Learning, Lecture Notes, Meta Parameters, Optimizers, Neural Networks]


Many optimizers (e.g., Adam, RMSprop) **adapt** the effective learning rate based on gradient statistics, but sometimes you want **explicit** control over how the learning rate **decays** across epochs. In PyTorch, you can achieve this via **learning rate schedulers** that *manually* or *automatically* reduce the learning rate at certain intervals or conditions.

This note explains **why** learning rate decay can help, and shows **how** to implement a **step-based** LR scheduler in PyTorch with a toy example (the three-class Qwerty dataset).

---

## 1. Recap: Why Decay the Learning Rate?

1. **Initial Rapid Learning**: Use a **larger** LR in early epochs to traverse the loss landscape quickly.  
2. **Refinement**: As training progresses, reduce LR to make **finer** adjustments and avoid overshooting the minima.  
3. **Combining**: Even if an optimizer like Adam adjusts LR internally, a global decay schedule can further **stabilize** or **speed** final convergence.

### 1.1 Different Decay Schemes

- **Step decay**: Halve LR every certain # of epochs or batches.  
- **Exponential decay**: Gradually decay each epoch: `lr = lr_0 * gamma^epoch`.  
- **Multi-step** or **Plateau**: Decrease LR when validation loss stops improving.

Here, we focus on **step decay** with a **StepLR** scheduler in PyTorch.

---

## 2. Data & Model Setup (Qwerty 3-Class)

We reuse the **three-class** synthetic dataset.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# 1) Generate 3-class dataset
np.random.seed(42)
n_per_class = 300
means = np.array([[2,1], [-2,1], [0,-3]])
cov = np.array([[1,0],[0,1]])

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

print("Train size:", len(train_ds), "Test size:", len(test_ds))
```

### 2.1 Model Definition

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
```

---

## 3. Learning Rate Scheduler in PyTorch

We create a helper function that:
1. Builds the model.  
2. Creates an optimizer with **initial learning rate**.  
3. **Optionally** sets up a **StepLR** scheduler that halves LR every certain # of calls.

```python
def create_model_with_scheduler(initialLR=0.01, use_scheduler=True):
    """
    Returns (model, loss_fn, optimizer, scheduler) with optional StepLR.
    """
    model = QwertyNet()
    loss_fn = nn.CrossEntropyLoss()
    
    # Basic SGD for demonstration
    optimizer = optim.SGD(model.parameters(), lr=initialLR)
    
    scheduler = None
    if use_scheduler:
        # Step size: after how many calls to 'scheduler.step()' do we reduce LR
        # gamma=0.5 means we multiply LR by 0.5 each step.
        step_size = len(train_loader)*5  # or 5 epoch equivalent 
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    
    return model, loss_fn, optimizer, scheduler
```

**Key**: `lr_scheduler.StepLR(optimizer, step_size, gamma)` reduces LR by `gamma` every `step_size` calls to `scheduler.step()`.

---

## 4. Training Function

We incorporate `scheduler.step()` after each `optimizer.step()` so that we reduce the LR at the **correct** frequency. Also, we record the **current LR** in each iteration to visualize it.

```python
def train_model(initialLR=0.01, use_scheduler=True, epochs=10):
    model, loss_fn, optimizer, scheduler = create_model_with_scheduler(
        initialLR=initialLR, use_scheduler=use_scheduler
    )
    
    train_loss_history = []
    test_loss_history  = []
    train_acc_history  = []
    test_acc_history   = []
    lr_history         = []  # track current learning rate
    
    for ep in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            
            # If using scheduler, update it after each optimizer step
            if scheduler is not None:
                scheduler.step()
            
            lr_history.append(optimizer.param_groups[0]["lr"])
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total   += len(yb)
        
        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total
        
        # EVALUATE
        model.eval()
        t_loss = 0
        c_test, tot_test = 0,0
        with torch.no_grad():
            for Xb, yb in test_loader:
                out = model(Xb)
                l_test = loss_fn(out, yb)
                t_loss += l_test.item()
                
                preds_test = torch.argmax(out, dim=1)
                c_test     += (preds_test == yb).sum().item()
                tot_test   += len(yb)
        
        test_loss = t_loss / len(test_loader)
        test_acc  = c_test / tot_test
        
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
    
    return {
        "train_loss": train_loss_history,
        "test_loss":  test_loss_history,
        "train_acc":  train_acc_history,
        "test_acc":   test_acc_history,
        "lr_history": lr_history
    }
```

---

## 5. Testing and Visualizing

Let’s do a quick test with `epochs=10` or `epochs=20`:

```python
# 1) Use scheduler
res_dynamic = train_model(initialLR=0.01, use_scheduler=True, epochs=20)

# 2) No scheduler
res_static  = train_model(initialLR=0.01, use_scheduler=False, epochs=20)

# Plot LR histories
plt.figure(figsize=(6,4))
plt.plot(res_dynamic["lr_history"], label="Dynamic LR")
plt.plot(res_static["lr_history"],  label="Static LR")
plt.title("Learning Rate across iterations")
plt.xlabel("Iteration (batches)")
plt.ylabel("LR")
plt.legend()
plt.show()
```

**Check** if the LR decays in **step** manner for `res_dynamic` vs. constant in `res_static`.

### 5.1 Plot Accuracy / Loss

```python
epochs = range(1, 21)

plt.figure(figsize=(14,4))
# Train vs Test Loss
plt.subplot(1,2,1)
plt.plot(epochs, res_dynamic["train_loss"], 'r-', label="Train Loss (dynamic)")
plt.plot(epochs, res_dynamic["test_loss"],  'r--', label="Test Loss (dynamic)")
plt.plot(epochs, res_static["train_loss"],  'b-', label="Train Loss (static)")
plt.plot(epochs, res_static["test_loss"],   'b--', label="Test Loss (static)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves (Dynamic LR vs Static LR)")
plt.legend()

# Train vs Test Accuracy
plt.subplot(1,2,2)
plt.plot(epochs, res_dynamic["train_acc"], 'r-', label="Train Acc (dynamic)")
plt.plot(epochs, res_dynamic["test_acc"],  'r--', label="Test Acc (dynamic)")
plt.plot(epochs, res_static["train_acc"],  'b-', label="Train Acc (static)")
plt.plot(epochs, res_static["test_acc"],   'b--', label="Test Acc (static)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves (Dynamic LR vs Static LR)")
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 6. Observations and Discussion

1. **Step Decay**:  
   - Lowers the LR after a certain number of **iterations** (or epochs).  
   - Helps fine-tune the model in **later** training steps.
2. **Comparison**:  
   - Sometimes **static** LR can do similarly well if the problem is simple.  
   - For more complex tasks, a decaying LR might yield **faster** or better final convergence.
3. **Implementation Detail**:  
   - The call to `scheduler.step()` typically goes **after** `optimizer.step()`.  
   - The `step_size` is typically in **iterations** (mini-batches) if you want more fine-grained control (you can also step each epoch, etc.).
4. **Compatibility**:  
   - **Adam** or **RMSprop** also can use a LR scheduler on top of their internal adaptive mechanism.

---

## 7. Key Takeaways

- **Learning Rate Decay** is an additional meta-parameter that **systematically** lowers LR over training.  
- In PyTorch, we typically use `torch.optim.lr_scheduler` classes (e.g. **StepLR**, **ExponentialLR**, **ReduceLROnPlateau**, etc.).
- **Simple Approach**: StepLR with a fixed `step_size` and `gamma` factor:
  \[
  \text{LR} \,\leftarrow \,\text{LR} \times \gamma \quad \text{every } \text{step_size} \text{ steps.}
  \]
- Even though adaptive optimizers (Adam, RMSprop) already adjust LR on a *per-parameter* basis, a global LR scheduler can still be beneficial or used to further fine-tune training.

---

## 8. References

- [**PyTorch** docs on LR Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)  
- **Goodfellow, Bengio, Courville**: *Deep Learning* – Chapter on optimization strategies.  
- [**CS231n**: Convolutional Neural Networks for Visual Recognition – notes on LR scheduling](http://cs231n.github.io/neural-networks-3/#anneal)

---

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: “Learning rate decay”  
```