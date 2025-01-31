aliases: [Optimizer Comparison, SGD vs. RMSprop vs. Adam]
tags: [Deep Learning, Lecture Notes, Optimizers, PyTorch, Neural Networks]

In this lecture, we **empirically compare** **three** popular optimizers on a simple 3-class classification problem (the “qwerty” dataset):

1. **SGD (Stochastic Gradient Descent)**  
2. **RMSprop**  
3. **Adam**

We want to see how each **optimizer** affects:
- **Training speed**: How fast the model reaches good accuracy.
- **Final performance**: Whether there are notable differences in the best achieved accuracy.

Although the dataset is small and simple (thus we don’t necessarily see dramatic differences), it provides a **hands-on** demonstration of how to switch optimizers in code and interpret training plots.

---

## 1. Dataset Setup (3-Class Qwerty)

We use the same **Qwerty** dataset as in previous sections:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# 1) Generate 3 Gaussian blobs ("qwerty" data)
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

# 2) Convert to PyTorch
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

# 3) Train / Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_t, y_t, test_size=0.2, shuffle=True, random_state=42
)

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

print("Train samples:", len(train_ds), "Test samples:", len(test_ds))

# Quick visualize
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap="Set2", alpha=0.8)
plt.title("Qwerty Dataset (3 Classes)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

---

## 2. Defining the Model & Helper Function

We create a simple **two-layer** MLP for 3-class classification:

```python
class QwertyNet(nn.Module):
    def __init__(self):
        super(QwertyNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # raw logits for 3 classes
        return x

def create_model(optimizerAlgo, lr=0.01):
    """ Returns (model, loss_fn, optimizer) for the chosen optimizerAlgo. """
    model = QwertyNet()
    loss_fn = nn.CrossEntropyLoss()
    
    # Convert string -> PyTorch optimizer class, e.g. "SGD" -> optim.SGD
    opt_class = getattr(optim, optimizerAlgo)
    optimizer = opt_class(model.parameters(), lr=lr)
    
    return model, loss_fn, optimizer
```

**Key**: We dynamically choose an optimizer by **string** (e.g., `"SGD"`, `"RMSprop"`, or `"Adam"`).

---

## 3. Training Function

A standard training loop:

```python
def train_model(optimizerAlgo, epochs=50, lr=0.01):
    model, loss_fn, optimizer = create_model(optimizerAlgo, lr=lr)
    
    train_loss_hist = []
    test_loss_hist  = []
    train_acc_hist  = []
    test_acc_hist   = []
    
    for ep in range(epochs):
        # ---- TRAINING ----
        model.train()
        total_loss = 0
        correct, total = 0, 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total   += len(yb)
        
        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        
        # ---- TESTING ----
        model.eval()
        t_loss = 0
        c_test, tot_test = 0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                logits_test = model(Xb)
                loss_test   = loss_fn(logits_test, yb)
                t_loss += loss_test.item()
                
                preds_test = torch.argmax(logits_test, dim=1)
                c_test     += (preds_test == yb).sum().item()
                tot_test   += len(yb)
        
        test_loss = t_loss / len(test_loader)
        test_acc  = c_test / tot_test
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
    
    return {
        "model": model,
        "train_loss": train_loss_hist,
        "test_loss":  test_loss_hist,
        "train_acc":  train_acc_hist,
        "test_acc":   test_acc_hist
    }
```

---

## 4. Visualization Function

We create a helper function to:

1. **Plot** the train/test loss and accuracy curves.  
2. Compute **per-class accuracy**.  
3. Plot a bar chart of per-class accuracy.

```python
def plot_results(results, optimizer_name):
    model = results["model"]
    
    train_loss = results["train_loss"]
    test_loss  = results["test_loss"]
    train_acc  = results["train_acc"]
    test_acc   = results["test_acc"]
    
    # Evaluate final model on ALL data
    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)
    
    model.eval()
    with torch.no_grad():
        logits_all = model(X_all)
        preds_all  = torch.argmax(logits_all, dim=1)
    
    accuracy_all = (preds_all == y_all).float().mean().item()
    
    # Per-class accuracy
    accuracies_per_class = []
    for c in [0,1,2]:
        mask = (y_all == c)
        class_acc = (preds_all[mask] == y_all[mask]).float().mean().item()
        accuracies_per_class.append(class_acc)
    
    # Plot
    epochs = range(1, len(train_loss)+1)
    
    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, test_loss,  label="Test Loss")
    axs[0].set_title(f"{optimizer_name} Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    axs[1].plot(epochs, train_acc, label="Train Acc")
    axs[1].plot(epochs, test_acc,  label="Test Acc")
    axs[1].set_title(f"{optimizer_name} Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    
    axs[2].bar([0,1,2], accuracies_per_class, tick_label=["Class0", "Class1", "Class2"])
    axs[2].set_title(f"Final Per-Class Acc\nOverall: {accuracy_all*100:.2f}%")
    axs[2].set_ylim([0,1])
    axs[2].set_ylabel("Accuracy")
    
    plt.tight_layout()
    plt.show()
```

---

## 5. Running the Comparison for Three Optimizers

We’ll test **SGD**, **RMSprop**, and **Adam** with the same learning rate `lr=0.01` (though in practice each optimizer might require its own tuning).

```python
optimizers = ["SGD", "RMSprop", "Adam"]
results_all = {}

for opt_name in optimizers:
    print(f"Training with {opt_name} optimizer...")
    res = train_model(opt_name, epochs=50, lr=0.01)
    results_all[opt_name] = res
    # plot results
    plot_results(res, opt_name)
```

### 5.1 Observations

- **SGD**: Possibly slower convergence; final accuracy might be slightly less or around the same if given enough epochs.  
- **RMSprop**: Faster convergence in early epochs. Often reaches near-final accuracy quickly.  
- **Adam**: Similar or even faster than RMSprop in many cases, typically considered a solid default.

---

## 6. Sample Outcomes

You might see something like:

- **SGD** final accuracy: ~88–90%  
- **RMSprop** final accuracy: ~88–90%  
- **Adam** final accuracy: ~88–90%

All end up **similar** on this simple dataset, but **Adam** or **RMSprop** likely reach that final state **faster**. On more **complex** tasks, Adam often outperforms SGD by a larger margin, both in speed and sometimes final accuracy.

---

## 7. Conclusion and Notes

1. **Small Dataset**: Differences can be minor if the task is not too challenging.  
2. **Faster Convergence**: RMSprop/Adam typically reduce the loss quickly within the first few epochs.  
3. **Same Final Performance**: With careful tuning, all optimizers can converge to a comparable solution here.  
4. **Beyond**: In larger models/datasets, Adam is often the **preferred** choice for robust performance.

**Key Lesson**: Try **SGD**, **RMSprop**, **Adam** with a common learning rate, see how quickly each optimizer converges. Then adjust the learning rate **per** optimizer if needed for deeper comparisons.

---

## 8. References

- [**PyTorch** docs on `torch.optim` (SGD, RMSprop, Adam)](https://pytorch.org/docs/stable/optim.html)  
- Goodfellow, Bengio, Courville, *Deep Learning* – Chapter on **Optimization**  
- [“Adam: A Method for Stochastic Optimization” (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980)

```
Created by: [Your Name / Lab / Date]
Lecture Reference: “Meta parameters: Optimizers comparison”
```
```