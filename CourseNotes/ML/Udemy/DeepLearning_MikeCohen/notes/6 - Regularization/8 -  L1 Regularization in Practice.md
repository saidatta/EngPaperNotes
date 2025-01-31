## Table of Contents
1. [[Overview]]
2. [[Dataset & Model Setup]]
3. [[Why Manual L1 in PyTorch?]]
4. [[Implementing L1 Regularization Manually]]
5. [[Training Function & Example Code]]
6. [[Experiment: Varying L1 Coefficients]]
7. [[Results and Observations]]
8. [[Key Takeaways & Further Exploration]]

---

## 1. Overview
- **Goal**: Apply **L1** regularization (lasso) to a simple deep learning model in **PyTorch**.  
- Unlike L2 (weight decay), there is **no** built-in parameter in PyTorch optimizers for L1.  
- We must **manually** compute the sum of the absolute values of the weights and **add** it to the model’s loss function.

### Why L1?
- **Promotes sparsity**: Some weights become exactly zero.  
- Potentially useful for **feature selection** or interpretability in larger models.

---

## 2. Dataset & Model Setup

### 2.1. Iris Dataset
We use the **Iris** dataset:
- 150 samples, 4 features (sepal/petal dimensions), 3 classes (setosa, versicolor, virginica).

```python
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Load Iris
iris_data = load_iris()
X = iris_data.data          # shape (150, 4)
y = iris_data.target        # 3 classes (0,1,2)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)
```

### 2.2. Defining a Simple Model
A small neural network with one or two hidden layers, e.g.:

```python
```python
def create_model():
    # A small MLP for Iris
    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3)  # final classification layer
    )
    return model
```

---

## 3. Why Manual L1 in PyTorch?
- **L2** regularization is straightforward (`weight_decay`) in optimizers.  
- **L1** is not directly built-in. Instead, we:
  1. Sum the absolute values of model weights each training step.  
  2. Multiply by a regularization parameter (\(\lambda\)).  
  3. Add this to the main loss.

This manual approach also:
- Teaches us how to **access** individual model parameters.  
- Lets us easily combine L1 with custom/advanced norms (elastic net, etc.).

---

## 4. Implementing L1 Regularization Manually
1. After calculating the **primary loss** (e.g., cross entropy), iterate over **model parameters**.  
2. For each parameter `p`, compute `torch.sum(torch.abs(p))`.  
3. Combine them into a single **L1 penalty**, multiply by **\(\lambda\)**.  
4. Add to total loss: `loss = main_loss + l1_lambda * l1_term`.

---

## 5. Training Function & Example Code

### 5.1. Training Function
```python
```python
def train_model(model, train_loader, test_loader, l1_lambda=0.0, epochs=1000):
    # We'll use standard cross-entropy for multi-class
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)  # no weight_decay for L1
    
    train_accs, train_losses, test_accs = [], [], []

    # Count how many parameters are not biases
    # (Optional step: some prefer not to regularize biases)
    num_params = 0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            num_params += param.numel()

    for epoch in range(epochs):
        # Training mode
        model.train()
        
        batch_accs = []
        batch_losses = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            # 1) Compute L1 penalty
            l1_penalty = 0
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l1_penalty += torch.sum(torch.abs(param))
            
            # 2) Average or scale by num_params if desired
            l1_penalty = l1_penalty / num_params
            
            # 3) Add to total loss
            total_loss = loss + l1_lambda * l1_penalty
            total_loss.backward()
            optimizer.step()
            
            # Track batch metrics
            preds = torch.argmax(y_pred, axis=1)
            acc   = (preds == y_batch).float().mean().item()
            batch_accs.append(acc)
            batch_losses.append(loss.item())
        
        # End of epoch, store average stats
        train_accs.append(np.mean(batch_accs))
        train_losses.append(np.mean(batch_losses))
        
        # Evaluate test accuracy
        model.eval()
        with torch.no_grad():
            for X_testb, y_testb in test_loader:
                y_test_pred = model(X_testb)
                preds_test  = torch.argmax(y_test_pred, axis=1)
                test_acc    = (preds_test == y_testb).float().mean().item()
        test_accs.append(test_acc)
        
        # model.train() # optional re-enable train mode if needed (no dropout or BN here)
    
    return train_accs, test_accs, train_losses
```

**Key Points**:
- We do **not** set `weight_decay` in the optimizer.  
- We manually compute the L1 penalty by summing absolute values of all *non-bias* parameters.

---

## 6. Experiment: Varying L1 Coefficients
We run a parametric sweep: \(\lambda\in [0.0,\dots,0.005]\) to observe how different L1 strengths affect training.

```python
```python
l1_vals = np.linspace(0.0, 0.005, 10)
store_train_acc = {}
store_test_acc  = {}

epochs = 1000

for val in l1_vals:
    model = create_model()
    train_accs, test_accs, _ = train_model(
        model, train_loader, test_loader, 
        l1_lambda=val, epochs=epochs
    )
    
    store_train_acc[val] = train_accs
    store_test_acc[val]  = test_accs
```

1. **0.0** => No L1 (baseline).  
2. Up to **0.005** => moderate L1.

---

### 6.1. Visualizing Accuracy vs. Epoch
```python
```python
plt.figure(figsize=(8,5))
for val, accs in store_train_acc.items():
    # optionally smooth
    smoothed_acc = np.convolve(accs, np.ones(5)/5, mode='same')
    plt.plot(smoothed_acc, label=f"L1={val:.4f} (Train)")
plt.title("Train Accuracy vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
for val, accs in store_test_acc.items():
    smoothed_acc = np.convolve(accs, np.ones(5)/5, mode='same')
    plt.plot(smoothed_acc, label=f"L1={val:.4f} (Test)")
plt.title("Test Accuracy vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

**Observation**: On **small** datasets, lines may overlap or vary only marginally.

---

### 6.2. Final or Mid-Epoch Comparison
We can average the accuracies over certain epochs to see if any \(\lambda\) value stands out.

```python
```python
start_epoch, end_epoch = 500, 900
train_final = []
test_final  = []
for val in l1_vals:
    accs_tr = store_train_acc[val][start_epoch:end_epoch]
    accs_te = store_test_acc[val][start_epoch:end_epoch]
    train_final.append(np.mean(accs_tr))
    test_final.append(np.mean(accs_te))

plt.figure(figsize=(6,4))
plt.plot(l1_vals, train_final, 'o-', label='Train (Avg. Over Selected Epochs)')
plt.plot(l1_vals, test_final,  'o-', label='Test')
plt.title("Accuracy vs L1 Coefficient")
plt.xlabel("L1 Lambda")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

---

## 7. Results and Observations
- Often, we see **no major improvement** from L1 on small, relatively easy datasets (like Iris).  
- In some runs, larger L1 might slow learning or degrade final accuracy. In others, it might produce marginal improvements early on.  
- L1 can **zero out** some weights, but with only ~64 hidden units and 4 input features, the network might already not overfit too heavily.

**Important**: L1 is more **impactful**:
1. In **larger, high-dimensional** models, or
2. Where **feature selection** is crucial, or
3. With **scant** data leading to big overfitting.

---

## 8. Key Takeaways & Further Exploration
1. **Manual L1**:  
   - Collect absolute values of the weights, sum them, multiply by \(\lambda\).  
   - Add to the standard loss.  
2. **Bias Exclusion**:  
   - Often, biases are excluded from L1.  
   - This is a design choice; can also penalize biases if desired.  
3. **Small vs. Large Models**:  
   - On simple tasks (Iris), L1 rarely shows large gains.  
   - On bigger tasks, L1 can improve generalization or create sparse solutions.  
4. **Combining**:  
   - L1 + L2 \(\rightarrow\) “Elastic Net.”  
   - You can similarly add custom penalty terms for other regularization strategies.

**Next Steps**:
- Experiment with **manual L2** similarly (if you haven’t tried the advanced exercises).  
- Explore **Elastic Net** by combining the L1 and L2 penalties.  
- Look at how **dropout** or **batch norm** interplay with L1 or L2 in deeper networks.

---

**End of Notes**  
You have now seen how to **manually implement** L1 (lasso) in PyTorch, which offers fine-grained control over which parameters to penalize and how strongly. Although it may not significantly help on a small dataset like Iris, L1 proves valuable in **larger-scale** deep learning for encouraging **sparser** models.