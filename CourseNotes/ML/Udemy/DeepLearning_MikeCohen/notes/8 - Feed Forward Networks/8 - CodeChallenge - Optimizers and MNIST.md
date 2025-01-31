aliases: [Optimizers, MNIST, CodeChallenge, Learning Rates, FFN]
tags: [Deep Learning, Classification, Feedforward, Metaparameters, PyTorch]
## Overview
In this CodeChallenge, we explore how **different optimizers** (SGD, RMSprop, Adam) and **various learning rates** affect classification performance on the MNIST dataset. This challenge continues the theme of understanding how **meta-parameters** (like the learning rate) and **optimization algorithms** can drastically influence model training efficiency and final accuracy.

We will:
1. Use a **standard feedforward architecture** for MNIST.
2. Evaluate **3 optimizers**: 
   - **SGD** (Stochastic Gradient Descent)
   - **RMSprop**
   - **Adam**
3. Test **6 logarithmically spaced learning rates** from \(10^{-4}\) to \(10^{-1}\).
4. Observe **train/test accuracy** over these settings and compare the results.

> **Note**: Each optimizer-learning-rate combination runs a full training session. With 3 optimizers \(\times\) 6 learning rates, that’s 18 runs. This can take **tens of minutes** depending on your hardware and the number of epochs.

---
## 1. Data Setup (MNIST)

We begin with the **MNIST dataset** (a partial version ~20k samples) in CSV format (e.g., `mnist_train_small.csv`). This is similar to previous challenges.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1. Load data
data_path = "/content/sample_data/mnist_train_small.csv"
data_np = np.loadtxt(data_path, delimiter=",")

# 2. Separate labels and pixels
labels_np = data_np[:, 0].astype(int)
pixels_np = data_np[:, 1:].astype(float)

# 3. Normalize [0,255] -> [0,1]
pixels_np /= 255.0

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    pixels_np, labels_np, test_size=0.10, random_state=42
)

# 5. Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# 6. Create DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)
```

---

## 2. Model Architecture

We’ll use a **fixed** feedforward network with two hidden layers (64 and 32 units). You can choose any architecture you prefer, but here’s a typical example:

```python
class SimpleMNIST(nn.Module):
    def __init__(self):
        super(SimpleMNIST, self).__init__()
        self.fc0 = nn.Linear(784, 64)  # input -> hidden1
        self.fc1 = nn.Linear(64, 32)   # hidden1 -> hidden2
        self.fc2 = nn.Linear(32, 10)   # hidden2 -> output (10 classes)
        
    def forward(self, x):
        # x shape: (batch_size, 784)
        x = self.fc0(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # Use CrossEntropyLoss externally or log_softmax here
        return x  # raw logits
```

> Note: We return **raw logits** so we can apply **CrossEntropyLoss** later. Alternatively, we could `return torch.log_softmax(x, dim=1)` and use **NLLLoss**.

---

## 3. Training Function with Flexible Optimizer & LR

We define a function that:
- Accepts the **optimizer name** (e.g., `"SGD"`, `"RMSprop"`, `"Adam"`) and **learning rate**.
- Creates the corresponding PyTorch optimizer dynamically using `getattr`.
- Trains for a certain number of epochs.
- Returns **train accuracy** and **test accuracy**.

```python
def train_model(optimizer_name, learning_rate, train_loader, test_loader, epochs=20):
    # 1) Instantiate model
    model = SimpleMNIST()
    
    # 2) Loss function
    criterion = nn.CrossEntropyLoss()
    
    # 3) Create optimizer using getattr
    OptimClass = getattr(optim, optimizer_name)
    optimizer = OptimClass(model.parameters(), lr=learning_rate)
    
    # 4) Training loop
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    
    # Compute train accuracy
    model.eval()
    correct_train, total_train = 0, 0
    with torch.no_grad():
        for Xb, yb in train_loader:
            logits = model(Xb)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == yb).sum().item()
            total_train   += len(yb)
    train_acc = correct_train / total_train
    
    # Compute test accuracy
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for Xb_t, yb_t in test_loader:
            logits_t = model(Xb_t)
            preds_t = torch.argmax(logits_t, dim=1)
            correct_test += (preds_t == yb_t).sum().item()
            total_test   += len(yb_t)
    test_acc = correct_test / total_test
    
    return train_acc, test_acc
```

---

## 4. Experiment Setup

We test:
1. **Optimizers** = \(\{\text{"SGD"}, \text{"RMSprop"}, \text{"Adam"}\}\)
2. **Learning Rates** = 6 values in \(\log\)-scale from \(10^{-4}\) to \(10^{-1}\).  
   - For example, \([0.0001, 0.001, 0.01, 0.1]\). We can use `np.logspace`.

```python
import numpy as np

optimizers_to_try = ["SGD", "RMSprop", "Adam"]
lr_values = np.logspace(-4, -1, 6)  # [0.0001, 0.000251..., 0.00063..., 0.00158..., 0.00398..., 0.01, 0.1]

results = {}

for opt_name in optimizers_to_try:
    results[opt_name] = []
    for lr in lr_values:
        print(f"Training with optimizer={opt_name}, learning_rate={lr:.4f}")
        train_acc, test_acc = train_model(
            optimizer_name=opt_name,
            learning_rate=lr,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=20
        )
        results[opt_name].append((lr, train_acc, test_acc))

print("Experiment complete!")
```

> This will run 3 optimizers \(\times\) 6 LRs = 18 total runs. Each run trains for 20 epochs.

---

## 5. Plotting the Results

We typically plot **test accuracy** (and optionally train accuracy) as a function of the learning rate, with separate lines for each optimizer.

```python
plt.figure(figsize=(10,6))

for opt_name in optimizers_to_try:
    lr_vals = [r[0] for r in results[opt_name]]
    test_accs = [r[2] for r in results[opt_name]]  # index=2 => test_acc
    plt.plot(lr_vals, test_accs, marker='o', label=opt_name)

plt.xscale("log")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Test Accuracy")
plt.title("Effect of Optimizer & Learning Rate on MNIST Classification")
plt.legend()
plt.grid(True)
plt.show()
```

> **Interpretation**:
> - For **SGD**, best results often happen at a **medium** or **larger** learning rate range, since small LRs might stall training.
> - **RMSprop** and **Adam** can do well for a broader range of lower LRs (they automatically adapt step sizes internally).

---

## 6. Typical Results

A common pattern:

- **SGD** might struggle with very low LRs (\(\leq 10^{-3}\)) because it doesn’t adapt learning rates on the fly, so training might be too slow or effectively stall at lower LR.  
- **RMSprop** and **Adam** often yield **similar** or **slightly better** performance, especially for smaller LRs, thanks to their **adaptive** nature.  
- Usually, there’s a **peak** region where test accuracy is highest before dropping off at too large an LR (training diverges) or too small an LR (not enough progress in limited epochs).

You might see a plot where **RMSprop** and **Adam** lines are grouped together, outperforming **SGD** in some LR regions, but possibly **SGD** might match them at a higher LR.

---

## 7. Discussion & Key Takeaways

1. **Adaptive Optimizers**:  
   - **Adam** & **RMSprop** scale gradients differently per parameter, improving convergence in many tasks.  
   - They’re more robust to suboptimal initial LRs, though extremely large LR can still break training.

2. **SGD**  
   - Requires careful **LR** tuning or **learning rate schedules** (momentum, decay) to perform competitively.  
   - Can work extremely well if properly tuned for large-scale tasks.

3. **No Universal Best**  
   - While **Adam** or **RMSprop** often converge faster, some tasks still favor carefully tuned SGD.  
   - Real-world experiments frequently combine Adam with a **learning-rate scheduler** to refine training.

4. **Experimentation**  
   - Deep learning demands **empirical** trials: each dataset, architecture, and time budget can lead to different optimal choices.

---

## 8. Further Explorations

- **Try More Epochs**: See if low LRs eventually catch up.  
- **Compare Training Curves**: Plot loss vs. epoch for each optimizer/LR to visualize convergence speed.  
- **Momentum**: Evaluate **SGD with momentum** to see if that narrows the gap with Adam/RMSprop.  
- **Learning-Rate Schedulers**: Gradually reduce LR during training (e.g., StepLR, ExponentialLR, Cosine Annealing) to see if test accuracy improves.

---

**End of Notes – "CodeChallenge: Optimizers and MNIST"** 
```