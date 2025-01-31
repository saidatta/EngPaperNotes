
aliases: [Code Challenge: ReLU Variants, Activation Functions, Wine Quality]
tags: [Deep Learning, Meta Parameters, Neural Networks, PyTorch]

This challenge extends our exploration of **activation functions** by comparing **ReLU variants** (classic **ReLU**, **ReLU6**, **LeakyReLU**) on the **Wine Quality** dataset. The aim is to see how these slight changes in activation behavior affect **training convergence** and **final accuracy**.

---
## 1. Overview

1. **Dataset**: Same [UCI Wine Quality (Red)](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) dataset.  
2. **Architecture**: A multi-layer perceptron (two hidden layers).  
3. **Activation Variants**:
   - **`nn.ReLU`** (standard ReLU)  
   - **`nn.ReLU6`** (clamps outputs at 6)  
   - **`nn.LeakyReLU`** (allows a small negative slope)

4. **Objective**: Compare training/test accuracies over epochs, see if any variant consistently learns faster or reaches higher accuracy.

---

## 2. Data Loading & Preprocessing

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load Wine Quality data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# 2) Remove outliers
data = data.loc[data["total sulfur dioxide"] < 200]

# 3) Z-score features (excluding 'quality')
feature_cols = [col for col in data.columns if col != "quality"]
for col in feature_cols:
    mean, std = data[col].mean(), data[col].std()
    data[col] = (data[col] - mean) / std

# 4) Binarize quality -> boolQuality
# Option A (may cause SettingWithCopyWarning)
data["boolQuality"] = 0
data.loc[data["quality"] > 5, "boolQuality"] = 1

# Option B (avoids warning)
# bool_qual = (data["quality"] > 5).astype(int)
# data["boolQuality"] = bool_qual

# Convert to Tensors
X_np = data[feature_cols].values
y_np = data["boolQuality"].values

X_t = torch.tensor(X_np, dtype=torch.float32)
y_t = torch.tensor(y_np, dtype=torch.long)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_t, y_t, test_size=0.2, random_state=42, shuffle=True
)

# Create DataLoaders
train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, drop_last=True)

print("Data shapes:", X_train.shape, X_test.shape)
print("Data ready for training/testing!")
```

- **Batch Size**: 64 (arbitrary choice).  
- **drop_last=True**: ensures each mini-batch has exactly 64 samples (discarding leftovers).

---

## 3. Defining the Model (Using `torch.nn` Activations)

The key difference:  
- We specify an activation name (`"ReLU"`, `"ReLU6"`, or `"LeakyReLU"`)  
- Dynamically retrieve the **nn module** for that activation (e.g., `nn.ReLU`, `nn.LeakyReLU`)  
- Instantiate it, then apply in the `forward` pass.

```python
class WineNet(nn.Module):
    def __init__(self, actfun="ReLU"):
        """
        actfun can be any valid torch.nn activation function name, e.g.,
        'ReLU', 'ReLU6', 'LeakyReLU', etc.
        """
        super(WineNet, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(11, 16)
        self.fc2 = nn.Linear(16, 32)
        self.out = nn.Linear(32, 1)
        
        # Save chosen activation name
        self.actfun_name = actfun
        
        # If the user picks LeakyReLU, we might specify a slope
        # Alternatively, we can parse "LeakyReLU(0.1)" in code if we want parameterization
        # but let's keep it simple for this challenge.
    
    def forward(self, x):
        # 1) Retrieve the activation class from torch.nn by name
        #    e.g. 'ReLU' -> nn.ReLU, 'LeakyReLU' -> nn.LeakyReLU, etc.
        ActClass = getattr(nn, self.actfun_name)  
        
        # 2) Instantiate it. For LeakyReLU, default negative_slope = 0.01
        #    If you want a custom slope, you'd parse that out or define separately.
        actfun = ActClass()
        
        # Pass data through the layers + activations
        x = actfun(self.fc1(x))
        x = actfun(self.fc2(x))
        
        # Output layer has no additional non-linear activation here
        x = self.out(x)
        return x

# Quick check
test_model = WineNet(actfun="ReLU")
test_input = torch.randn(5, 11)
test_output = test_model(test_input)
print(f"Output shape: {test_output.shape}")  # Expect [5,1]
```

---

## 4. Training Function

We’ll create a helper function that:
1. Builds a `WineNet` with the chosen activation.
2. Trains for a certain number of epochs.
3. Records train/test accuracies per epoch.

```python
def train_model(actfun_name, epochs=1000):
    model = WineNet(actfun=actfun_name)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    train_acc_history = []
    test_acc_history  = []
    
    for ep in range(epochs):
        # -- Train Loop --
        model.train()
        correct, total = 0, 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb).squeeze()
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            
            # Accuracy on train batch
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
        train_acc = correct / total
        train_acc_history.append(train_acc)
        
        # -- Test Loop --
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                test_logits = model(Xb).squeeze()
                test_preds  = (torch.sigmoid(test_logits) > 0.5).long()
                correct_test += (test_preds == yb).sum().item()
                total_test   += yb.size(0)
        test_acc = correct_test / total_test
        test_acc_history.append(test_acc)
    
    return {
        "train_acc": train_acc_history,
        "test_acc": test_acc_history
    }
```

---

## 5. Running the Experiment

We compare **3** different ReLU variants:

1. **`ReLU`** (classic rectified linear unit)  
2. **`ReLU6`** (clamps outputs at 6)  
3. **`LeakyReLU`** (default negative slope, typically 0.01)

```python
actfuns = ["ReLU", "ReLU6", "LeakyReLU"]
results = {}

for af in actfuns:
    print(f"Training WineNet with {af}...")
    outcome = train_model(af, epochs=1000)
    results[af] = outcome
    print("Done!\n")
```

*(This can take a minute or two depending on your hardware.)*

---

## 6. Visualizing Accuracy

### 6.1 Training Accuracy

```python
epochs = range(1, 1001)

plt.figure(figsize=(10,4))
for af in actfuns:
    plt.plot(epochs, results[af]["train_acc"], label=f"{af} - train")
plt.title("Training Accuracy (ReLU Variants)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

### 6.2 Test Accuracy

```python
plt.figure(figsize=(10,4))
for af in actfuns:
    plt.plot(epochs, results[af]["test_acc"], label=f"{af} - test")
plt.title("Test Accuracy (ReLU Variants)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

**Observations**:
- Sometimes **ReLU6** or **LeakyReLU** may converge faster or higher initially.  
- By ~final epochs, differences in test accuracy may diminish, but any of these variants might show better **early** performance.  
- **LeakyReLU** can help avoid “dead ReLUs” since negative inputs get a small slope.

---

## 7. Discussion & Observations

1. **Early Convergence**: Some runs show **ReLU6** or **LeakyReLU** learning faster in the first few hundred epochs.  
2. **Final Accuracy**: Over 1000 epochs, differences can become smaller, but sometimes the variants maintain a slight edge.  
3. **Random Initialization**: Running multiple times may yield slightly different outcomes. ReLU might sometimes catch up or even surpass the others, or the reverse.  
4. **Takeaway**: In many networks, classic **ReLU** suffices, but **Leaky** or **ReLU6** can handle certain edge cases or improve training speed.

---

## 8. Extensions & Experiments

1. **Try Different Neg Slope** in `LeakyReLU`:
   ```python
   # e.g. nn.LeakyReLU(negative_slope=0.1)
   ```
2. **Add BatchNorm**: See if BN interacts differently with these ReLU variants.  
3. **Check Computation Time**: Sometimes ReLU6 is marginally faster or slower on certain hardware.

---

## 9. Conclusion

In this code challenge, you:

- **Practiced** using `torch.nn` activation modules (instead of `torch.functional` calls).  
- **Compared** ReLU variants on a real dataset.  
- **Observed** how minor changes in activation shapes can affect learning curves, especially early on.

While **ReLU** remains a default for many tasks, **LeakyReLU** and **ReLU6** are viable alternatives that may provide quicker convergence or better handling of negative inputs. Empirical trials, as always, are essential to confirm which works best in a given scenario.

---

## 10. References

- [PyTorch Docs: Activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations)  
- [He et al. (2015) “Delving Deep into Rectifiers...”](https://arxiv.org/abs/1502.01852) (init + ReLU discussion)  
- [Maas et al. (2013) “Rectifier Nonlinearities Improve Neural Networks”](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf) (Leaky ReLU)

```
Remember: This type of experimentation is a standard part of model tuning: 
Try different activation functions (a meta parameter!) and compare on your dev set, 
then pick the best one for your final test or deployment.
```
```