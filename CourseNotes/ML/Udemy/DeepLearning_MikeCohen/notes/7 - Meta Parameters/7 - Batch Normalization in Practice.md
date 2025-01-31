
aliases: [BatchNorm in Practice, Batch Normalization Implementation]
tags: [Deep Learning, Lecture Notes, Meta Parameters, Normalization, PyTorch]

This note covers a practical **PyTorch implementation** of Batch Normalization (BN). We'll revisit the **Wine Quality** dataset, which we've used before, and augment our neural network with BN layers. We also compare training outcomes **with** and **without** BN to observe its effects on convergence and final accuracy.

---

## 1. Setup: Imports & Data Loading

We use the familiar **Wine Quality** dataset:

1. **Load** the CSV directly via URL (or from disk).  
2. **Remove outliers** and **z-score** features.  
3. **Partition** into train/test sets and **DataLoaders**.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# 1. Load the Wine Quality data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# 2. Remove outliers (optional: e.g., total sulfur dioxide > 200)
data = data.loc[data["total sulfur dioxide"] < 200]

# 3. Z-score each feature except 'quality'
feature_cols = [c for c in data.columns if c != "quality"]
for col in feature_cols:
    mean, std = data[col].mean(), data[col].std()
    data[col] = (data[col] - mean) / std

# 4. Binarize wine quality (e.g., 0 if <=5, 1 if >5)
data["label"] = (data["quality"] > 5).astype(int)

# Convert to torch Tensors
X = torch.tensor(data[feature_cols].values, dtype=torch.float32)
y = torch.tensor(data["label"].values, dtype=torch.long)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Create TensorDatasets
train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

print("Data ready! Train size:", len(train_ds), "Test size:", len(test_ds))
```

---

## 2. Defining the Model with Optional Batch Normalization

Below is a **PyTorch Module** that can switch **on/off** batch normalization through a boolean flag `doBN`.

### 2.1 Model Architecture

- **Input layer**: size 11 → hidden size 16  
- **Hidden layer 1**: 16 → 32  
- **Hidden layer 2**: 32 → 32 (or 64, up to you)  
- **Output**: 1 unit for binary classification  

**BatchNorm** is inserted after each hidden layer if `doBN` is `True`.

```python
class WineNetBN(nn.Module):
    def __init__(self):
        super(WineNetBN, self).__init__()
        
        # Fully connected layers
        self.input = nn.Linear(11, 16)
        self.fc1   = nn.Linear(16, 32)
        self.fc2   = nn.Linear(32, 32)
        self.output= nn.Linear(32, 1)
        
        # BatchNorm layers (for each hidden layer)
        self.bnorm1 = nn.BatchNorm1d(32)  # matches output dimension of self.fc1
        self.bnorm2 = nn.BatchNorm1d(32)  # matches output dimension of self.fc2
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x, doBN=True):
        # 1) Input layer
        x = self.input(x)
        x = self.relu(x)
        
        # 2) First hidden layer
        x = self.fc1(x)
        x = self.relu(x)
        # Conditionally apply BatchNorm
        if doBN:
            x = self.bnorm1(x)
        
        # 3) Second hidden layer
        x = self.fc2(x)
        x = self.relu(x)
        if doBN:
            x = self.bnorm2(x)
        
        # 4) Output layer (no BN here)
        x = self.output(x)
        return x
```

**Note**: 
- We keep `doBN` as a parameter to let us **toggle** batch normalization.  
- If `doBN=False`, it simply **bypasses** the `self.bnorm1/2` calls.

---

## 3. Training Function

We’ll write a helper function `train_model()` that accepts a boolean `doBN` to determine whether we pass that flag as `True/False` during each forward pass.

```python
def train_model(doBN=True, epochs=100):
    # Instantiate the model
    model = WineNetBN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Storage
    loss_history = []
    train_acc_history = []
    test_acc_history  = []
    
    for ep in range(epochs):
        # Training
        model.train()  # BN uses batch statistics here
        total_correct = 0
        total_samples = 0
        total_loss    = 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            # forward pass with BN toggled by doBN
            logits = model(Xb, doBN=doBN).squeeze()
            
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_correct += (preds == yb).sum().item()
            total_samples += yb.size(0)
        
        avg_loss   = total_loss / len(train_loader)
        train_acc  = total_correct / total_samples
        
        # Validation/Test
        model.eval()  # BN uses running averages here
        test_correct = 0
        test_samples = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                logits_test = model(Xb, doBN=doBN).squeeze()
                preds_test  = (torch.sigmoid(logits_test) > 0.5).long()
                
                test_correct += (preds_test == yb).sum().item()
                test_samples += yb.size(0)
        test_acc = test_correct / test_samples
        
        loss_history.append(avg_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        
    return {
        "loss": loss_history,
        "train_acc": train_acc_history,
        "test_acc": test_acc_history
    }
```

---

## 4. Running the Experiment: BN On vs. BN Off

We’ll run two training sessions:
1. **No BN**: `doBN=False`
2. **BN Enabled**: `doBN=True`

Then, we plot and compare.

```python
results_no_bn   = train_model(doBN=False, epochs=100)
results_with_bn = train_model(doBN=True,  epochs=100)
```

### 4.1 Plotting the Results

```python
epochs = range(1, 101)

plt.figure(figsize=(15,4))

# ---- Loss Plot
plt.subplot(1,3,1)
plt.plot(epochs, results_no_bn["loss"], label="No BN", color="red")
plt.plot(epochs, results_with_bn["loss"], label="With BN", color="blue")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# ---- Train Accuracy
plt.subplot(1,3,2)
plt.plot(epochs, results_no_bn["train_acc"], label="No BN", color="red")
plt.plot(epochs, results_with_bn["train_acc"], label="With BN", color="blue")
plt.title("Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# ---- Test Accuracy
plt.subplot(1,3,3)
plt.plot(epochs, results_no_bn["test_acc"], label="No BN", color="red")
plt.plot(epochs, results_with_bn["test_acc"], label="With BN", color="blue")
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
```

**Example Outcome**:

- **Loss**: Typically **faster decrease** with BN.  
- **Train Accuracy**: BN often shows higher or more stable accuracy earlier.  
- **Test Accuracy**: Gains are dataset/model-dependent; often improved, but can vary.

---

## 5. Observations & Notes

1. **Training Speed**:  
   - BN can lead to a **faster** drop in loss, especially in the early epochs.  
   - This is because BN stabilizes the magnitude of activations, reducing internal covariate shift.

2. **Test Accuracy**:  
   - Sometimes BN shows a clear improvement.  
   - In other runs or smaller networks, the improvement might be modest or occasionally even negative due to stochastic effects.  

3. **Model Depth**:  
   - This example is a **2-hidden-layer** MLP. BN is often **more crucial** in deeper networks (e.g., 10+ layers) but can still help here.

4. **Toggle**:
   - We used `model(Xb, doBN=True/False)` as a quick demonstration.  
   - In practice, you’d likely keep BN **always on** once you decide to use it.

---

## 6. Key Takeaways

- **Implementation Simplicity**: Adding BN in PyTorch is straightforward:
  - **Add** `nn.BatchNorm1d(layer_dim)` after each linear layer you want to normalize.
  - Insert BN calls in the `forward` pass.
- **Train vs. Eval**: 
  - `model.train()` uses mini-batch mean/variance.
  - `model.eval()` uses running mean/variance accumulated during training.
- **Performance Gains**: 
  - BN typically reduces **training loss** faster and can yield higher final accuracy.  
  - Gains in test accuracy may vary depending on dataset complexity and network depth.

---

## 7. Further Resources

1. **Original Paper**: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)  
2. **PyTorch Documentation**: [BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)  
3. **Other Normalizations**: Layer Norm, Group Norm, Instance Norm.  

---

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: “Meta parameters: Batch Normalization in practice”  
```