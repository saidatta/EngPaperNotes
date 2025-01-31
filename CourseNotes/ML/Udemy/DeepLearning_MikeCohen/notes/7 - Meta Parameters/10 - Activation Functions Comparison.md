aliases: [Activation Functions Comparison, Activation Benchmark, Wine Quality]
tags: [Deep Learning, Meta Parameters, Neural Networks, PyTorch]

This note demonstrates an **empirical comparison** of three different activation functions (**ReLU**, **Tanh**, and **Sigmoid**) on the **Wine Quality** dataset. We observe how each choice affects **training dynamics** and **final performance**.

---
## 1. Overview

1. **Dataset**: [UCI Wine Quality (Red Wine)](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
2. **Goal**: Binary classification (wine quality “above 5” vs. “5 or below”).  
3. **Activations** to compare in **hidden layers**:
   - **ReLU**
   - **Tanh**
   - **Sigmoid**

**Note**: The **output layer** is effectively **sigmoid** for all models since this is a binary classification problem. The difference lies in the **internal** (input & hidden) layers.

---

## 2. Imports & Data Preprocessing

We import **PyTorch** for building the model and **pandas/numpy** for data manipulation. The **Wine Quality** dataset is loaded directly from UCI, outliers removed, features z-scored, and then split into training/testing sets.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# 2. Optional: Remove extreme outliers
data = data.loc[data["total sulfur dioxide"] < 200]

# 3. Z-score features
feature_cols = [c for c in data.columns if c != "quality"]
for col in feature_cols:
    mu, sd = data[col].mean(), data[col].std()
    data[col] = (data[col] - mu) / sd

# 4. Binarize quality
data["label"] = (data["quality"] > 5).astype(int)

# Convert to Tensors
X = torch.tensor(data[feature_cols].values, dtype=torch.float32)
y = torch.tensor(data["label"].values, dtype=torch.long)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# DataLoaders
train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

print("Data ready! Train size:", len(train_ds), "Test size:", len(test_ds))
```

---

## 3. Defining the Model with a Configurable Activation

We build a **two-hidden-layer** MLP. The **key** difference is we pass in the string `actfun` (one of `"relu"`, `"tanh"`, `"sigmoid"`) and resolve it to the appropriate function in the forward pass.

```python
class WineNet(nn.Module):
    def __init__(self, actfun="relu"):
        super(WineNet, self).__init__()
        
        # Architecture: (11 -> 16 -> 32 -> 1)
        self.input = nn.Linear(11, 16)
        self.fc1   = nn.Linear(16, 32)
        self.output= nn.Linear(32, 1)
        
        # Save the activation function name
        self.actfun_name = actfun
        
    def forward(self, x):
        # Convert the string into the PyTorch functional call, e.g., torch.relu, torch.tanh, ...
        actfun = getattr(torch, self.actfun_name)
        
        x = actfun(self.input(x))
        x = actfun(self.fc1(x))
        
        # Output layer: no explicit activation here (we'll use BCEWithLogitsLoss or a final sigmoid for classification)
        x = self.output(x)
        return x

# Quick test
model = WineNet(actfun="relu")
test_input = torch.randn(10, 11)  # 10 samples, 11 features
test_output = model(test_input)
print("Model output shape:", test_output.shape)
# Should be [10, 1]
```

---

## 4. Training Function

We create a generic helper function that:

1. Instantiates `WineNet` with a chosen activation function.
2. Trains for a given number of epochs.
3. Tracks **train accuracy** and **test accuracy** per epoch.

```python
def train_model(actfun, epochs=1000):
    model = WineNet(actfun=actfun)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    train_acc_history = []
    test_acc_history  = []
    
    for ep in range(epochs):
        # Training
        model.train()
        correct, total = 0, 0
        
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb).squeeze()  # shape [batch_size]
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            
            # Accuracy on train batch
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
            
        train_acc = correct / total
        train_acc_history.append(train_acc)
        
        # Validation
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

We **loop** over the chosen activation functions, train, and record accuracy curves.

```python
import time

actfuns = ["relu", "tanh", "sigmoid"]
results = {}

start_time = time.time()
for af in actfuns:
    print(f"Training model with {af.upper()} activation...")
    outcome = train_model(af, epochs=1000)
    results[af] = outcome
print("Total experiment time: {:.1f} seconds".format(time.time() - start_time))
```

*(This might take a couple of minutes depending on your machine.)*

---

## 6. Visualizing Results

### 6.1 Training Accuracy Curves

```python
epochs = range(1, 1001)

plt.figure(figsize=(12,5))

for af in actfuns:
    plt.plot(epochs, results[af]["train_acc"], label=f"{af} train")

plt.title("Train Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

### 6.2 Test Accuracy Curves

```python
plt.figure(figsize=(12,5))

for af in actfuns:
    plt.plot(epochs, results[af]["test_acc"], label=f"{af} test")

plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

**Observations** (typical pattern):
- **Sigmoid** may lag behind (often due to vanishing gradients in deeper layers).
- **ReLU** and **Tanh** often converge faster and/or higher.  
- The final test accuracy might still be somewhat close, but **Sigmoid** typically takes longer to get there.

---

## 7. Empirical Findings

From sample runs, we often see:

1. **Sigmoid**:
   - Slow to train (training accuracy remains low for many epochs).
   - Eventually it can reach a decent accuracy but requires more time/epochs.
2. **ReLU**:
   - Tends to converge faster.
   - Sometimes yields the highest train accuracy.
3. **Tanh**:
   - Also can converge relatively quickly, sometimes similar to ReLU.
   - Might saturate or plateau earlier than ReLU depending on the data.

In **this** Wine Quality dataset scenario, ReLU and Tanh often **outperform** Sigmoid in the hidden layers, but their final test accuracies might end up close.

---

## 8. Remarks on Validation vs. Test

Since we are **varying a meta-parameter** (the hidden-layer activation), we are effectively **tuning** our architecture. This means:
- The set we call “test” in this code is more like a **dev/validation** set for activation choice.  
- Once we pick the **best** activation (e.g., ReLU), in a **real project** we would retrain on the combined train+dev and finally test on a **true hold-out** set to get an unbiased final score.

---

## 9. Conclusion

- **Sigmoid** in hidden layers often **underperforms** due to gradient issues.
- **ReLU** (and sometimes **Tanh**) typically yields **faster** training and at least comparable final accuracy.
- In practice, **ReLU** is the **default** choice for deep MLPs, with exceptions depending on domain or specialized architectures.

**Next Steps**:
- Experiment with **other** architectures or add **Batch Normalization** to see if it changes the ranking of activation functions.
- Try **LeakyReLU**, **ReLU6**, or **Swish/GELU** if you suspect ReLU has “dead neurons” or if you want to see alternative performance.

---

## 10. References

- **Deep Learning** (Goodfellow et al.), *Chapter 6*.  
- **Rectified Linear Units** – [Nair & Hinton, 2010](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf).  
- PyTorch Activation Docs: [Link](https://pytorch.org/docs/stable/nn.html#non-linear-activations)

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: *“Meta parameters: Activation Functions comparison”*  
```