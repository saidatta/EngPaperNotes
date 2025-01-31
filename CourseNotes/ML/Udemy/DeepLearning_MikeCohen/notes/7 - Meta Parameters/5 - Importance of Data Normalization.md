
aliases: [Importance of Normalization, Effects of Normalization]
tags: [Deep Learning, Lecture Notes, Data Preprocessing, Neural Networks]

In this lesson, we **empirically demonstrate** the powerful effects of **data normalization**. By toggling on/off data normalization (specifically **z-scoring**) in existing code examples, we observe **significant changes** in how quickly and effectively neural networks train and generalize.

---

## 1. Overview

- **Motivation**: Even if a dataset trains *eventually* without normalization, normalizing can:
  1. **Accelerate** training convergence (faster learning from the first few epochs).
  2. **Improve** final test accuracy and generalization (fewer “quirks” in weight updates).
  3. **Stabilize** gradient updates (less overfitting or explosive weight behavior).

- **Demonstrations**: We revisit two prior notebooks:
  1. **Regularization** notebook (`regular_minibatch.py`) using the **Iris** dataset.  
  2. **Mini-Batch Size** notebook (`codechallenge_minibatch_size.py`) using the **Wine Quality** dataset.

For each, we **switch** data normalization **on/off** and compare results.

---

## 2. Iris Dataset Example

### 2.1 Original Code (No Normalization)

In the `regular_minibatch.py` script, we initially had **data normalization commented out**:

```python
# (Loading the Iris dataset)
data = pd.read_csv("iris.csv")

# Visualize data
plt.scatter(data["petal_length"], data["sepal_width"], c=data["species"])
plt.title("Iris data (unscaled)")
plt.show()

# Normalization line was commented out:
# data[["petal_length", "petal_width", "sepal_length", "sepal_width"]] = stats.zscore(...)

# Convert to PyTorch, train...
```

- **Observation**: Without normalization, the model eventually learns decent accuracy, but:
  - It may **start slowly** (low accuracy in early epochs).
  - Potentially more **overfitting** or general instability.

### 2.2 Turning On Z-Score Normalization

We can **uncomment** (or add) the following lines:

```python
from scipy import stats

cols_to_normalize = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
data[cols_to_normalize] = stats.zscore(data[cols_to_normalize])
```

Now each feature (petal length, etc.) has:
- Mean \(\approx 0\)
- Std \(\approx 1\)

### 2.3 Empirical Results

![Iris Normalization Comparison](https://dummyimage.com/600x200/ccc/000.png&text=Placeholder)

1. **Loss**: 
   - With normalization: The loss curve **drops quickly** in early epochs. 
   - Without normalization: The loss might fluctuate more in the initial epochs.

2. **Accuracy**: 
   - With normalization: The **initial** accuracy often starts higher and climbs faster.
   - Final accuracy might be **slightly higher** (or achieved in fewer epochs) with normalization.

**Conclusion**: Even though Iris dataset is relatively small and “easy,” the difference is still noticeable—**faster** training and **less overfitting**.

---

## 3. Wine Quality Dataset Example

### 3.1 Parametric Experiment on Mini-Batch Size

Recall the **codechallenge_minibatch_size.py** which:
- Took ~15 minutes to run (testing multiple batch sizes).
- Already had **z-score normalization** turned **on** by default.

```python
# original code snippet
feature_cols = [c for c in data.columns if c != "quality"]
for col in feature_cols:
    col_mean = data[col].mean()
    col_std  = data[col].std()
    data[col] = (data[col] - col_mean) / col_std
```

This gave us **~80%** test accuracy for certain batch sizes.

### 3.2 Disabling Normalization

By commenting out or removing the z-score lines:

```python
# for col in feature_cols:
#    col_mean = data[col].mean()
#    col_std  = data[col].std()
#    data[col] = (data[col] - col_mean) / col_std
```

We let the raw Wine Quality features remain at their **original scales** (which vary drastically).

### 3.3 Observed Differences

- **Without Normalization**:
  1. Training accuracy **remains lower** or grows **very slowly**.  
  2. Test accuracy might plateau around **70%** instead of **80%**.  
  3. The network needs **more epochs** to reach comparable accuracy.

- **With Normalization**:
  1. Training accuracy can reach **~100%** for smaller batch sizes (though some overfitting).  
  2. Test accuracy around **80%**—significantly better performance.  
  3. Less time/epochs needed for stable convergence.

**Plot Example**:

| **Batch Size** | Normalized Accuracy (Test) | Non-Normalized Accuracy (Test) |
|----------------|----------------------------|--------------------------------|
| 2              | ~78% - 80%                | ~65% - 70%                     |
| 8              | ~80%                      | ~68% - 72%                     |
| 128            | ~75%                      | ~70%                           |

*(Exact numbers may vary slightly by random seed, but the trend is consistent.)*

---

## 4. Why This Matters

1. **Faster Initial Convergence**: The gradient steps are more **consistent**, as features do not overshadow each other.  
2. **Higher Final Accuracy**: Proper scaling **improves generalization**.  
3. **Fewer Epochs** (Time/Resource Savings): Normalized data typically require fewer updates to reach similar or better performance.

### 4.1 Practice Tip

> **“Just normalize your data!”** – In practice, it’s generally safer and more efficient to normalize or standardize your inputs by default.

---

## 5. Example Implementation Code

Below is a **minimal** structure combining these lessons:

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

# 1. Load Wine Quality data
data = pd.read_csv("winequality-red.csv", sep=";")

# 2. OPTIONAL: Remove outliers or handle missing data if needed
data = data.loc[data["total sulfur dioxide"] < 200]

# 3. Switch ON/OFF normalization
normalize_data = True
feature_cols = [c for c in data.columns if c != "quality"]

if normalize_data:
    for col in feature_cols:
        col_mean = data[col].mean()
        col_std = data[col].std()
        data[col] = (data[col] - col_mean) / col_std

# 4. Convert to PyTorch
X = torch.tensor(data[feature_cols].values, dtype=torch.float32)
y = torch.tensor((data["quality"] > 5).values, dtype=torch.long)

# 5. Create Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=0.2, random_state=42)

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

# 6. Simple Model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Train Loop
epochs = 50
for epoch in range(epochs):
    model.train()
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(Xb).squeeze()
        loss = criterion(logits, yb.float())
        loss.backward()
        optimizer.step()

    # Evaluate each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            logits = model(Xb).squeeze()
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f"Epoch {epoch+1}/{epochs}, Test Acc: {correct/total:.2f}")
```

> **Try** running with `normalize_data = True` vs. `False` to see how your accuracy changes.

---

## 6. Conclusions

1. **Normalization** consistently **improves**:
   - Training **speed** (initial performance).  
   - **Stability** (reduced overfitting or chaotic weight updates).  
   - Potentially **higher** final test accuracy.  

2. **Practical Advice**:
   - Always consider standardizing or min-max scaling your data.  
   - If your features are on drastically different scales, normalization is **essential**.  
   - If data is already in a tight, consistent range, the gain might be less dramatic—but it rarely **hurts**.

3. **Future Topics**:
   - **Batch Normalization**: Normalization *inside* the network layers to further stabilize training.  
   - **Other normalization methods** (layer norm, group norm, etc.).

---

## 7. Further Reading & References

- [Goodfellow, Bengio, Courville – *Deep Learning*]  
- [PyTorch `torchvision.transforms`](https://pytorch.org/vision/stable/transforms.html) for image normalizations  
- [Scikit-Learn `StandardScaler` & `MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: “Meta parameters: Importance of Data Normalization”  
```