aliases: [Wine Sugar Prediction, Code Challenge, Regression with MLP]
tags: [Deep Learning, Meta Parameters, Neural Networks, PyTorch]

In this challenge, we **shift** from predicting wine quality (a **classification** task) to predicting **residual sugar** (a **regression** task) using the same **Wine Quality** dataset. The main skills tested are:

1. **Re-purposing** existing code (originally for binary classification) to handle a **regression** problem.
2. **Selecting** the correct **loss function** and **output layer** for regression.
3. **Exploring** model performance through **plots** and **correlation analysis**.

---

## 1. Overview

- **Original Goal**: Predict if wine is *good* (binary label).  
- **New Goal**: Predict the **residual sugar** (\( \text{residual\_sugar} \)) value from the other chemical features.  
- **Data**: [UCI Red Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).  
- **Approach**:  
  - Keep a similar multi-layer perceptron (MLP) structure.  
  - **Regression** output → use **MSE** (Mean Squared Error) as the loss function.  
  - Evaluate performance via **loss** curves and **correlation** between predictions (\(\hat{y}\)) and true sugar (\(y\)).

---

## 2. Data Loading & Preparation

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

# 1) Load the Red Wine Quality data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# 2) Remove outliers in 'total sulfur dioxide'
data = data.loc[data["total sulfur dioxide"] < 200]

# 3) Z-score ALL columns, including 'quality'
for col in data.columns:
    mu = data[col].mean()
    sd = data[col].std()
    data[col] = (data[col] - mu) / sd

# 4) Separate features from the target 'residual sugar'
#    We will PREDICT 'residual sugar', so exclude it from X
feature_cols = [c for c in data.columns if c != "residual sugar"]
X_np = data[feature_cols].values  # shape [N, 11] (because total 12 columns minus 1 target)
y_np = data["residual sugar"].values  # shape [N,]

print("Feature columns:", feature_cols)
print("Data shapes:", X_np.shape, y_np.shape)

# Convert to PyTorch Tensors
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

print("X tensor shape:", X.shape, "| y tensor shape:", y.shape)

# Check distribution of target (residual sugar)
plt.figure(figsize=(6,3))
sns.histplot(y_np, kde=True)
plt.title("Distribution of Residual Sugar (Z-scored)")
plt.xlabel("Z-scored Residual Sugar")
plt.show()
```

- **Note**: We **z-score** every column, including `quality`, so everything is on comparable scales.  
- **Target**: `residual sugar`, now removed from `X`.

---

## 3. Train/Test Split & DataLoaders

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, drop_last=True)

print("Train samples:", len(train_ds), "Test samples:", len(test_ds))
```

---

## 4. Defining the Regression Model

We can keep a similar structure to our classification MLP, but:

1. **Output**: Only 1 unit (scalar) for the predicted sugar.  
2. Typically **no activation** on the final layer (for a **regression**).  
3. Use **ReLU** for hidden layers, though you could experiment with other activations.

```python
import torch.nn.functional as F

class WineSugarNet(nn.Module):
    def __init__(self):
        super(WineSugarNet, self).__init__()
        # Example architecture: 2 hidden layers
        self.fc1 = nn.Linear(len(feature_cols), 32)  # input_dim -> hidden_dim
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)                  # output_dim=1 (regression)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # No activation on final layer for regression
        x = self.out(x)
        return x
```

---

## 5. Training Loop

We’ll use **MSE** (mean squared error) as our loss function for regression.

```python
def train_model(epochs=1000, lr=0.001):
    model = WineSugarNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # regression -> MSE

    train_loss_history = []
    test_loss_history  = []

    for ep in range(epochs):
        # -- Training --
        model.train()
        batch_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)  # MSE
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
        train_loss_history.append(batch_loss / len(train_loader))
        
        # -- Testing --
        model.eval()
        batch_loss_test = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds_test = model(Xb)
                loss_test  = criterion(preds_test, yb)
                batch_loss_test += loss_test.item()
        test_loss_history.append(batch_loss_test / len(test_loader))

    return model, train_loss_history, test_loss_history
```

---

## 6. Run the Experiment

```python
epochs = 1000
model, train_loss, test_loss = train_model(epochs=epochs, lr=0.001)

# Plot the training vs. testing loss
plt.figure(figsize=(10,4))
plt.plot(range(1, epochs+1), train_loss, label="Train Loss")
plt.plot(range(1, epochs+1), test_loss,  label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("MSE Loss over Epochs for Wine Sugar Prediction")
plt.legend()
plt.show()
```

---

## 7. Evaluating Model Predictions

### 7.1 Train & Test Predictions vs. True Values

```python
model.eval()

# Entire training set
with torch.no_grad():
    yhat_train = model(X_train)  # shape [N_train, 1]
    yhat_test  = model(X_test)

# Detach from graph, convert to numpy for plotting
yhat_train_np = yhat_train.detach().numpy().flatten()
yhat_test_np  = yhat_test.detach().numpy().flatten()
y_train_np    = y_train.numpy().flatten()
y_test_np     = y_test.numpy().flatten()

# Scatter: Predicted vs. Actual
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(yhat_train_np, y_train_np, alpha=0.5)
plt.title("Train Pred vs. Actual")
plt.xlabel("Predicted Sugar (z-scored)")
plt.ylabel("Actual Sugar (z-scored)")

plt.subplot(1,2,2)
plt.scatter(yhat_test_np, y_test_np, alpha=0.5, color="orange")
plt.title("Test Pred vs. Actual")
plt.xlabel("Predicted Sugar (z-scored)")
plt.ylabel("Actual Sugar (z-scored)")

plt.tight_layout()
plt.show()
```

**Observation**: 
- Points close to a **diagonal** line (\(y = \hat{y}\)) → better fit.  
- More spread → less accurate.

### 7.2 Correlation Coefficients

We can compute the **Pearson correlation** between predicted and actual:

```python
def pearson_corrcoef(x, y):
    # x,y are 1D numpy arrays
    return np.corrcoef(x, y)[0,1]

corr_train = pearson_corrcoef(yhat_train_np, y_train_np)
corr_test  = pearson_corrcoef(yhat_test_np,  y_test_np)

print(f"Train correlation: {corr_train:.3f}")
print(f"Test correlation:  {corr_test:.3f}")
```

- Higher correlation (close to 1.0) indicates the model’s predictions strongly match the actual data.

---

## 8. Inspecting Feature Correlations

A **correlation matrix** of all features + `residual sugar` can show how strongly (or weakly) each variable is related:

```python
# We used 'data' which is already z-scored
corr_matrix = np.corrcoef(data.values.T)  # shape [num_features, num_features]
cols = data.columns.tolist()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", xticklabels=cols, yticklabels=cols)
plt.title("Correlation Matrix of Wine Features + Residual Sugar")
plt.xticks(rotation=90)
plt.show()
```

**Interpretation**:
- Does **residual sugar** strongly correlate with any single feature?  
- Typically, we might see a moderate correlation with `density` or `alcohol`.  
- If correlations are weak, it suggests the model must learn **non-linear** relationships across multiple features.

---

## 9. Observations & Next Steps

1. **Regression Performance**:
   - The MSE **train** loss might drop steadily; the **test** loss may level off or even slightly increase (overfitting).  
   - The correlation between predicted \(\hat{y}\) and actual \(y\) can be quite high for training, but lower for test (a sign of overfitting).

2. **Further Tuning**:
   - Adjust **model depth** or **width** (hidden layers).  
   - Use **regularization** (e.g., dropout, weight decay).  
   - Try **different optimizers** or **learning rates**.

3. **Correlation Matrix**:
   - If no single feature is strongly correlated with sugar, the network leverages **combined** non-linear patterns to achieve decent predictions.  
   - Demonstrates the power of **deep learning** in discovering relationships not evident from raw pairwise correlations.

---

## 10. Conclusion

- We **successfully repurposed** classification code to perform a **regression** on **residual sugar**.  
- Chose an **MSE** loss function and **no activation** for the final layer.  
- Observed how **overfitting** can manifest differently in regression (train correlation vs. test correlation).  
- Noted the **weak** direct correlations in the data, highlighting the **non-linear** predictive power of neural networks.

**Challenge**: Can you **beat** the instructor’s correlation on the test set by altering hyperparameters, architecture, or adding regularization?

---

## 11. References

- [UCI Wine Quality Dataset (Red)](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- [PyTorch Docs on MSE Loss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)  
- [Deep Learning Book (Goodfellow, Bengio, Courville) – Ch. 5: Machine Learning Basics (Regression)](https://www.deeplearningbook.org/)

```
Created by: [Your Name / Lab / Date]
Lecture Reference: “Meta parameters: CodeChallenge: Predict sugar”
```