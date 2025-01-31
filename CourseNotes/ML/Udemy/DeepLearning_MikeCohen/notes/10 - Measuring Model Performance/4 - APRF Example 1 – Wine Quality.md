title: "Measuring Model Performance: APRF Example 1 (Wine Quality)"
tags: [deep-learning, wine-quality, binary-classification, performance-metrics]
## 1. Overview

In this lecture, we apply the **ARPF metrics**—Accuracy, Recall, Precision, and F1—to a **real-world binary classification problem**: predicting wine quality. This dataset is **binary** (high-quality vs. low-quality wine), making it straightforward to demonstrate confusion matrices and ARPF measures.

### Key Points
- We use the **Wine Quality** dataset (commonly from UCI ML repository).
- The classification is **binary** (good vs. not-good wine).
- We compute Accuracy, Precision, Recall, and F1 directly via **scikit-learn**.
- We observe how **model bias** (e.g., predicting "good" too often) affects Precision vs. Recall.

---

## 2. Dataset Setup and Preprocessing

The dataset typically has:
- **Wine features** (e.g., acidity, chlorides, sugar, etc.).
- **Quality rating** from wine experts (integer scale).
- We **convert** the original multi-level rating into **binary**: good (≥7) or not good (<7).

### 2.1. Example Data Loading
```python
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# For scikit-learn metrics
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assume we have a function to load wine data
# X, y = load_wine_data()  # X ~ features, y ~ labels (0 or 1)

# Example: standardize features
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    Xz, y, test_size=0.10, random_state=42
)

# Convert to PyTorch tensors
X_train_t  = torch.tensor(X_train, dtype=torch.float32)
X_test_t   = torch.tensor(X_test,  dtype=torch.float32)
y_train_t  = torch.tensor(y_train, dtype=torch.float32)
y_test_t   = torch.tensor(y_test,  dtype=torch.float32)

# Confirm shapes
print(f"Train set: {X_train_t.shape}, Test set: {X_test_t.shape}")
```
```
**Note**: The **binary label** creation step might involve something like `y = (quality >= 7).astype(int)`.

---

## 3. Model Architecture and Training

We implement a simple **feed-forward network** (logistic or small MLP) to keep the example straightforward. Below is an illustration:

### 3.1 Model Definition
```python
```python
class WineNet(nn.Module):
    def __init__(self, n_features, n_hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        # Output a single raw value (no sigmoid/softmax)
        x = self.fc2(x)
        return x

model = WineNet(n_features=X_train_t.shape[1])
print(model)
```
```

### 3.2 Training Setup
```python
```python
def train_model(model, Xtrain, ytrain, learning_rate=0.01, epochs=1000):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    losses = []
    for epoch in range(epochs):
        # Forward pass
        y_hat = model(Xtrain).squeeze()   # shape: [batch_size]
        
        # Compute loss
        loss = loss_fn(y_hat, ytrain)
        losses.append(loss.item())
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return losses

# Instantiate and train
model = WineNet(n_features=X_train_t.shape[1])
losses = train_model(model, X_train_t, y_train_t, learning_rate=0.01, epochs=1000)

# Quick look at training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```
```

---

## 4. Predictions and Scikit-learn Metrics

After training, we obtain predictions from the model. We note:
- **No sigmoid**: The output is a **raw logit**.  
- We convert logits to binary predictions by checking if `output > 0`.  

### 4.1 Generating Predictions
```python
```python
# Get raw outputs for the entire training set
train_logits = model(X_train_t).detach().squeeze()  # shape: [num_train_samples]
# Convert logits to binary predictions
train_preds = (train_logits > 0).float()

# Same for test set
test_logits = model(X_test_t).detach().squeeze()
test_preds = (test_logits > 0).float()

print("Train logits shape:", train_logits.shape)
print("Test logits shape:",  test_logits.shape)
```
```

---

## 5. Accuracy, Precision, Recall, and F1

We use **scikit-learn’s** `metrics` to compute these:

```python
```python
# Convert y_train_t, y_test_t to numpy for sklearn
y_train_np = y_train_t.detach().numpy()
y_test_np  = y_test_t.detach().numpy()

train_preds_np = train_preds.detach().numpy()
test_preds_np  = test_preds.detach().numpy()

acc_train   = skm.accuracy_score(y_train_np, train_preds_np)
prec_train  = skm.precision_score(y_train_np, train_preds_np)
rec_train   = skm.recall_score(y_train_np, train_preds_np)
f1_train    = skm.f1_score(y_train_np, train_preds_np)

acc_test    = skm.accuracy_score(y_test_np, test_preds_np)
prec_test   = skm.precision_score(y_test_np, test_preds_np)
rec_test    = skm.recall_score(y_test_np, test_preds_np)
f1_test     = skm.f1_score(y_test_np, test_preds_np)

print("Train Metrics:")
print(f"  Accuracy : {acc_train:.3f}")
print(f"  Precision: {prec_train:.3f}")
print(f"  Recall   : {rec_train:.3f}")
print(f"  F1 Score : {f1_train:.3f}")

print("\nTest Metrics:")
print(f"  Accuracy : {acc_test:.3f}")
print(f"  Precision: {prec_test:.3f}")
print(f"  Recall   : {rec_test:.3f}")
print(f"  F1 Score : {f1_test:.3f}")
```
```

### 5.1 Typical Results Interpretation
- **Accuracy**: Often around 0.75–0.90 for training, lower on test set.  
- **Precision** vs. **Recall**: If the model **over-predicts** “good wine,” we’ll see more false positives → **precision** tends to be lower than recall.  
- **F1 Score**: Averages out these biases (often lies between precision and recall).

---

## 6. Visualization with Bar Plots

It’s useful to compare these metrics side-by-side:

```python
```python
import matplotlib.pyplot as plt

metrics_train = [acc_train, prec_train, rec_train, f1_train]
metrics_test  = [acc_test,  prec_test,  rec_test,  f1_test ]
labels = ['Accuracy','Precision','Recall','F1']

x = np.arange(len(labels))

plt.figure(figsize=(8,5))
plt.bar(x - 0.2, metrics_train,  width=0.4, label='Train')
plt.bar(x + 0.2, metrics_test,   width=0.4, label='Test')
plt.xticks(x, labels)
plt.ylim([0,1])
plt.ylabel('Metric Value')
plt.title('Train vs. Test Performance (Wine Quality)')
plt.legend()
plt.show()
```
```

Typical outcome:
- The test set metrics are usually **lower** than the training set (generalization gap).
- **Recall** is often higher than **Precision** if the model is biased to predict “high quality” wine.

---

## 7. Confusion Matrix

### 7.1 Using Scikit-Learn’s `confusion_matrix`
```python
```python
cm_train = skm.confusion_matrix(y_train_np, train_preds_np)
cm_test  = skm.confusion_matrix(y_test_np,  test_preds_np)

print("Train Confusion Matrix:\n", cm_train)
print("Test Confusion Matrix:\n", cm_test)
```
```
The confusion matrix layout by default is:
\[
\begin{bmatrix}
\text{TN} & \text{FP}\\
\text{FN} & \text{TP}
\end{bmatrix}
\]

### 7.2 Visualizing the Confusion Matrix
```python
```python
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(1, 2, figsize=(10,4))
ConfusionMatrixDisplay(cm_train, display_labels=['Not Good','Good']).plot(ax=ax[0], cmap=plt.cm.Blues)
ax[0].set_title('Train Confusion Matrix')

ConfusionMatrixDisplay(cm_test,  display_labels=['Not Good','Good']).plot(ax=ax[1], cmap=plt.cm.Blues)
ax[1].set_title('Test Confusion Matrix')

plt.tight_layout()
plt.show()
```
```

Common observations:
- **True Positive (TP)** block might show how many wines were correctly labeled as high quality.  
- **False Positive (FP)** often higher if the model is liberal in calling a wine “good.”  
- The difference between *FP* and *FN* typically reveals the model’s bias (too many positives vs. too many negatives).

---

## 8. Interpretation and Insights

1. **Accuracy** is high if the dataset isn’t extremely imbalanced.  
2. **Precision** < **Recall** suggests the model is *over-predicting* “good” wines.  
3. **F1 Score** provides a single metric that penalizes both false positives and false negatives.  
4. The **confusion matrix** confirms these findings in raw counts (e.g., more false positives than false negatives).

### 8.1 Practical Takeaways
- If **false positives** (overestimating wine quality) are worse (e.g., a buyer invests in “bad” wine), we want to **improve Precision**.  
- If **false negatives** (missing a “good” wine) are worse, we focus on **Recall**.  
- **Tuning** model complexity, learning rate, or using a different optimizer (e.g., Adam) can shift these metrics.

---

## 9. Summary

1. **ARPF in Practice**: For binary wine-quality classification, ARPF metrics reveal how well the model classifies “good” vs. “not good” wines and highlight bias.  
2. **Confusion Matrices**: The distribution of errors (FP vs. FN) indicates whether the model systematically favors “yes” or “no.”  
3. **scikit-learn** Integration: We often rely on built-in metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`) for convenience and standardization.  
4. **Model Bias**: Observed if `precision ≠ recall`—one type of misclassification is more frequent than the other.  

Next, we’ll see how to extend these ideas to **multi-class classification** (e.g., MNIST), where the confusion matrix becomes larger, and metrics like Precision and Recall need to be computed per class or via macro/micro averaging.

---

## 10. References and Further Reading

- **UCI ML Repository**: [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)  
- **Goodfellow, I., Bengio, Y., Courville, A.**: *Deep Learning* (Chapter on classification and metrics).  
- **scikit-learn metrics**: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)  
- **Signal Detection Theory**: *Green & Swets (1966)*, *Signal Detection Theory and Psychophysics*.  

---

```

**Usage in Obsidian**:

1. Create a new note in your Obsidian vault.  
2. Paste the entire code block (including frontmatter `---`) into the note.  
3. Adjust the file title if needed.  
4. Optionally add links to related notes, e.g., `[[Metrics Overview]]` or `[[Confusion Matrix]]`.  

These notes provide a thorough **theoretical** and **practical** reference for implementing and interpreting **ARPF metrics** using the **Wine Quality** dataset in PyTorch + scikit-learn.