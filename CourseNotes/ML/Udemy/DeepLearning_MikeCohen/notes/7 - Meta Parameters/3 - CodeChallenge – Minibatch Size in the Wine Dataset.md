
aliases: [Mini-Batch Size, Wine Dataset, Parametric Experiment]
tags: [Deep Learning, Lecture Notes, Hyperparameters, Neural Networks, PyTorch]

This code challenge focuses on **two main tasks**:

1. **Build and train** a **binary classification** model to predict **wine quality** (high vs. low) using the **Wine Quality** dataset.
2. **Run a parametric experiment** on **minibatch size** (e.g., 2, 8, 32, 128, 512) to see how it affects:
   - **Training accuracy** vs. **Test accuracy**  
   - **Computation time** for training

> **Hint**: If you are stuck, feel free to **start with the code** below or revisit the **previous code challenge** on mini-batch sizes for reference. You can also compare your results to the solution walkthrough.

---

## 1. Overview

### Dataset Recap

- **Source**: [UCI Wine Quality dataset (Red Wine)](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- **Features**: 11 chemical indicators (e.g., pH, acidity, sulfur dioxide, etc.).  
- **Label**: Originally integer in [3..8]. We **binarized** it:
  - `0` = Quality <= 5  
  - `1` = Quality >= 6  

### Objective

- **Part 1**: Construct a neural network that:
  - Takes **11 inputs** (one for each normalized feature).  
  - Outputs **1 unit** (for binary classification).  
  - Uses a suitable **loss function** (e.g., `BCELoss` or `BCEWithLogitsLoss` or `CrossEntropyLoss` with a single output).
  - Achieves reasonable training & test accuracy.

- **Part 2**: Investigate **various mini-batch sizes** and observe:
  - **Overfitting** behaviors (Train vs. Test accuracy).  
  - **Computation time** for each choice.

---

## 2. Imports & Setup

We use the same standard libraries as usual, plus Python’s built-in `time` to measure elapsed training times.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time  # For measuring computation time

sns.set(style="whitegrid")  # Aesthetics for plots
```

---

## 3. Data Loading and Preprocessing

### 3.1 Load Wine Quality Data from UCI

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

data = pd.read_csv(url, sep=";")
print("Initial data shape:", data.shape)
data.head()
```

### 3.2 Clean, Outlier Removal, & Binarize Labels

- **Remove outliers** in `total sulfur dioxide` (e.g., values > 200).  
- **Z-score normalize** features.  
- **Create** column `boolQuality` = 0/1 for binary classification.

```python
# 1) Remove outliers
data = data.loc[data["total sulfur dioxide"] < 200]

# 2) Z-score the features (except "quality")
features = [c for c in data.columns if c != "quality"]
for col in features:
    mu = data[col].mean()
    sd = data[col].std()
    data[col] = (data[col] - mu) / sd

# 3) Binarize the "quality" column (new label "boolQuality")
data["boolQuality"] = 0
data.loc[data["quality"] > 5, "boolQuality"] = 1
data.dropna(inplace=True)  # Just in case

print("Final data shape:", data.shape)
data.head()
```

---

## 4. Convert Data to PyTorch Tensors

We’ll drop the original `quality` column and use `boolQuality` as our target. Ensure:
- **Feature matrix** `X` has shape: (N, 11)  
- **Label vector** `y` has shape: (N, 1) for binary classification

```python
feature_cols = [c for c in data.columns if c not in ["quality", "boolQuality"]]
X_np = data[feature_cols].values
y_np = data["boolQuality"].values

X_t = torch.tensor(X_np, dtype=torch.float32)
y_t = torch.tensor(y_np, dtype=torch.long).view(-1, 1)

print("Features:", X_t.shape)
print("Labels:  ", y_t.shape)
```

### 4.1 Train / Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_t, y_t, test_size=0.2, shuffle=True, random_state=42
)

print("Train set:", X_train.shape, y_train.shape)
print("Test set: ", X_test.shape, y_test.shape)
```

### 4.2 Create TensorDatasets

```python
train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test,  y_test)
```

We won’t create `DataLoader` objects here yet, because we will **vary the batch size** in our experiment.

---

## 5. Defining the Model

We have 11 input features and 1 output (for binary classification).  
An example architecture (feel free to change!):

```python
class WineNet(nn.Module):
    def __init__(self):
        super(WineNet, self).__init__()
        self.fc1 = nn.Linear(11, 16)   # input 11 -> hidden 16
        self.fc2 = nn.Linear(16, 32)   # hidden 16 -> hidden 32
        self.fc3 = nn.Linear(32, 1)    # hidden 32 -> output 1
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # final output is 1 unit (logits)
        return x
```

**Note**: We’ll use a **single output** with a sigmoid or a logistic function for binary classification. If we use **`BCEWithLogitsLoss`**, we do **not** need to apply `Sigmoid()` in the forward pass (the loss function internally applies it).

---

## 6. Training Function

This function trains a given model for some number of epochs on a particular DataLoader, then reports final train/test accuracies.

```python
def train_model(model, train_loader, test_loader, epochs=100, lr=0.001):
    # Use BCEWithLogitsLoss if output is 1 logit
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_acc_history = []
    test_acc_history  = []
    
    for ep in range(epochs):
        # Switch to training mode
        model.train()
        
        correct = 0
        total   = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # forward pass
            logits = model(X_batch)
            # Convert logits -> probabilities for accuracy
            # but for the loss, we can use logits directly
            loss = criterion(logits, y_batch.float())
            
            # backward
            loss.backward()
            optimizer.step()
            
            # compute train accuracy
            preds = torch.sigmoid(logits)
            predicted_labels = (preds > 0.5).long()
            
            correct += (predicted_labels == y_batch).sum().item()
            total   += y_batch.size(0)
        
        train_acc = correct / total
        train_acc_history.append(train_acc)
        
        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            correct_test = 0
            total_test   = 0
            for Xb, yb in test_loader:
                test_logits = model(Xb)
                test_preds  = torch.sigmoid(test_logits)
                test_predicted_labels = (test_preds > 0.5).long()
                
                correct_test += (test_predicted_labels == yb).sum().item()
                total_test   += yb.size(0)
                
        test_acc = correct_test / total_test
        test_acc_history.append(test_acc)
        
        # Optional: print every few epochs
        # if (ep+1) % 20 == 0:
        #    print(f"Epoch {ep+1}/{epochs} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")
        
    return {
        "train_acc": train_acc_history,
        "test_acc": test_acc_history
    }
```

---

## 7. Running the Parametric Experiment

We will try **several mini-batch sizes** and track:

1. **Train accuracy** vs. **Epoch**  
2. **Test accuracy** vs. **Epoch**  
3. **Computation time** (seconds) for training.

### 7.1 Setup the Batch Sizes

```python
batch_sizes = [2, 8, 32, 128, 512]
epochs = 100

results = {}
times   = {}
```

### 7.2 Loop Over Batch Sizes

```python
for bs in batch_sizes:
    # Start time
    start_time = time.process_time()
    
    # Create new DataLoader objects with this batch size
    train_loader = DataLoader(train_dataset, batch_size=bs, 
                              shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=bs, 
                              shuffle=False, drop_last=True)
    
    # Instantiate a fresh model
    model = WineNet()
    
    # Train the model
    outcome = train_model(model, train_loader, test_loader, epochs=epochs, lr=0.001)
    
    # Stop time
    end_time = time.process_time()
    elapsed_seconds = end_time - start_time
    
    results[bs] = outcome
    times[bs]   = elapsed_seconds

print("Done with all experiments!")
```

> **Note**: `time.process_time()` measures CPU time in many environments. Some prefer `time.perf_counter()` for wall-clock time. Either is acceptable for relative comparisons.

---

## 8. Visualizing the Results

### 8.1 Accuracy Curves per Batch Size

We can plot the train and test accuracy curves for each batch size.

```python
plt.figure(figsize=(12,6))

for bs in batch_sizes:
    train_acc = results[bs]["train_acc"]
    test_acc  = results[bs]["test_acc"]
    
    plt.plot(train_acc, label=f"Train BS={bs}")
    plt.plot(test_acc,  label=f"Test BS={bs}", linestyle="--")

plt.title("Train (solid) vs. Test (dashed) Accuracy for Different Batch Sizes")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

- **Trend**: Smaller batch sizes often lead to faster updates (more frequent gradient steps) and can overfit more easily, especially if the dataset is not huge.
- **Overfitting**: Observe if training accuracy diverges far above test accuracy.

### 8.2 Final Accuracies

Compare final epoch accuracies across different batch sizes:

```python
final_train_acc = []
final_test_acc  = []
for bs in batch_sizes:
    final_train_acc.append(results[bs]["train_acc"][-1])
    final_test_acc.append(results[bs]["test_acc"][-1])

plt.figure(figsize=(8,4))
X_axis = np.arange(len(batch_sizes))
width = 0.35

plt.bar(X_axis - width/2, final_train_acc, width=width, label='Train')
plt.bar(X_axis + width/2, final_test_acc,  width=width, label='Test')
plt.xticks(X_axis, batch_sizes)
plt.xlabel("Batch Size")
plt.ylabel("Accuracy")
plt.title("Final Training vs. Testing Accuracy by Batch Size")
plt.legend()
plt.show()
```

### 8.3 Computation Time

We also tracked how long each training session took:

```python
batch_size_list = []
time_list       = []

for bs in batch_sizes:
    batch_size_list.append(bs)
    time_list.append(times[bs])

plt.figure(figsize=(6,4))
plt.bar(batch_size_list, time_list)
plt.xlabel("Batch Size")
plt.ylabel("Time (seconds)")
plt.title("Computation Time vs. Mini-Batch Size")
plt.show()

for bs in batch_sizes:
    print(f"Batch Size={bs}, Time={times[bs]:.2f} seconds")
```

Typically:
- **Very small batch sizes** → More frequent gradient updates, can cause longer overall training time.  
- **Larger batch sizes** → Less frequent updates, might train faster but can reduce generalization slightly.

---

## 9. Observations and Discussion

1. **Overfitting Trend**:  
   - Smaller batches (2, 8) might lead to very high training accuracy but a larger gap to the test accuracy → Overfitting.  
   - Larger batches (128, 512) might generalize better but can require more epochs to achieve comparable accuracy.

2. **Time Complexity**:  
   - Extremely small batch sizes often increase total compute time.  
   - Extremely large batch sizes can reduce training iterations but may converge slower or get stuck.

3. **Wine Quality Ceiling**:  
   - The dataset is somewhat noisy and subjective (human taste). Achieving >80% to 85% test accuracy can be challenging. Even 75%–80% can be decent, depending on architecture and hyperparameters.

4. **Future Experiments**:
   - Try deeper or wider networks.  
   - Vary learning rates or optimizers.  
   - Implement regularization methods (dropout, weight decay).  
   - Increase number of epochs (e.g., 500–2000) to see if test accuracy plateaus.

---

## 10. Key Takeaways

1. **Mini-Batch Size is a Meta Parameter**:  
   - Changing it has large effects on accuracy curves and training dynamics.  
   - There’s no single best “universal” batch size; it depends on your data, model, hardware.

2. **Wine Quality is Subjective**:  
   - **Perfect** classification is unlikely.  
   - Expect non-trivial overfitting if the model memorizes training samples.

3. **Time vs. Accuracy Trade-Off**:  
   - Smaller batch sizes → Potentially more fine-grained gradient steps, but longer runtime.  
   - Larger batch sizes → Fewer updates per epoch, possibly faster wall-clock time, sometimes decreased generalization.

This exercise exemplifies a fundamental balancing act in deep learning: **tuning hyperparameters (like batch size) while avoiding overfitting** and managing computational resources.

---

## 11. Additional Resources

- [UCI Wine Quality Dataset Description](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html)  
- [Deep Learning Book (Goodfellow et al.) – Hyperparameters](https://www.deeplearningbook.org/)  
- [On Large-Batch Training Generalization Issues (Keskar et al.)](https://arxiv.org/abs/1609.04836)  

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: “Meta parameters: CodeChallenge: Minibatch size in the wine dataset”  
```