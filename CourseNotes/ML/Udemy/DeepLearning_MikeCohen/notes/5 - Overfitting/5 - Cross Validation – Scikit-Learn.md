## Table of Contents
1. [[Motivation for Train/Test Splits]]
2. [[Choosing the Right Split Proportions]]
3. [[Using scikit-learn’s train_test_split]]
4. [[Implementing a Deep Learning Model with scikit-learn Split]]
5. [[Experiment: Varying Train/Test Proportions]]
6. [[Discussion of Overfitting & Researcher Overfitting]]
7. [[Key Takeaways]]

---

## 1. Motivation for Train/Test Splits
**Cross-validation** is a procedure to:
- Train a model on one subset of data (**train** set).
- Evaluate performance on a different subset (**test** set) the model **hasn’t seen**.

### When to Also Have a Dev Set
- If you plan to **continuously refine** or **tune** hyperparameters/architecture, you should typically have **train**, **dev**, and **test** sets to avoid “researcher overfitting” on the test set.

---

## 2. Choosing the Right Split Proportions
The proportion of data that goes into training vs. test (and dev) can vary. Common ratios include:
- **80/20** (train/test)
- **70/30** (train/test)
- **90/10** (train/test)
- **98/1/1** (train/dev/test) for **massive** datasets (e.g., ImageNet with 14M images).

> **Key Insight**: The chosen fraction depends on:
1. **Total amount of data** (e.g., a larger dataset can afford very small test percentage).  
2. **Model complexity** (complex models typically need more training data).  
3. **Variability** in the data (ensuring the test split is still representative).

---

## 3. Using scikit-learn’s train_test_split
scikit-learn has a **helper function** to handle splitting data randomly:

```python
```python
import numpy as np
from sklearn.model_selection import train_test_split

# Example dataset: 10 samples, 4 features
data = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    ...
])
labels = np.array([0, 0, 0, 0, 1, 1, ...])  # class labels

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    data, 
    labels,
    test_size=0.2,   # or train_size=0.8
    shuffle=True     # default is True
)
```

### Important Parameters
- **test_size** (float): The proportion of the dataset to allocate as test.  
- **train_size** (float): Alternative to test_size; either can be used.  
- **shuffle** (bool): Randomizes the rows to avoid ordered splits.  
- **random_state** (int): Sets a seed for reproducible splits.

---

## 4. Implementing a Deep Learning Model with scikit-learn Split

### 4.1. Loading the Iris Data
```python
```python
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
features = iris.data      # shape: (150, 4)
labels = iris.target      # shape: (150,)

# e.g., 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    labels,
    test_size=0.2,
    shuffle=True,
    random_state=42
)
```

### 4.2. Building and Training a Simple Model
```python
```python
model = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 3)  # 3 classes in Iris
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
X_test_torch  = torch.tensor(X_test,  dtype=torch.float32)
y_test_torch  = torch.tensor(y_test,  dtype=torch.long)

num_epochs = 100
train_accs = []
test_accs  = []

for epoch in range(num_epochs):
    # Forward
    y_pred = model(X_train_torch)
    loss   = loss_fn(y_pred, y_train_torch)
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Evaluate train accuracy
    train_preds = y_pred.argmax(dim=1)
    train_acc   = (train_preds == y_train_torch).float().mean().item()
    train_accs.append(train_acc)
    
    # Evaluate test accuracy (no backprop here!)
    with torch.no_grad():
        test_preds = model(X_test_torch).argmax(dim=1)
        test_acc   = (test_preds == y_test_torch).float().mean().item()
    test_accs.append(test_acc)

print(f"Final Train Accuracy: {train_accs[-1]*100:.2f}%")
print(f"Final Test Accuracy:  {test_accs[-1]*100:.2f}%")
```

> **Note**: Calculating **test accuracy** within the training loop is acceptable **as long as** you are **not** using it to update the model parameters (i.e., no gradient steps from the test loss).

---

## 5. Experiment: Varying Train/Test Proportions
**Goal**: See how Iris model accuracy changes when train data is 20%, 30%, …, 95%.

### 5.1. Experimental Setup
```python
```python
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
features = iris.data
labels = iris.target

train_sizes = np.linspace(0.2, 0.95, 10)  # e.g., [0.2, 0.288..., ..., 0.95]

def train_model(X_tr, y_tr, X_te, y_te, num_epochs=100):
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 3)
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    train_accs = []
    test_accs  = []
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.long)
    
    for epoch in range(num_epochs):
        # Forward
        y_pred = model(X_tr_t)
        loss = loss_fn(y_pred, y_tr_t)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Train accuracy
        pred_train = y_pred.argmax(dim=1)
        train_acc = (pred_train == y_tr_t).float().mean().item()
        
        # Test accuracy (no backprop)
        with torch.no_grad():
            y_pred_test = model(X_te_t)
            pred_test = y_pred_test.argmax(dim=1)
            test_acc = (pred_test == y_te_t).float().mean().item()
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    return train_accs, test_accs

all_train_accuracies = []
all_test_accuracies  = []

for ts in train_sizes:
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        train_size=ts,
        shuffle=True,
        random_state=42
    )
    
    train_accs, test_accs = train_model(X_train, y_train, X_test, y_test)
    all_train_accuracies.append(train_accs)
    all_test_accuracies.append(test_accs)

# Plot the results (heatmaps or line plots)
```

### 5.2. Possible Observations
- With **small splits** (e.g., 20% train), the model might struggle due to **insufficient training data**.  
- With **large splits** (e.g., 95% train), the test set is very small, so test accuracy will vary a lot and might even be unusually high or low due to **small sample size**.

> **In the lecture**: The Iris dataset is too small to confidently conclude strong trends, but the process illustrates **how** to systematically run such experiments.

---

## 6. Discussion of Overfitting & Researcher Overfitting
1. **Normal Overfitting**: Training accuracy is higher than test accuracy.  
   - *Expected* in nearly all neural network training scenarios.  
   - The gap ideally should be **small** if the model generalizes well.

2. **Researcher Overfitting**:
   - If you keep **changing** train/test proportions and **keep** re-training the model to see which split yields the best performance, you’re essentially “overfitting” to your dataset’s splitting scheme.  
   - Ideally, have a **fixed** train/test split and a separate **dev** set if you’re iterating on architecture or hyperparameters.

---

## 7. Key Takeaways
1. **Train/Test Splits**: There’s no single “correct” split ratio. It depends heavily on the **size** of your dataset and **how** you plan to use the model.  
2. **scikit-learn** `train_test_split`:
   - Provides a **convenient** way to do random partitions.  
   - Has parameters for **controlling** shuffle, test size, random seed, etc.
3. **Small Datasets**: The **Iris** dataset’s small size makes it **hard** to see clear trends when varying train/test splits. Random variations often dominate results.  
4. **Avoid Overfitting to Test Data**: Always keep your test set **completely** untouched for final performance checks; consider a separate dev set for active tuning.  

---

**End of Notes**.  
By leveraging scikit-learn’s `train_test_split` and understanding the dynamics of train vs. test proportions, you’ll be equipped to design experiments and achieve robust model evaluations in deep learning pipelines.