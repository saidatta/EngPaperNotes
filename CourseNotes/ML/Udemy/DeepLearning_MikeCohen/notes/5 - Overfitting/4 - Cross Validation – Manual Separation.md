## Table of Contents
1. [[Concept Overview]]
2. [[Manual Data Partition]]
3. [[Importance of Randomization]]
4. [[Implementing Cross-Validation in Python (Manual Method)]]
   - [[Data Setup: The Iris Dataset]]
   - [[Boolean Indexing for Train/Test Split]]
   - [[Balancing Classes]]
5. [[Training and Testing the Model]]
6. [[Observations and Edge Cases]]
7. [[Key Takeaways]]

---
## 1. Concept Overview
**Cross-validation** is a critical process for:
- **Training** a model on one subset of data (the “training” set).
- **Evaluating** it on another subset (the “dev” or “test” set) that the model **never** sees during training.

### Why Manual Partitioning?
- Sometimes you need direct, low-level control over exactly **which samples** go into training vs. test sets.
- This can be done with **NumPy** arrays or pandas, etc., rather than automated splits from libraries like Scikit-Learn.

> **Reminder**: If you **continuously refine** your model’s architecture or hyperparameters, you should maintain **three** splits: **train**, **dev**, and **test**. In this lecture’s example, we only need **train** and **test** because we are testing **one** specific model.

---

## 2. Manual Data Partition
### Standard Ratios
- Commonly: **80%** for training, **20%** for testing.
- These numbers aren’t sacred; 70/30, 75/25, 85/15, etc., are also used.
- The primary goal is to ensure the training set is **sufficiently large** to learn robust patterns.

> In a subsequent lecture, the instructor discusses the **trade-offs** in varying these ratios.

---

## 3. Importance of Randomization
- If data is **sorted** or **clustered** by label/class (e.g., all Setosa first, then Versicolor, then Virginica in the Iris dataset), a naive split will cause **imbalanced** (or even entirely missing) classes in the training or test sets.
- **Shuffling** or **random sampling** ensures each class is **fairly represented** in both splits.

---

## 4. Implementing Cross-Validation in Python (Manual Method)

### 4.1. Data Setup: The Iris Dataset
The **Iris dataset** consists of **150 samples** of iris flowers across **3 species**:
1. Iris Setosa
2. Iris Versicolor
3. Iris Virginica

Each sample has **4 features** (sepal length, sepal width, petal length, petal width) and a **label** (0, 1, or 2).

```python
```python
import numpy as np
import torch

# Assume `features` is a NumPy array of shape (150, 4)
# Assume `labels` is a NumPy array of shape (150,) with values in {0, 1, 2}
# (This loading/preprocessing code is omitted for brevity in the lecture.)
```

> **Note**: The lecturer pre-loads the data in a similar manner.

### 4.2. Boolean Indexing for Train/Test Split
1. Decide on **train_size** (e.g., 0.8 of total samples).  
2. Create a **Boolean array** (`train_test_bool`) of length `num_samples`, with `train_size * num_samples` set to `True` and the rest `False`.  
3. **Shuffle** or **permute** the Boolean array to ensure randomness.

```python
```python
num_samples = features.shape[0]  # e.g., 150
train_size = 0.8
num_train = int(num_samples * train_size)

# Create a boolean array initialized to False
train_test_bool = np.zeros(num_samples, dtype=bool)

# Set the first 'num_train' elements to True
train_test_bool[:num_train] = True

# Shuffle for randomness
np.random.shuffle(train_test_bool)

# Check distribution
print(train_test_bool)  # e.g., [True, False, True, True, ...]
```

- **True** = belongs to **training** set
- **False** = belongs to **test** set

Then we can do:
```python
train_features = features[train_test_bool]
train_labels   = labels[train_test_bool]

test_features  = features[~train_test_bool]
test_labels    = labels[~train_test_bool]
```

> **Tilde (`~`)** is the **logical NOT** operator, so it inverts `True` ↔ `False`.

### 4.3. Balancing Classes
- Ideally, the **average** label values (or class distributions) in the train and test sets should be **similar**.  
- With a **small** dataset (like Iris, only 150 samples), you might see slight imbalances.  
- A **larger** dataset typically yields better approximate balances with random sampling.

---

## 5. Training and Testing the Model
Once we have our train/test splits, we build and train a **simple neural network** in PyTorch. The code can look like this:

```python
```python
# Example: Simple 2-layer network for classification
model = torch.nn.Sequential(
    torch.nn.Linear(4, 16),   # 4 input features (Iris)
    torch.nn.ReLU(),
    torch.nn.Linear(16, 3)    # 3 classes output
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train_torch = torch.tensor(train_features, dtype=torch.float32)
y_train_torch = torch.tensor(train_labels, dtype=torch.long)

# Simple training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train_torch)
    loss   = loss_fn(y_pred, y_train_torch)
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # (Optional) track train accuracy each epoch
    pred_labels = torch.argmax(y_pred, axis=1)
    accuracy = (pred_labels == y_train_torch).float().mean().item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.4f}")

# Evaluate on train set
train_preds = model(X_train_torch).argmax(axis=1)
train_acc   = (train_preds == y_train_torch).float().mean().item()

# Evaluate on test set
X_test_torch = torch.tensor(test_features, dtype=torch.float32)
y_test_torch = torch.tensor(test_labels, dtype=torch.long)

test_preds = model(X_test_torch).argmax(axis=1)
test_acc   = (test_preds == y_test_torch).float().mean().item()

print(f"Final Train Accuracy: {train_acc*100:.2f}%")
print(f"Final Test Accuracy:  {test_acc*100:.2f}%")
```

**Note**:  
- We train only on the **True** subset (`train_test_bool`).
- We never run **backprop** on the **False** (test) subset.

---

## 6. Observations and Edge Cases
1. **Test Accuracy > Train Accuracy?**  
   - Unusual but possible, especially with **small datasets**.  
   - Random chance or easy test samples can lead to this situation.  
   - Not typically indicative of a major problem unless you see it in large datasets consistently.

2. **Small Sample Size**  
   - The Iris dataset has only **150 samples** total. A 30-sample test set might produce **perfect accuracy** occasionally.  
   - This isn’t necessarily an error but a reflection of the dataset’s small size and relative simplicity.

3. **Manual vs. Automated Splits**  
   - Libraries like Scikit-Learn’s `train_test_split` do the same partitioning automatically.  
   - Manual partitioning gives you **full control**, which can be crucial for certain designs, or domain constraints.

---

## 7. Key Takeaways
1. **Manual Cross-Validation**:  
   - Offers **fine-grained control** over which samples go into each split.  
   - Easy to implement with **boolean masking** in NumPy or pandas.

2. **Random Shuffling**:  
   - Crucial to avoid inadvertently placing **all** samples of a certain class into train **or** test.  
   - The dataset might be sorted by label or some other variable.

3. **Train vs. Test Accuracy**:  
   - Expect train accuracy to be >= test accuracy in most cases, but anomalies can occur.  
   - Always investigate large discrepancies.

4. **Future Videos**:  
   - Implementation in **Scikit-Learn** (built-in utilities).  
   - Implementation via **PyTorch DataLoader**.

---

**End of Notes**.  
By mastering manual cross-validation, you gain a solid foundation for more advanced or automated methods while retaining the flexibility to handle edge cases and special constraints in your projects.