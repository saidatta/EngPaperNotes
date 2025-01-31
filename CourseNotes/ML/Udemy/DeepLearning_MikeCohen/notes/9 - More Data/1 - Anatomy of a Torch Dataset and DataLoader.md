aliases: [Data, Torch Dataset, Torch Dataloader, PyTorch Data, DataLoader, TensorDataset]
tags: [Data Loading, PyTorch, TensorDataset, Dataloader, Beginner]
## Overview
In PyTorch, **datasets** and **dataloaders** are the core abstractions that let you handle and iterate over data in mini-batches for training neural networks. You have already seen these objects many times, but in this lecture we delve deeper into **how they store and organize** your data, including:
1. Converting NumPy arrays to **PyTorch Tensors**.
2. Creating **`TensorDataset`** objects (with features and optional labels).
3. Building **`DataLoader`** objects from those datasets.
4. Understanding mini-batch iteration, **shuffling**, and how to manually access batches.

---

## 1. Creating and Inspecting PyTorch Tensors

### 1.1 From NumPy Arrays
Let’s create a small random dataset in **NumPy**:

```python
import numpy as np
import torch

# Create a 100 x 20 NumPy array
data = np.random.randn(100, 20)  # 100 observations, 20 features
print("Type of data:", type(data))
print("data.shape:", data.shape)
print("data.dtype:", data.dtype)

# Convert to PyTorch tensor
dataT = torch.tensor(data)
print("\nType of dataT:", type(dataT))
print("dataT.size():", dataT.size())
print("dataT.dtype:", dataT.dtype)
```

**Notes**:
- `type(...)` tells us the **Python object class** (NumPy ndarray or torch.Tensor).
- `dtype` tells us the **data type** of the elements (float64, int64, etc.).
- `shape` in NumPy \(\leftrightarrow\) `size()` in PyTorch (though `dataT.shape` also works in PyTorch for convenience).

### 1.2 Converting and Changing Data Types
Sometimes we need **floating** data for features, and **integer** data for labels:

```python
data_float = torch.tensor(data, dtype=torch.float32)
data_long  = torch.tensor(data, dtype=torch.long)

print("data_float.dtype:", data_float.dtype)  # torch.float32
print("data_long.dtype:",  data_long.dtype)   # torch.int64
```

- **`.float()`** or **`.long()`** can be applied to a tensor to convert dtypes.

---

## 2. Building a `TensorDataset`

### 2.1 Single-Tensor Dataset
PyTorch’s **`TensorDataset`** can be created with one or more tensors. Let’s try **one** tensor first:

```python
from torch.utils.data import TensorDataset

# Attempt to directly use NumPy array (won't work)
try:
    dataset_fail = TensorDataset(data)
except Exception as e:
    print("Error message:\n", e)

# Correct approach (tensor form)
dataset_ok = TensorDataset(dataT)
print("\nDataset_ok:", dataset_ok)
print("dataset_ok.tensors:", dataset_ok.tensors)
```

- If you only pass **one** tensor, `dataset_ok.tensors` is a **1-element tuple**.  

### 2.2 Adding Labels
Usually, you have **(X, y)** pairs. Suppose we create **labels** for our 100 samples:

```python
# Fake labels (integers in [1..4])
labels = np.random.randint(1, 5, size=(100,))
print("labels.shape:", labels.shape)

# Reshape for clarity (100, 1)
labels = labels.reshape(-1, 1)
print("labels reshaped:", labels.shape)

# Convert to tensor
labelsT = torch.tensor(labels, dtype=torch.long)
```

Now we build a dataset with **2** tensors (features, labels):

```python
dataset = TensorDataset(dataT, labelsT)
print("Length of dataset:", len(dataset))  # should be 100

# Let's look at the first sample
print("First sample:", dataset[0])  
# -> (tensor of shape [20], tensor of shape [1])

# The underlying data is stored in dataset.tensors
print("\ndataset.tensors:", dataset.tensors)
print("dataset.tensors[0].size():", dataset.tensors[0].size())  # (100, 20)
print("dataset.tensors[1].size():", dataset.tensors[1].size())  # (100, 1)
```

**Key**: The first element of `dataset.tensors` is your **feature matrix** (100×20), the second is your **label matrix** (100×1).

---

## 3. Using the `DataLoader`

### 3.1 Creating a DataLoader
The **DataLoader** wraps a dataset and provides:
- **Mini-batches** of features/labels.
- Optional **shuffling** each epoch.
- Other settings (num_workers, pin_memory, etc. for performance).

```python
from torch.utils.data import DataLoader

batch_size = 25
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print("Type of data_loader:", type(data_loader))
print("data_loader.dataset:", data_loader.dataset)
print("Length of data_loader.dataset:", len(data_loader.dataset))
```

### 3.2 Inspecting the DataLoader
Unlike the `TensorDataset`, the DataLoader doesn’t directly store mini-batches. Instead, it *iterates* over the dataset, partitioning it into batches on-the-fly.

#### Looping Over Batches
```python
for i, (X_batch, y_batch) in enumerate(data_loader):
    print(f"Batch {i+1}:")
    print("  X_batch shape:", X_batch.size())  # [25, 20]
    print("  y_batch shape:", y_batch.size())  # [25, 1]
```

- **`X_batch.size(0)`** = 25 = `batch_size`.  
- The last batch might be smaller if `drop_last=False` and `len(dataset)` is not divisible by 25.

### 3.3 Shuffle
If we enable **shuffle**, each epoch (or each iteration over `data_loader`) reorders the dataset:

```python
# Recreate with shuffle=True
data_loader_shuffle = DataLoader(dataset, batch_size=25, shuffle=True)

for Xb, yb in data_loader_shuffle:
    # We'll see that the labels are now randomly ordered
    print(yb.squeeze().tolist())  # just show the label values
    break  # only show first batch
```

**Important**: The shuffle is applied **each time** you iterate over the `data_loader`. So if you iterate again, you get a new order.

---

## 4. Getting a Single Batch Manually
Sometimes you only want **one** batch (not the entire epoch). You can do:

```python
# next(iter(data_loader)) -> returns first batch
single_batch = next(iter(data_loader))
X_one, y_one = single_batch
print("X_one shape:", X_one.size())  
print("y_one shape:", y_one.size())
```

---

## 5. Key Insights

1. **`TensorDataset`**  
   - Holds one or more **matching** tensors (same length along dimension 0).  
   - Indexing `dataset[i]` → returns a tuple `(X_i, y_i, ...)`.
2. **`DataLoader`**  
   - Iterates over the `dataset` in **mini-batches**.  
   - Shuffling is done each time you loop over it.  
   - You can specify `batch_size`, `shuffle`, `drop_last`, etc.
3. **Data Types**  
   - **Features**: Usually floating-point (`.float()`).  
   - **Labels**: Usually integer (`.long()`) for classification.

---

## 6. Example Code Recap

Below is a compact summary:

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# 1) Create random data (NumPy)
num_samples, num_features = 100, 20
data_np = np.random.randn(num_samples, num_features)
labels_np = np.random.randint(0, 5, size=(num_samples, 1))

# 2) Convert to tensors
data_t   = torch.tensor(data_np, dtype=torch.float32)
labels_t = torch.tensor(labels_np, dtype=torch.long)

# 3) Build a dataset
dataset = TensorDataset(data_t, labels_t)
print("dataset.tensors:", dataset.tensors)

# 4) Build a dataloader
batch_size = 25
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5) Iterate over dataloader
for i, (Xb, yb) in enumerate(data_loader):
    print(f"Batch {i+1}")
    print("  Xb.shape:", Xb.shape, "yb.shape:", yb.shape)
    # Xb is [batch_size, num_features]
    # yb is [batch_size, 1]
```

---

## 7. Conclusions & Next Steps

- You can now see **how** PyTorch stores data internally:
  - **`TensorDataset.tensors`** for the underlying data.  
  - **`DataLoader`** simply iterates batches, optionally shuffling each epoch.
- **Real Datasets**: Next, you’ll learn how to handle:
  - **Unbalanced data** and sampling strategies.  
  - **Data augmentation** for images.  
  - **Loading external data** into Colab.  
  - **Saving/loading models** to avoid re-training from scratch.

**End of Notes – "Data: Anatomy of a Torch Dataset and DataLoader"** 
```