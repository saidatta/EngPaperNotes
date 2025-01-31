## Table of Contents
1. [[Why Three Splits?]]
2. [[Theory Recap: Train, Dev, Test]]
3. [[Method 1: scikit-learn (train_test_split Twice)]]
4. [[Method 2: Manual NumPy Partitioning]]
5. [[Additional Notes & Tips]]

---

## 1. Why Three Splits?
**Short Answer**:  
- **Train Set**: Used to **learn model parameters** (weights).  
- **Dev (Validation) Set**: Used to **tune** hyperparameters and make architectural decisions **without** contaminating the final test.  
- **Test Set**: Used **once** at the end to **estimate real-world performance**.

**Long Answer**:  
- If you only have train/test, you can easily end up **overfitting** your architecture or hyperparameters to the test data by repeatedly evaluating on it.  
- The **dev set** helps to prevent this form of “researcher overfitting” by absorbing the repeated model refinements.

---

## 2. Theory Recap: Train, Dev, Test

### Overfitting on the Dev Set
- Each time you evaluate on the **dev set** and adjust your model in response, you implicitly overfit to dev data.  
- That’s **why** you keep a final **test set** which remains **untouched** until you’re ready to confirm final performance.

### Typical Splits
- **80/10/10**, **70/15/15**, or other variations depending on:  
  - **Size** of dataset  
  - **Complexity** of model  
  - **Goals** of the project (e.g., more dev data if you need extensive hyperparameter tuning)

---

## 3. Method 1: scikit-learn (train_test_split Twice)
### 3.1. Example Setup
```python
```python
import numpy as np
from sklearn.model_selection import train_test_split

# Toy dataset: 10 samples, 4 features
X = np.array([
    [10,20,30,40],
    [50,60,70,80],
    [90,100,110,120],
    [130,140,150,160],
    [170,180,190,200],
    [210,220,230,240],
    [250,260,270,280],
    [290,300,310,320],
    [330,340,350,360],
    [370,380,390,400]
])
y = np.array([0,0,0,0,1,1,1,1,2,2])  # 3 classes, just an example

# We want 80% train, 10% dev, 10% test
train_prop = 0.8
dev_prop   = 0.1
test_prop  = 0.1
```

### 3.2. First Split (Train vs. “Temp”)
```python
# Step 1: Split into train set & "temp" set
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    train_size=train_prop,
    shuffle=True,
    random_state=42
)

print("Train set shape:", X_train.shape)
print("Temp set shape:", X_temp.shape)
```
- After this, **80%** of the data is in `X_train, y_train`.
- The remaining **20%** is in `X_temp, y_temp`.

### 3.3. Second Split (Dev vs. Test)
```python
# Step 2: Split temp set into dev and test
# We have dev_prop=0.1 and test_prop=0.1 out of the *original* dataset
# But we only have 0.2 of the data in X_temp
# So dev fraction is dev_prop / (dev_prop + test_prop) = 0.1/0.2 = 0.5
split_frac = dev_prop / (dev_prop + test_prop)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp,
    train_size=split_frac,
    shuffle=True,
    random_state=42
)

print("Dev set shape:", X_dev.shape)
print("Test set shape:", X_test.shape)
```

**Important**:  
- The second `train_test_split` call partitions the **temp** data 50/50 if dev_prop = test_prop.  
- If dev_prop and test_prop differ, adjust accordingly (e.g., `split_frac = dev_prop / (dev_prop + test_prop)`).

---

## 4. Method 2: Manual NumPy Partitioning
Sometimes you might not want to rely on scikit-learn for splitting, or you want full custom control.

### 4.1. Shuffle and Create Boundaries
```python
```python
# Toy dataset from above

# Proportions
train_prop = 0.8
dev_prop   = 0.1
test_prop  = 0.1

# Convert proportions to absolute indices
N = len(y)  # total samples
train_end = int(np.floor(train_prop * N))
dev_end   = train_end + int(np.floor(dev_prop * N))
# test_end  = dev_end + int(np.floor(test_prop * N)) # Should be = N

# Create random permutation of indices
indices = np.random.permutation(N)

# Assign based on the index boundaries
train_indices = indices[:train_end]
dev_indices   = indices[train_end:dev_end]
test_indices  = indices[dev_end:]

X_train_np = X[train_indices]
y_train_np = y[train_indices]
X_dev_np   = X[dev_indices]
y_dev_np   = y[dev_indices]
X_test_np  = X[test_indices]
y_test_np  = y[test_indices]

print("Train set shape:", X_train_np.shape)
print("Dev set shape:",   X_dev_np.shape)
print("Test set shape:",  X_test_np.shape)
```

### 4.2. Why Might You Do This?
- Full control over **randomization** process.  
- Potentially simpler if you’re not using scikit-learn.  
- Works seamlessly with **NumPy-only** pipelines.

---

## 5. Additional Notes & Tips
1. **Model Refinements**:
   - If you plan to iterate on model design, always keep a dedicated **dev set**.  
   - The **test set** is only for **final** performance evaluation.
2. **DataLoader** in PyTorch:
   - Typically, you’d do these splits first (train, dev, test).  
   - Then convert each subset into a **`TensorDataset`** and feed into a **`DataLoader`**.
3. **Edge Cases**:
   - Very small datasets (like Iris or 10-sample toy data) might render a 10% dev set **too small** to be representative. In real projects, you generally have **much larger** datasets.
4. **Fixed Seeds**:
   - If you need reproducibility, set `np.random.seed(…)` or `random_state` in `train_test_split`.
5. **Ratios vs. Exact Numbers**:
   - Sometimes it’s more convenient to specify **fixed numbers** of samples in each set rather than proportions (e.g., 100K dev images in ImageNet). The approach is the same.

---

## Key Takeaways
1. **3-Split Reasoning**: Train for parameters, Dev for hyperparameter tuning, Test for final performance.  
2. **Implementation**:
   - **scikit-learn**: Easiest approach; call `train_test_split` **twice**.  
   - **NumPy**: Manually shuffle and slice indices for more customization.  
3. **Small Data**: Splitting into multiple subsets can lead to very small dev/test sets; weigh the trade-offs if your dataset is tiny.  
4. **Best Practices**:
   - If you do **not** plan to do repeated hyperparameter tuning, you may only need train/test.  
   - Otherwise, always keep a separate dev set to **protect** your test set from repeated exposure.

---

**End of Notes**.  
By employing these partitioning strategies, you ensure robust model development with minimal risk of **overfitting** hyperparameters or **inadvertently** leaking test information into your training process.