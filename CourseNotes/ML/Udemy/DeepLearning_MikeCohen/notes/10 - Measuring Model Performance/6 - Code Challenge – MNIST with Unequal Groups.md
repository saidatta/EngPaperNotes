title: "Measuring Model Performance: Code Challenge – MNIST with Unequal Groups"
tags: [deep-learning, mnist, unbalanced-datasets, performance-metrics]
## 1. Overview

This **code challenge** explores how **unbalanced (unequal) class groups** in the MNIST dataset influence **Accuracy, Precision, Recall, and F1**. Specifically, we **reduce** the number of **7s** from ~2000 to **500** while leaving the other digit classes unchanged. We then observe how this imbalance affects the model’s performance—particularly **precision** and **recall** for the underrepresented digit.

### Key Objectives
1. **Create an unbalanced MNIST** by randomly removing the majority of “7” examples.  
2. **Train** a simple neural network model on this modified dataset.  
3. **Measure** and **compare** ARPF metrics across digits.  
4. **Inspect** the **confusion matrix** to see how the bias toward or against “7” manifests.

---

## 2. Hypothesis Formulation

**Before** coding, it’s crucial to form a **falsifiable hypothesis**:

> *“By drastically reducing the number of 7s, the model may learn to misclassify 7s more often. We might predict a **decrease** in **recall** (model says ‘not 7’ when it *is* 7) or a **decrease** in **precision** (model incorrectly predicting 7 for non-7 digits).”*

You can test whether the model:
- Becomes **biased** towards saying “7” less often (leading to **lower recall** for 7).  
- Or, in some runs, tries to overcompensate (potentially lowering **precision** for 7).  

---

## 3. Data Preparation: Removing 7s

### 3.1. Standard MNIST Setup

```python
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Basic MNIST transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    # Optionally normalize if you like, e.g. (0.1307,), (0.3081,)
])
    
# Load full MNIST
train_data_full = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
test_data       = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

print("Full Train Set Size:", len(train_data_full))
print("Test Set Size:      ", len(test_data))
```
```

### 3.2. Creating the Unbalanced Dataset

1. **Extract** all samples from the original training set.  
2. **Find all indices** where the label is “7.”  
3. **Randomly keep 500** of those “7” samples, **remove** the rest.  

```python
```python
# Convert train_data_full into NumPy arrays for easy slicing
# train_data_full.data:   (60000, 28, 28)
# train_data_full.targets: (60000,)

# We'll reduce the size for demonstration (e.g. subset to first 20k if you like).
# In the lecture, we see an example using ~20k samples. Adjust as you prefer.

N_SUBSET = 20000  # example
images_full = train_data_full.data[:N_SUBSET].numpy()
labels_full = train_data_full.targets[:N_SUBSET].numpy()

print("Initial subset size:", len(images_full))

# Count the label distribution (sanity check)
unique_labels, counts_labels = np.unique(labels_full, return_counts=True)
print("Initial label counts:", dict(zip(unique_labels, counts_labels)))

# Identify all '7' indices
sev_indices = np.where(labels_full == 7)[0]
print(f"Number of 7s initially: {len(sev_indices)}")

# We only keep 500
N_KEPT = 500
np.random.shuffle(sev_indices)
remove_indices = sev_indices[N_KEPT:]  # the ones to remove

# Mask out these indices from the dataset
keep_mask = np.ones(len(labels_full), dtype=bool)
keep_mask[remove_indices] = False

images_bal = images_full[keep_mask]
labels_bal = labels_full[keep_mask]
print("Final size after removing 7s:", len(images_bal))

# Confirm new distribution
unique_labels_bal, counts_labels_bal = np.unique(labels_bal, return_counts=True)
print("Label counts after removal:", dict(zip(unique_labels_bal, counts_labels_bal)))
```
```

**Result**: The digit “7” now has only **500** samples, while other digits maintain ~2,000 (assuming the initial subset was ~2,000 per digit in a balanced slice).

---

## 4. Visualization: Class Distribution Histogram

(Optional) plot a histogram to confirm the imbalance:

```python
```python
plt.bar(unique_labels_bal, counts_labels_bal, color='maroon')
plt.xlabel('Digit')
plt.ylabel('Number of samples')
plt.title('Digit Distribution After Removing Most 7s')
plt.show()
```
```
You should see something like a bar chart where digit `7` is significantly smaller than the rest.

---

## 5. Model Definition and Training

We build a **simple** feed-forward or CNN network; the architecture is up to you. Below, a minimal FC network:

```python
```python
class MNISTNet(nn.Module):
    def __init__(self, in_size=784, hidden=128, out_size=10):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden)
        self.fc2 = nn.Linear(hidden, out_size)
    
    def forward(self, x):
        x = x.view(-1, in_size)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
```
```

### 5.1. Creating a Custom Dataset and DataLoader

```python
```python
import torch.utils.data as data_utils

X_tensor = torch.from_numpy(images_bal).float()
y_tensor = torch.from_numpy(labels_bal).long()

# For grayscale images in [0,255], optionally scale them to [0,1]
X_tensor /= 255.0  # normalize

# Create a dataset and split
dataset_bal = data_utils.TensorDataset(X_tensor, y_tensor)

# For demonstration, let's do a quick train/test split from the balanced subset:
split_ratio = 0.9
N_total = len(dataset_bal)
N_train = int(split_ratio * N_total)
N_val   = N_total - N_train

train_ds, val_ds = data_utils.random_split(dataset_bal, [N_train, N_val])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
```
```

**Note**: This is separate from the original `test_data` MNIST set. You could **also** test on the original official test set if desired. The main demonstration is the effect on the *training distribution*.

### 5.2. Training Loop

```python
```python
epochs = 5
for epoch in range(epochs):
    model.train()
    correct = 0
    total   = 0
    
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(Xb)
        loss  = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        
        # Track accuracy in training loop
        pred_labels = preds.argmax(dim=1)
        correct += (pred_labels == yb).sum().item()
        total   += len(yb)
    
    acc = correct / total
    print(f"Epoch {epoch+1}/{epochs}, Train accuracy={acc:.3f}")
```
```

The unbalanced distribution **may** influence how quickly or how effectively the model learns to recognize “7.”

---

## 6. ARPF Metrics on the Unbalanced Dataset

### 6.1. Generate Predictions Over the Validation Set

```python
```python
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for Xv, yv in val_loader:
        out = model(Xv)
        preds_v = out.argmax(dim=1)
        all_preds.append(preds_v)
        all_labels.append(yv)

all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

val_acc = skm.accuracy_score(all_labels, all_preds)
print(f"\nValidation Accuracy: {val_acc:.3f}")
```
```

### 6.2. Compute Precision, Recall, and F1 (Per-Class)

```python
```python
prec_each_class = skm.precision_score(all_labels, all_preds, average=None)
rec_each_class  = skm.recall_score(all_labels,  all_preds, average=None)
f1_each_class   = skm.f1_score(all_labels,      all_preds, average=None)

print("Precision per digit:", prec_each_class)
print("Recall per digit:   ", rec_each_class)
print("F1 per digit:       ", f1_each_class)
```
```

- Expect **digit 7** to show **lower recall** if the model is biased to label *fewer* sevens (it sees fewer examples of them).  
- Alternatively, in some runs, the model may show **lower precision** if it erroneously labels non-7 digits as “7.”

### 6.3. Class-Averaged Metrics

```python
```python
prec_weighted = skm.precision_score(all_labels, all_preds, average='weighted')
rec_weighted  = skm.recall_score(all_labels,    all_preds, average='weighted')
f1_weighted   = skm.f1_score(all_labels,        all_preds, average='weighted')

print(f"Weighted Avg Precision: {prec_weighted:.3f}")
print(f"Weighted Avg Recall   : {rec_weighted:.3f}")
print(f"Weighted Avg F1       : {f1_weighted:.3f}")
```
```

Often, the **overall** weighted metrics remain relatively high (e.g., ~0.90+ in MNIST), **masking** the poorer performance on the underrepresented digit.

---

## 7. Confusion Matrix

Visualizing misclassifications reveals how the model confuses *7*s with other digits (e.g., *2* or *1*):

```python
```python
cm = skm.confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8,6))
disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
plt.title("Validation Confusion Matrix (Unbalanced 7s)")
plt.show()
```
```

- Look at row=7 (true digit=7).  
- Identify which columns have higher off-diagonal counts (e.g., 2 or 9).  

---

## 8. Key Observations

1. **Reduced 7 Examples** → The model often learns to **ignore** 7, leading to **low recall** for digit 7 (biased toward saying “not 7”).  
2. **Instability**: Running multiple times may yield **different** biases. Sometimes the model might have **low precision** on 7, labeling many other digits as 7.  
3. **Overall Accuracy** often remains **high** since only a small fraction (7s) is impacted, and **MNIST** is relatively easy.  
4. **Unbalanced Doesn’t Always Ruin Performance**. For MNIST, the difference might be subtle overall, but the minority digit can suffer.

### 8.1. Example Output
- **Accuracy** ~ 0.94 overall, but for **digit 7**, recall could drop to 0.6 or 0.7.  
- The **confusion matrix** might show 7 → 2 confusion.  

---

## 9. Conclusions and Lessons

1. **Form a Hypothesis**: Predict how model biases will emerge with class imbalance.  
2. **Per-Class vs. Weighted Metrics**: Class-level Precision/Recall is critical to detect where the performance collapses (digit 7).  
3. **Confusion Matrix**: Offers granular insight. Look at row=7 to see which digits are confused for 7.  
4. **Unbalanced Data**: Not always devastating, but can systematically harm performance on underrepresented classes. Real-world tasks (e.g., rare disease detection) must handle imbalance carefully.

---

## 10. Further Reading

- **He & Garcia (2009)**: *Learning from imbalanced data* (IEEE TKDE).  
- **Buda et al. (2018)**: *Systematic study of the class imbalance problem in convolutional neural networks* (Neural Networks).  
- **scikit-learn**: [Class Imbalance Documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).  
- **Data Augmentation**: In real-world scenarios, oversampling or other techniques can mitigate imbalance.

---

```

**How to Use These Notes in Obsidian**:

1. **Create a new note** in your Obsidian vault, e.g., `MNIST_Unbalanced_7s.md`.  
2. **Paste** the entire markdown content (including frontmatter `---`).  
3. Add or modify headings, internal links, or references to suit your organizational style.  

This provides a full guide for performing the **MNIST unbalanced code challenge**. It illustrates how an **unbalanced dataset** can lead to **biased** performance on certain digits—highlighting the importance of evaluating **per-class** metrics and using confusion matrices for deeper insight.