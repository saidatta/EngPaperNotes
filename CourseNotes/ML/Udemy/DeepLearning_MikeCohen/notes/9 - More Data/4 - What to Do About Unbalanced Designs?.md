aliases: [Unbalanced Data Solutions, Class Imbalance Strategies, Data Augmentation, SMOTE]
tags: [Data, Imbalanced Classes, Deep Learning, Practical Tips]
## Overview
When your dataset has **unbalanced** or **skewed** class distributions, standard deep learning approaches can fail—often defaulting to predicting the majority class, yielding misleadingly high accuracy. This lecture introduces **practical strategies** for dealing with unbalanced data, each with pros and cons.

---
## 1. Motivating Examples

1. **Fraud Detection**  
   - Legitimate transactions vastly outnumber fraudulent ones.  
2. **Disease Prediction**  
   - Most patients tested are healthy, so positives are rare.  
3. **Rare Event Classification**  
   - Earthquake detection, rare mechanical failures, etc.

In such scenarios, you might see a model **achieve 99% accuracy** simply by predicting the majority class every time, ignoring minority classes entirely.

---

## 2. Strategies for Unbalanced Data

Below are **six** approaches (though #3 and #4 are closely related):

1. **Get More Data (Best if Possible)**  
   - **Ideal solution**: Increase the minority class samples so classes are more balanced.  
   - Realistically, collecting more data can be **hard** (time, cost, rarity).
   - Could involve collaborating with colleagues or searching external datasets.

2. **Under-Sampling**  
   - **Remove** some observations from the majority class.  
   - This yields a balanced but **smaller** dataset.  
   - Pros:
     - Easy to implement.  
     - Forces balanced distribution.  
   - Cons:
     - You **throw away** potentially valuable majority-class examples.  
     - Leads to **less overall data**, risking underfitting.

3. **Over-Sampling**  
   - **Duplicate** minority class samples to match the majority class size.  
   - Pros:
     - Easy to implement; helps emphasize minority class in training.  
   - Cons:
     - No new information added; increases risk of **overfitting** the minority class.  
     - Model may memorize exact duplicate samples.

4. **Data Augmentation**  
   - Like over-sampling, but instead of exact copies, you apply **transformations** to create *new* (augmented) samples.  
   - Common in **image tasks**: random flips, rotations, color jitter, etc.  
   - In tabular or textual data, augmentation is trickier but sometimes possible.  
   - Helps add diversity, reduces overfitting.

5. **Synthetic Minority Over-Sampling**  
   - Rather than duplicates, generate **synthetic** examples in the minority class.  
   - E.g., **SMOTE** (Synthetic Minority Over-sampling Technique) creates new samples by interpolating feature vectors between existing minority samples.  
   - Adds more variety than pure duplication, but still may be simplistic in high-dimensional or complex feature spaces.

6. **Non-Deep-Learning Approach**  
   - If your data distribution is severely skewed, **consider** whether deep learning is the best solution.  
   - **Classical methods** (e.g., linear/quadratic discriminant analysis, clustering, linear regression, decision trees) can sometimes handle imbalance more gracefully or require smaller data volumes.

> None of these strategies are magic bullets—each has trade-offs.

---

## 3. Examples & Code Snippets

Below are illustrative Python snippets for a few key techniques. Assume you have:
- **X**: feature matrix
- **y**: label array (0 or 1, with class 1 the minority).

### 3.1 Under-Sampling
A simple approach: randomly select an equal number of majority samples:

```python
import numpy as np

def undersample(X, y):
    # Indices of minority class
    idx_minority = np.where(y==1)[0]
    idx_majority = np.where(y==0)[0]
    
    n_min = len(idx_minority)
    # Shuffle majority indices and pick the first n_min
    np.random.shuffle(idx_majority)
    idx_majority = idx_majority[:n_min]
    
    # Combine
    idx_new = np.concatenate([idx_minority, idx_majority])
    np.random.shuffle(idx_new)
    
    return X[idx_new], y[idx_new]

X_under, y_under = undersample(X, y)
print("Original:", X.shape, y.shape)
print("After Undersampling:", X_under.shape, y_under.shape)
```

### 3.2 Over-Sampling
Duplicate minority examples until classes match:

```python
def oversample(X, y):
    idx_minority = np.where(y==1)[0]
    idx_majority = np.where(y==0)[0]
    
    n_maj = len(idx_majority)
    n_min = len(idx_minority)
    
    # Number of extra samples needed
    extra_needed = n_maj - n_min
    
    # Randomly pick minority indices to copy
    idx_to_copy = np.random.choice(idx_minority, size=extra_needed, replace=True)
    
    X_extra = X[idx_to_copy]
    y_extra = y[idx_to_copy]
    
    X_over = np.concatenate([X, X_extra], axis=0)
    y_over = np.concatenate([y, y_extra], axis=0)
    
    # Shuffle
    idx_new = np.random.permutation(len(y_over))
    
    return X_over[idx_new], y_over[idx_new]

X_over, y_over = oversample(X, y)
print("After Oversampling:", X_over.shape, y_over.shape)
```

### 3.3 Data Augmentation Example (Images)
For **images**, PyTorch’s `torchvision.transforms` can randomize flips, rotations, color jitter, etc.

```python
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ToTensor()
])

# Use in a dataset
dataset_aug = ImageFolder(root="images_folder", transform=transform)
```

**Every epoch**, your model sees slightly different transformations for minority class images, effectively increasing data diversity.

### 3.4 Synthetic Data (SMOTE) 
Here’s a conceptual snippet for how SMOTE might work (simplified):

```python
def smote_simplified(X, y, k=5):
    # X,y are minority samples only
    # In practice, you only apply SMOTE to the minority class
    # and then combine with the majority class
    new_samples = []
    n = len(X)
    for i in range(n):
        # find k nearest neighbors, pick one at random
        # here we skip the actual neighbor step for brevity
        j = np.random.randint(0, n)
        lam = np.random.rand()
        x_synthetic = X[i] + lam * (X[j] - X[i])
        new_samples.append(x_synthetic)
    
    return np.array(new_samples)
```

> Real SMOTE includes nearest-neighbor searches in feature space to produce more realistic samples.

---

## 4. Pros & Cons Recap

| Strategy  | Pros                                          | Cons                                                         |
|-----------|-----------------------------------------------|--------------------------------------------------------------|
| **Get More Data**     | Best solution in principle | Expensive, time-consuming, or sometimes impossible           |
| **Under-Sampling**    | Easy, ensures balance     | Discards valuable data, risk of underfitting                 |
| **Over-Sampling**     | Simple to implement       | Risk of overfitting to duplicated samples                    |
| **Data Augmentation** | Creates variety in minority class; reduces overfitting | Implementation depends on data type; not always trivial  |
| **Synthetic Samples** (SMOTE) | More variety than duplication | May introduce unnatural examples if not carefully configured |
| **Non-Deep-Learning** | Could handle imbalance better in some cases | Might lose the power & flexibility of deep learning         |

---

## 5. Key Takeaways

1. **Check Class Distribution**  
   - Always inspect how many samples belong to each class before training.
2. **Monitor Per-Class Metrics**  
   - Overall accuracy can be misleading. Also look at **precision/recall/F1** for each class.
3. **Practical Solutions**  
   - **Get More Data** if possible. Otherwise, consider under-/over-sampling, augmentation, or generating synthetic samples (SMOTE).
4. **No Universal Best**  
   - All methods have trade-offs. Experiment with multiple approaches for your dataset.

---

## 6. Further Reading & Tips

- **SMOTE Paper**: *Chawla et al., 2002*. *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research.  
- **Imbalanced-learn** library in Python ([link](https://imbalanced-learn.org/)) provides ready-made implementations of over-sampling, under-sampling, and SMOTE.
- For images, explore [**Albumentations**](https://albumentations.ai/) or [**torchvision.transforms**](https://pytorch.org/vision/stable/transforms.html) for robust data augmentation techniques.

---

## 7. Code Example: Weighted Loss in PyTorch

Another approach is to use a **weighted loss** function, giving **higher penalty** to mistakes on minority class:

```python
# Suppose class 1 has freq < class 0
weight = torch.tensor([1.0, 3.0])  # heavier weight on minority
criterion = nn.CrossEntropyLoss(weight=weight)

# Then use as normal in your training loop
logits = model(X_batch)
loss = criterion(logits, y_batch)
loss.backward()
...
```

This does **not** add data, but **rebalances** how errors in minority classes contribute to loss, potentially mitigating bias.

---

## 8. Conclusion
Unbalanced data is common in real-world applications (fraud, rare diseases, etc.). You have multiple strategies at your disposal—**the best** depends on:
- Data availability
- Domain feasibility (e.g., images easily augmented, tabular data not so easy)
- Desired metrics (some prefer recall over precision, etc.)

**End of Notes – "Data: What to Do About Unbalanced Designs?"**
```