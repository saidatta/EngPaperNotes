
aliases: [Data Normalization, z-scoring, min-max scaling]
tags: [Deep Learning, Lecture Notes, Data Preprocessing, Neural Networks]

**Data Normalization** is a critical step in deep learning, helping ensure that all input features are on comparable scales. This facilitates stable training, prevents certain features from dominating, and helps avoid numerical instability in model parameters.  

In this lesson, we'll discuss:
1. **Motivation** for normalizing data.
2. The **two main methods** for data normalization:
   - **Z-scoring** (a.k.a. *z-transformation*)
   - **Min-max scaling**
3. **When** to use each method.
4. **Examples** (both equations and Python code).

---

## 1. Why We Need Data Normalization

### 1.1 Preventing Dominance by Large-Scale Features

- Neural networks compute **weighted sums** of inputs:
  \[
    z = \mathbf{w} \cdot \mathbf{x} + b
  \]
- If one feature \(\mathbf{x}_i\) has a much larger range of values than another \(\mathbf{x}_j\), it can **dominate** the dot product and disproportionately influence the gradient updates.
- This can lead to:
  - **Unstable training** (e.g., some weights may “explode”).
  - **Overemphasis** on large-valued features, even if they are **not** the most informative.

### 1.2 Equal Treatment of Samples

- We want each **data sample** (row) to similarly contribute to the training loss.  
- If one sample has very large values, it might push a **huge** gradient update compared to another sample with smaller values, causing **inconsistent** or **biased** learning.

### 1.3 Numerical Stability

- Keeping input features within a **similar range** (e.g., ~\([-2, +2]\) or \([-1, +1]\)) also helps with:
  - **Gradient-based optimization** (better behaved gradients).
  - **Faster convergence** (less time tuning learning rates).
  - Avoiding issues with **floating-point precision**.

---

## 2. Z-Score Normalization (Z-Scoring)

**Z-Scoring** re-centers and rescales data to have:
- Mean = 0
- Standard deviation = 1

### 2.1 Formula

For a given feature \( X \):

\[
x'_i = \frac{x_i - \mu}{\sigma}
\]
where:
- \(\mu = \frac{1}{N}\sum_{i=1}^N x_i\) is the **mean** of the feature.
- \(\sigma\) is the **standard deviation** of the feature.

### 2.2 Properties

- **Mean** of \(x'\) is 0.
- **Standard deviation** of \(x'\) is 1.
- **Correlation Preserved**: The *relative* distances between values remain the same; this is a **linear** shift and scale.

### 2.3 Example Visualization

Imagine a feature **Height** in cm, typically in \([150, 200]\).  
After z-scoring, the histogram shape stays the same, but is re-centered around 0 with a standard deviation of 1.

```
Height (cm) --> Z-scored Height
150 cm       --> (150 - mean)/std
170 cm       --> (170 - mean)/std
...
```

### 2.4 Python Snippet

```python
import numpy as np
import pandas as pd

def z_score_normalize(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sigma = series.std()
    return (series - mu) / sigma

# Example usage with random numeric data
df = pd.DataFrame({
    'height_cm': np.random.normal(loc=170, scale=10, size=1000)
})
df['height_z'] = z_score_normalize(df['height_cm'])

print(df[['height_cm', 'height_z']].describe())
```

---

## 3. Min-Max Scaling

**Min-max scaling** transforms data so that the **minimum** value becomes 0 and the **maximum** value becomes 1. 

### 3.1 Formula for [0, 1] Range

\[
x'_i = \frac{x_i - \min(X)}{\max(X) - \min(X)}
\]
- \(\min(X)\) is the smallest value in the feature.
- \(\max(X)\) is the largest value in the feature.
- After this transform, values lie in \([0, 1]\).

### 3.2 Generalization to [A, B] Range

If you need a different target range \([A, B]\):
\[
x''_i = A + \bigl(x'_i \times (B - A)\bigr)
\]
where \(x'_i\) is already scaled to \([0, 1]\) from the previous step.

- E.g., for \([-1, +1]\), set \(A=-1\), \(B=+1\).

### 3.3 Properties

- Also a **linear** transformation.  
- Preserves **relative** distances among data points (if no outliers exist).  
- If **outliers** are present, the entire distribution can get **squeezed** towards 0 for most data, and 1 for the extreme high outlier.

### 3.4 Python Snippet

```python
def min_max_scale(series: pd.Series, new_min=0.0, new_max=1.0) -> pd.Series:
    old_min, old_max = series.min(), series.max()
    # Scale to [0, 1]
    scaled = (series - old_min) / (old_max - old_min + 1e-12)
    # Then scale to [new_min, new_max]
    return scaled * (new_max - new_min) + new_min

# Example usage
df['height_01'] = min_max_scale(df['height_cm'], 0, 1)
print(df[['height_cm', 'height_01']].describe())
```

*(Added `1e-12` to avoid division by zero in edge cases.)*

---

## 4. Example: Visualizing Both Methods

Suppose we have a single feature `X` with a wide range. We’ll create a toy example:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Toy data
X = np.array([2, 5, 5.5, 20, 200]).astype(float)

# Z-scoring
X_z = (X - X.mean()) / X.std()

# Min-max scaling to [0,1]
X_mm = (X - X.min()) / (X.max() - X.min())

print("Original X:       ", X)
print("Z-score normalized:", np.round(X_z, 3))
print("Min-max scaled:    ", np.round(X_mm, 3))

# Visual comparison
plt.figure(figsize=(8,3))
sns.scatterplot(x=X, y=[1]*len(X), label='Original X', s=60)
sns.scatterplot(x=X_z, y=[2]*len(X), label='Z-scored', s=60)
sns.scatterplot(x=X_mm, y=[3]*len(X), label='Min-max', s=60)
plt.yticks([1,2,3], ["Original", "Z-scored", "Min-max"])
plt.show()
```

- Notice how the **order** of points remains the same, but the **numerical ranges** differ.

---

## 5. When to Use Which?

1. **Z-Scoring**:
   - More common for data that follows (or roughly follows) a **normal-like** distribution or has a meaningful mean and standard deviation.
   - Particularly helpful when the feature has **unbounded** range (e.g., negative to positive) or typical real-valued signals (like height, weight, etc.).
   - Results in **mean = 0**, **std = 1**.

2. **Min-Max Scaling**:
   - More common for data with **bounded** numerical ranges (e.g., pixel intensities in images often 0–255).
   - Makes sense for features that are inherently nonnegative, or for features you want to interpret on a [0,1] scale.
   - Susceptible to **outliers**: a single extreme value can shrink the majority of the data to a very narrow interval near 0.

### 5.1 Example: Image Data

- **Images**: Often store pixel intensities from 0 to 255. A typical approach is:
  - Min-max scale from [0, 255] → [0, 1] or [−1, +1].

### 5.2 Example: Sensor / Normal Distribution

- **Sensor** data (accelerometers, voltages, etc.) might be unbounded or have indefinite range:
  - Z-scoring ensures a standard distribution if the data is approximately normal.

---

## 6. Key Takeaways

1. **Normalization is Essential** in most deep learning applications:
   - Prevents a few features or samples from dominating gradient updates.
   - Stabilizes training and can improve convergence speed.

2. **Z-Scoring** (subtract mean, divide by std):
   - Best for data with a “center” and “spread”.
   - Distribution remains the same shape, but re-centered around 0 and scaled by std.

3. **Min-Max Scaling** (scale to [0,1] or another range):
   - Useful when feature values are bounded (like pixel intensities).
   - Extreme values can heavily influence the transformation.

4. **Next Steps**:  
   - **Batch Normalization** (and related methods) further address normalization needs *within* neural network layers, not just at the input level.
   - **Practice** on different datasets to see which method works best.

---

## 7. Further Reading & References

- [Goodfellow, Bengio, Courville. *Deep Learning*. MIT Press. (Chapter on data preprocessing)](https://www.deeplearningbook.org/)
- [PyTorch Docs on Transforms](https://pytorch.org/vision/stable/transforms.html) (common for image data)
- [Scikit-Learn Preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) (contains `StandardScaler`, `MinMaxScaler` implementations)
- [Blog: “Data Normalization in Deep Learning” by Kaggle/Medium](#)

---

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: *“Meta parameters: Data Normalization”*  
```