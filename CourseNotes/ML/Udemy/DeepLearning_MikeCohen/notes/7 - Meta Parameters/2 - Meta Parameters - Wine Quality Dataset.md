
aliases: [Wine Quality, Data Preprocessing, Classification, Meta Parameters]
tags: [Deep Learning, Lecture Notes, Dataset Exploration, Python, PyTorch]

In this lecture, we explore the **Wine Quality** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). The dataset includes chemical analyses of wines and corresponding subjective quality ratings. We will:

1. Discuss the origin and characteristics of the dataset.
2. Perform **data exploration** and **visualization** in Python.
3. **Clean** the data (handle outliers).
4. **Normalize** the data with z-scoring.
5. **Binarize** the quality column (creating a 0/1 label).
6. Prepare the dataset for use in PyTorch (tensors, DataLoader).

This notebook sets the stage for upcoming lessons on building and tuning deep learning models with multiple meta parameters, all using this dataset.

---

## 1. Overview of the Wine Quality Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- **Content**: Measurements of chemical properties such as acidity, sugar, pH, sulfur dioxide, etc.  
- **Goal**: Predict the wine quality rating (originally 3–8 for the red wine dataset).

### Why This Dataset Is Interesting
- **Subjectivity**: Wine quality is inherently subjective; different tasters give different scores.  
- **Chemical Features**: Each row includes 11 features derived from chemical analysis, like acidity and sulfur dioxide levels.  
- **Classification / Regression**: We can treat the original “quality” rating as a continuous or discrete label. Here, we will **binarize** it for a classification task (low vs. high quality).

---

## 2. Initial Setup and Imports

Below is an outline of the Python libraries we use:

- **Pandas** for data handling  
- **NumPy** (often implicitly used via Pandas, optional direct usage)  
- **Matplotlib** and **Seaborn** for visualizations  
- **PyTorch** for building data loaders and, later, neural networks  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader

# For nicer plots in notebooks
%matplotlib inline
sns.set(style="whitegrid")
```

---

## 3. Loading the Data

The Wine Quality dataset is available via a direct URL from UCI. We can read it into Pandas without needing to manually download the file.

```python
# URL for Wine dataset (red wine)
wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Notice that data is separated by semicolons (sep=";")
data = pd.read_csv(wine_url, sep=";")

print("Data shape:", data.shape)
data.head()
```

### Observations

1. **Shape**: Expect around 1600+ rows and 12 columns (11 chemical features + 1 quality column).  
2. **Columns**:  
   - `fixed acidity`  
   - `volatile acidity`  
   - `citric acid`  
   - `residual sugar`  
   - `chlorides`  
   - `free sulfur dioxide`  
   - `total sulfur dioxide`  
   - `density`  
   - `pH`  
   - `sulphates`  
   - `alcohol`  
   - `quality` (target label, originally an integer from 3 to 8 in this dataset).

---

## 4. Basic Exploration

### 4.1 Descriptive Statistics

```python
data.describe()
```

- **Count**: Each column should have the same number of non-null entries.  
- **Mean / Std**: Observe large variability in means and standard deviations across different columns.  
- **Min / Max**: Check for potential outliers.

**Key Point**: Different scales across features can **hurt neural network training**. We’ll address this by **normalizing** each column.

### 4.2 Unique Values

We can see how many unique values exist in each column:

```python
for col in data.columns:
    unique_vals = len(data[col].unique())
    print(f"{col}: {unique_vals} unique values")
```

This gives a quick sense of **categorical vs. continuous** columns. Notably:
- `quality` has only a handful of unique values (e.g., 3–8).

---

## 5. Visual Explorations

### 5.1 Pairwise Scatter Plots

A **pairplot** helps visualize potential correlations among features. We’ll take a subset of columns to keep the plot manageable and color-code by `quality`.

```python
subset_cols = ["fixed acidity", "citric acid", 
               "volatile acidity", "alcohol", "quality"]

sns.pairplot(data[subset_cols], hue="quality", diag_kind="kde")
plt.show()
```

- **Diagonal**: Distribution of each feature (KDE or histogram).  
- **Off-diagonal**: Relationship between two features with points colored by `quality`.  

**Insight**: It’s not trivial to see a direct relationship between any **two** features and `quality`. However, a network can combine **all 11 features** in higher-dimensional space, potentially uncovering patterns invisible in 2D projections.

### 5.2 Box Plots to Inspect Scales

```python
plt.figure(figsize=(12, 6))
sns.boxplot(data=data.drop(columns=["quality"]))
plt.xticks(rotation=45)
plt.title("Distribution of Wine Features (Unscaled)")
plt.show()
```

- We see different **ranges** and **potential outliers**.  
- Notably, `free sulfur dioxide` and `total sulfur dioxide` might have very large values for a few samples.  
- Outliers can degrade model training.

---

## 6. Handling Outliers

One outlier example is in `total sulfur dioxide`, where certain rows can have extremely large values. We may remove or clip such outliers.

```python
# For instance, remove rows where total sulfur dioxide > 200
before_shape = data.shape
data = data.loc[data["total sulfur dioxide"] < 200]
after_shape = data.shape

print("Shape before removing outliers:", before_shape)
print("Shape after removing outliers: ", after_shape)
```

**Trade-off**: We lose some data, but we gain a more stable distribution without extreme values.

---

## 7. Normalizing the Data (Z-Score)

Because the features span very different ranges, we use **z-score** normalization on all features **except** the target column `quality`.

\[
z = \frac{x - \mu}{\sigma}
\]

Where \(\mu\) is the mean and \(\sigma\) is the standard deviation of the feature.

### 7.1 Manual Loop Over Columns

```python
feature_cols = [c for c in data.columns if c != "quality"]

for col in feature_cols:
    col_mean = data[col].mean()
    col_std  = data[col].std()
    data[col] = (data[col] - col_mean) / col_std
```

### 7.2 Pandas Built-In

Alternatively, Pandas has built-ins for scaling, but we will keep it explicit here.

### 7.3 Verification

```python
data.describe()
```

- **Means** close to 0 for each feature.  
- **Std** close to 1 for each feature.

### 7.4 Post-Normalization Box Plot

```python
plt.figure(figsize=(12, 6))
sns.boxplot(data=data.drop(columns=["quality"]))
plt.xticks(rotation=45)
plt.title("Distribution of Wine Features (Normalized)")
plt.show()
```

Now each feature distribution should be centered around **0** and in a comparable range (roughly \([-3, 3]\) or so).

---

## 8. Binarizing the `quality` Column

Original `quality` values range roughly from **3** to **8**, but their distribution is **highly imbalanced** (most wines are labeled **5** or **6**).

### 8.1 Check the Distribution

```python
quality_counts = data["quality"].value_counts()
print(quality_counts)

plt.figure(figsize=(6,4))
sns.barplot(x=quality_counts.index, y=quality_counts.values)
plt.title("Histogram of Wine Quality Ratings")
plt.xlabel("Quality Rating")
plt.ylabel("Count")
plt.show()
```

**Insight**: The dataset is not balanced among the 6 rating levels.

### 8.2 Create a Boolean Quality Label

We decide to classify wines as “low” (0) or “high” (1). One common approach:
- **Low** = ratings ≤ 5  
- **High** = ratings ≥ 6  

```python
data["boolQuality"] = 0
data.loc[data["quality"] > 5, "boolQuality"] = 1

# Inspect a few rows
data[["quality", "boolQuality"]].head(10)
```

This approach merges classes (3, 4, 5) into **0** and (6, 7, 8) into **1**. 

> **Note**: This step is somewhat subjective; different thresholds could be used.

---

## 9. Preparing Data for PyTorch

### 9.1 Separate Features and Labels

We will drop the original `quality` (though you may keep it for reference) and use `boolQuality` as the label.

```python
feature_cols = [c for c in data.columns 
                if c not in ["quality", "boolQuality"]]
X = data[feature_cols].values  # Numpy array of features
y = data["boolQuality"].values # Numpy array of labels
```

### 9.2 Convert to Tensors

PyTorch expects **tensors** rather than Numpy arrays.

```python
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long).view(-1, 1)
```

- `dtype=torch.float32` for features.  
- `dtype=torch.long` (or `torch.int64`) for classification labels.  
- `.view(-1, 1)` or `unsqueeze(-1)` to ensure labels have shape `(N, 1)` instead of `(N,)`.

### 9.3 Train/Test Split

Let’s do a simple split (e.g., 80%-20%) to keep a subset for testing.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_t, y_t, test_size=0.2, shuffle=True, random_state=42
)

print("Train set:", X_train.shape, y_train.shape)
print("Test set: ", X_test.shape, y_test.shape)
```

### 9.4 Create DataLoaders

DataLoaders provide an easy way to batch data and shuffle for training.

```python
batch_size = 64
drop_last = True  # ensures each batch is the same size

train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True, drop_last=drop_last)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, 
                          shuffle=False, drop_last=drop_last)

print("Number of training batches:", len(train_loader))
print("Number of testing batches: ", len(test_loader))
```

- `shuffle=True` is typically used for **training** data.  
- `drop_last=True` discards any incomplete batch that doesn’t reach the full `batch_size`, e.g., the final batch might only have 29 samples leftover if total samples are not divisible by 64.

---

## 10. Summary and Next Steps

**We now have**:

1. **Cleaned** the Wine Quality dataset (removing extreme outliers).  
2. **Normalized** it via z-score scaling.  
3. **Binarized** the `quality` column into `boolQuality`.  
4. **Split** into training and testing sets.  
5. **Wrapped** these sets into PyTorch `DataLoaders`.

This dataset is ready for **model-building** and **hyperparameter tuning**. In the upcoming lessons, we will:

- Build neural networks that take the **11 normalized features** as input.  
- Output a **single sigmoid** or logistic classification for `boolQuality`.  
- Experiment with **meta parameters** (learning rate, number of layers, etc.).  
- Evaluate how well we can predict (low vs. high) wine quality based on chemical composition.

---

## 11. Additional Resources

- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/)  
- [Seaborn Documentation](https://seaborn.pydata.org/) (for pairplots, bar plots, etc.)

---

## 12. References and Further Reading

1. Cortez, P. et al. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4):547–553.
2. Goodfellow, Bengio, Courville, *Deep Learning*. (Section on Data Preprocessing)
3. Scholkopf & Smola, *Learning with Kernels*. (General references on data transformations)

---

**Created by**: [Your Name / Lab / Date]  
**Based on Lecture**: “Meta Parameters: Wine Quality Dataset”
```