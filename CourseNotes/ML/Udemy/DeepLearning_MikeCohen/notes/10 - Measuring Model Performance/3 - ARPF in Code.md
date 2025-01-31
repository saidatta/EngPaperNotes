Below is a **comprehensive set of Obsidian-style notes** for the lecture *“Measuring Model Performance: ARPF in Code”*. These notes are intended for a **PhD-level engineer**, combining **theoretical insight** and **practical code** to explore **accuracy, recall, precision, and F1** (ARPF) with **synthetic data**. Feel free to copy these into your Obsidian vault, modify headings, or integrate additional internal links (e.g., `[[Confusion Matrix]]`, `[[ROC Curves and AUC]]`).

---

```markdown
---
title: "Measuring Model Performance: ARPF in Code"
tags: [deep-learning, classification, model-evaluation, ARPF]
date: 2025-01-30
---

# Measuring Model Performance: 

## 1. Overview

In previous lectures, we established **four key metrics** for classification performance:

1. **Accuracy**  
2. **Recall** (Sensitivity)  
3. **Precision**  
4. **F1** (harmonic mean of Precision & Recall)

These metrics provide different **perspectives** on how a binary classifier (or detection system) performs. Here, we explore a **synthetic** experiment to:

- Generate **random confusion matrices** (TP, FP, TN, FN).  
- Compute **ARPF**.  
- Examine **correlations** and **bias** trade-offs (e.g., high precision vs. recall).  
- Visualize how **Accuracy** and **F1** relate, color-coding points by precision or recall to reveal model bias trends.

**Why synthetic data?**  
- We can systematically cover a wide range of performance outcomes.  
- No actual classifier or dataset is needed; we can purely sample possible confusion matrices.

---

## 2. Mathematical Recap

### 2.1. Confusion Matrix and Metrics

For a binary classification task, define:

- **TP**: True Positives  
- **FP**: False Positives  
- **TN**: True Negatives  
- **FN**: False Negatives  

\[
\begin{aligned}
\text{Accuracy} & = \frac{TP + TN}{TP + FP + TN + FN},\\
\text{Recall (Sensitivity)} & = \frac{TP}{TP + FN},\\
\text{Precision} & = \frac{TP}{TP + FP},\\
F1 & = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}.
\end{aligned}
\]

**Key**: 
- **Precision** penalizes *false positives*.  
- **Recall** penalizes *false negatives*.  
- **F1** balances both errors.  
- **Accuracy** can be misleading when **unbalanced** data or bias exist.

---

## 3. Generating Synthetic Data

### 3.1. Approaches

One simple method:
1. Fix a total number of “positives” = \(N\), and a total number of “negatives” = \(N\).  
2. Randomly choose \(\text{TP}\) out of \(N\) positives, then \(\text{FN} = N - \text{TP}\).  
3. Randomly choose \(\text{TN}\) out of \(N\) negatives, then \(\text{FP} = N - \text{TN}\).  

Hence, each run yields a confusion matrix \(\{TP, FP, TN, FN\}\) summing to \(2N\) total samples. Different distributions of \(\{\text{TP}, \text{FP}, \text{TN}, \text{FN}\}\) simulate varying biases or performance.

### 3.2. Code Example

```python
```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_experiments = 10000  # how many synthetic confusion matrices
N = 50  # half for positives, half for negatives => total = 2*N

acc_vals = []
prec_vals = []
rec_vals  = []
f1_vals   = []

for _ in range(num_experiments):
    # 1) Generate random TP in [1, N]
    TP = np.random.randint(1, N+1)
    FN = N - TP
    # 2) Generate random TN in [1, N]
    TN = np.random.randint(1, N+1)
    FP = N - TN
    
    # 3) Compute ARPF
    # avoid zero denominators
    denom = (TP + FP + TN + FN)
    acc  = (TP + TN) / denom
    
    prec = TP / (TP + FP) if (TP+FP) != 0 else 0.0
    rec  = TP / (TP + FN) if (TP+FN) != 0 else 0.0
    
    if prec+rec > 0:
        f1 = 2 * (prec * rec) / (prec + rec)
    else:
        f1 = 0.0
    
    acc_vals.append(acc)
    prec_vals.append(prec)
    rec_vals.append(rec)
    f1_vals.append(f1)
```
```

- We store each metric in an array.  
- Over many runs, these points fill out the “metric space,” capturing many possible confusion matrices.

---

## 4. Visualizing ARPF Relationships

We especially want to see how **Accuracy** and **F1** compare, color-coding points by **Precision** or **Recall** to reveal biases.

### 4.1. Scatter Plots

```python
```python
acc_vals = np.array(acc_vals)
f1_vals  = np.array(f1_vals)
prec_vals = np.array(prec_vals)
rec_vals  = np.array(rec_vals)

plt.figure(figsize=(12,5))

# 1) color by precision
plt.subplot(1,2,1)
plt.scatter(acc_vals, f1_vals, c=prec_vals, s=5, cmap='viridis')
plt.xlabel("Accuracy")
plt.ylabel("F1 Score")
plt.title("Color-coded by Precision")

# 2) color by recall
plt.subplot(1,2,2)
plt.scatter(acc_vals, f1_vals, c=rec_vals, s=5, cmap='viridis')
plt.xlabel("Accuracy")
plt.ylabel("F1 Score")
plt.title("Color-coded by Recall")

plt.tight_layout()
plt.show()
```
```

**Interpretation**:
- Points with **high** accuracy but **low** F1 often have **imbalanced** errors or a specific bias.  
- Points with moderate accuracy but **high** F1 typically balance **precision** and **recall** well.  
- The color scale reveals whether each dot’s “bias” is towards saying “yes” (leading to high recall, lower precision) or “no” (leading to high precision, lower recall).

---

## 5. Insights from the Scatter Plots

1. **Accuracy-F1 Correlation**:  
   - We see a **general** correlation: higher accuracy often corresponds to higher F1.  
   - But many points have moderate accuracy but also moderate F1 (the correlation is **not perfect**).

2. **Color Gradients**:  
   - If coloring by **precision**, dots with high precision might cluster at high accuracy but *not necessarily* at high F1.  
   - If coloring by **recall**, a different clustering emerges—some points have high recall but moderate accuracy.

3. **Model Bias**:  
   - High **precision** but low **recall** → model rarely says “positive,” but is correct when it does.  
   - High **recall** but low **precision** → model says “positive” frequently, capturing more positives but also more false alarms.

Hence, **accuracy** alone won’t highlight these **bias** trade-offs, but **ARPF** helps diagnosing them.

---

## 6. Additional Explorations

### 6.1. Alternative Summations
- Instead of randomizing **TP** & **TN**, one could systematically **grid** across \(\{TP, FP, TN, FN\}\) to ensure uniform coverage of confusion matrix space.

### 6.2. 3D or Multi-Panel Plots
- Plot **Precision** vs. **Recall** vs. **F1** or create 2D slices color-coded by **Accuracy**.

### 6.3. Impact of Data Imbalance
- Weighted sampling to reflect heavy imbalance (e.g., `FN` likely higher or smaller for certain distributions). Check how **accuracy** may overshadow the other metrics.

---

## 7. Summary

1. **ARPF**: We generated random confusion matrices to systematically explore **Accuracy, Recall, Precision, and F1**.  
2. **Scatter Plot Observations**:
   - They are **positively correlated** but not perfectly.  
   - High accuracy can coincide with moderate or even low F1 if the classifier is heavily biased in one direction.  
3. **Interpretation**:
   - **Precision** and **recall** color mappings highlight model biases that **accuracy** alone hides.  
   - **F1** merges these biases but doesn’t always correlate with accuracy (still valuable for unbalanced or bias-sensitive tasks).

**Next Steps**: 
- Apply these metrics to **real** datasets (wine quality, MNIST).  
- Observe **empirical** relationships, including how data distribution affects ARPF.

---

## 8. References

- **Signal Detection Theory** in classification metrics: *Green & Swets (1966)*  
- **Scikit-learn** metrics for classification: [https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)  
- **Bias-Variance** perspective: *Bishop, C. M. (2006)*, *Pattern Recognition and Machine Learning*

---
```

**How to Use These Notes in Obsidian**:

1. **Create** a new note (e.g. `APRF_in_Code.md`).  
2. **Paste** the entire content (including the YAML frontmatter) from above.  
3. Add your own internal links (e.g. `[[Confusion Matrix Code]]`, `[[Wine Quality ARPF Example]]`).  
4. Optionally expand on the plotting or distribution arguments if you want more advanced or specialized experiments.

This concludes a **detailed** exploration of **accuracy**, **recall**, **precision**, and **F1** using **synthetic** confusion matrices, visually illustrating how these metrics align and diverge—especially with respect to **model bias**.