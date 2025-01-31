Below is a **comprehensive set of Obsidian-style notes** covering the lecture on *“Measuring Model Performance: Accuracy, Precision, Recall, and F1”*. These notes are designed for a **PhD-level engineer**, providing both **conceptual depth** and **practical examples** (including code and visualizations). Feel free to copy/paste these into your Obsidian vault, modify headings, and add any internal note links.

---
title: "Measuring Model Performance: Accuracy, Precision, Recall, F1"
tags: [deep-learning, machine-learning, model-performance, classification]
## 1. Overview

In this lecture, we extend our understanding of model performance evaluation beyond the basic metrics of **loss** and **accuracy**. We focus on:

- **Accuracy**: Overall measure of correctness.  
- **Precision**: “When the model predicts *positive*, how often is it correct?”  
- **Recall** (aka **Sensitivity**): “Of all the actual positives, how many did the model *catch*?”  
- **F1 Score**: A harmonic mean of Precision and Recall, balancing both types of errors.

**Why do we need these additional metrics?**  
- A model can appear to have **high accuracy** but still perform poorly on certain classes (especially when data are unbalanced).  
- Different application contexts penalize **false positives** vs. **false negatives** differently.  

---

## 2. Recap: The Four Categories of Responses

From the lecture “Measuring Model Performance: Two Perspectives in the World,” we know each sample falls into one of four categories when comparing **Reality** (ground truth) vs. **Prediction** (model output):

|                        | Reality: Positive (Cat) | Reality: Negative (Not Cat) |
|------------------------|:-----------------------:|:---------------------------:|
| **Predicted: Positive** | **True Positive (TP)**  | **False Positive (FP)**     |
| **Predicted: Negative** | **False Negative (FN)** | **True Negative (TN)**      |

- **True Positive (TP)**: Model says *“Yes”* and the sample is actually positive.  
- **False Positive (FP) / “False Alarm”**: Model says *“Yes”* but the sample is actually negative.  
- **False Negative (FN) / “Miss”**: Model says *“No”* but the sample is actually positive.  
- **True Negative (TN)**: Model says *“No”* and the sample is actually negative.

These four numbers make up the **confusion matrix**.

---

## 3. Accuracy

### 3.1 Definition

\[
\text{Accuracy} \;=\; \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
\]

It is simply the ratio of **all correct** predictions (TP + TN) over **all predictions** (TP + TN + FP + FN).

### 3.2 Interpretation
- **High Accuracy** generally means the model is correct most of the time.
- However, **accuracy alone** can be misleading in:
  - **Unbalanced Datasets** (e.g., 99% of data is negative).
  - Situations where we specifically care about which *type* of mistake is more critical (false positives vs. false negatives).

---

## 4. Precision

### 4.1 Definition

\[
\text{Precision} \;=\; \frac{\text{TP}}{\text{TP} + \text{FP}}
\]

- Numerator: All true positives (TP).
- Denominator: All predicted positives (TP + FP).

### 4.2 Interpretation
- **Precision** answers: *“Of all the samples the model said were *positive*, how many truly are positive?”*
- If a model **over-predicts** positives (has a bias toward saying “Yes”), it will rack up more **false positives**, which **lowers** precision.
- **High Precision** is important when **false positives** are particularly costly.  
  - **Example**: A cancer diagnosis test that constantly flags healthy people as having cancer (FP) is problematic. High precision ensures that when we say “Positive,” we are really sure.

---

## 5. Recall (Sensitivity)

### 5.1 Definition

\[
\text{Recall} \;=\; \frac{\text{TP}}{\text{TP} + \text{FN}} \quad \text{(also known as Sensitivity)}
\]

- Numerator: All true positives (TP).
- Denominator: All actual positives (TP + FN).

### 5.2 Interpretation
- **Recall** answers: *“Of all the samples that are *actually* positive, how many did the model catch?”*
- If a model **under-predicts** positives (has a bias toward saying “No”), it will accumulate more **false negatives**, which **lowers** recall.
- **High Recall** is important when **false negatives** are especially costly.  
  - **Example**: A COVID-19 test that fails to identify sick people (FN) can let them spread the disease. We want high recall so we catch as many actual positives as possible.

---

## 6. F1 Score

### 6.1 Definition

\[
F1 \;=\; 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

Equivalently, using confusion matrix terms:

\[
F1 \;=\; \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}
\]

### 6.2 Interpretation
- F1 is the **harmonic mean** of Precision and Recall.
- A **high F1 score** requires both high precision and high recall.  
- It is a **balance**: the F1 score is useful as a single measure that captures both types of error (false positives and false negatives).

---

## 7. Numerical Examples

### 7.1 Example 1

**Confusion Matrix Values** (Total 100 samples):
- TP = 40  
- FP = 10  
- FN = 15  
- TN = 35  

\[
\begin{aligned}
\text{Accuracy} &= \frac{TP + TN}{TP + TN + FP + FN} = \frac{40 + 35}{40 + 35 + 10 + 15} = \frac{75}{100} = 0.75 \\
\text{Precision} &= \frac{TP}{TP + FP} = \frac{40}{40 + 10} = \frac{40}{50} = 0.80 \\
\text{Recall} &= \frac{TP}{TP + FN} = \frac{40}{40 + 15} = \frac{40}{55} \approx 0.73 \\
F1 &= 2 \times \frac{0.80 \times 0.73}{0.80 + 0.73} \approx 0.76 
\end{aligned}
\]

**Observations**:
- Accuracy = 0.75.  
- Precision = 0.80: The model is fairly good at *not* generating too many false positives.  
- Recall = 0.73: The model does miss some positives (FN=15).  
- F1 = 0.76: Roughly in between precision and recall.

The model has **slightly more** false positives than false negatives, but they’re not drastically unbalanced.

---

### 7.2 Example 2

**Confusion Matrix Values** (Total 100 samples, *same* accuracy but different distribution of errors):
- TP = 14  
- FP = 1  
- FN = 24  
- TN = 61  

Check the **total**: 14 + 1 + 24 + 61 = 100. Correct = TP + TN = 14 + 61 = 75 → Accuracy = 75%.

\[
\begin{aligned}
\text{Accuracy} &= \frac{14 + 61}{100} = 0.75 \\
\text{Precision} &= \frac{14}{14 + 1} = \frac{14}{15} \approx 0.93 \\
\text{Recall} &= \frac{14}{14 + 24} = \frac{14}{38} \approx 0.37 \\
F1 &= 2 \times \frac{0.93 \times 0.37}{0.93 + 0.37} \approx 0.53
\end{aligned}
\]

**Observations**:
- **Accuracy** = 0.75 (same as Example 1).  
- **Precision** = 0.93 → The model rarely makes a false positive.  
- **Recall** = 0.37 → The model *often* fails to detect true positives (many false negatives).  
- **F1** = 0.53 → Reflects the imbalance between precision and recall.

Despite the **same accuracy**, the *type* of mistakes is very different:
- In Example 2, the model is extremely conservative (rarely says “Yes”), leading to *few* false positives but *many* false negatives.  

---

## 8. Python Code Examples

### 8.1 Using Scikit-Learn to Compute the Metrics

```python
```python
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

# Example 1
y_true_1 = np.array([1]*40 + [0]*35 + [1]*15 + [0]*10)  # Just an illustrative ordering
y_pred_1 = np.array([1]*40 + [0]*35 + [0]*15 + [1]*10)

# Compute confusion matrix
cm1 = confusion_matrix(y_true_1, y_pred_1, labels=[1,0])
print("Confusion Matrix (Example 1):\n", cm1)

# Extract TP, FN, FP, TN
tp_1, fn_1, fp_1, tn_1 = cm1.ravel()
print(f"TP: {tp_1}, FN: {fn_1}, FP: {fp_1}, TN: {tn_1}")

acc_1 = accuracy_score(y_true_1, y_pred_1)
prec_1 = precision_score(y_true_1, y_pred_1)
rec_1 = recall_score(y_true_1, y_pred_1)
f1_1 = f1_score(y_true_1, y_pred_1)

print(f"Accuracy: {acc_1:.2f}")
print(f"Precision: {prec_1:.2f}")
print(f"Recall: {rec_1:.2f}")
print(f"F1: {f1_1:.2f}")

# Example 2
y_true_2 = np.array([1]*14 + [0]*61 + [1]*24 + [0]*1)  
y_pred_2 = np.array([1]*14 + [0]*61 + [0]*24 + [1]*1)

cm2 = confusion_matrix(y_true_2, y_pred_2, labels=[1,0])
print("\nConfusion Matrix (Example 2):\n", cm2)

tp_2, fn_2, fp_2, tn_2 = cm2.ravel()
print(f"TP: {tp_2}, FN: {fn_2}, FP: {fp_2}, TN: {tn_2}")

acc_2 = accuracy_score(y_true_2, y_pred_2)
prec_2 = precision_score(y_true_2, y_pred_2)
rec_2 = recall_score(y_true_2, y_pred_2)
f1_2 = f1_score(y_true_2, y_pred_2)

print(f"Accuracy: {acc_2:.2f}")
print(f"Precision: {prec_2:.2f}")
print(f"Recall: {rec_2:.2f}")
print(f"F1: {f1_2:.2f}")
```
```

**Note**:  
- We constructed `y_true_1, y_pred_1` and `y_true_2, y_pred_2` artificially to match the confusion matrix examples. In a real scenario, these arrays would be your test set labels and your model’s predictions.

### 8.2 Visualizing the Confusion Matrix
You can generate a heatmap to visualize these confusion matrices:

```python
```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion(cm, title):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive","Negative"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

plot_confusion(cm1, "Confusion Matrix (Example 1)")
plot_confusion(cm2, "Confusion Matrix (Example 2)")
```
```

---

## 9. When to Use Each Metric

### 9.1 Accuracy
- A good “first check” for overall correctness.
- Can be misleading in **heavily unbalanced** datasets.

### 9.2 Precision
- Use when **false positives** are especially **costly**.  
- **Example**: Medical diagnoses where a false positive may lead to invasive tests or anxiety.

### 9.3 Recall (Sensitivity)
- Use when **false negatives** are especially **costly**.  
- **Example**: Infectious disease screening or detection of malignant tumors—missing a positive is dangerous.

### 9.4 F1 Score
- **Balances** Precision and Recall into a single metric.  
- Ideal when you **want to track one metric** but also **care about** both types of errors.  
- Does not tell you *which* type of error is bigger—only that, on average, you have high (or low) performance across both.

---

## 10. Summary

1. **Accuracy**: \(\frac{TP + TN}{TP + TN + FP + FN}\)  
   - Overall success rate, but can be blind to imbalance.

2. **Precision**: \(\frac{TP}{TP + FP}\)  
   - Focuses on how **accurate** positive predictions are.  
   - High when **false positives** are minimized.

3. **Recall / Sensitivity**: \(\frac{TP}{TP + FN}\)  
   - Focuses on capturing **actual** positives.  
   - High when **false negatives** are minimized.

4. **F1 Score**: \(2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\)  
   - Harmonic mean: a single measure combining both precision and recall.

By examining **precision** and **recall** (and possibly **F1**), you gain **deeper insight** into your model’s biases—whether it tends to over- or under-predict positives—and how to **mitigate** those biases depending on the application.

---

## 11. Looking Ahead

The next steps will typically involve:
- **Hyperparameter tuning** to optimize one or more of these metrics (especially for unbalanced datasets).  
- Using **ROC curves** and **AUC** (Area Under the Curve) or **PR curves** (Precision-Recall curves) for a more continuous view of performance across different decision thresholds.
- Understanding how these metrics play out in **multi-class classification** scenarios.

---

## 12. References

- **Goodfellow, I., Bengio, Y., Courville, A.**: *Deep Learning* (MIT Press), Sections on classification metrics.  
- **Bishop, C.**: *Pattern Recognition and Machine Learning*, Chapter on performance measures.  
- **Scikit-learn documentation** on metrics: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)  
- **Signal Detection Theory** background: *Green & Swets (1966)*, *Signal Detection Theory and Psychophysics*.

---
```

**How to Use These Notes in Obsidian**:

1. **Create a new note** in your vault.  
2. Copy-paste the entire content above (including frontmatter `---`) into the new note.  
3. Rename the file and add your own internal links (e.g., `[[Unbalanced Datasets]]`) as desired.

These notes should give you a detailed theoretical and practical grounding in **Accuracy, Precision, Recall, and F1**—crucial metrics for **deep learning** and **machine learning** classification tasks.